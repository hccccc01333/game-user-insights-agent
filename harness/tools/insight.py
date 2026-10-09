# -*- coding: utf-8 -*-
"""insight.py —— 业务工具层：把 L1–L3 / 沉默预测 / 离线沙箱封装成 Agent 可调用的 8 个工具。

先说人话：
    Agent 要"做业务"，就得把已有能力（facts 快照、行为时间线、沉默预测工件、
    合成沙箱）摆成一排按钮。本模块就是这排按钮：
        get_user_behavior          公开行为摘要（L2 时间线只读）
        analyze_activity           活跃度画像（L3 activity 槽）
        analyze_interest_migration 兴趣迁移信号（L3 migration 槽）
        predict_silence_risk       未来 30 天沉默风险打分（离线工件推理）
        get_risk_cohort            按档位 + 风险分圈人（产出 cohort_id）
        plan_intervention          启发式分档生成触达队列（非因果估计）
        evaluate_strategy          离线沙箱评测策略（审计池冻结得分）
        generate_insight_report    汇总事实生成洞察报告（含证据与建议）

    每个工具都带三个刻意的工程约束：
      · 参数 Schema：模型给的参数先过 registry 校验（见 registry.py）；
      · 权限串：facts:read / model:predict / intervention:plan /
        sandbox:evaluate / report:generate（未授予不进候选、不可调用）；
      · 口径诚实：不适用 / 缺失显式返回 null + 原因，绝不静默填 0；
        干预建议标注"启发式分档，非因果收益估计"。

    故障注入（hooks）：评测需要"工具超时 / 返回故障"这类场景，builder 支持
    传入 {tool_name: wrapper} 包裹真实实现（tests / agent_eval 用，生产不传）。

隐私口径：所有工具只吐出 uid_hash 与聚合 / 派生字段；不含昵称、原文、ip/device。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import median
from pathlib import Path
from typing import Any, Callable

from silence_risk.artifact import DEFAULT_MODEL_PATH, SilencePredictor
from silence_risk.panel import DAY, TZ_CN, group_exact_events, load_jsonl
from silence_risk.run_silence import DEFAULT_INDEX, DEFAULT_TIMELINE

from ..facts import DEFAULT_FACTS_PATH, FactsStore, get_path, load_facts
from ..registry import ToolRegistry
from ..run_real_agent import DEFAULT_STAGES, PLAN_POLICY, InterventionPlanner, RealAgentConfig

TOOLSET_VERSION = "tools_v0"

# 8 个工具规范名（与 verifier.py 契约、agent_eval 场景共用）
TOOL_NAMES = (
    "get_user_behavior", "analyze_activity", "analyze_interest_migration",
    "predict_silence_risk", "get_risk_cohort", "plan_intervention",
    "evaluate_strategy", "generate_insight_report",
)
PERMISSIONS = ("facts:read", "model:predict", "intervention:plan",
               "sandbox:evaluate", "report:generate")

DEFAULT_RISK_HIGH = 80.0  # 召回触达阈值（与真实主链同口径）
DEFAULT_RISK_MID = 60.0   # 内容推荐阈值
DEFAULT_PLAN_COUNT = 50   # 单次干预队列默认名额
SUBSTANTIVE_TOOLS = ("get_risk_cohort", "predict_silence_risk",
                     "plan_intervention", "generate_insight_report")


def _fmt_date(ts: Any) -> str | None:
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts <= 0:
        return None
    return datetime.fromtimestamp(int(ts), TZ_CN).strftime("%Y-%m-%d")


def derive_as_of(store: FactsStore, predictor: Any = None) -> int | None:
    """快照时间锚点：取排序后首个用户的 int as_of；缺则退回预测工件的默认 as_of。"""
    for uid in store.uids():
        value = get_path(store.get(uid), "as_of")
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    if predictor is not None:
        default = getattr(predictor, "default_as_of_ts", None)
        if callable(default):
            try:
                return int(default())
            except Exception:  # noqa: BLE001 - 锚点推导失败不致命，返回 None
                return None
    return None


class TimelineView:
    """L2 时间线的只读视图：按 uid 归拢 exact 事件（公开时间戳口径）。"""

    def __init__(self, records: list[dict] | None = None) -> None:
        self._by_uid = group_exact_events(list(records or []))

    def events_of(self, uid_hash: str) -> list[dict]:
        return list(self._by_uid.get(uid_hash) or [])


# ── 沙箱配置与评测运行器 ────────────────────────────────────

@dataclass
class SandboxConfig:
    """离线合成沙箱口径（S1–S4；换口径必须升工具集版本）。"""

    n_users: int = 800
    days: int = 14
    window: int = 7
    seed: int = 13
    audit_frac: float = 0.3
    basis: str = "logsq"

    def __post_init__(self) -> None:
        if self.n_users < 200:
            raise ValueError(f"沙箱用户数需 ≥ 200（当前 {self.n_users}）")
        if self.days < 2:
            raise ValueError(f"沙箱天数需 ≥ 2（当前 {self.days}）")
        if not (1 <= self.window <= self.days):
            raise ValueError(f"奖励窗口需满足 1 ≤ window ≤ days（当前 {self.window}/{self.days}）")
        if not (0.0 < self.audit_frac < 1.0):
            raise ValueError(f"audit_frac 需在 (0,1) 内（当前 {self.audit_frac}）")
        if self.basis not in ("raw", "logsq"):
            raise ValueError(f"未知上下文基：{self.basis!r}（支持 raw / logsq）")

    def snapshot(self) -> dict:
        return asdict(self)


def make_sandbox_runner(cfg: SandboxConfig | None = None) -> Callable[[str], dict]:
    """造"真沙箱"运行器：合成人口 → 部分反馈世界 → 审计池冻结评测 → 汇总。

    评测口径 = 策略带着在线后验上场、审计池"只选不学"（run_audit）；
    oracle 是评测上界（读真值矩阵），显式拒绝作为可部署策略。
    """
    cfg = cfg or SandboxConfig()

    def runner(policy: str) -> dict:
        from bandit.evaluate import audit_summary, run_audit
        from bandit.policies import POLICIES, make_policy
        from bandit.protocol import build_world
        from simulator.run_sim import build_tables

        if policy not in POLICIES:
            raise ValueError(f"未知策略：{policy!r}（支持 {POLICIES}）")
        if policy == "oracle":
            raise ValueError("oracle 是评测上界（读真值矩阵），不可作为可部署策略评测")
        tables, _ = build_tables(cfg.n_users, cfg.days, cfg.seed, cfg.window, None, use_fit=False)
        built = build_world(tables, cfg.audit_frac, cfg.seed, cfg.basis)
        world = built["world"]
        pol = make_policy(policy, context_dim=built["meta"]["context_width"],
                          seed=cfg.seed, oracle_arm=world.oracle_arm)
        log = run_audit(world, built["audit"], pol)
        summary = audit_summary(log)
        clean = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in summary.items()
        }
        return {
            "policy": policy, **clean,
            "n_audit": int(len(log)),
            "sandbox": cfg.snapshot(),
            "note": "离线合成沙箱（S1–S4）：审计池冻结得分，非真实因果；oracle 对照仅供评测",
        }

    return runner


# ── 人群管理（每个 cohort_id 一个干预规划器）────────────────

class CohortAdvisor:
    """圈人与干预规划：get_risk_cohort 圈人 → plan_intervention 分档出队列。

    每次圈人产出一个 cohort_id（cohort-1 / cohort-2 …，确定性编号），并挂一个
    独立的路由器：复用真实主链的分档规则（risk_score 缺失 → 不打扰兜底）。
    """

    def __init__(
        self,
        store: FactsStore,
        *,
        risk_high: float = DEFAULT_RISK_HIGH,
        risk_mid: float = DEFAULT_RISK_MID,
        default_count: int = DEFAULT_PLAN_COUNT,
    ) -> None:
        self.store = store
        self.risk_high = float(risk_high)
        self.risk_mid = float(risk_mid)
        self.default_count = int(default_count)
        self._cohorts: dict[str, dict] = {}
        self._order: list[str] = []
        self.intervention_rows: list[dict] = []  # 全部队列行（落盘用）

    # ── 圈人 ────────────────────────────────────────────────

    def open_cohort(
        self,
        uids: list[str],
        *,
        stages: tuple[str, ...],
        min_score: float | None,
        max_size: int | None,
    ) -> dict:
        uids = [str(u) for u in uids]
        if not uids:
            raise ValueError("人群为空：检查圈人条件与 facts 覆盖")
        name = f"沉默风险人群（档位 {'/'.join(stages)}"
        name += "）" if min_score is None else f"；risk_score ≥ {min_score:g}）"
        config = RealAgentConfig(
            cohort_stages=tuple(stages), cohort_name=name, batch=self.default_count,
            risk_high=self.risk_high, risk_mid=self.risk_mid,
        )
        planner = InterventionPlanner(self.store, config)
        planner.open_cohort(uids, name=name)
        cohort_id = f"cohort-{len(self._order) + 1}"
        self._cohorts[cohort_id] = {
            "cohort_id": cohort_id, "name": name, "members": uids,
            "stages": tuple(stages), "min_score": min_score, "max_size": max_size,
            "planner": planner,
        }
        self._order.append(cohort_id)
        return {"cohort_id": cohort_id, "name": name}

    # ── 查询与规划 ──────────────────────────────────────────

    def has_cohorts(self) -> bool:
        return bool(self._order)

    def resolve(self, cohort_id: str | None) -> dict:
        """cohort_id 缺省 = 最近一次圈定的人群；未知 id 显式报错。"""
        if not self._order:
            raise ValueError("尚无人群：先调用 get_risk_cohort 圈定人群")
        if cohort_id is None:
            return self._cohorts[self._order[-1]]
        row = self._cohorts.get(str(cohort_id))
        if row is None:
            raise ValueError(
                f"未知 cohort_id：{cohort_id!r}（已圈定：{', '.join(self._order)}）"
            )
        return row

    def remaining(self, cohort_id: str | None = None) -> int:
        if not self._order and cohort_id is None:
            return 0
        return self.resolve(cohort_id)["planner"].remaining()

    def plan_available(self) -> bool:
        return any(self._cohorts[c]["planner"].remaining() > 0 for c in self._order)

    def plan(self, cohort_id: str | None = None, count: int | None = None) -> dict:
        row = self.resolve(cohort_id)
        planner = row["planner"]
        before = len(planner.queue_rows)
        obs = planner.plan(count if count is not None else self.default_count)
        new_rows = planner.queue_rows[before:]
        self.intervention_rows.extend(new_rows)
        return {
            **obs,
            "cohort_id": row["cohort_id"],
            "rows_sample": new_rows[:5],
            "note": ("启发式风险分档（宁不打扰，缺分数兜底不打扰），非因果收益估计；"
                     f"规则版本 {PLAN_POLICY}；效果验证走离线沙箱与 A/B 对照"),
        }


# ── 工具实现（Toolkit：方法即工具）──────────────────────────

class InsightToolkit:
    """8 个业务工具的实现集合；builder 把方法登记进 ToolRegistry。"""

    def __init__(
        self,
        store: FactsStore,
        records: list[dict] | None = None,
        predictor: Any = None,
        *,
        as_of_ts: int | None = None,
        sandbox: Callable[[str], dict] | None = None,
        sandbox_config: SandboxConfig | None = None,
        risk_high: float = DEFAULT_RISK_HIGH,
        risk_mid: float = DEFAULT_RISK_MID,
        cohort_stages: tuple[str, ...] = DEFAULT_STAGES,
        default_plan_count: int = DEFAULT_PLAN_COUNT,
    ) -> None:
        self.store = store
        self.records = list(records or [])
        self.predictor = predictor
        self.timeline = TimelineView(self.records)
        self._as_of = int(as_of_ts) if as_of_ts is not None else derive_as_of(store, predictor)
        self.sandbox = sandbox or make_sandbox_runner(sandbox_config)
        self.cohort_stages = tuple(cohort_stages)
        self.advisor = CohortAdvisor(
            store, risk_high=risk_high, risk_mid=risk_mid, default_count=default_plan_count,
        )

    # ── 单用户三个读取工具 ──────────────────────────────────

    def _require_uid(self, uid_hash: Any) -> str:
        if not isinstance(uid_hash, str) or not uid_hash.strip():
            raise ValueError("uid_hash 需为非空字符串")
        return uid_hash

    def _gap_days(self, events: list[dict]) -> float | None:
        if not events or self._as_of is None:
            return None
        return round(max(0.0, (self._as_of - int(events[-1]["event_ts"])) / DAY), 3)

    def get_user_behavior(self, uid_hash: str) -> dict:
        """公开行为摘要（时间线只读；不含事件原文）。"""
        uid = self._require_uid(uid_hash)
        fact = self.store.get(uid)
        events = self.timeline.events_of(uid)
        n = len(events)
        top = [
            {"event_type": t, "count": c}
            for t, c in Counter(str(e.get("event_type") or "unknown") for e in events).most_common(5)
        ]
        return {
            "uid_hash": uid,
            "found": fact is not None,
            "n_exact_events": n,
            "first_event_date": _fmt_date(events[0]["event_ts"]) if n else None,
            "last_event_date": _fmt_date(events[-1]["event_ts"]) if n else None,
            "gap_days": self._gap_days(events),
            "top_event_types": top,
            "note": "时间线只含页面公开时间戳事件（time_kind=exact）；found=False 表示不在本批 facts 快照",
        }

    def analyze_activity(self, uid_hash: str) -> dict:
        """活跃度画像（L3 activity 槽 + 时间线沉默天数）。"""
        uid = self._require_uid(uid_hash)
        fact = self.store.get(uid)
        events = self.timeline.events_of(uid)
        return {
            "uid_hash": uid,
            "found": fact is not None,
            "band": get_path(fact, "activity.band"),
            "score": get_path(fact, "activity.score"),
            "percentile": get_path(fact, "activity.percentile"),
            "stage": get_path(fact, "churn.stage"),
            "usable": bool(get_path(fact, "quality.usable")) if fact else False,
            "gap_days": self._gap_days(events),
            "as_of_date": _fmt_date(self._as_of),
            "note": "band ∈ dormant/low/mid/high/top（批内分位）；score 0–100 为锚点绝对分；缺失为 null",
        }

    def analyze_interest_migration(self, uid_hash: str) -> dict:
        """兴趣迁移信号（L3 migration 槽；数据不足显式标注）。"""
        uid = self._require_uid(uid_hash)
        fact = self.store.get(uid)
        if fact is None:
            return {
                "uid_hash": uid, "found": False, "insufficient_data": True,
                "genre_from": None, "genre_to": None, "genre_shift_score": None,
                "game_flow_net": None, "dropped_games": None, "top_genres": None,
                "note": "无 facts 记录：按数据不足处理，不做任何推断",
            }
        return {
            "uid_hash": uid,
            "found": True,
            "insufficient_data": bool(get_path(fact, "migration.insufficient_data")),
            "genre_from": get_path(fact, "migration.genre_from"),
            "genre_to": get_path(fact, "migration.genre_to"),
            "genre_shift_score": get_path(fact, "migration.genre_shift_score"),
            "game_flow_net": get_path(fact, "migration.game_flow_net"),
            "dropped_games": get_path(fact, "migration.dropped_games"),
            "top_genres": get_path(fact, "context.top_genres"),
            "note": "genre_shift_score ∈ [0,1] 为两窗品类分布 JS 散度；insufficient_data=true 时不给强度",
        }

    def predict_silence_risk(self, uid_hash: str) -> dict:
        """未来 30 天公开行为沉默风险打分（离线工件；不适用显式给原因）。"""
        uid = self._require_uid(uid_hash)
        if self.predictor is None:
            raise ValueError("沉默预测工件未加载：本次运行未注册该工具")
        result = dict(self.predictor.score_user(self.records, uid))
        meta = getattr(self.predictor, "meta", {}) or {}
        result["model"] = {
            "artifact_version": meta.get("artifact_version"),
            "learner": meta.get("learner"),
        }
        return result

    # ── 人群 / 干预 / 沙箱 / 报告 ───────────────────────────

    def get_risk_cohort(self, criteria: dict | None = None) -> dict:
        """按档位 + 风险分圈人；产出 cohort_id 供 plan_intervention 使用。"""
        criteria = dict(criteria or {})
        stages = tuple(criteria.get("stages") or self.cohort_stages)
        min_score = criteria.get("min_score")
        max_size = criteria.get("max_size")
        if min_score is not None and (isinstance(min_score, bool) or not isinstance(min_score, (int, float))):
            raise ValueError(f"min_score 需为数值或 null（当前 {min_score!r}）")
        if max_size is not None and (isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1):
            raise ValueError(f"max_size 需为 ≥1 的整数或 null（当前 {max_size!r}）")

        usable = sorted(self.store.usable())
        picked: list[str] = []
        for uid in usable:
            fact = self.store.get(uid)
            if get_path(fact, "churn.stage") not in stages:
                continue
            if min_score is not None:
                risk = get_path(fact, "churn.risk_score")
                if risk is None or float(risk) < float(min_score):
                    continue
            picked.append(uid)
        truncated = False
        if max_size is not None and len(picked) > max_size:
            picked = picked[:max_size]
            truncated = True
        if not picked:
            raise ValueError(
                "圈人结果为空：放宽 criteria（换档位 / 降 min_score），或先确认 facts 覆盖"
            )
        info = self.advisor.open_cohort(
            picked, stages=stages, min_score=min_score, max_size=max_size,
        )
        breakdown = Counter(str(get_path(self.store.get(uid), "churn.stage")) for uid in picked)
        return {
            "cohort_id": info["cohort_id"],
            "cohort": info["name"],
            "criteria": {"stages": list(stages), "min_score": min_score, "max_size": max_size},
            "size": len(picked),
            "share_of_usable": round(len(picked) / len(usable), 4) if usable else None,
            "stage_breakdown": dict(sorted(breakdown.items())),
            "sample_uids": picked[:10],
            "truncated": truncated,
            "note": "risk_score 为 L3 信号合成排序（非校准概率）；圈人依据 facts 只读快照",
        }

    def plan_intervention(self, cohort_id: str | None = None, count: int | None = None) -> dict:
        """对已圈定人群生成启发式触达队列（cohort_id 缺省 = 最近一次圈定）。"""
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
            raise ValueError(f"count 需为 ≥1 的整数或 null（当前 {count!r}）")
        return self.advisor.plan(cohort_id if cohort_id is None else str(cohort_id), count)

    def plan_available(self) -> bool:
        """前置条件：至少一个人群仍有未分配名额。"""
        return self.advisor.plan_available()

    def evaluate_strategy(self, policy: str = "linucb") -> dict:
        """在离线合成沙箱评测某 bandit 策略（审计池冻结得分；非真实因果）。"""
        if not isinstance(policy, str) or not policy.strip():
            raise ValueError("policy 需为非空字符串（如 linucb / thompson / random）")
        summary = dict(self.sandbox(policy))
        if "mean_reward_audit" not in summary:
            raise ValueError("沙箱返回缺 mean_reward_audit：产出不合契约，拒绝给出评测结论")
        summary.setdefault("policy", policy)
        summary.setdefault("note", "离线合成沙箱口径，非真实因果；仅供策略对照")
        return summary

    def generate_insight_report(self, scope: str = "cohort", cohort_id: str | None = None) -> dict:
        """汇总当前事实生成洞察报告（含证据字段与建议；scope=community/cohort）。"""
        if scope not in ("community", "cohort"):
            raise ValueError(f"scope 需为 community / cohort（当前 {scope!r}）")
        resolved_id: str | None = None
        if scope == "community":
            uids = sorted(self.store.usable())
            population = "全量可用用户（quality.usable）"
        else:
            row = self.advisor.resolve(cohort_id)
            uids = list(row["members"])
            population = row["name"]
            resolved_id = row["cohort_id"]
        facts = [self.store.get(u) or {} for u in uids]

        stage_mix = Counter(str(get_path(f, "churn.stage") or "unknown") for f in facts)
        risks = [float(v) for v in (get_path(f, "churn.risk_score") for f in facts) if v is not None]
        bands = {
            f"<{self.advisor.risk_mid:.0f}": 0,
            f"{self.advisor.risk_mid:.0f}-{self.advisor.risk_high:.0f}": 0,
            f">={self.advisor.risk_high:.0f}": 0,
            "unknown": 0,
        }
        for f in facts:
            risk = get_path(f, "churn.risk_score")
            if risk is None:
                bands["unknown"] += 1
            elif float(risk) >= self.advisor.risk_high:
                bands[f">={self.advisor.risk_high:.0f}"] += 1
            elif float(risk) >= self.advisor.risk_mid:
                bands[f"{self.advisor.risk_mid:.0f}-{self.advisor.risk_high:.0f}"] += 1
            else:
                bands[f"<{self.advisor.risk_mid:.0f}"] += 1

        drivers: Counter = Counter()
        for f in facts:
            for d in (get_path(f, "churn.drivers") or []):
                drivers[str(d)] += 1
        n = len(facts)
        negs = [v for v in (get_path(f, "context.sentiment_neg_rate") for f in facts) if v is not None]
        inactive = sum(1 for f in facts if str(get_path(f, "churn.stage")) in
                       ("silent", "churned", "dormant_90", "dormant_60"))
        headline = {
            "n_users": n,
            "stage_mix": dict(sorted(stage_mix.items())),
            "median_risk": round(median(risks), 2) if risks else None,
            "inactive_share": round(inactive / n, 4) if n else None,
        }
        high_key = f">={self.advisor.risk_high:.0f}"
        mid_key = f"{self.advisor.risk_mid:.0f}-{self.advisor.risk_high:.0f}"
        n_high, n_mid = bands[high_key], bands[mid_key]
        actions = []
        if n_high or n_mid:
            actions.append({
                "action": "生成触达候选队列（recall / rec 分档）",
                "rationale": f"风险分档中 {n_high} 人 ≥{self.advisor.risk_high:.0f}、"
                             f"{n_mid} 人处于 {mid_key}（启发式分档，非因果收益估计）",
                "next_step": "plan_intervention 生成队列 → 设 holdout 对照 → 观察 30 天公开沉默率",
            })
        else:
            actions.append({
                "action": "维持不打扰，继续观察",
                "rationale": "风险分档中无 ≥ 中档的用户",
                "next_step": "扩大观察窗或复核圈人条件",
            })
        actions.append({
            "action": "保留 holdout 对照组",
            "rationale": "触达效果需实验验证，不能以触达量代替效果结论",
            "next_step": "在实验分组中固定 treatment / holdout 比例",
        })
        return {
            "scope": scope,
            "cohort_id": resolved_id,
            "population": population,
            "headline": headline,
            "risk_bands": bands,
            "top_drivers": [
                {"driver": d, "count": c, "share": round(c / n, 4) if n else None}
                for d, c in drivers.most_common(3)
            ],
            "sentiment_neg_rate_median": round(median(negs), 4) if negs else None,
            "recommended_actions": actions,
            "evidence": [
                {"field": "churn.risk_score", "role": "风险排序主信号（信号合成，非校准概率）"},
                {"field": "churn.stage", "role": "沉默/流失档位（圈人口径）"},
                {"field": "churn.drivers", "role": "风险成因标签（计数聚合）"},
                {"field": "activity.band", "role": "活跃度批内分位档"},
                {"field": "migration.insufficient_data", "role": "兴趣迁移数据可得性"},
                {"field": "context.sentiment_neg_rate", "role": "负面情感占比（聚合）"},
            ],
            "caveats": [
                "facts 为单快照只读口径（as_of 锚点），跨批差异不代表趋势",
                "risk_score 阈值未经运营反馈校准（calibration_status=uncalibrated）",
                "本报告为趋势性提示，非因果结论；触达建议需实验（holdout 对照）验证",
            ],
        }


# ── 工具注册（builder：Schema + 权限 + 前置条件 + 故障注入）──

_SCHEMA_UID = {
    "type": "object",
    "properties": {"uid_hash": {"type": "string"}},
    "required": ["uid_hash"],
}
_SCHEMA_COHORT = {
    "type": "object",
    "properties": {"criteria": {"type": "object", "properties": {
        "stages": {"type": "array", "items": {"type": "string"}},
        "min_score": {"type": "number"},
        "max_size": {"type": "integer"},
    }}},
}
_SCHEMA_PLAN = {
    "type": "object",
    "properties": {"cohort_id": {"type": "string"}, "count": {"type": "integer"}},
}
_SCHEMA_EVAL = {
    "type": "object",
    "properties": {"policy": {"type": "string"}},
}
_SCHEMA_REPORT = {
    "type": "object",
    "properties": {"scope": {"type": "string", "enum": ["community", "cohort"]},
                   "cohort_id": {"type": "string"}},
}


def build_insight_registry(
    toolkit: InsightToolkit,
    *,
    granted_permissions: set[str] | tuple[str, ...] | None = None,
    hooks: dict[str, Callable[[Callable], Callable]] | None = None,
) -> ToolRegistry:
    """把 toolkit 的 8 个工具登记进 registry（权限收窄 + 故障注入）。

    · predictor 为 None 时 predict_silence_risk 不注册（工件缺失不假装能预测）；
    · hooks：{工具名: wrapper}，在登记前包裹真实实现（评测故障注入用）。
    """
    reg = ToolRegistry(granted_permissions=granted_permissions)
    hooks = dict(hooks or {})

    def add(name: str, fn: Callable, description: str, schema: dict,
            permission: str, when: Callable | None = None) -> None:
        wrapped = hooks[name](fn) if name in hooks else fn
        reg.register(name, wrapped, description, when=when, schema=schema, permission=permission)

    add("get_user_behavior", toolkit.get_user_behavior,
        "读取某用户的公开行为摘要（事件数 / 首末日期 / 类型分布；只读）",
        _SCHEMA_UID, "facts:read")
    add("analyze_activity", toolkit.analyze_activity,
        "读取某用户的活跃度画像（band / 分数 / 分位 / 沉默天数）",
        _SCHEMA_UID, "facts:read")
    add("analyze_interest_migration", toolkit.analyze_interest_migration,
        "读取某用户的兴趣迁移信号（换出换入品类 / 迁移强度；不足显式标注）",
        _SCHEMA_UID, "facts:read")
    if toolkit.predictor is not None:
        add("predict_silence_risk", toolkit.predict_silence_risk,
            "对某用户做未来 30 天公开行为沉默风险打分（离线工件；不适用给原因）",
            _SCHEMA_UID, "model:predict")
    add("get_risk_cohort", toolkit.get_risk_cohort,
        "按档位与风险分圈定风险人群（产出 cohort_id；risk_score 为信号排序非概率）",
        _SCHEMA_COHORT, "facts:read")
    add("plan_intervention", toolkit.plan_intervention,
        "对已圈定人群生成启发式触达队列（分档建议 + 不打扰兜底；cohort_id 缺省=最近人群）",
        _SCHEMA_PLAN, "intervention:plan",
        when=lambda state: toolkit.plan_available())
    add("evaluate_strategy", toolkit.evaluate_strategy,
        "在离线合成沙箱评测某策略的审计得分（非真实因果；oracle 不可用）",
        _SCHEMA_EVAL, "sandbox:evaluate")
    add("generate_insight_report", toolkit.generate_insight_report,
        "汇总事实生成洞察报告（含证据字段与建议；scope=community/cohort）",
        _SCHEMA_REPORT, "report:generate")
    return reg


# ── 真实工件加载（CLI 用；缺工件显式降级，不假装有）─────────

def load_tool_inputs(
    *,
    facts_path: Path | str = DEFAULT_FACTS_PATH,
    timeline_path: Path | str | None = None,
    model_path: Path | str | None = None,
    index_path: Path | str | None = None,
) -> dict:
    """读真实工件：facts（必须；缺文件显式报错）/ 时间线（可选）/ 沉默模型（可选）。"""
    facts_path = Path(facts_path)
    store = load_facts(facts_path)
    timeline_path = Path(timeline_path or DEFAULT_TIMELINE)
    records = load_jsonl(timeline_path) if timeline_path.exists() else []
    predictor = None
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    if model_path.exists():
        idx = Path(index_path or DEFAULT_INDEX)
        predictor = SilencePredictor.load(model_path, idx if idx.exists() else None)
    return {
        "store": store, "records": records, "predictor": predictor,
        "facts_path": facts_path, "timeline_path": timeline_path, "model_path": model_path,
    }


# ── 定案记录 ────────────────────────────────────────────────
# 1. 工具粒度 = 8 个业务动作（用户级 4 + 人群/干预/沙箱/报告 4）；
#    参数用 JSON-Schema 子集收口，权限串一工具一串。
# 2. plan_intervention 复用真实主链的分档规则（risk_band_heuristic_v0），
#    不引入第二套逻辑；沙箱评测与报告生成均标注"非因果 / 未校准"。
# 3. cohort_id 采用确定性编号（cohort-1/2/…），保证同参双跑可对拍。
# ────────────────────────────────────────────────────────────
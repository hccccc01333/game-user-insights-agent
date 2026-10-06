# -*- coding: utf-8 -*-
"""run_real_agent.py —— ① 真实主链：L3 facts → Agent 六环节决策（realv0）。

先说人话：
    run_agent.py 走的是 S1 合成人口 + Bandit 干预台，定位是**离线评测沙箱**
    （回答"如果干预会发生什么"——策略效果在这里验证与对照）；本文件是另一条
    主链：直接吃 L3 产出的 `facts.jsonl`（一用户一行的事实契约，真实、只读），
    让六环节（异常 → 人群 → 原因 → 风险 → 干预 → 实验）在真实状态上跑一遍，
    产出给运营的**干预队列**与**实验分组**（不是策略评测）。

    与沙箱链的两处口径差异（诚实标注，不许混淆）：
      · 干预臂来自启发式风险分档（risk_score → recall / rec / control），
        不是 bandit 学习，也**不是因果收益估计**；效果验证仍走沙箱（S1–S4）；
      · 实验分组对"触达候选"做确定性哈希分流（treatment / holdout），
        outcome（未来 30 天公开行为沉默率）待观察窗回填。

运行（在仓库根目录）：
    python -m harness.run_real_agent                       # 默认读 L3 facts
    python -m harness.run_real_agent --batch 50 --risk-high 80 --risk-mid 60

产出（默认 data/processed/harness_real/；仅 uid_hash 与派生字段，无隐私原文）：
    real_trace.jsonl              主循环轨迹（每步：决策 → 观察 → 校验）
    real_artifacts.json           六环节产物（键=工具名）
    intervention_queue.jsonl      干预队列（逐人：臂 + 理由 + 规则版本）
    experiment_assignments.jsonl  实验分组（treatment / holdout）
    _manifest.json                版本 / facts 指纹 / 配置 / 输出指纹 / 双跑一致
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

from . import HARNESS_VERSION
from .facts import DEFAULT_FACTS_PATH, FactsStore, get_path, load_coverage, load_facts
from .loop import run
from .model import MockModel
from .planner import FINISH_TOOL, Planner
from .registry import ToolRegistry
from .verifier import Verifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "harness_real"
TZ_CN = timezone(timedelta(hours=8))

REAL_CHAIN_VERSION = "realv0"
GOAL = "降低未来30日公开行为沉默风险"

# facts 的 churn.stage 已知档位（其余落 unknown，不做静默归档）
KNOWN_STAGES = (
    "churned", "silent", "dormant_90", "dormant_60", "dormant_30", "active", "unknown",
)

DEFAULT_STAGES = ("dormant_60", "dormant_90")
DEFAULT_COHORT_NAME = "沉默风险人群（公开行为中断 ≥60 天、账号可触达）"

PLAN_POLICY = "risk_band_heuristic_v0"  # 干预分档规则版本（换规则必须升版本）
EXPERIMENT_GROUPING = "A/B（treatment=触达 / holdout=不打扰）"
EXPERIMENT_METRIC = "未来 30 天公开行为沉默率（public_inactivity_rate_30d）"

# 六环节字段契约（键=工具名；RealChainVerifier 与 tests 共用）
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "detect_anomaly": ("metric", "inactive_share", "stage_mix", "n_total", "exceeds_reference"),
    "locate_cohort": ("cohort", "rule", "size", "share_of_usable"),
    "analyze_cause": ("cohort", "n_total", "top_drivers"),
    "assess_risk": ("cohort", "bands", "n_total", "median_risk"),
    "plan_interventions": (
        "batch", "batch_size", "arms", "cohort", "stream_remaining", "policy", "rule_version",
    ),
    "design_experiment": (
        "experiment_id", "grouping", "primary_metric", "n_treated", "n_control", "assignment_rule",
    ),
}


@dataclass
class RealAgentConfig:
    """真实主链配置（CLI 与测试共用；构造时即校验）。"""

    cohort_stages: tuple[str, ...] = DEFAULT_STAGES
    cohort_name: str = ""  # 空 = 自动派生（默认档位用 DEFAULT_COHORT_NAME）
    batch: int = 50
    risk_high: float = 80.0
    risk_mid: float = 60.0
    reference_share: float = 0.5
    max_steps: int = 8
    experiment_key: str = "silence-recall-v0"

    def __post_init__(self) -> None:
        self.cohort_stages = tuple(self.cohort_stages)
        self.validate()
        if not self.cohort_name:
            self.cohort_name = (
                DEFAULT_COHORT_NAME if self.cohort_stages == tuple(DEFAULT_STAGES)
                else f"沉默风险人群（档位 {'/'.join(self.cohort_stages)}、账号可触达）"
            )

    def validate(self) -> None:
        if self.batch < 1:
            raise ValueError(f"batch 需 ≥ 1（当前 {self.batch}）")
        if not (0.0 < self.risk_mid < self.risk_high <= 100.0):
            raise ValueError(
                f"需满足 0 < risk_mid < risk_high ≤ 100（当前 {self.risk_mid} / {self.risk_high}）"
            )
        if not (0.0 < self.reference_share < 1.0):
            raise ValueError(f"reference_share 需在 (0,1) 内（当前 {self.reference_share}）")
        if self.max_steps < 7:
            raise ValueError(f"max_steps 需 ≥ 7（六环节 + 收工；当前 {self.max_steps}）")
        if not self.cohort_stages:
            raise ValueError("cohort_stages 不能为空")
        unknown = [s for s in self.cohort_stages if s not in KNOWN_STAGES]
        if unknown:
            raise ValueError(f"未知档位 {unknown}（支持 {KNOWN_STAGES}）")


class InterventionPlanner:
    """真实主链的干预规划器：风险分档启发式（无学习、无反馈通道）。

    与沙箱链的 bandit 干预台对照：那边有 world.observe 反馈与净奖励结算，
    这边没有——facts 里的 risk_score 是 L3 模型输出（口径已冻结），分档规则
    （宁不打扰）：
        risk_score 缺失 → control（缺失不填 0）
        ≥ risk_high     → recall（召回触达）
        ≥ risk_mid      → rec（内容推荐）
        其余            → control
    每行队列都带 reason；obs 明示"启发式分档，非因果收益估计"。
    """

    def __init__(self, store: FactsStore, config: RealAgentConfig) -> None:
        self.store = store
        self.config = config
        self._members: list[str] | None = None
        self._cursor = 0
        self._batch_no = 0
        self._last_rows: list[dict] = []
        self.batch_logs: list[dict] = []       # 每批摘要（Agent 可见口径）
        self.queue_rows: list[dict] = []       # 逐人干预队列（落盘）
        self.assignment_rows: list[dict] = []  # 实验分组（落盘）

    # ── 人群 ────────────────────────────────────────────────

    @property
    def cohort_ready(self) -> bool:
        return self._members is not None

    def members(self) -> list[str]:
        return list(self._members or [])

    def remaining(self) -> int:
        """人群中尚未分配的名额数（未圈人时为 0）。"""
        if self._members is None:
            return 0
        return len(self._members) - self._cursor

    def open_cohort(self, uids: list[str], name: str | None = None) -> dict:
        """圈定人群（须在首次 plan 前调用；重复调用显式报错）。"""
        if self._members is not None:
            raise ValueError("人群已开跑，不可更换（一次运行只服务一个人群）")
        uids = [str(u) for u in uids]
        if not uids:
            raise ValueError("人群为空：检查圈人条件与 facts 覆盖")
        self._members = uids
        return {"cohort": name or self.config.cohort_name, "size": len(uids)}

    # ── 分档与队列 ──────────────────────────────────────────

    def arm_for(self, risk: float | None) -> tuple[str, str]:
        """风险分 → (触达臂, 理由)；缺失与低分都兜底"不打扰"。"""
        if risk is None:
            return "control", "risk_score 缺失→不打扰"
        if risk >= self.config.risk_high:
            return "recall", "召回触达"
        if risk >= self.config.risk_mid:
            return "rec", "内容推荐"
        return "control", "低风险→不打扰"

    def plan(self, count: int) -> dict:
        """从人群按序取 count 个名额，逐人生成干预队列行并返回批次摘要。"""
        if self._members is None:
            raise ValueError("尚未圈定人群：先调用 locate_cohort（干预规划器只服务已圈人群）")
        count = int(count)
        if count < 1:
            raise ValueError(f"名额数需 ≥ 1（当前 {count}）")
        remaining = self.remaining()
        if remaining == 0:
            raise ValueError("人群名额已跑完（stream_remaining = 0）")
        actual = min(count, remaining)

        self._batch_no += 1
        counts = {"control": 0, "rec": 0, "recall": 0}
        rows: list[dict] = []
        for _ in range(actual):
            uid = self._members[self._cursor]
            self._cursor += 1
            fact = self.store.get(uid) or {}
            risk = get_path(fact, "churn.risk_score")
            arm, reason = self.arm_for(risk)
            counts[arm] += 1
            row = {
                "batch": self._batch_no,
                "seq": len(self.queue_rows),
                "uid_hash": uid,
                "cohort": self.config.cohort_name,
                "stage": get_path(fact, "churn.stage"),
                "risk_score": risk,
                "arm": arm,
                "reason": reason,
                "policy": PLAN_POLICY,
            }
            self.queue_rows.append(row)
            rows.append(row)
        self._last_rows = rows

        obs = {
            "batch": self._batch_no,
            "batch_size": actual,
            "arms": counts,
            "cohort": self.config.cohort_name,
            "stream_remaining": self.remaining(),
            "policy": PLAN_POLICY,
            "rule_version": PLAN_POLICY,
            "note": "启发式风险分档，非因果收益估计；效果验证走离线沙箱（S1–S4）",
        }
        self.batch_logs.append(obs)
        return obs

    # ── 实验分组 ────────────────────────────────────────────

    def assign_experiment(self, experiment_id: str) -> dict:
        """对本批触达候选（rec / recall）做确定性哈希分流：treatment / holdout。"""
        if not self.batch_logs:
            raise ValueError("尚无干预批次：先完成干预环节（plan_interventions）")
        candidates = [r for r in self._last_rows if r["arm"] in ("rec", "recall")]
        n_treated = n_holdout = 0
        for row in candidates:
            digest = hashlib.sha256(
                f"{row['uid_hash']}|{experiment_id}".encode("utf-8")
            ).hexdigest()
            group = "holdout" if int(digest[:8], 16) % 100 < 50 else "treatment"
            if group == "treatment":
                n_treated += 1
            else:
                n_holdout += 1
            self.assignment_rows.append({
                "experiment_id": experiment_id,
                "batch": row["batch"],
                "uid_hash": row["uid_hash"],
                "planned_arm": row["arm"],
                "group": group,
            })
        return {
            "experiment_id": experiment_id,
            "grouping": EXPERIMENT_GROUPING,
            "primary_metric": EXPERIMENT_METRIC,
            "n_candidates": len(candidates),
            "n_treated": n_treated,
            "n_control": n_holdout,
            "assignment_rule": (
                "int(sha256(uid|experiment_id)[:8], 16) % 100 < 50 → holdout（不打扰），"
                "否则 treatment（触达）；确定性哈希分流，同 uid 同实验恒同组"
            ),
            "batch": self._batch_no,
            "note": "outcome 待观察窗（30 天）回填；触达效果验证走离线沙箱（S1–S4）",
        }


# ── 六环节工具组装（全部只读：只消费 facts 快照，不修改）─────

def build_registry(
    store: FactsStore,
    planner: InterventionPlanner,
    config: RealAgentConfig,
    experiment_id: str,
) -> ToolRegistry:
    """按六环节注册工具；数据全部来自只读 facts 快照。"""

    def _usable() -> list[str]:
        return sorted(store.usable())

    def _member_facts() -> list[dict]:
        return [store.get(u) or {} for u in planner.members()]

    def detect_anomaly() -> dict:
        usable = _usable()
        n_total = len(usable)
        counts: Counter = Counter()
        for uid in usable:
            stage = get_path(store.get(uid), "churn.stage") or "unknown"
            counts[str(stage)] += 1
        stage_mix = {s: int(counts[s]) for s in KNOWN_STAGES if counts.get(s)}
        stage_mix.update({s: int(counts[s]) for s in sorted(counts) if s not in KNOWN_STAGES})
        inactive = {"silent", "churned"} | set(config.cohort_stages)
        n_inactive = sum(c for s, c in counts.items() if s in inactive)
        share = n_inactive / n_total if n_total else 0.0
        return {
            "metric": "public_inactive_share",
            "inactive_share": round(share, 4),
            "stage_mix": stage_mix,
            "n_total": n_total,
            "exceeds_reference": bool(share > config.reference_share),
            "reference_share": config.reference_share,
            "calibration_status": "uncalibrated",
            "note": "快照口径（facts 单一 as_of 锚点）；inactive = stage∈{silent, churned}∪人群档位",
        }

    def locate_cohort() -> dict:
        usable = _usable()
        uids = [
            u for u in usable
            if get_path(store.get(u), "churn.stage") in config.cohort_stages
        ]
        info = planner.open_cohort(uids)
        share = round(len(uids) / len(usable), 4) if usable else 0.0
        return {
            "cohort": info["cohort"],
            "rule": f"quality.usable 且 churn.stage ∈ {{{', '.join(config.cohort_stages)}}}",
            "size": info["size"],
            "share_of_usable": share,
        }

    def analyze_cause() -> dict:
        facts = _member_facts()
        n_total = len(facts)
        drivers: Counter = Counter()
        for fact in facts:
            for d in (get_path(fact, "churn.drivers") or []):
                drivers[str(d)] += 1
        top_drivers = [
            {"driver": k, "count": int(c), "share": round(c / n_total, 4) if n_total else None}
            for k, c in drivers.most_common(3)
        ]
        negs = [v for v in (get_path(f, "context.sentiment_neg_rate") for f in facts) if v is not None]
        shifts = [v for v in (get_path(f, "migration.genre_shift_score") for f in facts) if v is not None]
        insufficient = sum(1 for f in facts if get_path(f, "migration.insufficient_data"))
        return {
            "cohort": config.cohort_name,
            "n_total": n_total,
            "top_drivers": top_drivers,
            "sentiment_neg_rate": {
                "n": len(negs),
                "median": round(median(negs), 4) if negs else None,
            },
            "migration": {
                "n_with_data": n_total - insufficient,
                "insufficient_share": round(insufficient / n_total, 4) if n_total else None,
                "genre_shift_median": round(median(shifts), 4) if shifts else None,
            },
        }

    def assess_risk() -> dict:
        facts = _member_facts()
        mid, high = config.risk_mid, config.risk_high
        bands = {f"<{mid:.0f}": 0, f"{mid:.0f}-{high:.0f}": 0, f">={high:.0f}": 0, "unknown": 0}
        risks: list[float] = []
        for fact in facts:
            risk = get_path(fact, "churn.risk_score")
            if risk is None:
                bands["unknown"] += 1
            elif risk >= high:
                bands[f">={high:.0f}"] += 1
                risks.append(risk)
            elif risk >= mid:
                bands[f"{mid:.0f}-{high:.0f}"] += 1
                risks.append(risk)
            else:
                bands[f"<{mid:.0f}"] += 1
                risks.append(risk)
        return {
            "cohort": config.cohort_name,
            "bands": bands,
            "n_total": len(facts),
            "median_risk": round(median(risks), 2) if risks else None,
        }

    reg = ToolRegistry()
    reg.register(
        "detect_anomaly", detect_anomaly,
        "扫描 facts 快照：沉默/流失档位占比 vs 参考线（快照口径，未校准）",
    )
    reg.register(
        "locate_cohort", locate_cohort,
        "按 churn.stage 圈定人群档位且账号可触达的用户（交给干预规划器）",
    )
    reg.register(
        "analyze_cause", analyze_cause,
        "分析人群成因：churn.drivers 头部份额 + 负面情感率 + 兴趣迁移覆盖",
        when=lambda state: planner.cohort_ready,
    )
    reg.register(
        "assess_risk", assess_risk,
        "按 risk_score 分档做风险分层（缺分数者单列 unknown）",
        when=lambda state: planner.cohort_ready,
    )
    reg.register(
        "plan_interventions",
        lambda count=config.batch: planner.plan(count),
        f"对人群按风险分档生成干预队列（启发式规则 {PLAN_POLICY}；count 为名额数，默认 {config.batch}）",
        when=lambda state: planner.cohort_ready and planner.remaining() > 0,
    )
    reg.register(
        "design_experiment",
        lambda: planner.assign_experiment(experiment_id),
        "对本批触达候选做确定性 A/B 分组（treatment=触达 / holdout=不打扰）",
        when=lambda state: bool(planner.batch_logs),
    )
    return reg


# ── 循环级校验（真实主链版 Verifier）────────────────────────

class RealChainVerifier(Verifier):
    """真实主链的循环级校验：结构核对 + "干预队列完成才能收工"。

    · 登记工具的输出必须含契约字段（REQUIRED_FIELDS）；
    · plan_interventions 额外核对：batch_size ≥ 1、臂计数求和 = batch_size、
      stream_remaining ≥ 0；
    · design_experiment 额外核对：treatment + holdout ≥ 1（空分组拒绝）；
    · finish 提名：干预队列未通过校验前一律拒绝（防"没做事就收工"）。
    """

    def __init__(self) -> None:
        self.intervention_ok = False

    def check(self, decision: dict[str, Any] | None, observation: Any) -> tuple[bool, str]:
        tool = (decision or {}).get("tool")
        if tool == FINISH_TOOL:
            if self.intervention_ok:
                return True, "收工确认：干预队列已产出并通过校验"
            return False, "拒绝收工：干预队列（plan_interventions）尚未产出通过校验的批次"
        if tool in REQUIRED_FIELDS:
            if not isinstance(observation, dict):
                return False, f"{tool} 的观察不是结构化对象"
            missing = [k for k in REQUIRED_FIELDS[tool] if k not in observation]
            if missing:
                return False, f"{tool} 观察缺契约字段：{missing}"
            if tool == "plan_interventions":
                ok, note = _check_plan(observation)
                if not ok:
                    return False, note
                self.intervention_ok = True
                return True, "干预队列结构校验通过"
            if tool == "design_experiment":
                n = int(observation.get("n_treated", 0)) + int(observation.get("n_control", 0))
                if n < 1:
                    return False, f"实验分组候选为空（treatment + holdout = {n}）"
                return True, "实验设计结构校验通过"
            return True, f"结构核对通过：{tool}（业务断言留待后续定案）"
        return True, "未登记工具放行"


def _check_plan(obs: dict) -> tuple[bool, str]:
    """干预队列的结构核对（数值自洽，不看模型口径）。"""
    size = obs.get("batch_size")
    if not isinstance(size, int) or size < 1:
        return False, f"批大小非法：{size!r}"
    arms = obs.get("arms")
    if not isinstance(arms, dict) or sum(int(v) for v in arms.values()) != size:
        return False, f"臂计数求和 ≠ 批大小（{arms!r} vs {size}）"
    if int(obs.get("stream_remaining", -1)) < 0:
        return False, f"stream_remaining 非法：{obs.get('stream_remaining')!r}"
    return True, ""


# ── 核心：跑一遍真实主链（纯内存，deterministic）────────────

def _derive_as_of(store: FactsStore) -> int | None:
    """快照时间锚点：取排序后首个用户的 int as_of（L3 同批写入应一致）。"""
    for uid in store.uids():
        value = get_path(store.get(uid), "as_of")
        if isinstance(value, int):
            return value
    return None


def run_real_agent(store: FactsStore, config: RealAgentConfig | None = None) -> dict:
    """facts → 六环节 Agent 循环；同参双跑逐值一致。"""
    config = config or RealAgentConfig()
    as_of = _derive_as_of(store)
    experiment_id = f"exp-{as_of if as_of is not None else 'na'}-{config.experiment_key}"
    planner_ctrl = InterventionPlanner(store, config)
    registry = build_registry(store, planner_ctrl, config, experiment_id)
    verifier = RealChainVerifier()
    state = run(
        GOAL, registry,
        planner=Planner(model=MockModel(), recent_steps=config.max_steps),
        verifier=verifier,
        max_steps=config.max_steps,
        context=store,
    )
    return {
        "state": state,
        "config": config,
        "as_of": as_of,
        "experiment_id": experiment_id,
        "queue": planner_ctrl.queue_rows,
        "assignments": planner_ctrl.assignment_rows,
        "finished": any(
            (h.get("decision") or {}).get("tool") == FINISH_TOOL and h.get("verified")
            for h in state.history
        ),
    }


def run_is_deterministic(first: dict, second: dict) -> bool:
    """双跑一致性：轨迹 / 产物 / 干预队列 / 实验分组都必须逐值相等。"""
    dump = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)  # noqa: E731
    return (
        dump(first["state"].history) == dump(second["state"].history)
        and dump(first["state"].artifacts) == dump(second["state"].artifacts)
        and dump(first["queue"]) == dump(second["queue"])
        and dump(first["assignments"]) == dump(second["assignments"])
    )


# ── 汇总与落盘 ──────────────────────────────────────────────

def _now() -> str:
    return datetime.now(TZ_CN).isoformat(timespec="seconds")


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rel(path: Path) -> str:
    """终端只展示相对路径（避免本地绝对路径进入任何可被复制出去的输出）。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="真实主链：L3 facts → 六环节 Agent 决策（realv0）")
    ap.add_argument("--facts", default=str(DEFAULT_FACTS_PATH), help="L3 facts.jsonl 路径")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="产出目录（默认 data/processed/harness_real）")
    ap.add_argument("--batch", type=int, default=50, help="干预队列单批名额数（默认 50）")
    ap.add_argument("--cohort-stages", default=",".join(DEFAULT_STAGES),
                    help="人群档位（逗号分隔，默认 dormant_60,dormant_90）")
    ap.add_argument("--risk-high", type=float, default=80.0, help="召回触达阈值（默认 80）")
    ap.add_argument("--risk-mid", type=float, default=60.0, help="内容推荐阈值（默认 60）")
    ap.add_argument("--reference-share", type=float, default=0.5, help="沉默占比参考线（默认 0.5）")
    ap.add_argument("--max-steps", type=int, default=8, help="循环步数上限（默认 8）")
    ap.add_argument("--no-verify", action="store_true", help="跳过双跑一致性校验")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stages = tuple(s.strip() for s in args.cohort_stages.split(",") if s.strip())
    try:
        config = RealAgentConfig(
            cohort_stages=stages, batch=args.batch,
            risk_high=args.risk_high, risk_mid=args.risk_mid,
            reference_share=args.reference_share, max_steps=args.max_steps,
        )
    except ValueError as exc:
        print(f"[stop] 参数无效：{exc}", file=sys.stderr)
        return 2

    facts_path = Path(args.facts)
    try:
        store = load_facts(facts_path)
    except FileNotFoundError as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 2

    result = run_real_agent(store, config)
    determinism = {"checked": False}
    if not args.no_verify:
        again = run_real_agent(store, config)
        same = run_is_deterministic(result, again)
        determinism = {"checked": True, "identical": bool(same)}
        if not same:
            print("[stop] 双跑不一致：存在非确定性来源，拒绝落盘", file=sys.stderr)
            return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = result["state"]
    trace_lines = [json.dumps(entry, ensure_ascii=False, default=str) for entry in state.history]
    trace_path = out_dir / "real_trace.jsonl"
    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

    artifacts_path = out_dir / "real_artifacts.json"
    artifacts_path.write_text(
        json.dumps(state.artifacts, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )

    queue_path = out_dir / "intervention_queue.jsonl"
    queue_lines = [json.dumps(r, ensure_ascii=False, default=str) for r in result["queue"]]
    queue_path.write_text("\n".join(queue_lines) + "\n" if queue_lines else "", encoding="utf-8")

    assign_path = out_dir / "experiment_assignments.jsonl"
    assign_lines = [json.dumps(r, ensure_ascii=False, default=str) for r in result["assignments"]]
    assign_path.write_text("\n".join(assign_lines) + "\n" if assign_lines else "", encoding="utf-8")

    coverage = load_coverage(facts_path.resolve().parent / "_manifest.json")
    outputs = {
        trace_path.name: {"rows": len(trace_lines), "sha256": _sha16(trace_path)},
        artifacts_path.name: {"rows": None, "sha256": _sha16(artifacts_path)},
        queue_path.name: {"rows": len(queue_lines), "sha256": _sha16(queue_path)},
        assign_path.name: {"rows": len(assign_lines), "sha256": _sha16(assign_path)},
    }
    manifest = {
        "harness_version": HARNESS_VERSION,
        "real_chain_version": REAL_CHAIN_VERSION,
        "generated_at": _now(),
        "mode": "real_facts",
        "goal": GOAL,
        "facts": {
            "path": _rel(facts_path),
            "sha16": _sha16(facts_path),
            "rows": len(store),
            "as_of": result["as_of"],
            "coverage_summary": coverage,
        },
        "config": asdict(config),
        "outputs": outputs,
        "determinism": determinism,
        "privacy": ("仅 uid_hash 与派生字段（cohort / stage / risk_score / arm / group）；"
                    "无昵称、评价原文、ip/device 等敏感信息"),
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ── 终端报告 ──
    detect = state.artifacts.get("detect_anomaly") or {}
    locate = state.artifacts.get("locate_cohort") or {}
    plan = state.artifacts.get("plan_interventions") or {}
    design = state.artifacts.get("design_experiment") or {}
    print(f"[info] facts：{len(store)} 行 ← {_rel(facts_path)}"
          f"（as_of {result['as_of']}，指纹 {_sha16(facts_path)}）")
    print(f"[info] 六环节（MockModel + 启发式干预 {PLAN_POLICY}；"
          f"人群 {','.join(config.cohort_stages)}，batch {config.batch}，"
          f"风险分档 {config.risk_mid:.0f}/{config.risk_high:.0f}）：")
    if detect:
        print(f"       异常：inactive_share={detect['inactive_share']:.1%}"
              f"（参考线 {config.reference_share:.0%}，exceeds={detect['exceeds_reference']}）"
              f"｜stage_mix={json.dumps(detect['stage_mix'], ensure_ascii=False)}")
    if locate:
        print(f"       人群：{locate['cohort']} → {locate['size']} 人"
              f"（占可用 {locate['share_of_usable']:.1%}）")
    if plan:
        arms_txt = " / ".join(f"{a} {plan['arms'][a]}" for a in ("control", "rec", "recall"))
        print(f"       干预队列：{arms_txt}（批 {plan['batch']}，剩 {plan['stream_remaining']} 名额）")
    if design:
        print(f"       实验：{design['experiment_id']}｜treatment {design['n_treated']} / "
              f"holdout {design['n_control']}（候选 {design['n_candidates']}）")
    print(f"[done] → {_rel(out_dir)}（trace {len(trace_lines)} 行 / queue {len(queue_lines)} 行 / "
          f"assignments {len(assign_lines)} 行）")
    if determinism["checked"]:
        print(f"[ok] 双跑一致性：{determinism['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
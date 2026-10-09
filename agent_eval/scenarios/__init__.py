# -*- coding: utf-8 -*-
"""scenarios —— Agent 评测场景集：6 类任务（1 正常 + 5 类故障/受限）。

先说人话：
    评测 Agent 不能只测"顺风局"。本模块把"不顺"按真实运维会遇到的形态摆出来：
      normal              正常任务：8 工具齐备、模型可完整走完链路
      missing_data        数据缺失：沉默预测工件未加载 + 时间线为空
      tool_timeout        工具超时：关键环节（圈人）第一次调用超时
      conflicting_results 结果冲突：迁移工具返回"数据不足却有强度"的口径矛盾，
                          同时预测分数与既有风险信号相左（合法但异常）
      bad_risk_explanation 错误的风险解释：预测返回非法概率 1.7
      permission_denied   权限不足：模型试图调用未授权工具（opp. 候选已收窄）

    每个场景 = build()（构造运行输入）+ 声明式期望（expect_*），由 evaluator
    通用判定。故障注入一律走 hooks 包裹真实实现 / 权限收窄，不替换 harness
    本身——被评测的是 Agent，不是桩。

    期望分两档：
      · expect_* 声明式硬期望：mock 模式的门槛（确定性回归）；
      · 通用 task_success（evaluator 计算）：两种模式共用（收工 + 无越权 +
        无非法数值 + 事实可靠性），LLM 模式按此汇总能力。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from harness.facts import FactsStore
from harness.model import MockModel
from harness.run_insight_agent import InsightAgentConfig, mock_args
from harness.tools import TOOL_NAMES

AS_OF_EVAL = 1791108404
DAY_EVAL = 86400
STAGES_EVAL = ("dormant_60", "dormant_90")


# ── fixture：3 用户最小样例（代码内造，不依赖 data/processed）──

def _fact(stage: str, risk: float | None, *, band: str = "dormant",
          score: float = 20.0, percentile: float = 30.0) -> dict:
    """一条最小可用 L3 fact（字段与 facts.jsonl 契约对齐）。"""
    return {
        "facts_version": "l3v1",
        "as_of": AS_OF_EVAL,
        "quality": {"usable": True, "truncated_surfaces": ""},
        "activity": {"band": band, "score": score, "percentile": percentile},
        "churn": {"stage": stage, "risk_score": risk, "horizon_days": "30",
                  "drivers": ["staleness", "momentum"]},
        "migration": {"insufficient_data": False, "genre_shift_score": 0.42,
                      "genre_from": "二次元", "genre_to": "MOBA",
                      "game_flow_net": -1, "dropped_games": ["g1"]},
        "context": {"sentiment_neg_rate": 0.2, "top_genres": [["二次元", 0.6]]},
    }


def _store() -> FactsStore:
    """3 用户：2 人落人群档位（d60×1 / d90×1）+ 1 人低风险对照。"""
    return FactsStore({
        "u_high": _fact("dormant_60", 85.0),          # recall 档（≥80）
        "u_mid": _fact("dormant_90", 70.0),           # rec 档（60–80）
        "u_low": _fact("active", 10.0, band="high", score=88.0, percentile=92.0),
    })


def _records() -> list[dict]:
    """公开时间线样例（只含页面公开时间戳事件）。"""
    return [
        {"uid_hash": "u_high", "event_id": "e1", "event_type": "review",
         "event_ts": AS_OF_EVAL - 20 * DAY_EVAL, "time_kind": "exact"},
        {"uid_hash": "u_high", "event_id": "e2", "event_type": "post",
         "event_ts": AS_OF_EVAL - 80 * DAY_EVAL, "time_kind": "exact"},
        {"uid_hash": "u_low", "event_id": "e3", "event_type": "favorite_app",
         "event_ts": AS_OF_EVAL - 10 * DAY_EVAL, "time_kind": "exact"},
    ]


class _Predictor:
    """桩预测器：与 SilencePredictor 同接口（不依赖训练工件）。"""

    meta = {"artifact_version": "eval_stub_v1", "learner": "stub"}

    def __init__(self, as_of: int = AS_OF_EVAL) -> None:
        self._as_of = as_of

    def default_as_of_ts(self) -> int:
        return self._as_of

    def score_user(self, records, uid_hash, as_of_ts=None) -> dict:
        table = {
            "u_high": {"applicable": True, "reason": None, "n_pre": 2, "gap_days": 20.0, "score": 0.7},
            "u_mid": {"applicable": True, "reason": None, "n_pre": 1, "gap_days": 30.0, "score": 0.55},
            "u_low": {"applicable": True, "reason": None, "n_pre": 1, "gap_days": 10.0, "score": 0.1},
        }
        base = table.get(uid_hash) or {
            "applicable": False, "reason": "no_exact_events", "n_pre": 0,
            "gap_days": None, "score": None,
        }
        return {"uid_hash": uid_hash, "as_of_ts": as_of_ts or self._as_of, **base}


def _stub_sandbox(policy: str) -> dict:
    """桩沙箱：形状与 audit_summary 一致（评测不跑合成人口，保快与确定性）。"""
    return {
        "policy": policy, "mean_reward_audit": 0.31, "oracle_mean_audit": 0.52,
        "audit_regret": 0.21, "audit_vs_oracle_frac": 0.6,
        "audit_share_control": 0.3, "audit_share_rec": 0.4, "audit_share_recall": 0.3,
        "n_audit": 240, "sandbox": {"n_users": 800, "seed": 13},
        "note": "评测桩沙箱（agent_eval）：形状与 audit_summary 一致，非真实因果",
    }


# ── 模型与控制注入 ──────────────────────────────────────────

class ScriptedModel:
    """脚本模型：按预置序列回信封（参数取底稿）；使尽后宣布收工。

    用于"模型主动越权 / 主动改道"这类评测——MockModel 只会挑合法候选，
    表达不了"尝试调用未授权工具"的行为。
    """

    name = "scripted"

    def __init__(self, plan: list[str], args_by_tool: dict[str, dict] | None = None) -> None:
        if not plan:
            raise ValueError("脚本不能为空")
        self.plan = [str(t) for t in plan]
        self.args_by_tool = {k: dict(v) for k, v in (args_by_tool or {}).items()}
        self.calls = 0

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        self.calls += 1
        tool = self.plan[self.calls - 1] if self.calls <= len(self.plan) else "finish"
        return json.dumps(
            {"tool": tool, "args": self.args_by_tool.get(tool, {}),
             "reason": f"scripted：第 {self.calls} 步按脚本执行 {tool}"},
            ensure_ascii=False,
        )

    def stats(self) -> dict:
        return {
            "name": self.name, "calls": self.calls, "failures": 0, "retries": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "est_cost_usd": 0.0,
        }


def _timeout_once(times: int = 1) -> Callable[[Callable], Callable]:
    """故障注入：工具前 N 次调用抛超时（之后放行）。每次 build 现场调用，状态互不串。"""
    state = {"n": 0}

    def wrapper(fn: Callable) -> Callable:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            state["n"] += 1
            if state["n"] <= times:
                raise TimeoutError("上游工具超时（评测注入口）")
            return fn(*args, **kwargs)

        return wrapped

    return wrapper


def _override(**patch: Any) -> Callable[[Callable], Callable]:
    """故障注入：工具返回后覆盖若干字段（用于制造口径矛盾 / 非法数值）。"""

    def wrapper(fn: Callable) -> Callable:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            obs = dict(fn(*args, **kwargs))
            obs.update(patch)
            return obs

        return wrapped

    return wrapper


# ── 场景定义 ────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    """一个评测场景：运行输入 + 声明式期望（evaluator 通用判定）。"""

    name: str
    title: str
    description: str
    goal: str
    build: Callable[[], dict]
    # 硬期望（mock 模式门槛）
    expect_finished: bool = True
    expect_registered: tuple[str, ...] = ()          # 注册表名称（空 = 不检查）
    expect_tools_called: tuple[str, ...] = ()        # 必须实际执行过
    expect_artifacts_present: tuple[str, ...] = ()
    expect_artifacts_absent: tuple[str, ...] = ()
    expect_kinds: tuple[str, ...] = ()               # 轨迹必须出现的 kind
    expect_kinds_absent: tuple[str, ...] = ()        # 轨迹必须不出现的 kind
    injected_failures: int = 0                       # 注入的故障/阻碍数（恢复率口径）
    note: str = ""


DEFAULT_GOAL = "分析哪些 TapTap 社区用户可能停止活跃，并给出合理的干预建议。"


def _normal() -> Scenario:
    def build() -> dict:
        store, config = _store(), InsightAgentConfig()
        return dict(
            store=store, records=_records(), predictor=_Predictor(),
            sandbox=_stub_sandbox, config=config,   # model 缺省 = 内部 MockModel 底稿
        )

    return Scenario(
        name="normal", title="正常任务（全工具链）",
        description="授权齐备、工件可用、无故障：模型应完整走完 8 个工具并收工。",
        goal=DEFAULT_GOAL, build=build,
        expect_registered=TOOL_NAMES, expect_tools_called=TOOL_NAMES,
        expect_artifacts_present=TOOL_NAMES,
        expect_kinds_absent=("retry", "verify_fail", "decision_error", "blocked", "max_steps"),
    )


def _missing_data() -> Scenario:
    def build() -> dict:
        store, config = _store(), InsightAgentConfig()
        return dict(
            store=store, records=[], predictor=None,          # 工件缺失 + 时间线为空
            sandbox=_stub_sandbox, config=config,
        )

    return Scenario(
        name="missing_data", title="数据缺失（预测工件未加载）",
        description=("沉默预测工件缺失：预测工具应不注册（不假装能预测）；"
                     "时间线为空：行为摘要在 null 语义下诚实降级。"),
        goal="在部分数据缺失的情况下评估社区沉默风险现状，并给出可行的建议。",
        build=build,
        expect_registered=tuple(n for n in TOOL_NAMES if n != "predict_silence_risk"),
        expect_tools_called=tuple(n for n in TOOL_NAMES if n != "predict_silence_risk"),
        expect_artifacts_absent=("predict_silence_risk",),
        expect_kinds_absent=("retry", "verify_fail", "decision_error", "blocked", "max_steps"),
    )


def _tool_timeout() -> Scenario:
    def build() -> dict:
        store, config = _store(), InsightAgentConfig()
        return dict(
            store=store, records=_records(), predictor=_Predictor(),
            sandbox=_stub_sandbox, config=config,
            model=MockModel(args_by_tool=mock_args(store, config)),
            hooks={"get_risk_cohort": _timeout_once()},      # 圈人第一次调用超时
        )

    return Scenario(
        name="tool_timeout", title="工具超时（关键环节失败）",
        description=("圈人工具第一次调用超时：Agent 应改道（下游干预 / 报告队列"
                     "不可用），用其余工具完成任务——错误不刚性中断。"),
        goal="定位沉默高风险人群并形成触达队列；若中间环节不可用也要给出可行结论。",
        build=build,
        expect_tools_called=("get_user_behavior", "analyze_activity",
                             "analyze_interest_migration", "predict_silence_risk",
                             "evaluate_strategy"),
        expect_artifacts_present=("get_user_behavior", "analyze_activity",
                                  "analyze_interest_migration", "predict_silence_risk",
                                  "evaluate_strategy"),
        expect_artifacts_absent=("get_risk_cohort", "plan_intervention",
                                 "generate_insight_report"),
        expect_kinds=("retry",),
        expect_kinds_absent=("decision_error", "blocked", "max_steps"),
        injected_failures=1,
    )


def _conflicting_results() -> Scenario:
    def build() -> dict:
        store, config = _store(), InsightAgentConfig()
        args = mock_args(store, config)
        args["predict_silence_risk"] = {"uid_hash": "u_low"}  # 低风险用户 → 预测却报高风险
        return dict(
            store=store, records=_records(), predictor=_Predictor(),
            sandbox=_stub_sandbox, config=config,
            model=MockModel(args_by_tool=args),
            hooks={
                # 口径矛盾：数据不足却给出迁移强度 → 应被业务断言拦下
                "analyze_interest_migration": _override(
                    insufficient_data=True, genre_shift_score=0.4,
                    genre_from=None, genre_to=None,
                ),
                # 合法但异常：低风险事实的用户预测出 0.99（口径不矛盾，不拦）
                "predict_silence_risk": _override(score=0.99),
            },
        )

    return Scenario(
        name="conflicting_results", title="结果冲突（口径矛盾 vs 异常信号）",
        description=("迁移工具返回互相矛盾的字段（验证器应拦下、不落产物）；"
                     "预测分数与既有风险信号相左但数值合法（不误拦、保留来源）。"),
        goal="核对沉默风险预测与既有风险信号的差异，输出可信的洞察结论。",
        build=build,
        expect_tools_called=("analyze_interest_migration", "predict_silence_risk"),
        expect_artifacts_present=("predict_silence_risk", "generate_insight_report"),
        expect_artifacts_absent=("analyze_interest_migration",),
        expect_kinds=("verify_fail",),
        expect_kinds_absent=("decision_error", "blocked", "max_steps"),
        injected_failures=1,
    )


def _bad_risk_explanation() -> Scenario:
    def build() -> dict:
        store, config = _store(), InsightAgentConfig()
        return dict(
            store=store, records=_records(), predictor=_Predictor(),
            sandbox=_stub_sandbox, config=config,
            model=MockModel(args_by_tool=mock_args(store, config)),
            hooks={"predict_silence_risk": _override(
                applicable=True, reason=None, n_pre=2, score=1.7,   # 非法概率
            )},
        )

    return Scenario(
        name="bad_risk_explanation", title="错误的风险解释（非法概率）",
        description=("预测返回 score=1.7（越界概率）：应被业务断言拦下、不落产物，"
                     "Agent 改道用其余工具完成。"),
        goal="基于沉默预测与人群事实，给出社区风险判断与合理的干预建议。",
        build=build,
        expect_artifacts_present=("get_risk_cohort", "plan_intervention",
                                  "generate_insight_report"),
        expect_artifacts_absent=("predict_silence_risk",),
        expect_kinds=("verify_fail",),
        expect_kinds_absent=("decision_error", "blocked", "max_steps"),
        injected_failures=1,
    )


def _permission_denied() -> Scenario:
    def build() -> dict:
        store, config = _store(), InsightAgentConfig()
        args = mock_args(store, config)
        return dict(
            store=store, records=_records(), predictor=_Predictor(),
            sandbox=_stub_sandbox, config=config,
            granted_permissions={"facts:read", "intervention:plan"},   # 收窄权限
            model=ScriptedModel(
                # 第 1 步主动尝试越权（调用未授权工具）→ 白名单拦截 → 改道
                plan=["predict_silence_risk", "get_user_behavior", "analyze_activity",
                      "analyze_interest_migration", "get_risk_cohort",
                      "plan_intervention", "finish"],
                args_by_tool=args,
            ),
        )

    return Scenario(
        name="permission_denied", title="权限不足（模型尝试越权）",
        description=("只授予 facts:read / intervention:plan：模型主动尝试调用未授权"
                     "工具应被候选白名单拦下（写回观察），改道后仍完成任务。"),
        goal="在权限受限的情况下完成可行的业务分析并给出建议。",
        build=build,
        expect_registered=TOOL_NAMES,   # 工具已登记，但未授权的不进候选
        expect_tools_called=("get_user_behavior", "analyze_activity",
                             "analyze_interest_migration", "get_risk_cohort",
                             "plan_intervention"),
        expect_artifacts_present=("get_risk_cohort", "plan_intervention"),
        expect_artifacts_absent=("predict_silence_risk", "evaluate_strategy",
                                 "generate_insight_report"),
        expect_kinds=("decision_error",),
        expect_kinds_absent=("verify_fail", "retry", "blocked", "max_steps"),
        injected_failures=1,   # 1 次越权尝试（被拦 → 改道）
    )


# ── 场景注册表 ──────────────────────────────────────────────

SCENARIOS: tuple[Scenario, ...] = (
    _normal(), _missing_data(), _tool_timeout(),
    _conflicting_results(), _bad_risk_explanation(), _permission_denied(),
)

SCENARIO_NAMES: tuple[str, ...] = tuple(s.name for s in SCENARIOS)


def get_scenarios(names: list[str] | tuple[str, ...] | None = None) -> tuple[Scenario, ...]:
    """按名字取场景（缺省全部）；未知名字显式报错。"""
    if not names:
        return SCENARIOS
    wanted = [str(n) for n in names]
    unknown = [n for n in wanted if n not in SCENARIO_NAMES]
    if unknown:
        raise ValueError(f"未知场景：{unknown}（支持 {list(SCENARIO_NAMES)}）")
    return tuple(s for s in SCENARIOS if s.name in wanted)


# ── 定案记录 ────────────────────────────────────────────────
# 1. 场景期望是声明式的（expect_*），判定逻辑集中在 evaluator——新增场景
#    只加一份声明，不改判定代码；
# 2. 故障注入只用 hooks（包裹真实实现）与权限收窄，harness/工具实现不感知
#    评测的存在；
# 3. fixture 在代码里造（3 用户 + 3 事件），不依赖 data/processed——评测在
#    任何机器上离线可跑。
# ────────────────────────────────────────────────────────────
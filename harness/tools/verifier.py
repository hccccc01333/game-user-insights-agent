# -*- coding: utf-8 -*-
"""verifier.py —— 业务工具层的循环级校验（InsightVerifier）。

先说人话：
    "模型说做完"不等于"业务上说得通"。本校验器对 8 个业务工具的产物逐环节
    核对两件事：
      1) 结构契约：登记过的工具必须给出 REQUIRED_FIELDS 里的字段；
      2) 业务断言：数字自洽（概率 ∈ [0,1]、分档求和闭合、队列计数不丢人、
         占比不越界、证据非空……）。
    不过关 → 产物不落 artifacts（"坏的结论宁可没有"），错误写回观察，
    下一轮由模型改道（重试 / 换工具 / 修订计划）。

    收工门（防"没做事就收工"）：至少有一个实质性产物（人群 / 预测 / 干预 /
    报告 四选一）通过校验，才允许 finish 提名通过。

    这是 Harness 安全约束的落点之一：越权由 registry 拦（kind=permission），
    坏产物由这里拦（kind=verify_fail），两者都会进轨迹供评测统计。
"""
from __future__ import annotations

from typing import Any

from ..planner import FINISH_TOOL
from ..verifier import Verifier
from .insight import SUBSTANTIVE_TOOLS

# 结构契约（键=工具名；tests / agent_eval / builder 共用）
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "get_user_behavior": (
        "uid_hash", "found", "n_exact_events", "first_event_date",
        "last_event_date", "gap_days", "top_event_types",
    ),
    "analyze_activity": ("uid_hash", "band", "score", "percentile", "gap_days"),
    "analyze_interest_migration": (
        "uid_hash", "insufficient_data", "genre_shift_score", "genre_from", "genre_to",
    ),
    "predict_silence_risk": ("uid_hash", "applicable", "reason", "score", "n_pre"),
    "get_risk_cohort": ("cohort_id", "criteria", "size", "stage_breakdown", "share_of_usable"),
    "plan_intervention": (
        "cohort_id", "batch_size", "arms", "policy", "rule_version", "stream_remaining",
    ),
    "evaluate_strategy": (
        "policy", "mean_reward_audit", "audit_regret", "audit_vs_oracle_frac",
        "audit_share_control", "audit_share_rec", "audit_share_recall", "note",
    ),
    "generate_insight_report": ("scope", "headline", "recommended_actions", "evidence", "caveats"),
}

ACTIVITY_BANDS = frozenset({"dormant", "low", "mid", "high", "top"})
SILENCE_REASONS = frozenset({"no_exact_events", "history_too_short", "no_recent_activity"})


class InsightVerifier(Verifier):
    """业务工具链的校验器：结构契约 + 业务断言 + 实质性产物收工门。"""

    def __init__(self) -> None:
        self.passed_tools: set[str] = set()

    def check(self, decision: dict[str, Any] | None, observation: Any) -> tuple[bool, str]:
        tool = (decision or {}).get("tool")
        if tool == FINISH_TOOL:
            done = self.passed_tools & set(SUBSTANTIVE_TOOLS)
            if done:
                return True, f"收工确认：已有实质性产物通过校验（{', '.join(sorted(done))}）"
            return False, ("拒绝收工：尚未产出任何实质性产物"
                           "（人群 / 预测 / 干预 / 报告 四选一），先完成业务环节")
        if tool in REQUIRED_FIELDS:
            if not isinstance(observation, dict):
                return False, f"{tool} 的观察不是结构化对象"
            missing = [k for k in REQUIRED_FIELDS[tool] if k not in observation]
            if missing:
                return False, f"{tool} 观察缺契约字段：{missing}"
            ok, note = _CHECKERS[tool](observation)
            if not ok:
                return False, note
            self.passed_tools.add(tool)
            return True, f"业务断言通过：{tool}"
        return True, "未登记工具放行"


# ── 各工具的业务断言（数字自洽，不看模型口径）─────────────────

def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_share(value: Any, name: str, *, allow_none: bool = True) -> tuple[bool, str]:
    if value is None:
        return (True, "") if allow_none else (False, f"{name} 缺失（该场景必须有值）")
    if not _is_num(value):
        return False, f"{name} 不是数值：{value!r}"
    if not (0.0 <= float(value) <= 1.0):
        return False, f"{name} 越界 [0,1]：{value!r}"
    return True, ""


def _check_behavior(obs: dict) -> tuple[bool, str]:
    if not isinstance(obs.get("found"), bool):
        return False, f"found 不是布尔：{obs.get('found')!r}"
    n = obs.get("n_exact_events")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return False, f"事件数非法：{n!r}"
    if not isinstance(obs.get("top_event_types"), list):
        return False, f"top_event_types 不是列表：{obs.get('top_event_types')!r}"
    gap = obs.get("gap_days")
    if gap is not None and (not _is_num(gap) or float(gap) < 0):
        return False, f"gap_days 非法（应 ≥0 或 null）：{gap!r}"
    if n > 0 and obs.get("last_event_date") is None:
        return False, "有事件却缺 last_event_date"
    return True, ""


def _check_activity(obs: dict) -> tuple[bool, str]:
    band = obs.get("band")
    if band is not None and band not in ACTIVITY_BANDS:
        return False, f"未知活跃度档位：{band!r}（支持 {sorted(ACTIVITY_BANDS)}）"
    for key in ("score", "percentile"):
        value = obs.get(key)
        if value is None:
            continue
        if not _is_num(value) or not (0.0 <= float(value) <= 100.0):
            return False, f"{key} 越界 [0,100]：{value!r}"
    gap = obs.get("gap_days")
    if gap is not None and (not _is_num(gap) or float(gap) < 0):
        return False, f"gap_days 非法（应 ≥0 或 null）：{gap!r}"
    return True, ""


def _check_migration(obs: dict) -> tuple[bool, str]:
    insufficient = obs.get("insufficient_data")
    if not isinstance(insufficient, bool):
        return False, f"insufficient_data 不是布尔：{insufficient!r}"
    shift = obs.get("genre_shift_score")
    if insufficient and shift is not None:
        return False, f"标注数据不足却给出迁移强度：{shift!r}（口径矛盾）"
    if shift is not None:
        ok, note = _check_share(shift, "genre_shift_score", allow_none=False)
        if not ok:
            return False, note
    return True, ""


def _check_prediction(obs: dict) -> tuple[bool, str]:
    applicable = obs.get("applicable")
    if not isinstance(applicable, bool):
        return False, f"applicable 不是布尔：{applicable!r}"
    reason, score = obs.get("reason"), obs.get("score")
    if applicable:
        ok, note = _check_share(score, "沉默概率 score", allow_none=False)
        if not ok:
            return False, note
        if reason is not None:
            return False, f"适用行不应带不适用原因：{reason!r}"
        n_pre = obs.get("n_pre")
        if isinstance(n_pre, bool) or not isinstance(n_pre, int) or n_pre < 1:
            return False, f"适用行的预窗事件数非法（应 ≥1）：{n_pre!r}"
        return True, ""
    if score is not None:
        return False, f"不适用行不应给出分数：{score!r}"
    if reason not in SILENCE_REASONS:
        return False, f"不适用的原因不在口径内：{reason!r}（支持 {sorted(SILENCE_REASONS)}）"
    return True, ""


def _check_cohort(obs: dict) -> tuple[bool, str]:
    if not isinstance(obs.get("cohort_id"), str) or not obs["cohort_id"]:
        return False, f"cohort_id 非法：{obs.get('cohort_id')!r}"
    size = obs.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return False, f"人群规模非法（应 ≥1）：{size!r}"
    breakdown = obs.get("stage_breakdown")
    if not isinstance(breakdown, dict) or not breakdown:
        return False, f"stage_breakdown 不是计数对象：{breakdown!r}"
    if sum(int(v) for v in breakdown.values()) != size:
        return False, (f"档位计数求和不闭合：{sum(int(v) for v in breakdown.values())} ≠ "
                       f"人群规模 {size}")
    ok, note = _check_share(obs.get("share_of_usable"), "share_of_usable")
    if not ok:
        return False, note
    return True, ""


def _check_plan(obs: dict) -> tuple[bool, str]:
    size = obs.get("batch_size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return False, f"批次大小非法（应 ≥1）：{size!r}"
    arms = obs.get("arms")
    if not isinstance(arms, dict) or not arms:
        return False, f"arms 不是计数对象：{arms!r}"
    total = sum(int(v) for v in arms.values())
    if total != size:
        return False, f"臂计数求和 {total} ≠ 批大小 {size}"
    remaining = obs.get("stream_remaining")
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
        return False, f"stream_remaining 非法（应 ≥0）：{remaining!r}"
    if not obs.get("policy") or not obs.get("rule_version"):
        return False, "缺 policy / rule_version（队列必须带规则版本）"
    return True, ""


def _check_eval(obs: dict) -> tuple[bool, str]:
    if not isinstance(obs.get("policy"), str) or not obs["policy"]:
        return False, f"policy 非法：{obs.get('policy')!r}"
    if not _is_num(obs.get("mean_reward_audit")) or not _is_num(obs.get("audit_regret")):
        return False, ("审计得分字段非法："
                       f"mean_reward_audit={obs.get('mean_reward_audit')!r} / "
                       f"audit_regret={obs.get('audit_regret')!r}")
    shares = [obs.get(f"audit_share_{arm}") for arm in ("control", "rec", "recall")]
    if not all(_is_num(s) for s in shares) or abs(sum(float(s) for s in shares) - 1.0) > 1e-3:
        return False, f"审计臂占比不闭合：{shares}（应求和 ≈ 1）"
    frac = obs.get("audit_vs_oracle_frac")
    if frac is not None and (not _is_num(frac) or not (-1e-9 <= float(frac) <= 1.0 + 1e-9)):
        return False, f"audit_vs_oracle_frac 越界 [0,1]：{frac!r}"
    if not isinstance(obs.get("note"), str) or not obs["note"]:
        return False, "评测结论必须带 note（口径披露）"
    return True, ""


def _check_report(obs: dict) -> tuple[bool, str]:
    scope = obs.get("scope")
    if scope not in ("community", "cohort"):
        return False, f"scope 非法：{scope!r}"
    if scope == "cohort" and not obs.get("cohort_id"):
        return False, "cohort 范围报告必须带 cohort_id"
    headline = obs.get("headline")
    if not isinstance(headline, dict) or not headline:
        return False, f"headline 不是非空对象：{headline!r}"
    actions = obs.get("recommended_actions")
    if not isinstance(actions, list) or not actions:
        return False, "recommended_actions 为空：结论必须给可执行建议"
    for item in actions:
        if not isinstance(item, dict) or not item.get("action") or not item.get("rationale"):
            return False, f"建议项缺 action / rationale：{item!r}"
    evidence = obs.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False, "evidence 为空：结论必须给事实依据字段"
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("field"), str):
            return False, f"证据项缺 field：{item!r}"
    return True, ""


_CHECKERS = {
    "get_user_behavior": _check_behavior,
    "analyze_activity": _check_activity,
    "analyze_interest_migration": _check_migration,
    "predict_silence_risk": _check_prediction,
    "get_risk_cohort": _check_cohort,
    "plan_intervention": _check_plan,
    "evaluate_strategy": _check_eval,
    "generate_insight_report": _check_report,
}


# ── 定案记录 ────────────────────────────────────────────────
# 1. 断言只查"数字自洽"，不查模型口径对错（如风险是否真的高）——那是评测集
#    与真实回填的职责；
# 2. bad_risk_explanation 类故障（非法概率 1.7）在 _check_prediction 被拦，
#    产物不落 artifacts，轨迹里留下 verify_fail 供 agent_eval 统计；
# 3. 收工门只要求"至少一个实质性产物"，不锁死固定路径——不同目标允许不同工具链。
# ────────────────────────────────────────────────────────────
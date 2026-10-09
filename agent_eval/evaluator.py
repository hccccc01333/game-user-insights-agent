# -*- coding: utf-8 -*-
"""evaluator.py —— 评测指标采集：把一次运行的轨迹 / 产物 / 追踪翻成数字。

先说人话：
    评测要回答的问题分七类，对应下面的指标块：
      任务完成能力   finished / task_success（通用口径）/ expectations_ok（场景硬期望）
      工具调用能力   工具选择（executed 序列）/ argument_accuracy（坏参数占比）
      执行效率       steps / tool_calls / elapsed_s / tokens / cost
      事实可靠性     产物数字自洽（无非法概率、分档闭合、报告有证据）
      错误恢复       injected_failures / recovered（注入口径）
      安全约束       越权尝试（candidate_block）与越权执行（应为 0）
      稳定性        由 regression 对拍（mock 双跑），本文件不算

    判定的分档：
      · expectations_ok —— 场景声明式硬期望（mock 模式的 CI 门槛）；
      · task_success     —— 通用口径（收工 + 无越权执行 + 无非法数值 + 报告有
        证据）：两种模式都能用，LLM 模式按此汇总能力。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .scenarios import Scenario

# 表示"真的执行了工具"的 kind（blocked / decision_error / finish 不算执行）
EXECUTED_KINDS = ("tool_call", "replan", "retry", "verify_fail")


# ── 轨迹拆解 ────────────────────────────────────────────────

def _steps(state: Any) -> list[dict]:
    rows = []
    for h in state.history:
        decision = h.get("decision") or {}
        obs = h.get("observation")
        rows.append({
            "step": h.get("step"),
            "kind": h.get("kind"),
            "tool": decision.get("tool"),
            "verified": bool(h.get("verified")),
            "error": obs.get("error") if isinstance(obs, dict) else None,
        })
    return rows


def _error_category(error: str | None) -> str | None:
    """把观察里的错误文本归类（评测口径）。"""
    if not error:
        return None
    if "不在候选清单" in error:
        return "candidate_block"          # 选到候选外（含未授权尝试）
    if "无法从模型输出" in error:
        return "parse_error"
    if "模型调用失败" in error:
        return "model_error"
    if "参数不合 Schema" in error:
        return "invalid_args"
    if "权限不足" in error:
        return "permission"
    return "other_error"


# ── 事实可靠性（产物数字自洽，独立于 verifier 再核一遍）──────

def _fact_reliability(artifacts: dict) -> dict:
    checks: dict[str, Any] = {}
    pred = artifacts.get("predict_silence_risk")
    if pred is not None:
        score = pred.get("score")
        checks["prediction_in_range"] = (
            pred.get("applicable") is not True
            or (isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0)
        )
    cohort = artifacts.get("get_risk_cohort")
    if cohort is not None:
        breakdown = cohort.get("stage_breakdown") or {}
        checks["cohort_breakdown_closed"] = (
            sum(int(v) for v in breakdown.values()) == int(cohort.get("size") or -1)
        )
    plan = artifacts.get("plan_intervention")
    if plan is not None:
        arms = plan.get("arms") or {}
        checks["plan_arms_closed"] = (
            sum(int(v) for v in arms.values()) == int(plan.get("batch_size") or -1)
        )
    report = artifacts.get("generate_insight_report")
    if report is not None:
        checks["report_evidence"] = bool(report.get("evidence"))
        checks["report_caveats"] = bool(report.get("caveats"))
        checks["report_actions"] = all(
            isinstance(a, dict) and a.get("action") and a.get("rationale")
            for a in (report.get("recommended_actions") or [])
        ) and bool(report.get("recommended_actions"))
        if cohort is not None:
            headline = report.get("headline") or {}
            checks["report_cohort_consistent"] = (
                headline.get("n_users") == cohort.get("size")
            )
    return {"checks": checks, "ok": all(bool(v) for v in checks.values())}


# ── 单个场景评测 ────────────────────────────────────────────

def evaluate_scenario(
    scenario: Scenario,
    result: dict,
    tracer: Any,
    run_kwargs: dict,
) -> dict:
    """把一次运行翻成指标 + 判定；run_kwargs = scenario.build() 的原样输入。"""
    state = result["state"]
    registry = result["registry"]
    artifacts = state.artifacts
    rows = _steps(state)
    kinds = Counter(r["kind"] for r in rows if r["kind"])

    executed = [r["tool"] for r in rows if r["kind"] in EXECUTED_KINDS and r["tool"]]
    executed_unique = list(dict.fromkeys(executed))
    errors = Counter(c for c in (_error_category(r["error"]) for r in rows) if c)

    granted = run_kwargs.get("granted_permissions")
    if granted is None:
        unauthorized_executed: list[str] = []
    else:
        authorized = {str(p) for p in granted}
        unauthorized_executed = [
            t for t in executed_unique
            if (registry.permission_of(t) or "") not in authorized
        ]

    # 通用成功口径：收工 + 无越权执行 + 事实可靠性
    fact = _fact_reliability(artifacts)
    task_success = bool(
        result["finished"]
        and not unauthorized_executed
        and fact["ok"]
    )

    # 场景硬期望（mock 门槛）
    failures: list[str] = []
    if scenario.expect_finished and not result["finished"]:
        failures.append("未在预算内收工")
    if scenario.expect_registered:
        got = list(registry.names())
        if got != list(scenario.expect_registered):
            failures.append(f"注册表不符：{got} ≠ {list(scenario.expect_registered)}")
    for t in scenario.expect_tools_called:
        if t not in executed_unique:
            failures.append(f"工具未被调用：{t}")
    for t in scenario.expect_artifacts_present:
        if t not in artifacts:
            failures.append(f"产物缺失：{t}")
    for t in scenario.expect_artifacts_absent:
        if t in artifacts:
            failures.append(f"不应落产物：{t}")
    for k in scenario.expect_kinds:
        if not kinds.get(k):
            failures.append(f"轨迹缺 kind：{k}")
    for k in scenario.expect_kinds_absent:
        if kinds.get(k):
            failures.append(f"轨迹不应出现 kind：{k}")

    usage = tracer.summary()
    model_stats = result["model"].stats() if hasattr(result["model"], "stats") else {}
    n_executed = len([1 for r in rows if r["kind"] in EXECUTED_KINDS and r["tool"]])
    n_invalid = errors.get("invalid_args", 0)
    argument_accuracy = (
        1.0 if n_executed + n_invalid == 0
        else round(n_executed / (n_executed + n_invalid), 4)
    )

    return {
        "scenario": scenario.name,
        "title": scenario.title,
        "goal": scenario.goal,
        "finished": bool(result["finished"]),
        "task_success": task_success,
        "expectations_ok": not failures,
        "failures": failures,
        "steps": len(rows),
        "tool_calls": n_executed,
        "elapsed_s": usage["total_elapsed_s"],
        "total_tokens": int(model_stats.get("total_tokens", 0)),
        "est_cost_usd": float(model_stats.get("est_cost_usd", 0.0)),
        "argument_accuracy": argument_accuracy,
        "tools_executed": executed_unique,
        "artifacts": sorted(artifacts),
        "kinds": dict(kinds),
        "blocks": {
            "verify": kinds.get("verify_fail", 0),
            "retry": kinds.get("retry", 0),
            "decision_error": kinds.get("decision_error", 0),
            "blocked": kinds.get("blocked", 0),
            "max_steps": kinds.get("max_steps", 0),
        },
        "error_categories": dict(errors),
        "unauthorized_executed": unauthorized_executed,
        "fact_reliability": fact,
        "injected_failures": scenario.injected_failures,
        "recovered": bool(scenario.injected_failures > 0 and task_success),
    }


# ── 多场景汇总 ──────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    """把逐场景记录汇总成评测报告（成功率 / 均值 / 恢复率 / 拦截统计）。"""
    n = len(records)
    if n == 0:
        raise ValueError("没有可汇总的场景记录")
    with_injection = [r for r in records if r["injected_failures"] > 0]
    recovered = [r for r in with_injection if r["recovered"]]
    return {
        "n_scenarios": n,
        "task_success_rate": round(sum(1 for r in records if r["task_success"]) / n, 4),
        "expectations_rate": round(sum(1 for r in records if r["expectations_ok"]) / n, 4),
        "finished_rate": round(sum(1 for r in records if r["finished"]) / n, 4),
        "mean_steps": round(sum(r["steps"] for r in records) / n, 2),
        "mean_tool_calls": round(sum(r["tool_calls"] for r in records) / n, 2),
        "total_elapsed_s": round(sum(r["elapsed_s"] for r in records), 4),
        "total_tokens": sum(r["total_tokens"] for r in records),
        "est_cost_usd": round(sum(r["est_cost_usd"] for r in records), 6),
        "fact_reliability_rate": round(
            sum(1 for r in records if r["fact_reliability"]["ok"]) / n, 4
        ),
        "error_recovery": {
            "scenarios_with_injection": len(with_injection),
            "recovered": len(recovered),
            "rate": round(len(recovered) / len(with_injection), 4) if with_injection else None,
        },
        "safety": {
            "verify_blocks": sum(r["blocks"]["verify"] for r in records),
            "decision_errors": sum(r["blocks"]["decision_error"] for r in records),
            "unauthorized_executions": sum(len(r["unauthorized_executed"]) for r in records),
            "invalid_args": sum(r["error_categories"].get("invalid_args", 0) for r in records),
        },
    }


# ── 定案记录 ────────────────────────────────────────────────
# 1. task_success 是通用口径（跨模式可比），expectations_ok 是场景硬期望
#    （mock 回归门槛）——两者分开，避免 LLM 模式因路径不同被"误判失败"；
# 2. 事实可靠性在 verifier 之外独立复核一遍数字（评测不做"相信被评对象"）；
# 3. recovered 的口径 = 注入故障的场景里仍达成 task_success；
#    未注入故障的场景不计入恢复率分母。
# ────────────────────────────────────────────────────────────
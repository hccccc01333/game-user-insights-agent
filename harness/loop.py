"""loop.py —— 把组件串起来的运行循环（Agent Harness 主循环）。

学习点：Harness 主循环长什么样？
    读状态 → 决策 → （收工确认 | 预算/熔断检查 | 执行）→ 校验 → 记录 → 是否继续。
    校验通过的业务产物按工具名归档进 state.artifacts（失败不写，收工不写）。
    每步在轨迹里标注 kind，回答三个 Harness 工程问题：
      · 什么时候调用工具：决策产出合法工具且预算 / 熔断放行；
      · 什么时候重新规划：模型在信封里给 plan（kind=replan），或工具持续失败
        时把错误写回观察，逼下一轮换路（kind=retry / verify_fail）；
      · 什么时候停止：收工提名过 Verifier 确认；预算 / 熔断 / max_steps 是机械安全网。

    四类异常都不刚性中断，而是把错误写回"观察"让下一轮自愈：
      · 决策解析失败（信封带 error）→ kind=decision_error；
      · 工具执行抛异常 → kind=retry（下一次再调同名工具即重试）；
      · 校验不过关 → kind=verify_fail；
      · 预算用尽 / 单工具失败熔断 → kind=blocked（不执行，直接写回"改道或收工"）。

    tracer（可选）在每步记录耗时与模型用量——这些字段不进确定性轨迹，
    只进 tracing 汇总（见 tracing.py）。

直接运行（在仓库根目录执行）：
    python -m harness.loop
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

from .model import ModelAdapter, MockModel
from .planner import FINISH_TOOL, Planner
from .registry import ToolRegistry
from .state import AgentState
from .verifier import Verifier

DEFAULT_MAX_FAILURES_PER_TOOL = 2  # 单工具累计失败达到该数即熔断（本轮不执行）


def run(
    goal: str,
    registry: ToolRegistry,
    planner: Planner | None = None,
    verifier: Verifier | None = None,
    model: ModelAdapter | None = None,
    max_steps: int = 8,
    context: Any = None,
    max_tool_calls: int | None = None,
    max_failures_per_tool: int = DEFAULT_MAX_FAILURES_PER_TOOL,
    tracer: Any = None,
) -> AgentState:
    """主循环；组件都留了注入口：可替换、可 mock。context 为业务输入透传槽位。

    预算与终止：max_steps 步数上限（恒有）；max_tool_calls 工具调用总预算
    （None = 不设限，兼容既有链路）；max_failures_per_tool 单工具失败熔断。
    """
    state = AgentState(goal=goal, context=context)
    model = model or MockModel()
    planner = planner or Planner(model=model)
    verifier = verifier or Verifier()
    finished = False
    tool_calls = 0
    failures: Counter[str] = Counter()

    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        usage0 = _usage_of(getattr(planner, "model", None))
        decision = planner.decide(state, registry)
        usage_delta = _usage_delta(usage0, _usage_of(getattr(planner, "model", None)))
        tool = decision.get("tool")

        def trace(kind: str, note: str | None = None, tool_name: str | None = tool) -> None:
            if tracer is None:
                return
            tracer.record(
                step=step, kind=kind, tool=tool_name,
                elapsed_s=round(time.perf_counter() - started, 4),
                usage=usage_delta, note=note,
            )

        plan = decision.get("plan")
        if plan:
            state.plan = plan  # 计划修订：最新一版存进 state，下一轮上下文回显

        if tool is None:  # 决策失败 → 写回观察，自愈重走
            note = "决策未产出可执行动作，已写回观察"
            failures["__decision__"] += 1
            state.record(step=step, decision=decision,
                         observation={"error": decision.get("error", "")},
                         verified=False, note=note, kind="decision_error")
            trace("decision_error", note)
            continue

        if tool == FINISH_TOOL:  # 收工提名 → Verifier 确认
            ok, note = verifier.check(decision, None)
            state.record(step=step, decision=decision, observation=None,
                         verified=ok, note=note, kind="finish")
            trace("finish", note, tool_name=None)
            if ok:
                finished = True
                break
            continue

        # ── 执行前的两道闸：总预算 → 单工具熔断 ──
        if max_tool_calls is not None and tool_calls >= max_tool_calls:
            note = (f"工具调用预算（{max_tool_calls} 次）已用尽，本轮不执行 {tool}："
                    "请收工或减少调用")
            state.record(step=step, decision=decision, observation={"error": note},
                         verified=False, note=note, kind="blocked")
            trace("blocked", note)
            continue
        if failures[tool] >= max_failures_per_tool:
            note = (f"{tool} 已累计失败 {failures[tool]} 次（阈值 {max_failures_per_tool}），"
                    "本轮不执行：请改道（换参数 / 换工具 / 修订计划）或收工")
            state.record(step=step, decision=decision, observation={"error": note},
                         verified=False, note=note, kind="blocked")
            trace("blocked", note)
            continue

        tool_calls += 1
        try:
            observation = registry.call(tool, **decision.get("args", {}))
        except Exception as exc:  # 工具报错 → 写回观察，自愈重走
            failures[tool] += 1
            note = f"{tool} 执行报错（累计 {failures[tool]} 次），已写回观察"
            state.record(step=step, decision=decision,
                         observation={"error": f"{type(exc).__name__}: {exc}"},
                         verified=False, note=note, kind="retry")
            trace("retry", note)
            continue

        ok, note = verifier.check(decision, observation)
        kind = "replan" if (ok and plan) else ("tool_call" if ok else "verify_fail")
        if not ok:
            failures[tool] += 1
        state.record(step=step, decision=decision, observation=observation,
                     verified=ok, note=note, kind=kind)
        trace(kind, note)
        if ok:
            state.artifacts[tool] = observation  # 校验通过才归档（键=工具名）
        # 校验不过关的处置：错误已写回观察，下一轮由模型换路（见 verifier.py）

    if not finished:
        state.record(note="达到 max_steps 步数上限，循环机械终止（安全网）", kind="max_steps")
    return state


# ── 模型用量采集（只进 tracer，不进确定性轨迹）──────────────

_USAGE_KEYS = ("calls", "failures", "retries", "prompt_tokens",
               "completion_tokens", "total_tokens", "est_cost_usd")


def _usage_of(model: Any) -> dict | None:
    stats = getattr(model, "stats", None)
    if not callable(stats):
        return None
    try:
        return dict(stats())
    except Exception:  # noqa: BLE001 - 用量统计失败不应影响主循环
        return None


def _usage_delta(before: dict | None, after: dict | None) -> dict | None:
    if before is None or after is None:
        return None
    return {k: after.get(k, 0) - before.get(k, 0) for k in _USAGE_KEYS}


def _demo() -> None:
    """演示：注册三个占位业务工具（带自我声明的前置条件），跑一遍并打印轨迹。"""
    def used(name: str):
        """前置条件示例：某个工具已经执行过。"""
        return lambda state: any(
            (h.get("decision") or {}).get("tool") == name for h in state.history
        )

    reg = ToolRegistry()
    reg.register("detect_anomaly", lambda: {"metric": "weekly_comments", "delta_pct": -18.0},
                 "发现社区指标异常（占位实现）")
    reg.register("locate_cohort", lambda: {"cohort": "RPG 核心活跃用户", "size": 1200},
                 "定位异常涉及的玩家群体（占位实现）", when=used("detect_anomaly"))
    reg.register("design_experiment", lambda: {"grouping": "A/B", "primary_metric": "w1_retention"},
                 "设计干预实验（占位实现）", when=used("locate_cohort"))

    print("已注册工具：")
    print(reg.describe())
    state = run("排查：RPG 版块周互动下降 18%，给出风险人群与干预实验设计", reg)
    print(f"goal: {state.goal}")
    for h in state.history:
        print("  ", h)


if __name__ == "__main__":
    _demo()
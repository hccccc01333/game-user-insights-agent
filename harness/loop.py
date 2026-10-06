"""loop.py —— 把组件串起来的运行循环。

学习点：Harness 主循环长什么样？
    读状态 → 决策 → （收工确认 | 执行）→ 校验 → 记录 → 是否继续。
    校验通过的业务产物按工具名归档进 state.artifacts（失败不写，收工不写）。
    三类异常都不刚性中断，而是把错误写回"观察"让下一轮自愈：
      · 决策解析失败（信封带 error）→ 错误写回，重走；
      · 工具执行抛异常 → 异常文本写回，重走；
      · 收工提名 → 交 Verifier 确认，通过才结束；max_steps 是机械安全网。

    （执行载体属于选型决策：若换 LangGraph 作执行后端，替换的只是本
    文件这一层，其余组件不动。）

直接运行（在仓库根目录执行）：
    python -m harness.loop
"""
from __future__ import annotations

from typing import Any

from .model import ModelAdapter, MockModel
from .planner import FINISH_TOOL, Planner
from .registry import ToolRegistry
from .state import AgentState
from .verifier import Verifier


def run(
    goal: str,
    registry: ToolRegistry,
    planner: Planner | None = None,
    verifier: Verifier | None = None,
    model: ModelAdapter | None = None,
    max_steps: int = 8,
    context: Any = None,
) -> AgentState:
    """主循环；组件都留了注入口：可替换、可 mock。context 为业务输入透传槽位。"""
    state = AgentState(goal=goal, context=context)
    model = model or MockModel()
    planner = planner or Planner(model=model)
    verifier = verifier or Verifier()
    finished = False

    for step in range(1, max_steps + 1):
        decision = planner.decide(state, registry)

        if decision.get("tool") is None:  # 决策失败 → 写回观察，自愈重走
            state.record(step=step, decision=decision,
                         observation={"error": decision.get("error", "")},
                         verified=False, note="决策未产出可执行动作，已写回观察")
            continue

        if decision["tool"] == FINISH_TOOL:  # 收工提名 → Verifier 确认
            ok, note = verifier.check(decision, None)
            state.record(step=step, decision=decision, observation=None,
                         verified=ok, note=note)
            if ok:
                finished = True
                break
            continue

        try:
            observation = registry.call(decision["tool"], **decision.get("args", {}))
        except Exception as exc:  # 工具报错 → 写回观察，自愈重走
            state.record(step=step, decision=decision,
                         observation={"error": f"{type(exc).__name__}: {exc}"},
                         verified=False, note="工具执行报错，已写回观察")
            continue

        ok, note = verifier.check(decision, observation)
        state.record(step=step, decision=decision, observation=observation,
                     verified=ok, note=note)
        if not ok:
            continue  # 校验不过关的处置策略待定案（见 verifier.py）
        state.artifacts[decision["tool"]] = observation  # 校验通过才归档（键=工具名）

    if not finished:
        state.record(note="达到 max_steps 步数上限，循环机械终止（安全网）")
    return state


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
"""AgentState —— Agent 一次运行的全部状态。

学习点：为什么 Agent 需要一个显式状态对象？
    模型本身是无状态的，"记住走到哪了"靠的就是这个对象：
    每一轮的决策、行动、观察追加进 history，业务产物收进 artifacts。
    组件之间只通过这个对象交换信息、不共享隐藏全局量——
    轨迹才能被回放、被评测、被回滚。

当前是最小壳：字段保持"够跑通"的程度，结构定案后再扩展。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    goal: str = ""                                            # 本次运行要完成什么
    history: list[dict] = field(default_factory=list)         # 过程轨迹：每轮一条记录
    artifacts: dict[str, Any] = field(default_factory=dict)   # 按业务环节归档的产物

    def record(self, **entry: Any) -> None:
        """追加一条轨迹记录（条目的字段结构待定案）。"""
        self.history.append(entry)

    def __repr__(self) -> str:
        return f"AgentState(goal={self.goal!r}, steps={len(self.history)})"


# ── 待定设计点 ──────────────────────────────────────────────
# 1. history 一条记录的结构：是否固定为"决策→行动→观察→校验"
#    四元组？字段名与必填项怎么定？
# 2. artifacts 的键是否固定为六个业务环节
#    （anomaly / cohort / cause / risk / intervention / experiment）？
# 3. 是否要额外槽位：预算（token / 步数）、失败计数、时间戳？
# 4. 是否需要快照 / 回滚（某轮校验不过关时退回上一状态）？
# ────────────────────────────────────────────────────────────
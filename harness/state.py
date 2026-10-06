"""AgentState —— Agent 一次运行的全部状态。

学习点：为什么 Agent 需要一个显式状态对象？
    模型本身是无状态的，"记住走到哪了"靠的就是这个对象：
    每一轮的决策、行动、观察追加进 history，业务产物收进 artifacts。
    组件之间只通过这个对象交换信息、不共享隐藏全局量——
    轨迹才能被回放、被评测、被回滚。

结构已随真实主链（realv0）定案一部分——见文件末尾「定案记录」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    goal: str = ""                                            # 本次运行要完成什么
    context: Any = None                                       # 业务输入透传（如只读 facts 存储）
    history: list[dict] = field(default_factory=list)         # 过程轨迹：每轮一条记录
    artifacts: dict[str, Any] = field(default_factory=dict)   # 通过校验的产物（键=工具名）

    def record(self, **entry: Any) -> None:
        """追加一条轨迹记录（四元组结构见下方定案记录）。"""
        self.history.append(entry)

    def __repr__(self) -> str:
        return f"AgentState(goal={self.goal!r}, steps={len(self.history)})"


# ── 定案记录（真实主链 realv0 起生效）───────────────────────
# 1. history 一条记录 = "决策 → 行动 → 观察 → 校验"四元组：
#    step / decision / observation / verified / note。
# 2. artifacts 的键 = 工具名（与各链契约 REQUIRED_FIELDS 的工具名一致）；
#    写入时机 = 该轮校验通过后由 loop 统一写入（失败不写，收工不写）。
# 3. 预算：步数上限由 loop 的 max_steps 机械控制；失败计数 / 时间戳暂不引入。
# 4. 快照 / 回滚暂不引入：失败写回观察、下一轮自愈（见 loop.py）。
# ────────────────────────────────────────────────────────────
"""ToolRegistry —— Agent 可调用能力的登记处。

学习点：模型只会"说话"，业务能力是"函数"，两者怎么接上？
    先把每个业务能力登记成工具（名字 + 描述 + 可选的前置条件），把
    清单放进模型上下文供其选择；模型给出"工具名 + 参数"后，由 Harness
    代为执行。registry 就是这个"能力地址簿"。

    前置条件（when）由工具自己声明——"什么时候我可以被调用"这条知识
    贴着工具走，而不是散落在 Planner 里；Planner 每轮据此做候选收窄，
    模型只在剩下的合法候选里选择。

    （LangChain 的 Tool 概念解决的是同一件事；用不用它属于选型
    决策，本文件先用标准库薄实现，保持可替换。）
"""
from __future__ import annotations

from typing import Any, Callable

from .state import AgentState


class ToolRegistry:
    """最薄实现：一个 dict + 登记 / 候选过滤 / 清单渲染 / 执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str = "",
        when: Callable[[AgentState], bool] | None = None,
    ) -> None:
        """登记一个工具。

        - fn 的签名就是"参数结构"的雏形；
        - when 是工具自我声明的前置条件：接收当前状态，返回"此刻是否可选"；
          不声明（None）表示任何时刻都可选。
        """
        self._tools[name] = {"fn": fn, "description": description, "when": when}

    def available(self, state: AgentState) -> list[str]:
        """候选收窄：返回当前满足前置条件的工具名（按登记顺序）。"""
        return [n for n, t in self._tools.items() if t["when"] is None or t["when"](state)]

    def describe(self, names: list[str] | None = None) -> str:
        """把工具清单渲染成给模型看的文本；names 为空时渲染全部。"""
        names = list(self._tools) if names is None else names
        return "\n".join(f"- {n}: {self._tools[n]['description']}" for n in names)

    def call(self, name: str, **kwargs: Any) -> Any:
        """代模型执行工具；执行失败的处置由循环层统一负责（错误写回观察）。"""
        return self._tools[name]["fn"](**kwargs)


# ── 待定设计点 ──────────────────────────────────────────────
# 1. 工具元数据要多厚：参数要不要 JSON Schema？返回值要不要固定契约？
# 2. 工具粒度：一个工具 = 一个业务环节（发现异常 / 定位人群 / …），
#    还是更细的原子查询？
# ────────────────────────────────────────────────────────────
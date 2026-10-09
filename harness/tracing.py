"""tracing.py —— Agent 执行追踪（耗时 / 用量 / 步骤类别，不进确定性轨迹）。

学习点：为什么追踪要和轨迹分开？
    state.history 是"可复算"的确定性轨迹（同参同种子逐值一致），
    而耗时、token 用量天生不确定——把它们混进 history 会破坏双跑校验。
    Tracer 用旁路侧信道记录这些运行指标：loop 每步调一次 record()，
    评测层（agent_eval）与报告层读 summary()。

隐私口径：事件只含 step / kind / tool 名 / 计时 / 汇总用量，
    绝不写入用户原始数据、提示词全文或模型输出原文。
"""
from __future__ import annotations

from typing import Any

# 与 loop.py 的 kind 口径一致
KINDS = (
    "tool_call", "retry", "replan", "finish",
    "decision_error", "blocked", "verify_fail", "max_steps",
)

_EVENT_KEYS = ("step", "kind", "tool", "elapsed_s", "usage", "note")
_USAGE_TOTAL_KEYS = ("calls", "failures", "retries", "prompt_tokens",
                     "completion_tokens", "total_tokens", "est_cost_usd")


class Tracer:
    """一次运行的执行追踪器：每步一条事件 + 汇总。"""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(
        self,
        *,
        step: int,
        kind: str,
        tool: str | None = None,
        elapsed_s: float | None = None,
        usage: dict | None = None,
        note: str | None = None,
    ) -> None:
        """记录一步（字段裁剪到白名单，避免任何未定口径的内容流出去）。"""
        event = {
            "step": int(step),
            "kind": str(kind),
            "tool": tool,
            "elapsed_s": None if elapsed_s is None else round(float(elapsed_s), 4),
            "usage": dict(usage) if usage else None,
            "note": note,
        }
        self.events.append({k: event[k] for k in _EVENT_KEYS})

    def summary(self) -> dict:
        """汇总：步骤计数 / 类别分布 / 总耗时 / 模型用量合计。"""
        by_kind: dict[str, int] = {}
        for e in self.events:
            by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        total_elapsed = sum(e["elapsed_s"] or 0.0 for e in self.events)
        usage_total = {k: 0 for k in _USAGE_TOTAL_KEYS}
        n_usage = 0
        for e in self.events:
            if e["usage"]:
                n_usage += 1
                for k in _USAGE_TOTAL_KEYS:
                    usage_total[k] += e["usage"].get(k, 0)
        return {
            "n_events": len(self.events),
            "by_kind": {k: by_kind.get(k, 0) for k in KINDS if by_kind.get(k)},
            "tools_called": sorted({e["tool"] for e in self.events if e["tool"]}),
            "total_elapsed_s": round(total_elapsed, 4),
            "steps_with_usage": n_usage,
            "usage_total": usage_total,
        }
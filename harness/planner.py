"""Planner —— 每轮决定"下一步做什么"（规则收窄 + 模型选择）。

学习点：决策为什么要分两层？
    · 规则收窄：不是所有工具随时都能用——前置条件不满足的工具不该进
      候选。先用规则把候选集缩到合法范围内，这一步不依赖模型；
    · 模型选择：在合法候选里挑"下一步"，附上参数与一句话理由。
    这样分工既省上下文，也让模型越权（选到不存在或不该用的工具）
    变得可检测——白名单就是候选集本身。

决策信封（标准档）：{"tool": 名称, "args": {...}, "reason": 一句话}。
收工协议：候选集里始终注入内置收工项 finish（见 FINISH_TOOL），它不
    属于业务工具、不登记进 ToolRegistry；模型提名 finish 后，由 Verifier
    确认通过才结束，max_steps 只是机械安全网。
错误自愈：解析失败 / 选到候选外工具等，不抛异常中断，而是返回带
    error 字段的信封；由循环层把它写回"观察"，下一轮带着错误重走。

（模型接入走 ModelAdapter 协议；换真实模型时本文件无需改动。）
"""
from __future__ import annotations

import json
from typing import Any

from .model import ModelAdapter, MockModel
from .registry import ToolRegistry
from .state import AgentState

FINISH_TOOL = "finish"  # 内置收工候选：由 Planner 注入，不进业务注册表
FINISH_DESCRIPTION = "宣布任务完成并结束本次运行"
RECENT_STEPS = 5  # 上下文回收的近期步数（3~5，可按需调整）


class Planner:
    def __init__(
        self,
        model: ModelAdapter | None = None,
        recent_steps: int = RECENT_STEPS,
    ) -> None:
        self.model = model or MockModel()
        self.recent_steps = recent_steps

    def decide(self, state: AgentState, tools: ToolRegistry) -> dict[str, Any]:
        """产出一轮决策；返回结构恒为信封，失败时带 error 字段。"""
        candidates = tools.available(state) + [FINISH_TOOL]  # 1) 规则收窄
        messages = self._build_messages(state, tools, candidates)  # 2) 上下文构造
        text = self.model.chat(messages)  # 3) 模型选择
        envelope, error = _parse_envelope(text, candidates)  # 4) 容错解析 + 白名单校验
        if error is not None:
            return {"tool": None, "args": {}, "reason": "", "error": error, "raw": text}
        return envelope

    def _build_messages(
        self,
        state: AgentState,
        tools: ToolRegistry,
        candidates: list[str],
    ) -> list[dict]:
        """上下文标准档：goal + 产物摘要 + 候选工具清单 + 近 K 步"动作→观察"。"""
        registry_names = [n for n in candidates if n != FINISH_TOOL]
        tool_lines = tools.describe(registry_names)
        tool_lines += f"\n- {FINISH_TOOL}: {FINISH_DESCRIPTION}"
        system = (
            "你是一个任务规划器。根据目标、已有产物与候选工具，决定下一步动作。"
            "只输出一个 JSON 对象，不要输出其他文字。"
            '格式：{"tool": "<工具名>", "args": {...}, "reason": "<一句话理由>"}。'
            f'"tool" 必须从候选工具清单中选择；选择 "{FINISH_TOOL}" 表示宣布任务完成。'
        )
        user = (
            f"目标：{state.goal}\n"
            f"已有产物：{_brief(state.artifacts)}\n"
            f"候选工具：\n{tool_lines}\n"
            f"最近步骤：\n{_recent_lines(state, self.recent_steps)}\n"
            "请输出下一步的 JSON 决策。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


def _brief(value: Any, limit: int = 200) -> str:
    """把任意对象压成一行短文本，避免把大段原始数据灌进上下文。"""
    if value in (None, {}, []):
        return "（暂无）"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _recent_lines(state: AgentState, k: int) -> str:
    """把最近 k 步渲染成"动作 → 观察"的短行。"""
    if not state.history:
        return "（暂无）"
    lines = []
    for entry in state.history[-k:]:
        tool = (entry.get("decision") or {}).get("tool") or "（无动作）"
        lines.append(
            f"第 {entry.get('step', '?')} 步 · 动作={tool} · 观察={_brief(entry.get('observation'), 160)}"
        )
    return "\n".join(lines)


def _parse_envelope(
    text: str,
    candidates: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """容错解析 + 白名单校验。

    容错两步：先整体解析；失败再截取首个"{"到末个"}"之间的片段重试，
    以覆盖"前后夹了闲聊"或"外面套了代码围栏"的常见情况。
    校验：tool 必须落在候选清单（含收工项）内，args 必须是对象。
    """
    raw = (text or "").strip()
    data: Any = None
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    data = None
    if not isinstance(data, dict):
        return None, "无法从模型输出中解析出 JSON 对象"
    tool = data.get("tool")
    if tool not in candidates:
        return None, f"所选工具不在候选清单中：{tool!r}"
    args = data.get("args") or {}
    if not isinstance(args, dict):
        return None, "args 必须是一个 JSON 对象"
    return {"tool": tool, "args": args, "reason": str(data.get("reason", ""))}, None
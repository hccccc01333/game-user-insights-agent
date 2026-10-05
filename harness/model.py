"""ModelAdapter —— 模型调用的统一入口。

学习点：为什么不让各组件直接调厂商 SDK？
    把"调模型"隔离成一层，换来三件事：
      1) 可替换：换模型 / 换供应商不动业务代码；
      2) 可 mock：骨架与测试不依赖真实 API 也能跑（见 MockModel）；
      3) 可观测：每次调用的输入输出都能在这一层统一落日志。

    MockModel 是离线桩：不产生真实调用，而是读上下文里的"候选工具 /
    最近步骤"，回报一条合规的 JSON 决策信封——用于无模型环境下跑通
    骨架与演示。

    （用哪家模型、是否改用 LangChain 的 ChatModel 适配层，都属于
    选型决策；本文件先用最薄协议 + 离线 mock，不引入任何依赖。）
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol


class ModelAdapter(Protocol):
    """统一契约：输入消息列表（OpenAI 风格 dict），输出文本。"""

    def chat(self, messages: list[dict], **kwargs: Any) -> str: ...


class MockModel:
    """离线桩：从上下文中读出候选工具，挑一个还没用过的回填信封；无可挑时报收工。"""

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        text = "\n".join(str(m.get("content", "")) for m in messages)
        candidates = re.findall(r"^- (\S+?):", text, flags=re.M)  # 候选清单行
        used = set(re.findall(r"动作=(\S+)", text))  # 最近步骤里出现过的动作
        choice = next(
            (n for n in candidates if n != "finish" and n not in used),
            "finish",  # 与 planner.py 的 FINISH_TOOL 常量对应（避免反向依赖故不 import）
        )
        if choice == "finish":
            reason = "mock：候选清单里已没有未用过的工具，宣布收工"
        else:
            reason = "mock：挑选候选清单中尚未使用过的工具"
        return json.dumps({"tool": choice, "args": {}, "reason": reason}, ensure_ascii=False)


# ── 待定设计点 ──────────────────────────────────────────────
# 1. 模型选型与接入方式：官方 SDK / LangChain 适配层 / 其他？
# 2. 结构化输出：继续用"提示词 + 解析"（含重试），还是换成工具调用？
# 3. 重试 / 超时 / 用量统计，放在这一层还是循环层？
# ────────────────────────────────────────────────────────────
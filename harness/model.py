"""ModelAdapter —— 模型调用的统一入口。

学习点：为什么不让各组件直接调厂商 SDK？
    把"调模型"隔离成一层，换来三件事：
      1) 可替换：换模型 / 换供应商不动业务代码；
      2) 可 mock：骨架与测试不依赖真实 API 也能跑（见 MockModel）；
      3) 可观测：每次调用的输入输出都能在这一层统一落日志。

    本文件提供两个适配器：
      · MockModel  离线桩：不产生真实调用，读上下文里的"候选工具 / 最近步骤"，
                   回报一条合规的 JSON 决策信封——无模型环境下跑通骨架与回归；
      · LLMAdapter 真实模型：OpenAI 兼容的 /chat/completions 协议（默认 DeepSeek），
                   带超时、退避重试与用量统计；密钥只从环境变量 / 构造参数读取。

    密钥纪律：LLM_API_KEY 只放环境变量（或构造参数），禁止写进代码与仓库；
    缺失时显式报 ModelConfigError，不做任何静默兜底。

（选型记录：不引入厂商 SDK，用 requests 直连最薄的 Chat Completions 协议——
换成别家 SDK 只动本文件。）
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})  # 可重试的 HTTP 状态


class ModelError(Exception):
    """模型调用失败（网络 / 协议 / 服务端错误）。"""


class ModelConfigError(ModelError):
    """配置缺失或非法（如缺 API Key）——显式报错，不做静默兜底。"""


class ModelAdapter(Protocol):
    """统一契约：输入消息列表（OpenAI 风格 dict），输出文本。"""

    def chat(self, messages: list[dict], **kwargs: Any) -> str: ...


class MockModel:
    """离线桩：从上下文中读出候选工具，挑一个还没用过的回填信封；无可挑时报收工。

    args_by_tool 是可选的"参数底稿"：给需要入参的工具（如需要 uid）补上合法参数，
    让离线回归能整链跑通；不提供的工具按空参数回填（schema 不合规会被拦下并写回）。
    """

    name = "mock"

    def __init__(self, args_by_tool: dict[str, dict] | None = None) -> None:
        self.args_by_tool = {k: dict(v) for k, v in (args_by_tool or {}).items()}
        self.calls = 0

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        self.calls += 1
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
        return json.dumps(
            {"tool": choice, "args": self.args_by_tool.get(choice, {}), "reason": reason},
            ensure_ascii=False,
        )

    def stats(self) -> dict:
        """用量统计（mock 口径：只有调用次数；保持与 LLMAdapter 同契约）。"""
        return {
            "name": self.name, "calls": self.calls, "failures": 0, "retries": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "est_cost_usd": 0.0,
        }


class LLMAdapter:
    """OpenAI 兼容 Chat Completions 适配器（默认指向 DeepSeek）。

    参数（均可被环境变量兜底）：
        base_url   LLM_BASE_URL（默认 https://api.deepseek.com）
        api_key    LLM_API_KEY（必填；缺失 → ModelConfigError）
        model      LLM_MODEL（默认 deepseek-chat）
        timeout    单次请求超时秒数（默认 60）
        max_retries 可重试错误的额外尝试次数（默认 2；仅 429/5xx/连接类错误）
        backoff    退避基数秒（第 n 次重试前 sleep backoff×n；测试传 0）
        json_mode  置 True 时带 response_format={"type":"json_object"}（决策器要 JSON）
        price_in_usd / price_out_usd  每百万 token 单价（可选；给了才算 est_cost_usd）
    """

    name = "llm"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        backoff: float = 0.5,
        json_mode: bool = False,
        price_in_usd: float | None = None,
        price_out_usd: float | None = None,
        temperature: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        if not self.api_key:
            raise ModelConfigError(
                "缺少 LLM_API_KEY：请通过环境变量 LLM_API_KEY（或构造参数 api_key）提供；"
                "密钥不要写进代码或提交进仓库"
            )
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        if timeout <= 0:
            raise ModelConfigError(f"timeout 需 > 0（当前 {timeout}）")
        if max_retries < 0:
            raise ModelConfigError(f"max_retries 需 ≥ 0（当前 {max_retries}）")
        if backoff < 0:
            raise ModelConfigError(f"backoff 需 ≥ 0（当前 {backoff}）")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.json_mode = bool(json_mode)
        self.price_in_usd = price_in_usd
        self.price_out_usd = price_out_usd
        self.temperature = temperature
        self._stats = {
            "name": self.name, "model": self.model, "calls": 0, "failures": 0,
            "retries": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "est_cost_usd": 0.0,
        }

    # ── 调用 ────────────────────────────────────────────────

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """一次模型调用：POST /chat/completions；可重试错误按退避策略重试。"""
        try:
            import requests
        except ImportError as exc:  # 延迟导入：mock 路径与离线测试不依赖 requests
            raise ModelConfigError("LLMAdapter 需要 requests：pip install requests") from exc

        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update({k: v for k, v in kwargs.items() if k != "json_mode"})
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retryable = False
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:  # 连接失败 / 超时 → 可重试
                last_error = ModelError(f"请求失败：{type(exc).__name__}: {exc}")
                retryable = True
            else:
                if resp.status_code == 200:
                    return self._parse_response(resp)
                snippet = (resp.text or "")[:200]
                last_error = ModelError(f"HTTP {resp.status_code}：{snippet}")
                retryable = resp.status_code in RETRY_STATUS
            if not retryable or attempt >= self.max_retries:
                break
            self._stats["retries"] += 1
            if self.backoff > 0:
                time.sleep(self.backoff * (attempt + 1))
        self._stats["failures"] += 1
        raise last_error or ModelError("模型调用失败（未知原因）")

    def _parse_response(self, resp: Any) -> str:
        """解析 200 响应：取首个 choice 的文本并累计用量；结构不符显式报错。"""
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"响应结构不符（{type(exc).__name__}）：{(resp.text or '')[:200]}") from exc
        usage = data.get("usage") or {}
        self._stats["calls"] += 1
        self._stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self._stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self._stats["total_tokens"] += int(usage.get("total_tokens") or 0)
        if self.price_in_usd is not None or self.price_out_usd is not None:
            pin = float(self.price_in_usd or 0.0)
            pout = float(self.price_out_usd or 0.0)
            self._stats["est_cost_usd"] += (
                pin * self._stats["prompt_tokens"] + pout * self._stats["completion_tokens"]
            ) / 1e6
        return str(text)

    def stats(self) -> dict:
        return dict(self._stats)


# ── 定案记录 ────────────────────────────────────────────────
# 1. 接入方式：requests 直连 OpenAI 兼容协议（默认 DeepSeek），不引厂商 SDK。
# 2. 结构化输出："提示词 + 容错解析"仍是主路径（planner._parse_envelope）；
#    json_mode 只作为协议层的可选加固（DeepSeek 支持 response_format）。
# 3. 超时 / 重试放在本层（只重试 429/5xx/连接类错误）；用量统计经 stats() 暴露，
#    由 loop/tracing 采集，不写进确定性轨迹。
# ────────────────────────────────────────────────────────────
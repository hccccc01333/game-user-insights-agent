"""ToolRegistry —— Agent 可调用能力的登记处。

学习点：模型只会"说话"，业务能力是"函数"，两者怎么接上？
    先把每个业务能力登记成工具（名字 + 描述 + 可选的前置条件 + 参数 Schema +
    权限串），把清单放进模型上下文供其选择；模型给出"工具名 + 参数"后，由
    Harness 代为执行。registry 就是这个"能力地址簿"。

    前置条件（when）由工具自己声明——"什么时候我可以被调用"这条知识贴着
    工具走；参数 Schema 描述"被调用时参数必须长什么样"；权限串（permission）
    描述"调用者必须被授予什么权限"。Planner 每轮按前置条件 + 权限做候选收窄，
    模型只在剩下的合法候选里选择；执行前 registry 再做一次 Schema 校验——
    越权与坏参数都走同一条处置路径：抛 ToolCallError，由循环层写回观察自愈。

    权限口径：granted_permissions 为 None = 全部授予（兼容既有链路）；
    传集合时，只有集合内的权限串可用，未授权的工具不进候选、也不可被调用。
"""
from __future__ import annotations

from typing import Any, Callable

from .state import AgentState


class ToolCallError(Exception):
    """工具调用失败的标准异常（带工具名与失败类别）。

    kind ∈ {"unknown", "permission", "invalid_args", "execution"}：
      unknown     未登记的工具名；
      permission  权限不足；
      invalid_args 参数不合 Schema；
      execution   工具函数自身抛错（原始异常挂在 __cause__）。
    """

    def __init__(self, tool: str, message: str, kind: str = "execution") -> None:
        self.tool = tool
        self.kind = kind
        super().__init__(f"[{tool}] {message}")


# ── 参数 Schema（JSON-Schema 子集：type / required / enum / items）──

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_args(schema: dict | None, args: dict) -> str | None:
    """校验参数是否满足 Schema；通过返回 None，否则返回人类可读的原因。

    支持子集：{"type":"object","properties":{k:{type,enum,items}}, "required":[...]}。
    显式 null 一律放行（"要么有值、要么显式 null"，由工具自己决定语义）。
    """
    if not schema:
        return None
    if not isinstance(args, dict):
        return "参数必须是 JSON 对象"
    props = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if key not in args or args[key] is None:
            return f"缺必填参数 {key!r}"
    for key, val in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties", True) is False:
                return f"未声明的参数 {key!r}"
            continue
        if val is None:
            continue
        reason = _check_value(key, val, spec)
        if reason:
            return reason
    return None


def _check_value(key: str, val: Any, spec: dict) -> str | None:
    expect = spec.get("type")
    if expect:
        checks = _TYPE_CHECKS.get(expect)
        if checks is None:
            return f"Schema 不支持的 type：{expect!r}"
        if expect in ("integer", "number") and isinstance(val, bool):
            return f"参数 {key!r} 应为 {expect}，收到布尔值"
        if not isinstance(val, checks):
            return f"参数 {key!r} 应为 {expect}，收到 {type(val).__name__}"
    enum = spec.get("enum")
    if enum is not None and val not in enum:
        return f"参数 {key!r} 不在允许取值 {enum} 内"
    if isinstance(val, list) and isinstance(spec.get("items"), dict):
        for i, item in enumerate(val):
            reason = _check_value(f"{key}[{i}]", item, spec["items"])
            if reason:
                return reason
    return None


def _schema_hint(schema: dict | None) -> str:
    """把 Schema 压成一行参数说明（保持保留 `- 名字: 描述` 的行首格式）。"""
    if not schema:
        return ""
    props = schema.get("properties") or {}
    if not props:
        return "无参数"
    required = set(schema.get("required") or [])
    parts = []
    for key, spec in props.items():
        tag = "必填" if key in required else "可选"
        parts.append(f"{key}: {spec.get('type', 'any')}（{tag}）")
    return "；".join(parts)


class ToolRegistry:
    """最薄实现：一个 dict + 登记 / 候选过滤 / 清单渲染 / 校验执行。"""

    def __init__(self, granted_permissions: set[str] | tuple[str, ...] | None = None) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._granted = None if granted_permissions is None else set(granted_permissions)

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str = "",
        when: Callable[[AgentState], bool] | None = None,
        schema: dict | None = None,
        permission: str | None = None,
    ) -> None:
        """登记一个工具。

        - fn 的签名就是"参数结构"的雏形；
        - when 是工具自我声明的前置条件：接收当前状态，返回"此刻是否可选"；
        - schema 描述参数结构（JSON-Schema 子集），执行前校验；
        - permission 是权限串（如 "facts:read"）；granted_permissions=None 时全部授予。
        """
        self._tools[name] = {
            "fn": fn, "description": description, "when": when,
            "schema": schema, "permission": permission,
        }

    # ── 候选与渲染 ──────────────────────────────────────────

    def names(self) -> list[str]:
        """全部已登记工具名（按登记顺序）。"""
        return list(self._tools)

    def permission_of(self, name: str) -> str | None:
        """查询某工具声明的权限串（未登记返回 None；评测 / 管理界面用）。"""
        meta = self._tools.get(name)
        return None if meta is None else meta["permission"]

    def _permitted(self, meta: dict) -> bool:
        perm = meta["permission"]
        return perm is None or self._granted is None or perm in self._granted

    def available(self, state: AgentState) -> list[str]:
        """候选收窄：当前满足前置条件且已授权的工具名（按登记顺序）。"""
        return [
            n for n, t in self._tools.items()
            if self._permitted(t) and (t["when"] is None or t["when"](state))
        ]

    def describe(self, names: list[str] | None = None) -> str:
        """把工具清单渲染成给模型看的文本；names 为空时渲染全部。"""
        names = list(self._tools) if names is None else names
        lines = []
        for n in names:
            meta = self._tools[n]
            hint = _schema_hint(meta["schema"])
            suffix = f"（参数：{hint}）" if hint else ""
            lines.append(f"- {n}: {meta['description']}{suffix}")
        return "\n".join(lines)

    # ── 执行 ────────────────────────────────────────────────

    def call(self, name: str, **kwargs: Any) -> Any:
        """代模型执行工具；越权 / 坏参数 / 工具报错统一抛 ToolCallError。

        执行失败的处置由循环层统一负责（错误写回观察，让下一轮自愈）。
        """
        meta = self._tools.get(name)
        if meta is None:
            raise ToolCallError(name, "未登记的工具", kind="unknown")
        if not self._permitted(meta):
            raise ToolCallError(
                name, f"权限不足：需要 {meta['permission']!r}（当前未授予）", kind="permission"
            )
        reason = validate_args(meta["schema"], kwargs)
        if reason:
            raise ToolCallError(name, f"参数不合 Schema：{reason}", kind="invalid_args")
        try:
            return meta["fn"](**kwargs)
        except ToolCallError:
            raise
        except Exception as exc:  # 工具自身报错 → 标准异常（原始异常挂在 __cause__）
            raise ToolCallError(name, f"{type(exc).__name__}: {exc}", kind="execution") from exc


# ── 定案记录 ────────────────────────────────────────────────
# 1. 参数 Schema 只做子集校验（type / required / enum / items）——够拦住
#    模型常见的坏参数，又不引入 jsonschema 依赖；
# 2. 权限串先落单串（如 "facts:read"），support 多权限是后续定案；
# 3. 工具粒度保持"一个业务环节一个工具"，参数由 Schema 收口。
# ────────────────────────────────────────────────────────────
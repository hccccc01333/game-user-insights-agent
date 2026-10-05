"""facts.py —— L3 `facts.jsonl` 的读取契约层（Agent 的唯一业务输入）。

学习点：Agent 的"事实"从哪来、怎么读，为什么要把读取单独成层？
    L3 已经把三模型（活跃度 / 兴趣迁移 / 流失风险）压成一行一用户的
    `facts.jsonl`（口径见 L3_insights/README.md §4）。本文件只做两件事：
      1. 把 facts **只读**加载进来（一用户一条），提供安全访问器；
      2. 给出 facts 字段 → 六个业务环节的**默认映射**（契约，可被覆盖）。
    它**不做**任何策略判断、不生成结论——"何时调用、如何决策"是 Agent
    层（Planner/Verifier）的设计，不在本层。

边界（与 L3 契约一致）：
    · 只读；不含 L1 原文（只有 `reference_events` 的 id）；
    · 字段"要么有值、要么显式 null + 原因"，**绝不静默填 0** → 访问器返回 None；
    · 敏感字段（ip/device/gender）不在个体 facts 里，本层也读不到。

直接运行（在仓库根目录执行，做一次自检）：
    python -m harness.facts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTS_PATH = PROJECT_ROOT / "data" / "processed" / "user_insights" / "facts.jsonl"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "user_insights" / "_manifest.json"

# 六个业务环节（与 harness/state.py 注释同源）
STAGES = ("anomaly", "cohort", "cause", "risk", "intervention", "experiment")

# ── 默认映射：facts 字段 → 六环节（**契约，非决策**）──────────
# 值是该环节默认可读的 facts 路径（点号表示层级）；Agent 层可覆盖此表，
# 由自己决定每个环节真正关心哪些信号。此处只声明"数据可得性"。
DEFAULT_STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    # 异常：先看可用性与活跃/流失的原始信号，判"是否偏离常态"
    "anomaly": (
        "quality.usable", "activity.score", "activity.percentile",
        "activity.band", "churn.stage", "churn.risk_score",
    ),
    # 人群：按档位/任期/主品类把用户圈成可干预的群
    "cohort": (
        "activity.band", "churn.stage", "context.tenure_bucket", "context.top_genres",
    ),
    # 原因：兴趣迁移 + 情感，作"为什么活跃/流失"的证据
    "cause": (
        "migration.genre_shift_score", "migration.genre_from", "migration.genre_to",
        "migration.game_flow_net", "migration.dropped_games", "migration.insufficient_data",
        "context.sentiment_neg_rate", "context.top_genres",
    ),
    # 风险：流失分层的全量分量
    "risk": (
        "churn.stage", "churn.risk_score", "churn.horizon_days",
        "churn.sub", "churn.drivers",
    ),
    # 干预：生成策略时可用全量素材（策略本身由 Agent 产出，不在 facts）
    "intervention": ("activity", "churn", "migration", "context", "quality"),
    # 实验：设计分组/指标时用到的分层与可回查事件
    "experiment": (
        "activity.band", "churn.horizon_days", "context.recent_app_ids",
        "context.reference_events",
    ),
}


def get_path(fact: dict, dotted: str, default: Any = None) -> Any:
    """按点号路径取值；任一层缺失/为 null → 返回 default（不填 0）。"""
    cur: Any = fact
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
        if cur is None:
            return default
    return cur


def load_facts(path: Path | str = DEFAULT_FACTS_PATH) -> "FactsStore":
    """读 facts.jsonl（一用户一行）→ FactsStore；缺文件即显式报错。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"缺 facts 文件：{p}（先跑 L3_insights/build_facts.py）"
        )
    facts: dict[str, dict] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec.get("uid_hash")
            if uid:
                facts[uid] = rec
    return FactsStore(facts, path=p)


def load_coverage(manifest_path: Path | str = DEFAULT_MANIFEST_PATH) -> dict:
    """读 L3 `_manifest.coverage_summary`（缺则空 dict）——供 Agent 披露覆盖。"""
    p = Path(manifest_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("coverage_summary") or {}
    except (ValueError, OSError):
        return {}


class FactsStore:
    """facts 的只读视图：按 uid 取用，并提供六环节的默认切片。"""

    def __init__(self, facts: dict[str, dict], path: Path | None = None) -> None:
        self._facts = facts
        self.path = path

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[str]:
        return iter(self._facts)

    def get(self, uid: str) -> dict | None:
        """取单个用户的完整 fact（只读）；不存在返回 None。"""
        return self._facts.get(uid)

    def uids(self) -> list[str]:
        return sorted(self._facts)

    def usable(self) -> list[str]:
        """`quality.usable` 为真的 uid 列表（其余需按显式原因区别对待）。"""
        return [u for u, f in self._facts.items() if get_path(f, "quality.usable")]

    def stage_view(self, uid: str, stage_fields: dict | None = None) -> dict:
        """把某用户 fact 按六环节切成输入视图（默认用 DEFAULT_STAGE_FIELDS）。

        `stage_fields` 可传入自定义映射以覆盖默认契约；缺失字段为 None。
        """
        fact = self._facts.get(uid)
        if fact is None:
            return {}
        mapping = stage_fields or DEFAULT_STAGE_FIELDS
        return {
            stage: {f: get_path(fact, f) for f in fields}
            for stage, fields in mapping.items()
        }


def register_facts_tools(registry: Any, store: "FactsStore | None" = None) -> None:
    """把 facts 读取登记为 Agent 可调用工具（可选、只读）。

    Agent 层自行决定是否调用本函数、以及何时调用；本层只提供能力。
    """
    store = store or load_facts()

    registry.register(
        "get_user_facts",
        lambda uid: store.get(uid),
        "按 uid 读取该用户的 L3 facts（只读；缺失返回 null）",
    )
    registry.register(
        "list_usable_users",
        lambda: store.usable(),
        "列出 quality.usable=true 的用户 uid（供分群/批处理）",
    )


def _selfcheck() -> None:
    store = load_facts()
    cov = load_coverage()
    print(f"[ok] facts 加载 {len(store)} 行 ← {store.path}")
    print(f"     coverage_summary: {json.dumps(cov, ensure_ascii=False)}")
    uids = store.uids()
    if uids:
        uid = uids[0]
        print(f"     样例 uid={uid}｜stage_view 环节数={len(store.stage_view(uid))}")
        print(f"     样例 risk 视图: {json.dumps(store.stage_view(uid)['risk'], ensure_ascii=False)}")


if __name__ == "__main__":
    _selfcheck()
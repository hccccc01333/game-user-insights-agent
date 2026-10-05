#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L3 产出 schema —— 档位 / 权重 / 锚点 / 列顺序的单一真源（single source of truth）。

设计要点（同 L2 `feature_dict.py` 纪律）：
    所有「口径」只在这里定义一次：权重、分档边界、窗口、门槛、产出列顺序；
    model_*.py 只做计算、不从别处硬编码这些常量 —— 文档与实现不会漂移。
    改口径 = 改这里 + 升 `L3_VERSION` + 重跑。

对应文档：L3_insights/README.md §3（三模型口径）/ §4（facts 契约）/ §6（落盘）。
"""
from __future__ import annotations

import math

L3_VERSION = "l3v1"

# ── 分位锚点（从批内分布计算后冻结，见 README §3.0）──────────
ANCHOR_LOW_PCT = 0.10   # 归一化下锚（低频端）
ANCHOR_CAP_PCT = 0.95   # 归一化上锚（重尾截断端）

# ── M1 活跃度：四维权重 + 分档（README §3.1）─────────────────
ACTIVITY_WEIGHTS = {
    "intensity": 0.30,          # 近期产出强度
    "recency": 0.35,            # 近端新鲜度
    "breadth": 0.20,            # 参与广度
    "social_influence": 0.15,   # 社交与影响
}
# 分档按批内分位（<p25 / p25-p50 / p50-p75 / p75-p90 / >p90）
ACTIVITY_BANDS = (
    (0.25, "dormant"),
    (0.50, "low"),
    (0.75, "mid"),
    (0.90, "high"),
    (1.00, "top"),
)

# 近端新鲜度半衰（天）：recency_days 每 +HALFLIFE 天，新鲜度减半
RECENCY_HALFLIFE_DAYS = 60.0

# ── M3 流失风险：五分量权重 + 停更档位（README §3.3）─────────
CHURN_WEIGHTS = {
    "staleness": 0.35,       # 停更天数
    "momentum": 0.25,        # 动量衰减（近 90 天在近 180 天中的占比反演）
    "sentiment": 0.15,       # 情绪/口碑
    "social_erosion": 0.10,  # 社交收缩（粗糙代理）
    "account_flag": 0.15,    # 账号标记（命中则覆盖置顶）
}
CHURN_STAGE_BOUNDS = (30, 60, 90)   # 停更天数梯度
CHURN_FLAG_OVERLAY = 95.0           # 账号标记命中时的风险下限
SENTIMENT_NEUTRAL = 0.5             # 情绪不可得时的中性值

# ── M2 兴趣迁移（README §3.2，S2 实现）────────────────────────
MIGRATION_WINDOW_DAYS = 180         # 近期窗
MIGRATION_MIN_EVENTS = 5            # 两窗各需的最小带 tag 事件数
MIGRATION_RETURN_GAP_DAYS = 90      # 回流判定：同一游戏中断 ≥ 该天数后近窗再现
MIGRATION_MAX_GAMES = 20            # 流动列表截断长度（确定性、避免超长）

# ── 归一化锚点涉及的原始指标 ─────────────────────────────────
ANCHOR_METRICS = (
    "act_30d", "act_90d", "act_180d", "content_per_year", "decay_ratio",
    "played_app_count", "distinct_tag_count", "following_app_count", "forum_count",
    "fans_count", "following_count", "voteup_received", "avg_ups_per_moment",
    "recency_days", "momentum_ratio",
)

# ── 情绪维度（与 L2 dim_neg_rate_* 对应）─────────────────────
SENTIMENT_DIMS = ("degree_of_freedom", "gameplay", "operation", "visual_music")

# ── 产出列顺序 ────────────────────────────────────────────────
ACTIVITY_COLUMNS = [
    "uid_hash", "l3_version",
    "activity_score", "activity_percentile", "activity_band",
    "activity_intensity", "activity_recency", "activity_breadth", "activity_social_influence",
    "activity_drivers",
    "recency_days", "act_30d", "act_90d", "decay_ratio",   # 原始引用（透明）
]

CHURN_COLUMNS = [
    "uid_hash", "l3_version",
    "churn_stage", "churn_risk_score", "churn_horizon_days",
    "churn_staleness", "churn_momentum", "churn_sentiment",
    "churn_social_erosion", "churn_account_flag",
    "churn_drivers",
    "recency_days", "decay_ratio", "momentum_ratio",        # 原始引用（透明）
]

MIGRATION_COLUMNS = [
    "uid_hash", "l3_version", "window_days",
    "genre_shift_score", "genre_entropy_delta", "genre_from", "genre_to",
    "entered_games", "dropped_games", "returned_games", "game_flow_net",
    "insufficient_data",
    "n_tag_events_early", "n_tag_events_recent",            # 两窗规模（透明）
    "n_games_early", "n_games_recent",
]


# ── 归一化工具 ────────────────────────────────────────────────

def _log1p(x: float) -> float:
    return math.log1p(max(float(x), 0.0))


def norm(x, lo: float, hi: float):
    """重尾压缩 + 锚点线性归一到 [0,1]；x 为 None 时返回 None。"""
    if x is None:
        return None
    a, b = _log1p(lo), _log1p(hi)
    if b <= a:
        return 0.0
    return min(max((_log1p(x) - a) / (b - a), 0.0), 1.0)


def norm_rev(x, lo: float, hi: float):
    """反向归一：越小越强。"""
    n = norm(x, lo, hi)
    return None if n is None else 1.0 - n


def recency_decay(days, halflife: float = RECENCY_HALFLIFE_DAYS) -> float:
    """近端新鲜度：half-life 指数衰减，越大越新。"""
    if days is None:
        return 0.0
    return 0.5 ** (max(float(days), 0.0) / halflife)


def avg(vals) -> float:
    """对非 None 取均值；全空返回 0.0（用于子分内部聚合）。"""
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else 0.0


def band_of(percentile: float) -> str:
    """活跃度分档：按批内分位。"""
    for edge, name in ACTIVITY_BANDS:
        if percentile < edge * 100:
            return name
    return ACTIVITY_BANDS[-1][1]


def top_drivers(subs: dict[str, float], k: int = 2) -> list[str]:
    """贡献最大的 k 个子分（按值降序，同值按名升序，确定性）。"""
    return [name for name, _ in sorted(subs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


# ── 共享运行时：加载 / 派生 / 锚点（供 model_*.py 复用）──────

def load_features(in_dir):
    """读 L2 特征表（features.csv），返回 DataFrame；缺失即报错。"""
    import pandas as pd  # 局部导入，保持本模块在无 IO 场景下可轻量引用

    path = in_dir / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺 L2 特征表：{path}（先跑 L2_features/extract_features.py）")
    return pd.read_csv(path)


def add_derived(df):
    """派生列：momentum_ratio = act_90d / max(act_180d, 1)（停更动量的原始比率）。"""
    df = df.copy()
    df["momentum_ratio"] = df["act_90d"] / df["act_180d"].clip(lower=1)
    return df


def compute_anchors(df) -> dict:
    """从批内分布计算冻结锚点（p10 / p95）；确定性、可写入 _manifest。"""
    anchors: dict[str, dict] = {}
    for m in ANCHOR_METRICS:
        if m not in df.columns:
            continue
        s = df[m].dropna()
        if s.empty:
            continue
        anchors[m] = {
            "p10": round(float(s.quantile(ANCHOR_LOW_PCT)), 4),
            "p95": round(float(s.quantile(ANCHOR_CAP_PCT)), 4),
        }
    return anchors


def anchor_of(anchors: dict, metric: str, default_lo: float = 0.0, default_hi: float = 1.0) -> tuple[float, float]:
    a = anchors.get(metric) or {}
    return a.get("p10", default_lo), a.get("p95", default_hi)


def input_fingerprint(paths) -> str:
    """对一组产出文件取 sha256（判「是否需重跑」）。"""
    import hashlib

    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            st = p.stat()
        except OSError:
            continue
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
    return h.hexdigest()[:16]
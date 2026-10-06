# -*- coding: utf-8 -*-
"""沉默预测 · 特征：多窗计数主干 + 180d 类型占比。

单一真源（改口径先升 SILENCE_VERSION，重跑双跑校验）：
    WINDOW_DAYS     计数窗（天）：(T−w, T] 内 exact 事件数 n{w}
    TRUNK_FEATURES  主干 = n7..n180 + gap_days（距最近一次 ≤T 公开事件的天数）
    TYPE_BUCKETS    类型桶（180d 占比）：review / post / wishlist / badge / follow_user
    FEATURE_SETS    {"trunk", "trunk_types"}——run_silence 只从这里取列，防泄漏

无泄漏硬规则：任一特征只用锚点 T 及之前的 exact 事件；未来事件不得影响既往行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import DAY, PanelConfig, group_exact_events

WINDOW_DAYS = (7, 14, 30, 60, 90, 180)
TRUNK_FEATURES = tuple(f"n{w}" for w in WINDOW_DAYS) + ("gap_days",)
TYPE_BUCKETS = ("review", "post", "wishlist", "badge", "follow_user")
TYPE_SHARE_WINDOW_DAYS = 180
TYPE_SHARE_FEATURES = tuple(f"t{TYPE_SHARE_WINDOW_DAYS}_{b}_share" for b in TYPE_BUCKETS)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "trunk": TRUNK_FEATURES,
    "trunk_types": TRUNK_FEATURES + TYPE_SHARE_FEATURES,
}
BASE_COLUMNS = ("uid_hash", "anchor_ts")
ALL_FEATURE_COLUMNS = (*TRUNK_FEATURES, *TYPE_SHARE_FEATURES)


def feature_columns(feature_set: str) -> tuple[str, ...]:
    """特征集名 → 列元组（唯一入口；未知名显式报错）。"""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"未知特征集：{feature_set!r}（支持 {tuple(FEATURE_SETS)}）")
    return FEATURE_SETS[feature_set]


def build_features(panel: pd.DataFrame, timeline: list[dict], cfg: PanelConfig) -> pd.DataFrame:
    """面板 → 特征帧（含全部特征列；调用方按 FEATURE_SETS 选列交给模型）。

    行为与面板行一一对应、顺序一致；列序 = BASE_COLUMNS + 主干 + 类型占比。
    """
    by_uid = group_exact_events(timeline)
    share_s = TYPE_SHARE_WINDOW_DAYS * DAY
    rows: list[dict] = []
    for r in panel.itertuples(index=False):
        events = by_uid.get(r.uid_hash, [])
        ts_arr = np.asarray([e["event_ts"] for e in events], dtype=np.int64)
        t = int(r.anchor_ts)
        feat: dict[str, float] = {}
        hi = int(np.searchsorted(ts_arr, t, "right"))
        for w in WINDOW_DAYS:
            lo = int(np.searchsorted(ts_arr, t - w * DAY, "right"))
            feat[f"n{w}"] = hi - lo
        last_pos = hi - 1  # 面板保证 (T−pre, T] 内有事件；防御性兜底见下
        feat["gap_days"] = (
            float((t - int(ts_arr[last_pos])) / DAY) if last_pos >= 0 else float(cfg.pre_window)
        )
        lo = int(np.searchsorted(ts_arr, t - share_s, "right"))
        seg = [events[i].get("event_type") for i in range(lo, hi)]
        total = len(seg)
        for bucket in TYPE_BUCKETS:
            feat[f"t{TYPE_SHARE_WINDOW_DAYS}_{bucket}_share"] = (
                sum(1 for etype in seg if etype == bucket) / total if total else 0.0
            )
        rows.append({"uid_hash": r.uid_hash, "anchor_ts": t, **feat})
    return pd.DataFrame(rows, columns=[*BASE_COLUMNS, *ALL_FEATURE_COLUMNS])
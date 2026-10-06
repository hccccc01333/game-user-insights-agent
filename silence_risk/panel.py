# -*- coding: utf-8 -*-
"""沉默预测 · 面板构造：(用户, 锚点 T) 样本 / 标签 / 截断保护。

口径（Future Public Inactivity Prediction）：
    锚点 T 从「首个 exact 事件 + min_history 天」起、按 step 天推进；
    样本条件： (T−pre_window, T] 内有 ≥1 个 exact 事件（对"近期活跃用户"提问）；
    标签 y = 1 当且仅当 (T, T+horizon] 内没有任何 exact 事件（公开沉默）；
    右截断保护：T + horizon ≤ 采集时刻 fetched_at（标签窗口必须完整落在观测期内）；
    左截断标记：T − pred_window 早于首个 exact 事件 → 长特征窗不完整（left_truncated）。

隐私口径：只用页面公开显示时间戳的事件（time_kind=exact）；产出仅 uid_hash。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TZ_CN = timezone(timedelta(hours=8))
DAY = 86400

PANEL_COLUMNS = ("uid_hash", "anchor_ts", "label", "first_ts", "fetched_at_ts", "left_truncated")


@dataclass(frozen=True)
class PanelConfig:
    """面板口径参数（全部为正整数天数；冻结后不可变，便于快照进 _manifest）。"""

    horizon: int = 30       # 标签窗口（天）：(T, T+h] 内无事件 → 沉默
    step: int = 30          # 锚点步长（天）
    pre_window: int = 30    # 预窗（天）：(T−pre, T] 需有活动
    pred_window: int = 180  # 最长特征窗（天），用于左截断标记
    min_history: int = 30   # 首个锚点距首个 exact 事件的最小历史（天）

    def __post_init__(self) -> None:
        for name in ("horizon", "step", "pre_window", "pred_window", "min_history"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"PanelConfig.{name} 需为正整数（当前 {v!r}）")

    def snapshot(self) -> dict:
        return {
            "horizon": self.horizon,
            "step": self.step,
            "pre_window": self.pre_window,
            "pred_window": self.pred_window,
            "min_history": self.min_history,
        }


def load_jsonl(path: Path) -> list[dict]:
    """读 JSONL（缺文件显式报错，避免静默空面板）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_fetched_at(value) -> int | None:
    """fetched_at（ISO8601，带 +08:00）→ epoch 秒；空值/非法返回 None。"""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_CN)
    return int(dt.timestamp())


def group_exact_events(records: list[dict]) -> dict[str, list[dict]]:
    """按 uid 归拢 exact 时间戳事件（time_kind=exact 且 event_ts 为正）；按 (ts, event_id) 排序。

    unknown 时间的事件不进任何时间序列（口径硬约束）。
    """
    by_uid: dict[str, list[dict]] = {}
    for e in records:
        ts = e.get("event_ts")
        if e.get("time_kind") != "exact" or not isinstance(ts, (int, float)) or ts <= 0:
            continue
        uid = e.get("uid_hash")
        if uid:
            by_uid.setdefault(uid, []).append(e)
    for events in by_uid.values():
        events.sort(key=lambda e: (e["event_ts"], str(e.get("event_id"))))
    return by_uid


def build_panel(
    timeline: list[dict], index: list[dict], cfg: PanelConfig
) -> tuple[pd.DataFrame, dict]:
    """构造面板（纯函数）：返回 (面板 DataFrame, stats)。逐值 deterministic。

    面板列见 PANEL_COLUMNS；排序 (anchor_ts, uid_hash) 固定。
    """
    by_uid = group_exact_events(timeline)
    fetched: dict[str, int] = {}
    for row in index:
        uid = row.get("uid_hash")
        ts = parse_fetched_at(row.get("fetched_at"))
        if uid and ts is not None:
            fetched[uid] = ts

    h_s, step_s = cfg.horizon * DAY, cfg.step * DAY
    pre_s, pred_s = cfg.pre_window * DAY, cfg.pred_window * DAY
    rows: list[dict] = []
    n_missing, n_skipped = 0, 0
    for uid in sorted(by_uid):
        events = by_uid[uid]
        ts_arr = np.asarray([e["event_ts"] for e in events], dtype=np.int64)
        first_ts = int(ts_arr[0])
        fetched_ts = fetched.get(uid)
        if fetched_ts is None:
            n_missing += 1
            continue
        limit = fetched_ts - h_s  # 右截断：T + h ≤ fetched_at
        t = first_ts + cfg.min_history * DAY
        while t <= limit:
            lo, hi = t - pre_s, t
            n_pre = int(np.searchsorted(ts_arr, hi, "right")) - int(
                np.searchsorted(ts_arr, lo, "right")
            )
            if n_pre == 0:
                n_skipped += 1
                t += step_s
                continue
            n_label = int(np.searchsorted(ts_arr, t + h_s, "right")) - int(
                np.searchsorted(ts_arr, t, "right")
            )
            rows.append(
                {
                    "uid_hash": uid,
                    "anchor_ts": int(t),
                    "label": int(n_label == 0),
                    "first_ts": first_ts,
                    "fetched_at_ts": int(fetched_ts),
                    "left_truncated": bool(t - pred_s < first_ts),
                }
            )
            t += step_s

    panel = pd.DataFrame(rows, columns=list(PANEL_COLUMNS))
    if not panel.empty:
        panel = panel.sort_values(["anchor_ts", "uid_hash"], kind="stable").reset_index(drop=True)

    stats: dict = {
        "n_users_with_exact_events": len(by_uid),
        "n_users_used": int(panel["uid_hash"].nunique()) if not panel.empty else 0,
        "n_users_missing_index": n_missing,
        "n_samples": int(len(panel)),
        "silent_rate": float(panel["label"].mean()) if not panel.empty else None,
        "left_truncated_share": float(panel["left_truncated"].mean()) if not panel.empty else None,
        "anchors_skipped_no_pre_activity": n_skipped,
    }
    if not panel.empty:
        per_user = panel.groupby("uid_hash").size()
        stats["samples_per_user"] = {"median": float(per_user.median()), "max": int(per_user.max())}
        years = [datetime.fromtimestamp(int(ts), TZ_CN).year for ts in panel["anchor_ts"]]
        stats["anchors_by_year"] = {str(y): int(c) for y, c in sorted(pd.Series(years).value_counts().items())}
    return panel, stats
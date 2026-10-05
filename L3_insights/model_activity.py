#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L3 · M1 活跃度评分（activity）。

读 L2 特征表，按 l3_schema 的四维权重与锚点，给每用户一个 0-100 的当前活跃度：
    子分：intensity（近期产出强度）/ recency（近端新鲜度）
          breadth（参与广度）/ social_influence（社交与影响）
    合成：activity_score = 100 × Σ(权重 × 子分)
    band：按批内分位切五档（dormant/low/mid/high/top）

产出：
    data/processed/user_insights/activity.csv
    data/processed/user_insights/_manifest.json （补 anchors / coverage 段）

口径见 L3_insights/README.md §3.0 / §3.1。

运行（在仓库根目录）：
    python L3_insights/model_activity.py
    python L3_insights/model_activity.py --limit 50         # 小样试跑
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import l3_schema as s

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "data" / "processed" / "user_features"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "user_insights"
TZ_CN = timezone(timedelta(hours=8))


def _num(row, key):
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def activity_subscores(row, anchors: dict) -> dict:
    lo, hi = s.anchor_of(anchors, "act_30d");        n30 = s.norm(_num(row, "act_30d"), lo, hi)
    lo, hi = s.anchor_of(anchors, "act_90d");        n90 = s.norm(_num(row, "act_90d"), lo, hi)
    lo, hi = s.anchor_of(anchors, "content_per_year")
    ncpy = s.norm(_num(row, "content_per_year"), lo, hi)
    intensity = s.avg([n30, n30, n90, n90, n90, ncpy])  # 30d 权重 2/6，90d 3/6，cpy 1/6

    lo, hi = s.anchor_of(anchors, "decay_ratio")
    recency = 0.65 * s.recency_decay(_num(row, "recency_days")) + 0.35 * s.avg([s.norm(_num(row, "decay_ratio"), lo, hi)])

    b_played = s.norm(_num(row, "played_app_count"), *s.anchor_of(anchors, "played_app_count"))
    b_tags = s.norm(_num(row, "distinct_tag_count"), *s.anchor_of(anchors, "distinct_tag_count"))
    b_follow = s.norm(_num(row, "following_app_count"), *s.anchor_of(anchors, "following_app_count"))
    b_forum = s.norm(_num(row, "forum_count"), *s.anchor_of(anchors, "forum_count"))
    breadth = s.avg([b_played, b_tags, b_follow, b_forum])

    s_fans = s.norm(_num(row, "fans_count"), *s.anchor_of(anchors, "fans_count"))
    s_following = s.norm(_num(row, "following_count"), *s.anchor_of(anchors, "following_count"))
    s_voteup = s.norm(_num(row, "voteup_received"), *s.anchor_of(anchors, "voteup_received"))
    s_ups = s.norm(_num(row, "avg_ups_per_moment"), *s.anchor_of(anchors, "avg_ups_per_moment"))
    social_influence = s.avg([s_fans, s_following, s_voteup, s_ups])

    return {
        "intensity": round(intensity, 4),
        "recency": round(recency, 4),
        "breadth": round(breadth, 4),
        "social_influence": round(social_influence, 4),
    }


def compute(df: pd.DataFrame, anchors: dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        subs = activity_subscores(row, anchors)
        score = round(100.0 * sum(s.ACTIVITY_WEIGHTS[k] * subs[k] for k in s.ACTIVITY_WEIGHTS), 2)
        rows.append({
            "uid_hash": row["uid_hash"],
            "l3_version": s.L3_VERSION,
            "activity_score": score,
            "activity_intensity": subs["intensity"],
            "activity_recency": subs["recency"],
            "activity_breadth": subs["breadth"],
            "activity_social_influence": subs["social_influence"],
            "activity_drivers": ";".join(s.top_drivers(subs)),
            "recency_days": _num(row, "recency_days"),
            "act_30d": _num(row, "act_30d"),
            "act_90d": _num(row, "act_90d"),
            "decay_ratio": _num(row, "decay_ratio"),
        })
    out = pd.DataFrame(rows)
    out["activity_percentile"] = (out["activity_score"].rank(pct=True) * 100).round(1)
    out["activity_band"] = out["activity_percentile"].map(s.band_of)
    return out[s.ACTIVITY_COLUMNS]


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 M1：活跃度评分")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN), help="L2 产出目录")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="L3 产出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个用户（小样试跑）")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    try:
        df = s.load_features(in_dir)
    except FileNotFoundError as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 2
    if args.limit:
        df = df.head(args.limit)
    df = s.add_derived(df)
    anchors = s.compute_anchors(df)

    out = compute(df, anchors)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "activity.csv", index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["l3_version"] = s.L3_VERSION
    manifest["anchors"] = anchors
    manifest["activity"] = {
        "n_users": int(len(out)),
        "bands": out["activity_band"].value_counts().to_dict(),
        "mean_score": round(float(out["activity_score"].mean()), 2),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    dist = out["activity_band"].value_counts().reindex(["dormant", "low", "mid", "high", "top"]).fillna(0).astype(int)
    print(f"[done] 活跃度 {len(out)} 行 → {out_dir / 'activity.csv'}")
    print("       分档：" + " ".join(f"{k}={v}" for k, v in dist.items()) +
          f"｜均分 {manifest['activity']['mean_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
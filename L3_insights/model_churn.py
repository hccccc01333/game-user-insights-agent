#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L3 · M3 流失风险分层（churn）。

无纵向标签 → 不做监督训练，用「信号合成」给出可排序的风险：
    停更档位 churn_stage  ∈ churned/silent/dormant_90/dormant_60/dormant_30/active
    综合评分 churn_risk_score 0-100（staleness/momentum/sentiment/social_erosion/account_flag 五分量）
    预警视界 churn_horizon_days ∈ {90,60,30,none}（距 90 天沉默线的剩余跑道）

产出：
    data/processed/user_insights/churn.csv
    data/processed/user_insights/_manifest.json （补 churn 段）

⚠️ 阈值未经运营反馈校准（calibration_status=uncalibrated）。口径见 L3_insights/README.md §3.3。

运行（在仓库根目录）：
    python L3_insights/model_churn.py
    python L3_insights/model_churn.py --limit 50
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


def _flag(row) -> bool:
    return bool(row.get("is_silent") or row.get("is_deactivated") or row.get("is_deleted"))


def churn_stage(row) -> str:
    if bool(row.get("is_deleted")) or bool(row.get("is_deactivated")):
        return "churned"
    if bool(row.get("is_silent")):
        return "silent"
    r = _num(row, "recency_days")
    if r is None:
        return "unknown"
    b30, b60, b90 = s.CHURN_STAGE_BOUNDS
    if r > b90:
        return "dormant_90"
    if r > b60:
        return "dormant_60"
    if r > b30:
        return "dormant_30"
    return "active"


def churn_horizon(stage: str, row) -> object:
    if stage in ("churned", "silent"):
        return "none"
    r = _num(row, "recency_days")
    if r is None:
        return "none"
    b30, b60, b90 = s.CHURN_STAGE_BOUNDS
    if r <= b30:
        return b90      # 健康跑道
    if r <= b60:
        return b60
    if r <= b90:
        return b30
    return "none"


def _sentiment_risk(row) -> float:
    parts: list[tuple[float, float]] = []
    negs = [_num(row, f"dim_neg_rate_{d}") for d in s.SENTIMENT_DIMS]
    negs = [v for v in negs if v is not None]
    if negs:
        parts.append((0.6, sum(negs) / len(negs)))
    score = _num(row, "review_score_mean")
    if score is not None:
        parts.append((0.4, 1.0 - min(max((score - 1.0) / 4.0, 0.0), 1.0)))
    if not parts:
        return s.SENTIMENT_NEUTRAL
    w = sum(x[0] for x in parts)
    return sum(x[0] * x[1] for x in parts) / w


def churn_subscores(row, anchors: dict, flag: bool) -> dict:
    staleness = s.avg([s.norm(_num(row, "recency_days"), *s.anchor_of(anchors, "recency_days"))])
    momentum = 1.0 - s.avg([s.norm(_num(row, "momentum_ratio"), *s.anchor_of(anchors, "momentum_ratio"))])
    sentiment = _sentiment_risk(row)
    social = 1.0 - s.avg([
        s.norm(_num(row, "fans_count"), *s.anchor_of(anchors, "fans_count")),
        s.norm(_num(row, "mutual_count"), *s.anchor_of(anchors, "mutual_count")),
    ])
    return {
        "staleness": round(staleness, 4),
        "momentum": round(momentum, 4),
        "sentiment": round(sentiment, 4),
        "social_erosion": round(social, 4),
        "account_flag": 1.0 if flag else 0.0,
    }


def compute(df: pd.DataFrame, anchors: dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        flag = _flag(row)
        subs = churn_subscores(row, anchors, flag)
        score = 100.0 * sum(s.CHURN_WEIGHTS[k] * subs[k] for k in s.CHURN_WEIGHTS)
        if flag:
            score = max(score, s.CHURN_FLAG_OVERLAY)
        stage = churn_stage(row)
        rows.append({
            "uid_hash": row["uid_hash"],
            "l3_version": s.L3_VERSION,
            "churn_stage": stage,
            "churn_risk_score": round(score, 2),
            "churn_horizon_days": churn_horizon(stage, row),
            "churn_staleness": subs["staleness"],
            "churn_momentum": subs["momentum"],
            "churn_sentiment": subs["sentiment"],
            "churn_social_erosion": subs["social_erosion"],
            "churn_account_flag": subs["account_flag"],
            "churn_drivers": ";".join(s.top_drivers({
                k: subs[k] * s.CHURN_WEIGHTS[k] for k in s.CHURN_WEIGHTS
            })),
            "recency_days": _num(row, "recency_days"),
            "decay_ratio": _num(row, "decay_ratio"),
            "momentum_ratio": round(_num(row, "momentum_ratio") or 0.0, 4),
        })
    return pd.DataFrame(rows)[s.CHURN_COLUMNS]


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 M3：流失风险分层")
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
    out.to_csv(out_dir / "churn.csv", index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["l3_version"] = s.L3_VERSION
    manifest["anchors"] = anchors
    order = ["active", "dormant_30", "dormant_60", "dormant_90", "silent", "churned", "unknown"]
    dist = out["churn_stage"].value_counts().reindex(order).fillna(0).astype(int)
    manifest["churn"] = {
        "n_users": int(len(out)),
        "stages": {k: int(v) for k, v in dist.items()},
        "mean_risk": round(float(out["churn_risk_score"].mean()), 2),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[done] 流失风险 {len(out)} 行 → {out_dir / 'churn.csv'}")
    print("       档位：" + " ".join(f"{k}={int(v)}" for k, v in dist.items()) +
          f"｜均分 {manifest['churn']['mean_risk']}")
    print("       ⚠️ 阈值未经运营反馈校准（uncalibrated）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
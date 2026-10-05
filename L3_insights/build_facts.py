#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L3 · M4 汇总：把 M1/M2/M3 合成 facts.jsonl —— Agent 的唯一输入契约。

一用户一行，只读、不含 L1 原文、不含策略结论（策略由 Agent 生成）。
契约见 L3_insights/README.md §4。

输入（均只读）：
    data/processed/user_features/features.csv      L2 特征（quality / context 素材）
    data/processed/user_features/timeline.jsonl    L2 时间线（仅取 recent_app_ids / reference_events）
    data/processed/user_features/_manifest.json    as_of / tag_vocab
    data/processed/user_insights/activity.csv      M1（缺失则显式报错）
    data/processed/user_insights/churn.csv         M3（缺失则显式报错）
    data/processed/user_insights/migration.jsonl   M2（可选；缺则 migration=null）

产出：
    data/processed/user_insights/facts.jsonl       ★ Agent 唯一入口
    data/processed/user_insights/_manifest.json    （补 coverage_summary / input_fingerprint）

运行（在仓库根目录）：
    python L3_insights/build_facts.py
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
DEFAULT_L2 = PROJECT_ROOT / "data" / "processed" / "user_features"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "user_insights"
TZ_CN = timezone(timedelta(hours=8))

RECENT_APP_WINDOW_DAYS = 180
MAX_RECENT_APPS = 5
MAX_REFERENCE_EVENTS = 10
CONTENT_EVENTS = ("review", "post", "wishlist")


def _clean(v):
    """把 pandas NaN 归一为 None，便于 JSON 序列化（绝不静默填 0）。"""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _num(row, key):
    v = _clean(row.get(key))
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _index_by_uid(rows: list[dict]) -> dict[str, dict]:
    return {r["uid_hash"]: r for r in rows if r.get("uid_hash")}


def _timeline_by_uid(events: list[dict], as_of: int):
    """按 uid 归拢：近期内容事件（取 app_id）与最近事件 id。"""
    by_uid: dict[str, dict] = {}
    for e in events:
        uid = e.get("uid_hash")
        if not uid:
            continue
        slot = by_uid.setdefault(uid, {"apps": [], "events": []})
        slot["events"].append(e)
        ts = e.get("event_ts")
        if (e.get("event_type") in CONTENT_EVENTS and e.get("app_id")
                and ts and e.get("time_kind") == "exact"
                and ts >= as_of - RECENT_APP_WINDOW_DAYS * 86400):
            slot["apps"].append((ts, e["app_id"]))
    for slot in by_uid.values():
        slot["apps"] = [a for _, a in sorted(slot["apps"], key=lambda x: -x[0])]
        slot["events"] = sorted(
            slot["events"], key=lambda e: (e.get("event_ts") is None, -(e.get("event_ts") or 0))
        )
    return by_uid


def _top_genres(row, vocab: list[str]) -> list:
    pairs = []
    for i, genre in enumerate(vocab, 1):
        v = _num(row, f"tag_top{i:02d}")
        if v and v > 0:
            pairs.append((genre, round(v, 4)))
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return [[g, r] for g, r in pairs[:2]]


def _distinct(seq: list) -> list:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_fact(row, act, churn, tl, vocab: list[str], as_of: int) -> dict:
    apps = _distinct((tl or {}).get("apps") or [])[:MAX_RECENT_APPS]
    events = _distinct([e.get("event_id") for e in ((tl or {}).get("events") or [])])[:MAX_REFERENCE_EVENTS]
    negs = [_num(row, f"dim_neg_rate_{d}") for d in s.SENTIMENT_DIMS]
    negs = [v for v in negs if v is not None]

    subs_a = {k: _num(act, f"activity_{k}") for k in s.ACTIVITY_WEIGHTS}
    subs_c = {k: _num(churn, f"churn_{k}") for k in s.CHURN_WEIGHTS}
    usable = act is not None and churn is not None

    return {
        "uid_hash": row["uid_hash"],
        "facts_version": s.L3_VERSION,
        "as_of": as_of,
        "calibration_status": "uncalibrated",
        "quality": {
            "usable": bool(usable),
            "truncated_surfaces": _clean(row.get("truncated_surfaces")) or "",
            "account_status": _clean(row.get("account_status")) or "unknown",
            "wishlist_locked": bool(row.get("wishlist_locked")) if _clean(row.get("wishlist_locked")) is not None else None,
        },
        "activity": None if act is None else {
            "score": _num(act, "activity_score"),
            "percentile": _num(act, "activity_percentile"),
            "band": act.get("activity_band"),
            "sub": subs_a,
            "drivers": (act.get("activity_drivers") or "").split(";") if act.get("activity_drivers") else [],
        },
        "churn": None if churn is None else {
            "stage": churn.get("churn_stage"),
            "risk_score": _num(churn, "churn_risk_score"),
            "horizon_days": churn.get("churn_horizon_days"),
            "sub": subs_c,
            "drivers": (churn.get("churn_drivers") or "").split(";") if churn.get("churn_drivers") else [],
        },
        "migration": None,   # S2 实现后回填（见 README §3.2）
        "context": {
            "top_genres": _top_genres(row, vocab),
            "recent_app_ids": apps,
            "sentiment_neg_rate": round(sum(negs) / len(negs), 4) if negs else None,
            "tenure_bucket": _clean(row.get("tenure_bucket")) or "unknown",
            "reference_events": events,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 M4：合成 facts.jsonl（Agent 输入契约）")
    ap.add_argument("--l2-dir", default=str(DEFAULT_L2), help="L2 产出目录")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="L3 产出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个用户（小样试跑）")
    args = ap.parse_args()

    l2_dir, out_dir = Path(args.l2_dir), Path(args.out_dir)
    for name in ("activity.csv", "churn.csv"):
        if not (out_dir / name).exists():
            print(f"[stop] 缺 {name}（先跑 model_activity.py / model_churn.py）", file=sys.stderr)
            return 2

    try:
        features = s.load_features(l2_dir)
    except FileNotFoundError as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 2
    if args.limit:
        features = features.head(args.limit)
    features = features.set_index("uid_hash", drop=False)

    activity = _index_by_uid(pd.read_csv(out_dir / "activity.csv").to_dict("records"))
    churn = _index_by_uid(pd.read_csv(out_dir / "churn.csv").to_dict("records"))
    migration = _index_by_uid(_load_jsonl(out_dir / "migration.jsonl"))

    l2_manifest_path = l2_dir / "_manifest.json"
    l2_manifest = json.loads(l2_manifest_path.read_text(encoding="utf-8")) if l2_manifest_path.exists() else {}
    as_of = int(l2_manifest.get("as_of") or datetime.now(TZ_CN).timestamp())
    vocab = l2_manifest.get("tag_vocab") or []

    timeline = _timeline_by_uid(_load_jsonl(l2_dir / "timeline.jsonl"), as_of)

    facts = []
    for _, row in features.iterrows():
        uid = row["uid_hash"]
        fact = build_fact(row, activity.get(uid), churn.get(uid), timeline.get(uid), vocab, as_of)
        if uid in migration:
            fact["migration"] = {
                k: v for k, v in migration[uid].items() if k not in ("uid_hash", "l3_version")
            }
        facts.append(fact)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "facts.jsonl").open("w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    coverage = {
        "facts_rows": len(facts),
        "activity_available": sum(1 for f in facts if f["activity"]),
        "churn_available": sum(1 for f in facts if f["churn"]),
        "migration_available": sum(1 for f in facts if f["migration"]),
        "usable_false": sum(1 for f in facts if not f["quality"]["usable"]),
        "migration_status": "run" if migration else "not_run",
    }
    manifest_path = out_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["l3_version"] = s.L3_VERSION
    manifest["generated_at"] = datetime.now(TZ_CN).isoformat(timespec="seconds")
    manifest["n_users"] = len(facts)
    manifest["calibration_status"] = "uncalibrated"
    manifest["input_fingerprint"] = s.input_fingerprint([l2_dir / "features.csv", l2_dir / "timeline.jsonl"])
    manifest["coverage_summary"] = coverage
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[done] facts {len(facts)} 行 → {out_dir / 'facts.jsonl'}")
    print(f"       覆盖：activity={coverage['activity_available']} churn={coverage['churn_available']} "
          f"migration={coverage['migration_available']} usable=false={coverage['usable_false']}")
    print("       ⚠️ calibration_status=uncalibrated（未经运营反馈校准）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
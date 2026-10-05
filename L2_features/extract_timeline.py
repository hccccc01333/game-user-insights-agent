#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L2 特征抽取②：行为时间线 + 收藏快照表。

读 L1 原始 JSON，产出：

    data/processed/user_features/
    ├── timeline.jsonl          行为时间线（一事件一行，含 time_kind 三态标记）
    └── favorite_snapshot.jsonl 收藏快照表（无动作时间，不进时间线，见 README §10 C）

口径：
    · 时间可得性：feed/wishlist/badge 为 exact；following_user 实测=关注时间（exact）；
      favorite/fans/following_app 无时间（unknown，不参与序列分析）。
    · 隐私：follow/fan 事件 target_id 一律留空，不落任何对方标识（含哈希）。
    · event_id = sha1(uid_hash + event_type + source_id) 前 16 位，幂等键。

运行（在仓库根目录）：
    python L2_features/extract_timeline.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feature_dict as fd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "data" / "raw" / "user_profile"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "user_features"
TZ_CN = timezone(timedelta(hours=8))

FAVORITE_SURFACES = (
    "favorite_app",
    "favorite_moment",
    "favorite_collection",
    "favorite_hashtag",
    "favorite_event",
)


def _event_id(uid_hash: str, event_type: str, source_id) -> str:
    raw = f"{uid_hash}:{event_type}:{source_id}"
    return "ev_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _ts(v):
    try:
        t = int(v)
        return t if t > 0 else None
    except (TypeError, ValueError):
        return None


def _app_fields(app: dict | None) -> tuple:
    if not isinstance(app, dict):
        return None, None
    return app.get("id"), app.get("title")


def _event(
    uid_hash: str, event_type: str, source_id, ts, time_kind: str,
    app_id=None, app_name=None, target_id=None, subtype=None,
    metrics: dict | None = None, source_surface: str = "",
) -> dict:
    return {
        "event_id": _event_id(uid_hash, event_type, source_id),
        "uid_hash": uid_hash,
        "feature_version": fd.FEATURE_VERSION,
        "event_type": event_type,
        "event_ts": ts,
        "time_kind": time_kind,
        "app_id": app_id,
        "app_name": app_name,
        "target_id": target_id,
        "subtype": subtype,
        "metrics": metrics or {},
        "source_surface": source_surface,
    }


def build_timeline(rec: dict) -> list[dict]:
    uid = rec.get("uid_hash", "")
    surfaces = rec["surfaces"]
    events: list[dict] = []

    for m in (item.get("moment") or {} for item in surfaces.get("feed_review") or []):
        if not isinstance(m, dict):
            continue
        review = m.get("review") or {}
        app_id, app_name = _app_fields(m.get("app"))
        stat = m.get("stat") or {}
        events.append(_event(
            uid, "review", m.get("id_str"), _ts(m.get("publish_time")), "exact",
            app_id, app_name, m.get("id_str"),
            subtype=f"score={review.get('score')}" if review.get("score") else None,
            metrics={"score": review.get("score"), "ups": stat.get("ups"),
                     "supports": stat.get("supports"), "played_spent": review.get("played_spent")},
            source_surface="feed_review",
        ))

    for m in (item.get("moment") or {} for item in surfaces.get("feed_moment") or []):
        if not isinstance(m, dict):
            continue
        app_id, app_name = _app_fields(m.get("app"))
        stat = m.get("stat") or {}
        events.append(_event(
            uid, "post", m.get("id_str"), _ts(m.get("publish_time")), "exact",
            app_id, app_name, m.get("id_str"),
            metrics={"ups": stat.get("ups"), "supports": stat.get("supports")},
            source_surface="feed_moment",
        ))

    for w in surfaces.get("wishlist") or []:
        app_id, app_name = _app_fields(w.get("app"))
        events.append(_event(
            uid, "wishlist", w.get("app_id"), _ts(w.get("created_time")), "exact",
            app_id or w.get("app_id"), app_name,
            source_surface="wishlist",
        ))

    for b in surfaces.get("badge") or []:
        events.append(_event(
            uid, "badge", b.get("id"), _ts(b.get("time")), "exact",
            target_id=str(b.get("id")), subtype=b.get("title"),
            metrics={"level": b.get("level")},
            source_surface="badge",
        ))

    for u in surfaces.get("following_user") or []:
        # 隐私：target_id 留空，不落对方标识
        events.append(_event(
            uid, "follow_user", u.get("id"), _ts(u.get("created_time")), "exact",
            subtype=u.get("follow_source") or None,
            metrics={"verified": bool(u.get("verified"))},
            source_surface="following_user",
        ))

    for u in surfaces.get("fans") or []:
        events.append(_event(
            uid, "fan", u.get("id"), None, "unknown",
            metrics={"deactivated": bool(u.get("is_deactivated") or u.get("is_deleted"))},
            source_surface="fans",
        ))

    events.sort(key=lambda e: (e["event_ts"] is None, e["event_ts"] or 0, e["event_id"]))
    return events


def build_favorite_snapshot(rec: dict) -> list[dict]:
    uid = rec.get("uid_hash", "")
    rows = []
    for surface in FAVORITE_SURFACES:
        for item in rec["surfaces"].get(surface) or []:
            app_id = app_name = target_id = title = None
            if isinstance(item.get("app"), dict):
                app = item["app"]
                app_id, app_name, title = app.get("id"), app.get("title"), app.get("title")
                target_id = app_id
            elif isinstance(item.get("moment"), dict):
                mom = item["moment"]
                target_id = mom.get("id_str")
                app_id, app_name = _app_fields(mom.get("app"))
            else:
                for key in ("hashtag", "collection", "event"):
                    if isinstance(item.get(key), dict):
                        obj = item[key]
                        target_id = obj.get("id") or obj.get("id_str")
                        title = obj.get("title") or obj.get("name")
                        break
            rows.append({
                "uid_hash": uid,
                "feature_version": fd.FEATURE_VERSION,
                "surface": surface,
                "app_id": app_id,
                "app_name": app_name,
                "target_id": target_id,
                "title": title,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="L2 特征抽取②：行为时间线 + 收藏快照表")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN), help="L1 原始 JSON 目录")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="产出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个用户（小样试跑）")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.in_dir).glob("*.json") if not p.name.startswith("_"))
    files = files[: args.limit] if args.limit else files
    if not files:
        print("[stop] 没有可处理的输入", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_events = n_snapshot = n_users = 0
    with (out_dir / "timeline.jsonl").open("w", encoding="utf-8") as f_tl, \
            (out_dir / "favorite_snapshot.jsonl").open("w", encoding="utf-8") as f_fs:
        for p in files:
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[skip] {p.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            rec.setdefault("uid_hash", p.stem)
            n_users += 1
            for e in build_timeline(rec):
                f_tl.write(json.dumps(e, ensure_ascii=False) + "\n")
                n_events += 1
            for r in build_favorite_snapshot(rec):
                f_fs.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_snapshot += 1

    manifest_path = out_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["timeline"] = {"events": n_events, "favorite_snapshot_rows": n_snapshot, "n_users": n_users}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[done] 时间线 {n_events} 事件｜收藏快照 {n_snapshot} 行｜用户 {n_users} → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
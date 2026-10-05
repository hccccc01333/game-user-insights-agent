#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L3 · M2 兴趣迁移（migration）。

两个子模型（口径见 L3_insights/README.md §3.2）：

（a）品类迁移 genre migration
    事件 → app_id → tags（借 `app_tag_lookup.json` 补品类）；
    按 `event_ts` 切「早期窗 vs 近 180 天窗」，各聚合 tags 分布 →
        genre_shift_score   两窗分布的 JS 散度（base-2，∈[0,1]）
        genre_entropy_delta 香农熵（bits）之差：>0 变散 / <0 变聚焦
        genre_from / genre_to 份额降/升最大的品类（迁移方向）
    门槛：两窗各需 ≥ MIGRATION_MIN_EVENTS 个带 tag 事件，否则置 null + insufficient_data。

（b）游戏流动 game flow
    基于带 app_id 的 review/post/wishlist（exact）内容事件（近窗默认 180 天）：
        entered_games 近窗首现、早期未见
        dropped_games 早期活跃、近窗消失
        returned_games 两窗均有且中断 ≥ MIGRATION_RETURN_GAP_DAYS 再现
        game_flow_net   entered − dropped

产出：
    data/processed/user_insights/migration.jsonl   （一用户一行）
    data/processed/user_insights/_manifest.json    （补 migration 段）

运行（在仓库根目录；前置：先跑 app_tag_lookup.py）：
    python L3_insights/model_migration.py
    python L3_insights/model_migration.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import l3_schema as s

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_L2 = PROJECT_ROOT / "data" / "processed" / "user_features"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "user_insights"
TZ_CN = timezone(timedelta(hours=8))

CONTENT_EVENTS = ("review", "post", "wishlist")
EPS = 1e-12


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _content_events_by_uid(timeline: Path) -> dict[str, list[dict]]:
    """按 uid 归拢带 app_id 的内容事件（review/post/wishlist，且 exact 有时间）。"""
    by_uid: dict[str, list[dict]] = {}
    for e in _read_jsonl(timeline):
        ts = e.get("event_ts")
        if (e.get("event_type") in CONTENT_EVENTS and e.get("app_id") is not None
                and e.get("time_kind") == "exact" and ts):
            by_uid.setdefault(e["uid_hash"], []).append(e)
    for events in by_uid.values():
        events.sort(key=lambda e: e["event_ts"])
    return by_uid


def _feature_uids(l2_dir: Path) -> list[str] | None:
    """L2 特征表的 uid 全集（与 facts 行域对齐；缺则返回 None）。"""
    import csv

    path = l2_dir / "features.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [row["uid_hash"] for row in csv.DictReader(f) if row.get("uid_hash")]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _tag_dist(events: list[dict], lookup: dict) -> tuple[dict, int]:
    """事件 → 品类分布（每事件对其 app 的每个 tag 计 1）；返回 (dist, 带 tag 事件数)。"""
    c: Counter = Counter()
    n_tag = 0
    for e in events:
        tags = lookup.get(str(e["app_id"]))
        if not tags:
            continue
        n_tag += 1
        c.update(tags)
    total = sum(c.values())
    dist = {t: c[t] / total for t in c} if total else {}
    return dist, n_tag


def _js_divergence(p: dict, q: dict) -> float:
    """JS 散度（base-2，∈[0,1]）：0 = 分布一致。"""
    vocab = set(p) | set(q)
    m = {t: 0.5 * (p.get(t, 0.0) + q.get(t, 0.0)) for t in vocab}

    def kl(a: dict, b: dict) -> float:
        acc = 0.0
        for t in vocab:
            av = a.get(t, 0.0)
            if av > 0.0:
                bv = b.get(t, 0.0)
                if bv > 0.0:
                    acc += av * math.log2(av / bv)
        return acc

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _entropy(dist: dict) -> float:
    return -sum(v * math.log2(v) for v in dist.values() if v > EPS)


def _shift_direction(p: dict, q: dict) -> tuple:
    """份额变化最大的品类：from = 降幅最大，to = 升幅最大（无变化侧为 None）。"""
    vocab = set(p) | set(q)
    diffs = sorted(((q.get(t, 0.0) - p.get(t, 0.0), t) for t in vocab), key=lambda kv: (kv[0], kv[1]))
    from_tag = diffs[0][1] if diffs and diffs[0][0] < 0 else None
    to_tag = diffs[-1][1] if diffs and diffs[-1][0] > 0 else None
    return from_tag, to_tag


def _game_flow(events: list[dict], recent_cut: int) -> dict:
    """游戏流动（entered/dropped/returned/net）。"""
    early = [e for e in events if e["event_ts"] < recent_cut]
    recent = [e for e in events if e["event_ts"] >= recent_cut]
    games_early = {e["app_id"] for e in early}
    games_recent = {e["app_id"] for e in recent}

    entered = sorted(games_recent - games_early)
    dropped = sorted(games_early - games_recent)

    by_app: dict[int, list[int]] = {}
    for e in events:
        by_app.setdefault(e["app_id"], []).append(e["event_ts"])
    gap_s = s.MIGRATION_RETURN_GAP_DAYS * 86400
    returned = []
    for aid in sorted(games_early & games_recent):
        ts = by_app[aid]
        if any(ts[i + 1] - ts[i] >= gap_s and ts[i] < recent_cut <= ts[i + 1]
               for i in range(len(ts) - 1)):
            returned.append(aid)

    return {
        "entered": entered[: s.MIGRATION_MAX_GAMES],
        "dropped": dropped[: s.MIGRATION_MAX_GAMES],
        "returned": returned[: s.MIGRATION_MAX_GAMES],
        "net": len(entered) - len(dropped),
        "n_games_early": len(games_early),
        "n_games_recent": len(games_recent),
        "has_content": bool(events),
    }


def compute_user(uid: str, events: list[dict], lookup: dict, as_of: int) -> dict:
    recent_cut = as_of - s.MIGRATION_WINDOW_DAYS * 86400
    early = [e for e in events if e["event_ts"] < recent_cut]
    recent = [e for e in events if e["event_ts"] >= recent_cut]

    p, n_tag_early = _tag_dist(early, lookup)
    q, n_tag_recent = _tag_dist(recent, lookup)
    genre_ok = n_tag_early >= s.MIGRATION_MIN_EVENTS and n_tag_recent >= s.MIGRATION_MIN_EVENTS

    if genre_ok:
        shift = round(_js_divergence(p, q), 4)
        entropy_delta = round(_entropy(q) - _entropy(p), 4)
        from_tag, to_tag = _shift_direction(p, q)
    else:
        shift = entropy_delta = from_tag = to_tag = None

    flow = _game_flow(events, recent_cut)

    return {
        "uid_hash": uid,
        "l3_version": s.L3_VERSION,
        "window_days": s.MIGRATION_WINDOW_DAYS,
        "genre_shift_score": shift,
        "genre_entropy_delta": entropy_delta,
        "genre_from": from_tag,
        "genre_to": to_tag,
        "entered_games": flow["entered"],
        "dropped_games": flow["dropped"],
        "returned_games": flow["returned"],
        "game_flow_net": flow["net"] if flow["has_content"] else None,
        "insufficient_data": not genre_ok,
        "n_tag_events_early": n_tag_early,
        "n_tag_events_recent": n_tag_recent,
        "n_games_early": flow["n_games_early"],
        "n_games_recent": flow["n_games_recent"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 M2：兴趣迁移（品类迁移 + 游戏流动）")
    ap.add_argument("--l2-dir", default=str(DEFAULT_L2), help="L2 产出目录")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="L3 产出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个用户（小样试跑）")
    args = ap.parse_args()

    l2_dir, out_dir = Path(args.l2_dir), Path(args.out_dir)
    lookup_path = out_dir / "app_tag_lookup.json"
    if not lookup_path.exists():
        print(f"[stop] 缺 {lookup_path.name}（先跑 L3_insights/app_tag_lookup.py）", file=sys.stderr)
        return 2
    timeline_path = l2_dir / "timeline.jsonl"
    if not timeline_path.exists():
        print(f"[stop] 缺 L2 时间线：{timeline_path}", file=sys.stderr)
        return 2

    lookup = _load_json(lookup_path).get("apps") or {}
    as_of = int(_load_json(l2_dir / "_manifest.json").get("as_of") or datetime.now(TZ_CN).timestamp())

    by_uid = _content_events_by_uid(timeline_path)
    uids = _feature_uids(l2_dir) or sorted(by_uid.keys())   # 与 facts 行域对齐
    uids = sorted(uids)
    if args.limit:
        uids = uids[: args.limit]

    rows = [compute_user(uid, by_uid.get(uid, []), lookup, as_of) for uid in uids]
    rows.sort(key=lambda r: r["uid_hash"])

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "migration.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    insuf = sum(1 for r in rows if r["insufficient_data"])
    genre_ok = [r for r in rows if not r["insufficient_data"]]
    shifts = [r["genre_shift_score"] for r in genre_ok]
    manifest_path = out_dir / "_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["l3_version"] = s.L3_VERSION
    manifest["migration"] = {
        "n_users": len(rows),
        "genre_available": len(genre_ok),
        "insufficient_data": insuf,
        "mean_genre_shift": round(sum(shifts) / len(shifts), 4) if shifts else None,
        "game_flow_net_nonzero": sum(1 for r in rows if r["game_flow_net"]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[done] 兴趣迁移 {len(rows)} 行 → {out_dir / 'migration.jsonl'}")
    print(f"       品类可用={len(genre_ok)}｜insufficient_data={insuf}"
          f"｜平均 JS={manifest['migration']['mean_genre_shift']}"
          f"｜游戏净流动非零={manifest['migration']['game_flow_net_nonzero']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
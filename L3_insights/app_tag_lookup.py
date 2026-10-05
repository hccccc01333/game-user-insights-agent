#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L3 · M2 前置：app_id → tags 冻结词表（app_tag_lookup）。

背景（L2 口径）：时间线的 `review/post/wishlist` 事件带 `app_id` 但**无 tags**。
app 品类是静态属性、非动作 → 从 L1 各用户面（`following_app` / `favorite_app` /
`feed_review` / `feed_moment` / `wishlist`，均携带 `app.tags`）汇总 `app_id → tags[]`
并冻结，供 `model_migration.py` 给事件补品类。**无需重跑 L2**。

产出：
    data/processed/user_insights/app_tag_lookup.json
    data/processed/user_insights/_manifest.json （补 app_tag_lookup 段）

口径见 L3_insights/README.md §3.2（a）。

运行（在仓库根目录）：
    python L3_insights/app_tag_lookup.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import l3_schema as s

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "user_profile"
DEFAULT_L2 = PROJECT_ROOT / "data" / "processed" / "user_features"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "user_insights"
TZ_CN = timezone(timedelta(hours=8))

CONTENT_EVENTS = ("review", "post", "wishlist")


def _tags_of(app_obj) -> list[str]:
    """从一个 app 对象取 tag value 列表（去空）。"""
    out = []
    for t in (app_obj or {}).get("tags") or []:
        v = (t or {}).get("value")
        if v:
            out.append(v)
    return out


def _collect_from_surface(surface: str, items: list) -> list[tuple]:
    """按面解析 (app_id, tags[])；路径差异在此集中处理。"""
    pairs = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        if surface == "following_app":
            app = it
        elif surface == "favorite_app":
            app = it.get("app") or {}
        elif surface in ("feed_review", "feed_moment"):
            app = ((it.get("moment") or {}).get("app")) or {}
        elif surface == "wishlist":
            app = it.get("app") or {}
        else:
            continue
        aid = app.get("id") or it.get("app_id")
        if aid is None:
            continue
        tags = _tags_of(app)
        if tags:
            pairs.append((int(aid), tags))
    return pairs


def build_lookup(raw_dir: Path) -> tuple[dict, dict]:
    """汇总所有用户面 → {app_id: [tags...]}（确定性排序）+ 统计。"""
    acc: dict[int, Counter] = {}
    n_files = 0
    files = sorted(p for p in raw_dir.glob("*.json") if not p.name.startswith("_"))
    for p in files:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # 单文件坏 → 跳过并显式报告
            print(f"[skip] {p.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        n_files += 1
        surfaces = rec.get("surfaces") or {}
        for surface in ("following_app", "favorite_app", "feed_review", "feed_moment", "wishlist"):
            for aid, tags in _collect_from_surface(surface, surfaces.get(surface)):
                c = acc.setdefault(aid, Counter())
                c.update(tags)

    apps = {str(aid): sorted(c) for aid, c in sorted(acc.items())}
    stats = {
        "n_files": n_files,
        "n_apps": len(apps),
        "n_app_tag_pairs": int(sum(len(v) for v in apps.values())),
    }
    return apps, stats


def timeline_coverage(l2_dir: Path, apps: dict) -> dict:
    """用时间线中带 app_id 的内容事件衡量词表覆盖度（供探索与诚实披露）。"""
    path = l2_dir / "timeline.jsonl"
    if not path.exists():
        return {"status": "no_timeline"}
    probe, covered, events = set(), set(), 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("event_type") in CONTENT_EVENTS and e.get("app_id") is not None:
                events += 1
                aid = str(e["app_id"])
                probe.add(aid)
                if aid in apps:
                    covered.add(aid)
    return {
        "status": "ok",
        "content_events_with_app": events,
        "distinct_app_ids": len(probe),
        "covered_app_ids": len(covered),
        "coverage_ratio": round(len(covered) / len(probe), 4) if probe else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 M2 前置：app_id → tags 冻结词表")
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW), help="L1 原始 user_profile 目录")
    ap.add_argument("--l2-dir", default=str(DEFAULT_L2), help="L2 产出目录（取 timeline 估覆盖度）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="L3 产出目录")
    args = ap.parse_args()

    raw_dir, l2_dir, out_dir = Path(args.raw_dir), Path(args.l2_dir), Path(args.out_dir)
    if not raw_dir.exists():
        print(f"[stop] 缺原始目录：{raw_dir}", file=sys.stderr)
        return 2

    apps, stats = build_lookup(raw_dir)
    cov = timeline_coverage(l2_dir, apps)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "l3_version": s.L3_VERSION,
        "generated_at": datetime.now(TZ_CN).isoformat(timespec="seconds"),
        "source": str(raw_dir.relative_to(PROJECT_ROOT)),
        "stats": stats,
        "timeline_coverage": cov,
        "apps": apps,
    }
    (out_dir / "app_tag_lookup.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    manifest_path = out_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["l3_version"] = s.L3_VERSION
    manifest["app_tag_lookup"] = {"stats": stats, "timeline_coverage": cov}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[done] app_tag_lookup {stats['n_apps']} 个 app（{stats['n_app_tag_pairs']} tag 对）"
          f" → {out_dir / 'app_tag_lookup.json'}")
    if cov.get("status") == "ok":
        print(f"       timeline 覆盖：{cov['covered_app_ids']}/{cov['distinct_app_ids']} "
              f"= {cov['coverage_ratio']:.1%}（内容事件 {cov['content_events_with_app']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
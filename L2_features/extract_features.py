#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L2 特征抽取①：用户级特征表（每用户一行）。

读 L1 原始 JSON（data/raw/user_profile/*.json），按 feature_dict.py
的口径算出五维特征 + 固定品类向量，落盘：

    data/processed/user_features/
    ├── features.csv          用户级特征表
    ├── features.parquet      同上（parquet 版）
    ├── feature_dict.csv      特征字典（从 feature_dict.py 自动导出）
    └── _manifest.json        版本 / 输入指纹 / 词表 / 截断与错误汇总

口径与坑清单见 L2_features/README.md §3 / §6 / §10。

运行（在仓库根目录）：
    python L2_features/extract_features.py
    python L2_features/extract_features.py --limit 20          # 小样试跑
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import feature_dict as fd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "data" / "raw" / "user_profile"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "user_features"
TZ_CN = timezone(timedelta(hours=8))

# 截断检测映射：数据面 → detail.stat 里的总量字段
TRUNCATION_MAP = [
    ("feed_review", "created_review_count"),
    ("feed_moment", "created_moment_count"),
    ("favorite_app", "favorite_app_count"),
    ("favorite_moment", "favorite_moment_count"),
    ("following_app", "following_app_count"),
    ("following_user", "following_count"),
    ("following_hashtag", "following_hashtag_count"),
    ("fans", "fans_count"),
    ("badge", "badges_count"),
]


# ── 基础工具 ────────────────────────────────────────────────

def _int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _mean(xs: list[float]):
    return statistics.fmean(xs) if xs else None


def _std(xs: list[float]):
    return statistics.pstdev(xs) if len(xs) >= 2 else None


def _mode(xs: list) -> str:
    xs = [x for x in xs if x not in (None, "")]
    if not xs:
        return "unknown"
    return Counter(xs).most_common(1)[0][0]


def _pct(num: int, den: int):
    return (num / den) if den else None


def _parse_ts(v) -> int | None:
    ts = _int(v, -1)
    return ts if ts > 0 else None


# ── 输入加载与全量词表 ──────────────────────────────────────

def load_records(in_dir: Path, limit: int) -> list[tuple[Path, dict]]:
    records = []
    files = sorted(p for p in in_dir.glob("*.json") if not p.name.startswith("_"))
    for p in (files[:limit] if limit else files):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # 单文件坏 → 跳过并显式报告
            print(f"[skip] {p.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rec.setdefault("uid_hash", p.stem)
        records.append((p, rec))
    return records


def tag_counter(rec: dict) -> Counter:
    c: Counter = Counter()
    for item in rec["surfaces"].get("following_app") or []:
        for t in item.get("tags") or []:
            value = (t or {}).get("value")
            if value:
                c[value] += 1
    return c


def build_tag_vocab(records: list[tuple[Path, dict]]) -> list[str]:
    """全量词表：按频次取 top-N，同频按字典序（确定性）。"""
    total: Counter = Counter()
    for _, rec in records:
        total.update(tag_counter(rec))
    ranked = sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[: fd.TAG_VECTOR_N]]


def resolve_as_of(records: list[tuple[Path, dict]]) -> int:
    """统一基准时刻 = 全量 fetched_at 最大值（确定性；抗重跑漂移）。"""
    stamps = []
    for _, rec in records:
        try:
            stamps.append(int(datetime.fromisoformat(rec["fetched_at"]).timestamp()))
        except (KeyError, ValueError):
            continue
    if stamps:
        return max(stamps)
    return int(datetime.now(TZ_CN).timestamp())


# ── 各维度特征计算 ──────────────────────────────────────────

def _moments(rec: dict, *surfaces: str) -> list[dict]:
    out = []
    for s in surfaces:
        for item in rec["surfaces"].get(s) or []:
            m = item.get("moment")
            if isinstance(m, dict):
                out.append(m)
    return out


def _exact_times(rec: dict) -> list[int]:
    """全部 exact 时间事件（评价/动态发布、心愿单、徽章、关注）。"""
    times = [t for t in (m.get("publish_time") for m in _moments(rec, "feed_review", "feed_moment")) if t]
    for w in rec["surfaces"].get("wishlist") or []:
        if w.get("created_time"):
            times.append(w["created_time"])
    for b in rec["surfaces"].get("badge") or []:
        if b.get("time"):
            times.append(b["time"])
    for u in rec["surfaces"].get("following_user") or []:
        if u.get("created_time"):
            times.append(u["created_time"])
    return [int(t) for t in times if t]


def compute_features(
    rec: dict, tag_vocab: list[str], as_of: int
) -> dict:
    surfaces = rec["surfaces"]
    detail = surfaces.get("detail") or {}
    stat = detail.get("stat") or {}
    errors = rec.get("errors") or {}
    row: dict = {"uid_hash": rec.get("uid_hash", ""), "feature_version": fd.FEATURE_VERSION}

    # ── 身份 / 生命周期 ──────────────────────────────
    tenure = _int(stat.get("created_days"), -1)
    row["tenure_days"] = tenure
    if tenure < 0:
        row["tenure_bucket"] = "unknown"
    elif tenure < fd.TENURE_BUCKET_BOUNDS[0]:
        row["tenure_bucket"] = "<1y"
    elif tenure < fd.TENURE_BUCKET_BOUNDS[1]:
        row["tenure_bucket"] = "1-3y"
    elif tenure < fd.TENURE_BUCKET_BOUNDS[2]:
        row["tenure_bucket"] = "3-5y"
    else:
        row["tenure_bucket"] = ">5y"
    row["gender"] = detail.get("gender") or "unknown"
    row["country"] = detail.get("country") or "unknown"
    row["language"] = detail.get("language") or "unknown"
    row["ip_location"] = detail.get("ip_location") or "unknown"
    row["is_silent"] = bool(detail.get("is_silent"))
    row["is_deactivated"] = bool(detail.get("is_deactivated"))
    row["is_deleted"] = bool(detail.get("is_deleted"))
    if not detail:
        row["account_status"] = "unknown"
    elif row["is_deleted"]:
        row["account_status"] = "deleted"
    elif row["is_deactivated"]:
        row["account_status"] = "deactivated"
    elif row["is_silent"]:
        row["account_status"] = "silent"
    else:
        row["account_status"] = "active"
    row["badge_count"] = _int(stat.get("badges_count"))
    row["badge_wear_count"] = len(detail.get("wear_badges") or [])
    badge_times = [int(b["time"]) for b in surfaces.get("badge") or [] if b.get("time")]
    row["first_seen_proxy_ts"] = min(badge_times) if badge_times else None

    truncated = []
    for surface, stat_key in TRUNCATION_MAP:
        total = stat.get(stat_key)
        if total is None:
            continue
        if len(surfaces.get(surface) or []) < _int(total):
            truncated.append(surface)
    row["truncated_surfaces"] = ";".join(truncated)

    # ── 活跃 / 参与强度 ──────────────────────────────
    n_review = _int(stat.get("created_review_count"))
    n_moment = _int(stat.get("created_moment_count"))
    n_post = _int(stat.get("created_post_count"))
    n_topic = _int(stat.get("created_topic_count"))
    n_video = _int(stat.get("created_video_count"))
    row["review_count"], row["moment_count"] = n_review, n_moment
    row["post_count"], row["topic_count"], row["video_count"] = n_post, n_topic, n_video
    content_total = n_review + n_moment + n_post + n_topic + n_video
    row["content_total"] = content_total
    row["content_per_year"] = round(content_total / max(tenure / 365, 0.08), 3) if tenure > 0 else 0
    row["played_app_count"] = _int(stat.get("played_app_count"))
    row["playing_app_count"] = _int(stat.get("playing_app_count"))
    row["history_app_count"] = _int(stat.get("history_app_count"))
    row["played_spent_total"] = _int(stat.get("played_spent"))
    row["reserved_count"] = _int(stat.get("reserved_count"))
    row["cloud_game_played_count"] = _int(stat.get("cloud_game_played_count"))

    own_times = [int(m["publish_time"]) for m in _moments(rec, "feed_review", "feed_moment") if m.get("publish_time")]
    row["recency_days"] = round((as_of - max(own_times)) / 86400, 3) if own_times else None
    for days, key in ((30, "act_30d"), (90, "act_90d"), (180, "act_180d")):
        row[key] = sum(1 for t in own_times if t >= as_of - days * 86400)
    row["decay_ratio"] = round(row["act_30d"] / max(row["act_180d"], 1), 3)
    hours = [datetime.fromtimestamp(t, TZ_CN).hour for t in own_times]
    row["active_hour_top"] = Counter(hours).most_common(1)[0][0] if hours else None
    all_times = _exact_times(rec)
    weekend = sum(1 for t in all_times if datetime.fromtimestamp(t, TZ_CN).weekday() >= 5)
    row["weekend_ratio"] = round(_pct(weekend, len(all_times)), 3) if all_times else 0
    row["device_top"] = _mode([m.get("device") for m in _moments(rec, "feed_review", "feed_moment")])

    # ── 兴趣 / 游戏图谱 ──────────────────────────────
    row["following_app_count"] = _int(stat.get("following_app_count"))
    row["favorite_app_count"] = _int(stat.get("favorite_app_count"))
    wishlist = surfaces.get("wishlist") or []
    row["wishlist_count"] = len(wishlist)
    row["want_app_count"] = _int(stat.get("app_wishlist_count"))
    show = detail.get("show_setting") or {}
    row["wishlist_locked"] = bool(show.get("show_app_wishlist") is False or errors.get("wishlist"))

    tc = tag_counter(rec)
    t_total = sum(tc.values())
    row["distinct_tag_count"] = len(tc)
    row["tag_entropy"] = round(-sum((c / t_total) * math.log2(c / t_total) for c in tc.values()), 3) if t_total else 0
    row["genre_top1_ratio"] = round(max(tc.values()) / t_total, 3) if t_total else 0
    for i, tag in enumerate(tag_vocab, 1):
        row[f"tag_top{i:02d}"] = round(_pct(tc.get(tag, 0), t_total) or 0, 4)

    follow_ratings = [
        _float((item.get("stat") or {}).get("rating", {}).get("score") if isinstance((item.get("stat") or {}).get("rating"), dict) else None)
        for item in surfaces.get("following_app") or []
    ]
    follow_ratings = [r for r in follow_ratings if r is not None]
    row["avg_follow_rating"] = round(_mean(follow_ratings), 3) if follow_ratings else None
    row["follow_rating_std"] = round(_std(follow_ratings), 3) if _std(follow_ratings) is not None else None

    scores = []
    dims: dict[str, list[int]] = {"degree_of_freedom": [], "gameplay": [], "operation": [], "visual_music": []}
    for m in _moments(rec, "feed_review"):
        review = m.get("review") or {}
        s = _float(review.get("score"))
        if s is not None:
            scores.append(s)
        for r in review.get("ratings") or []:
            t, v = (r or {}).get("type"), (r or {}).get("value")
            if t in dims and v in ("up", "down"):
                dims[t].append(1 if v == "down" else 0)
    row["review_score_mean"] = round(_mean(scores), 3) if scores else None
    row["review_score_std"] = round(_std(scores), 3) if _std(scores) is not None else None
    dist = Counter(int(s) for s in scores)
    row["review_score_dist"] = ";".join(f"{k}:{dist.get(k, 0)}" for k in range(1, 6))
    for dim, vals in dims.items():
        row[f"dim_neg_rate_{dim}"] = round(_pct(sum(vals), len(vals)), 3) if vals else None
    recent_wish = sum(1 for w in wishlist if _parse_ts(w.get("created_time")) and _parse_ts(w["created_time"]) >= as_of - 365 * 86400)
    row["wishlist_recent_ratio"] = round(_pct(recent_wish, len(wishlist)), 3) if wishlist else None

    # ── 社交 / 关系 ──────────────────────────────────
    following, fans = _int(stat.get("following_count")), _int(stat.get("fans_count"))
    row["following_count"], row["fans_count"] = following, fans
    row["follower_ratio"] = round(fans / max(following, 1), 3)
    fu = surfaces.get("following_user") or []
    fa = surfaces.get("fans") or []
    verified = [u.get("verified") for u in fu]
    present = [v for v in verified if v is not None]
    row["verified_following_ratio"] = round(_pct(sum(1 for v in present if v), len(present)), 3) if present else None
    row["verified_following_cov"] = round(_pct(len(present), len(fu)), 3) if fu else None
    sources = [u.get("follow_source") for u in fu]
    src_present = [s for s in sources if s]
    row["follow_source_cov"] = round(_pct(len(src_present), len(fu)), 3) if fu else None
    row["following_alive_ratio"] = round(_pct(sum(1 for u in fu if not u.get("is_deleted") and not u.get("is_deactivated")), len(fu)), 3) if fu else None
    row["fans_alive_ratio"] = round(_pct(sum(1 for u in fa if not u.get("is_deleted") and not u.get("is_deactivated")), len(fa)), 3) if fa else None
    row["follow_source_top"] = _mode(src_present)
    fu_ids = {u.get("id") for u in fu if u.get("id") is not None}
    fan_ids = {u.get("id") for u in fa if u.get("id") is not None}
    row["mutual_count"] = len(fu_ids & fan_ids) if (fu_ids and fan_ids) else None
    row["following_hashtag_count"] = _int(stat.get("following_hashtag_count"))
    row["following_developer_count"] = _int(stat.get("following_developer_count"))
    row["forum_count"] = _int(stat.get("forum_count"))

    # ── 内容影响力 / 口碑 ────────────────────────────
    row["voteup_received"] = _int(stat.get("voteup_count"))
    row["votefunny_received"] = _int(stat.get("votefunny_count"))
    row["be_voted_up_review"] = _int(stat.get("be_voted_up_review_count"))
    row["be_voted_up_moment"] = _int(stat.get("be_voted_up_moment_count"))
    row["be_favorited_count"] = _int(stat.get("be_favorited_count"))
    ups = [_int((m.get("stat") or {}).get("ups")) for m in _moments(rec, "feed_review", "feed_moment")]
    ups = [u for u in ups if u > 0]
    row["avg_ups_per_moment"] = round(_mean(ups), 3) if ups else None
    row["max_ups"] = max(ups) if ups else 0
    speeds = []
    for m in _moments(rec, "feed_review", "feed_moment"):
        p, c = _parse_ts(m.get("publish_time")), _parse_ts(m.get("commented_time"))
        if p and c and c >= p:
            speeds.append(c - p)
    row["interaction_speed_median"] = float(statistics.median(speeds)) if speeds else None
    row["favorite_moment_count"] = _int(stat.get("favorite_moment_count"))
    row["purchased_app_count"] = _int(stat.get("purchased_app_count"))
    row["app_achievement_count"] = _int(stat.get("app_achievement_count"))

    return row


# ── 落盘 ────────────────────────────────────────────────────

def input_fingerprint(records: list[tuple[Path, dict]]) -> str:
    h = hashlib.sha256()
    for p, _ in records:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description="L2 特征抽取①：用户级特征表")
    ap.add_argument("--in-dir", default=str(DEFAULT_IN), help="L1 原始 JSON 目录")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="产出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个用户（小样试跑）")
    ap.add_argument("--no-parquet", action="store_true", help="跳过 parquet 产出")
    args = ap.parse_args()

    records = load_records(Path(args.in_dir), args.limit)
    if not records:
        print("[stop] 没有可处理的输入", file=sys.stderr)
        return 2
    vocab = build_tag_vocab(records)
    as_of = resolve_as_of(records)
    print(f"[info] 用户 {len(records)}｜词表 top{fd.TAG_VECTOR_N}｜基准时刻 {datetime.fromtimestamp(as_of, TZ_CN).isoformat(timespec='seconds')}")

    rows = [compute_features(rec, vocab, as_of) for _, rec in records]
    df = pd.DataFrame(rows, columns=fd.column_names())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "features.csv", index=False, encoding="utf-8-sig")
    if not args.no_parquet:
        df.to_parquet(out_dir / "features.parquet", index=False)
    pd.DataFrame(fd.export_rows()).to_csv(out_dir / "feature_dict.csv", index=False, encoding="utf-8-sig")

    trunc_by_surface: Counter = Counter()
    users_with_errors = 0
    for r in rows:
        for s in filter(None, str(r["truncated_surfaces"]).split(";")):
            trunc_by_surface[s] += 1
    for _, rec in records:
        if rec.get("errors"):
            users_with_errors += 1

    manifest = {
        "feature_version": fd.FEATURE_VERSION,
        "generated_at": datetime.now(TZ_CN).isoformat(timespec="seconds"),
        "n_users": len(records),
        "as_of": as_of,
        "input_fingerprint": input_fingerprint(records),
        "tag_vocab": vocab,
        "truncation_summary": {"users_truncated": sum(1 for r in rows if r["truncated_surfaces"]), "by_surface": dict(trunc_by_surface)},
        "users_with_errors": users_with_errors,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[done] 特征表 {df.shape[0]} 行 × {df.shape[1]} 列 → {out_dir}")
    print(f"       截断用户 {manifest['truncation_summary']['users_truncated']}｜错误用户 {users_with_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
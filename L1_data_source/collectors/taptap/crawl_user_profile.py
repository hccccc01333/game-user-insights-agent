#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TapTap 用户画像采集（用户锚点轨 v0）。

对每个种子用户抓取以下公开数据面并落盘（原始 JSON）：
  user/v1/detail                     用户档案（Id 参数）
  feed/v7/by-user                    评价（type=review）与动态
  favorite/v2/by-user                收藏（app/moment/collection/hashtag/event）
  app-wishlist/v1/list-by-user       心愿单
  user-badge/v1/by-user              徽章
  friendship/v1/following-by-user    关注（app/user/hashtag）
  friendship/v1/fans-by-user         粉丝（游标翻页）

产出目录（默认 data/raw/user_profile/）：
  <uid_hash>.json    单个用户的全部原始响应
  _index.jsonl       采集索引（每人一行汇总）
  _checkpoint 机制：已存在的用户文件默认跳过（--force 覆盖），可断点续跑

用户标识一律加盐哈希落盘；明文只从本地种子文件读入内存，不写进任何产出文件。
限速：请求之间 sleep；403/429 停止当前数据面并记录错误，不重试、不绕过。

环境变量（或 --env-file 指定的本地 .env，均不入库）：
  TAPTAP_X_UA        必填：浏览器里任取一个 TapTap webapiv2 请求的 X-UA
  TAPTAP_HASH_SALT   必填：哈希盐（可用 --salt 传）。延续历史数据必须沿用原盐
                     （存本地 .env，勿提交/公开），否则 uid_hash 与新采不一致。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlparse

import requests

BASE = "https://www.taptap.cn/webapiv2"
TZ_CN = timezone(timedelta(hours=8))
UA_HEADER = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEEDS = PROJECT_ROOT / "seeds" / "user_seeds.csv"  # 本地种子（不入库，可 --seeds-csv 改）
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "user_profile"

# (名称, 路径, 固定参数, 每页条数)
OFFSET_SURFACES = [
    ("feed_review", "/feed/v7/by-user", {"type": "review", "with_top_in_type": "true"}, 10),
    ("feed_moment", "/feed/v7/by-user", {}, 10),
    ("favorite_app", "/favorite/v2/by-user", {"type": "app"}, 10),
    ("favorite_moment", "/favorite/v2/by-user", {"type": "moment"}, 10),
    ("favorite_collection", "/favorite/v2/by-user", {"type": "collection"}, 10),
    ("favorite_hashtag", "/favorite/v2/by-user", {"type": "hashtag"}, 10),
    ("favorite_event", "/favorite/v2/by-user", {"type": "event"}, 10),
    ("wishlist", "/app-wishlist/v1/list-by-user", {}, 10),
    ("badge", "/user-badge/v1/by-user", {}, 20),
    ("following_app", "/friendship/v1/following-by-user", {"type": "app"}, 10),
    ("following_user", "/friendship/v1/following-by-user", {"type": "user"}, 10),
    ("following_hashtag", "/friendship/v1/following-by-user", {"type": "hashtag"}, 10),
]

# 游标翻页的数据面（跟随 next_page 原样请求）
CURSOR_SURFACES = [
    ("fans", "/friendship/v1/fans-by-user", {}),
]


def hash_uid(uid: str, salt: str) -> str:
    return "h_" + hashlib.sha256(f"{salt}:{uid}".encode("utf-8")).hexdigest()[:16]


def load_env_file(env_file: str) -> dict[str, str]:
    """读取 .env 风格文件（KEY=VALUE；忽略 # 注释与空行），返回键值字典。"""
    values: dict[str, str] = {}
    path = Path(env_file) if env_file else None
    if path and path.exists():
        # utf-8-sig：容忍 Windows 记事本另存的 BOM 头
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def resolve_xua(env_file: str) -> str | None:
    """TAPTAP_X_UA：环境变量优先，其次 .env 文件；都没有 → None（调用方报错）。

    从浏览器拷贝的 X-UA 常为百分号编码（%3D/%26），服务端按明文解析（实测
    编码形式会报 INVALID_XUA），这里统一解码；已是明文的值不受影响。
    """
    raw = os.environ.get("TAPTAP_X_UA") or load_env_file(env_file).get("TAPTAP_X_UA")
    return unquote(raw) if raw else None


def resolve_salt(env_file: str, cli_salt: str) -> str | None:
    """TAPTAP_HASH_SALT：CLI 优先，其次环境变量，再次 .env 文件；都没有 → None。

    刻意不给默认值：换盐会让 uid_hash 与历史数据不一致，必须显式提供。
    延续旧数据请沿用原盐，写进本地 .env（勿提交）即可。
    """
    return (
        cli_salt
        or os.environ.get("TAPTAP_HASH_SALT")
        or load_env_file(env_file).get("TAPTAP_HASH_SALT")
    )


def load_seeds(csv_path: Path, limit: int) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uid = (row.get("user_id") or row.get("uid") or "").strip()
            if uid and uid not in seen:
                seen.add(uid)
                ordered.append(uid)
                if limit and len(ordered) >= limit:
                    break
    return ordered


class Client:
    """带限速与错误归一的会话。"""

    def __init__(self, xua: str, sleep_sec: float) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"X-UA": xua, "User-Agent": UA_HEADER, "Referer": "https://www.taptap.cn/"}
        )
        self.sleep_sec = sleep_sec
        self.calls = 0

    def get(self, path: str, params: dict) -> tuple[dict | None, str | None]:
        url = path if path.startswith("http") else f"{BASE}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=20)
        except requests.RequestException as exc:
            return None, f"请求异常：{exc.__class__.__name__}"
        self.calls += 1
        time.sleep(self.sleep_sec)
        if resp.status_code in (403, 429):
            return None, f"HTTP {resp.status_code}（停止该数据面）"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        try:
            body = resp.json()
        except ValueError:
            return None, "响应不是 JSON"
        if not body.get("success"):
            reason = body.get("error_description") or body.get("msg") or "unknown"
            return None, f"接口错误：{reason}"
        return body.get("data") or {}, None


def fetch_offset(
    client: Client, path: str, params: dict, limit: int, max_pages: int
) -> tuple[list, str | None]:
    """offset 翻页（from += limit，读到空页或没有 next_page 为止）。"""
    items: list = []
    err: str | None = None
    from_ = 0
    for _ in range(max_pages):
        page_params = dict(params)
        page_params.update({"from": from_, "limit": limit})
        data, err = client.get(path, page_params)
        if data is None:
            break
        batch = data.get("list") or []
        items.extend(batch)
        if not batch or not data.get("next_page"):
            break
        from_ += limit
    return items, err


def fetch_cursor(
    client: Client, path: str, params: dict, max_pages: int
) -> tuple[list, str | None]:
    """游标翻页：next_page 是相对路径，原样跟随其 query。"""
    items: list = []
    err: str | None = None
    next_path, next_params = path, dict(params)
    for _ in range(max_pages):
        data, err = client.get(next_path, next_params)
        if data is None:
            break
        batch = data.get("list") or []
        items.extend(batch)
        nxt = data.get("next_page") or ""
        if not nxt or not batch:
            break
        parsed = urlparse(nxt)
        next_path = parsed.path.removeprefix("/webapiv2")
        next_params = dict(parse_qsl(parsed.query))
    return items, err


def crawl_user(client: Client, uid: str, salt: str, max_pages: int) -> dict:
    record: dict = {
        "uid_hash": hash_uid(uid, salt),
        "fetched_at": datetime.now(TZ_CN).isoformat(timespec="seconds"),
        "surfaces": {},
        "errors": {},
    }
    detail, err = client.get("/user/v1/detail", {"Id": uid})
    record["surfaces"]["detail"] = detail or {}
    if err:
        record["errors"]["detail"] = err
    for name, path, params, limit in OFFSET_SURFACES:
        items, err = fetch_offset(client, path, {"user_id": uid, **params}, limit, max_pages)
        record["surfaces"][name] = items
        if err:
            record["errors"][name] = err
    for name, path, params in CURSOR_SURFACES:
        items, err = fetch_cursor(client, path, {"user_id": uid, **params}, max_pages)
        record["surfaces"][name] = items
        if err:
            record["errors"][name] = err
    return record


def process_user(
    client: Client,
    uid: str,
    salt: str,
    max_pages: int,
    out_dir: Path,
    index_path: Path,
    force: bool = False,
) -> dict:
    """采集并落盘单个用户，返回小结（供本文件 main 与批次运行器复用）。

    已存在同名文件且未 force → 直接返回 skipped=True（断点续跑语义）。
    """
    uid_h = hash_uid(uid, salt)
    target = out_dir / f"{uid_h}.json"
    if target.exists() and not force:
        return {"uid_hash": uid_h, "skipped": True, "counts": {}, "errors": {}}
    record = crawl_user(client, uid, salt, max_pages)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {k: len(v) for k, v in record["surfaces"].items() if isinstance(v, list)}
    with index_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "uid_hash": uid_h,
                    "file": target.name,
                    "fetched_at": record["fetched_at"],
                    "counts": counts,
                    "has_detail": bool(record["surfaces"].get("detail")),
                    "errors": record["errors"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return {
        "uid_hash": uid_h,
        "skipped": False,
        "fetched_at": record["fetched_at"],
        "counts": counts,
        "errors": record["errors"],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TapTap 用户画像采集（用户锚点轨 v0）")
    ap.add_argument("--seeds-csv", default=str(DEFAULT_SEEDS), help="种子来源 CSV（取 user_id 列；默认 seeds/user_seeds.csv，仅本地）")
    ap.add_argument("--limit-users", type=int, default=5, help="最多取多少个种子用户")
    ap.add_argument("--user-ids", default="", help="直接指定用户（逗号分隔），优先于种子文件")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="产出目录")
    ap.add_argument("--sleep", type=float, default=1.0, help="请求间隔秒数")
    ap.add_argument("--max-pages", type=int, default=3, help="每个数据面最多翻几页")
    ap.add_argument("--env-file", default="", help="可选：从 .env 文件读取 TAPTAP_X_UA / TAPTAP_HASH_SALT")
    ap.add_argument("--salt", default="", help="加盐哈希盐（或环境变量 TAPTAP_HASH_SALT；延续历史数据须沿用原盐）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的用户文件")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    xua = resolve_xua(args.env_file)
    if not xua:
        print("[stop] 缺少 TAPTAP_X_UA：请设置环境变量或用 --env-file 指定", file=sys.stderr)
        return 2
    salt = resolve_salt(args.env_file, args.salt)
    if not salt:
        print(
            "[stop] 缺少哈希盐：请传 --salt、设置 TAPTAP_HASH_SALT 或写入 --env-file。\n"
            "       延续历史数据必须沿用原盐（存本地 .env，勿公开），否则 uid_hash 不一致。",
            file=sys.stderr,
        )
        return 2

    if args.user_ids:
        seeds = [s.strip() for s in args.user_ids.split(",") if s.strip()]
    else:
        seeds_path = Path(args.seeds_csv)
        if not seeds_path.exists():
            print(
                f"[stop] 种子文件不存在：{seeds_path}\n"
                "       可用 --seeds-csv 指定，或用 collect_seed_pool.py 生成种子池，或 --user-ids 直接给 uid。",
                file=sys.stderr,
            )
            return 2
        seeds = load_seeds(seeds_path, args.limit_users)
    if not seeds:
        print("[stop] 没有可用种子用户", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(xua, args.sleep)
    index_path = out_dir / "_index.jsonl"
    fresh = 0

    for idx, uid in enumerate(seeds, 1):
        uid_h = hash_uid(uid, salt)
        print(f"[{idx}/{len(seeds)}] 采集 {uid_h} ...")
        summary = process_user(
            client, uid, salt, args.max_pages, out_dir, index_path, force=args.force
        )
        if summary["skipped"]:
            print(f"[{idx}/{len(seeds)}] 跳过（已采过） {uid_h}")
            continue
        fresh += 1
        total_items = sum(summary["counts"].values())
        print(
            f"    完成：列表 {total_items} 条；错误 {len(summary['errors'])} 项；"
            f"累计请求 {client.calls}"
        )

    print(f"[done] 新采 {fresh} 人；累计请求 {client.calls} 次；产出目录 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
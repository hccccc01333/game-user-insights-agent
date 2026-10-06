#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TapTap 种子池采集（多游戏评价 → 用户 id 池）。

作用：给用户画像采集器（crawl_user_profile.py）持续补充**多游戏、多品类**的
种子用户，降低"单款游戏评价者"带来的样本偏置。

做法：调用 TapTap 网页端同源的评价列表接口（review/v2/list-by-app，字段位置
与网页端一致），按游戏翻页抽取每条评价作者的公开用户 id，去重后写入本地
种子池 CSV（默认 seeds/user_pool.csv）。

合规与限速：
  · 只采集公开可见的评价作者 id，不抓取非公开信息，不绕过权限；
  · 请求间隔限速（--sleep）；403/429 立即停止本次采集（不重试、不绕过）；
  · 种子池 CSV 含明文 user_id，**仅存本地**（seeds/ 已在 .gitignore，禁止提交）。

用法（仓库根目录执行）：
  python L1_data_source/collectors/taptap/collect_seed_pool.py --app-ids 70253,168332,283303
  # 常用：--pages-per-app 5 --sleep 1.5 --out seeds/user_pool.csv
  # 只看配置不联网：--dry-run

建议：一次选 5–8 款不同品类的游戏（RPG / 策略 / 休闲 / 二次元 等），
让扩采样本在品类上更均衡；app_id 即 TapTap 游戏页 URL 中的数字
（https://www.taptap.cn/app/<app_id>）。
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crawl_user_profile import (  # noqa: E402
    Client,
    PROJECT_ROOT,
    hash_uid,
    resolve_salt,
    resolve_xua,
)

REVIEW_PATH = "/review/v2/list-by-app"  # 与网页端同源
PAGE_LIMIT = 10
TZ_CN = timezone(timedelta(hours=8))
DEFAULT_OUT = PROJECT_ROOT / "seeds" / "user_pool.csv"
DEFAULT_CRAWLED_DIR = PROJECT_ROOT / "data" / "raw" / "user_profile"
FIELDS = ["user_id", "source_app_id", "source_review_id", "collected_at"]


def review_author_id(item: dict) -> str:
    """从一条评价里抽作者公开 id（字段位置：moment.author.user.id）。"""
    moment = item.get("moment") or {}
    author = ((moment.get("author") or {}).get("user")) or {}
    uid = author.get("id")
    return str(uid) if uid else ""


def review_id(item: dict) -> str:
    moment = item.get("moment") or {}
    review = moment.get("review") or {}
    rid = review.get("id")
    return str(rid) if rid else ""


def collect_app(
    client: Client,
    app_id: str,
    pages: int,
    xua: str,
    seen: set[str],
    crawled_dir: Path,
    salt: str,
    skip_crawled: bool,
) -> tuple[list[dict], str]:
    """翻页采集单款游戏的评价作者，返回 (新增行, 状态说明)。

    已采过（uid_hash 文件存在）且 skip_crawled 的用户不写入池。
    """
    new_rows: list[dict] = []
    note = f"完成（{pages} 页）"
    for page in range(pages):
        params = {
            "app_id": app_id,
            "from": page * PAGE_LIMIT,
            "limit": PAGE_LIMIT,
            "sort": "new",
        }  # X-UA 已由 Client 统一放请求头
        data, err = client.get(REVIEW_PATH, params)
        if data is None:
            note = f"停止：{err}"
            break
        items = data.get("list") or []
        if not items:
            note = f"完成（第 {page + 1} 页无数据）"
            break
        for item in items:
            uid = review_author_id(item)
            if not uid or uid in seen:
                continue
            seen.add(uid)
            if skip_crawled and (crawled_dir / f"{hash_uid(uid, salt)}.json").exists():
                continue
            new_rows.append(
                {
                    "user_id": uid,
                    "source_app_id": app_id,
                    "source_review_id": review_id(item),
                    "collected_at": datetime.now(TZ_CN).isoformat(timespec="seconds"),
                }
            )
    return new_rows, note


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TapTap 种子池采集（多游戏评价 → 用户 id 池）")
    ap.add_argument("--app-ids", default="", help="游戏 id 列表（逗号分隔），如 70253,168332")
    ap.add_argument("--pages-per-app", type=int, default=3, help="每款游戏翻几页（每页 10 条）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="种子池 CSV（默认 seeds/user_pool.csv，仅本地）")
    ap.add_argument("--crawled-dir", default=str(DEFAULT_CRAWLED_DIR), help="已采用户目录（跳过用）")
    ap.add_argument("--skip-crawled", action="store_true", default=True, help="跳过已采过的用户（默认开）")
    ap.add_argument("--no-skip-crawled", dest="skip_crawled", action="store_false", help="不跳过已采过的用户")
    ap.add_argument("--sleep", type=float, default=1.5, help="请求间隔秒数")
    ap.add_argument("--env-file", default="", help="可选：从 .env 文件读取 TAPTAP_X_UA / TAPTAP_HASH_SALT")
    ap.add_argument("--salt", default="", help="加盐哈希盐（或环境变量 TAPTAP_HASH_SALT）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划（不联网）")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    app_ids = [a.strip() for a in args.app_ids.split(",") if a.strip()]
    if not app_ids:
        print("[stop] 请用 --app-ids 指定至少一个游戏（逗号分隔）", file=sys.stderr)
        return 2
    out_path = Path(args.out)
    crawled_dir = Path(args.crawled_dir)

    if args.dry_run:
        print("[dry-run] 计划：")
        print(f"  游戏：{', '.join(app_ids)}（每游戏 {args.pages_per_app} 页 × {PAGE_LIMIT} 条）")
        print(f"  产出：{out_path}")
        print(f"  去重：池内去重 + 已采跳过={args.skip_crawled}（已采目录 {crawled_dir}）")
        return 0

    xua = resolve_xua(args.env_file)
    if not xua:
        print("[stop] 缺少 TAPTAP_X_UA：请设置环境变量或用 --env-file 指定", file=sys.stderr)
        return 2
    salt = resolve_salt(args.env_file, args.salt)
    if not salt:
        print(
            "[stop] 缺少哈希盐：请传 --salt、设置 TAPTAP_HASH_SALT 或写入 --env-file。",
            file=sys.stderr,
        )
        return 2

    seen: set[str] = set()
    rows_existing: list[dict] = []
    if out_path.exists():
        with out_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                uid = (row.get("user_id") or row.get("uid") or "").strip()
                if uid and uid not in seen:
                    seen.add(uid)
                    rows_existing.append({k: row.get(k, "") for k in FIELDS})

    client = Client(xua, args.sleep)
    new_rows: list[dict] = []
    blocked = False
    for app_id in app_ids:
        rows, note = collect_app(
            client, app_id, args.pages_per_app, xua, seen, crawled_dir, salt, args.skip_crawled
        )
        new_rows.extend(rows)
        print(f"[app {app_id}] 新增 {len(rows)} 人；{note}")
        if "HTTP 403" in note or "HTTP 429" in note:
            blocked = True
            break

    if new_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_existing + new_rows)

    print(
        f"[done] 池内累计 {len(seen)} 人；本次新增 {len(new_rows)}；"
        f"请求 {client.calls} 次；产出 {out_path}"
    )
    if blocked:
        print("[pause] 触发 403/429，已停止本次采集（请间隔一段时间再试，勿提高频率）", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
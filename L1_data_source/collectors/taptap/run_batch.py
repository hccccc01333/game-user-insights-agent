#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批次运行器：按日预算慢跑用户画像采集（扩采到 1000–2000 人）。

职责（把 crawl_user_profile.py 包成可日常慢跑的任务）：
  1. 从种子池（默认 seeds/user_pool.csv）按序取人；已采过（uid_hash 文件存在）自动跳过；
  2. 每日小批（--daily-users，默认 20）+ 请求限速（--sleep）+ 用户间长间隔；
  3. 403/429 自动暂停：命中即停止整批（wishlist 的 403 属用户隐私设置，不计入）；
  4. 失败队列（与种子池同目录的 failed_queue.jsonl）：detail 缺失或错误数达到
     --fail-min-errors 的用户入队；--retry-failed 时优先重试，成功后移出队列；
  5. 批次记录追加 data/raw/user_profile/_batches.jsonl；累计跨过 500/1000/2000
     人台阶时提示复盘（重跑 L2/L3 查截断率与分布漂移）。

用法（仓库根目录执行）：
  # ① 小批验证（先跑 5–10 人，确认接口与限速正常）
  python L1_data_source/collectors/taptap/run_batch.py --daily-users 10
  # ② 日常慢跑（每天 20 人，累计到 2000 人即停）
  python L1_data_source/collectors/taptap/run_batch.py --daily-users 20 --target-total 2000
  # ③ 只评估计划（不联网）：--dry-run

注意：失败队列含明文 user_id，仅存本地（seeds/ 已在 .gitignore，禁止提交）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crawl_user_profile import (  # noqa: E402
    Client,
    PROJECT_ROOT,
    hash_uid,
    process_user,
    resolve_salt,
    resolve_xua,
)

TZ_CN = timezone(timedelta(hours=8))
DEFAULT_POOL = PROJECT_ROOT / "seeds" / "user_pool.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "user_profile"
MILESTONES = (500, 1000, 2000)


def now_cn() -> str:
    return datetime.now(TZ_CN).isoformat(timespec="seconds")


def load_pool(path: Path) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uid = (row.get("user_id") or row.get("uid") or "").strip()
            if uid and uid not in seen:
                seen.add(uid)
                ordered.append(uid)
    return ordered


def load_failed_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except ValueError:
            continue
    return items


def save_failed_queue(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
        encoding="utf-8",
    )


def total_users(out_dir: Path) -> int:
    return len(list(out_dir.glob("h_*.json")))


def block_signal(errors: dict[str, str]) -> str:
    """403/429 暂停信号；wishlist 的 403 属用户隐私设置（预期内），不计入。"""
    hits: list[str] = []
    for surface, msg in errors.items():
        if "429" in msg:
            hits.append(f"{surface}:429")
        elif "403" in msg and surface != "wishlist":
            hits.append(f"{surface}:403")
    return ";".join(hits)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="批次运行器：按日预算慢跑用户画像采集")
    ap.add_argument("--pool", default=str(DEFAULT_POOL), help="种子池 CSV（含 user_id 列；默认 seeds/user_pool.csv）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="产出目录（默认 data/raw/user_profile）")
    ap.add_argument("--daily-users", type=int, default=20, help="本批最多新采多少人（默认 20）")
    ap.add_argument("--target-total", type=int, default=2000, help="累计达到该值即停（默认 2000）")
    ap.add_argument("--sleep", type=float, default=1.0, help="请求间隔秒数（传给采集器）")
    ap.add_argument("--sleep-per-user", type=float, default=3.0, help="用户之间的额外间隔秒数")
    ap.add_argument("--max-pages", type=int, default=3, help="每个数据面最多翻几页")
    ap.add_argument("--fail-min-errors", type=int, default=3, help="错误数达到该值记入失败队列")
    ap.add_argument("--env-file", default="", help="可选：从 .env 文件读取 TAPTAP_X_UA / TAPTAP_HASH_SALT")
    ap.add_argument("--salt", default="", help="加盐哈希盐（或环境变量 TAPTAP_HASH_SALT）")
    ap.add_argument("--retry-failed", action="store_true", help="优先重试失败队列中的用户")
    ap.add_argument("--dry-run", action="store_true", help="只评估计划（不联网）")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pool_path = Path(args.pool)
    out_dir = Path(args.out_dir)
    index_path = out_dir / "_index.jsonl"
    failed_path = pool_path.parent / "failed_queue.jsonl"

    if not pool_path.exists():
        print(
            f"[stop] 种子池不存在：{pool_path}（先跑 collect_seed_pool.py 生成）",
            file=sys.stderr,
        )
        return 2
    pool = load_pool(pool_path)
    failed = load_failed_queue(failed_path)
    already_failed = {f.get("uid") for f in failed if f.get("uid")}

    salt = resolve_salt(args.env_file, args.salt)
    if not salt:
        print(
            "[stop] 缺少哈希盐：请传 --salt、设置 TAPTAP_HASH_SALT 或写入 --env-file。",
            file=sys.stderr,
        )
        return 2

    # 本批名单：--retry-failed 时失败队列优先；已采过（uid_hash 文件存在）跳过
    retry_uids = [f.get("uid", "") for f in failed if f.get("uid")] if args.retry_failed else []
    ordered: list[str] = []
    seen: set[str] = set()
    for uid in retry_uids + pool:
        if uid and uid not in seen:
            seen.add(uid)
            ordered.append(uid)
    todo = [uid for uid in ordered if not (out_dir / f"{hash_uid(uid, salt)}.json").exists()]
    skipped = len(ordered) - len(todo)
    planned = todo[: args.daily_users]

    if args.dry_run:
        print("[dry-run] 计划：")
        print(f"  池内 {len(pool)} 人；失败队列 {len(failed)} 人；已采跳过 {skipped} 人")
        print(f"  本批计划采集 {len(planned)} 人" + (f"（第一步 {planned[0]}）" if planned else "（无）"))
        print(f"  当前总量 {total_users(out_dir)} → 目标 {args.target_total}")
        return 0

    xua = resolve_xua(args.env_file)
    if not xua:
        print("[stop] 缺少 TAPTAP_X_UA：请设置环境变量或用 --env-file 指定", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(xua, args.sleep)
    started = now_cn()
    total_before = total_users(out_dir)
    ok = 0
    queued = 0
    attempts = 0
    stop_reason = "nothing_to_do" if not planned else "queue_exhausted"

    for uid in planned:
        if total_users(out_dir) >= args.target_total:
            stop_reason = "target_total"
            break
        uid_h = hash_uid(uid, salt)
        print(f"[{attempts + 1}/{len(planned)}] 采集 {uid_h} ...")
        summary = process_user(client, uid, salt, args.max_pages, out_dir, index_path)
        attempts += 1
        if summary["skipped"]:
            continue
        errs: dict[str, str] = summary["errors"]

        if ("detail" in errs) or (len(errs) >= args.fail_min_errors):
            queued += 1
            record = {
                "uid": uid,
                "uid_hash": uid_h,
                "reason": "; ".join(f"{k}:{v}" for k, v in errs.items())[:300],
                "at": now_cn(),
            }
            failed = [f for f in failed if f.get("uid") != uid] + [record]
            save_failed_queue(failed_path, failed)
            print(f"    入失败队列：{record['reason'][:100]}")
        else:
            ok += 1
            if uid in already_failed:
                failed = [f for f in failed if f.get("uid") != uid]
                save_failed_queue(failed_path, failed)
            print(f"    完成：列表 {sum(summary['counts'].values())} 条；错误 {len(errs)} 项")

        sig = block_signal(errs)
        if sig:
            stop_reason = f"auto_pause_blocked({sig})"
            print(f"[pause] 检测到 {sig} → 立即停止本批（疑似反爬信号，勿自动重试）")
            break
        time.sleep(args.sleep_per_user)
    else:
        if planned:
            stop_reason = "daily_budget_done"

    total_after = total_users(out_dir)
    batch = {
        "batch": datetime.now(TZ_CN).strftime("%Y%m%d_%H%M%S"),
        "started_at": started,
        "ended_at": now_cn(),
        "pool_total": len(pool),
        "attempted": attempts,
        "ok": ok,
        "queued": queued,
        "skipped_existing": skipped,
        "stop_reason": stop_reason,
        "total_users": total_after,
        "target_total": args.target_total,
        "requests": client.calls,
    }
    with (out_dir / "_batches.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(batch, ensure_ascii=False) + "\n")

    for m in MILESTONES:
        if total_before < m <= total_after:
            print(
                f"[milestone] 累计 {total_after} 人，已跨过 {m} 人台阶：建议暂停扩采，"
                "重跑 L2/L3 并复盘截断率/分布漂移（见根 README「扩采节奏」）"
            )
    print(
        f"[done] 本批新采 {ok} 人（入失败队列 {queued}）；总量 {total_after}/{args.target_total}；"
        f"停止原因 {stop_reason}；请求 {client.calls} 次"
    )
    return 0 if not stop_reason.startswith("auto_pause") else 3


if __name__ == "__main__":
    raise SystemExit(main())
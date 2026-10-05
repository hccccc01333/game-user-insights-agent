#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reddit 校准脚本：为游戏社区合成数据生成器提取真实行为分布锚点（数据获取阶段产物）。

定位与边界
  本脚本只做「数据获取阶段」的事：从 Arctic Shift（Reddit 归档 API）采集真实游戏社区
  行为分布，产出生成器参数对照表（reddit_params.csv）供用户审核。不包含任何
  Agent / 生成器实现（那部分由用户设计）。

  三类口径边界（报告与后续 README 中须保持区分，不得混用）：
    1) 合成数据      —— 未来生成器产出的模拟用户行为（本脚本不产出）
    2) Reddit 校准  —— 本脚本产出的锚点值（真实社区，有口径与局限）
    3) 论文口径引用 —— 文献中的社区行为参数（本脚本不处理）

用法（Windows PowerShell，勿用 &&）
  python scripts/reddit_calibration.py selftest          # 分页游标逻辑自检（无网络）
  python scripts/reddit_calibration.py all --pilot       # 小样本全链路验证（1 天 / tenure=8）
  python scripts/reddit_calibration.py collect           # 全量采集（可中断，重跑自动续传）
  python scripts/reddit_calibration.py analyze           # 全量分析（产出 stats / params / report）
  python scripts/reddit_calibration.py all               # 采集 + 分析

冻结配置（2023 Q1，全程 UTC）
  采集窗口    2023-02-20 ~ 2023-02-26（周一~周日，逐日全量采集）
  核心子版    gaming / Genshin_Impact / truegaming（评论 + 帖子）
  留存子版    truegaming / patientgamers（2023-01-02 周一起 12 个周的逐周 author 聚合）
  tenure 抽样 gaming 采集周评论作者，按活跃度（1 条 / 2-4 条 / 5+ 条）分层系统抽样，n=50

关键 API 事实（探针实测，已固化进本脚本）
  · /api/{comments,posts}/search：after 严格排除边界（>）→ 分页游标 = last_ts − 1 回看 +
    id 去重；整页无新行（同一秒内容超过一页容量）时跳过该秒（skipped_same_ts 计数）。
  · limit=auto 实测页容量 100~225 行；空页即窗口结束（不依赖"短页结束"启发式）。
  · /api/*/search/aggregate?aggregate=author 对 gaming/Genshin 必然超时（422），
    仅小版（truegaming/patientgamers）可用；返回 count 为字符串类型。
  · /api/time_series 日桶标签存在 −1h 偏移，覆盖校验按周求和、容差 ±10%。
  · /api/users/search?author=X&limit=1 的 _meta 提供最早活动时间（账号年龄代理）与
    lifetime 计数（num_comments / num_posts / total_karma）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

# ============================================================ 冻结配置（2023 Q1）
BASE = "https://arctic-shift.photon-reddit.com"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "calibration"
RAW_DIR = OUT_DIR / "raw"
CACHE_DIR = RAW_DIR / "cache"

WEEK_START = date(2023, 2, 20)          # 周一（UTC）
WEEK_DAYS = 7                           # 2023-02-20 ~ 2023-02-26
CORE_SUBS = ["gaming", "Genshin_Impact", "truegaming"]
RETENTION_SUBS = ["truegaming", "patientgamers"]
RETENTION_W0 = date(2023, 1, 2)         # 留存统计起始周一
RETENTION_WEEKS = 12
TENURE_N = 50
PACE_SECONDS = 0.5                      # 全局请求间隔（~2 req/s，礼貌限速）
FIELDS = {
    "comments": "id,author,created_utc,score,parent_id,link_id",
    "posts": "id,author,created_utc,score,num_comments",
}


def build_config(pilot: bool) -> dict:
    n_days = 1 if pilot else WEEK_DAYS
    days = [(WEEK_START + timedelta(days=i)).isoformat() for i in range(n_days)]
    return {
        "pilot": pilot,
        "days": days,
        "raw_dir": (RAW_DIR / "pilot") if pilot else RAW_DIR,
        "suffix": "_pilot" if pilot else "",
        "tenure_n": 8 if pilot else TENURE_N,
        "n_retention_weeks": 3 if pilot else RETENTION_WEEKS,
    }


# ============================================================ 基础工具
class ApiError(RuntimeError):
    pass


def day_start_epoch(day: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def day_end_epoch(day: str) -> int:
    return day_start_epoch(day) + 86400


def ts_of(row: dict) -> int:
    return int(row["created_utc"])


def utc_dt(ts) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def is_human(author) -> bool:
    """数据卫生：剔除已删号、AutoModerator、版务号与 bot 后缀账号。"""
    if not isinstance(author, str) or not author:
        return False
    if author == "[deleted]":
        return False
    low = author.lower()
    if low == "automoderator":
        return False
    if "modteam" in low:
        return False
    if low.endswith("bot"):
        return False
    return True


def safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)


def _parse_ts(v):
    """把 epoch 数值或 ISO 字符串统一解析为 epoch 秒。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dtv = datetime.fromisoformat(s)
        if dtv.tzinfo is None:
            dtv = dtv.replace(tzinfo=timezone.utc)
        return dtv.timestamp()
    except ValueError:
        return None


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if abs(x) >= 100:
            return f"{x:,.0f}"
        if abs(x) >= 10:
            return f"{x:.1f}"
        return f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return str(x)


def pct(x, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def line_count(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def dig(d, *keys):
    """安全读取嵌套 dict。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# ============================================================ HTTP 层（限速 / 重试 / 缓存）
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "game-community-insight-agent/0.1 (research calibration)"})
_last_request_at = [0.0]


def _pace():
    wait = PACE_SECONDS - (time.time() - _last_request_at[0])
    if wait > 0:
        time.sleep(wait)


def _rate_limit_wait(resp) -> float:
    """429 时优先读 X-RateLimit-Reset（兼容绝对 epoch 与相对秒数两种形式）。"""
    h = resp.headers.get("X-RateLimit-Reset")
    if h:
        try:
            v = float(h)
            now = time.time()
            wait = (v - now) if v > 1e9 else v
            return min(max(wait, 1.0) + 1.0, 180.0)
        except ValueError:
            pass
    return 5.0


def api_get(path: str, params: dict, cache_tag: str | None = None, retries: int = 5) -> dict:
    """带限速、429/422 处理与磁盘缓存的 GET。失败抛 ApiError。"""
    cache_file = None
    if cache_tag:
        key = hashlib.sha1(
            json.dumps({"p": path, "q": params}, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        cache_file = CACHE_DIR / f"{cache_tag}_{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    backoff = 3.0
    last_err = "?"
    for _attempt in range(retries):
        _pace()
        try:
            r = SESSION.get(BASE + path, params=params, timeout=180)
            _last_request_at[0] = time.time()
        except requests.RequestException as e:
            last_err = f"网络异常 {e}"
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        if r.status_code == 200:
            data = r.json()
            if cache_file is not None:
                cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        if r.status_code == 429:
            wait = _rate_limit_wait(r)
            print(f"    [429] 触发限速，等待 {wait:.0f}s 重试 ...")
            time.sleep(wait)
            continue
        if r.status_code in (422, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code} {r.text[:120]}"
            time.sleep(backoff)
            backoff = min(backoff * 2, 90)
            continue
        raise ApiError(f"GET {path} -> HTTP {r.status_code}: {r.text[:300]}")
    raise ApiError(f"GET {path} 重试 {retries} 次仍失败：{last_err}")


# ============================================================ 断点续传
class Progress:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, **fields):
        st = self.data.setdefault(key, {})
        st.update(fields)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


# ============================================================ 分页游标（核心逻辑）
def paginate(fetch_page, seen: set, start_cursor: int, end_epoch: int, on_page) -> tuple[int, int]:
    """通用分页循环。

    after 严格排除边界（>）→ 游标 = last_ts − 1 回看，用 id 去重处理重叠；
    整页无新行（说明同一秒内容超过一页容量）→ 跳过该秒剩余（skipped 计数），保证终止。
    返回 (本次新增行数, 同秒跳过事件数)。
    """
    cursor = start_cursor
    added = 0
    skipped = 0
    while cursor < end_epoch:
        rows = fetch_page(cursor, end_epoch)
        if not rows:
            break
        fresh = [r for r in rows if r["id"] not in seen]
        for r in fresh:
            seen.add(r["id"])
        last = ts_of(rows[-1])
        if fresh:
            next_cursor = max(cursor + 1, last - 1)
            added += len(fresh)
        else:
            next_cursor = max(cursor + 1, last)
            skipped += 1
        on_page(fresh, next_cursor)
        cursor = next_cursor
    return added, skipped


def cmd_selftest():
    """分页游标自检（合成数据，无网络）：验证无缺口、同秒聚集不丢尾、循环必终止。"""
    print("=== 分页游标自检（合成数据，无网络）===")

    def make_series(n: int, cluster: tuple | None = None):
        items = []
        ts = 1_000_000
        for i in range(n):
            if cluster and i == cluster[0]:
                for j in range(cluster[1]):
                    items.append({"id": f"c{j:04d}", "created_utc": ts})
                ts += 2
                continue
            items.append({"id": f"i{i:06d}", "created_utc": ts})
            ts += 2
        items.sort(key=lambda r: (r["created_utc"], r["id"]))
        return items

    def run(items, page_size):
        seen = set()

        def fetch(after, before):
            return [r for r in items if after < r["created_utc"] < before][:page_size]

        return paginate(fetch, seen, items[0]["created_utc"] - 1, items[-1]["created_utc"] + 10, lambda f, c: None)

    ok = True

    items = make_series(5000)
    added, skipped = run(items, 120)
    good = added == len(items)
    ok &= good
    print(f"  场景A 无同秒聚集 5000 行：collected={added} expected={len(items)} "
          f"skipped_events={skipped} -> {'PASS' if good else 'FAIL'}")

    items = make_series(3000, cluster=(1500, 250))
    added, skipped = run(items, 120)
    lost = len(items) - added
    good = lost == 130 and skipped >= 1
    ok &= good
    print(f"  场景B 同秒聚集 250 行（>单页 120）：collected={added} lost={lost} "
          f"skipped_events={skipped} -> {'PASS' if good else 'FAIL'}"
          "（预期：仅该秒超出单页的 130 行被跳过，且循环终止、不产生缺口）")

    print("=== 自检" + ("通过" if ok else "未通过") + " ===")
    if not ok:
        sys.exit(1)


# ============================================================ 采集
def collect_search(cfg: dict, prog: Progress, sub: str, kind: str, day: str):
    raw = cfg["raw_dir"]
    task = f"{kind}:{sub}:{day}"
    out_path = raw / f"{kind}_{sub}_{day}.jsonl"
    st = prog.get(task) or {}
    if st.get("done"):
        print(f"  [skip] {task} 已完成（{st.get('count', 0)} 行）")
        return
    seen = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    seen.add(json.loads(line)["id"])
    cursor = int(st.get("cursor", day_start_epoch(day) - 1))
    if seen:
        print(f"  [resume] {task}：已有 {len(seen)} 行，从 cursor={cursor} 续采")
    t0 = time.time()

    def fetch_page(after: int, before: int):
        data = api_get(
            f"/api/{kind}/search",
            {
                "subreddit": sub,
                "after": str(after),
                "before": str(before),
                "sort": "asc",
                "limit": "auto",
                "fields": FIELDS[kind],
            },
        )
        return data.get("data") or []

    with out_path.open("a", encoding="utf-8") as fh:

        def on_page(fresh, next_cursor):
            if fresh:
                fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fresh))
                fh.flush()
            prog.set(task, cursor=next_cursor, count=len(seen), done=False)
            print(f"    [page] {task} +{len(fresh)} (cum {len(seen)}) t={time.time() - t0:.0f}s")

        added, skipped = paginate(fetch_page, seen, cursor, day_end_epoch(day), on_page)

    prog.set(task, cursor=day_end_epoch(day), count=len(seen), skipped_same_ts=skipped, done=True)
    print(f"  [done] {task}：共 {len(seen)} 行（本次新增 {added}，同秒跳过事件 {skipped}）"
          f"耗时 {time.time() - t0:.0f}s")


def collect_time_series(cfg: dict):
    raw = cfg["raw_dir"]
    d0 = cfg["days"][0]
    d1 = (date.fromisoformat(cfg["days"][-1]) + timedelta(days=1)).isoformat()
    for sub in CORE_SUBS:
        for kind in ("posts", "comments"):
            data = api_get(
                "/api/time_series",
                {"key": f"r/{sub}/{kind}/count", "precision": "day", "after": d0, "before": d1},
                cache_tag=f"ts_{kind}_{sub}_{d0}_{d1}",
            )
            rows = data.get("data") or []
            (raw / f"time_series_{kind}_{sub}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"  [ts] {kind:8s} {sub:14s}: {len(rows)} 个日桶")
        data = api_get(
            "/api/time_series",
            {"key": f"r/{sub}/subscribers", "precision": "month", "after": "2023-01-01", "before": "2023-03-31"},
            cache_tag=f"subs_{sub}",
        )
        rows = data.get("data") or []
        (raw / f"subscribers_{sub}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"  [subs] {sub:14s}: {rows[-1] if rows else 'n/a'}")


def collect_retention(cfg: dict, failures: list):
    raw = cfg["raw_dir"]
    for sub in RETENTION_SUBS:
        f = raw / f"retention_{sub}.json"
        existing = {}
        if f.exists():
            try:
                for w in json.loads(f.read_text(encoding="utf-8")).get("weeks", []):
                    existing[w.get("start")] = w
            except Exception:
                existing = {}
        weeks = []
        for i in range(cfg["n_retention_weeks"]):
            w0 = RETENTION_W0 + timedelta(days=7 * i)
            key = w0.isoformat()
            if existing.get(key, {}).get("rows"):
                weeks.append(existing[key])
                continue
            try:
                data = api_get(
                    "/api/comments/search/aggregate",
                    {
                        "aggregate": "author",
                        "subreddit": sub,
                        "after": key,
                        "before": (w0 + timedelta(days=7)).isoformat(),
                        "min_count": 1,
                        "limit": "",
                    },
                    cache_tag=f"ret_{sub}_{key}",
                    retries=3,
                )
                rows = data.get("data") or []
                weeks.append({"start": key, "rows": rows})
                print(f"  [retention] {sub} {key}：{len(rows)} 位作者")
            except ApiError as e:
                weeks.append({"start": key, "error": str(e)})
                failures.append(f"retention:{sub}:{key} -> {e}")
                print(f"  [FAIL] retention {sub} {key}：{e}")
            f.write_text(json.dumps({"sub": sub, "weeks": weeks}, ensure_ascii=False), encoding="utf-8")
        f.write_text(json.dumps({"sub": sub, "weeks": weeks}, ensure_ascii=False), encoding="utf-8")


def stratified_systematic(counts: dict, n_total: int) -> list:
    """按活跃度分层（1 条 / 2-4 条 / 5+ 条）+ 层内等间隔系统抽样（确定性可复现）。"""
    strata = [("1条", lambda c: c == 1), ("2-4条", lambda c: 2 <= c <= 4), ("5+条", lambda c: c >= 5)]
    buckets = []
    for name, pred in strata:
        buckets.append((name, sorted(a for a, c in counts.items() if pred(c))))
    total = sum(len(b) for _, b in buckets)
    if total == 0 or n_total <= 0:
        return []
    alloc = []
    for _, b in buckets:
        k = round(n_total * len(b) / total) if b else 0
        alloc.append(min(len(b), max(1, k)) if b else 0)
    diff = n_total - sum(alloc)
    order = sorted(range(len(buckets)), key=lambda i: -len(buckets[i][1]))
    guard = 0
    while diff != 0 and guard < 1000:
        guard += 1
        for i in order:
            if diff == 0:
                break
            _, b = buckets[i]
            if diff > 0 and alloc[i] < len(b):
                alloc[i] += 1
                diff -= 1
            elif diff < 0 and alloc[i] > 1:
                alloc[i] -= 1
                diff += 1
    picks = []
    for (name, b), k in zip(buckets, alloc):
        if not b or k <= 0:
            continue
        step = len(b) / k
        idx = sorted({min(len(b) - 1, int((i + 0.5) * step)) for i in range(k)})
        picks.extend((b[i], name) for i in idx)
    return sorted(picks)


def collect_tenure(cfg: dict, failures: list):
    raw = cfg["raw_dir"]
    counts = Counter()
    for day in cfg["days"]:
        fp = raw / f"comments_gaming_{day}.jsonl"
        if not fp.exists():
            continue
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if is_human(r.get("author")):
                    counts[r["author"]] += 1
    if not counts:
        failures.append("tenure：无 gaming 评论数据，跳过抽样")
        print("  [FAIL] tenure：无 gaming 评论数据，跳过抽样")
        return
    sample = stratified_systematic(counts, cfg["tenure_n"])
    (raw / "tenure_sample.json").write_text(
        json.dumps(
            [{"author": a, "stratum": s, "week_comments": counts[a]} for a, s in sample],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"  [tenure] 分层抽样 {len(sample)} 人："
          + ", ".join(f"{a}({s})" for a, s in sample[:10]) + (" ..." if len(sample) > 10 else ""))
    for a, _s in sample:
        fp = raw / f"tenure_{safe_name(a)}.json"
        if fp.exists():
            continue
        try:
            data = api_get("/api/users/search", {"author": a, "limit": 1}, cache_tag=f"user_{a}", retries=3)
            row = (data.get("data") or [None])[0]
            fp.write_text(json.dumps({"author": a, "row": row}, ensure_ascii=False), encoding="utf-8")
        except ApiError as e:
            fp.write_text(json.dumps({"author": a, "error": str(e)}), encoding="utf-8")
            failures.append(f"tenure:{a} -> {e}")
            print(f"  [FAIL] tenure {a}：{e}")


def cmd_collect(cfg: dict):
    raw = cfg["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prog = Progress(raw / "progress.json")
    failures = []
    t0 = time.time()
    print(f"=== 采集开始（{'pilot' if cfg['pilot'] else '全量'}，输出目录 {raw}）===")

    for day in cfg["days"]:
        for sub in CORE_SUBS:
            for kind in ("comments", "posts"):
                try:
                    collect_search(cfg, prog, sub, kind, day)
                except ApiError as e:
                    failures.append(f"{kind}:{sub}:{day} -> {e}")
                    print(f"  [FAIL] {kind}:{sub}:{day}：{e}")

    try:
        collect_time_series(cfg)
    except ApiError as e:
        failures.append(f"time_series -> {e}")
        print(f"  [FAIL] time_series：{e}")

    collect_retention(cfg, failures)
    collect_tenure(cfg, failures)

    cov = analyze_coverage(cfg)
    print("=== 覆盖校验（collected=jsonl 行数 / series=time_series 日桶合计）===")
    for kind in ("comments", "posts"):
        for sub, v in cov[kind].items():
            ratio = "n/a" if v["ratio"] is None else f"{v['ratio']:.3f}"
            print(f"  {kind:8s} {sub:14s} collected={v['collected']:>7d} series={v['series_sum']:>9.0f} "
                  f"ratio={ratio} pass={v['pass']}")
    print(f"=== 采集结束，耗时 {(time.time() - t0) / 60:.1f} min，失败任务 {len(failures)} 个 ===")
    if failures:
        for f in failures:
            print("  FAILED:", f)
        sys.exit(1)


# ============================================================ 分析
def load_kind(raw: Path, kind: str, sub: str, days: list) -> tuple[list, dict]:
    rows = []
    tasks = {}
    for day in days:
        p = raw / f"{kind}_{sub}_{day}.jsonl"
        n = 0
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
                    n += 1
        tasks[day] = n
    dedup = {}
    for r in rows:
        dedup.setdefault(r["id"], r)
    return list(dedup.values()), {"tasks": tasks, "raw": len(rows), "dupes": len(rows) - len(dedup)}


def gini(arr) -> float | None:
    x = np.sort(np.asarray(arr, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return None
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def activity_stats(counts) -> dict:
    vals = [int(c) for c in counts if c > 0]
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=float)
    desc = np.sort(arr)[::-1]

    def top_share(frac: float) -> float:
        k = max(1, int(len(desc) * frac))
        return float(desc[:k].sum() / desc.sum())

    return {
        "n": len(vals),
        "total": int(arr.sum()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "share_eq1": float((arr == 1).mean()),
        "top1pct_share": top_share(0.01),
        "top10pct_share": top_share(0.10),
        "gini": gini(arr),
    }


def score_stats(scores) -> dict:
    a = np.asarray([int(s or 0) for s in scores], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "share_le0": float((a <= 0).mean()),
        "share_ge10": float((a >= 10).mean()),
        "share_ge100": float((a >= 100).mean()),
    }


def read_subscribers(raw: Path, sub: str):
    p = raw / f"subscribers_{sub}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not rows:
        return None
    v = rows[-1].get("value", rows[-1].get("count"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyze_sub(sub: str, comments: list, c_meta: dict, posts: list, p_meta: dict, cfg: dict) -> dict:
    days = cfg["days"]
    out = {"rows": {"comments": c_meta, "posts": p_meta}}
    c_h = [r for r in comments if is_human(r.get("author"))]
    p_h = [r for r in posts if is_human(r.get("author"))]

    def hyg(rows, hrows):
        total = len(rows)
        dele = sum(1 for r in rows if r.get("author") == "[deleted]")
        return {
            "total": total,
            "human": len(hrows),
            "deleted_share": (dele / total) if total else None,
            "filtered_share": ((total - len(hrows)) / total) if total else None,
        }

    out["hygiene"] = {"comments": hyg(comments, c_h), "posts": hyg(posts, p_h)}

    c_ud = Counter((r["author"], utc_dt(ts_of(r)).date().isoformat()) for r in c_h)
    p_ud = Counter((r["author"], utc_dt(ts_of(r)).date().isoformat()) for r in p_h)
    c_users = Counter(r["author"] for r in c_h)
    p_users = Counter(r["author"] for r in p_h)
    out["comment_user_day"] = activity_stats(c_ud.values())
    out["comment_user_week"] = activity_stats(c_users.values())
    out["post_user_day"] = activity_stats(p_ud.values())
    out["post_user_week"] = activity_stats(p_users.values())
    out["commenters"] = len(c_users)
    out["posters"] = len(p_users)
    out["commenter_poster_ratio"] = (len(c_users) / len(p_users)) if p_users else None

    per_day = defaultdict(set)
    for r in c_h:
        per_day[utc_dt(ts_of(r)).date().isoformat()].add(r["author"])
    daily = [len(per_day.get(d, ())) for d in days]
    out["dau_commenters_per_day"] = daily
    out["dau_commenters_avg"] = (sum(daily) / len(daily)) if daily else None
    subs_n = read_subscribers(cfg["raw_dir"], sub)
    out["subscribers"] = subs_n
    out["dau_over_subscribers"] = (
        (out["dau_commenters_avg"] / subs_n) if (subs_n and out["dau_commenters_avg"] is not None) else None
    )

    out["comment_score"] = score_stats([r.get("score") for r in c_h])
    out["post_score"] = score_stats([r.get("score") for r in p_h])

    nc = np.asarray([int(r.get("num_comments") or 0) for r in p_h], dtype=float)
    out["comments_per_post"] = (
        {
            "p50": float(np.percentile(nc, 50)),
            "p90": float(np.percentile(nc, 90)),
            "mean": float(nc.mean()),
            "zero_share": float((nc == 0).mean()),
        }
        if nc.size
        else {"n": 0}
    )
    out["comment_volume_per_post"] = (len(c_h) / len(p_h)) if p_h else None

    parents = [str(r.get("parent_id") or "") for r in c_h]
    known = [v for v in parents if v.startswith(("t3_", "t1_"))]
    top_level = sum(1 for v in known if v.startswith("t3_"))
    out["top_level_comment_share"] = (top_level / len(known)) if known else None

    hours = Counter()
    wd = Counter()
    for r in c_h:
        dtv = utc_dt(ts_of(r))
        hours[dtv.hour] += 1
        wd[dtv.weekday()] += 1
    hv = [hours.get(h, 0) for h in range(24)]
    mean_h = (sum(hv) / 24) if sum(hv) else 0
    dows = [date.fromisoformat(d).weekday() for d in days]
    wk_cnt = {i: dows.count(i) for i in range(7)}
    wknd_n = wk_cnt[5] + wk_cnt[6]
    wkday_n = sum(wk_cnt[i] for i in range(5))
    wknd = wd.get(5, 0) + wd.get(6, 0)
    wkday = sum(wd.get(i, 0) for i in range(5))
    out["rhythm"] = {
        "hours_utc": hv,
        "peak_hour_utc": (max(range(24), key=lambda h: hv[h]) if sum(hv) else None),
        "peak_over_mean": (max(hv) / mean_h) if mean_h else None,
        "weekday_counts": [wd.get(i, 0) for i in range(7)],
        "weekend_over_weekday": ((wknd / wknd_n) / (wkday / wkday_n)) if (wknd_n and wkday_n and wkday) else None,
    }
    out["daily_counts"] = {"comments": c_meta["tasks"], "posts": p_meta["tasks"]}
    return out


def norm_agg_row(row):
    """归一化 author 聚合行（实测形状 {"key","count"}，兼容其他返回格式）。"""
    if isinstance(row, dict):
        a = row.get("author") or row.get("key") or row.get("_id") or row.get("name") or row.get("id")
        if a is None:
            if len(row) == 1:
                k, v = next(iter(row.items()))
                try:
                    return str(k), int(float(v))
                except (TypeError, ValueError):
                    return None
            return None
        c = row.get("count", row.get("value", row.get("n", 1)))
        try:
            return str(a), int(float(c))
        except (TypeError, ValueError):
            return str(a), 1
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        try:
            return str(row[0]), int(float(row[1]))
        except (TypeError, ValueError):
            return None
    return None


def analyze_coverage(cfg: dict) -> dict:
    raw = cfg["raw_dir"]
    out = {
        "note": "series 为 time_series 日桶合计（桶标签有 −1h 偏移）；collected 为 jsonl 行数"
                "（含日界重复行）；容差 ±10%",
        "comments": {},
        "posts": {},
    }
    for kind in ("comments", "posts"):
        for sub in CORE_SUBS:
            f = raw / f"time_series_{kind}_{sub}.json"
            series = []
            if f.exists():
                for r in json.loads(f.read_text(encoding="utf-8")):
                    v = r.get("value", r.get("count"))
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = 0.0
                    series.append({"label": r.get("date") or r.get("created_utc") or "?", "value": v})
            ssum = sum(x["value"] for x in series)
            csum = sum(line_count(raw / f"{kind}_{sub}_{d}.jsonl") for d in cfg["days"])
            ratio = (csum / ssum) if ssum else None
            out[kind][sub] = {
                "collected": csum,
                "series_sum": ssum,
                "ratio": ratio,
                "pass": bool(ratio is not None and 0.9 <= ratio <= 1.1),
                "series": series,
            }
    return out


def analyze_retention(cfg: dict) -> dict:
    raw = cfg["raw_dir"]
    out = {}
    for sub in RETENTION_SUBS:
        f = raw / f"retention_{sub}.json"
        if not f.exists():
            out[sub] = {"error": "未采集"}
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        weeks = []
        errs = []
        for w in doc.get("weeks", []):
            if w.get("rows") is None and w.get("error"):
                errs.append(w)
                weeks.append(None)
                continue
            m = {}
            for row in w.get("rows") or []:
                nr = norm_agg_row(row)
                if not nr:
                    continue
                a, c = nr
                if is_human(a):
                    m[a] = c
            weeks.append({"start": w.get("start"), "authors": m})
        trans = []
        pool = {"1条": [0, 0], "2-4条": [0, 0], "5+条": [0, 0]}
        for i in range(len(weeks) - 1):
            w0, w1 = weeks[i], weeks[i + 1]
            if not w0 or not w1:
                continue
            nxt = set(w1["authors"])
            cohort = w0["authors"]
            retained = sum(1 for a in cohort if a in nxt)
            trans.append({
                "w0": w0["start"],
                "cohort": len(cohort),
                "retained": retained,
                "rate": (retained / len(cohort)) if cohort else None,
            })
            for a, c in cohort.items():
                k = "1条" if c == 1 else ("2-4条" if c <= 4 else "5+条")
                pool[k][0] += 1
                pool[k][1] += 1 if a in nxt else 0
        coh = sum(t["cohort"] for t in trans)
        ret = sum(t["retained"] for t in trans)
        out[sub] = {
            "transitions": trans,
            "cohort_total": coh,
            "retained_total": ret,
            "overall_rate": (ret / coh) if coh else None,
            "by_intensity": {
                k: {"cohort": v[0], "retained": v[1], "rate": (v[1] / v[0]) if v[0] else None}
                for k, v in pool.items()
            },
            "errors": [{"start": e.get("start"), "error": e.get("error")} for e in errs],
        }
    return out


def analyze_tenure(cfg: dict, gaming_comments: list) -> dict:
    raw = cfg["raw_dir"]
    out = {
        "n_target": cfg["tenure_n"],
        "n_files": 0,
        "n_ok": 0,
        "strata": {},
        "delay_days": {},
        "lifetime_comments": {},
        "lifetime_posts": {},
        "samples": [],
        "errors": [],
    }
    first = {}
    for r in gaming_comments:
        a = r.get("author")
        if not is_human(a):
            continue
        t = ts_of(r)
        if a not in first or t < first[a]:
            first[a] = t
    sample_doc = []
    sp = raw / "tenure_sample.json"
    if sp.exists():
        sample_doc = json.loads(sp.read_text(encoding="utf-8"))
    delays, lc, lp = [], [], []
    for item in sample_doc:
        a = item["author"]
        st = item.get("stratum", "?")
        out["strata"][st] = out["strata"].get(st, 0) + 1
        fp = raw / f"tenure_{safe_name(a)}.json"
        if not fp.exists():
            out["errors"].append({"author": a, "error": "未采集"})
            continue
        doc = json.loads(fp.read_text(encoding="utf-8"))
        out["n_files"] += 1
        if doc.get("error"):
            out["errors"].append({"author": a, "error": doc["error"]})
            continue
        meta = (doc.get("row") or {}).get("_meta") or {}
        e1 = _parse_ts(meta.get("earliest_comment_at")) or _parse_ts(meta.get("earliest_post_at"))
        if e1 is None or a not in first:
            out["errors"].append({"author": a, "error": "缺最早活动时间或首条时间"})
            continue
        d = max(0.0, (first[a] - e1) / 86400.0)
        delays.append(d)
        ncm = meta.get("num_comments")
        npst = meta.get("num_posts")
        if isinstance(ncm, (int, float)):
            lc.append(float(ncm))
        if isinstance(npst, (int, float)):
            lp.append(float(npst))
        out["n_ok"] += 1
        if len(out["samples"]) < 15:
            out["samples"].append({
                "author": a,
                "stratum": st,
                "week_comments": item.get("week_comments"),
                "delay_days": round(d, 1),
                "lifetime_comments": ncm,
                "lifetime_posts": npst,
                "earliest_activity": meta.get("earliest_comment_at"),
            })
    if delays:
        da = np.asarray(delays)
        out["delay_days"] = {
            "n": len(delays),
            "p50": float(np.percentile(da, 50)),
            "p90": float(np.percentile(da, 90)),
            "p99": float(np.percentile(da, 99)),
            "share_le7d": float((da <= 7).mean()),
            "share_le30d": float((da <= 30).mean()),
        }
    if lc:
        la = np.asarray(lc)
        out["lifetime_comments"] = {
            "p50": float(np.percentile(la, 50)),
            "p90": float(np.percentile(la, 90)),
            "p99": float(np.percentile(la, 99)),
        }
    if lp:
        pa = np.asarray(lp)
        out["lifetime_posts"] = {"p50": float(np.percentile(pa, 50)), "p90": float(np.percentile(pa, 90))}
    return out


# ============================================================ 参数对照表 / 报告
def build_param_rows(stats: dict, cfg: dict) -> list:
    g = stats["subs"].get("gaming", {})
    gs = stats["subs"].get("Genshin_Impact", {})
    tg = stats["subs"].get("truegaming", {})
    ret = stats.get("retention", {})
    ten = stats.get("tenure", {})
    win = f"{cfg['days'][0]}~{cfg['days'][-1]}"
    src_g = f"Reddit校准 gaming@{win}"
    get = dig

    def cross(keys, f=lambda v: fmt(v)):
        parts = []
        for name, d in (("Gen", gs), ("truegaming", tg)):
            v = get(d, *keys)
            parts.append(f"{name} {f(v)}")
        return "对照：" + " / ".join(parts)

    rows = []

    def add(label, value, source, note=""):
        rows.append([label, value, source, note])

    add("用户日评论条数 p50", fmt(get(g, "comment_user_day", "p50")), src_g,
        "按 user-day（当日有评论的用户×天）" + "；" + cross(("comment_user_day", "p50")))
    add("用户日评论条数 p90", fmt(get(g, "comment_user_day", "p90")), src_g, cross(("comment_user_day", "p90")))
    add("用户日评论条数 p99", fmt(get(g, "comment_user_day", "p99")), src_g, cross(("comment_user_day", "p99")))
    add("用户日评论条数 max（观察极值）", fmt(get(g, "comment_user_day", "max")), src_g,
        "重度个体的日上限参考" + "；" + cross(("comment_user_day", "max")))
    add("用户日恰 1 条评论占比", pct(get(g, "comment_user_day", "share_eq1")), src_g,
        "偶发发言（潜水者冒泡）形态" + "；" + cross(("comment_user_day", "share_eq1"), pct))
    add("周内评论量 top1% 用户份额", pct(get(g, "comment_user_week", "top1pct_share")), src_g,
        "头部集中度" + "；" + cross(("comment_user_week", "top1pct_share"), pct))
    add("周内评论量 top10% 用户份额", pct(get(g, "comment_user_week", "top10pct_share")), src_g,
        cross(("comment_user_week", "top10pct_share"), pct))
    add("评论量 Gini", fmt(get(g, "comment_user_week", "gini")), src_g,
        "不均匀度（0=均匀，1=极端集中）" + "；" + cross(("comment_user_week", "gini")))
    add("用户日发帖条数 p50 / p90 / 恰 1 条占比",
        f"{fmt(get(g, 'post_user_day', 'p50'))} / {fmt(get(g, 'post_user_day', 'p90'))} / "
        f"{pct(get(g, 'post_user_day', 'share_eq1'))}", src_g, "发帖远低于评论频次")
    add("评论者 : 发帖者 比",
        f"{fmt(get(g, 'commenter_poster_ratio'))} : 1", src_g, cross(("commenter_poster_ratio",)))
    add("日活跃评论者 / 订阅数", pct(get(g, "dau_over_subscribers"), 2),
        f"{src_g}；订阅=2023-02 存量",
        "订阅含逐年膨胀与僵尸号，仅作量级参考" + "；" + cross(("dau_over_subscribers",), lambda v: pct(v, 2)))
    add("评论 score ≤0 占比", pct(get(g, "comment_score", "share_le0")), src_g,
        "score 为归档时点值（非发布后短期值）" + "；" + cross(("comment_score", "share_le0"), pct))
    add("评论 score ≥10 占比", pct(get(g, "comment_score", "share_ge10")), src_g,
        cross(("comment_score", "share_ge10"), pct))
    add("帖子 score ≤0 / ≥10 / ≥100 占比",
        f"{pct(get(g, 'post_score', 'share_le0'))} / {pct(get(g, 'post_score', 'share_ge10'))} / "
        f"{pct(get(g, 'post_score', 'share_ge100'))}", src_g, "爆款帖长尾")
    add("每帖评论数 p50", fmt(get(g, "comments_per_post", "p50")), src_g,
        cross(("comments_per_post", "p50")))
    add("每帖评论数 p90 / 0 评论帖占比",
        f"{fmt(get(g, 'comments_per_post', 'p90'))} / {pct(get(g, 'comments_per_post', 'zero_share'))}",
        src_g, "冷启动帖占比" + "；" + cross(("comments_per_post", "zero_share"), pct))
    add("顶层评论（直接回主帖）占比", pct(get(g, "top_level_comment_share")), src_g,
        "回复结构：直接回帖 vs 楼中楼" + "；" + cross(("top_level_comment_share",), pct))
    add("24h 评论节律峰值（UTC）/ 峰均比",
        f"{fmt(get(g, 'rhythm', 'peak_hour_utc'))}:00 / {fmt(get(g, 'rhythm', 'peak_over_mean'))}",
        src_g, "UTC 口径；美区社区峰值约在 UTC 20:00 前后")
    add("周末 / 工作日 日均评论量比", fmt(get(g, "rhythm", "weekend_over_weekday")), src_g,
        cross(("rhythm", "weekend_over_weekday")))
    ret_rate = (get(ret, "truegaming", "overall_rate") or 0) if get(ret, "truegaming", "overall_rate") else None
    ret_pg = get(ret, "patientgamers", "overall_rate")
    add("社区周留存 P(w+1|w)", pct(ret_rate), "Reddit校准 truegaming@2023-01-02~03-27",
        f"对照 patientgamers {pct(ret_pg)}；仅小版可算（gaming/Genshin 聚类聚合 API 超时）")
    add("周留存·分层（周 1 条 / 2-4 条 / 5+ 条）",
        " / ".join(pct(get(ret, "truegaming", "by_intensity", k, "rate")) for k in ("1条", "2-4条", "5+条")),
        "Reddit校准 truegaming@2023-01-02~03-27", "轻度用户流失更快（真实社区特征）")
    add("新账号占比（首条评论距其最早活动 ≤7 天）", pct(get(ten, "delay_days", "share_le7d")),
        f"Reddit校准 gaming@{win} 分层抽样 n={cfg['tenure_n']}",
        "起始活动延迟代理账号年龄（无账号创建日字段）")
    add("新账号占比（≤30 天）", pct(get(ten, "delay_days", "share_le30d")),
        f"Reddit校准 gaming@{win} 分层抽样 n={cfg['tenure_n']}", "")
    add("首次发言延迟 p50 / p90（天）",
        f"{fmt(get(ten, 'delay_days', 'p50'))} / {fmt(get(ten, 'delay_days', 'p90'))}",
        f"Reddit校准 gaming@{win} 分层抽样 n={cfg['tenure_n']}", "“潜水期”长度参考")
    add("账号 lifetime 评论量 p50 / p90",
        f"{fmt(get(ten, 'lifetime_comments', 'p50'))} / {fmt(get(ten, 'lifetime_comments', 'p90'))}",
        f"Reddit校准 gaming@{win} 分层抽样 n={cfg['tenure_n']}", "注册时长的存量代理")
    add("情感信号（负向率 / 争议度）", "不适用：不可观测", "设计取值",
        "Reddit 采集字段无情感标注；生成器按设计分布注入，不伪装为校准值")
    add("浏览深度 / 曝光 / 停留时长", "不适用：不可观测", "设计取值",
        "采集字段无浏览行为；生成器按设计取值，不伪装为校准值")
    return rows


def write_report(stats: dict, rows: list, cfg: dict):
    win = f"{cfg['days'][0]} ~ {cfg['days'][-1]}"
    get = dig
    L = []
    L.append(f"# Reddit 校准报告{'（PILOT 小样本，仅供链路验证，不可作为最终锚点）' if cfg['pilot'] else ''}")
    L.append("")
    L.append(f"- 生成时间（UTC）：{stats['generated_at']}")
    L.append("- 数据源：Arctic Shift API（Reddit 归档）；采集窗口 2023 Q1，全程 UTC")
    L.append(f"- 采集窗口：{win}（逐日全量）；留存窗口：2023-01-02 起 {cfg['n_retention_weeks']} 个周")
    L.append("- 用途：**仅作为合成数据生成器的参数锚点**；不代表任何对外结论，也不等同于论文口径引用")
    L.append("")
    L.append("## 一、采集口径与覆盖校验")
    L.append("")
    L.append("| 子版 | 类型 | collected（jsonl 行数） | time_series 日桶合计 | 比值 | 通过 |")
    L.append("| --- | --- | ---: | ---: | ---: | --- |")
    for kind in ("comments", "posts"):
        for sub in CORE_SUBS:
            v = stats["coverage"][kind][sub]
            ratio = "n/a" if v["ratio"] is None else f"{v['ratio']:.3f}"
            L.append(f"| {sub} | {kind} | {v['collected']:,} | {v['series_sum']:,.0f} | {ratio} | "
                     f"{'✅' if v['pass'] else '❌'} |")
    L.append("")
    L.append("数据卫生（剔除 `[deleted]` / AutoModerator / \\*modteam\\* / \\*bot 后缀）：")
    L.append("")
    L.append("| 子版 | 评论剔除占比 | 帖子剔除占比 | 评论者 | 发帖者 | 评论者:发帖者 |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for sub in CORE_SUBS:
        d = stats["subs"][sub]
        L.append(f"| {sub} | {pct(get(d, 'hygiene', 'comments', 'filtered_share'))} | "
                 f"{pct(get(d, 'hygiene', 'posts', 'filtered_share'))} | {d['commenters']:,} | {d['posters']:,} | "
                 f"{fmt(d['commenter_poster_ratio'])} : 1 |")
    L.append("")
    L.append("## 二、核心指标（逐子版）")
    L.append("")
    L.append("| 指标 | gaming | Genshin_Impact | truegaming |")
    L.append("| --- | ---: | ---: | ---: |")

    def line(label, fn):
        vals = [fn(stats["subs"][s]) for s in CORE_SUBS]
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    line("日评论量（均值）", lambda d: fmt(sum(d["daily_counts"]["comments"].values()) / max(len(d["daily_counts"]["comments"]), 1)))
    line("日均活跃评论者", lambda d: fmt(d["dau_commenters_avg"]))
    line("订阅数（2023-02）", lambda d: fmt(d["subscribers"]))
    line("DAU 评论者/订阅", lambda d: pct(d["dau_over_subscribers"], 2))
    line("用户日评论 p50/p90/p99", lambda d: f"{fmt(get(d, 'comment_user_day', 'p50'))} / {fmt(get(d, 'comment_user_day', 'p90'))} / {fmt(get(d, 'comment_user_day', 'p99'))}")
    line("top1% 用户评论份额", lambda d: pct(get(d, "comment_user_week", "top1pct_share")))
    line("评论 Gini", lambda d: fmt(get(d, "comment_user_week", "gini")))
    line("评论 score ≤0 / ≥10 / ≥100", lambda d: f"{pct(get(d, 'comment_score', 'share_le0'))} / {pct(get(d, 'comment_score', 'share_ge10'))} / {pct(get(d, 'comment_score', 'share_ge100'))}")
    line("帖子 score ≤0 / ≥10 / ≥100", lambda d: f"{pct(get(d, 'post_score', 'share_le0'))} / {pct(get(d, 'post_score', 'share_ge10'))} / {pct(get(d, 'post_score', 'share_ge100'))}")
    line("每帖评论 p50/p90；0 评论占比", lambda d: f"{fmt(get(d, 'comments_per_post', 'p50'))} / {fmt(get(d, 'comments_per_post', 'p90'))}；{pct(get(d, 'comments_per_post', 'zero_share'))}")
    line("顶层评论占比", lambda d: pct(get(d, "top_level_comment_share")))
    line("峰值时段（UTC）/ 峰均比", lambda d: f"{fmt(get(d, 'rhythm', 'peak_hour_utc'))}:00 / {fmt(get(d, 'rhythm', 'peak_over_mean'))}")
    line("周末/工作日日均评论比", lambda d: fmt(get(d, "rhythm", "weekend_over_weekday")))
    L.append("")
    L.append("## 三、周留存（P(w+1 | w)，按周评论活跃）")
    L.append("")
    L.append("| 子版 | 总留存 | 周 1 条 | 周 2-4 条 | 周 5+ 条 | 说明 |")
    L.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for sub in RETENTION_SUBS:
        d = stats["retention"].get(sub, {})
        if d.get("error"):
            L.append(f"| {sub} | n/a | n/a | n/a | n/a | {d['error']} |")
            continue
        bi = d.get("by_intensity", {})
        L.append(f"| {sub} | {pct(d.get('overall_rate'))} | {pct(get(bi, '1条', 'rate'))} | "
                 f"{pct(get(bi, '2-4条', 'rate'))} | {pct(get(bi, '5+条', 'rate'))} | cohort={d.get('cohort_total', 0):,} |")
    L.append("")
    L.append("## 四、Tenure / 账号年龄（gaming 采集周评论作者分层抽样）")
    L.append("")
    ten = stats.get("tenure", {})
    L.append(f"- 成功/目标：{ten.get('n_ok')} / {ten.get('n_target')}；分层：{ten.get('strata')}")
    L.append(f"- 首次发言延迟（距最早活动）p50/p90/p99：{fmt(get(ten, 'delay_days', 'p50'))} / "
             f"{fmt(get(ten, 'delay_days', 'p90'))} / {fmt(get(ten, 'delay_days', 'p99'))} 天")
    L.append(f"- 新账号占比：≤7 天 {pct(get(ten, 'delay_days', 'share_le7d'))}；"
             f"≤30 天 {pct(get(ten, 'delay_days', 'share_le30d'))}")
    L.append(f"- lifetime 评论量 p50/p90/p99：{fmt(get(ten, 'lifetime_comments', 'p50'))} / "
             f"{fmt(get(ten, 'lifetime_comments', 'p90'))} / {fmt(get(ten, 'lifetime_comments', 'p99'))}")
    L.append("")
    L.append("## 五、生成器参数锚点对照表")
    L.append("")
    L.append("| " + " | ".join(["生成器参数", "锚点取值", "来源", "备注"]) + " |")
    L.append("| --- | --- | --- | --- |")
    for r in rows:
        L.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    L.append("")
    L.append("## 六、口径与局限（重要）")
    L.append("")
    L.append("1. **订阅数膨胀**：subscribers 为历史存量（含多年累积与僵尸账号），DAU/订阅 仅作量级参考。")
    L.append("2. **单周采样**：采集仅覆盖 2023-02-20~26 一个自然周，存在版本活动/季节偏差。")
    L.append("3. **UTC 时区**：所有时间口径为 UTC；真实用户以美区为主，当地时区节律需自行换算。")
    L.append("4. **score 为归档时点值**：非发布后短期分数，分位数整体向上漂移。")
    L.append("5. **留存仅小版代理**：gaming/Genshin 的 author 聚合在 API 侧超时，留存与强度分桶来自 "
             "truegaming/patientgamers，可能高估小社区的粘性。")
    L.append("6. **同秒截断**：同一秒内容超过单页容量时该秒剩余行会被跳过（expected=0 事件；"
             "各任务 skipped_same_ts 见 progress.json）。")
    L.append("7. **过滤规则**：剔除 [deleted]/bot/modteam/AutoModerator，占比见第一节卫生表。")
    L.append("8. **边界桶 −1h 标签偏移**：time_series 日桶标签有偏移，覆盖校验按周求和、容差 ±10%。")
    L.append("9. Reddit 与 TapTap 型游戏社区在文化与产品形态上存在差异，锚点用于**量级与分布形状**，"
             "不宜逐值照搬。")
    L.append("")
    (OUT_DIR / f"reddit_report{cfg['suffix']}.md").write_text("\n".join(L), encoding="utf-8")


# ============================================================ 主流程
def cmd_analyze(cfg: dict):
    raw = cfg["raw_dir"]
    if not raw.exists():
        print(f"[ERROR] 未找到采集数据目录 {raw}，请先运行 collect")
        sys.exit(1)
    print(f"=== 分析开始（{'pilot' if cfg['pilot'] else '全量'}，读取 {raw}）===")
    stats = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pilot": cfg["pilot"],
        "config": {
            "days": cfg["days"],
            "core_subs": CORE_SUBS,
            "retention_subs": RETENTION_SUBS,
            "retention_weeks": cfg["n_retention_weeks"],
            "tenure_n": cfg["tenure_n"],
            "fields": FIELDS,
            "source": "Arctic Shift API (Reddit archive)",
        },
        "subs": {},
        "coverage": {},
        "retention": {},
        "tenure": {},
    }
    gaming_comments = []
    for sub in CORE_SUBS:
        c, c_meta = load_kind(raw, "comments", sub, cfg["days"])
        p, p_meta = load_kind(raw, "posts", sub, cfg["days"])
        if sub == "gaming":
            gaming_comments = c
        stats["subs"][sub] = analyze_sub(sub, c, c_meta, p, p_meta, cfg)
        print(f"  [load] {sub}: comments={c_meta['raw']:,}（去重 {c_meta['dupes']}） posts={p_meta['raw']:,}")
    stats["coverage"] = analyze_coverage(cfg)
    stats["retention"] = analyze_retention(cfg)
    stats["tenure"] = analyze_tenure(cfg, gaming_comments)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUT_DIR / f"reddit_stats{cfg['suffix']}.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = build_param_rows(stats, cfg)
    csv_path = OUT_DIR / f"reddit_params{cfg['suffix']}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["生成器参数", "锚点取值", "来源", "备注"])
        w.writerows(rows)

    write_report(stats, rows, cfg)
    print("")
    print("=== 参数锚点（摘要）===")
    for r in rows:
        print(f"  {r[0]}：{r[1]}   [{r[2]}]")
    print("")
    print(f"=== 产出：{stats_path.name} / {csv_path.name} / reddit_report{cfg['suffix']}.md ===")


def main():
    ap = argparse.ArgumentParser(description="Reddit 校准：游戏社区行为分布 → 生成器参数锚点")
    ap.add_argument("command", choices=["collect", "analyze", "all", "selftest"])
    ap.add_argument("--pilot", action="store_true", help="小样本模式（1 天 / tenure=8），产物带 _pilot 后缀")
    args = ap.parse_args()
    if args.command == "selftest":
        cmd_selftest()
        return
    cfg = build_config(args.pilot)
    if args.command in ("collect", "all"):
        cmd_collect(cfg)
    if args.command in ("analyze", "all"):
        cmd_analyze(cfg)


if __name__ == "__main__":
    main()
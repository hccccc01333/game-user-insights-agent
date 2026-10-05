#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S1 模拟器 · 运行入口：三臂配对实验 → 真值 CATE 表。

先说人话：
    这支脚本把模拟器跑成一桌"对照实验结果"：每个合成用户都完整经历
    control / rec / recall 三个臂（同一随机数串的配对轨迹），于是每个
    用户×臂都有一行真值——结构 CATE、个体 CATE、实现增量、净奖励。
    下游 Uplift（S2）用这些行训练与判分，Bandit（S3）直接对 reward 列
    做决策——因果靶子在数据层面就位，不需要真实世界先跑一遍。

运行（在仓库根目录）：
    python -m simulator.run_sim --users 500 --days 14 --seed 7
    python -m simulator.run_sim --no-fit          # 强制默认分布（离线可复现路径）

产出（默认 data/synthetic/，合成数据不入公开仓）：
    sim_users.csv      每个合成用户的初始状态（观测四件 + 潜在节奏）
    sim_effects.csv    ★ 用户 × 臂：tau_struct / tau_ind / increment / reward
    sim_daily.csv      天 × 臂 人口聚合（含未触达基线；用来看效应衰减）
    _manifest.json     版本 / 种子 / 参数快照 / 人口口径 / 输出指纹 / 双跑一致
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import params as P
from . import response
from . import reward as R
from .env import SimEnv
from .user_generator import DEFAULT_FEATURES, generate_users

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "synthetic"
TZ_CN = timezone(timedelta(hours=8))


# ── 核心：跑一遍完整模拟（纯内存，deterministic）─────────────

def build_tables(
    n_users: int,
    days: int,
    seed: int,
    window: int,
    features_path: Path | None,
    use_fit: bool = True,
    params: P.SimParams = P.SIM,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """跑一遍完整模拟，返回 (三张产出表, meta)。同参同种子必须逐字节一致。"""
    if n_users < 1:
        raise ValueError(f"用户数需 ≥ 1（当前 {n_users}）")
    if days < 1:
        raise ValueError(f"天数需 ≥ 1（当前 {days}）")
    if not (1 <= window <= days):
        raise ValueError(f"奖励窗口需满足 1 ≤ window ≤ days（当前 window={window}, days={days}）")

    pop = generate_users(n_users, seed, features_path if use_fit else None, params)
    env = SimEnv(pop.users, seed, params)

    user_rows: list[dict] = []
    effect_rows: list[dict] = []
    daily_sum = {arm: [0] * days for arm in params.arms}
    daily_void = [0] * days
    active_days = 0   # 条数采样器的校准自检（只在未触达轨迹上统计）
    one_days = 0

    for i, u in enumerate(pop.users):
        y_void = env.rollout(i, [], days)
        for d, v in enumerate(y_void):
            daily_void[d] += v
            daily_sum["control"][d] += v
            if v > 0:
                active_days += 1
                if v == 1:
                    one_days += 1
        y_void_win = sum(y_void[:window])
        user_rows.append({
            "uid": u.uid,
            "act_30d": u.act_30d,
            "silence_days": round(u.silence_days, 3),
            "interest_concentration": round(u.interest_concentration, 6),
            "tenure_days": round(u.tenure_days, 2),
            "a_daily": round(u.a_daily, 6),
            "y_void_window": y_void_win,
        })

        noises = env.noise_by_arm(i)
        for arm in params.arms:
            if arm == "control":
                y_arm, tau_s, tau_i = y_void, 0.0, 0.0
            else:
                y_arm = env.rollout(i, [(0, arm)], days, window)
                tau_s = response.structural_effect(u, arm, params)
                tau_i = response.individual_effect(u, arm, noises[arm], params)
                for d, v in enumerate(y_arm):
                    daily_sum[arm][d] += v
            inc = R.window_increment(y_arm, y_void, window)
            effect_rows.append({
                "uid": u.uid,
                "arm": arm,
                "tau_struct": round(tau_s, 6),
                "tau_ind": round(tau_i, 6),
                "cost": R.marginal_cost(arm, params),
                "y_void_window": y_void_win,
                "y_treated_window": sum(y_arm[:window]),
                "increment": inc,
                "reward": round(R.reward(inc, arm, params), 6),
            })

    users_df = pd.DataFrame(user_rows)
    effects_df = pd.DataFrame(effect_rows)
    daily_df = pd.DataFrame(
        [
            {
                "day": d,
                "arm": arm,
                "events_population": daily_sum[arm][d],
                "void_population": daily_void[d],
                "increment_population": daily_sum[arm][d] - daily_void[d],
            }
            for arm in params.arms
            for d in range(days)
        ]
    )

    summary = {}
    for arm in params.arms:
        sub = effects_df[effects_df["arm"] == arm]
        summary[arm] = {
            "label": P.ARM_LABELS[arm],
            "mean_tau_struct": round(float(sub["tau_struct"].mean()), 4),
            "mean_tau_ind": round(float(sub["tau_ind"].mean()), 4),
            "mean_increment": round(float(sub["increment"].mean()), 4),
            "mean_reward": round(float(sub["reward"].mean()), 4),
            "positive_reward_share": round(float((sub["reward"] > 0).mean()), 4),
        }

    meta = {
        "population": pop.meta,
        "effects_summary": summary,
        "daily_count_calibration": {
            "active_days": int(active_days),
            "share_1_given_active": round(one_days / active_days, 4) if active_days else None,
            "anchor_share_1": params.count_p1,
        },
    }
    return {"users": users_df, "effects": effects_df, "daily": daily_df}, meta


# ── 落盘 ────────────────────────────────────────────────────

def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rel(path: Path) -> str:
    """终端只展示相对路径（避免本地绝对路径进入任何可被复制出去的输出）。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="S1 模拟器：三臂配对实验 → 真值 CATE 表")
    ap.add_argument("--users", type=int, default=500, help="合成用户数（默认 500）")
    ap.add_argument("--days", type=int, default=14, help="模拟天数（默认 14，日步长）")
    ap.add_argument("--window", type=int, default=P.SIM.reward_window_days, help="奖励窗口天数（默认 7）")
    ap.add_argument("--seed", type=int, default=7, help="随机种子（默认 7）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 data/synthetic）")
    ap.add_argument("--features", default=str(DEFAULT_FEATURES), help="L2 特征表路径（fit 路径使用）")
    ap.add_argument("--no-fit", action="store_true", help="强制默认分布：不读特征表（离线路径）")
    ap.add_argument("--no-verify", action="store_true", help="跳过双跑一致性校验（大人口时可省一半耗时）")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = dict(
        n_users=args.users,
        days=args.days,
        seed=args.seed,
        window=args.window,
        features_path=Path(args.features),
        use_fit=not args.no_fit,
    )
    try:
        tables, meta = build_tables(**kwargs)
    except ValueError as exc:
        print(f"[stop] 参数无效：{exc}", file=sys.stderr)
        return 2

    determinism = {"checked": False}
    if not args.no_verify:
        tables_again, _ = build_tables(**kwargs)
        same = all(tables[k].equals(tables_again[k]) for k in tables)
        determinism = {"checked": True, "identical": bool(same)}
        if not same:
            print("[stop] 双跑不一致：存在非确定性来源，拒绝落盘", file=sys.stderr)
            return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "sim_users.csv": tables["users"],
        "sim_effects.csv": tables["effects"],
        "sim_daily.csv": tables["daily"],
    }
    for name, df in files.items():
        df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    manifest = {
        "sim_version": P.SIM_VERSION,
        "generated_at": datetime.now(TZ_CN).isoformat(timespec="seconds"),
        "seed": args.seed,
        "n_users": args.users,
        "days": args.days,
        "reward_window_days": args.window,
        "population": meta["population"],
        "effects_summary": meta["effects_summary"],
        "daily_count_calibration": meta["daily_count_calibration"],
        "params": P.SIM.snapshot(),
        "outputs": {name: {"rows": int(len(df)), "sha256": _sha16(out_dir / name)} for name, df in files.items()},
        "determinism": determinism,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 终端报告 ──
    pop_meta, realized = meta["population"], meta["population"]["realized"]
    src = pop_meta["path"]
    if src == "fit":
        src = f"fit ← {pop_meta['features_file']}（指纹 {pop_meta['features_fingerprint']}）"
    print(f"[info] 人口 {realized['n']} 人｜路径 {src}")
    print(f"       实收分布：act30>0 {realized['act30_gt0_share']:.1%}｜沉默中位 {realized['silence_median']} 天"
          f"｜集中度中位 {realized['concentration_median']}｜资历中位 {realized['tenure_median']} 天"
          f"｜节奏均值 {realized['a_daily_mean']}")
    print("[info] 效应汇总（真值，单位=事件）：")
    print(f"       {'arm':<8}{'mean_tau_struct':>16}{'mean_tau_ind':>14}{'mean_incr':>11}{'mean_reward':>13}{'reward>0':>10}")
    for arm in P.SIM.arms:
        s = meta["effects_summary"][arm]
        print(f"       {arm:<8}{s['mean_tau_struct']:>16.3f}{s['mean_tau_ind']:>14.3f}"
              f"{s['mean_increment']:>11.3f}{s['mean_reward']:>13.3f}{s['positive_reward_share']:>10.1%}")
    cal = meta["daily_count_calibration"]
    if cal["share_1_given_active"] is not None:
        print(f"[info] 条数采样器自检：活跃日恰 1 条 {cal['share_1_given_active']:.1%}（锚点 {cal['anchor_share_1']:.0%}）"
              f"｜活跃日 n={cal['active_days']}")
    print(f"[done] → {_rel(out_dir)}（sim_users {len(tables['users'])} 行 / sim_effects {len(tables['effects'])} 行"
          f" / sim_daily {len(tables['daily'])} 行）")
    if determinism["checked"]:
        print(f"[ok] 双跑一致性：{determinism['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
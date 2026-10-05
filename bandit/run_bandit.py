#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 Bandit · 运行入口：在模拟环境里学"每次触达选哪条臂"，并回答能学多快。

先说人话：
    S1 的模拟器给了每个用户 × 三条臂的净奖励真值；这支脚本让策略戴上
    "部分反馈"的镣铐上场：每人只到访一次，选完一条臂只看得到这一条臂的
    结果——LinUCB 与 Thompson 边决策边学，对照 random / 固定臂 / oracle
    上界，回答三个问题：
      · 在线跑一遍，学习策略比不学多赚多少？（学习曲线 + 累计遗憾）
      · 学到的知识本身值多少？（审计池冻结评测：不再探索，只考知识）
      · 离上帝视角（oracle）还有多远？（遗憾的剩余空间）

运行（在仓库根目录）：
    python -m bandit.run_bandit --users 6000 --seed 13
    python -m bandit.run_bandit --no-fit --policies linucb,thompson,random --no-verify

产出（默认 data/synthetic/bandit/，合成数据不进公开仓）：
    bandit_metrics.json   配置 / 世界口径 / 超参 / 每策略汇总（宽表 records）
    bandit_curve.csv      在线学习曲线逐人次日志（全策略拼接）
    bandit_summary.csv    每策略一行汇总（在线 + 审计）
    bandit_audit.csv      审计池冻结评测逐人次日志
    _manifest.json        版本 / 模拟参数快照 / 输出指纹 / 双跑一致
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from simulator import params as P
from simulator.run_sim import build_tables
from simulator.user_generator import DEFAULT_FEATURES

from . import BANDIT_VERSION
from .evaluate import run_audit, run_online, summary_table
from .policies import (
    DEFAULT_ALPHA, DEFAULT_RIDGE, DEFAULT_SIGMA, POLICIES,
    hyperparams_snapshot, make_policy,
)
from .protocol import ARMS, BASES, DEFAULT_BASIS, build_world

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "synthetic" / "bandit"
TZ_CN = timezone(timedelta(hours=8))


# ── 核心：跑一遍完整流程（纯内存，deterministic）────────────

def validate_config(
    *,
    n_users: int,
    days: int,
    window: int,
    audit: float,
    policies: tuple[str, ...],
    alpha: float,
    ridge: float,
    sigma: float,
) -> tuple[str, ...]:
    """入口参数校验（CLI 与测试共用）；返回按 POLICIES 规范序去重后的策略元组。"""
    if n_users < 200:
        raise ValueError(f"用户数需 ≥ 200（当前 {n_users}）：切分与学习都需要足够样本")
    if days < 1:
        raise ValueError(f"天数需 ≥ 1（当前 {days}）")
    if not (1 <= window <= days):
        raise ValueError(f"奖励窗口需满足 1 ≤ window ≤ days（当前 window={window}, days={days}）")
    if not (0.0 < audit < 1.0):
        raise ValueError(f"审计比例需在 (0,1) 内（当前 {audit}）")
    if not policies:
        raise ValueError("policies 不能为空")
    bad = [p for p in policies if p not in POLICIES]
    if bad:
        raise ValueError(f"未知策略：{bad}（支持 {POLICIES}）")
    for name, value in (("alpha", alpha), ("ridge", ridge), ("sigma", sigma)):
        if not (value > 0):
            raise ValueError(f"{name} 需 > 0（当前 {value}）")
    return tuple(p for p in POLICIES if p in set(policies))


def run_all(
    *,
    n_users: int = 6000,
    days: int = 14,
    window: int = P.SIM.reward_window_days,
    seed: int = 13,
    audit: float = 0.3,
    policies: tuple[str, ...] = POLICIES,
    alpha: float = DEFAULT_ALPHA,
    ridge: float = DEFAULT_RIDGE,
    sigma: float = DEFAULT_SIGMA,
    basis: str = DEFAULT_BASIS,
    features_path: Path | None = None,
    use_fit: bool = True,
) -> dict:
    """S1 模拟 → 部分反馈世界 → 逐策略在线学习 + 审计冻结评测；同参同种子逐字节一致。"""
    if basis not in BASES:
        raise ValueError(f"未知上下文基：{basis!r}（支持 {BASES}）")
    policies = validate_config(
        n_users=n_users, days=days, window=window, audit=audit,
        policies=policies, alpha=alpha, ridge=ridge, sigma=sigma,
    )
    tables, sim_meta = build_tables(
        n_users, days, seed, window, features_path if use_fit else None, use_fit
    )
    built = build_world(tables, audit, seed, basis)
    world, online_idx, audit_idx = built["world"], built["online"], built["audit"]

    online_logs: dict[str, pd.DataFrame] = {}
    audit_logs: dict[str, pd.DataFrame] = {}
    for name in policies:
        policy = make_policy(
            name,
            context_dim=built["meta"]["context_width"],
            seed=seed,
            oracle_arm=world.oracle_arm,
            alpha=alpha,
            ridge=ridge,
            sigma=sigma,
        )
        online_logs[name] = run_online(world, online_idx, policy)
        audit_logs[name] = run_audit(world, audit_idx, policy)

    frames = {
        "curve": pd.concat(online_logs.values(), ignore_index=True),
        "summary": summary_table(online_logs, audit_logs),
        "audit": pd.concat(audit_logs.values(), ignore_index=True),
    }
    config = {
        "n_users": int(n_users),
        "days": int(days),
        "window": int(window),
        "seed": int(seed),
        "audit": float(audit),
        "policies": list(policies),
        "alpha": float(alpha),
        "ridge": float(ridge),
        "sigma": float(sigma),
        "basis": basis,
        "use_fit": bool(use_fit),
        "features_file": str(features_path) if (use_fit and features_path) else None,
    }
    return {
        "tables": tables,
        "sim_meta": sim_meta,
        "world_meta": built["meta"],
        "config": config,
        "frames": frames,
    }


def run_is_deterministic(first: dict, second: dict) -> bool:
    """双跑一致性：三个产出帧都必须逐值相等（模拟表 + 世界构造 + 策略学习全链）。"""
    if set(first["frames"]) != set(second["frames"]):
        return False
    return all(first["frames"][k].equals(second["frames"][k]) for k in first["frames"])


# ── 汇总与落盘 ──────────────────────────────────────────────

def _now() -> str:
    return datetime.now(TZ_CN).isoformat(timespec="seconds")


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rel(path: Path) -> str:
    """终端只展示相对路径（避免本地绝对路径进入任何可被复制出去的输出）。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def build_metrics(run: dict) -> dict:
    """bandit_metrics.json：主结果文件（配置 + 世界口径 + 超参 + 每策略汇总）。"""
    cfg, sim_meta, wm = run["config"], run["sim_meta"], run["world_meta"]
    summary: pd.DataFrame = run["frames"]["summary"]
    return {
        "bandit_version": BANDIT_VERSION,
        "generated_at": _now(),
        "config": {**cfg, "population_path": sim_meta["population"]["path"]},
        "sim": {
            "sim_version": P.SIM_VERSION,
            "effects_summary": sim_meta["effects_summary"],
        },
        "world": wm,
        "hyperparams": hyperparams_snapshot(
            tuple(cfg["policies"]), alpha=cfg["alpha"], ridge=cfg["ridge"], sigma=cfg["sigma"]
        ),
        "observations": {
            "online_steps_per_policy": wm["n_online"],
            "audit_steps_per_policy": wm["n_audit"],
            "total_observations": (wm["n_online"] + wm["n_audit"]) * len(cfg["policies"]),
        },
        "summary": summary.to_dict("records"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="S3 Bandit：模拟环境里学触达臂选择，并回答能学多快")
    ap.add_argument("--users", type=int, default=6000, help="合成用户数（默认 6000）")
    ap.add_argument("--days", type=int, default=14, help="模拟天数（默认 14）")
    ap.add_argument("--window", type=int, default=P.SIM.reward_window_days, help="奖励窗口天数（默认 7）")
    ap.add_argument("--seed", type=int, default=13, help="随机种子（默认 13）")
    ap.add_argument("--audit", type=float, default=0.3, help="审计池比例（默认 0.3）")
    ap.add_argument("--policies", default=",".join(POLICIES), help="策略，逗号分隔（默认全部）")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="LinUCB 探索强度（默认 1.0）")
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE, help="岭回归正则（默认 1.0）")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Thompson 后验尺度（默认 2.0）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 data/synthetic/bandit）")
    ap.add_argument("--features", default=str(DEFAULT_FEATURES), help="L2 特征表路径（fit 路径使用）")
    ap.add_argument("--no-fit", action="store_true", help="强制默认分布：不读特征表（离线路径）")
    ap.add_argument("--no-verify", action="store_true", help="跳过双跑一致性校验")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    policies = tuple(s.strip() for s in args.policies.split(",") if s.strip())
    kwargs = dict(
        n_users=args.users, days=args.days, window=args.window, seed=args.seed,
        audit=args.audit, policies=policies,
        alpha=args.alpha, ridge=args.ridge, sigma=args.sigma,
        features_path=Path(args.features), use_fit=not args.no_fit,
    )
    try:
        run = run_all(**kwargs)
    except ValueError as exc:
        print(f"[stop] 参数无效：{exc}", file=sys.stderr)
        return 2

    determinism = {"checked": False}
    if not args.no_verify:
        run_again = run_all(**kwargs)
        same = run_is_deterministic(run, run_again)
        determinism = {"checked": True, "identical": bool(same)}
        if not same:
            print("[stop] 双跑不一致：存在非确定性来源，拒绝落盘", file=sys.stderr)
            return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(run)
    frames = {
        "bandit_curve.csv": run["frames"]["curve"],
        "bandit_summary.csv": run["frames"]["summary"],
        "bandit_audit.csv": run["frames"]["audit"],
    }
    metrics_path = out_dir / "bandit_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, df in frames.items():
        df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    outputs = {metrics_path.name: {"rows": None, "sha256": _sha16(metrics_path)}}
    for name, df in frames.items():
        outputs[name] = {"rows": int(len(df)), "sha256": _sha16(out_dir / name)}
    manifest = {
        "bandit_version": BANDIT_VERSION,
        "generated_at": _now(),
        "config": run["config"],
        "sim": {
            "sim_version": P.SIM_VERSION,
            "population": run["sim_meta"]["population"],
            "params": P.SIM.snapshot(),
            "effects_summary": run["sim_meta"]["effects_summary"],
        },
        "world": run["world_meta"],
        "hyperparams": metrics["hyperparams"],
        "outputs": outputs,
        "determinism": determinism,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ── 终端报告 ──
    cfg, wm, pop = run["config"], run["world_meta"], run["sim_meta"]["population"]
    src = pop["path"]
    if src == "fit":
        src = f"fit ← {pop['features_file']}（指纹 {pop['features_fingerprint']}）"
    print(f"[info] 模拟：{cfg['n_users']} 人 × {cfg['days']} 天（窗 {cfg['window']} 天，seed {cfg['seed']}）"
          f"｜人口 {src}")
    arms_txt = " / ".join(f"{a} {wm['arm_mean_reward'][a]:+.3f}" for a in ARMS)
    print(f"[info] 世界：在线 {wm['n_online']} 人｜审计 {wm['n_audit']} 人（audit {cfg['audit']:.0%}）"
          f"｜上下文基 {wm['basis']}（宽 {wm['context_width']}）｜oracle 均值 {wm['oracle_mean_reward']:+.3f}"
          f"｜三臂真值均值 {arms_txt}")
    print(f"[info] 策略对比（在线 {wm['n_online']} 人 → 审计 {wm['n_audit']} 人冻结，净奖励口径）：")
    print(f"       {'policy':<15}{'mean':>9}{'head20':>9}{'tail20':>9}{'cum_regret':>12}"
          f"{'audit_mean':>11}{'audit/oracle':>13}{'rec%':>7}{'recall%':>9}")
    for r in run["frames"]["summary"].itertuples():
        print(f"       {r.policy:<15}{r.mean_reward:>9.3f}{r.head20:>9.3f}{r.tail20:>9.3f}"
              f"{r.cum_regret:>12.1f}{r.mean_reward_audit:>11.3f}{r.audit_vs_oracle_frac:>13.3f}"
              f"{r.share_rec:>7.1%}{r.share_recall:>9.1%}")
    curve, summary, audit = frames["bandit_curve.csv"], frames["bandit_summary.csv"], frames["bandit_audit.csv"]
    print(f"[done] → {_rel(out_dir)}（metrics / curve {len(curve)} 行 / summary {len(summary)} 行"
          f" / audit {len(audit)} 行）")
    if determinism["checked"]:
        print(f"[ok] 双跑一致性：{determinism['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
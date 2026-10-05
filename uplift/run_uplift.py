#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S2 Uplift · 运行入口：在模拟环境里学"谁值得干预"，并回答能学多准。

先说人话：
    S1 的模拟器给了完整反事实（每人每臂真值）；这支脚本故意"假装看不见"：
    把训练用户按现实口径降维——rct（每人只观测一条臂）或 full（全都看得见）
    ——用 T-learner 学 τ̂，然后在从未参与训练的留出用户上对三种口径判分：
    与真值像不像（保真度）、名额花给谁最值（策略价值曲线，对照 oracle）。

运行（在仓库根目录）：
    python -m uplift.run_uplift --users 6000 --seed 11
    python -m uplift.run_uplift --no-fit --protocol rct --no-verify   # 离线快路径

产出（默认 data/synthetic/uplift/，合成数据不进公开仓）：
    uplift_metrics.json    配置 / 保真度汇总 / 参考 k 下的策略价值
    uplift_calibration.csv 按 τ̂ 十等分的校准表（每协议 × 臂 × 学习器）
    uplift_policy.csv      策略价值曲线（k% × 策略 × 臂）
    uplift_holdout.csv     留出集逐行评测帧（τ̂ / 真值 / 观测净奖励）
    _manifest.json         版本 / 模拟参数快照 / 模型超参 / 输出指纹 / 双跑一致
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import params as P
from simulator.run_sim import build_tables
from simulator.user_generator import DEFAULT_FEATURES

from . import UPLIFT_VERSION
from .evaluate import KS_DEFAULT, calibration_table, fidelity_table, policy_at, policy_table
from .models import LEARNERS, TLearnerUplift, hyperparams_snapshot
from .protocol import ARMS, FEATURES, PROTOCOLS, TREATED_ARMS, build_observations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "synthetic" / "uplift"
TZ_CN = timezone(timedelta(hours=8))


# ── 核心：跑一遍完整流程（纯内存，deterministic）────────────

def validate_config(
    *, holdout: float, protocols: tuple[str, ...], learners: tuple[str, ...], ks
) -> tuple[int, ...]:
    """入口参数校验（CLI 与测试共用）；返回去重排序后的 ks。"""
    if not (0.0 < holdout < 1.0):
        raise ValueError(f"holdout 需在 (0,1) 内（当前 {holdout}）")
    if not protocols:
        raise ValueError("protocols 不能为空")
    bad_p = [p for p in protocols if p not in PROTOCOLS]
    if bad_p:
        raise ValueError(f"未知协议：{bad_p}（支持 {PROTOCOLS}）")
    bad_l = [l for l in learners if l not in LEARNERS]
    if bad_l:
        raise ValueError(f"未知学习器：{bad_l}（支持 {LEARNERS}）")
    ks_out = tuple(sorted({int(k) for k in ks}))
    if not ks_out or any(k < 1 or k > 100 for k in ks_out):
        raise ValueError(f"ks 需为 [1,100] 内的整数集合（当前 {ks}）")
    return ks_out


def run_all(
    *,
    n_users: int = 6000,
    days: int = 14,
    window: int = P.SIM.reward_window_days,
    seed: int = 11,
    holdout: float = 0.3,
    protocols: tuple[str, ...] = PROTOCOLS,
    learners: tuple[str, ...] = LEARNERS,
    ks: tuple[int, ...] = KS_DEFAULT,
    features_path: Path | None = None,
    use_fit: bool = True,
) -> dict:
    """S1 模拟 → 协议降维 → T-learner 拟合 → 留出评测；同参同种子逐字节一致。"""
    ks = validate_config(holdout=holdout, protocols=protocols, learners=learners, ks=ks)
    tables, sim_meta = build_tables(
        n_users, days, seed, window, features_path if use_fit else None, use_fit
    )
    results: dict[str, dict] = {}
    for proto in protocols:
        obs = build_observations(tables, holdout, seed, proto)
        hold = obs["holdout"].copy()
        for learner in learners:
            model = TLearnerUplift(learner, seed).fit(obs["train"])
            hat = np.zeros(len(hold))
            for arm in ARMS:
                mask = (hold["arm"] == arm).to_numpy()
                if mask.any():  # 只把特征白名单交给模型（真值列永不入模）
                    hat[mask] = model.predict_tau(hold.loc[mask, list(FEATURES)], arm)
            hold[f"tau_hat_{learner}"] = hat

        fid = fidelity_table(hold, learners)
        cal = calibration_table(hold, learners)
        pol = policy_table(hold, learners, ks)
        for frame in (fid, cal, pol):
            frame.insert(0, "protocol", proto)
        hold_cols = ["uid", "arm", *FEATURES, "tau_struct", "tau_ind", "increment",
                     "cost", "reward", *[f"tau_hat_{l}" for l in learners]]
        hold_out = hold.loc[:, hold_cols].copy()
        hold_out.insert(0, "protocol", proto)
        results[proto] = {
            "meta": obs["meta"],
            "fidelity": fid,
            "calibration": cal,
            "policy": pol,
            "holdout": hold_out,
        }
    config = {
        "n_users": int(n_users),
        "days": int(days),
        "window": int(window),
        "seed": int(seed),
        "holdout": float(holdout),
        "protocols": list(protocols),
        "learners": list(learners),
        "ks": list(ks),
        "use_fit": bool(use_fit),
        "features_file": str(features_path) if (use_fit and features_path) else None,
    }
    return {"tables": tables, "sim_meta": sim_meta, "results": results, "config": config}


def run_is_deterministic(first: dict, second: dict) -> bool:
    """双跑一致性：模拟三表 + 每个协议的四个产出帧都必须逐值相等。"""
    if set(first["tables"]) != set(second["tables"]):
        return False
    if not all(first["tables"][k].equals(second["tables"][k]) for k in first["tables"]):
        return False
    if set(first["results"]) != set(second["results"]):
        return False
    for proto in first["results"]:
        for name in ("fidelity", "calibration", "policy", "holdout"):
            if not first["results"][proto][name].equals(second["results"][proto][name]):
                return False
    return True


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
    """uplift_metrics.json：主结果文件（配置 + 保真度 + 参考 k 策略价值）。"""
    cfg, sim_meta = run["config"], run["sim_meta"]
    ref_k = 20 if 20 in cfg["ks"] else cfg["ks"][len(cfg["ks"]) // 2]
    metrics = {
        "uplift_version": UPLIFT_VERSION,
        "generated_at": _now(),
        "config": {**cfg, "reference_k_pct": ref_k, "population_path": sim_meta["population"]["path"]},
        "sim": {"sim_version": P.SIM_VERSION, "effects_summary": sim_meta["effects_summary"]},
        "model_hyperparams": hyperparams_snapshot(tuple(cfg["learners"])),
        "results": {},
    }
    for proto, res in run["results"].items():
        metrics["results"][proto] = {
            "n_train_users": res["meta"]["n_train_users"],
            "n_holdout_users": res["meta"]["n_holdout_users"],
            "train_rows": res["meta"]["train_rows"],
            "holdout_rows": res["meta"]["holdout_rows"],
            "train_arm_counts": res["meta"]["train_arm_counts"],
            "fidelity": res["fidelity"].to_dict("records"),
            "policy_at_reference_k": {
                arm: policy_at(res["policy"], arm, ref_k) for arm in TREATED_ARMS
            },
        }
    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="S2 Uplift：模拟环境里学干预效应，并回答能学多准")
    ap.add_argument("--users", type=int, default=6000, help="合成用户数（默认 6000）")
    ap.add_argument("--days", type=int, default=14, help="模拟天数（默认 14）")
    ap.add_argument("--window", type=int, default=P.SIM.reward_window_days, help="奖励窗口天数（默认 7）")
    ap.add_argument("--seed", type=int, default=11, help="随机种子（默认 11）")
    ap.add_argument("--holdout", type=float, default=0.3, help="留出用户比例（默认 0.3）")
    ap.add_argument("--protocol", choices=["both", *PROTOCOLS], default="both", help="训练协议（默认 both）")
    ap.add_argument("--learners", default=",".join(LEARNERS), help="学习器，逗号分隔（默认 hgb,ridge）")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS_DEFAULT), help="策略价值曲线的 k%% 列表")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 data/synthetic/uplift）")
    ap.add_argument("--features", default=str(DEFAULT_FEATURES), help="L2 特征表路径（fit 路径使用）")
    ap.add_argument("--no-fit", action="store_true", help="强制默认分布：不读特征表（离线路径）")
    ap.add_argument("--no-verify", action="store_true", help="跳过双跑一致性校验")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    protocols = PROTOCOLS if args.protocol == "both" else (args.protocol,)
    learners = tuple(s.strip() for s in args.learners.split(",") if s.strip())
    try:
        ks = tuple(int(s) for s in args.ks.split(",") if s.strip())
    except ValueError:
        print("[stop] --ks 需为逗号分隔的整数", file=sys.stderr)
        return 2
    kwargs = dict(
        n_users=args.users, days=args.days, window=args.window, seed=args.seed,
        holdout=args.holdout, protocols=protocols, learners=learners, ks=ks,
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
    calibration = pd.concat([r["calibration"] for r in run["results"].values()], ignore_index=True)
    policy = pd.concat([r["policy"] for r in run["results"].values()], ignore_index=True)
    holdout = pd.concat([r["holdout"] for r in run["results"].values()], ignore_index=True)
    frames = {
        "uplift_calibration.csv": calibration,
        "uplift_policy.csv": policy,
        "uplift_holdout.csv": holdout,
    }
    metrics_path = out_dir / "uplift_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, df in frames.items():
        df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    outputs = {metrics_path.name: {"rows": None, "sha256": _sha16(metrics_path)}}
    for name, df in frames.items():
        outputs[name] = {"rows": int(len(df)), "sha256": _sha16(out_dir / name)}
    manifest = {
        "uplift_version": UPLIFT_VERSION,
        "generated_at": _now(),
        "config": run["config"],
        "sim": {
            "sim_version": P.SIM_VERSION,
            "population": run["sim_meta"]["population"],
            "params": P.SIM.snapshot(),
            "effects_summary": run["sim_meta"]["effects_summary"],
        },
        "models": hyperparams_snapshot(tuple(run["config"]["learners"])),
        "outputs": outputs,
        "determinism": determinism,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ── 终端报告 ──
    cfg, pop = run["config"], run["sim_meta"]["population"]
    src = pop["path"]
    if src == "fit":
        src = f"fit ← {pop['features_file']}（指纹 {pop['features_fingerprint']}）"
    print(f"[info] 模拟：{cfg['n_users']} 人 × {cfg['days']} 天（窗 {cfg['window']} 天，seed {cfg['seed']}）"
          f"｜人口 {src}")
    for proto in cfg["protocols"]:
        m = run["results"][proto]["meta"]
        counts = " / ".join(f"{a} {m['train_arm_counts'][a]}" for a in ARMS)
        print(f"[info] 协议 {proto:<5}：训练 {m['n_train_users']} 人 · {m['train_rows']} 行（{counts}）"
              f"｜留出 {m['n_holdout_users']} 人 · {m['holdout_rows']} 行")
    for proto in cfg["protocols"]:
        print(f"[info] 保真度（{proto}）：")
        print(f"       {'arm':<7}{'learner':>8}{'n':>7}{'mean_tau_hat':>13}{'mean_tau_str':>13}"
              f"{'bias':>9}{'spear_str':>10}{'spear_ind':>10}")
        for r in run["results"][proto]["fidelity"].itertuples():
            print(f"       {r.arm:<7}{r.learner:>8}{r.n:>7}{r.mean_tau_hat:>13.3f}"
                  f"{r.mean_tau_struct:>13.3f}{r.bias:>9.3f}{r.spearman_struct:>10.3f}"
                  f"{r.spearman_ind:>10.3f}")
    ref_k = metrics["config"]["reference_k_pct"]
    order = [f"model_{l}" for l in cfg["learners"]] + ["random", "oracle_struct", "oracle_ind"]
    for proto in cfg["protocols"]:
        for arm in TREATED_ARMS:
            vals = policy_at(run["results"][proto]["policy"], arm, ref_k)
            body = "｜".join(f"{name} {vals[name]:+.3f}" for name in order if name in vals)
            print(f"[info] 策略价值（{proto} · {arm} · k={ref_k}%）：{body}")
    print(f"[done] → {_rel(out_dir)}（metrics / calibration {len(calibration)} 行"
          f" / policy {len(policy)} 行 / holdout {len(holdout)} 行）")
    if determinism["checked"]:
        print(f"[ok] 双跑一致性：{determinism['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
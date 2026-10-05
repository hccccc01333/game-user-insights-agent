#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S4 · 接入 harness：Bandit 干预工具 + Critic，让 Agent 决策循环真正调用。

先说人话：
    S3 的策略对象此前只活在离线评测脚本里；S4 把它接进 harness 的决策循环——
    六环节（异常 → 人群 → 原因 → 风险 → 干预 → 实验）里的"干预"环节，由
    Agent 调用真正的工具：对已圈定人群批量分配触达臂（select → Critic 复核 →
    结算 → 学习），Critic 作为安全门与循环级校验把关，收工前必须完成干预。
    同一套流程跑"带 / 不带 Critic"两条路径：一眼看出安全门的代价值不值。

    演示数据全部来自 S1 合成人口（离线可复现、逐字节一致），不触任何真实数据。

运行（在仓库根目录）：
    python -m harness.run_agent --no-fit --users 6000     # 强制默认分布（离线路径）
    python -m harness.run_agent --no-fit --users 800 --batch 120   # 小规模快跑

产出（默认 data/synthetic/harness/，合成数据不进公开仓）：
    s4_metrics.json   配置 / 世界口径 / Critic 口径 / 两模式汇总 / 审计对照
    s4_batches.csv    逐人次分配日志（mode / proposed / final / vetoed / reward）
    s4_trace.jsonl    两模式的循环轨迹（每步：决策 → 观察 → 校验）
    _manifest.json    版本 / 输入指纹 / 输出指纹 / 双跑一致
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from bandit.evaluate import audit_summary, run_audit
from bandit.policies import POLICIES, make_policy
from bandit.protocol import ARMS, BASES, DEFAULT_BASIS, build_world
from simulator import params as P
from simulator.run_sim import build_tables
from simulator.user_generator import DEFAULT_FEATURES

from . import HARNESS_VERSION
from .bandit_tools import (
    DEFAULT_COUNT, INTERVENTION_TOOL, InterventionConsole, register_intervention_tools,
)
from .critic import (
    DEFAULT_BETA, DEFAULT_EXPLORE_FRAC, DEFAULT_EXPLORE_MIN, DEFAULT_MIN_OBS,
    CriticVerifier, InterventionCritic,
)
from .loop import run
from .model import MockModel
from .planner import FINISH_TOOL, Planner
from .registry import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "synthetic" / "harness"
TZ_CN = timezone(timedelta(hours=8))

GOAL = "排查：合成人口互动走低与沉默风险——定位高风险人群，批量分配触达，并设计验证实验"
COHORT_NAME = "沉默高风险人群"
COHORT_RULE = "act_30d = 0 且 silence_days ≥ 90（召回效应峰值附近）"
COHORT_SILENCE_MIN = 90.0
MODES = ("with_critic", "no_critic")


# ── 六环节工具组装（干预环节 = bandit 工具 + Critic）────────

def _used(name: str):
    """前置条件：某个工具已经执行过（与 loop 演示同款）。"""
    return lambda state: any((h.get("decision") or {}).get("tool") == name for h in state.history)


def build_registry(tables: dict[str, pd.DataFrame], built: dict,
                   console: InterventionConsole, batch: int = DEFAULT_COUNT) -> ToolRegistry:
    """按六环节注册工具；人群与干预两个环节是"真数"（合成世界可观测口径）。

    · detect_anomaly：对照基线互动量的后半段 vs 前半段环比 + 人口沉默占比
    · locate_cohort ：按可观测条件从在线池圈人（真圈人）并交给干预台
    · analyze_cause ：人群状态中位数（沉默 / 资历 / 集中度）
    · assess_risk   ：沉默时长分档
    · allocate_interventions：bandit 批量分配（真干预，含 Critic 安全门）
    · design_experiment：基于实际分配批次的 A/B 设计
    """
    users, daily = tables["users"], tables["daily"]
    online_idx = built["online"]

    def detect_anomaly() -> dict:
        void = daily[daily["arm"] == "control"].sort_values("day")["void_population"].to_numpy(float)
        half = len(void) // 2
        w1, w2 = float(void[:half].sum()), float(void[half:].sum())
        if w1 <= 0:
            raise ValueError("对照基线为 0，无法计算环比")
        silent_share = float(((users["act_30d"] == 0) & (users["silence_days"] >= COHORT_SILENCE_MIN)).mean())
        return {
            "metric": "void_events_second_half_vs_first_half_pct",
            "delta_pct": round((w2 - w1) / w1 * 100.0, 2),
            "silent_share": round(silent_share, 4),
        }

    def locate_cohort() -> dict:
        mask = (users["act_30d"] == 0) & (users["silence_days"] >= COHORT_SILENCE_MIN)
        idx = [int(i) for i in online_idx if bool(mask.iloc[int(i)])]
        info = console.open_cohort(idx, COHORT_NAME)
        return {
            "cohort": COHORT_NAME,
            "rule": COHORT_RULE,
            "size": info["size"],
            "share_of_online": round(len(idx) / len(online_idx), 4),
        }

    def analyze_cause() -> dict:
        sub = users.iloc[console.cohort_index()]
        return {
            "cohort": COHORT_NAME,
            "silence_median": round(float(sub["silence_days"].median()), 1),
            "tenure_median": round(float(sub["tenure_days"].median()), 1),
            "concentration_median": round(float(sub["interest_concentration"].median()), 4),
            "n_total": int(len(sub)),
        }

    def assess_risk() -> dict:
        s = users.iloc[console.cohort_index()]["silence_days"]
        bands = {
            "90-180": int(((s >= 90) & (s < 180)).sum()),
            "180-365": int(((s >= 180) & (s < 365)).sum()),
            "365+": int((s >= 365).sum()),
        }
        return {"cohort": COHORT_NAME, "bands": bands, "n_total": int(len(s))}

    def design_experiment() -> dict:
        if not console.batch_logs:
            raise ValueError("尚无分配批次：先完成干预环节")
        last = console.batch_logs[-1]
        return {
            "grouping": "A/B",
            "primary_metric": "w1_void_events",
            "n_treated": int(last["batch_size"]),
            "arms": dict(last["arms"]),
            "note": "触达组 vs 不打扰组；批次名额来自实际分配账本",
        }

    reg = ToolRegistry()
    reg.register("detect_anomaly", detect_anomaly,
                 "发现社区指标异常（对照基线环比 + 沉默人口占比；合成人口口径）")
    reg.register("locate_cohort", locate_cohort,
                 "定位异常涉及的玩家群体（按可观测条件圈人，并交给干预台）",
                 when=_used("detect_anomaly"))
    reg.register("analyze_cause", analyze_cause,
                 "分析该人群的沉默时长 / 资历 / 兴趣集中度中位数",
                 when=_used("locate_cohort"))
    reg.register("assess_risk", assess_risk,
                 "按沉默时长分档做风险分层",
                 when=_used("analyze_cause"))
    register_intervention_tools(reg, console, default_count=batch)
    reg.register("design_experiment", design_experiment,
                 "基于实际分配批次设计 A/B 实验（分组 + 主指标）",
                 when=_used(INTERVENTION_TOOL))
    return reg


# ── 核心：跑一遍完整流程（纯内存，deterministic）────────────

def validate_config(
    *,
    n_users: int,
    days: int,
    window: int,
    audit: float,
    policy: str,
    basis: str,
    batch: int,
    critic_beta: float,
    critic_min_obs: int,
    critic_explore_frac: float,
    critic_explore_min: int,
    max_steps: int,
) -> None:
    """入口参数校验（CLI 与测试共用）。"""
    if n_users < 200:
        raise ValueError(f"用户数需 ≥ 200（当前 {n_users}）：圈人、分配与审计都需要足够样本")
    if days < 2:
        raise ValueError(f"天数需 ≥ 2（当前 {days}）：异常环比需要前后两段")
    if not (1 <= window <= days):
        raise ValueError(f"奖励窗口需满足 1 ≤ window ≤ days（当前 window={window}, days={days}）")
    if not (0.0 < audit < 1.0):
        raise ValueError(f"审计比例需在 (0,1) 内（当前 {audit}）")
    if policy not in POLICIES:
        raise ValueError(f"未知策略：{policy!r}（支持 {POLICIES}）")
    if policy == "oracle":
        raise ValueError("oracle 是评测上界（读真值矩阵），不可作为可部署策略接入")
    if basis not in BASES:
        raise ValueError(f"未知上下文基：{basis!r}（支持 {BASES}）")
    if batch < 1:
        raise ValueError(f"批大小需 ≥ 1（当前 {batch}）")
    if critic_beta <= 0:
        raise ValueError(f"critic_beta 需 > 0（当前 {critic_beta}）")
    if critic_min_obs < 0:
        raise ValueError(f"critic_min_obs 需 ≥ 0（当前 {critic_min_obs}）")
    if not (0.0 <= critic_explore_frac <= 1.0):
        raise ValueError(f"explore_frac 需在 [0,1] 内（当前 {critic_explore_frac}）")
    if critic_explore_min < 0:
        raise ValueError(f"explore_min 需 ≥ 0（当前 {critic_explore_min}）")
    if max_steps < 7:
        raise ValueError(f"max_steps 需 ≥ 7（六环节 + 收工；当前 {max_steps}）")


def run_agent_demo(
    *,
    n_users: int = 6000,
    days: int = 14,
    window: int = P.SIM.reward_window_days,
    seed: int = 13,
    audit: float = 0.3,
    policy: str = "linucb",
    basis: str = DEFAULT_BASIS,
    batch: int = DEFAULT_COUNT,
    critic_beta: float = DEFAULT_BETA,
    critic_min_obs: int = DEFAULT_MIN_OBS,
    critic_explore_frac: float = DEFAULT_EXPLORE_FRAC,
    critic_explore_min: int = DEFAULT_EXPLORE_MIN,
    max_steps: int = 8,
    features_path: Path | None = None,
    use_fit: bool = True,
) -> dict:
    """S1 模拟 → S3 世界 → 六环节 Agent 循环（带 / 不带 Critic 两条路径）；同参同种子逐字节一致。"""
    validate_config(
        n_users=n_users, days=days, window=window, audit=audit, policy=policy, basis=basis,
        batch=batch, critic_beta=critic_beta, critic_min_obs=critic_min_obs,
        critic_explore_frac=critic_explore_frac,
        critic_explore_min=critic_explore_min, max_steps=max_steps,
    )
    tables, sim_meta = build_tables(
        n_users, days, seed, window, features_path if use_fit else None, use_fit
    )
    built = build_world(tables, audit, seed, basis)
    world, online_idx, audit_idx = built["world"], built["online"], built["audit"]
    dim = built["meta"]["context_width"]

    modes: dict[str, dict] = {}
    traces: dict[str, list] = {}
    frames: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        policy_obj = make_policy(policy, context_dim=dim, seed=seed, oracle_arm=world.oracle_arm)
        critic = (
            InterventionCritic(
                ARMS, dim, beta=critic_beta, min_obs=critic_min_obs,
                explore_frac=critic_explore_frac, explore_min=critic_explore_min,
            )
            if mode == "with_critic" else None
        )
        console = InterventionConsole(world, policy_obj, critic)
        registry = build_registry(tables, built, console, batch=batch)
        state = run(
            GOAL, registry,
            planner=Planner(model=MockModel(), recent_steps=max_steps),
            verifier=CriticVerifier(),
            max_steps=max_steps,
        )
        audit_log = run_audit(world, audit_idx, policy_obj)  # 冻结评测：只选不学

        df = console.rows_frame()
        df.insert(0, "mode", mode)
        frames[mode] = df
        traces[mode] = state.history
        modes[mode] = {
            "finished": any(
                (h.get("decision") or {}).get("tool") == FINISH_TOOL and h.get("verified")
                for h in state.history
            ),
            "loop_steps": len(state.history),
            "batches": list(console.batch_logs),
            "online": console.eval_summary() if console.batch_logs else None,
            "audit": audit_summary(audit_log),
            "critic": critic.stats() if critic is not None else None,
        }

    config = {
        "n_users": int(n_users),
        "days": int(days),
        "window": int(window),
        "seed": int(seed),
        "audit": float(audit),
        "policy": policy,
        "basis": basis,
        "batch": int(batch),
        "critic_beta": float(critic_beta),
        "critic_min_obs": int(critic_min_obs),
        "critic_explore_frac": float(critic_explore_frac),
        "critic_explore_min": int(critic_explore_min),
        "max_steps": int(max_steps),
        "use_fit": bool(use_fit),
        "features_file": _rel(features_path) if (use_fit and features_path) else None,
    }
    return {
        "tables": tables,
        "sim_meta": sim_meta,
        "world_meta": built["meta"],
        "config": config,
        "modes": modes,
        "traces": traces,
        "frames": {"s4_batches.csv": pd.concat(frames.values(), ignore_index=True)},
    }


def run_is_deterministic(first: dict, second: dict) -> bool:
    """双跑一致性：分配日志 / 循环轨迹 / 两模式汇总都必须逐值相等。"""
    if set(first["frames"]) != set(second["frames"]):
        return False
    for k in first["frames"]:
        if not first["frames"][k].equals(second["frames"][k]):
            return False
    if first["traces"] != second["traces"]:
        return False
    dump = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)  # noqa: E731
    return dump(first["modes"]) == dump(second["modes"])


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
    """s4_metrics.json：主结果文件（配置 + 世界口径 + Critic 口径 + 两模式汇总）。"""
    cfg, sim_meta, wm = run["config"], run["sim_meta"], run["world_meta"]
    return {
        "harness_version": HARNESS_VERSION,
        "generated_at": _now(),
        "goal": GOAL,
        "config": {**cfg, "population_path": sim_meta["population"]["path"]},
        "sim": {
            "sim_version": P.SIM_VERSION,
            "effects_summary": sim_meta["effects_summary"],
        },
        "world": wm,
        "critic": {
            "rule": ("证据门槛：所选臂观测 ≥ min_obs 次；价值门 lcb(所选臂) > lcb(control) 才放行；"
                     "否则按每批探索额度放行，额度用尽则否决并降级 control"),
            "beta": cfg["critic_beta"],
            "min_obs": cfg["critic_min_obs"],
            "explore_frac": cfg["critic_explore_frac"],
            "explore_min": cfg["critic_explore_min"],
        },
        "modes": run["modes"],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="S4：Bandit 干预工具 + Critic 接入 harness 决策循环")
    ap.add_argument("--users", type=int, default=6000, help="合成用户数（默认 6000）")
    ap.add_argument("--days", type=int, default=14, help="模拟天数（默认 14）")
    ap.add_argument("--window", type=int, default=P.SIM.reward_window_days, help="奖励窗口天数（默认 7）")
    ap.add_argument("--seed", type=int, default=13, help="随机种子（默认 13）")
    ap.add_argument("--audit", type=float, default=0.3, help="审计池比例（默认 0.3）")
    ap.add_argument("--policy", default="linucb", help="学习策略（默认 linucb）")
    ap.add_argument("--basis", default=DEFAULT_BASIS, help="上下文基（默认 logsq）")
    ap.add_argument("--batch", type=int, default=DEFAULT_COUNT, help="单批名额数（默认 200）")
    ap.add_argument("--critic-beta", type=float, default=DEFAULT_BETA, help="Critic 保守系数（默认 1.0）")
    ap.add_argument("--min-obs", type=int, default=DEFAULT_MIN_OBS,
                    help="Critic 证据门槛：每条臂至少观测该次数才开价值门（默认 3）")
    ap.add_argument("--explore-frac", type=float, default=DEFAULT_EXPLORE_FRAC,
                    help="每批探索额度比例（默认 0.05）")
    ap.add_argument("--explore-min", type=int, default=DEFAULT_EXPLORE_MIN,
                    help="每批探索额度下限（默认 2）")
    ap.add_argument("--max-steps", type=int, default=8, help="循环步数上限（默认 8）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 data/synthetic/harness）")
    ap.add_argument("--features", default=str(DEFAULT_FEATURES), help="L2 特征表路径（fit 路径使用）")
    ap.add_argument("--no-fit", action="store_true", help="强制默认分布：不读特征表（离线路径）")
    ap.add_argument("--no-verify", action="store_true", help="跳过双跑一致性校验")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = dict(
        n_users=args.users, days=args.days, window=args.window, seed=args.seed, audit=args.audit,
        policy=args.policy, basis=args.basis, batch=args.batch,
        critic_beta=args.critic_beta, critic_min_obs=args.min_obs,
        critic_explore_frac=args.explore_frac,
        critic_explore_min=args.explore_min, max_steps=args.max_steps,
        features_path=Path(args.features), use_fit=not args.no_fit,
    )
    try:
        run_result = run_agent_demo(**kwargs)
    except ValueError as exc:
        print(f"[stop] 参数无效：{exc}", file=sys.stderr)
        return 2

    determinism = {"checked": False}
    if not args.no_verify:
        again = run_agent_demo(**kwargs)
        same = run_is_deterministic(run_result, again)
        determinism = {"checked": True, "identical": bool(same)}
        if not same:
            print("[stop] 双跑不一致：存在非确定性来源，拒绝落盘", file=sys.stderr)
            return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(run_result)
    batches = run_result["frames"]["s4_batches.csv"]
    trace_lines = [
        json.dumps({"mode": mode, **entry}, ensure_ascii=False, default=str)
        for mode, entries in run_result["traces"].items()
        for entry in entries
    ]

    metrics_path = out_dir / "s4_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    batches_path = out_dir / "s4_batches.csv"
    batches.to_csv(batches_path, index=False, encoding="utf-8-sig")
    trace_path = out_dir / "s4_trace.jsonl"
    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

    outputs = {
        metrics_path.name: {"rows": None, "sha256": _sha16(metrics_path)},
        batches_path.name: {"rows": int(len(batches)), "sha256": _sha16(batches_path)},
        trace_path.name: {"rows": len(trace_lines), "sha256": _sha16(trace_path)},
    }
    manifest = {
        "harness_version": HARNESS_VERSION,
        "generated_at": _now(),
        "config": run_result["config"],
        "sim": {
            "sim_version": P.SIM_VERSION,
            "population": run_result["sim_meta"]["population"],
            "params": P.SIM.snapshot(),
            "effects_summary": run_result["sim_meta"]["effects_summary"],
        },
        "world": run_result["world_meta"],
        "critic": metrics["critic"],
        "outputs": outputs,
        "determinism": determinism,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ── 终端报告 ──
    cfg, wm, pop = run_result["config"], run_result["world_meta"], run_result["sim_meta"]["population"]
    src = pop["path"]
    if src == "fit":
        src = f"fit ← {pop['features_file']}（指纹 {pop['features_fingerprint']}）"
    print(f"[info] 模拟：{cfg['n_users']} 人 × {cfg['days']} 天（窗 {cfg['window']} 天，seed {cfg['seed']}）"
          f"｜人口 {src}")
    arms_txt = " / ".join(f"{a} {wm['arm_mean_reward'][a]:+.3f}" for a in ARMS)
    print(f"[info] 世界：在线 {wm['n_online']} 人｜审计 {wm['n_audit']} 人（audit {cfg['audit']:.0%}）"
          f"｜基 {wm['basis']}（宽 {wm['context_width']}）｜三臂真值均值 {arms_txt}")
    print(f"[info] 六环节演示（MockModel + 真干预台；policy {cfg['policy']}，batch {cfg['batch']}，"
          f"Critic β={cfg['critic_beta']}，min_obs={cfg['critic_min_obs']}，"
          f"探索额度 {cfg['critic_explore_frac']:.0%}/批）：")
    print(f"       {'mode':<14}{'steps':>6}{'alloc':>7}{'veto%':>7}{'online':>9}"
          f"{'vs_oracle':>11}{'audit/oracle':>13}{'finished':>9}")
    for mode in MODES:
        m = metrics["modes"][mode]
        online = m["online"] or {}
        veto = online.get("veto_rate")
        avg = online.get("avg_reward")
        frac = online.get("reward_vs_oracle_frac")
        veto_txt = f"{veto:>7.1%}" if veto is not None else f"{'-':>7}"
        avg_txt = f"{avg:>9.3f}" if avg is not None else f"{'-':>9}"
        frac_txt = f"{frac:>11.3f}" if frac is not None else f"{'-':>11}"
        print(f"       {mode:<14}{m['loop_steps']:>6}{online.get('steps', 0):>7}{veto_txt}"
              f"{avg_txt}{frac_txt}"
              f"{m['audit']['audit_vs_oracle_frac']:>13.3f}{str(m['finished']):>9}")
    print(f"[done] → {_rel(out_dir)}（batches {len(batches)} 行 / trace {len(trace_lines)} 行）")
    if determinism["checked"]:
        print(f"[ok] 双跑一致性：{determinism['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
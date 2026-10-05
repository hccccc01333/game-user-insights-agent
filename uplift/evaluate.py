# -*- coding: utf-8 -*-
"""S2 Uplift · 评测：保真度 / 分位校准 / 策略价值曲线。

先说人话：
    模型学到的 τ̂ 好不好，分三层看：
      1) 保真度——τ̂ 与结构真值像不像：偏差、相关（Pearson / Spearman）、
         MSE / MAE；τ_ind 是加了不可观测个体噪声的个体真值，作另一面镜子。
      2) 分位校准——按 τ̂ 十等分，看每档"预测均值 vs 真值均值"是否贴合：
         排序对了但整体放大 / 缩小，都能在这里现形。
      3) 策略价值——触达名额按 τ̂ 排序花出去，看每单位触达的真实净收益：
         参照 random（不排序）、oracle_struct（知道结构真值排序）、
         oracle_ind（知道个体真值排序）——模型离 oracle 还有多远，一目了然。

硬规则：本模块只读留出集（训练从未见过这些用户；真值只用于判分）。
策略价值的口径是 reward = increment − cost（模拟器已按窗内增量 − 触达成本算好）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .protocol import TREATED_ARMS

KS_DEFAULT: tuple[int, ...] = (10, 20, 30, 50, 100)
CALIBRATION_BINS = 10


# ── 相关性工具（无 scipy 依赖：秩相关 = 先取秩再算 Pearson）──

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    return _pearson(ra, rb)


# ── 1. 保真度 ───────────────────────────────────────────────

FIDELITY_COLUMNS = (
    "arm", "learner", "n",
    "mean_tau_hat", "mean_tau_struct", "bias",
    "pearson_struct", "spearman_struct", "spearman_ind",
    "mse_struct", "mae_struct",
)


def fidelity_table(hold: pd.DataFrame, learners: tuple[str, ...]) -> pd.DataFrame:
    """每臂每学习器一行：τ̂ 与真值的偏差 / 相关 / 误差（臂内按留出用户判分）。"""
    rows = []
    for arm in TREATED_ARMS:
        sub = hold[hold["arm"] == arm]
        ts = sub["tau_struct"].to_numpy(dtype=float)
        ti = sub["tau_ind"].to_numpy(dtype=float)
        for learner in learners:
            hat = sub[f"tau_hat_{learner}"].to_numpy(dtype=float)
            rows.append({
                "arm": arm,
                "learner": learner,
                "n": int(len(sub)),
                "mean_tau_hat": round(float(hat.mean()), 4),
                "mean_tau_struct": round(float(ts.mean()), 4),
                "bias": round(float(hat.mean() - ts.mean()), 4),
                "pearson_struct": round(_pearson(hat, ts), 4),
                "spearman_struct": round(_spearman(hat, ts), 4),
                "spearman_ind": round(_spearman(hat, ti), 4),
                "mse_struct": round(float(np.mean((hat - ts) ** 2)), 4),
                "mae_struct": round(float(np.mean(np.abs(hat - ts))), 4),
            })
    return pd.DataFrame(rows, columns=list(FIDELITY_COLUMNS))


# ── 2. 分位校准（按 τ̂ 十等分）───────────────────────────────

CALIBRATION_COLUMNS = (
    "arm", "learner", "bin", "n",
    "mean_tau_hat", "mean_tau_struct", "mean_tau_ind",
)


def calibration_table(
    hold: pd.DataFrame, learners: tuple[str, ...], bins: int = CALIBRATION_BINS
) -> pd.DataFrame:
    """按 τ̂ 名次分箱（每箱人数尽量相等）：每箱预测均值 vs 两种真值均值。"""
    if bins < 2:
        raise ValueError(f"分箱数需 ≥ 2（当前 {bins}）")
    rows = []
    for arm in TREATED_ARMS:
        sub = hold[hold["arm"] == arm]
        n = len(sub)
        if n < bins:
            raise ValueError(f"留出集 {arm} 臂人数不足分箱（n={n}, bins={bins}）")
        for learner in learners:
            hat = sub[f"tau_hat_{learner}"]
            rank = hat.rank(method="first")
            bin_idx = np.ceil(rank.to_numpy() / n * bins).astype(int).clip(1, bins)
            for b in range(1, bins + 1):
                m = bin_idx == b
                rows.append({
                    "arm": arm,
                    "learner": learner,
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_tau_hat": round(float(hat[m].mean()), 4),
                    "mean_tau_struct": round(float(sub.loc[m, "tau_struct"].mean()), 4),
                    "mean_tau_ind": round(float(sub.loc[m, "tau_ind"].mean()), 4),
                })
    return pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))


# ── 3. 策略价值曲线 ─────────────────────────────────────────

POLICY_COLUMNS = ("arm", "policy", "k_pct", "n_selected", "value")


def policy_names(learners: tuple[str, ...]) -> tuple[str, ...]:
    """策略集合：模型（每学习器一条）+ 不排序 + 两条 oracle 参照。"""
    return tuple(f"model_{l}" for l in learners) + ("random", "oracle_struct", "oracle_ind")


def policy_table(
    hold: pd.DataFrame, learners: tuple[str, ...], ks: tuple[int, ...] = KS_DEFAULT
) -> pd.DataFrame:
    """策略价值：按各策略得分取前 k% 用户触达，value = 其真实净奖励均值。

    · random        不排序（解析期望 = 该臂全体均值，与 k 无关）
    · oracle_struct 按 tau_struct 排序（给定可观测状态下的最优排名）
    · oracle_ind    按 tau_ind 排序（个体真值排名，含不可观测噪声）
    · model_*       按 τ̂ 排序（学习器排名）
    k=100% 时所有策略取全体，value 必然相等（tests 锁死该端点）。
    """
    rows = []
    for arm in TREATED_ARMS:
        sub = hold[hold["arm"] == arm]
        n = len(sub)
        if n < 1:
            raise ValueError(f"留出集 {arm} 臂为空，无法评估策略价值")
        reward = sub["reward"].to_numpy(dtype=float)
        mean_reward = float(reward.mean())
        scores: dict[str, np.ndarray | None] = {
            f"model_{l}": sub[f"tau_hat_{l}"].to_numpy(dtype=float) for l in learners
        }
        scores["random"] = None
        scores["oracle_struct"] = sub["tau_struct"].to_numpy(dtype=float)
        scores["oracle_ind"] = sub["tau_ind"].to_numpy(dtype=float)
        for k in ks:
            if not (1 <= k <= 100):
                raise ValueError(f"k 需在 [1,100]（当前 {k}）")
            n_sel = max(1, int(round(n * k / 100.0)))
            for policy in policy_names(learners):
                score = scores[policy]
                if score is None:
                    value = mean_reward  # 不排序的解析期望：各 k 相同
                else:
                    order = np.argsort(-score, kind="stable")[:n_sel]
                    value = float(reward[order].mean())
                rows.append({
                    "arm": arm,
                    "policy": policy,
                    "k_pct": int(k),
                    "n_selected": int(n_sel),
                    "value": round(value, 4),
                })
    return pd.DataFrame(rows, columns=list(POLICY_COLUMNS))


def policy_at(policy_df: pd.DataFrame, arm: str, k: int) -> dict[str, float]:
    """终端汇总用：某臂某 k 下所有策略的 value（policy → value）。"""
    sub = policy_df[(policy_df["arm"] == arm) & (policy_df["k_pct"] == k)]
    return {str(r.policy): float(r.value) for r in sub.itertuples()}
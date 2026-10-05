# -*- coding: utf-8 -*-
"""S3 Bandit · 评测：在线学习曲线 + 审计池冻结评测 + 每策略一行汇总。

先说人话：
    一条 bandit 策略好不好，看两层：
      1) 在线层——边决策边学习的过程：前 20 步与后 20 步的平均净奖励差
         （学习信号）、累计遗憾（cum_regret = Σ(最优臂奖励 − 实得奖励)）；
      2) 审计层——学完之后把策略当场冻结（只读后验均值、不再探索），
         到从未参与学习的审计池上跑一遍：得分对比 oracle 上界，看"学到的
         知识"本身值多少（把"探索"与"知识"分开计价）。
    口径：
      · 每个策略跑在线池时从零开始（冷启动），策略之间共享同一个世界；
      · 记忆里只允许出现 observe 返回过的净奖励——日志里 reward 列即所选臂
        实际观测值，oracle_reward 列仅供判分（策略不可读）；
      · 遗憾 regret = oracle_reward − reward ≥ 0（oracle 恒为 0 上界）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .protocol import ARMS

ONLINE_COLUMNS = (
    "policy", "step", "uid", "arm", "reward", "oracle_reward",
    "cum_reward", "avg_reward", "cum_regret", "avg_regret",
)
AUDIT_COLUMNS = (
    "policy", "uid", "arm", "reward", "oracle_arm", "oracle_reward", "regret",
)
SUMMARY_COLUMNS = (
    "policy", "n_online", "mean_reward", "head20", "tail20", "cum_regret",
    "regret_per_step", "oracle_mean", "reward_vs_oracle_frac",
    "share_control", "share_rec", "share_recall",
    "n_audit", "mean_reward_audit", "oracle_mean_audit", "audit_regret",
    "audit_vs_oracle_frac", "audit_share_control", "audit_share_rec", "audit_share_recall",
)


def _policy_arm(world, policy, x: np.ndarray, i: int, *, frozen: bool) -> str:
    """让策略选臂并校验下标合法（越界 = 策略实现错误，显式报错）。"""
    a = policy.greedy(x, i) if frozen else policy.select(x, i)
    if not (0 <= int(a) < len(world.arms)):
        raise ValueError(f"策略 {policy.name} 返回非法臂下标：{a}")
    return world.arms[int(a)]


def run_online(world, order: np.ndarray, policy) -> pd.DataFrame:
    """在线跑一遍：逐人次 select → observe（唯一反馈）→ update。

    返回逐人次日志（列 ONLINE_COLUMNS；展示值 round 6，累计用原始浮点累加）。
    每个人次恰好观测一条臂——world.revealed 记录可用于核对"没偷看"。
    """
    world.reset_revealed()
    rows = []
    cum_reward = 0.0
    cum_regret = 0.0
    for step, i in enumerate(order):
        i = int(i)
        x = world.context_of(i)
        arm = _policy_arm(world, policy, x, i, frozen=False)
        r = world.observe(i, arm)
        policy.update(x, world.arms.index(arm), r)
        _, oracle_r = world.oracle_best(i)
        regret = oracle_r - r
        cum_reward += r
        cum_regret += regret
        rows.append({
            "policy": policy.name,
            "step": int(step),
            "uid": str(world.uid[i]),
            "arm": arm,
            "reward": round(r, 6),
            "oracle_reward": round(oracle_r, 6),
            "cum_reward": round(cum_reward, 6),
            "avg_reward": round(cum_reward / (step + 1), 6),
            "cum_regret": round(cum_regret, 6),
            "avg_regret": round(cum_regret / (step + 1), 6),
        })
    return pd.DataFrame(rows, columns=list(ONLINE_COLUMNS))


def run_audit(world, audit_idx: np.ndarray, policy) -> pd.DataFrame:
    """冻结评测：审计池逐人次"只选不学"（greedy，无 update）。

    策略带着在线阶段学到的后验上场；审计用户从未参与任何 update。
    """
    world.reset_revealed()
    rows = []
    for i in audit_idx:
        i = int(i)
        x = world.context_of(i)
        arm = _policy_arm(world, policy, x, i, frozen=True)
        r = world.observe(i, arm)
        oracle_arm, oracle_r = world.oracle_best(i)
        rows.append({
            "policy": policy.name,
            "uid": str(world.uid[i]),
            "arm": arm,
            "reward": round(r, 6),
            "oracle_arm": oracle_arm,
            "oracle_reward": round(oracle_r, 6),
            "regret": round(oracle_r - r, 6),
        })
    return pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))


# ── 汇总（每策略一行）───────────────────────────────────────

def _arm_shares(log: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    """三臂占比（固定顺序输出，未选过的臂记 0）。"""
    counts = log["arm"].value_counts()
    n = len(log)
    return {f"{prefix}share_{arm}": round(float(counts.get(arm, 0)) / n, 6) for arm in ARMS}


def _frac(molecule: float, denominator: float) -> float:
    """oracle 均值为 0（全员最优臂=control 且奖励恰 0）时无法比较，记 NaN。"""
    return round(molecule / denominator, 6) if denominator > 0 else float("nan")


def online_summary(log: pd.DataFrame) -> dict:
    """在线过程汇总：均值 / 头尾 20 步（学习信号）/ 遗憾 / 臂占比。"""
    if len(log) == 0:
        raise ValueError("在线日志为空，无法汇总")
    reward = log["reward"].to_numpy(dtype=float)
    oracle = log["oracle_reward"].to_numpy(dtype=float)
    n = len(log)
    mean_reward = float(reward.mean())
    oracle_mean = float(oracle.mean())
    k = min(20, n)
    return {
        "mean_reward": round(mean_reward, 6),
        "head20": round(float(reward[:k].mean()), 6),
        "tail20": round(float(reward[-k:].mean()), 6),
        "cum_regret": round(float(log["cum_regret"].iloc[-1]), 6),
        "regret_per_step": round(float(log["cum_regret"].iloc[-1]) / n, 6),
        "oracle_mean": round(oracle_mean, 6),
        "reward_vs_oracle_frac": _frac(mean_reward, oracle_mean),
        **_arm_shares(log),
    }


def audit_summary(log: pd.DataFrame) -> dict:
    """审计汇总：冻结得分 / 人均遗憾 / 相对 oracle 上界 / 臂占比。"""
    if len(log) == 0:
        raise ValueError("审计日志为空，无法汇总")
    reward = log["reward"].to_numpy(dtype=float)
    oracle = log["oracle_reward"].to_numpy(dtype=float)
    mean_reward = float(reward.mean())
    oracle_mean = float(oracle.mean())
    return {
        "mean_reward_audit": round(mean_reward, 6),
        "oracle_mean_audit": round(oracle_mean, 6),
        "audit_regret": round(float(log["regret"].mean()), 6),
        "audit_vs_oracle_frac": _frac(mean_reward, oracle_mean),
        **_arm_shares(log, prefix="audit_"),
    }


def summary_table(
    online_logs: dict[str, pd.DataFrame], audit_logs: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """合并成每策略一行宽表（策略顺序 = 传入字典顺序 = POLICIES 规范序）。"""
    rows = []
    for name, olog in online_logs.items():
        row: dict = {"policy": name, "n_online": int(len(olog))}
        row.update(online_summary(olog))
        alog = audit_logs[name]
        row["n_audit"] = int(len(alog))
        row.update(audit_summary(alog))
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
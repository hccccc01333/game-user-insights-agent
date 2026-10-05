# -*- coding: utf-8 -*-
"""S3 Bandit · 部分反馈世界：切分 / 上下文基 / observe 唯一反馈通道。

先说人话：
    真实营销里每个人只被触达一次，选了一条臂就只能看到这条臂的结果，
    其它臂"本会怎样"永远看不到——这就是部分反馈（bandit 的核心约束）。
    本模块把 S1 模拟器的真值表（每人 × 三臂净奖励）封装成一个世界：
      · 先按用户切分：在线池（策略边决策边学）与审计池（冻结评测，不参与学习）；
      · 把四件可观测状态展开成上下文向量（基展开 + z 标准化 + 截距）；
      · 世界只开一个信息口 observe(i, arm)——返回所选臂的净奖励并记账，
        真值矩阵对学习策略不可见（oracle 与评测专用）。

随机流纪律：
    切分随机流由 (种子, 固定盐, 用途) 派生——同参永远同结果；
    策略自己的探索随机流在 policies.py 单独派生（盐不重叠）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 上下文特征（单一真源）：只有四件可观测状态允许进策略（与 S2 同源）
FEATURES: tuple[str, ...] = (
    "act_30d",
    "silence_days",
    "interest_concentration",
    "tenure_days",
)

ARMS: tuple[str, ...] = ("control", "rec", "recall")

# 上下文基：raw 直接线性；logsq 对右偏变量做 log1p 加平方项（探针锁定默认）
BASES: tuple[str, ...] = ("raw", "logsq")
DEFAULT_BASIS = "logsq"

# 协议随机流的用途盐（与模拟器、S2 的流彼此独立）
_SALT = {"split": 301}


def _rng(seed: int, purpose: str) -> np.random.Generator:
    """协议随机流：由 (种子, 固定盐, 用途) 派生——同参永远同结果。"""
    return np.random.default_rng([seed, 20261005, _SALT[purpose]])


def split_online_audit(
    users: pd.DataFrame, audit_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按用户切分在线池 / 审计池：种子洗牌后，前 audit 比例进审计池。

    · 在线池保持洗牌序——即"在线决策顺序"（每人依次到访一次）；
    · 审计池按原始位置升序——冻结评测顺序，与决策顺序无关；
    · 两个池互不相交且并集为全人口（tests 断言）。
    """
    if not (0.0 < audit_frac < 1.0):
        raise ValueError(f"audit_frac 需在 (0,1) 内（当前 {audit_frac}）")
    if len(users) < 50:
        raise ValueError(f"切分至少需要 50 个用户（当前 {len(users)}）")
    rng = _rng(seed, "split")
    order = rng.permutation(len(users))
    n_audit = max(1, int(round(len(users) * audit_frac)))
    n_online = len(users) - n_audit
    if n_online < 10:
        raise ValueError(f"在线池用户过少（{n_online} 人）：调小 audit_frac")
    online = users.iloc[order[n_audit:]].reset_index(drop=True)  # 洗牌序 = 决策顺序
    audit = users.iloc[np.sort(order[:n_audit])].reset_index(drop=True)  # 升序
    return online, audit


# ── 上下文构造（基展开 → 在线池 z 标准化 → 前置截距）────────

def _expand(df: pd.DataFrame, basis: str) -> tuple[np.ndarray, list[str]]:
    """基展开：raw 四列直出；logsq 把右偏的沉默 / 资历做 log1p（沉默平方项）。"""
    act = df["act_30d"].to_numpy(dtype=float)
    silence = df["silence_days"].to_numpy(dtype=float)
    conc = df["interest_concentration"].to_numpy(dtype=float)
    tenure = df["tenure_days"].to_numpy(dtype=float)
    if basis == "raw":
        cols = [act, silence, conc, tenure]
        names = list(FEATURES)
    elif basis == "logsq":
        ls = np.log1p(silence)
        cols = [act, ls, ls * ls, conc, np.log1p(tenure)]
        names = ["act_30d", "log1p_silence", "log1p_silence_sq", "interest_concentration", "log1p_tenure"]
    else:
        raise ValueError(f"未知上下文基：{basis!r}（支持 {BASES}）")
    return np.column_stack(cols), names


def build_context(
    df: pd.DataFrame, basis: str, stats: dict | None = None
) -> tuple[np.ndarray, dict]:
    """把用户帧展开成上下文矩阵（含截距列）；返回 (X, 标准化统计量)。

    stats 为 None 时由传入 df 计算——调用方应传"在线池"（统计口径只用
    在线池）；给审计池构造上下文时必须沿用同一 stats，口径才一致。
    std=0 的列保护为 1（常量列标准化后恒 0），截距列不做标准化。
    """
    expanded, names = _expand(df, basis)
    if stats is None:
        mean = expanded.mean(axis=0)
        std = expanded.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        stats = {"basis": basis, "columns": names, "mean": mean, "std": std}
    z = (expanded - stats["mean"]) / stats["std"]
    X = np.column_stack([np.ones(len(df)), z])
    return X, stats


# ── 部分反馈世界 ────────────────────────────────────────────

class BanditWorld:
    """每人 × 三臂净奖励的真值世界；唯一反馈通道是 observe(i, arm)。

    属性：
        uid / context / rewards（n × 3 真值矩阵，评测与 oracle 专用）/
        oracle_arm / oracle_reward（逐用户最优臂与上界奖励）/
        revealed（已观测 (i, arm) 记录，供测试核对"只看了选过的臂"）
    """

    def __init__(
        self,
        uid: np.ndarray,
        context: np.ndarray,
        rewards: np.ndarray,
        arms: tuple[str, ...] = ARMS,
    ):
        uid = np.asarray(uid)
        context = np.asarray(context, dtype=float)
        rewards = np.asarray(rewards, dtype=float)
        if rewards.ndim != 2 or rewards.shape[1] != len(arms):
            raise ValueError(f"奖励矩阵需为 n × {len(arms)}（当前形状 {rewards.shape}）")
        if context.ndim != 2 or context.shape[0] != rewards.shape[0]:
            raise ValueError("上下文行数需与奖励矩阵一致")
        if len(uid) != rewards.shape[0]:
            raise ValueError("uid 数需与奖励矩阵一致")
        self.uid = uid
        self.context = context
        self.rewards = rewards
        self.arms = tuple(arms)
        best = np.argmax(rewards, axis=1)
        self.oracle_arm = np.array([self.arms[j] for j in best])
        self.oracle_reward = rewards[np.arange(len(uid)), best]
        self.revealed: list[tuple[int, str]] = []

    def _check_index(self, i: int) -> int:
        i = int(i)
        if not (0 <= i < len(self.uid)):
            raise ValueError(f"用户下标越界：{i}（共 {len(self.uid)} 人）")
        return i

    def context_of(self, i: int) -> np.ndarray:
        """第 i 个用户的上下文向量（只读）。"""
        return self.context[self._check_index(i)]

    def observe(self, i: int, arm: str) -> float:
        """★ 唯一反馈通道：返回第 i 个用户在第 arm 条臂下的净奖励并记账。"""
        i = self._check_index(i)
        if arm not in self.arms:
            raise ValueError(f"未知臂：{arm!r}（支持 {self.arms}）")
        self.revealed.append((i, arm))
        return float(self.rewards[i, self.arms.index(arm)])

    def oracle_best(self, i: int) -> tuple[str, float]:
        """上帝视角：第 i 个用户的最优臂与上界奖励（评测与 oracle 策略专用）。"""
        i = self._check_index(i)
        return str(self.oracle_arm[i]), float(self.oracle_reward[i])

    def reset_revealed(self) -> None:
        """清空观测记录（同一世界连跑多个策略时，每个策略单独记账）。"""
        self.revealed.clear()


def build_world(
    tables: dict[str, pd.DataFrame], audit_frac: float, seed: int, basis: str = DEFAULT_BASIS
) -> dict:
    """把模拟器真值表封装成部分反馈世界。

    返回 {"world", "online", "audit", "meta"}：
        world    BanditWorld（全人口；uid 顺序 = 模拟器用户表顺序）
        online   在线决策顺序（世界下标，洗牌序）
        audit    审计评测顺序（世界下标，升序）
        meta     切分 / 基 / 上下文口径 / 人数 / 奖励真值概况（供落盘）
    """
    if basis not in BASES:
        raise ValueError(f"未知上下文基：{basis!r}（支持 {BASES}）")
    users, effects = tables["users"], tables["effects"]
    online_df, audit_df = split_online_audit(users, audit_frac, seed)
    pos = pd.Series(np.arange(len(users)), index=users["uid"].to_numpy())
    online_idx = pos.loc[online_df["uid"].to_numpy()].to_numpy()
    audit_idx = np.sort(pos.loc[audit_df["uid"].to_numpy()].to_numpy())

    pivot = effects.pivot(index="uid", columns="arm", values="reward")
    missing_arms = [a for a in ARMS if a not in pivot.columns]
    if missing_arms:
        raise ValueError(f"真值表缺臂：{missing_arms}（需要 {ARMS}）")
    rewards_df = pivot[list(ARMS)].reindex(users["uid"])
    if rewards_df.isna().any().any():
        raise ValueError("真值表存在缺口：有用户 × 臂没有 reward 行")
    rewards = rewards_df.to_numpy(dtype=float)

    # 上下文：统计量只取在线池；全体用户（含审计池）用同一套统计量
    _, stats = build_context(users.iloc[online_idx], basis)
    context, _ = build_context(users, basis, stats)

    world = BanditWorld(users["uid"].to_numpy(), context, rewards)
    meta = {
        "audit_frac": float(audit_frac),
        "split_seed": int(seed),
        "basis": basis,
        "features": list(FEATURES),
        "arms": list(ARMS),
        "context_width": int(context.shape[1]),
        "context_stats": {
            "columns": list(stats["columns"]),
            "mean": [round(float(v), 6) for v in stats["mean"]],
            "std": [round(float(v), 6) for v in stats["std"]],
        },
        "n_online": int(len(online_idx)),
        "n_audit": int(len(audit_idx)),
        "oracle_mean_reward": round(float(world.oracle_reward.mean()), 6),
        "arm_mean_reward": {
            a: round(float(rewards[:, j].mean()), 6) for j, a in enumerate(ARMS)
        },
    }
    return {"world": world, "online": online_idx, "audit": audit_idx, "meta": meta}
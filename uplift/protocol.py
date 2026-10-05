# -*- coding: utf-8 -*-
"""S2 Uplift · 实验协议：切分 / 随机分配 / 观测标签（rct 与 full 两条口径）。

先说人话：
    模拟器手里有"每个用户 × 每条臂"的完整反事实，但真实世界里只能观测到
    用户实际经历的那一条世界线。本模块把真值表降维成两种训练数据：
      · rct   每人随机分到 1 条臂，只观测这条臂的结果（现实世界的标准做法）
      · full  每人 3 条臂的结果全都看得见（理论上限参照，现实中不存在）
    两条口径跑同一套模型与评测流程，差距就是"反事实不可观测"的代价。

硬规则（反循环论证）：
    · 训练帧只带 uid / arm / y / 四件可观测状态——任何 tau_*、increment、
      reward、cost、a_daily、y_*_window 都不允许进训练帧（tests 断言列集合）；
    · 按用户切分（同一用户不会同时出现在训练与留出），留出集真值只用于评测，
      不参与任何拟合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 特征白名单（单一真源）：只有四件可观测状态允许进模型
FEATURES: tuple[str, ...] = (
    "act_30d",
    "silence_days",
    "interest_concentration",
    "tenure_days",
)

ARMS: tuple[str, ...] = ("control", "rec", "recall")
TREATED_ARMS: tuple[str, ...] = ("rec", "recall")
PROTOCOLS: tuple[str, ...] = ("rct", "full")

# 训练帧 / 留出评测帧的合法列（无泄漏测试以此为断言）
TRAIN_COLUMNS: tuple[str, ...] = ("uid", "arm", "y") + FEATURES
HOLDOUT_COLUMNS: tuple[str, ...] = ("uid", "arm") + FEATURES + (
    "tau_struct", "tau_ind", "increment", "cost", "reward",
)

OUTCOME_VOID = "y_void_window"       # 模拟器真值表里的"不触达"窗内结果
OUTCOME_TREATED = "y_treated_window"  # 模拟器真值表里的"触达后"窗内结果

# 协议随机流的用途盐（与模拟器的流彼此独立）
_SALT = {"split": 101, "assign": 202}


def _rng(seed: int, purpose: str) -> np.random.Generator:
    """协议随机流：由 (种子, 固定盐, 用途) 派生——同参永远同结果。"""
    return np.random.default_rng([seed, 20261005, _SALT[purpose]])


def split_users(
    users: pd.DataFrame, holdout: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按用户切分训练 / 留出：种子洗牌后，前 holdout 比例做留出集。"""
    if not (0.0 < holdout < 1.0):
        raise ValueError(f"holdout 需在 (0,1) 内（当前 {holdout}）")
    if len(users) < 10:
        raise ValueError(f"切分至少需要 10 个用户（当前 {len(users)}）")
    rng = _rng(seed, "split")
    order = rng.permutation(len(users))
    n_hold = max(1, int(round(len(users) * holdout)))
    n_train = len(users) - n_hold
    if n_train < 6:
        raise ValueError(f"训练用户过少（{n_train} 人）：RCT 三臂各需观测")
    train = users.iloc[np.sort(order[n_hold:])].reset_index(drop=True)
    hold = users.iloc[np.sort(order[:n_hold])].reset_index(drop=True)
    return train, hold


def assign_rct(users: pd.DataFrame, seed: int) -> pd.DataFrame:
    """RCT 分配：训练集里每人随机均匀分 1 条臂（现实可执行的标准做法）。"""
    rng = _rng(seed, "assign")
    idx = rng.integers(0, len(ARMS), size=len(users))
    return pd.DataFrame({"uid": users["uid"].to_numpy(), "arm": [ARMS[i] for i in idx]})


def _observed_y(df: pd.DataFrame) -> np.ndarray:
    """观测标签：对照臂看"不触达结果"，触达臂看"触达后结果"。"""
    return np.where(
        df["arm"].to_numpy() == "control",
        df[OUTCOME_VOID].to_numpy(dtype=float),
        df[OUTCOME_TREATED].to_numpy(dtype=float),
    )


def build_observations(
    tables: dict[str, pd.DataFrame], holdout: float, seed: int, protocol: str
) -> dict:
    """把模拟器真值表降维成（训练观测帧, 留出评测帧, 口径元信息）。

    · rct ：训练集每人恰 1 行（随机分配臂 + 该臂观测结果）
    · full：训练集每人 3 行（对照用 y_void，两条触达臂用各自 y_treated）
    留出集：每用户 × 每臂评测帧（含真值 tau_struct / tau_ind / increment /
    cost / reward），只用于判分。
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"协议只支持 {PROTOCOLS}（当前 {protocol!r}）")
    users, effects = tables["users"], tables["effects"]
    train_users, hold_users = split_users(users, holdout, seed)

    keep = ["uid", "arm", OUTCOME_VOID, OUTCOME_TREATED]
    if protocol == "rct":
        obs = assign_rct(train_users, seed).merge(
            effects[keep], on=["uid", "arm"], how="left", validate="one_to_one"
        )
    else:  # full：训练用户 × 全部臂
        cross = train_users[["uid"]].assign(_k=1).merge(
            pd.DataFrame({"arm": list(ARMS), "_k": 1}), on="_k"
        ).drop(columns="_k")
        obs = cross.merge(effects[keep], on=["uid", "arm"], how="left", validate="one_to_one")
    obs["y"] = _observed_y(obs)
    if obs["y"].isna().any():  # pragma: no cover - 防御：模拟器真值表不完整
        raise ValueError("模拟器真值表缺少对应 (uid, arm) 行，无法构造观测标签")

    train = obs[["uid", "arm", "y"]].merge(
        train_users[["uid", *FEATURES]], on="uid", how="left", validate="many_to_one"
    )
    train = train[list(TRAIN_COLUMNS)]

    hold = hold_users[["uid", *FEATURES]].merge(
        effects[["uid", "arm", "tau_struct", "tau_ind", "increment", "cost", "reward"]],
        on="uid",
        how="inner",
        validate="one_to_many",
    )
    hold = hold[list(HOLDOUT_COLUMNS)]

    meta = {
        "protocol": protocol,
        "holdout": holdout,
        "split_seed": seed,
        "n_train_users": int(len(train_users)),
        "n_holdout_users": int(len(hold_users)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(hold)),
        "train_arm_counts": {arm: int((train["arm"] == arm).sum()) for arm in ARMS},
        "features": list(FEATURES),
    }
    return {"train": train, "holdout": hold, "meta": meta}
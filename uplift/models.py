# -*- coding: utf-8 -*-
"""S2 Uplift · 学习器：T-learner（HistGBM 主 + Ridge 线性基线）。

先说人话：
    要估"给这个人上这条臂，比不触达多出多少"，最直接的做法是分别学两个
    结果模型——"不触达时他会做多少"（f_control）与"触达时会做多少"（f_arm），
    两者相减就是干预效应 τ̂。这就是 T-learner：结构简单，且当干预会改变
    结果分布时比 S-learner 更稳。

    两个学习器：
      · hgb   梯度提升树（HistGBM）：能学非线性与交互（召回的钟形、门槛效应）
      · ridge 线性 + 标准化：低方差基线，用来对照"树模型到底带来了什么"

硬规则：fit 只接收 protocol.build_observations 产出的训练帧（无真值列）；
τ̂ 永远是 f_arm − f_control，对照臂恒 0（对照无干预效应）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .protocol import ARMS, FEATURES, TREATED_ARMS

LEARNERS: tuple[str, ...] = ("hgb", "ridge")

# 主学习器超参（设计取值：小深度 + 叶下限 + L2，抑制对个体噪声的过拟合）
HGB_PARAMS = dict(
    max_iter=400,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
)
RIDGE_ALPHA = 1.0


def make_regressor(learner: str, seed: int):
    """按名字造一个结果模型（种子只影响 hgb；ridge 本身确定）。"""
    if learner == "hgb":
        return HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS)
    if learner == "ridge":
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])
    raise ValueError(f"未知学习器：{learner!r}（支持 {LEARNERS}）")


def hyperparams_snapshot(learners: tuple[str, ...] | list[str]) -> dict:
    """给 manifest 的超参快照（不依赖已拟合对象，纯配置复述）。"""
    snap = {}
    for learner in learners:
        if learner == "hgb":
            snap["hgb"] = {**HGB_PARAMS, "random_state": "run_seed"}
        elif learner == "ridge":
            snap["ridge"] = {"alpha": RIDGE_ALPHA, "scaler": "StandardScaler"}
        else:
            raise ValueError(f"未知学习器：{learner!r}（支持 {LEARNERS}）")
    return snap


class TLearnerUplift:
    """T-learner：每臂一个结果模型；τ̂(x, arm) = f_arm(x) − f_control(x)。"""

    def __init__(self, learner: str, seed: int):
        if learner not in LEARNERS:
            raise ValueError(f"未知学习器：{learner!r}（支持 {LEARNERS}）")
        self.learner = learner
        self.seed = seed
        self.models: dict[str, object] = {}

    def fit(self, train: pd.DataFrame) -> "TLearnerUplift":
        """只吃协议帧（uid / arm / y / 特征白名单），每臂单独拟合。"""
        missing = [c for c in ("arm", "y", *FEATURES) if c not in train.columns]
        if missing:
            raise ValueError(f"训练帧缺少列：{missing}")
        for arm in ARMS:
            sub = train[train["arm"] == arm]
            if sub.empty:
                raise ValueError(f"训练集中臂 {arm!r} 无观测（RCT 分配样本不足？）")
            model = make_regressor(self.learner, self.seed)
            model.fit(sub.loc[:, list(FEATURES)], sub["y"].to_numpy(dtype=float))
            self.models[arm] = model
        return self

    def predict_arm(self, x: pd.DataFrame, arm: str) -> np.ndarray:
        """预测"该臂之下会做多少"（结果面，不是效应面）。"""
        if arm not in self.models:
            raise ValueError(f"模型未拟合或臂未知：{arm!r}")
        return self.models[arm].predict(x.loc[:, list(FEATURES)])

    def predict_tau(self, x: pd.DataFrame, arm: str) -> np.ndarray:
        """τ̂ = f_arm − f_control；对照臂恒 0（模型输入只取四件可观测状态）。"""
        if arm == "control":
            return np.zeros(len(x))
        if arm not in TREATED_ARMS:
            raise ValueError(f"未知臂：{arm!r}（支持 {TREATED_ARMS}）")
        return self.predict_arm(x, arm) - self.predict_arm(x, "control")
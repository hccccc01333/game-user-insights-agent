# -*- coding: utf-8 -*-
"""沉默预测 · 分类器工厂（超参单一真源 + 快照）。

两个挑战者（对照 = heuristic gap_days 规则，见 run_silence.py）：
    logreg  StandardScaler + LogisticRegression：线性基线，概率天然贴近基准率
    hgb     HistGradientBoostingClassifier：主力，能学非线性与交互

与可行性快检的一个有意差异：logreg **不加 class_weight**——概率直接反映基准率
（Brier / 校准曲线可读）；class_weight 只改概率缩放，不改排序，AUC 不受影响。
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LEARNERS = ("logreg", "hgb")

LOGREG_PARAMS = {"max_iter": 2000}
HGB_PARAMS = {
    "learning_rate": 0.06,
    "max_iter": 400,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def make_classifier(name: str, seed: int = 7):
    """按名字建分类器（未拟合）；未知名显式报错。"""
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=seed, **LOGREG_PARAMS)),
            ]
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS)
    raise ValueError(f"未知学习器：{name!r}（支持 {LEARNERS}）")


def hyperparams_snapshot(names: tuple[str, ...]) -> dict:
    """超参快照（进 _manifest，冻结"当时跑的是什么模型"）。"""
    snap: dict = {}
    for name in names:
        if name == "logreg":
            snap[name] = {
                "class": "Pipeline(StandardScaler, LogisticRegression)",
                "class_weight": None,
                **LOGREG_PARAMS,
            }
        elif name == "hgb":
            snap[name] = {"class": "HistGradientBoostingClassifier", **HGB_PARAMS}
        else:
            raise ValueError(f"未知学习器：{name!r}（支持 {LEARNERS}）")
    return snap
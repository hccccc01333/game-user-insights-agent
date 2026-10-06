# -*- coding: utf-8 -*-
"""沉默预测 · 评测：指标 / 分位校准 / ΔAUC 显著性（聚类 bootstrap）。

指标口径：
    roc_auc         排序判别力（主指标）
    pr_auc          average precision（沉默是不均衡类时更敏感）
    recall_top      Top-k%（k=round(frac·n)）内的沉默召回
    precision_top   同上命中率
    brier           概率校准误差（仅概率分数；heuristic 规则分不上）
    calibration_table  分位十等分：mean_score vs observed_rate

显著性：
    cluster_bootstrap_auc_diff  按用户（uid）聚类重采样，估计 ΔAUC 的 95% 区间；
    同一用户在面板里有多行（自相关），按行重采样会低估方差，必须按 uid 聚类。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CALIBRATION_COLUMNS = ("bin", "n", "score_low", "score_high", "mean_score", "observed_rate")


def top_k(n: int, frac: float) -> int:
    """Top-k 名额：k = max(1, round(frac·n))。"""
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"top_frac 需在 (0,1] 内（当前 {frac}）")
    return max(1, int(round(frac * n)))


def _check_two_classes(y) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("指标需要两类标签（当前单一类别）")
    return y


def roc_auc(y, score) -> float:
    return float(roc_auc_score(_check_two_classes(y), np.asarray(score, dtype=float)))


def pr_auc(y, score) -> float:
    return float(average_precision_score(_check_two_classes(y), np.asarray(score, dtype=float)))


def _top_idx(score, frac: float) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    k = top_k(len(score), frac)
    return np.argsort(-score, kind="stable")[:k]


def recall_at_top(y, score, frac: float = 0.10) -> float:
    """Top-k% 内命中沉默用户的比例（分母 = 全体沉默数）。"""
    y = np.asarray(y, dtype=int)
    idx = _top_idx(score, frac)
    n_pos = int(y.sum())
    return float(y[idx].sum() / n_pos) if n_pos else float("nan")


def precision_at_top(y, score, frac: float = 0.10) -> float:
    """Top-k% 内的沉默命中率。"""
    y = np.asarray(y, dtype=int)
    return float(y[_top_idx(score, frac)].mean())


def brier(y, prob) -> float:
    """Brier 分数：mean((p − y)²)；仅对概率分数计算。"""
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    return float(np.mean((prob - y) ** 2))


def metrics(y, score, *, probabilistic: bool, top_frac: float = 0.10) -> dict:
    """一组 (y, score) 的指标汇总；probabilistic=False 时 brier=None（规则分非概率）。"""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score):
        raise ValueError(f"y 与 score 长度不一致（{len(y)} vs {len(score)}）")
    if len(np.unique(y)) < 2:
        raise ValueError("指标需要两类标签（当前单一类别）")
    out = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "roc_auc": roc_auc(y, score),
        "pr_auc": pr_auc(y, score),
        "recall_top": recall_at_top(y, score, top_frac),
        "precision_top": precision_at_top(y, score, top_frac),
        "top_frac": float(top_frac),
        "brier": brier(y, score) if probabilistic else None,
    }
    return out


def calibration_table(y, score, bins: int = 10) -> pd.DataFrame:
    """分位校准表：按 score 的分位边界分箱，比 mean_score 与 observed_rate。"""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score):
        raise ValueError("y 与 score 长度不一致")
    if len(y) == 0:
        raise ValueError("校准表需要非空样本")
    edges = np.unique(np.quantile(score, np.linspace(0.0, 1.0, bins + 1)))
    labels = (
        np.zeros(len(y), dtype=int)
        if len(edges) < 3  # 分数近乎常数 → 单箱
        else np.searchsorted(edges[1:-1], score, side="right")
    )
    rows: list[dict] = []
    for b in range(int(labels.max()) + 1):
        m = labels == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "score_low": float(score[m].min()),
                "score_high": float(score[m].max()),
                "mean_score": float(score[m].mean()),
                "observed_rate": float(y[m].mean()),
            }
        )
    return pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))


def cluster_bootstrap_auc_diff(
    y, score_a, score_b, groups, n_boot: int = 1000, seed: int = 7
) -> dict:
    """ΔAUC = AUC(a) − AUC(b) 的按用户聚类 bootstrap 区间。

    groups 为行级 uid 数组（同一用户的多行必须同进同出）。
    返回 delta / ci_low / ci_high（95%）/ frac_positive / n_boot_effective。
    """
    y = np.asarray(y, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    groups = np.asarray(groups)
    if not (len(y) == len(score_a) == len(score_b) == len(groups)):
        raise ValueError("y / score_a / score_b / groups 长度不一致")
    if len(np.unique(y)) < 2:
        raise ValueError("Bootstrap 需要两类标签（当前单一类别）")
    if n_boot < 1:
        raise ValueError(f"n_boot 需为正整数（当前 {n_boot}）")

    uniq, inv = np.unique(groups, return_inverse=True)
    index_by_group = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    delta = roc_auc(y, score_a) - roc_auc(y, score_b)

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([index_by_group[i] for i in pick])
        y_b = y[idx]
        if y_b.min() == y_b.max():
            continue
        deltas.append(roc_auc(y_b, score_a[idx]) - roc_auc(y_b, score_b[idx]))
    if not deltas:
        raise ValueError("Bootstrap 无有效重采样（样本类别过少）")
    arr = np.asarray(deltas)
    return {
        "delta": float(delta),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "frac_positive": float(np.mean(arr > 0)),
        "n_boot": int(n_boot),
        "n_boot_effective": int(len(arr)),
    }
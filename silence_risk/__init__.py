# -*- coding: utf-8 -*-
"""M1 · ML 沉默预测（silence_risk v1）：公开行为痕上的「未来沉默风险」建模与验收。

口径名：Future Public Inactivity Prediction（为什么不用 churn：公开时间线只能观测
"还有没有公开活动"，观测不到真实流失）。

模块分工：
    panel.py       (用户, 锚点 T) 面板构造：锚点推进 / 标签 / 右截断保护 / 左截断标记
    features.py    多窗计数主干（n7..n180 + gap_days）+ 180d 类型占比（FEATURE_SETS 单一真源）
    models.py      分类器工厂（logreg / hgb，超参单一真源 + 快照）
    evaluate.py    指标（AUC / PR-AUC / Recall@Top / Precision@Top / Brier）+ 分位校准
                   + 聚类 bootstrap 的 ΔAUC 显著性
    run_silence.py CLI：双协议评测（group_cv / time_extrap）、落盘、_manifest、双跑校验
    tests.py       冒烟与一致性测试（手算面板 / 特征 / 指标端点 / 协议完整性 / 无泄漏 / 边界）

M1 达标线（唯一验收口径）：时间外推（训旧段 → 测新段）**任一挑战者**的 AUC
显著高于 recency 基线（heuristic = gap_days）；同期拆分 / 训练集 AUC 不算数。
"""

SILENCE_VERSION = "silencev1"
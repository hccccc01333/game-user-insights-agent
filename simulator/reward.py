# -*- coding: utf-8 -*-
"""S1 模拟器 · 奖励契约（单一真源）。

先说人话：
    一次触达值不值，用「未来 N 天多出来的活跃事件 − 触达成本」衡量。
    增量与成本用同一单位——"事件当量"（1 ≈ 一条公开互动行为），奖励可正可负，
    下游（Bandit / 策略评估）直接看它，不再自己发明口径。

    reward(u, arm) = increment_window(u, arm) − cost(arm)
    · increment_window = Σ 触达轨迹前 N 天事件 − Σ 未触达轨迹前 N 天事件
      （同一用户、同一随机数串配对，见 env.py 的 CRN 说明）
    · cost：对照 0 / 推荐 0.5 / 召回 1.0（设计取值，无公开数据可校准；
      改 params.cost_events 即可，改完全链路生效）

边界（v0 已知简化，README 同步声明）：
    · 只计 N 天窗内事件；窗外只做诊断（sim_daily.csv 的衰减曲线），不计奖励；
    · 除成本外不建模负效应（打扰 / 取关 / 口碑损失）——奖励框架已留位，
      等有数据可校准时再进 v1。
"""
from __future__ import annotations

from collections.abc import Sequence

from .params import SIM, SimParams


def marginal_cost(arm: str, p: SimParams = SIM) -> float:
    """该臂的触达成本（事件当量）。"""
    if arm not in p.cost_events:
        raise ValueError(f"未知臂：{arm}（可选 {tuple(p.cost_events)}）")
    return float(p.cost_events[arm])


def window_increment(treated: Sequence[float], void: Sequence[float], window: int) -> float:
    """奖励窗内的配对增量：Σ(触达 − 未触达)。两条轨迹必须是同一用户的配对 rollout。"""
    if window < 1:
        raise ValueError(f"窗口需 ≥ 1 天（当前 {window}）")
    if len(treated) < window or len(void) < window:
        raise ValueError(f"轨迹长度不足窗口：len(treated)={len(treated)}, len(void)={len(void)}, window={window}")
    return float(sum(treated[:window]) - sum(void[:window]))


def reward(increment_value: float, arm: str, p: SimParams = SIM) -> float:
    """净奖励 = 窗内配对增量 − 触达成本（可正可负）。"""
    return float(increment_value) - marginal_cost(arm, p)
# -*- coding: utf-8 -*-
"""S1 模拟器 · 效应模型（真实 CATE 的来源）。

先说人话：
    三个臂里，对照臂没有效应；推荐臂与召回臂各有一套"对谁有效、效果多大"的
    公式。模拟器手握这套公式 = 手握真实 CATE（条件平均处理效应）——下游 Uplift
    模型学得准不准，拿这里的真值一对照就知道。没有真值靶子，因果建模无从验收。

公式（τ = 未来奖励窗内的活跃增量期望，单位 = 事件数）：

    个性化内容推荐  τ_rec = A_rec · c_eff · g(r) · f(s)
        c_eff = clip(c / conc_ref, 0.2, 1.2)    兴趣越集中，推荐越有的放矢
        g(r)  = sqrt(clip(r / rec_act_full, 0, 1))  近 30 天越活跃响应越强（次线性；
               r=0 严格为零——对完全沉默者不做内容推荐，那属于召回臂的活）
        f(s)  = exp(-s / rec_fresh_days)        沉默越久越难唤起

    沉默用户召回    τ_recall = A_recall · hump(s) · t(T)
        hump(s) = exp(-(ln(s / peak))² / (2σ²)) 对数钟形：沉默 peak 天上下最可召回，
                 太嫩的（刚活跃过）与太久远的（>1.5 年）两端衰减
        t(T)    = clip(ref / (T + shift), 0.4, 1.25)  资历折减：新账号更易召回

    疲劳：同一臂第 n 次触达，效应乘 ρ^(n-1)（参数 fatigue_rho）
    噪声：个体效应 τ_i = τ · exp(ν − σ²/2)，ν ~ N(0, σ²)
          （对数正态：中位数 ≈ 结构值 τ，期望仍为 τ；σ=0 时全部同质）

职责边界：本模块只实现公式、纯函数（不碰随机数生成器，噪声由 env 按用户流
提供）；参数与来源标注见 params.py；量级与单调性的验收见 tests.py。
"""
from __future__ import annotations

import math

from .params import SIM, SimParams
from .user_generator import SimUser


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def tau_rec(user: SimUser, p: SimParams = SIM) -> float:
    """个性化内容推荐的效应（结构值，未含个体噪声与疲劳）。"""
    c_eff = _clip(user.interest_concentration / p.conc_ref, *p.conc_clip)
    g = math.sqrt(_clip(user.act_30d / p.rec_act_full, 0.0, 1.0))
    fresh = math.exp(-max(user.silence_days, 0.0) / p.rec_fresh_days)
    return p.a_rec * c_eff * g * fresh


def tau_recall(user: SimUser, p: SimParams = SIM) -> float:
    """沉默用户召回的效应（结构值）。"""
    s = max(user.silence_days, 0.5)  # 下界保护：ln(0) 无定义，hump(0)≈0
    hump = math.exp(-(math.log(s / p.recall_peak_days) ** 2) / (2.0 * p.recall_sigma_ln ** 2))
    t_eff = _clip(p.recall_tenure_ref / (user.tenure_days + p.recall_tenure_shift), *p.recall_tenure_clip)
    return p.a_recall * hump * t_eff


def structural_effect(user: SimUser, arm: str, p: SimParams = SIM) -> float:
    """真实 CATE 的结构部分 τ(x)：只依赖可观测状态 x，不依赖个体噪声。"""
    if arm == "control":
        return 0.0
    if arm == "rec":
        return tau_rec(user, p)
    if arm == "recall":
        return tau_recall(user, p)
    raise ValueError(f"未知臂：{arm}（可选 {p.arms}）")


def individual_effect(user: SimUser, arm: str, noise_ln: float, p: SimParams = SIM) -> float:
    """个体真实效应 τ_i：结构值 × 乘性噪声（对数正态，中位数≈结构值）。

    `noise_ln` 由调用方（env）按用户随机流提供 ν ~ N(0, σ²)——本模块保持纯函数，
    同一 (user, arm, ν) 永远得到同一结果，便于测试与复现。
    """
    tau = structural_effect(user, arm, p)
    if tau <= 0.0:
        return 0.0
    return tau * math.exp(noise_ln - 0.5 * p.effect_noise_sigma ** 2)


def day_weights(window: int, decay: float) -> list[float]:
    """效应在窗内的日权重（∝ decay^d，归一化到 Σ=1）。"""
    if window < 1:
        raise ValueError(f"窗口需 ≥ 1 天（当前 {window}）")
    raw = [decay ** d for d in range(window)]
    total = sum(raw)
    return [x / total for x in raw]


def effect_schedule(
    user: SimUser,
    actions: list[tuple[int, str]],
    days: int,
    noise_by_arm: dict[str, float],
    window: int | None = None,
    p: SimParams = SIM,
) -> list[float]:
    """把触达计划折算成"逐日活跃概率增量" Δa（已封顶）。

    actions：[(触达日, 臂)]，按时间先后传入（疲劳计数依赖传入顺序）；
    同一天可多次触达（各自独立计疲劳）。
    """
    w = window if window is not None else p.reward_window_days
    weights = day_weights(w, p.effect_daily_decay)
    delta = [0.0] * days
    seen: dict[str, int] = {}
    for day, arm in actions:
        if arm == "control":
            continue
        n = seen.get(arm, 0) + 1
        seen[arm] = n
        tau = individual_effect(user, arm, noise_by_arm.get(arm, 0.0), p) * (p.fatigue_rho ** (n - 1))
        for k in range(min(w, days - day)):
            # τ（事件）÷ 活跃日平均条数 → 概率当量；再乘当日权重
            delta[day + k] += tau * weights[k] / p.count_mean_active
    room = max(p.max_active_prob - user.a_daily, 0.0)  # 概率余量：Δa 不越界
    return [min(d, room) for d in delta]
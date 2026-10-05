# -*- coding: utf-8 -*-
"""S1 模拟器 · 环境（日步长 + 配对 rollout）。

先说人话：
    环境只干一件事：给一个用户和一份"哪天对他做什么"的触达计划，按天推进，
    返回每天的事件数。同一个用户的两次 rollout（触达 / 不触达）共用同一串
    随机数——共同随机数（CRN, common random numbers）——增量里就只剩干预
    本身造成的差异，运气被配对消掉。这等价于"同一个人的反事实"，是 A/B
    实验在模拟世界里的替身。

    为什么要有日循环：策略不是一次性的——召回第二天要不要再推、推荐推几次
    会疲劳，都发生在日粒度上。Agent（harness）后续做策略迭代时逐日调用它即可，
    这个 env 就是 Bandit / 决策循环将来用的沙盒。

每日推进规则（固定消耗 3 个随机数，保证跨臂逐日对齐）：
    u_act < a_base + Δa   → 当天活跃：条数 K ~ 校准混合分布
                              （恰 1 条 w.p. 76%；否则 2+Geom，见 count_from_draws）
    u_act ≥ a_base + Δa   → 当天不活跃：0 条
    （无论活跃与否都消耗 u_cnt1 / u_cnt2，保证配对轨迹的随机数逐日同序）

确定性：每个用户两条独立随机流（行为流 / 响应噪声流），都由
SeedSequence([seed, 流号, 用户下标]) 派生——同 seed 同用户永远同轨迹；
在人口末尾追加新用户不会扰动已有用户的轨迹。
"""
from __future__ import annotations

import math

import numpy as np

from .params import SIM, SimParams
from .response import effect_schedule
from .user_generator import SimUser

# 随机流编号（与 user_generator.STREAM_POP=0 对齐，互不串扰）
STREAM_ACTIVITY = 1   # 行为推进：活跃判定 + 条数
STREAM_RESPONSE = 2   # 个体效应噪声 ν


def count_from_draws(u1: float, u2: float, p: SimParams = SIM) -> int:
    """由两个均匀随机数决定"活跃日条数"（纯函数，唯一实现）。

    校准混合分布：恰 1 条 w.p. count_p1，否则 2 + Geom(count_tail_p)
    （尾部从 2 起，否则"1+Geom"会让恰 1 条占比虚高）。
    校准口径（calibration/reddit_params.csv）：恰 1 条 76%、p90=3、p99=9。
    """
    if u1 < p.count_p1:
        return 1
    geo = int(math.floor(math.log1p(-u2) / math.log1p(-p.count_tail_p)))
    return min(2 + geo, p.count_cap)


def sample_count(rng: np.random.Generator, p: SimParams = SIM) -> int:
    """从随机流抽一个活跃日条数（校准测试 / 分布验证用）。"""
    return count_from_draws(rng.random(), rng.random(), p)


class SimEnv:
    """日步长模拟环境：用户 × 触达计划 → 每日事件数（配对 rollout 的载体）。"""

    def __init__(self, users: list[SimUser], seed: int, params: SimParams = SIM) -> None:
        if not users:
            raise ValueError("用户列表为空")
        self.users = users
        self.seed = int(seed)
        self.params = params

    # ── 随机流 ──────────────────────────────────────────────

    def _activity_rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, STREAM_ACTIVITY, index]))

    def noise_by_arm(self, index: int) -> dict[str, float]:
        """该用户的个体效应噪声 ν（每臂一个，固定不变——个体响应是稳定属性）。"""
        p = self.params
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, STREAM_RESPONSE, index]))
        return {arm: float(rng.normal(0.0, p.effect_noise_sigma)) for arm in ("rec", "recall")}

    # ── rollout ─────────────────────────────────────────────

    def rollout(
        self,
        index: int,
        actions: list[tuple[int, str]],
        days: int,
        window: int | None = None,
    ) -> list[int]:
        """推进一名用户 `days` 天，返回逐日事件数。

        actions：[(触达日, 臂)]，触达日为 0 基天序号；同一计划内按传入顺序计疲劳。
        window：效应分配窗口（默认取 params.reward_window_days）。
        同一 index 的多次调用共用同一随机数串 → 可做配对（CRN）。
        """
        p = self.params
        if days < 1:
            raise ValueError(f"天数需 ≥ 1（当前 {days}）")
        user = self.users[index]
        for day, arm in actions:
            if arm not in p.arms:
                raise ValueError(f"未知臂：{arm}（可选 {p.arms}）")
            if not (0 <= day < days):
                raise ValueError(f"触达日越界：day={day}，有效范围 [0, {days - 1}]")

        schedule = effect_schedule(user, actions, days, self.noise_by_arm(index), window, p)
        rng = self._activity_rng(index)
        events: list[int] = []
        for d in range(days):
            a = min(max(user.a_daily + schedule[d], 0.0), p.max_active_prob)
            u_act = rng.random()
            u_cnt1 = rng.random()   # 固定消耗：无论是否活跃都抽同样两个数，
            u_cnt2 = rng.random()   # 保证配对轨迹的随机数逐日同序（CRN 对齐）
            if u_act < a:
                events.append(count_from_draws(u_cnt1, u_cnt2, p))
            else:
                events.append(0)
        return events

    def baseline_rollout(self, index: int, days: int) -> list[int]:
        """未触达轨迹（对照臂 / 反事实基线）。"""
        return self.rollout(index, [], days)
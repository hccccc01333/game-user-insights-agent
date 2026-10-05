# -*- coding: utf-8 -*-
"""S1 用户模拟器 · 参数与口径（单一真源）。

先说人话：
    这里是模拟世界的"宪法"——有哪些实验组、用户长什么样、干预效果多大、
    触达成本怎么算，全部只在本文件定义；其它模块只读，不写死数字。
    改口径 = 改这里 + 升 SIM_VERSION，然后重跑。

证据分级（每条参数的来源都在 README 参数表里逐条标注）：
    · 校准值 calibrated：有公开统计支撑。锚点来自 calibration/reddit_params.csv
      ——Reddit 游戏社区（gaming / truegaming）日评论形态：活跃日恰 1 条占
      76%、p90≈3、p99≈9。
    · 设计取值 designed：没有数据可校准的部分——效应强度与形状、疲劳、
      成本、默认人口混合。量级参照"单次触达的周增量 ≈ 一天活跃产出"。
    · 派生值 derived：由上述两类按公式推出（如活跃日平均条数 count_mean_active）。

边界（与 README §边界 一致）：
    · 本模拟器产出纯合成用户（uid 前缀 sim_），不复制任何真实用户记录；
    · 无监督标签、无负效应（打扰/取关）、无网络效应——v0 已知简化，README 列明。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

# 模拟器版本：换口径（效应公式 / 分布 / 成本）必须升版本
SIM_VERSION = "s1v0"

# 三个实验臂（顺序即产出列顺序/遍历顺序）
ARM_LABELS = {
    "control": "对照（不触达）",
    "rec": "个性化内容推荐",
    "recall": "沉默用户召回提醒",
}


@dataclass(frozen=True)
class SimParams:
    """一次模拟的全部可调参数（默认值 = 本文件定稿口径，测试可局部覆盖）。"""

    # ── 实验设计 ────────────────────────────────────────────
    arms: tuple[str, ...] = ("control", "rec", "recall")
    # 奖励窗口：与社区周留存锚点同尺度（设计取值）
    reward_window_days: int = 7
    # 触达成本（事件当量：1 ≈ 一条公开互动行为；设计取值，无公开数据可校准）
    cost_events: dict = field(
        default_factory=lambda: {"control": 0.0, "rec": 0.5, "recall": 1.0}
    )

    # ── 日活跃条数采样（校准 + 设计）────────────────────────
    # 口径：给定"当天活跃"，条数 K ~ { 恰 1 条 w.p. p1；否则 2+Geom(tail_p) }
    # （尾部从 2 起，否则"1+Geom"会把恰 1 条占比抬高到 p1+(1-p1)·p）
    count_p1: float = 0.76        # 校准：活跃日恰 1 条占比 76%（Reddit gaming）
    count_tail_p: float = 0.36    # 设计：几何尾（0.36 → p90=3、p99=9 对齐锚点）
    count_cap: int = 200          # 设计：单日封顶（锚点观察极值 146 的量级保护）

    # ── 效应：个性化内容推荐（设计取值）─────────────────────
    a_rec: float = 2.5            # 满档周增量（事件）
    rec_act_full: float = 6.0     # act_30d 达到该值记满档；门槛内 sqrt 次线性
    rec_fresh_days: float = 60.0  # 新鲜度衰减尺度：exp(-silence/60)
    conc_ref: float = 0.12        # 兴趣集中度参考值（对齐 L2 观测形状的取整）
    conc_clip: tuple[float, float] = (0.2, 1.2)

    # ── 效应：沉默用户召回（设计取值）───────────────────────
    a_recall: float = 3.0         # 峰值周增量（事件）
    recall_peak_days: float = 90.0  # 对数钟形峰值：与"沉默中位约 81 天"形态衔接
    recall_sigma_ln: float = 1.2    # 对数尺度宽度（1.2 → 过久沉默（>1.5 年）衰减到近 0）
    recall_tenure_ref: float = 1500.0   # 资历折减：ref/(tenure+shift)
    recall_tenure_shift: float = 250.0
    recall_tenure_clip: tuple[float, float] = (0.4, 1.25)

    # ── 效应通用（设计取值）─────────────────────────────────
    effect_daily_decay: float = 0.7   # 效应在窗内的日衰减（权重 ∝ decay^d，归一化）
    effect_noise_sigma: float = 0.35  # 个体乘性噪声 σ（对数正态；0 = 全同质）
    fatigue_rho: float = 0.5          # 同一臂第 n 次触达乘 ρ^(n-1)
    max_active_prob: float = 0.98     # 活跃概率上限保护（Δa 封顶）

    # ── 用户生成：默认分布（设计取值；仅在无 L2 特征表时使用）──
    # (档位, 占比, act_30d 下/上, a_daily 下/上, 沉默天数 下/上)
    # 口径对齐说明：silent 档 act_30d∈[0,0.45) 取整后为 0，叠加 low+active 共 12%
    # ≈ 已发表口径 "act_30d>0 约 11.5%"；沉默分布取更宽的先验（中位约 2–4 个月），
    # 不逐点复刻本地紧分布（那是本地观测形状，fit 路径负责对齐）。
    pop_mix: tuple[tuple, ...] = (
        ("dormant", 0.55, 0.0, 0.0, 0.0010, 0.0040, 60.0, 720.0),
        ("silent", 0.33, 0.0, 0.45, 0.0040, 0.0200, 35.0, 150.0),
        ("low", 0.09, 1.0, 6.0, 0.0250, 0.1400, 7.0, 70.0),
        ("active", 0.03, 6.0, 60.0, 0.1400, 0.5500, 0.5, 21.0),
    )
    # 兴趣集中度默认：base + span × Beta(a, b)（0.03 + 0.27×Beta(2,3)，中位≈0.13）
    default_conc: tuple[float, float, float, float] = (0.03, 0.27, 2.0, 3.0)
    # 资历默认：对数正态（中位 1100 天），截断 [60, 4200]
    default_tenure_lognorm: tuple[float, float] = (math.log(1100.0), 0.85)
    default_tenure_clip: tuple[float, float] = (60.0, 4200.0)
    # fit 路径中 act_30d=0 用户的节奏带（很低但非零——偶发冒泡）
    quiet_a_band: tuple[float, float] = (0.0008, 0.0050)
    # act_30d>0 用户由事件数反推节奏时的乘性抖动
    act_a_jitter: tuple[float, float] = (0.85, 1.15)
    a_min_if_active: float = 0.004   # act_30d>0 用户节奏下限
    a_ceiling: float = 0.6           # 节奏上限（个体日活跃概率封顶）

    @property
    def count_mean_active(self) -> float:
        """派生：活跃日平均条数 = p1·1 + (1-p1)·(2 + E[Geom]) = 1.667（默认口径）。"""
        geo_mean = (1.0 - self.count_tail_p) / self.count_tail_p
        return self.count_p1 * 1.0 + (1.0 - self.count_p1) * (2.0 + geo_mean)

    def snapshot(self) -> dict:
        """给 manifest 的参数快照（含派生值）。"""
        snap = asdict(self)
        snap["count_mean_active"] = round(self.count_mean_active, 4)
        return snap


# 默认参数实例：其它模块的默认引用（`from .params import SIM`）
SIM = SimParams()
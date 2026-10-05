# -*- coding: utf-8 -*-
"""critic.py —— S4 决策安全层（Critic）：保守下界门 + 循环级校验。

先说人话：
    学习策略（Actor）天生乐观——选臂时用"上界"打分，不确定的方向先试一试。
    但业务里一次触达有成本、有打扰风险，"试错"应当被限制在可控额度内。
    Critic 就是这层安全门：它与 Actor 看到的是同一份信息（只经 world.observe
    返回的净奖励，绝不看真值矩阵），但做相反方向的事：
      · Actor 用乐观上界（UCB）提出臂；
      · Critic 用镜像后验算保守下界（LCB）——若所选臂连悲观估计都不如对照，
        则否决并降级 control（宁不打扰）；
      · 先看清再开价值门（证据门槛 min_obs）：单次观测会让后验下界"过度自信"，
        所以每条臂至少被观测过 min_obs 次，价值门才生效；否则只能走探索额度；
      · 每批保留少量"探索额度"：证据不足时允许小额度试错——否则被否决的臂
        永远拿不到观测，Critic 的谨慎会把学习饿死；额度用尽后严格按门把关。
    镜像后验意味着 Critic 和 Actor 一样对"未观测臂"一无所知：零泄漏，
    "带 / 不带 Critic"两条路径可直接对照。

    另外，Critic 兼任循环级校验（CriticVerifier）：对六环节产物做结构核对，
    并确认"收工"提名前干预环节确实完成（防"没做事就收工"）。

边界：保守门规则、额度口径、六环节字段契约都在本文件集中定义（S4 定案）；
换口径 = 升 HARNESS_VERSION 并重跑。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .verifier import Verifier

# Critic 默认口径（设计取值）
DEFAULT_BETA = 1.0            # 保守系数：下界 = 后验均值 − beta × 后验标准差
DEFAULT_MIN_OBS = 3           # 证据门槛：每条臂至少观测过这么多次，才允许价值门放行
DEFAULT_EXPLORE_FRAC = 0.05   # 每批探索额度比例（宁不打扰，但留试错口）
DEFAULT_EXPLORE_MIN = 2       # 每批探索额度下限（批很小也要能学）

# 六环节工具名与字段契约（与 run_agent 注册的工具一一对应，tests 有专项核对）
INTERVENTION_TOOL = "allocate_interventions"
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "detect_anomaly": ("metric", "delta_pct", "silent_share"),
    "locate_cohort": ("cohort", "rule", "size"),
    "analyze_cause": ("cohort", "silence_median", "tenure_median"),
    "assess_risk": ("cohort", "bands", "n_total"),
    INTERVENTION_TOOL: (
        "batch", "batch_size", "arms", "mean_reward", "cum_reward",
        "veto_count", "veto_rate", "explore_count", "policy",
        "cohort", "stream_remaining",
    ),
    "design_experiment": ("grouping", "primary_metric", "n_treated"),
}
FINISH_TOOL = "finish"  # 与 planner.FINISH_TOOL 同值（避免反向依赖故不 import）


class InterventionCritic:
    """Actor 提出臂 → Critic 复核（保守下界门）→ 否决则降级对照。

    属性（供测试与报告核对）：
        trials / passed / explored / vetoed —— 复核计数；
        n_obs —— 每条臂累计观测次数（证据门槛的依据）；
        budget_used —— 本批已消耗的探索额度。
    """

    def __init__(
        self,
        arms: tuple[str, ...],
        context_dim: int,
        *,
        ridge: float = 1.0,
        beta: float = DEFAULT_BETA,
        min_obs: int = DEFAULT_MIN_OBS,
        explore_frac: float = DEFAULT_EXPLORE_FRAC,
        explore_min: int = DEFAULT_EXPLORE_MIN,
    ):
        arms = tuple(arms)
        if "control" not in arms:
            raise ValueError(f"Critic 需要对照臂 control（当前 {arms}）")
        if ridge <= 0:
            raise ValueError(f"ridge 需 > 0（当前 {ridge}）")
        if beta <= 0:
            raise ValueError(f"beta 需 > 0（当前 {beta}）")
        if min_obs < 0:
            raise ValueError(f"min_obs 需 ≥ 0（当前 {min_obs}）")
        if not (0.0 <= explore_frac <= 1.0):
            raise ValueError(f"explore_frac 需在 [0,1] 内（当前 {explore_frac}）")
        if explore_min < 0:
            raise ValueError(f"explore_min 需 ≥ 0（当前 {explore_min}）")
        self.arms = arms
        self.control = arms.index("control")
        self.d = int(context_dim)
        self.ridge = float(ridge)
        self.beta = float(beta)
        self.min_obs = int(min_obs)
        self.explore_frac = float(explore_frac)
        self.explore_min = int(explore_min)
        # 镜像后验：与学习策略同款岭回归（只吃 observe 返回过的净奖励）
        self.A = [ridge * np.eye(self.d) for _ in arms]
        self.b = [np.zeros(self.d) for _ in arms]
        self.n_obs = [0] * len(arms)
        self._budget = 0
        self.trials = 0
        self.passed = 0
        self.explored = 0
        self.vetoed = 0

    # ── 批次准备与复核 ──────────────────────────────────────

    def new_batch(self, batch_size: int) -> int:
        """开一批：重置探索额度，返回本批额度（max(下限, ⌈批大小×比例⌉)）。"""
        if batch_size < 1:
            raise ValueError(f"批大小需 ≥ 1（当前 {batch_size}）")
        self._budget = max(self.explore_min, int(math.ceil(batch_size * self.explore_frac)))
        return self._budget

    @property
    def budget_left(self) -> int:
        return self._budget

    def _lcb(self, x: np.ndarray, a: int) -> float:
        """保守下界：μ̂ᵀx − β·√(xᵀA⁻¹x)（悲观估计）。"""
        A_inv = np.linalg.inv(self.A[a])
        mu = float((A_inv @ self.b[a]) @ x)
        std = float(np.sqrt(max(0.0, x @ A_inv @ x)))
        return mu - self.beta * std

    def review(self, x: np.ndarray, proposed: int) -> tuple[int, dict]:
        """复核 Actor 的提议臂：返回 (最终臂下标, 复核记录)。

        规则（宁不打扰）：
          · 提议就是 control → 恒放行（不打扰无需复核）；
          · 证据门槛：该臂累计观测 ≥ min_obs 才允许价值门判断——单次观测会让
            后验下界"过度自信"，先看清再开价值门；
          · lcb(提议) > lcb(control) → 放行（悲观估计也不亏于不触达）；
          · 否则若本批探索额度未尽 → 按额度放行（小额度试错）；
          · 额度用尽 → 否决并降级 control。
        """
        proposed = int(proposed)
        if not (0 <= proposed < len(self.arms)):
            raise ValueError(f"提议臂下标越界：{proposed}（共 {len(self.arms)} 条臂）")
        lcb_c = self._lcb(x, self.control)
        if proposed == self.control:
            return self.control, {"action": "control", "lcb": round(lcb_c, 6),
                                  "lcb_control": round(lcb_c, 6)}
        lcb_p = self._lcb(x, proposed)
        self.trials += 1
        info = {"lcb": round(lcb_p, 6), "lcb_control": round(lcb_c, 6),
                "n_obs": self.n_obs[proposed], "min_obs": self.min_obs}
        if self.n_obs[proposed] >= self.min_obs and lcb_p > lcb_c:
            self.passed += 1
            return proposed, {"action": "pass", **info}
        if self._budget > 0:
            self._budget -= 1
            self.explored += 1
            return proposed, {"action": "explore", **info}
        self.vetoed += 1
        return self.control, {"action": "veto", **info}

    def observe(self, x: np.ndarray, a: int, r: float) -> None:
        """镜像观测：只接收"实际发生"的 (上下文, 臂, 净奖励)，与 Actor 同步。"""
        a = int(a)
        if not (0 <= a < len(self.arms)):
            raise ValueError(f"观测臂下标越界：{a}")
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
        self.n_obs[a] += 1

    def stats(self) -> dict:
        return {
            "trials": self.trials,
            "passed": self.passed,
            "explored": self.explored,
            "vetoed": self.vetoed,
            "budget_left": self._budget,
            "min_obs": self.min_obs,
            "n_obs": {arm: int(n) for arm, n in zip(self.arms, self.n_obs)},
        }


# ── 循环级校验（接进 harness 的 Verifier 位）─────────────────

class CriticVerifier(Verifier):
    """S4 定案：六环节产物结构核对 + 干预完成度确认收工。

    · 登记工具的输出必须含契约字段（REQUIRED_FIELDS）；
    · 干预批次额外核对：三臂计数求和 = 批大小、veto_rate ∈ [0,1]、批大小 ≥ 1；
    · finish 提名：干预环节未通过校验前一律拒绝（防"没做事就收工"）。
    其余工具的"业务断言"（数字是否自洽等）留待后续环节逐条定案——这里先按
    "结构完整即放行"的占位口径处理，并在说明里如实标注。
    """

    def __init__(self) -> None:
        self.intervention_ok = False

    def check(self, decision: dict[str, Any] | None, observation: Any) -> tuple[bool, str]:
        tool = (decision or {}).get("tool")
        if tool == FINISH_TOOL:
            if self.intervention_ok:
                return True, "收工确认：干预环节已完成并通过校验"
            return False, "拒绝收工：干预环节尚未产出通过校验的分配批次"
        if tool in REQUIRED_FIELDS:
            if not isinstance(observation, dict):
                return False, f"{tool} 的观察不是结构化对象"
            missing = [k for k in REQUIRED_FIELDS[tool] if k not in observation]
            if missing:
                return False, f"{tool} 观察缺契约字段：{missing}"
            if tool == INTERVENTION_TOOL:
                ok, note = _check_batch(observation)
                if not ok:
                    return False, note
                self.intervention_ok = True
                return True, "干预批次结构校验通过"
            return True, f"占位校验：{tool} 结构完整（业务断言留待后续定案）"
        return True, "占位校验：未登记工具放行"


def _check_batch(obs: dict) -> tuple[bool, str]:
    """干预批次的结构核对（数值自洽，不看真值）。"""
    size = obs.get("batch_size")
    if not isinstance(size, int) or size < 1:
        return False, f"批大小非法：{size!r}"
    arms = obs.get("arms")
    if not isinstance(arms, dict):
        return False, "arms 不是计数对象"
    if sum(int(v) for v in arms.values()) != size:
        return False, f"三臂计数求和 {sum(int(v) for v in arms.values())} ≠ 批大小 {size}"
    veto_rate = obs.get("veto_rate")
    if not isinstance(veto_rate, (int, float)) or not (0.0 <= float(veto_rate) <= 1.0):
        return False, f"veto_rate 非法：{veto_rate!r}"
    if int(obs.get("stream_remaining", -1)) < 0:
        return False, f"stream_remaining 非法：{obs.get('stream_remaining')!r}"
    return True, ""
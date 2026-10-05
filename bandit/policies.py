# -*- coding: utf-8 -*-
"""S3 Bandit · 策略：LinUCB + Thompson 线性高斯 + 基线（随机 / 固定臂 / oracle）。

先说人话：
    每人到访一次、选一条臂、只看得到这条臂的净奖励——策略的全部工作就是
    "每次选臂"。两条学习策略都建立在同一个线性假设上：净奖励 ≈ θᵀx（x 是
    可观测状态展开的上下文），差别在"怎么应对不确定性"：
      · linucb  ——每臂维护岭回归的置信椭圆，选"上界最高"的臂（乐观主义：
                   不确定的方向先试一试），alpha 控制探索强度；
      · thompson——每臂维护后验（高斯），从后验采样一组 θ̃ 再贪心，
                   不确定性自然会带来自动衰减的探索（概率匹配）。
    基线：random（不学习）、fixed_*（三条固定臂：纯利用不求知）、
    oracle（上帝视角选最优臂，上界参照）。oracle 只在评测里跑，永远不是
    "可部署策略"——它读到的是真值矩阵。

冻结评估口径（greedy）：
    在线跑完学习后，要评估"学到的知识在没见过的用户上值多少"。此时策略
    冻结：LinUCB 用后验均值打分（去掉 bonus），Thompson 用后验均值（不采样）
    ——这就是部署时"不再探索"的口径。

硬规则：学习策略的 update 只接收 observe 返回的净奖励；越界下标 / 未知臂名
由世界与工厂显式报错；随机流一律外部注入（无弱默认）。
"""
from __future__ import annotations

import numpy as np

from .protocol import ARMS

POLICY_LEARNERS: tuple[str, ...] = ("linucb", "thompson")
BASELINES: tuple[str, ...] = ("random", "fixed_control", "fixed_rec", "fixed_recall", "oracle")
POLICIES: tuple[str, ...] = POLICY_LEARNERS + BASELINES

# 默认超参（设计取值；在探针里实测扫描后锁定，见 README 结果表）：
# alpha 0.5/1.0/2.0 三档在线均值几乎持平（1.045–1.090）取中档；
# sigma 1.0/2.0/4.0 扫描中 1.0 跨种子略优（在线 +0.01~0.03、审计持平）取 1.0。
DEFAULT_ALPHA = 1.0
DEFAULT_RIDGE = 1.0
DEFAULT_SIGMA = 1.0

# 策略随机流的盐基址（与协议流 301、模拟器流彼此独立）
_POLICY_SALT_BASE = 303


def policy_rng(seed: int, name: str) -> np.random.Generator:
    """策略随机流：由 (种子, 固定盐, 策略下标) 派生——同参永远同结果。"""
    if name not in POLICIES:
        raise ValueError(f"未知策略：{name!r}（支持 {POLICIES}）")
    return np.random.default_rng([seed, 20261005, _POLICY_SALT_BASE + POLICIES.index(name)])


# ── 学习策略 ────────────────────────────────────────────────

class LinUCBPolicy:
    """LinUCB：每臂岭回归 + 置信上界打分；alpha 越大探索越强。

    A_a = ridge·I + Σxxᵀ，b_a = Σrx → θ̂_a = A_a⁻¹b_a；
    score_a = θ̂_aᵀx + alpha·√(xᵀA_a⁻¹x)（后半段 = 不确定性奖励）。
    """

    name = "linucb"

    def __init__(self, context_dim: int, *, alpha: float = DEFAULT_ALPHA, ridge: float = DEFAULT_RIDGE):
        if alpha <= 0:
            raise ValueError(f"alpha 需 > 0（当前 {alpha}）")
        if ridge <= 0:
            raise ValueError(f"ridge 需 > 0（当前 {ridge}）")
        self.d = int(context_dim)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.A = [ridge * np.eye(self.d) for _ in range(len(ARMS))]
        self.b = [np.zeros(self.d) for _ in range(len(ARMS))]

    def _scores(self, x: np.ndarray, *, explore: bool) -> np.ndarray:
        scores = np.empty(len(ARMS))
        for a in range(len(ARMS)):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            score = float(theta @ x)
            if explore:
                score += self.alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
            scores[a] = score
        return scores

    def select(self, x: np.ndarray, i: int = 0) -> int:
        """在线决策：上界最高者（并列取靠前的臂，确定性）。"""
        return int(np.argmax(self._scores(x, explore=True)))

    def greedy(self, x: np.ndarray, i: int = 0) -> int:
        """冻结评估：只用后验均值（无 bonus），即"不再探索"的部署口径。"""
        return int(np.argmax(self._scores(x, explore=False)))

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


class ThompsonLinearPolicy:
    """Thompson 线性高斯：每臂后验 N(θ̂, σ²A⁻¹)，采样 θ̃ 后贪心选臂。

    每臂每步消耗 d 个标准正态（随机流由调用方注入，保证可复现）；
    sigma 是观测噪声尺度——越大后验越宽、探索越多。
    """

    name = "thompson"

    def __init__(
        self,
        context_dim: int,
        rng: np.random.Generator,
        *,
        sigma: float = DEFAULT_SIGMA,
        ridge: float = DEFAULT_RIDGE,
    ):
        if rng is None:
            raise ValueError("thompson 需要显式传入随机流 rng（无弱默认）")
        if sigma <= 0:
            raise ValueError(f"sigma 需 > 0（当前 {sigma}）")
        if ridge <= 0:
            raise ValueError(f"ridge 需 > 0（当前 {ridge}）")
        self.d = int(context_dim)
        self.rng = rng
        self.sigma = float(sigma)
        self.ridge = float(ridge)
        self.A = [ridge * np.eye(self.d) for _ in range(len(ARMS))]
        self.b = [np.zeros(self.d) for _ in range(len(ARMS))]

    def select(self, x: np.ndarray, i: int = 0) -> int:
        """在线决策：从每臂后验采样 θ̃，选 θ̃ᵀx 最大者。"""
        best_a, best_score = 0, -np.inf
        for a in range(len(ARMS)):
            mu = np.linalg.solve(self.A[a], self.b[a])
            L = np.linalg.cholesky(self.A[a])
            z = self.rng.standard_normal(self.d)
            theta_tilde = mu + self.sigma * np.linalg.solve(L.T, z)
            score = float(theta_tilde @ x)
            if score > best_score:
                best_a, best_score = a, score
        return int(best_a)

    def greedy(self, x: np.ndarray, i: int = 0) -> int:
        """冻结评估：后验均值贪心（不采样、不消耗随机流）。"""
        best_a, best_score = 0, -np.inf
        for a in range(len(ARMS)):
            mu = np.linalg.solve(self.A[a], self.b[a])
            score = float(mu @ x)
            if score > best_score:
                best_a, best_score = a, score
        return int(best_a)

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


# ── 基线 ────────────────────────────────────────────────────

class RandomPolicy:
    """不学习的均匀随机（探索拉满、利用为零的下界参照）。"""

    name = "random"

    def __init__(self, rng: np.random.Generator):
        if rng is None:
            raise ValueError("random 需要显式传入随机流 rng（无弱默认）")
        self.rng = rng

    def select(self, x: np.ndarray, i: int = 0) -> int:
        return int(self.rng.integers(0, len(ARMS)))

    def greedy(self, x: np.ndarray, i: int = 0) -> int:
        return self.select(x, i)

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        pass


class FixedArmPolicy:
    """固定臂：永远推荐同一条臂（纯利用 / 不探索的对照）。"""

    def __init__(self, arm: str):
        if arm not in ARMS:
            raise ValueError(f"未知臂：{arm!r}（支持 {ARMS}）")
        self.arm = arm
        self.name = f"fixed_{arm}"

    def select(self, x: np.ndarray, i: int = 0) -> int:
        return ARMS.index(self.arm)

    def greedy(self, x: np.ndarray, i: int = 0) -> int:
        return self.select(x, i)

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        pass


class OraclePolicy:
    """上帝视角：永远选该用户的最优臂（评测上界，不可部署）。"""

    name = "oracle"

    def __init__(self, oracle_arm: np.ndarray):
        self.oracle_arm = np.asarray(oracle_arm)

    def select(self, x: np.ndarray, i: int = 0) -> int:
        i = int(i)
        if not (0 <= i < len(self.oracle_arm)):
            raise ValueError(f"oracle 用户下标越界：{i}")
        return ARMS.index(str(self.oracle_arm[i]))

    def greedy(self, x: np.ndarray, i: int = 0) -> int:
        return self.select(x, i)

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        pass


# ── 工厂与超参快照 ──────────────────────────────────────────

def make_policy(
    name: str,
    *,
    context_dim: int,
    seed: int,
    oracle_arm: np.ndarray | None = None,
    alpha: float = DEFAULT_ALPHA,
    ridge: float = DEFAULT_RIDGE,
    sigma: float = DEFAULT_SIGMA,
):
    """按名字造策略（学习策略用种子派生随机流；基线不依赖种子）。"""
    if name not in POLICIES:
        raise ValueError(f"未知策略：{name!r}（支持 {POLICIES}）")
    if name == "linucb":
        return LinUCBPolicy(context_dim, alpha=alpha, ridge=ridge)
    if name == "thompson":
        return ThompsonLinearPolicy(context_dim, policy_rng(seed, name), sigma=sigma, ridge=ridge)
    if name == "random":
        return RandomPolicy(policy_rng(seed, name))
    if name == "oracle":
        if oracle_arm is None:
            raise ValueError("oracle 策略需要 oracle_arm（来自 world.oracle_arm）")
        return OraclePolicy(oracle_arm)
    return FixedArmPolicy(name.removeprefix("fixed_"))


def hyperparams_snapshot(
    policies: tuple[str, ...] | list[str],
    alpha: float = DEFAULT_ALPHA,
    ridge: float = DEFAULT_RIDGE,
    sigma: float = DEFAULT_SIGMA,
) -> dict:
    """给 manifest 的超参快照（不依赖已建对象，纯配置复述）。"""
    snap = {}
    for name in policies:
        if name not in POLICIES:
            raise ValueError(f"未知策略：{name!r}（支持 {POLICIES}）")
        if name == "linucb":
            snap[name] = {"alpha": float(alpha), "ridge": float(ridge)}
        elif name == "thompson":
            snap[name] = {"sigma": float(sigma), "ridge": float(ridge)}
        elif name == "random":
            snap[name] = {"dist": "uniform"}
        elif name == "oracle":
            snap[name] = {"type": "upper_bound（评测专用，不可部署）"}
        else:
            snap[name] = {"arm": name.removeprefix("fixed_")}
    return snap
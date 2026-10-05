# -*- coding: utf-8 -*-
"""bandit_tools.py —— S4 干预台：把 S3 的 bandit 策略接成 harness 可调用工具。

先说人话：
    S3 的策略对象会"选臂"，但它只有在部分反馈世界里才完整——选完必须结算
    奖励、更新后验。干预台（InterventionConsole）把这条链收进一次工具调用：
        选臂（Actor）→ 复核（Critic）→ 结算（world.observe，唯一反馈通道）
        → 学习（policy.update / critic.observe）
    每次调用 = 一批名额的分配（`allocate_interventions(count)`），返回
    "运营口径"的批次摘要（臂占比 / 净奖励 / 否决率 / 剩余名额）——**不含
    任何真值信息**；oracle 只进内部评测账本（落盘供对照，不进 Agent 上下文）。

    人群收窄：干预台只对"已圈定人群"分配——locate_cohort 环节通过
    open_cohort 把人群（在线池内按可观测条件圈出）交给干预台，之后的名额
    都从这个人群里按在线顺序发放。

硬规则（与 S3 同一套，tests 有专项断言）：
    · 策略只能通过 world.observe 拿到"所选臂"的净奖励；真值矩阵不可见；
    · 每次分配恰好结算一条臂的记录；
    · 同参同种子逐字节一致（策略随机流由调用方种子派生）。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# 干预工具规范名（Critic 契约与 run_agent 注册共用）
INTERVENTION_TOOL = "allocate_interventions"
DEFAULT_COUNT = 200  # 默认批大小（名额数）

BATCH_COLUMNS = (
    "mode", "batch", "seq", "uid", "proposed_arm", "arm", "gate",
    "vetoed", "reward", "oracle_reward",
)


class InterventionConsole:
    """干预台：一次调用完成一批人次"选臂→复核→结算→学习"。

    参数：
        world   bandit.protocol.BanditWorld（部分反馈世界；评测与 oracle 专用真值）
        policy  bandit.policies 的策略对象（select/update 接口）
        critic  harness.critic.InterventionCritic 或 None（None = 无安全门对照路径）
    """

    def __init__(self, world: Any, policy: Any, critic: Any | None = None):
        self.world = world
        self.policy = policy
        self.critic = critic
        self._stream: list[int] | None = None  # open_cohort 后 = 人群的在线决策顺序
        self._cohort: str | None = None
        self._cursor = 0
        self._rows: list[dict] = []
        self._batch_no = 0
        self.batch_logs: list[dict] = []  # 每批摘要（Agent 可见口径）
        self._cum_reward = 0.0
        self._cum_oracle = 0.0
        self._cum_regret = 0.0

    # ── 人群 ────────────────────────────────────────────────

    @property
    def cohort_ready(self) -> bool:
        return self._stream is not None

    @property
    def cohort_name(self) -> str | None:
        return self._cohort

    def remaining(self) -> int:
        """人群中尚未分配的名额数（未圈人时为 0）。"""
        if self._stream is None:
            return 0
        return len(self._stream) - self._cursor

    def cohort_index(self) -> list[int]:
        """人群在世界里的下标列表（按在线决策顺序；供原因/风险环节取数）。"""
        return list(self._stream or [])

    def open_cohort(self, idx: list[int], name: str) -> dict:
        """圈定人群（须在首次分配前调用；重复调用显式报错）。"""
        if self._stream is not None:
            raise ValueError("人群已开跑，不可更换（干预台一次运行只服务一个人群）")
        idx = [int(i) for i in idx]
        if not idx:
            raise ValueError("人群为空：检查圈人条件")
        self._stream = idx
        self._cohort = str(name)
        return {"cohort": self._cohort, "size": len(idx)}

    # ── 分配（核心）─────────────────────────────────────────

    def allocate(self, count: int = DEFAULT_COUNT) -> dict:
        """批量分配：从人群按在线顺序取 count 个名额，逐人走完整条链。

        返回批次摘要（Agent 可见）：臂计数 / 批次均值净奖励 / 累计净奖励 /
        否决与探索计数 / 剩余名额。真值（oracle_reward）只进内部账本。
        """
        if self._stream is None:
            raise ValueError("尚未圈定人群：先调用 locate_cohort（干预台只服务已圈人群）")
        count = int(count)
        if count < 1:
            raise ValueError(f"名额数需 ≥ 1（当前 {count}）")
        remaining = self.remaining()
        if remaining == 0:
            raise ValueError("人群名额已跑完（stream_remaining = 0）")
        actual = min(count, remaining)

        self._batch_no += 1
        if self.critic is not None:
            self.critic.new_batch(actual)
        arms = self.world.arms
        counts = {a: 0 for a in arms}
        rewards: list[float] = []
        vetoed = explored = 0

        for _ in range(actual):
            i = self._stream[self._cursor]
            self._cursor += 1
            x = self.world.context_of(i)
            proposed = int(self.policy.select(x, i))
            if not (0 <= proposed < len(arms)):
                raise ValueError(f"策略 {self.policy.name} 返回非法臂下标：{proposed}")
            if self.critic is not None:
                chosen, gate_info = self.critic.review(x, proposed)
                action = gate_info["action"]
            else:
                chosen, action = proposed, None
            if action == "veto":
                vetoed += 1
            elif action == "explore":
                explored += 1

            r = self.world.observe(i, arms[chosen])  # ★ 唯一反馈通道
            self.policy.update(x, chosen, r)
            if self.critic is not None:
                self.critic.observe(x, chosen, r)
            _, oracle_r = self.world.oracle_best(i)  # 仅评测账本

            counts[arms[chosen]] += 1
            rewards.append(r)
            self._cum_reward += r
            self._cum_oracle += oracle_r
            self._cum_regret += oracle_r - r
            self._rows.append({
                "batch": self._batch_no,
                "seq": len(self._rows),
                "uid": str(self.world.uid[i]),
                "proposed_arm": arms[proposed],
                "arm": arms[chosen],
                "gate": action or "off",
                "vetoed": bool(action == "veto"),
                "reward": round(r, 6),
                "oracle_reward": round(oracle_r, 6),
            })

        obs = {
            "batch": self._batch_no,
            "requested": count,
            "batch_size": actual,
            "arms": counts,
            "mean_reward": round(sum(rewards) / actual, 6),
            "cum_reward": round(self._cum_reward, 6),
            "veto_count": vetoed,
            "veto_rate": round(vetoed / actual, 6),
            "explore_count": explored,
            "policy": self.policy.name,
            "cohort": self._cohort,
            "stream_remaining": self.remaining(),
        }
        self.batch_logs.append(obs)
        return obs

    # ── 账本（评测口径；不进 Agent 上下文）──────────────────

    def eval_summary(self) -> dict:
        """累计账本：净奖励 / 遗憾 / 臂占比 / 否决率（含 oracle 对照，评测专用）。"""
        n = len(self._rows)
        if n == 0:
            raise ValueError("尚无分配记录，无法汇总")
        shares = {a: 0 for a in self.world.arms}
        for row in self._rows:
            shares[row["arm"]] += 1
        return {
            "policy": self.policy.name,
            "n_batches": self._batch_no,
            "steps": n,
            "cum_reward": round(self._cum_reward, 6),
            "cum_oracle": round(self._cum_oracle, 6),
            "cum_regret": round(self._cum_regret, 6),
            "avg_reward": round(self._cum_reward / n, 6),
            "reward_vs_oracle_frac": (
                round(self._cum_reward / self._cum_oracle, 6) if self._cum_oracle > 0 else None
            ),
            "share_control": round(shares["control"] / n, 6),
            "share_rec": round(shares["rec"] / n, 6),
            "share_recall": round(shares["recall"] / n, 6),
            "veto_count": sum(1 for r in self._rows if r["vetoed"]),
            "veto_rate": round(sum(1 for r in self._rows if r["vetoed"]) / n, 6),
            "explore_count": sum(1 for r in self._rows if r["gate"] == "explore"),
            "cohort": self._cohort,
            "stream_remaining": self.remaining(),
        }

    def rows_frame(self) -> pd.DataFrame:
        """逐人次日志（含真值对照列 oracle_reward——评测专用，落盘但不进上下文）。"""
        return pd.DataFrame(self._rows, columns=[c for c in BATCH_COLUMNS if c != "mode"])


# ── 工具注册 ────────────────────────────────────────────────

def register_intervention_tools(registry: Any, console: InterventionConsole,
                                default_count: int = DEFAULT_COUNT) -> None:
    """把干预台登记成 harness 工具（前置条件：人群已圈定且名额未尽）。"""

    registry.register(
        INTERVENTION_TOOL,
        lambda count=default_count: console.allocate(count),
        f"对已圈定人群按 bandit 策略批量分配触达臂（含 Critic 安全门）；"
        f"count 为名额数（默认 {default_count}）",
        when=lambda state: console.cohort_ready and console.remaining() > 0,
    )
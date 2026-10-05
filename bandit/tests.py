# -*- coding: utf-8 -*-
"""S3 Bandit · 冒烟与一致性测试（无外部依赖，纯断言）。

运行（在仓库根目录，任选其一）：
    python -m bandit.tests
    pytest bandit/tests.py

覆盖：
    1. 确定性：同参同种子双跑逐值一致；换种子曲线不同
    2. 协议完整性：在线/审计池不交且并全、审计升序、切分可复现；
       上下文宽与在线池 z 统计（均值≈0 / std≈1）；真值矩阵与 effects 抽查一致；
       control 列恒 0；oracle 数组 = 逐用户 argmax
    3. 无泄漏：把"未观测 (i, arm)"的奖励全部篡改后同种子重跑，决策逐值相同；
       冻结审计不 update（A/b 不变）且不消耗 Thompson 随机流
    4. 端点：oracle 零遗憾、所有策略 regret≥0、vs_oracle 比例≤1、
       固定臂 share 恒 1、fixed_control 奖励恒 0
    5. 学习信号：linucb/thompson 在线均值明显高于 random 与 fixed_rec；
       tail20 ≥ head20 + 余量（确实在学）；审计冻结得分 ≥ 0.5× 审计 oracle
    6. 边界：audit 0/1/1.5、未知策略 / 基、alpha / ridge / sigma≤0、
       人数不足、observe 越界、未知臂、非法切分等均显式报错
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from simulator.run_sim import build_tables

from .evaluate import run_audit, run_online
from .policies import (
    POLICIES, FixedArmPolicy, LinUCBPolicy, RandomPolicy, ThompsonLinearPolicy,
    hyperparams_snapshot, make_policy,
)
from .protocol import ARMS, BanditWorld, build_world, split_online_audit
from .run_bandit import run_all, run_is_deterministic

# 测试用小人口（跑得快；信号阈值按此规模 × seed 13 的实测数字锁定，留 ≥40% 余量）
N_TEST = 1200
SEED_TEST = 13
COMMON = dict(
    n_users=N_TEST, days=14, window=7, seed=SEED_TEST, audit=0.3,
    policies=POLICIES, features_path=None, use_fit=False,
)


@lru_cache(maxsize=1)
def _tables():
    tables, meta = build_tables(N_TEST, 14, SEED_TEST, 7, None, False)
    return tables, meta


@lru_cache(maxsize=1)
def _run():
    return run_all(**COMMON)


# ── 1. 确定性 ───────────────────────────────────────────────

def test_determinism():
    a = run_all(**COMMON)
    b = run_all(**COMMON)
    assert run_is_deterministic(a, b), "同参同种子双跑不一致"

    c = run_all(**{**COMMON, "seed": 12})
    assert not a["frames"]["curve"].equals(c["frames"]["curve"]), "换种子曲线居然一致"
    assert not a["frames"]["summary"].equals(c["frames"]["summary"]), "换种子汇总居然一致"


# ── 2. 协议完整性 ───────────────────────────────────────────

def test_protocol_integrity():
    tables, _ = _tables()
    users = tables["users"]
    online_df, audit_df = split_online_audit(users, 0.3, SEED_TEST)

    assert len(online_df) + len(audit_df) == len(users), "切分未覆盖全人口"
    assert set(online_df["uid"]).isdisjoint(set(audit_df["uid"])), "在线池与审计池重叠"
    assert set(online_df["uid"]) | set(audit_df["uid"]) == set(users["uid"]), "并集不完整"

    # 切分可复现；换种子不同
    o2, a2 = split_online_audit(users, 0.3, SEED_TEST)
    assert online_df.equals(o2) and audit_df.equals(a2), "切分不可复现"
    o3, _ = split_online_audit(users, 0.3, 12)
    assert not online_df.equals(o3), "换种子切分居然不变"

    # 审计池按原始位置升序；在线池保持洗牌序（不等于升序）
    pos = {uid: k for k, uid in enumerate(users["uid"].tolist())}
    audit_pos = [pos[u] for u in audit_df["uid"]]
    assert audit_pos == sorted(audit_pos), "审计池应为升序"
    online_pos = [pos[u] for u in online_df["uid"]]
    assert online_pos != sorted(online_pos), "在线池应为洗牌序（决策顺序）"


def test_world_integrity():
    tables, _ = _tables()
    users, effects = tables["users"], tables["effects"]
    online_df, audit_df = split_online_audit(users, 0.3, SEED_TEST)
    built = build_world(tables, 0.3, SEED_TEST, "logsq")
    world, meta = built["world"], built["meta"]

    assert meta["n_online"] == len(built["online"]) == len(online_df)
    assert meta["n_audit"] == len(built["audit"]) == len(audit_df)
    assert [world.uid[i] for i in built["online"]] == online_df["uid"].tolist(), "在线顺序错位"
    assert [world.uid[i] for i in built["audit"]] == audit_df["uid"].tolist(), "审计顺序错位"
    assert set(built["online"].tolist()).isdisjoint(set(built["audit"].tolist()))
    assert sorted(built["online"].tolist() + built["audit"].tolist()) == list(range(len(users)))

    # 上下文：宽度 = 1（截距）+ 5（logsq 展开）；在线池 z 统计均值≈0 / std≈1
    assert meta["basis"] == "logsq" and meta["context_width"] == 6
    assert world.context.shape == (len(users), 6)
    assert np.allclose(world.context[:, 0], 1.0), "截距列应恒 1"
    sub = world.context[built["online"]][:, 1:]
    assert np.allclose(sub.mean(axis=0), 0.0, atol=1e-9), "在线池展开列均值应为 0"
    assert np.allclose(sub.std(axis=0), 1.0, atol=1e-9), "在线池展开列 std 应为 1"
    stats = meta["context_stats"]
    assert stats["columns"] == ["act_30d", "log1p_silence", "log1p_silence_sq",
                                "interest_concentration", "log1p_tenure"]
    assert abs(stats["mean"][0] - online_df["act_30d"].mean()) < 1e-5, "统计量口径漂移"

    # 真值矩阵与 effects 抽查一致；control 列恒 0；oracle = 逐用户 argmax
    eff = effects.set_index(["uid", "arm"])
    for k in (0, 1, 137, len(users) - 1):
        uid = world.uid[k]
        for j, arm in enumerate(ARMS):
            assert abs(world.rewards[k, j] - eff.loc[(uid, arm), "reward"]) < 1e-9, \
                f"真值矩阵与 effects 不一致：({uid}, {arm})"
    assert (world.rewards[:, ARMS.index("control")] == 0.0).all(), "control 列应恒 0"
    best = world.rewards.argmax(axis=1)
    assert (world.oracle_reward == world.rewards.max(axis=1)).all(), "oracle 奖励应恒为上界"
    assert list(world.oracle_arm) == [ARMS[j] for j in best], "oracle 臂应为逐用户 argmax"


# ── 3. 无泄漏（未观测的臂不产生任何影响）────────────────────

def test_no_leakage():
    """关键不变量：把"没观测过的 (i, arm)"奖励篡改掉，学习策略决策必须逐值不变。"""
    tables, _ = _tables()
    for name in ("linucb", "thompson"):
        built = build_world(tables, 0.3, SEED_TEST, "logsq")
        world, online_idx = built["world"], built["online"]
        p1 = make_policy(name, context_dim=built["meta"]["context_width"], seed=SEED_TEST)
        log1 = run_online(world, online_idx, p1)

        observed = set(world.revealed)
        assert len(observed) == len(online_idx), f"{name}：每人次应恰被观测一次"
        assert observed == set(zip(online_idx.tolist(), log1["arm"].tolist())), \
            f"{name}：观测集合与所选臂不一致（有偷看嫌疑）"
        for (i, arm), r in zip(world.revealed, log1["reward"].tolist()):
            assert abs(world.rewards[i, ARMS.index(arm)] - r) < 5e-7, f"{name}：观测奖励与真值不符"

        # 变体世界：O 之外的臂奖励全部 +777+i，其余不动
        rewards2 = world.rewards.copy()
        for i in range(len(world.uid)):
            for j, arm in enumerate(ARMS):
                if (i, arm) not in observed:
                    rewards2[i, j] += 777.0 + i
        world2 = BanditWorld(world.uid, world.context, rewards2)
        p2 = make_policy(name, context_dim=built["meta"]["context_width"], seed=SEED_TEST)
        log2 = run_online(world2, online_idx, p2)

        cols = ["step", "arm"]
        assert log1[cols].equals(log2[cols]), \
            f"{name}：未观测臂的奖励改动影响了决策（潜在泄漏）"


def test_frozen_audit():
    """冻结审计：走 observe（有观测记录）、绝不 update（A/b 不变）、不消耗随机流。"""
    tables, _ = _tables()
    built = build_world(tables, 0.3, SEED_TEST, "logsq")
    world, online_idx, audit_idx = built["world"], built["online"], built["audit"]

    p = make_policy("linucb", context_dim=built["meta"]["context_width"], seed=SEED_TEST)
    run_online(world, online_idx, p)
    A_snap = [A.copy() for A in p.A]
    b_snap = [b.copy() for b in p.b]
    log_audit = run_audit(world, audit_idx, p)
    assert all(np.array_equal(A, s) for A, s in zip(p.A, A_snap)), "审计阶段不得 update（A 变了）"
    assert all(np.array_equal(b, s) for b, s in zip(p.b, b_snap)), "审计阶段不得 update（b 变了）"
    assert set(i for i, _ in world.revealed) == set(audit_idx.tolist()), "审计应逐人次走 observe"
    assert world.revealed and len(world.revealed) == len(audit_idx)
    assert (log_audit["arm"] == log_audit["arm"]).all()

    pt = make_policy("thompson", context_dim=built["meta"]["context_width"], seed=SEED_TEST)
    run_online(world, online_idx, pt)
    state_before = dict(pt.rng.bit_generator.state)
    run_audit(world, audit_idx, pt)
    assert pt.rng.bit_generator.state == state_before, "冻结评估不应消耗 Thompson 随机流"


# ── 4. 端点 ─────────────────────────────────────────────────

def test_endpoints():
    run = _run()
    summ = run["frames"]["summary"].set_index("policy")
    curve, audit = run["frames"]["curve"], run["frames"]["audit"]

    assert (audit["regret"] >= 0).all(), "审计遗憾出现负值"
    assert (summ["cum_regret"] >= 0).all(), "累计遗憾出现负值"
    assert summ.loc["oracle", "cum_regret"] == 0.0, "oracle 累计遗憾应为 0"
    assert (audit[audit["policy"] == "oracle"]["regret"] == 0.0).all(), "oracle 审计遗憾应为 0"
    assert (summ["reward_vs_oracle_frac"] <= 1.0).all(), "在线得分超过 oracle 上界"
    assert (summ["audit_vs_oracle_frac"] <= 1.0).all(), "审计得分超过 oracle 上界"
    assert summ.loc["oracle", "reward_vs_oracle_frac"] == 1.0
    assert summ.loc["oracle", "audit_vs_oracle_frac"] == 1.0

    # 固定臂端点：control 奖励恒 0（效应表口径），固定臂 share 恒 1
    fc = curve[curve["policy"] == "fixed_control"]
    assert (fc["reward"] == 0.0).all(), "control 奖励应恒 0"
    assert summ.loc["fixed_control", "mean_reward"] == 0.0
    assert summ.loc["fixed_control", "mean_reward_audit"] == 0.0
    assert summ.loc["fixed_rec", "share_rec"] == 1.0
    assert summ.loc["fixed_rec", "share_control"] == 0.0
    assert summ.loc["fixed_rec", "audit_share_rec"] == 1.0
    assert summ.loc["fixed_recall", "share_recall"] == 1.0
    assert summ.loc["fixed_recall", "audit_share_recall"] == 1.0

    # random 应三臂都抽（均匀性软检查：每臂至少 10%）
    for arm in ARMS:
        assert summ.loc["random", f"share_{arm}"] > 0.1, f"random 几乎不抽 {arm}"


# ── 5. 学习信号（阈值 = 1200 人 × seed 13 实测值的 ~50-70%）──

def test_learning_signal():
    run = _run()
    summ = run["frames"]["summary"].set_index("policy")
    base_random = summ.loc["random", "mean_reward"]
    base_fixed_rec = summ.loc["fixed_rec", "mean_reward"]

    for name in ("linucb", "thompson"):
        m = summ.loc[name]
        assert m["mean_reward"] >= base_random + 0.4, \
            f"{name} 在线均值未明显优于 random（{m['mean_reward']:.3f} vs {base_random:.3f}）"
        assert m["mean_reward"] >= base_fixed_rec + 0.4, \
            f"{name} 在线均值未明显优于 fixed_rec（{m['mean_reward']:.3f} vs {base_fixed_rec:.3f}）"
        assert m["tail20"] >= m["head20"] + 0.2, \
            f"{name} 未见学习信号（head20 {m['head20']:.3f} → tail20 {m['tail20']:.3f}）"

    oracle_audit = summ.loc["oracle", "mean_reward_audit"]
    assert summ.loc["linucb", "mean_reward_audit"] >= 0.5 * oracle_audit, "linucb 审计得分低于半个 oracle"
    assert summ.loc["thompson", "mean_reward_audit"] >= 0.5 * oracle_audit, "thompson 审计得分低于半个 oracle"


# ── 6. 边界与显式报错 ───────────────────────────────────────

def test_boundaries():
    def expect_error(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except ValueError:
            return
        raise AssertionError(f"{fn} 未按预期抛 ValueError")

    tables, _ = _tables()
    users = tables["users"]

    # 入口参数
    expect_error(run_all, **{**COMMON, "audit": 0.0})
    expect_error(run_all, **{**COMMON, "audit": 1.0})
    expect_error(run_all, **{**COMMON, "audit": 1.5})
    expect_error(run_all, **{**COMMON, "policies": ("nope",)})
    expect_error(run_all, **{**COMMON, "policies": ()})
    expect_error(run_all, **{**COMMON, "alpha": 0.0})
    expect_error(run_all, **{**COMMON, "ridge": -1.0})
    expect_error(run_all, **{**COMMON, "sigma": 0.0})
    expect_error(run_all, **{**COMMON, "n_users": 100})
    expect_error(run_all, **{**COMMON, "days": 0})
    expect_error(run_all, **{**COMMON, "window": 0})
    expect_error(run_all, **{**COMMON, "window": 99})
    expect_error(run_all, **{**COMMON, "basis": "cube"})

    # 切分与建世界
    expect_error(split_online_audit, users, 0.0, SEED_TEST)
    expect_error(split_online_audit, users, 1.0, SEED_TEST)
    expect_error(split_online_audit, users.head(49), 0.3, SEED_TEST)
    expect_error(split_online_audit, users.head(60), 0.97, SEED_TEST)  # 在线池过少
    expect_error(build_world, tables, 0.0, SEED_TEST, "raw")
    expect_error(build_world, tables, 0.3, SEED_TEST, "nope")

    # 策略工厂与超参
    expect_error(make_policy, "nope", context_dim=6, seed=1)
    expect_error(make_policy, "oracle", context_dim=6, seed=1)  # 缺 oracle_arm
    expect_error(LinUCBPolicy, 6, alpha=0.0)
    expect_error(LinUCBPolicy, 6, ridge=-1.0)
    expect_error(ThompsonLinearPolicy, 6, None)
    expect_error(ThompsonLinearPolicy, 6, np.random.default_rng(0), sigma=0.0)
    expect_error(RandomPolicy, None)
    expect_error(FixedArmPolicy, "nope")
    expect_error(hyperparams_snapshot, ("nope",))
    op = make_policy("oracle", context_dim=6, seed=1, oracle_arm=np.array(["rec"]))
    expect_error(op.select, np.zeros(6), 5)  # oracle 下标越界

    # 世界反馈通道
    built = build_world(tables, 0.3, SEED_TEST, "logsq")
    world = built["world"]
    expect_error(world.observe, -1, "rec")
    expect_error(world.observe, len(world.uid), "rec")
    expect_error(world.observe, 0, "nope")
    expect_error(world.context_of, -1)
    expect_error(world.oracle_best, 10 ** 6)
    expect_error(BanditWorld, np.array(["a", "b"]), np.zeros((2, 5)), np.zeros((2, 2)))


# ── 运行器 ──────────────────────────────────────────────────

def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[ok]   {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - 测试运行器需要收集所有失败
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"[done] {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
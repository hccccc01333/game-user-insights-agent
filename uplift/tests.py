# -*- coding: utf-8 -*-
"""S2 Uplift · 冒烟与一致性测试（无外部依赖，纯断言）。

运行（在仓库根目录，任选其一）：
    python -m uplift.tests
    pytest uplift/tests.py

覆盖：
    1. 确定性：同参同种子双跑一致；换种子 τ̂ 与模拟结果都不同
    2. 协议完整性：rct 每人恰 1 行、分配近似均匀、观测标签与臂对应、
       切分不相交且并集完整、分配可复现
    3. full 协议：每人 3 行；对照行取 y_void、触达行取 y_treated
    4. 无泄漏：训练/留出帧列集合断言；模型对白名单外的列不敏感
    5. 策略端点：k=100% 各策略全等且 = 该臂均值；random 各 k 相同
    6. 模型信号：召回臂 τ̂ 排序信号与定向价值显著为正；full 保真度优于 rct
    7. 边界：非法 holdout / 协议 / 学习器 / ks / 用户数 / 缺臂 均显式报错
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from simulator.run_sim import build_tables

from .evaluate import calibration_table, policy_names
from .models import TLearnerUplift, make_regressor
from .protocol import (
    ARMS, FEATURES, HOLDOUT_COLUMNS, PROTOCOLS, TRAIN_COLUMNS, TREATED_ARMS,
    assign_rct, build_observations, split_users,
)
from .run_uplift import run_all, run_is_deterministic

# 测试用小人口（跑得快；模型信号阈值按此规模的实测数字锁定）
N_TEST = 600
SEED_TEST = 11
COMMON = dict(
    n_users=N_TEST, days=14, window=7, seed=SEED_TEST, holdout=0.3,
    learners=("hgb", "ridge"), ks=(10, 20, 100), features_path=None, use_fit=False,
)


@lru_cache(maxsize=1)
def _tables():
    tables, meta = build_tables(N_TEST, 14, SEED_TEST, 7, None, False)
    return tables, meta


@lru_cache(maxsize=1)
def _run():
    return run_all(protocols=("rct", "full"), **COMMON)


# ── 1. 确定性 ───────────────────────────────────────────────

def test_determinism():
    a = run_all(protocols=("rct",), **COMMON)
    b = run_all(protocols=("rct",), **COMMON)
    assert run_is_deterministic(a, b), "同参同种子双跑不一致"

    c = run_all(protocols=("rct",), **{**COMMON, "seed": 12})
    hat_a = a["results"]["rct"]["holdout"]["tau_hat_hgb"].to_numpy()
    hat_c = c["results"]["rct"]["holdout"]["tau_hat_hgb"].to_numpy()
    assert not np.allclose(hat_a, hat_c), "换种子 τ̂ 居然一致"
    assert not a["tables"]["effects"].equals(c["tables"]["effects"]), "换种子模拟结果居然一致"


# ── 2. 协议完整性（rct）─────────────────────────────────────

def test_protocol_integrity():
    tables, _ = _tables()
    obs = build_observations(tables, 0.3, SEED_TEST, "rct")
    train, hold, meta = obs["train"], obs["holdout"], obs["meta"]

    assert train["uid"].is_unique, "rct 训练集每人应恰 1 行"
    share = train["arm"].value_counts() / len(train)
    assert len(share) == len(ARMS) and share.between(0.25, 0.42).all(), \
        f"RCT 分配明显不均匀：{share.to_dict()}"

    eff = tables["effects"].set_index(["uid", "arm"])
    for r in train.itertuples():
        src = eff.loc[(r.uid, r.arm)]
        expect = src["y_void_window"] if r.arm == "control" else src["y_treated_window"]
        assert abs(r.y - expect) < 1e-9, f"({r.uid}, {r.arm}) 观测标签与臂不对应"

    assert set(train["uid"]).isdisjoint(set(hold["uid"])), "训练与留出用户重叠"
    assert set(train["uid"]) | set(hold["uid"]) == set(tables["users"]["uid"]), "切分未覆盖全人口"
    assert len(hold) == 3 * hold["uid"].nunique(), "留出集应每用户 × 全部臂"
    assert sorted(hold["arm"].unique()) == list(ARMS)
    assert meta["train_rows"] == len(train) == meta["n_train_users"]

    users_train, _ = split_users(tables["users"], 0.3, SEED_TEST)
    a1 = assign_rct(users_train, SEED_TEST)
    a2 = assign_rct(users_train, SEED_TEST)
    a3 = assign_rct(users_train, 12)
    assert a1.equals(a2), "RCT 分配不可复现"
    assert not a1.equals(a3), "换种子分配居然不变"


# ── 3. full 协议 ────────────────────────────────────────────

def test_full_protocol():
    tables, _ = _tables()
    obs = build_observations(tables, 0.3, SEED_TEST, "full")
    train, meta = obs["train"], obs["meta"]
    assert len(train) == 3 * meta["n_train_users"], "full 应每训练用户 3 行"
    assert sorted(train["arm"].unique()) == list(ARMS)
    assert (train["arm"].value_counts() == meta["n_train_users"]).all(), "每臂应覆盖全部训练用户"

    eff = tables["effects"].set_index(["uid", "arm"])
    for r in train.itertuples():
        src = eff.loc[(r.uid, r.arm)]
        expect = src["y_void_window"] if r.arm == "control" else src["y_treated_window"]
        assert abs(r.y - expect) < 1e-9, f"({r.uid}, {r.arm}) full 标签口径漂移"


# ── 4. 无泄漏 ───────────────────────────────────────────────

def test_no_leakage():
    assert FEATURES == ("act_30d", "silence_days", "interest_concentration", "tenure_days")
    tables, _ = _tables()
    forbidden = ("tau_struct", "tau_ind", "increment", "reward", "cost", "a_daily",
                 "y_void_window", "y_treated_window")
    for proto in PROTOCOLS:
        obs = build_observations(tables, 0.3, SEED_TEST, proto)
        assert list(obs["train"].columns) == list(TRAIN_COLUMNS), f"{proto} 训练帧列集合漂移"
        assert list(obs["holdout"].columns) == list(HOLDOUT_COLUMNS)
        for col in forbidden:
            assert col not in obs["train"].columns, f"{proto} 训练帧混入真值列 {col}"

    # 模型只认特征白名单：把真值列硬塞进去也不影响预测
    obs = build_observations(tables, 0.3, SEED_TEST, "rct")
    model = TLearnerUplift("hgb", SEED_TEST).fit(obs["train"])
    x = obs["holdout"].loc[:, list(FEATURES)]
    x_adv = x.assign(tau_struct=999.0, reward=123.0)
    for arm in TREATED_ARMS:
        assert np.allclose(model.predict_tau(x, arm), model.predict_tau(x_adv, arm)), \
            f"{arm}：预测被白名单外的列影响（潜在泄漏）"


# ── 5. 策略端点 ─────────────────────────────────────────────

def test_policy_endpoints():
    res = _run()["results"]["rct"]
    pol, hold = res["policy"], res["holdout"]
    n_policies = len(policy_names(("hgb", "ridge")))
    # 注意：策略值落盘保留 4 位小数，端点断言用 1e-3 容差（远小于策略间差异）
    for arm in TREATED_ARMS:
        mean_reward = float(hold.loc[hold["arm"] == arm, "reward"].mean())
        sub = pol[(pol["arm"] == arm) & (pol["k_pct"] == 100)]
        assert len(sub) == n_policies, "k=100% 应包含全部策略"
        assert np.allclose(sub["value"].to_numpy(), mean_reward, atol=1e-3), \
            "k=100% 端点：所有策略都应等于该臂均值"
        rnd = pol[(pol["arm"] == arm) & (pol["policy"] == "random")]
        assert np.allclose(rnd["value"].to_numpy(), mean_reward, atol=1e-3), \
            "random 的解析期望应等于该臂均值且与 k 无关"


# ── 6. 模型信号 ─────────────────────────────────────────────

def test_model_signal():
    res = _run()["results"]
    fid, pol = res["rct"]["fidelity"], res["rct"]["policy"]

    row = fid[(fid["arm"] == "recall") & (fid["learner"] == "hgb")].iloc[0]
    assert row["spearman_struct"] >= 0.45, f"召回臂 τ̂ 排序信号过弱：{row['spearman_struct']}"
    assert abs(row["bias"]) <= 0.9, f"召回臂 τ̂ 平均偏差过大：{row['bias']}"

    def value(arm: str, policy: str, k: int) -> float:
        return float(pol[(pol["arm"] == arm) & (pol["policy"] == policy)
                         & (pol["k_pct"] == k)]["value"].iloc[0])

    assert value("recall", "model_hgb", 20) >= value("recall", "random", 20) + 0.5, \
        "召回臂定向触达未明显优于不排序"
    assert value("recall", "oracle_struct", 20) >= value("recall", "random", 20), \
        "结构 oracle 不应差于随机"

    # 反事实可观测性的价值：full 协议保真度应优于 rct（同种子同人口同模型）
    for arm in TREATED_ARMS:
        rct_row = fid[(fid["arm"] == arm) & (fid["learner"] == "hgb")].iloc[0]
        full_fid = res["full"]["fidelity"]
        full_row = full_fid[(full_fid["arm"] == arm) & (full_fid["learner"] == "hgb")].iloc[0]
        assert full_row["spearman_struct"] >= rct_row["spearman_struct"], f"{arm}：full 保真度未优于 rct"
        assert abs(full_row["bias"]) <= abs(rct_row["bias"]) + 1e-9, f"{arm}：full 偏差未优于 rct"


# ── 7. 边界与显式报错 ───────────────────────────────────────

def test_boundaries():
    def expect_error(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except ValueError:
            return
        raise AssertionError(f"{fn} 未按预期抛 ValueError")

    tables, _ = _tables()
    expect_error(build_observations, tables, 0.0, SEED_TEST, "rct")
    expect_error(build_observations, tables, 1.0, SEED_TEST, "rct")
    expect_error(build_observations, tables, 0.3, SEED_TEST, "nope")
    expect_error(split_users, tables["users"], 0.0, SEED_TEST)
    expect_error(split_users, tables["users"].head(5), 0.3, SEED_TEST)  # 用户数不足

    expect_error(run_all, **{**COMMON, "protocols": ("nope",)})
    expect_error(run_all, **{**COMMON, "learners": ("svm",)})
    expect_error(run_all, **{**COMMON, "ks": (0, 20)})
    expect_error(run_all, **{**COMMON, "n_users": 5})

    expect_error(make_regressor, "nope", 1)
    expect_error(TLearnerUplift, "nope", 1)

    obs = build_observations(tables, 0.3, SEED_TEST, "rct")
    expect_error(TLearnerUplift("hgb", SEED_TEST).fit, obs["train"].drop(columns=["arm"]))
    missing_arm = obs["train"][obs["train"]["arm"] != "recall"]
    expect_error(TLearnerUplift("hgb", SEED_TEST).fit, missing_arm)
    expect_error(calibration_table, obs["holdout"], ("hgb",), 1)
    x = obs["holdout"].loc[:, list(FEATURES)]
    expect_error(TLearnerUplift("hgb", SEED_TEST).predict_tau, x, "rec")  # 未拟合
    expect_error(TLearnerUplift("hgb", SEED_TEST).predict_tau, x, "nope")


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
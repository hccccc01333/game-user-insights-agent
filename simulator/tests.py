# -*- coding: utf-8 -*-
"""S1 模拟器 · 冒烟与一致性测试（无外部依赖，纯断言）。

运行（在仓库根目录，任选其一）：
    python -m simulator.tests
    pytest simulator/tests.py

覆盖：
    1. 确定性：同参同种子双跑一致；换种子结果不同
    2. 配对性质：对照臂增量恒 0；因 CRN，所有臂增量 ≥ 0；reward ≥ −cost
    3. 条数采样器校准：恰 1 条占比 76%、p90=3、p99=9（对齐 Reddit 锚点）
    4. 效应单调性：召回对"沉默 90 天上下"最有效、两端衰减；推荐随集中度/
       活跃度单调、随沉默衰减；对照臂恒 0
    5. 疲劳：同臂第二次触达效应严格变弱（1 + ρ 与单次之比）
    6. 边界：window>days / 触达日越界 / 未知臂 / 空人口 均显式报错
    7. 人口生成：离线默认路径可用且确定；本地有 L2 特征表时走 fit 路径
"""
from __future__ import annotations

import numpy as np

from . import params as P
from . import reward as R
from . import response
from .env import SimEnv, sample_count
from .run_sim import DEFAULT_FEATURES, build_tables
from .user_generator import SimUser, generate_users

# 测试用小参数（跑得快）
TINY = dict(n_users=40, days=7, seed=11, window=7)


def _user(**kw) -> SimUser:
    base = dict(uid="sim_test", a_daily=0.01, silence_days=80.0, act_30d=0.0,
                interest_concentration=0.12, tenure_days=1200.0)
    base.update(kw)
    return SimUser(**base)


# ── 1. 确定性 ───────────────────────────────────────────────

def test_determinism():
    kw = dict(TINY, features_path=None, use_fit=False)
    t1, _ = build_tables(**kw)
    t2, _ = build_tables(**kw)
    assert all(t1[k].equals(t2[k]) for k in t1), "同参同种子双跑不一致"

    t3, _ = build_tables(**{**kw, "seed": 12})
    assert not t1["effects"].equals(t3["effects"]), "换种子结果居然一致"


# ── 2. 配对性质 ─────────────────────────────────────────────

def test_pairing_properties():
    tables, _ = build_tables(**{**TINY, "features_path": None, "use_fit": False})
    eff = tables["effects"]
    ctrl = eff[eff["arm"] == "control"]
    assert (ctrl["increment"] == 0).all() and (ctrl["reward"] == 0).all(), "对照臂应为恒零"
    assert (eff["increment"] >= 0).all(), "CRN 配对下增量不应为负"
    for arm, cost in P.SIM.cost_events.items():
        sub = eff[eff["arm"] == arm]
        assert (sub["reward"] >= -cost - 1e-9).all(), f"{arm} 的 reward 低于 −cost"
    # 场外口径：reward = increment − cost
    sub = eff[eff["arm"] == "recall"]
    assert np.allclose(sub["reward"], sub["increment"] - R.marginal_cost("recall")), "reward 口径漂移"


# ── 3. 条数采样器校准（对齐 Reddit 锚点）────────────────────

def test_count_sampler_calibration():
    rng = np.random.default_rng(20261005)
    n = 200_000
    draws = np.array([sample_count(rng) for _ in range(n)], dtype=float)
    share1 = float((draws == 1).mean())
    p90 = float(np.quantile(draws, 0.90, method="lower"))
    p99 = float(np.quantile(draws, 0.99, method="lower"))
    assert abs(share1 - 0.76) < 0.01, f"恰 1 条占比 {share1:.3f} 偏离锚点 0.76"
    assert p90 == 3.0, f"p90={p90} 偏离锚点 3"
    assert p99 == 9.0, f"p99={p99} 偏离锚点 9"


# ── 4. 效应单调性 ───────────────────────────────────────────

def test_effect_monotonicity():
    u_silent = _user(silence_days=90.0)
    assert response.structural_effect(u_silent, "control") == 0.0
    # 召回：90 天附近最强，两端衰减（刚活跃过 / 沉默过久都难召回）
    t90 = response.tau_recall(u_silent)
    assert t90 > response.tau_recall(_user(silence_days=15.0)) > 0
    assert t90 > response.tau_recall(_user(silence_days=600.0)) > 0
    # 召回：新账号更易召回
    assert response.tau_recall(_user(silence_days=90, tenure_days=300)) > t90
    # 推荐：随集中度、随近 30 天活跃度单调递增；随沉默时长单调递减
    base = _user(act_30d=3.0, silence_days=10.0, interest_concentration=0.06)
    assert response.tau_rec(_user(act_30d=3.0, silence_days=10.0, interest_concentration=0.15)) > response.tau_rec(base)
    assert response.tau_rec(_user(act_30d=8.0, silence_days=10.0, interest_concentration=0.06)) > response.tau_rec(base)
    assert response.tau_rec(_user(act_30d=3.0, silence_days=40.0, interest_concentration=0.06)) < response.tau_rec(base)
    # 推荐：对完全沉默（act_30d=0）严格为零——那是召回臂的活
    assert response.tau_rec(_user(act_30d=0.0)) == 0.0


# ── 5. 疲劳 ─────────────────────────────────────────────────

def test_fatigue():
    u = _user(silence_days=80.0)
    one = response.effect_schedule(u, [(0, "recall")], days=14, noise_by_arm={})
    two = response.effect_schedule(u, [(0, "recall"), (1, "recall")], days=14, noise_by_arm={})
    s1, s2 = sum(one), sum(two)
    assert s1 > 0
    # 第二次乘 ρ^(2-1)=0.5，且权重归一 → 总量严格小于"两次满效应"
    assert abs(s2 / s1 - (1 + P.SIM.fatigue_rho)) < 1e-9, f"疲劳乘子不符：{s2 / s1:.4f}"


# ── 6. 边界与显式报错 ───────────────────────────────────────

def test_boundaries():
    def expect_error(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except ValueError:
            return
        raise AssertionError(f"{fn} 未按预期抛 ValueError")

    expect_error(build_tables, **{**TINY, "window": 8, "features_path": None, "use_fit": False})
    expect_error(build_tables, **{**TINY, "n_users": 0, "features_path": None, "use_fit": False})
    expect_error(generate_users, 0, 1, None)
    expect_error(R.window_increment, [1, 2], [0, 0], 3)
    expect_error(R.marginal_cost, "nope")

    pop = generate_users(8, 3, None)
    env = SimEnv(pop.users, 3)
    expect_error(env.rollout, 0, [(2, "rec")], 2)          # 触达日越界
    expect_error(env.rollout, 0, [(0, "nope")], 4)          # 未知臂
    expect_error(env.rollout, 0, [], 0)                     # 天数非法
    assert (env.rollout(0, [], 3) == env.baseline_rollout(0, 3)), "基线与对照应完全一致（同一随机流）"


# ── 7. 人口生成 ─────────────────────────────────────────────

def test_population():
    pop_a = generate_users(64, 5, None)
    pop_b = generate_users(64, 5, None)
    assert [u.__dict__ for u in pop_a.users] == [u.__dict__ for u in pop_b.users], "默认路径不确定"
    assert all(u.uid.startswith("sim_") for u in pop_a.users), "uid 必须带 sim_ 前缀"
    assert all(0.0 < u.a_daily <= 0.6 for u in pop_a.users)
    assert all(u.silence_days >= 0 and u.tenure_days > 0 for u in pop_a.users)
    assert pop_a.meta["path"] == "default"

    if DEFAULT_FEATURES.exists():
        pop_fit = generate_users(64, 5, DEFAULT_FEATURES)
        assert pop_fit.meta["path"] == "fit" and "features_fingerprint" in pop_fit.meta
        assert all(0.0 < u.a_daily <= 0.6 for u in pop_fit.users)
    else:  # pragma: no cover - 公开仓离线场景
        pop_fb = generate_users(32, 5, DEFAULT_FEATURES)  # 文件缺失 → 显式兜底
        assert pop_fb.meta["path"] == "default" and "fallback_reason" in pop_fb.meta


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
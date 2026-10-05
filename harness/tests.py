# -*- coding: utf-8 -*-
"""S4 harness · 冒烟与一致性测试（无外部依赖，纯断言）。

运行（在仓库根目录，任选其一）：
    python -m harness.tests
    pytest harness/tests.py

覆盖：
    1. 确定性：同参同种子双跑逐值一致；换种子轨迹不同
    2. 无泄漏：把"未观测 (i, arm)"的奖励全部篡改后同种子重跑，决策逐值相同
    3. 门控行为：探索额度为 0 时否决冻结（全对照）；额度 > 0 时仅小额放行；
       证据足够（过 min_obs 门槛且下界转正）后门自动打开、否决归零
    4. 校验器：finish 前置确认干预完成；批次结构非法被拒；契约字段核对
    5. 契约与前置条件：六环节工具名与 Critic 契约一致；when 链收窄；
       名额跑完后干预工具退出候选；圈人只落在在线池且满足可观测规则
    6. 端到端：六环节顺序执行 + 收工通过校验；干预批次人数；
       no_critic 路径零否决；两条路径审计得分都在 (0, 1] 内
"""
from __future__ import annotations

from functools import lru_cache

from bandit.policies import make_policy
from bandit.protocol import ARMS, BanditWorld, build_world
from simulator.run_sim import build_tables

from .bandit_tools import INTERVENTION_TOOL, InterventionConsole
from .critic import (
    DEFAULT_MIN_OBS, REQUIRED_FIELDS, CriticVerifier, InterventionCritic,
)
from .planner import FINISH_TOOL
from .run_agent import COHORT_NAME, build_registry, run_agent_demo, run_is_deterministic
from .state import AgentState

# 测试用小人口（跑得快；逐值断言，非阈值型）
N_TEST = 800
SEED_TEST = 13
DAYS = 14
WIN = 7
COMMON = dict(
    n_users=N_TEST, days=DAYS, window=WIN, seed=SEED_TEST, audit=0.3,
    policy="linucb", basis="logsq", batch=120,
    critic_beta=1.0, critic_min_obs=DEFAULT_MIN_OBS,
    critic_explore_frac=0.05, critic_explore_min=2,
    max_steps=8, features_path=None, use_fit=False,
)


@lru_cache(maxsize=1)
def _tables() -> tuple:
    return build_tables(N_TEST, DAYS, SEED_TEST, WIN, None, False)


@lru_cache(maxsize=1)
def _built() -> dict:
    tables, _ = _tables()
    return build_world(tables, 0.3, SEED_TEST, "logsq")


def _console(world=None, stream=None, critic=None, policy="linucb"):
    """造一个干预台（默认挂全局共享世界；stream 给定时直接圈人）。"""
    built = _built()
    world = world if world is not None else built["world"]
    pol = make_policy(policy, context_dim=built["meta"]["context_width"],
                      seed=SEED_TEST, oracle_arm=world.oracle_arm)
    console = InterventionConsole(world, pol, critic)
    if stream is not None:
        console.open_cohort(list(stream), "测试人群")
    return console, pol


def expect_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - 测试助手需要吞掉预期异常
        return
    raise AssertionError(f"未按预期报错：{fn}")


# ── 1. 确定性 ───────────────────────────────────────────────

def test_determinism():
    a = run_agent_demo(**COMMON)
    b = run_agent_demo(**COMMON)
    assert run_is_deterministic(a, b), "同参同种子双跑不一致"
    c = run_agent_demo(**{**COMMON, "seed": 12})
    assert a["traces"] != c["traces"], "换种子轨迹居然一致"
    assert not run_is_deterministic(a, c), "换种子居然判定为一致"


# ── 2. 无泄漏（未观测臂篡改不影响决策）──────────────────────

def test_no_leak_tamper():
    built = _built()
    world = built["world"]
    stream = list(built["online"][:120])
    world.reset_revealed()
    c1, _ = _console(stream=stream)
    c1.allocate(120)
    revealed = {(int(i), arm) for i, arm in world.revealed}
    assert len(revealed) == 120, "每人次应恰好观测一条臂"

    rewards = world.rewards.copy()
    for j, arm in enumerate(world.arms):
        for i in range(len(world.uid)):
            if (i, arm) not in revealed:
                rewards[i, j] = 1e6  # 未观测臂全部篡改为"天上掉馅饼"
    world2 = BanditWorld(world.uid, world.context, rewards)
    c2, _ = _console(world=world2, stream=stream)
    c2.allocate(120)

    r1 = c1.rows_frame()
    r2 = c2.rows_frame()
    assert r1["proposed_arm"].tolist() == r2["proposed_arm"].tolist(), "未观测臂篡改影响了提议：存在泄漏"
    assert r1["arm"].tolist() == r2["arm"].tolist(), "未观测臂篡改影响了决策：存在泄漏"


# ── 3. 门控行为 ─────────────────────────────────────────────

def test_gate_freeze():
    """探索额度为 0：证据不足时门全部关闭 → 整批降级对照。"""
    built = _built()
    dim = built["meta"]["context_width"]
    critic = InterventionCritic(ARMS, dim, explore_frac=0.0, explore_min=0)
    console, _ = _console(stream=list(built["online"][:50]), critic=critic)
    obs = console.allocate(50)
    assert obs["arms"]["control"] == 50, f"额度为 0 时应全对照（当前 {obs['arms']}）"
    assert obs["veto_count"] > 0 and obs["explore_count"] == 0, "应当全部走否决路径"


def test_gate_quota_and_open():
    """探索额度 > 0：只放行小额试错；证据足够（过 min_obs 门槛且下界转正）后门自动打开。

    min_obs=20 构造严格场景：批内最多 10 个探索观测，任何臂都到不了门槛，
    价值门在批 1 完全关闭——保底断言"额度 = 非对照放行数"。
    """
    built = _built()
    dim = built["meta"]["context_width"]
    critic = InterventionCritic(ARMS, dim, min_obs=20, explore_frac=0.2, explore_min=0)
    console, pol = _console(stream=list(built["online"][:80]), critic=critic)
    obs1 = console.allocate(50)
    n_off = obs1["batch_size"] - obs1["arms"]["control"]
    assert n_off == 10, f"额度 10 应恰好放行 10 个非对照（当前 {n_off}）"
    assert obs1["explore_count"] == 10, f"探索放行数应恰为额度（当前 {obs1['explore_count']}）"
    assert obs1["veto_count"] == 32, f"批 1 其余非对照提议应全被否决（当前 {obs1['veto_count']}）"

    # 快速注入"召回值得"的知识（Actor 与 Critic 看到同一份观测，且跨多样上下文），门应打开
    world = built["world"]
    recall = ARMS.index("recall")
    for i in built["online"][:40]:
        x = world.context_of(int(i))
        critic.observe(x, recall, 2.0)
        pol.update(x, recall, 2.0)
    obs2 = console.allocate(30)
    assert obs2["arms"]["recall"] == 30, f"门打开后应放行召回臂（当前 {obs2['arms']}）"
    assert obs2["veto_count"] == 0, "证据足够时不应再有否决"


def test_critic_unit():
    dim = _built()["meta"]["context_width"]
    critic = InterventionCritic(ARMS, dim)
    x = _built()["world"].context_of(0)
    chosen, info = critic.review(x, ARMS.index("control"))
    assert chosen == ARMS.index("control") and info["action"] == "control", "提议对照应恒放行"
    chosen, info = critic.review(x, ARMS.index("rec"))
    assert chosen == ARMS.index("control") and info["action"] in ("explore", "veto"), "未知臂应被拦下"

    # 证据门槛：min_obs=2 时，观测 0/1 次都被挡在价值门外；到 2 次且下界转正才放行
    strict = InterventionCritic(ARMS, dim, min_obs=2, explore_frac=0.0, explore_min=0)
    rec = ARMS.index("rec")
    world = _built()["world"]
    xa, xb = world.context_of(0), world.context_of(1)
    chosen, info = strict.review(xa, rec)
    assert chosen == ARMS.index("control") and info["action"] == "veto" and info["n_obs"] == 0, \
        "零证据时不得开价值门"
    strict.observe(xa, rec, 4.0)
    chosen, info = strict.review(xb, rec)
    assert chosen == ARMS.index("control") and info["n_obs"] == 1, "1 次观测仍应在门槛外"
    strict.observe(xb, rec, 4.0)
    chosen, info = strict.review(xa, rec)
    assert chosen == rec and info["action"] == "pass", "过门槛且下界转正应放行"
    assert strict.stats()["n_obs"]["rec"] == 2, "观测计数应与注入一致"

    expect_error(critic.review, x, 5)
    expect_error(critic.observe, x, -1, 0.0)
    expect_error(critic.new_batch, 0)
    expect_error(InterventionCritic, ARMS, dim, beta=0.0)
    expect_error(InterventionCritic, ARMS, dim, ridge=-1.0)
    expect_error(InterventionCritic, ARMS, dim, explore_frac=1.5)
    expect_error(InterventionCritic, ARMS, dim, explore_min=-1)
    expect_error(InterventionCritic, ARMS, dim, min_obs=-1)
    expect_error(InterventionCritic, ("rec", "recall"), dim)  # 缺对照臂


# ── 4. 校验器 ───────────────────────────────────────────────

def test_verifier():
    v = CriticVerifier()
    ok, note = v.check({"tool": FINISH_TOOL}, None)
    assert not ok and "干预" in note, "未完成干预前不得收工"

    good = {"batch": 1, "batch_size": 2, "arms": {"control": 1, "rec": 1},
            "mean_reward": 0.1, "cum_reward": 0.1, "veto_count": 0, "veto_rate": 0.0,
            "explore_count": 0, "policy": "linucb", "cohort": "x", "stream_remaining": 3}
    ok, _ = v.check({"tool": INTERVENTION_TOOL}, good)
    assert ok and v.intervention_ok, "合法批次应通过并解锁收工"
    ok, _ = v.check({"tool": FINISH_TOOL}, None)
    assert ok, "干预完成后应收工通过"

    bad_sum = {**good, "batch_size": 2, "arms": {"control": 1, "rec": 2}}
    ok, note = v.check({"tool": INTERVENTION_TOOL}, bad_sum)
    assert not ok and "计数" in note
    bad_rate = {**good, "veto_rate": 1.5}
    ok, _ = v.check({"tool": INTERVENTION_TOOL}, bad_rate)
    assert not ok, "非法 veto_rate 应被拒"
    missing = {k: v_ for k, v_ in good.items() if k != "policy"}
    ok, note = v.check({"tool": INTERVENTION_TOOL}, missing)
    assert not ok and "契约字段" in note
    ok, _ = v.check({"tool": "detect_anomaly"}, {"metric": "m", "delta_pct": 1.0})
    assert not ok, "缺字段的环节产物应被拒"


# ── 5. 契约与前置条件 ───────────────────────────────────────

def test_contract_and_preconditions():
    tables, _ = _tables()
    built = _built()
    console, _ = _console()  # 未圈人
    reg = build_registry(tables, built, console, batch=100)
    assert set(reg._tools) == set(REQUIRED_FIELDS), "六环节工具名与 Critic 契约不一致"

    state = AgentState(goal="t")
    assert reg.available(state) == ["detect_anomaly"], "首步只应开放异常检测"
    state.record(step=1, decision={"tool": "detect_anomaly"}, observation={}, verified=True, note="")
    assert "locate_cohort" in reg.available(state)
    assert INTERVENTION_TOOL not in reg.available(state), "未圈人前干预工具不可用"

    obs = reg.call("locate_cohort")
    assert obs["cohort"] == COHORT_NAME and obs["size"] > 0
    state.record(step=2, decision={"tool": "locate_cohort"}, observation=obs, verified=True, note="")
    assert INTERVENTION_TOOL in reg.available(state)

    # 圈人合规：只落在在线池，且满足可观测规则
    users = tables["users"]
    idx = console.cohort_index()
    online = {int(i) for i in built["online"]}
    assert set(idx) <= online, "人群越出在线池"
    assert all(
        users.iloc[i]["act_30d"] == 0 and users.iloc[i]["silence_days"] >= 90 for i in idx
    ), "人群不满足可观测规则"

    # 名额跑完 → 干预工具退出候选
    while console.remaining() > 0:
        console.allocate(min(200, console.remaining()))
    assert INTERVENTION_TOOL not in reg.available(state), "名额跑完后干预工具不应仍在候选"
    expect_error(console.allocate, 10)
    expect_error(console.open_cohort, idx, "另一个人群")


# ── 6. 端到端 ───────────────────────────────────────────────

def test_e2e():
    run = run_agent_demo(**COMMON)
    wc, nc = run["modes"]["with_critic"], run["modes"]["no_critic"]
    assert wc["finished"] and nc["finished"], "两条路径都应在 max_steps 内收工"

    tools = [(h["decision"] or {}).get("tool") for h in run["traces"]["with_critic"]]
    assert tools == [
        "detect_anomaly", "locate_cohort", "analyze_cause", "assess_risk",
        INTERVENTION_TOOL, "design_experiment", FINISH_TOOL,
    ], f"六环节顺序异常：{tools}"
    assert all(h.get("verified") for h in run["traces"]["with_critic"]), "有步骤未通过校验"

    alloc = [h["observation"] for h in run["traces"]["with_critic"]
             if (h["decision"] or {}).get("tool") == INTERVENTION_TOOL][0]
    assert alloc["batch_size"] == COMMON["batch"], "干预批次人数异常"
    assert alloc["arms"]["control"] >= 1, "批次应含对照名额"

    assert nc["online"]["veto_count"] == 0 and nc["online"]["veto_rate"] == 0.0, "无安全门路径不应有否决"
    assert wc["online"]["veto_count"] > 0, "带安全门路径应出现否决"
    for mode in ("with_critic", "no_critic"):
        frac = run["modes"][mode]["audit"]["audit_vs_oracle_frac"]
        assert 0.0 < frac <= 1.0, f"{mode} 审计得分异常：{frac}"


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
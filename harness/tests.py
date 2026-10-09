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
    7. 真实主链：facts→六环节顺序 + artifacts 键=工具名；双跑确定性；
       干预队列分档数学 + 缺分数兜底；verifier 收工门槛与坏批次拒绝；
       缺 facts 文件显式报错（纯内存 fixture，不依赖 data/processed）
    8. LLM 适配器：本地假 HTTP 服务验证重试 / 用量统计 / 缺密钥显式报错
    9. Registry：Schema 校验、权限收窄、越权与坏参数的错误类别
   10. 循环控制：工具预算、单工具失败熔断、计划修订（replan）、错误信封
   11. Verifier 业务断言：分档不闭合 / 占比越界被拒
   12. 追踪器：kind 分布与用量汇总（不进确定性轨迹）
   13. 业务工具层：8 工具直调（圈人→队列→报告）、Schema / 权限 / 候选收窄、
       预测件缺失不注册、故障注入（超时）后任务仍可收工
   14. 洞察主链（insightv0）：mock 双跑确定性、工具链顺序、InsightVerifier 收工门
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from bandit.policies import make_policy
from bandit.protocol import ARMS, BanditWorld, build_world
from simulator.run_sim import build_tables

from .bandit_tools import INTERVENTION_TOOL, InterventionConsole
from .critic import (
    DEFAULT_MIN_OBS, REQUIRED_FIELDS, CriticVerifier, InterventionCritic,
)
from .loop import run
from .model import LLMAdapter, ModelConfigError, MockModel
from .planner import FINISH_TOOL, Planner
from .registry import ToolCallError, ToolRegistry
from .state import AgentState
from .tracing import Tracer
from .facts import FactsStore, load_facts
from .run_agent import COHORT_NAME, build_registry, run_agent_demo, run_is_deterministic
from .run_real_agent import (
    REQUIRED_FIELDS as REAL_REQUIRED_FIELDS,
    RealAgentConfig,
    RealChainVerifier,
    run_is_deterministic as real_run_is_deterministic,
    run_real_agent,
)
from .run_insight_agent import (
    mock_args,
    run_insight_agent,
    run_is_deterministic as insight_run_is_deterministic,
)
from .tools import TOOL_NAMES, InsightToolkit, InsightVerifier, build_insight_registry

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


# ── 7. 真实主链（facts → Agent 决策；纯内存 fixture，不依赖 data/processed）──

def _real_fact(stage: str, risk: float | None) -> dict:
    """造一条最小可用的 L3 fact（字段与 facts.jsonl 契约对齐）。"""
    return {
        "facts_version": "l3v1",
        "as_of": 1791108404,
        "quality": {"usable": True, "truncated_surfaces": ""},
        "churn": {
            "stage": stage, "risk_score": risk, "horizon_days": "30",
            "drivers": ["staleness", "momentum"],
        },
        "activity": {"band": "dormant", "score": 10.0},
        "migration": {"insufficient_data": True, "genre_shift_score": None, "game_flow_net": 0},
        "context": {"sentiment_neg_rate": 0.1, "top_genres": [["二次元", 0.1]]},
    }


def _real_store() -> FactsStore:
    """12 用户 fixture：8 人落人群档位（d60×5 / d90×3）+ 4 人对照档位。"""
    facts = {
        "h_01": _real_fact("dormant_60", 90.0),  # recall
        "h_02": _real_fact("dormant_90", 85.0),  # recall
        "h_03": _real_fact("dormant_60", 81.0),  # recall
        "h_04": _real_fact("dormant_90", 70.0),  # rec
        "h_05": _real_fact("dormant_60", 60.0),  # rec
        "h_06": _real_fact("dormant_60", None),  # 缺分数 → control（批内第 6 顺位）
        "h_07": _real_fact("dormant_90", 30.0),  # control（第 7 顺位，不在批内）
        "h_08": _real_fact("dormant_60", 40.0),  # control（第 8 顺位，不在批内）
        "h_09": _real_fact("silent", 95.0),      # 非人群档位（只为沉默占比贡献）
        "h_10": _real_fact("active", 10.0),
        "h_11": _real_fact("dormant_30", 55.0),
        "h_12": _real_fact("churned", 88.0),
    }
    return FactsStore(facts)


def test_real_chain_flow():
    result = run_real_agent(_real_store(), RealAgentConfig(batch=6))
    state = result["state"]
    tools = [(h["decision"] or {}).get("tool") for h in state.history]
    assert tools == [
        "detect_anomaly", "locate_cohort", "analyze_cause", "assess_risk",
        "plan_interventions", "design_experiment", FINISH_TOOL,
    ], f"六环节顺序异常：{tools}"
    assert all(h.get("verified") for h in state.history), "有步骤未通过校验"
    assert set(state.artifacts) == set(REAL_REQUIRED_FIELDS), "artifacts 键应为六环节工具名"
    assert state.context is not None, "context 未透传业务输入（facts 存储）"
    assert result["finished"], "主链应在 max_steps 内收工"

    plan = state.artifacts["plan_interventions"]
    assert plan["batch_size"] == 6, "批大小异常"
    assert plan["arms"] == {"control": 1, "rec": 2, "recall": 3}, f"分档结果异常：{plan['arms']}"
    assert plan["stream_remaining"] == 2, "剩余名额异常"

    design = state.artifacts["design_experiment"]
    assert design["n_candidates"] == 5, "触达候选数异常"
    assert design["n_treated"] + design["n_control"] == 5, "分组数应等于候选数"


def test_real_chain_determinism():
    a = run_real_agent(_real_store(), RealAgentConfig(batch=6))
    b = run_real_agent(_real_store(), RealAgentConfig(batch=6))
    assert real_run_is_deterministic(a, b), "真实主链双跑不一致"
    assert a["experiment_id"] == "exp-1791108404-silence-recall-v0", "实验 id 派生异常"
    assert len(a["queue"]) == 6 and len(a["assignments"]) == 5, "队列/分组行数异常"


def test_real_plan_math():
    result = run_real_agent(_real_store(), RealAgentConfig(batch=6))
    queue = result["queue"]
    assert [r["arm"] for r in queue] == ["recall", "recall", "recall", "rec", "rec", "control"]
    missing = [r for r in queue if r["risk_score"] is None]
    assert len(missing) == 1 and missing[0]["arm"] == "control", "缺分数应兜底不打扰"
    assert all(r["policy"] == "risk_band_heuristic_v0" for r in queue), "队列应带规则版本"

    groups = {r["uid_hash"]: r["group"] for r in result["assignments"]}
    assert set(groups) == {f"h_0{i}" for i in range(1, 6)}, "候选应为本批触达行"
    for row in result["assignments"]:
        digest = hashlib.sha256(
            f"{row['uid_hash']}|{result['experiment_id']}".encode("utf-8")
        ).hexdigest()
        expect = "holdout" if int(digest[:8], 16) % 100 < 50 else "treatment"
        assert row["group"] == expect, "哈希分流与规则实现不符"


def test_real_verifier():
    v = RealChainVerifier()
    ok, note = v.check({"tool": FINISH_TOOL}, None)
    assert not ok and "干预" in note, "未完成干预队列前不得收工"

    good = {
        "batch": 1, "batch_size": 2, "arms": {"control": 1, "rec": 1, "recall": 0},
        "cohort": "x", "stream_remaining": 3,
        "policy": "risk_band_heuristic_v0", "rule_version": "risk_band_heuristic_v0",
    }
    ok, _ = v.check({"tool": "plan_interventions"}, good)
    assert ok and v.intervention_ok, "合法批次应通过并解锁收工"
    ok, _ = v.check({"tool": FINISH_TOOL}, None)
    assert ok, "干预队列完成后应收工通过"

    ok, _ = RealChainVerifier().check({"tool": "plan_interventions"}, {**good, "batch_size": 3})
    assert not ok, "臂计数不闭合应被拒"
    ok, _ = RealChainVerifier().check(
        {"tool": "plan_interventions"}, {k: val for k, val in good.items() if k != "policy"}
    )
    assert not ok, "缺契约字段应被拒"

    empty = {
        "experiment_id": "e", "grouping": "A/B", "primary_metric": "m",
        "n_treated": 0, "n_control": 0, "assignment_rule": "r",
    }
    ok, note = RealChainVerifier().check({"tool": "design_experiment"}, empty)
    assert not ok and "空" in note, "空分组应被拒"


def test_real_missing_file():
    try:
        load_facts(Path("__no_such_facts__.jsonl"))
    except FileNotFoundError:
        return
    raise AssertionError("缺 facts 文件应显式报 FileNotFoundError")


# ── 8. LLM 适配器（本地假 HTTP 服务，不触外网）──────────────

def _start_fake_llm(responses: list[tuple[int, dict]]):
    """起一个本地假 /chat/completions 服务；responses 按请求序号取（超出取最后一条）。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    state = {"responses": list(responses), "requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server 接口名
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            state["requests"] += 1
            idx = min(state["requests"] - 1, len(state["responses"]) - 1)
            status, payload = state["responses"][idx]
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # 静音
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def test_llm_adapter():
    ok_body = {
        "choices": [{"message": {"content": '{"tool": "finish"}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    server, state = _start_fake_llm([(429, {"error": "rate limited"}), (200, ok_body)])
    try:
        adapter = LLMAdapter(
            base_url=f"http://127.0.0.1:{server.server_port}",
            api_key="test-key", backoff=0.0, max_retries=2,
        )
        text = adapter.chat([{"role": "user", "content": "hi"}])
        assert text == '{"tool": "finish"}', f"应答解析异常：{text!r}"
        assert state["requests"] == 2, "429 应先重试一次再成功"
        stats = adapter.stats()
        assert stats["calls"] == 1 and stats["retries"] == 1, f"用量统计异常：{stats}"
        assert stats["prompt_tokens"] == 10 and stats["total_tokens"] == 15, f"token 统计异常：{stats}"
        assert stats["failures"] == 0
    finally:
        server.shutdown()

    # 不可重试错误（400）：一次即失败，不重试
    server2, state2 = _start_fake_llm([(400, {"error": "bad request"})])
    try:
        adapter2 = LLMAdapter(
            base_url=f"http://127.0.0.1:{server2.server_port}",
            api_key="test-key", backoff=0.0, max_retries=2,
        )
        expect_error(adapter2.chat, [{"role": "user", "content": "x"}])
        assert state2["requests"] == 1, "400 不应重试"
        assert adapter2.stats()["failures"] == 1
    finally:
        server2.shutdown()

    # 超重试仍失败：错误上抛，failures 计数
    server3, state3 = _start_fake_llm([(503, {"error": "unavailable"})])
    try:
        adapter3 = LLMAdapter(
            base_url=f"http://127.0.0.1:{server3.server_port}",
            api_key="test-key", backoff=0.0, max_retries=1,
        )
        expect_error(adapter3.chat, [{"role": "user", "content": "x"}])
        assert state3["requests"] == 2, "5xx 应重试 1 次（共 2 次请求）"
        assert adapter3.stats()["retries"] == 1 and adapter3.stats()["failures"] == 1
    finally:
        server3.shutdown()

    # 缺密钥：显式报 ModelConfigError（不做静默兜底）
    expect_error(LLMAdapter, api_key="")
    expect_error(LLMAdapter, api_key="k", timeout=0)


# ── 9. Registry：Schema / 权限 / 错误类别 ───────────────────

def test_registry_schema_and_permission():
    reg = ToolRegistry(granted_permissions={"facts:read"})
    reg.register(
        "read_thing", lambda uid: {"uid": uid}, "读（占位）",
        schema={"type": "object", "properties": {"uid": {"type": "string"}}, "required": ["uid"]},
        permission="facts:read",
    )
    reg.register("write_thing", lambda: {"ok": 1}, "写（占位）", permission="facts:write")
    state = AgentState(goal="t")
    assert reg.available(state) == ["read_thing"], "未授权工具不应进候选"
    assert reg.describe(["read_thing"]).startswith("- read_thing: 读（占位）（参数：uid: string（必填））")
    assert reg.call("read_thing", uid="u1") == {"uid": "u1"}

    for kwargs, kind in (({}, "invalid_args"), ({"uid": 7}, "invalid_args")):
        try:
            reg.call("read_thing", **kwargs)
        except ToolCallError as exc:
            assert exc.kind == kind, f"坏参数应记 {kind}：{exc.kind}"
        else:
            raise AssertionError(f"坏参数未被拦下：{kwargs}")
    try:
        reg.call("write_thing")
    except ToolCallError as exc:
        assert exc.kind == "permission"
    else:
        raise AssertionError("越权调用未被拦下")
    try:
        reg.call("nope")
    except ToolCallError as exc:
        assert exc.kind == "unknown"
    else:
        raise AssertionError("未登记工具未被拦下")

    # 工具自身报错 → execution 类别（原始异常挂在 __cause__）
    reg.register("boom", lambda: 1 / 0, "会炸（占位）")
    try:
        reg.call("boom")
    except ToolCallError as exc:
        assert exc.kind == "execution" and isinstance(exc.__cause__, ZeroDivisionError)
    else:
        raise AssertionError("工具报错应转 ToolCallError")

    # granted_permissions=None = 全部授予（兼容既有链路）
    reg2 = ToolRegistry()
    reg2.register("write_thing", lambda: {"ok": 1}, "写", permission="facts:write")
    assert reg2.available(AgentState(goal="t")) == ["write_thing"]


# ── 10. 循环控制：预算 / 熔断 / 计划修订 / 错误信封 ─────────

def test_loop_budget_and_breaker():
    def used(name):
        return lambda state: any((h.get("decision") or {}).get("tool") == name for h in state.history)

    reg = ToolRegistry()
    reg.register("a", lambda: {"v": 1}, "占位 a")
    reg.register("b", lambda: {"v": 2}, "占位 b", when=used("a"))
    state = run("t", reg, max_steps=5, max_tool_calls=1)
    kinds = [h["kind"] for h in state.history]
    assert kinds == ["tool_call", "blocked", "finish"], f"预算控制轨迹异常：{kinds}"
    assert "b" not in state.artifacts and state.artifacts.get("a") == {"v": 1}

    # 熔断：同一工具被反复选择（脚本模型），累计失败到阈值后不再执行
    def scripted(model_plan):
        """按脚本回信封的模型：每一项是 tool 名或 None（None → 非 JSON 垃圾输出）。"""

        class Scripted:
            def __init__(self):
                self.i = 0

            def chat(self, messages, **kwargs):
                tool = model_plan[min(self.i, len(model_plan) - 1)]
                self.i += 1
                if tool is None:
                    return "垃圾输出"
                return json.dumps({"tool": tool, "args": {}, "reason": "r"})

        return Scripted()

    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("炸了")

    reg2 = ToolRegistry()
    reg2.register("boom", boom, "占位爆炸")
    state2 = run("t", reg2, planner=Planner(model=scripted(["boom", "boom", "boom", FINISH_TOOL])), max_steps=6)
    kinds2 = [h["kind"] for h in state2.history]
    assert kinds2 == ["retry", "retry", "blocked", "finish"], f"熔断轨迹异常：{kinds2}"
    assert calls["n"] == 2, f"阈值=2 应恰好执行 2 次（实际 {calls['n']}）"
    assert "boom" not in state2.artifacts, "熔断后不落产物"
    assert "改道" in state2.history[2]["note"], "熔断应提示改道或收工"

    # 阈值=1：一次失败即熔断
    calls2 = {"n": 0}

    def boom_once():
        calls2["n"] += 1
        raise ValueError("炸了")

    reg3 = ToolRegistry()
    reg3.register("boom_once", boom_once, "占位爆炸")
    state3 = run("t", reg3, planner=Planner(model=scripted(["boom_once", "boom_once", FINISH_TOOL])),
                 max_steps=6, max_failures_per_tool=1)
    kinds3 = [h["kind"] for h in state3.history]
    assert kinds3 == ["retry", "blocked", "finish"], f"阈值=1 时应一次即熔断：{kinds3}"
    assert calls2["n"] == 1, "熔断后不应再执行"


def test_plan_revision_and_tracer():
    class PlanModel:
        def __init__(self):
            self.i = 0

        def chat(self, messages, **kwargs):
            self.i += 1
            if self.i == 1:
                return json.dumps({"tool": "a", "args": {}, "reason": "r", "plan": ["先 a", "再收工"]})
            return json.dumps({"tool": FINISH_TOOL, "args": {}, "reason": "done"})

    reg = ToolRegistry()
    reg.register("a", lambda: {"v": 1}, "占位 a")
    tracer = Tracer()
    state = run("t", reg, planner=Planner(model=PlanModel()), max_steps=4, tracer=tracer)
    assert state.plan == ["先 a", "再收工"], "计划修订未写入 state.plan"
    assert [h["kind"] for h in state.history] == ["replan", "finish"], "replan/finish 标注异常"
    assert state.artifacts["a"] == {"v": 1}
    summary = tracer.summary()
    assert summary["by_kind"] == {"replan": 1, "finish": 1}, f"tracer 汇总异常：{summary}"
    assert summary["total_elapsed_s"] >= 0.0 and summary["tools_called"] == ["a"]
    # 追踪事件只含白名单字段（隐私口径）
    assert all(set(e) == {"step", "kind", "tool", "elapsed_s", "usage", "note"} for e in tracer.events)


def test_planner_error_envelope():
    class BoomModel:
        def chat(self, messages, **kwargs):
            raise RuntimeError("api down")

    reg = ToolRegistry()
    reg.register("t1", lambda: {"ok": 1}, "占位")
    decision = Planner(model=BoomModel()).decide(AgentState(goal="g"), reg)
    assert decision["tool"] is None and "模型调用失败" in decision["error"], decision

    # 决策失败写回观察后仍能自愈（第 1 轮垃圾输出 → 第 2 轮工具 → 第 3 轮收工）
    class FlakyModel:
        def __init__(self):
            self.i = 0

        def chat(self, messages, **kwargs):
            self.i += 1
            if self.i == 1:
                return "这不是 JSON"
            if self.i == 2:
                return json.dumps({"tool": "t1", "args": {}, "reason": "r"})
            return json.dumps({"tool": FINISH_TOOL, "args": {}, "reason": "done"})

    state = run("g", reg, planner=Planner(model=FlakyModel()), max_steps=4)
    kinds = [h["kind"] for h in state.history]
    assert kinds == ["decision_error", "tool_call", "finish"], f"决策错误自愈轨迹异常：{kinds}"
    assert state.artifacts["t1"] == {"ok": 1}


def test_mock_args_and_schema_path():
    model = MockModel(args_by_tool={"t1": {"uid": "u9"}})
    reg = ToolRegistry()
    reg.register(
        "t1", lambda uid: {"uid": uid}, "占位",
        schema={"type": "object", "properties": {"uid": {"type": "string"}}, "required": ["uid"]},
    )
    state = run("g", reg, model=model, max_steps=3)
    assert state.artifacts["t1"] == {"uid": "u9"}, "mock 参数底稿未生效"


# ── 11. Verifier 业务断言 ───────────────────────────────────

def test_business_assertions():
    v = CriticVerifier()
    ok, note = v.check({"tool": "assess_risk"}, {"cohort": "c", "bands": {"a": 1, "b": 1}, "n_total": 3})
    assert not ok and "不闭合" in note, note
    ok, _ = v.check({"tool": "detect_anomaly"}, {"metric": "m", "delta_pct": 1.0, "silent_share": 1.5})
    assert not ok, "越界占比应被拒"
    ok, _ = v.check({"tool": "locate_cohort"}, {"cohort": "c", "rule": "r", "size": 0, "share_of_online": 0.1})
    assert not ok, "空人群应被拒"

    rv = RealChainVerifier()
    ok, note = rv.check({"tool": "detect_anomaly"}, {
        "metric": "m", "inactive_share": 0.5, "stage_mix": {"active": 2}, "n_total": 3,
        "exceeds_reference": True,
    })
    assert not ok and "不闭合" in note, note
    ok, note = rv.check({"tool": "assess_risk"}, {"cohort": "c", "bands": {"<60": 2}, "n_total": 2, "median_risk": 10.0})
    assert ok, note


# ── 13. 业务工具层（harness/tools）──────────────────────────

AS_OF_TEST = 1791108404
DAY_S = 86400
STAGES_TEST = ["dormant_60", "dormant_90"]


class _StubPredictor:
    """测试桩预测器：与 SilencePredictor 同接口（不依赖训练工件）。"""

    def __init__(self, table: dict, as_of: int) -> None:
        self.meta = {"artifact_version": "stub_v1", "learner": "stub"}
        self._table, self._as_of = table, as_of

    def default_as_of_ts(self) -> int:
        return self._as_of

    def score_user(self, records, uid_hash, as_of_ts=None) -> dict:
        base = self._table.get(uid_hash) or {
            "applicable": False, "reason": "no_exact_events", "n_pre": 0,
            "gap_days": None, "score": None,
        }
        return {"uid_hash": uid_hash, "as_of_ts": as_of_ts or self._as_of, **base}


def _stub_sandbox(policy: str) -> dict:
    """测试桩沙箱：形状与 audit_summary 一致（不跑合成人口）。"""
    return {
        "policy": policy, "mean_reward_audit": 0.31, "oracle_mean_audit": 0.52,
        "audit_regret": 0.21, "audit_vs_oracle_frac": 0.6,
        "audit_share_control": 0.3, "audit_share_rec": 0.4, "audit_share_recall": 0.3,
        "note": "stub 沙箱（测试口径）",
    }


def _tool_fact(stage, risk, *, band="dormant", score=20.0, percentile=30.0, migration=None) -> dict:
    return {
        "facts_version": "l3v1",
        "as_of": AS_OF_TEST,
        "quality": {"usable": True, "truncated_surfaces": ""},
        "activity": {"band": band, "score": score, "percentile": percentile},
        "churn": {"stage": stage, "risk_score": risk, "horizon_days": "30",
                  "drivers": ["staleness", "momentum"]},
        "migration": migration if migration is not None else {
            "insufficient_data": False, "genre_shift_score": 0.42, "genre_from": "二次元",
            "genre_to": "MOBA", "game_flow_net": -1, "dropped_games": ["g1"],
        },
        "context": {"sentiment_neg_rate": 0.2, "top_genres": [["二次元", 0.6]]},
    }


def _tool_fixture():
    """tools 层 fixture：5 用户 facts（含冲突 / 全空信号）+ 3 条时间线事件 + 桩预测器。"""
    facts = {
        "u_ok": _tool_fact("dormant_60", 85.0),
        "u_ok2": _tool_fact("dormant_90", 70.0),
        "u_low": _tool_fact("dormant_60", 30.0),
        "u_conflict": _tool_fact(
            "dormant_90", 95.0, band="top", score=88.0, percentile=95.0,
            migration={"insufficient_data": True, "genre_shift_score": None, "genre_from": None,
                       "genre_to": None, "game_flow_net": 0, "dropped_games": []},
        ),
        "u_nulls": _tool_fact(
            "active", None, band=None, score=None, percentile=None,
            migration={"insufficient_data": True, "genre_shift_score": None, "genre_from": None,
                       "genre_to": None, "game_flow_net": None, "dropped_games": None},
        ),
    }
    store = FactsStore(facts)
    records = [
        {"uid_hash": "u_ok", "event_id": "e1", "event_type": "review",
         "event_ts": AS_OF_TEST - 20 * DAY_S, "time_kind": "exact"},
        {"uid_hash": "u_ok", "event_id": "e2", "event_type": "post",
         "event_ts": AS_OF_TEST - 80 * DAY_S, "time_kind": "exact"},
        {"uid_hash": "u_conflict", "event_id": "e3", "event_type": "favorite_app",
         "event_ts": AS_OF_TEST - 15 * DAY_S, "time_kind": "exact"},
    ]
    pred = _StubPredictor({
        "u_ok": {"applicable": True, "reason": None, "n_pre": 2, "gap_days": 20.0, "score": 0.7},
        "u_conflict": {"applicable": True, "reason": None, "n_pre": 3, "gap_days": 15.0, "score": 0.82},
        "u_nulls": {"applicable": False, "reason": "no_recent_activity", "n_pre": 0,
                    "gap_days": None, "score": None},
    }, AS_OF_TEST)
    return store, records, pred


def test_insight_tools_direct():
    store, records, pred = _tool_fixture()
    tk = InsightToolkit(store, records, pred, sandbox=_stub_sandbox)
    reg = build_insight_registry(tk)
    assert reg.names() == list(TOOL_NAMES), f"工具集与规范名不一致：{reg.names()}"

    beh = reg.call("get_user_behavior", uid_hash="u_ok")
    assert beh["found"] and beh["n_exact_events"] == 2 and beh["gap_days"] == 20.0
    assert {d["event_type"] for d in beh["top_event_types"]} == {"review", "post"}
    act = reg.call("analyze_activity", uid_hash="u_nulls")
    assert act["band"] is None and act["score"] is None, "缺失信号应为 null（不静默填 0）"
    mig = reg.call("analyze_interest_migration", uid_hash="u_ok")
    assert mig["genre_shift_score"] == 0.42 and mig["insufficient_data"] is False
    pr = reg.call("predict_silence_risk", uid_hash="u_ok")
    assert pr["applicable"] and pr["score"] == 0.7 and pr["model"]["artifact_version"] == "stub_v1"

    cohort = reg.call("get_risk_cohort", criteria={"stages": STAGES_TEST, "min_score": 60})
    assert cohort["size"] == 3 and cohort["stage_breakdown"] == {"dormant_60": 1, "dormant_90": 2}, cohort
    plan = reg.call("plan_intervention", count=4)
    assert plan["batch_size"] == 3 and plan["arms"] == {"control": 0, "rec": 1, "recall": 2}, plan["arms"]
    assert plan["stream_remaining"] == 0 and plan["cohort_id"] == "cohort-1"
    assert len(tk.advisor.intervention_rows) == 3
    assert tk.plan_available() is False, "名额跑完后干预工具应退出候选"

    ev = reg.call("evaluate_strategy", policy="linucb")
    assert ev["mean_reward_audit"] == 0.31
    rep = reg.call("generate_insight_report", scope="cohort")
    assert rep["headline"]["n_users"] == 3
    assert rep["risk_bands"] == {">=80": 2, "60-80": 1, "<60": 0, "unknown": 0}, rep["risk_bands"]
    assert rep["recommended_actions"] and rep["evidence"], "报告必须给建议与证据"

    expect_error(reg.call, "get_user_behavior")  # 缺必填参数 → invalid_args
    try:
        reg.call("plan_intervention", cohort_id="cohort-9")
    except ToolCallError as exc:
        assert exc.kind == "execution", f"未知 cohort_id 应报 execution：{exc.kind}"
    else:
        raise AssertionError("未知 cohort_id 未被拦下")


def test_insight_registry_permission_and_candidates():
    store, records, pred = _tool_fixture()
    tk = InsightToolkit(store, records, pred, sandbox=_stub_sandbox)
    reg = build_insight_registry(tk, granted_permissions={"facts:read", "intervention:plan"})
    state = AgentState(goal="g")
    available = reg.available(state)
    assert "predict_silence_risk" not in available, "未授权工具不应进候选"
    assert "evaluate_strategy" not in available
    assert "plan_intervention" not in available, "未圈人前干预工具不应进候选"
    try:
        reg.call("predict_silence_risk", uid_hash="u_ok")
    except ToolCallError as exc:
        assert exc.kind == "permission", f"越权应报 permission：{exc.kind}"
    else:
        raise AssertionError("越权调用未被拦下")
    reg.call("get_risk_cohort", criteria={"stages": STAGES_TEST, "min_score": 60})
    assert "plan_intervention" in reg.available(state), "圈人后干预工具应进入候选"

    # 预测工件缺失 → 不注册预测工具（不假装能预测）
    tk2 = InsightToolkit(store, records, None, sandbox=_stub_sandbox)
    reg2 = build_insight_registry(tk2)
    assert "predict_silence_risk" not in reg2.names() and len(reg2.names()) == 7


def test_insight_verifier_assertions():
    v = InsightVerifier()
    ok, note = v.check({"tool": FINISH_TOOL}, None)
    assert not ok and "拒绝收工" in note, note

    bad = {"uid_hash": "u", "applicable": True, "reason": None, "score": 1.7, "n_pre": 2}
    ok, note = v.check({"tool": "predict_silence_risk"}, bad)
    assert not ok and "越界" in note, f"非法概率应被拦下：{note}"
    ok, _ = v.check({"tool": "predict_silence_risk"}, {**bad, "score": 0.7})
    assert ok and "predict_silence_risk" in v.passed_tools
    ok, _ = v.check({"tool": FINISH_TOOL}, None)
    assert ok, "有实质性产物后应收工通过"

    ok, note = v.check({"tool": "plan_intervention"}, {
        "cohort_id": "c", "batch_size": 2, "arms": {"control": 1},
        "policy": "p", "rule_version": "r", "stream_remaining": 1,
    })
    assert not ok and "≠" in note, note
    ok, _ = v.check({"tool": "generate_insight_report"}, {
        "scope": "cohort", "cohort_id": "c", "headline": {"n": 1},
        "recommended_actions": [{"action": "a", "rationale": "r"}],
        "evidence": [], "caveats": [],
    })
    assert not ok, "无证据的报告应被拒"
    ok, _ = v.check({"tool": "evaluate_strategy"}, {
        "policy": "p", "mean_reward_audit": 0.3, "audit_regret": 0.1,
        "audit_vs_oracle_frac": 0.6, "audit_share_control": 0.5,
        "audit_share_rec": 0.5, "audit_share_recall": 0.5, "note": "n",
    })
    assert not ok, "审计臂占比不闭合应被拒（0.5×3）"
    ok, _ = v.check({"tool": "analyze_interest_migration"}, {
        "uid_hash": "u", "insufficient_data": True, "genre_shift_score": 0.4,
        "genre_from": None, "genre_to": None,
    })
    assert not ok, "数据不足却给出迁移强度应被拒（口径矛盾）"


def test_insight_agent_mock_run():
    store, records, pred = _tool_fixture()
    common = dict(store=store, records=records, predictor=pred, sandbox=_stub_sandbox)
    a = run_insight_agent(**common)
    b = run_insight_agent(**common)
    assert insight_run_is_deterministic(a, b), "洞察主链 mock 双跑不一致"
    assert a["finished"], "mock 应能在预算内收工"
    state = a["state"]
    seq = [(h.get("decision") or {}).get("tool") for h in state.history]
    assert seq == [*TOOL_NAMES, FINISH_TOOL], f"工具链顺序异常：{seq}"
    assert all(h.get("verified") for h in state.history), "有步骤未通过校验"
    assert set(state.artifacts) == set(TOOL_NAMES), f"产物不全：{sorted(state.artifacts)}"
    assert len(a["toolkit"].advisor.intervention_rows) == 3
    assert mock_args(store, a["config"])["get_user_behavior"]["uid_hash"] == "u_ok"


def test_insight_fault_hook_recovery():
    store, records, pred = _tool_fixture()
    calls = {"n": 0}

    def timeout_once(fn):
        def wrapper(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("上游超时（注入口）")
            return fn(*args, **kwargs)

        return wrapper

    result = run_insight_agent(
        store=store, records=records, predictor=pred, sandbox=_stub_sandbox,
        hooks={"predict_silence_risk": timeout_once},
    )
    state = result["state"]
    kinds = [h["kind"] for h in state.history]
    assert "retry" in kinds, f"超时应记 retry：{kinds}"
    assert "predict_silence_risk" not in state.artifacts, "失败轮不应落产物"
    assert result["finished"], "单工具超时不应拖垮任务（改用其余工具收工）"


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
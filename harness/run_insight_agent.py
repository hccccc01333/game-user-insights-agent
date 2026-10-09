# -*- coding: utf-8 -*-
"""run_insight_agent.py —— ② 洞察主链：业务目标 → LLM/Mock 决策循环（insightv0）。

先说人话：
    这是"Agent 真的在决策"的那条链：给一个业务目标（如"哪些用户可能停止活跃？
    给出干预建议"），把 8 个业务工具（harness/tools）摆上桌，由模型逐轮决定
    下一步调用什么工具、给什么参数；每轮观察写回上下文，模型据此继续 / 重试 /
    修订计划 / 收工。与另外两条链的分工：
      · run_agent.py       S1–S4 沙箱：回答"如果干预会发生什么"（策略评测）；
      · run_real_agent.py  realv0：固定六环节工作流，产出干预队列与实验分组；
      · 本文件             insightv0：**模型驱动的动态决策**（工具调用路径不写死）。
    两条模型路径：
      · --model mock  离线桩（确定性，可双跑对拍）——CI 与回归用；
      · --model llm   DeepSeek/OpenAI 兼容（环境变量 LLM_API_KEY），端到端能力评测用。

运行（在仓库根目录）：
    python -m harness.run_insight_agent                    # mock 双跑（默认）
    python -m harness.run_insight_agent --model llm        # 真实模型（需 LLM_API_KEY）

产出（默认 data/processed/harness_insight/；仅 uid_hash 与派生字段）：
    insight_trace.jsonl      决策循环轨迹（每步：决策 → 观察 → 校验）
    insight_artifacts.json   通过校验的产物（键=工具名）
    intervention_queue.jsonl 干预队列（若模型走到了干预环节）
    insight_report.json      洞察报告（若模型产出了报告产物）
    insight_metrics.json     执行指标（步骤 / 工具序列 / 耗时 / 模型用量）
    _manifest.json           版本 / 输入指纹 / 配置 / 输出指纹 / 双跑一致性
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import HARNESS_VERSION
from .loop import run
from .model import LLMAdapter, MockModel, ModelConfigError
from .planner import FINISH_TOOL, Planner
from .tools import (
    TOOLSET_VERSION,
    InsightToolkit,
    InsightVerifier,
    SandboxConfig,
    build_insight_registry,
    load_tool_inputs,
)
from .tracing import Tracer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "harness_insight"
TZ_CN = timezone(timedelta(hours=8))

INSIGHT_CHAIN_VERSION = "insightv0"
DEFAULT_GOAL = "分析哪些 TapTap 社区用户可能停止活跃，并给出合理的干预建议。"
DEFAULT_COHORT_STAGES = ("dormant_60", "dormant_90")


@dataclass
class InsightAgentConfig:
    """洞察主链配置（CLI 与测试共用；构造时即校验）。"""

    goal: str = DEFAULT_GOAL
    max_steps: int = 12
    max_tool_calls: int = 14
    max_failures_per_tool: int = 2
    cohort_stages: tuple[str, ...] = DEFAULT_COHORT_STAGES
    plan_count: int = 20
    mock_criteria_min_score: float = 60.0
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    def __post_init__(self) -> None:
        self.cohort_stages = tuple(self.cohort_stages)
        if not self.goal.strip():
            raise ValueError("goal 不能为空")
        if self.max_steps < 3:
            raise ValueError(f"max_steps 需 ≥ 3（当前 {self.max_steps}）")
        if self.max_tool_calls < 1:
            raise ValueError(f"max_tool_calls 需 ≥ 1（当前 {self.max_tool_calls}）")
        if self.max_failures_per_tool < 1:
            raise ValueError(f"max_failures_per_tool 需 ≥ 1（当前 {self.max_failures_per_tool}）")
        if self.plan_count < 1:
            raise ValueError(f"plan_count 需 ≥ 1（当前 {self.plan_count}）")
        if not self.cohort_stages:
            raise ValueError("cohort_stages 不能为空")


def mock_args(store: Any, config: InsightAgentConfig) -> dict[str, dict]:
    """MockModel 的参数底稿：让离线回归能整链跑满 8 个工具。"""
    uids = store.usable() or store.uids()
    uid = uids[0] if uids else ""
    return {
        "get_user_behavior": {"uid_hash": uid},
        "analyze_activity": {"uid_hash": uid},
        "analyze_interest_migration": {"uid_hash": uid},
        "predict_silence_risk": {"uid_hash": uid},
        "get_risk_cohort": {"criteria": {
            "stages": list(config.cohort_stages),
            "min_score": config.mock_criteria_min_score,
        }},
        "plan_intervention": {"count": config.plan_count},
        "evaluate_strategy": {"policy": "linucb"},
        "generate_insight_report": {"scope": "cohort"},
    }


def run_insight_agent(
    *,
    store: Any,
    records: list[dict] | None = None,
    predictor: Any = None,
    sandbox: Any = None,
    model: Any = None,
    config: InsightAgentConfig | None = None,
    tracer: Tracer | None = None,
    hooks: dict | None = None,
    granted_permissions: set[str] | tuple[str, ...] | None = None,
) -> dict:
    """跑一遍洞察主链；model 缺省 = MockModel（带参数底稿，离线可跑）。"""
    config = config or InsightAgentConfig()
    toolkit = InsightToolkit(
        store, records, predictor,
        sandbox=sandbox, sandbox_config=config.sandbox,
        cohort_stages=config.cohort_stages, default_plan_count=config.plan_count,
    )
    registry = build_insight_registry(
        toolkit, granted_permissions=granted_permissions, hooks=hooks,
    )
    if model is None:
        model = MockModel(args_by_tool=mock_args(store, config))
    state = run(
        config.goal, registry,
        planner=Planner(model=model, recent_steps=config.max_steps),
        verifier=InsightVerifier(),
        model=model,
        max_steps=config.max_steps,
        context=store,
        max_tool_calls=config.max_tool_calls,
        max_failures_per_tool=config.max_failures_per_tool,
        tracer=tracer,
    )
    finished = any(
        (h.get("decision") or {}).get("tool") == FINISH_TOOL and h.get("verified")
        for h in state.history
    )
    return {"state": state, "toolkit": toolkit, "registry": registry,
            "config": config, "model": model, "finished": finished}


def run_is_deterministic(first: dict, second: dict) -> bool:
    """双跑一致性：轨迹 / 产物 / 干预队列都必须逐值相等（mock 路径专用）。"""
    dump = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)  # noqa: E731
    return (
        dump(first["state"].history) == dump(second["state"].history)
        and dump(first["state"].artifacts) == dump(second["state"].artifacts)
        and dump(first["toolkit"].advisor.intervention_rows)
        == dump(second["toolkit"].advisor.intervention_rows)
    )


# ── 落盘与报告 ──────────────────────────────────────────────

def _now() -> str:
    return datetime.now(TZ_CN).isoformat(timespec="seconds")


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _tool_sequence(state: Any) -> list[str]:
    return [(h.get("decision") or {}).get("tool") or "（无动作）" for h in state.history]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="洞察主链：业务目标 → 工具调用决策循环（insightv0）")
    ap.add_argument("--goal", default=DEFAULT_GOAL, help="业务目标（自然语言）")
    ap.add_argument("--model", choices=("mock", "llm"), default="mock",
                    help="决策模型：mock 离线桩（双跑对拍）/ llm 真实模型（需 LLM_API_KEY）")
    ap.add_argument("--facts", default=None, help="L3 facts.jsonl 路径（默认按约定位置）")
    ap.add_argument("--timeline", default=None, help="L2 timeline.jsonl 路径（可选）")
    ap.add_argument("--model-path", default=None, help="沉默预测工件路径（可选；缺则不注册预测工具）")
    ap.add_argument("--index", default=None, help="采集索引路径（取 fetched_at 做观测终点，可选）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 data/processed/harness_insight）")
    ap.add_argument("--max-steps", type=int, default=12, help="循环步数上限（默认 12）")
    ap.add_argument("--max-tool-calls", type=int, default=14, help="工具调用总预算（默认 14）")
    ap.add_argument("--plan-count", type=int, default=20, help="单批干预名额数（默认 20）")
    ap.add_argument("--no-verify", action="store_true", help="跳过 mock 双跑一致性校验")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = InsightAgentConfig(
            goal=args.goal, max_steps=args.max_steps,
            max_tool_calls=args.max_tool_calls, plan_count=args.plan_count,
        )
    except ValueError as exc:
        print(f"[stop] 参数无效：{exc}", file=sys.stderr)
        return 2

    kwargs = {}
    if args.facts:
        kwargs["facts_path"] = args.facts
    if args.timeline:
        kwargs["timeline_path"] = args.timeline
    if args.model_path:
        kwargs["model_path"] = args.model_path
    if args.index:
        kwargs["index_path"] = args.index
    try:
        inputs = load_tool_inputs(**kwargs)
    except FileNotFoundError as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 2

    model: Any
    if args.model == "llm":
        try:
            model = LLMAdapter(json_mode=True, temperature=0.0)
        except ModelConfigError as exc:
            print(f"[stop] {exc}", file=sys.stderr)
            return 2
    else:
        model = None  # run_insight_agent 内部按 store 造 MockModel 底稿

    tracer = Tracer()
    common = dict(
        store=inputs["store"], records=inputs["records"], predictor=inputs["predictor"],
        model=model, config=config, tracer=tracer,
    )
    result = run_insight_agent(**common)
    determinism = {"checked": False, "mode": args.model}
    if args.model == "mock" and not args.no_verify:
        again = run_insight_agent(**common)
        same = run_is_deterministic(result, again)
        determinism = {"checked": True, "mode": "mock", "identical": bool(same)}
        if not same:
            print("[stop] 双跑不一致：存在非确定性来源，拒绝落盘", file=sys.stderr)
            return 3

    # ── 落盘 ──
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = result["state"]
    toolkit = result["toolkit"]

    trace_path = out_dir / "insight_trace.jsonl"
    trace_lines = [json.dumps(h, ensure_ascii=False, default=str) for h in state.history]
    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

    artifacts_path = out_dir / "insight_artifacts.json"
    artifacts_path.write_text(
        json.dumps(state.artifacts, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )

    queue_path = out_dir / "intervention_queue.jsonl"
    rows = toolkit.advisor.intervention_rows
    queue_lines = [json.dumps(r, ensure_ascii=False, default=str) for r in rows]
    queue_path.write_text("\n".join(queue_lines) + "\n" if queue_lines else "", encoding="utf-8")

    report = state.artifacts.get("generate_insight_report")
    report_path = out_dir / "insight_report.json"
    if report is not None:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )

    model_stats = result["model"].stats() if hasattr(result["model"], "stats") else {}
    usage = tracer.summary()
    metrics = {
        "insight_chain_version": INSIGHT_CHAIN_VERSION,
        "goal": config.goal,
        "model": args.model,
        "finished": result["finished"],
        "steps": len(state.history),
        "tool_sequence": _tool_sequence(state),
        "tools_called": usage["tools_called"],
        "artifacts": sorted(state.artifacts),
        "blocks": usage["by_kind"],
        "elapsed_s": usage["total_elapsed_s"],
        "model_stats": model_stats,
        "intervention_rows": len(rows),
        "report_ready": report is not None,
    }
    metrics_path = out_dir / "insight_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    outputs = {
        trace_path.name: {"rows": len(trace_lines), "sha256": _sha16(trace_path)},
        artifacts_path.name: {"rows": None, "sha256": _sha16(artifacts_path)},
        queue_path.name: {"rows": len(queue_lines), "sha256": _sha16(queue_path)},
        metrics_path.name: {"rows": None, "sha256": _sha16(metrics_path)},
    }
    if report_path.exists():
        outputs[report_path.name] = {"rows": None, "sha256": _sha16(report_path)}
    manifest = {
        "harness_version": HARNESS_VERSION,
        "insight_chain_version": INSIGHT_CHAIN_VERSION,
        "toolset_version": TOOLSET_VERSION,
        "generated_at": _now(),
        "mode": f"insight_{args.model}",
        "config": {**asdict(config), "sandbox": config.sandbox.snapshot()},
        "inputs": {
            "facts": {"path": _rel(inputs["facts_path"]), "sha16": _sha16(inputs["facts_path"]),
                      "rows": len(inputs["store"])},
            "timeline": {"path": _rel(inputs["timeline_path"]),
                         "rows": len(inputs["records"]),
                         "sha16": _sha16(inputs["timeline_path"]) if inputs["timeline_path"].exists() else None},
            "silence_model": {"path": _rel(inputs["model_path"]),
                              "loaded": inputs["predictor"] is not None},
        },
        "outputs": outputs,
        "determinism": determinism,
        "privacy": ("仅 uid_hash 与派生字段（stage / risk_score / arm / 分数）；"
                    "无昵称、评价原文、ip/device 等敏感信息"),
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )

    # ── 终端报告 ──
    print(f"[info] 目标：{config.goal}")
    print(f"[info] 输入：facts {len(inputs['store'])} 行（{_rel(inputs['facts_path'])}）｜"
          f"时间线 {len(inputs['records'])} 事件｜沉默工件 "
          f"{'已加载' if inputs['predictor'] is not None else '未加载（预测工具不注册）'}")
    print(f"[info] 模型：{args.model}｜步骤 {metrics['steps']}｜"
          f"工具序列：{' → '.join(metrics['tool_sequence'])}")
    if result["finished"]:
        print(f"[ok] 任务完成（收工经 Verifier 确认）；产物：{', '.join(metrics['artifacts']) or '无'}")
    else:
        print(f"[warn] 未在 {config.max_steps} 步内确认收工；产物：{', '.join(metrics['artifacts']) or '无'}")
    cohort = state.artifacts.get("get_risk_cohort")
    if cohort:
        print(f"       人群：{cohort['cohort_id']}｜{cohort['size']} 人"
              f"（占可用 {cohort['share_of_usable']:.1%}）")
    plan = state.artifacts.get("plan_intervention")
    if plan:
        print(f"       干预队列：{plan['batch_size']} 人｜臂计数 {plan['arms']}")
    if report:
        h = report["headline"]
        print(f"       报告：n={h['n_users']}｜风险分档 {report['risk_bands']}｜"
              f"建议 {len(report['recommended_actions'])} 条")
    print(f"[done] → {_rel(out_dir)}（trace {len(trace_lines)} 行 / queue {len(rows)} 行）")
    if determinism.get("checked"):
        print(f"[ok] 双跑一致性：{determinism['identical']}")
    elif args.model == "llm":
        print("[note] llm 路径不做双跑校验（采样不确定性；评测口径见 agent_eval）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
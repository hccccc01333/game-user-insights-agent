#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""regression.py —— Agent 评测 CLI（v1）：6 场景 × 双模式（mock / llm）。

先说人话：
    一条命令回答"自研 Harness 让 Agent 在可靠性 / 可控性 / 可观测性上拿到了
    什么"。两种模式的分工：
      · --model mock  MockModel 离线桩：逐值确定的回归——同参双跑必须一致
                      （CI 门），场景硬期望必须全过；这是"可控性"的证据。
      · --model llm   真实模型（LLM_API_KEY）：端到端能力评测——按通用成功
                      率与指标汇总（--repeat N 观察多次执行的稳定性），
                      不作 CI 门（采样路径不确定）。

运行（在仓库根目录）：
    python -m agent_eval.regression                          # mock 全场景 + 双跑
    python -m agent_eval.regression --scenarios normal,tool_timeout
    python -m agent_eval.regression --model llm --repeat 3   # 需 LLM_API_KEY

产出（默认 data/processed/agent_eval/）：
    agent_eval_report.json  每场景指标 + 汇总（成功率 / 均值 / 恢复率 / 拦截）
    agent_eval_trace.jsonl  逐场景逐步轨迹（kind / tool / 校验结果，隐私口径）
    _manifest.json          版本 / 模式 / 场景 / 双跑一致性 / 输出指纹
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from harness.model import LLMAdapter, ModelConfigError
from harness.run_insight_agent import run_insight_agent, run_is_deterministic
from harness.tracing import Tracer

from . import AGENT_EVAL_VERSION
from .evaluator import aggregate, evaluate_scenario
from .scenarios import SCENARIO_NAMES, get_scenarios

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "agent_eval"
TZ_CN = timezone(timedelta(hours=8))


def _now() -> str:
    return datetime.now(TZ_CN).isoformat(timespec="seconds")


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _run_once(scenario, model_factory):
    """跑一次场景（每次全新 build，避免运行间串状态）；返回 result / tracer / kwargs。"""
    kwargs = scenario.build()
    tracer = Tracer()
    if model_factory is not None:
        kwargs["model"] = model_factory()
    result = run_insight_agent(tracer=tracer, **kwargs)
    return result, tracer, kwargs


def _trace_rows(scenario_name: str, result: dict) -> list[dict]:
    rows = []
    for h in result["state"].history:
        decision = h.get("decision") or {}
        obs = h.get("observation")
        rows.append({
            "scenario": scenario_name,
            "step": h.get("step"),
            "kind": h.get("kind"),
            "tool": decision.get("tool"),
            "verified": bool(h.get("verified")),
            "error": obs.get("error") if isinstance(obs, dict) else None,
        })
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Agent 评测（v1）：场景集 × 模式回归")
    ap.add_argument("--model", choices=("mock", "llm"), default="mock",
                    help="决策模型：mock 离线桩（双跑对拍）/ llm 真实模型（需 LLM_API_KEY）")
    ap.add_argument("--scenarios", default=None,
                    help=f"场景名逗号分隔（默认全部；支持 {', '.join(SCENARIO_NAMES)}）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="llm 模式每场景重复次数（默认 1；mock 恒为双跑对拍）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 data/processed/agent_eval）")
    ap.add_argument("--no-verify", action="store_true", help="mock 模式跳过双跑一致性校验")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        scenarios = get_scenarios(args.scenarios.split(",") if args.scenarios else None)
    except ValueError as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 2
    if args.model == "mock" and args.repeat != 1:
        print("[stop] mock 模式固定双跑对拍，不支持 --repeat", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("[stop] --repeat 需 ≥ 1", file=sys.stderr)
        return 2

    model_factory = None
    if args.model == "llm":
        try:
            LLMAdapter(json_mode=True, temperature=0.0)  # 探针：缺密钥立即显式报错
        except ModelConfigError as exc:
            print(f"[stop] {exc}", file=sys.stderr)
            return 2
        model_factory = lambda: LLMAdapter(json_mode=True, temperature=0.0)  # noqa: E731

    records: list[dict] = []
    trace_rows: list[dict] = []
    stability = {"mode": args.model, "checked": args.model == "mock" and not args.no_verify,
                 "identical": True if args.model == "mock" else None}

    for scenario in scenarios:
        result, tracer, kwargs = _run_once(scenario, model_factory)
        record = evaluate_scenario(scenario, result, tracer, kwargs)
        trace_rows.extend(_trace_rows(scenario.name, result))
        runs, success_runs = 1, int(record["task_success"])

        if args.model == "mock" and not args.no_verify:
            again, again_tracer, _ = _run_once(scenario, model_factory)
            same = run_is_deterministic(result, again)
            record["stable"] = bool(same)
            stability["identical"] = stability["identical"] and same
            again_record = evaluate_scenario(scenario, again, again_tracer, kwargs)
            runs += 1
            success_runs += int(again_record["task_success"])
        elif args.model == "llm":
            for _ in range(args.repeat - 1):
                later, later_tracer, _ = _run_once(scenario, model_factory)
                later_record = evaluate_scenario(scenario, later, later_tracer, kwargs)
                trace_rows.extend(_trace_rows(scenario.name, later))
                runs += 1
                success_runs += int(later_record["task_success"])
        record["runs"], record["success_runs"] = runs, success_runs
        records.append(record)

        flag = "[ok]  " if record["expectations_ok"] else "[FAIL]"
        extra = f" 稳定={record['stable']}" if "stable" in record else ""
        print(f"{flag} {scenario.name:<20} 步骤={record['steps']:<3} "
              f"执行={record['tool_calls']:<3} 产物={len(record['artifacts']):<3} "
              f"拦截={record['blocks']['verify'] + record['blocks']['decision_error']} "
              f"耗时={record['elapsed_s']:.3f}s{extra}")
        for item in record["failures"]:
            print(f"        - {item}")

    summary = aggregate(records)
    summary["execution_stability"] = {
        "runs": sum(r["runs"] for r in records),
        "success_runs": sum(r["success_runs"] for r in records),
        "rate": round(sum(r["success_runs"] for r in records)
                      / sum(r["runs"] for r in records), 4),
    }

    # ── 落盘 ──
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "agent_eval_version": AGENT_EVAL_VERSION,
        "generated_at": _now(),
        "mode": args.model,
        "scenarios": [s.name for s in scenarios],
        "stability": stability,
        "summary": summary,
        "records": records,
    }
    report_path = out_dir / "agent_eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    trace_path = out_dir / "agent_eval_trace.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in trace_rows) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "agent_eval_version": AGENT_EVAL_VERSION,
        "generated_at": report["generated_at"],
        "mode": args.model,
        "scenarios": [{"name": s.name, "title": s.title,
                       "injected_failures": s.injected_failures} for s in scenarios],
        "fixture": "in-code（agent_eval/scenarios：3 用户 + 3 事件 + 桩预测器/沙箱）",
        "stability": stability,
        "outputs": {
            report_path.name: {"sha256": _sha16(report_path)},
            trace_path.name: {"rows": len(trace_rows), "sha256": _sha16(trace_path)},
        },
        "privacy": "仅 uid_hash 与派生字段；无昵称、原文、ip/device 等敏感信息",
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ── 终端汇总 ──
    print(f"[info] 模式 {args.model}｜场景 {summary['n_scenarios']}｜"
          f"任务成功率 {summary['task_success_rate']}｜"
          f"硬期望通过率 {summary['expectations_rate']}")
    print(f"[info] 步骤均值 {summary['mean_steps']}｜执行均值 {summary['mean_tool_calls']}｜"
          f"总耗时 {summary['total_elapsed_s']}s｜tokens {summary['total_tokens']}｜"
          f"成本 ${summary['est_cost_usd']}")
    rec = summary["error_recovery"]
    print(f"[info] 错误恢复：注入 {rec['scenarios_with_injection']} 场景，"
          f"恢复 {rec['recovered']}（{rec['rate']}）｜事实可靠性 "
          f"{summary['fact_reliability_rate']}")
    safety = summary["safety"]
    print(f"[info] 安全约束：校验拦截 {safety['verify_blocks']}｜决策拦截 "
          f"{safety['decision_errors']}｜越权执行 {safety['unauthorized_executions']}｜"
          f"坏参数 {safety['invalid_args']}")
    if stability["checked"]:
        print(f"[{'ok' if stability['identical'] else 'FAIL'}] 稳定性（mock 双跑逐值一致）："
              f"{stability['identical']}")
    elif args.model == "llm":
        es = summary["execution_stability"]
        print(f"[info] 稳定性（llm {es['runs']} 次执行）：成功 {es['success_runs']}/{es['runs']}"
              f"（{es['rate']}）")
    print(f"[done] → {_rel(out_dir)}")

    if args.model == "mock" and not args.no_verify:
        all_ok = all(r["expectations_ok"] for r in records)
        return 0 if (all_ok and stability["identical"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
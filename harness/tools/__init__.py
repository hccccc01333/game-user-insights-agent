# -*- coding: utf-8 -*-
"""harness.tools —— 业务工具层：把 L1–L3 / 沉默预测 / 沙箱封装成 Agent 可调用能力。

组件地图：
    insight.py    InsightToolkit（8 个工具实现）+ CohortAdvisor + SandboxConfig
                  + build_insight_registry（Schema / 权限 / 前置条件 / 故障注入）
    verifier.py   InsightVerifier（结构契约 + 业务断言 + 实质性产物收工门）

快速用法（在仓库根目录）：
    from harness.tools import InsightToolkit, build_insight_registry, InsightVerifier
    inputs = load_tool_inputs()                      # facts（必须）+ 时间线 + 沉默工件（可选）
    toolkit = InsightToolkit(inputs["store"], inputs["records"], inputs["predictor"])
    registry = build_insight_registry(toolkit)

完整性入口：python -m harness.run_insight_agent（CLI：goal → 决策循环 → 报告落盘）
"""
from .insight import (
    DEFAULT_PLAN_COUNT,
    DEFAULT_RISK_HIGH,
    DEFAULT_RISK_MID,
    PERMISSIONS,
    SUBSTANTIVE_TOOLS,
    TOOL_NAMES,
    TOOLSET_VERSION,
    CohortAdvisor,
    InsightToolkit,
    SandboxConfig,
    TimelineView,
    build_insight_registry,
    derive_as_of,
    load_tool_inputs,
    make_sandbox_runner,
)
from .verifier import REQUIRED_FIELDS, InsightVerifier

__all__ = [
    "TOOLSET_VERSION", "TOOL_NAMES", "PERMISSIONS", "SUBSTANTIVE_TOOLS",
    "DEFAULT_RISK_HIGH", "DEFAULT_RISK_MID", "DEFAULT_PLAN_COUNT",
    "InsightToolkit", "CohortAdvisor", "TimelineView", "SandboxConfig",
    "build_insight_registry", "load_tool_inputs", "make_sandbox_runner", "derive_as_of",
    "InsightVerifier", "REQUIRED_FIELDS",
]
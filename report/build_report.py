#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S5 展示层 · 把全链路落盘产出汇总为一张静态自包含 HTML 报告。

先说人话：
    真实轨（L1–L3）回答"用户现在什么状态"，模拟轨（S1–S4）回答"干预下去会发生
    什么、怎么决定、谁踩刹车"。这支脚本把两条轨的落盘产出（各模块 _manifest /
    metrics / summary / curve / trace）读成一份**聚合统计**，渲染成一张零依赖、
    零 CDN、离线直开的 HTML——所有图表是 Python 生成的 inline SVG，页面不含
    任何脚本、外链与时间戳（逐字节可复现），也不含任何逐用户隐私数据。

运行（在仓库根目录）：
    python -m report.build_report            # 生成 report/index.html + _manifest.json（默认双跑校验）
    python -m report.build_report --no-verify --out report

退出码：参数无效 2；双跑不一致 3（拒绝落盘）；合成轨输入缺失 4（附重跑指引）。
真实轨（L2 / L3）产出缺失时自动跳过对应章节——报告仍可生成。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import REPORT_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "report"
TZ_CN = timezone(timedelta(hours=8))

# 合成轨（S1–S4）：缺任意一份即拒绝生成（退出码 4，附重跑指引）
SYNTH_INPUTS = (
    "data/synthetic/_manifest.json",
    "data/synthetic/uplift/uplift_metrics.json",
    "data/synthetic/uplift/uplift_calibration.csv",
    "data/synthetic/bandit/bandit_summary.csv",
    "data/synthetic/bandit/bandit_curve.csv",
    "data/synthetic/harness/s4_metrics.json",
    "data/synthetic/harness/s4_trace.jsonl",
)
# 真实轨（L1–L3）：缺失即跳过对应章节（聚合口径，仅本地存在）
REAL_INPUTS = (
    "data/processed/user_features/_manifest.json",
    "data/processed/user_insights/_manifest.json",
)
RERUN_HINT = (
    "python -m simulator.run_sim --no-fit --users 500",
    "python -m uplift.run_uplift --no-fit --users 6000",
    "python -m bandit.run_bandit --no-fit --users 6000",
    "python -m harness.run_agent --no-fit --users 6000",
)
CURVE_SAMPLE_EVERY = 105          # 学习曲线采样步长（4200 步 → 40 点）
CURVE_POLICIES = ("linucb", "thompson", "random", "fixed_recall")

# ── 统一配色（与各模块 README 的叙事色一致）─────────────────
INK = "#1f2937"
MUTED = "#6b7280"
ACCENT = "#2563eb"
GOOD = "#059669"
BAD = "#dc2626"
GRID = "#e5e7eb"
VIOLET = "#7c3aed"
WARN = "#d97706"
DEEP = "#7f1d1d"


class MissingInputsError(RuntimeError):
    """合成轨必需产出缺失（消息内附重跑指引）。"""


# ── 小工具 ──────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(TZ_CN).isoformat(timespec="seconds")


def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _rel(path: Path) -> str:
    """终端只展示相对路径（避免本地绝对路径进入任何可被复制出去的输出）。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _pct(x: float, nd: int = 1) -> str:
    return f"{float(x) * 100:.{nd}f}%"


def _signed(x: float, nd: int = 2) -> str:
    return f"{float(x):+.{nd}f}"


def _num(x: float, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}"


def _thousands(x: float) -> str:
    return f"{float(x):,.0f}"


def _write_text(path: Path, text: str) -> None:
    """定死 LF：跨平台逐字节一致的产出（避免 Windows 文本模式把 \\n 换成 \\r\\n）。"""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _data_rows(path: Path) -> int | None:
    """CSV / JSONL 的数据行数（CSV 不含表头；JSON 返回 None）。"""
    if path.suffix not in (".csv", ".jsonl"):
        return None
    with path.open("r", encoding="utf-8-sig") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return max(len(lines) - 1, 0) if path.suffix == ".csv" else len(lines)


# ── 数据收集（只读落盘产出 → 聚合统计）──────────────────────

def collect(root: Path = PROJECT_ROOT) -> tuple[dict, dict]:
    """读取全部输入产出 → (stats, sources)。合成轨缺失抛 MissingInputsError。"""
    missing = [rel for rel in SYNTH_INPUTS if not (root / rel).exists()]
    if missing:
        listing = "\n".join(f"  - {rel}" for rel in missing)
        hint = "\n".join(f"  {cmd}" for cmd in RERUN_HINT)
        raise MissingInputsError(
            f"缺少合成轨产出 {len(missing)} 份：\n{listing}\n先按序重跑（仓库根目录）：\n{hint}"
        )

    stats: dict = {}
    stats["s1"] = _collect_s1(_load_json(root / SYNTH_INPUTS[0]))
    stats["s2"] = _collect_s2(
        _load_json(root / SYNTH_INPUTS[1]), _load_csv(root / SYNTH_INPUTS[2])
    )
    stats["s3"] = _collect_s3(
        _load_csv(root / SYNTH_INPUTS[3]), _load_csv(root / SYNTH_INPUTS[4])
    )
    stats["s4"] = _collect_s4(
        _load_json(root / SYNTH_INPUTS[5]), _load_jsonl(root / SYNTH_INPUTS[6])
    )
    stats["real"] = _collect_real(root)

    sources: dict = {}
    for rel in SYNTH_INPUTS + tuple(r for r in REAL_INPUTS if (root / r).exists()):
        sources[rel] = {
            "sha256": _sha16((root / rel).read_bytes()),
            "rows": _data_rows(root / rel),
        }
    fp_src = "\n".join(f"{rel}:{src['sha256']}" for rel, src in sources.items())
    stats["inputs_fingerprint"] = hashlib.sha256(fp_src.encode("utf-8")).hexdigest()[:16]
    return stats, sources


def _load_jsonl(path: Path) -> list:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _collect_s1(manifest: dict) -> dict:
    eff = manifest["effects_summary"]
    return {
        "seed": int(manifest["seed"]),
        "n_users": int(manifest["n_users"]),
        "days": int(manifest["days"]),
        "window": int(manifest["reward_window_days"]),
        "population_path": str(manifest["population"]["path"]),
        "fit_rows": int(manifest["population"].get("fit_rows") or 0),
        "act30_gt0_share": float(manifest["population"]["realized"]["act30_gt0_share"]),
        "arms": {
            name: {
                "label": str(eff[name]["label"]),
                "tau": float(eff[name]["mean_tau_struct"]),
                "reward": float(eff[name]["mean_reward"]),
            }
            for name in ("control", "rec", "recall")
        },
    }


def _collect_s2(metrics: dict, calibration_rows: list[dict]) -> dict:
    cfg = metrics["config"]
    fidelity = []
    for proto in ("rct", "full"):
        for row in metrics["results"][proto]["fidelity"]:
            fidelity.append({
                "protocol": str(row["protocol"]),
                "arm": str(row["arm"]),
                "learner": str(row["learner"]),
                "tau_hat": float(row["mean_tau_hat"]),
                "tau_struct": float(row["mean_tau_struct"]),
                "bias": float(row["bias"]),
                "spearman_struct": float(row["spearman_struct"]),
            })
    calibration = [
        {
            "bin": int(r["bin"]),
            "hat": float(r["mean_tau_hat"]),
            "struct": float(r["mean_tau_struct"]),
        }
        for r in calibration_rows
        if r["protocol"] == "rct" and r["arm"] == "recall" and r["learner"] == "hgb"
    ]
    rct = metrics["results"]["rct"]
    return {
        "seed": int(cfg["seed"]),
        "n_users": int(cfg["n_users"]),
        "reference_k_pct": int(cfg["reference_k_pct"]),
        "population_path": str(cfg["population_path"]),
        "n_train": int(rct["n_train_users"]),
        "n_holdout": int(rct["n_holdout_users"]),
        "fidelity": fidelity,
        "policy_reference": {
            proto: metrics["results"][proto]["policy_at_reference_k"] for proto in ("rct", "full")
        },
        "calibration": calibration,
    }


def _collect_s3(summary_rows: list[dict], curve_rows: list[dict]) -> dict:
    policies = [
        {
            "policy": str(r["policy"]),
            "mean_reward": float(r["mean_reward"]),
            "head20": float(r["head20"]),
            "tail20": float(r["tail20"]),
            "cum_regret": float(r["cum_regret"]),
            "audit_mean": float(r["mean_reward_audit"]),
            "audit_frac": float(r["audit_vs_oracle_frac"]),
            "share_recall": float(r["share_recall"]),
            "oracle_mean": float(r["oracle_mean"]),
        }
        for r in summary_rows
    ]
    curve: dict[str, list] = {name: [] for name in CURVE_POLICIES}
    for r in curve_rows:
        step = int(r["step"])
        # 从第 105 步起采样：第 0 步是单次抽样，尖峰会把学习曲线压扁
        if step > 0 and step % CURVE_SAMPLE_EVERY == 0 and r["policy"] in curve:
            curve[r["policy"]].append((step, float(r["avg_reward"])))
    first = summary_rows[0]
    return {
        "n_online": int(first["n_online"]),
        "n_audit": int(first["n_audit"]),
        "policies": policies,
        "curve": {name: sorted(pts) for name, pts in curve.items()},
    }


def _collect_s4(metrics: dict, trace_lines: list[dict]) -> dict:
    modes = {}
    for mode in ("with_critic", "no_critic"):
        m = metrics["modes"][mode]
        batch = m["batches"][0]
        modes[mode] = {
            "arms": {k: int(v) for k, v in batch["arms"].items()},
            "batch_size": int(batch["batch_size"]),
            "batch_mean_reward": float(batch["mean_reward"]),
            "veto_count": int(batch["veto_count"]),
            "veto_rate": float(batch["veto_rate"]),
            "online_vs_oracle": float(m["online"]["reward_vs_oracle_frac"]),
            "audit_vs_oracle": float(m["audit"]["audit_vs_oracle_frac"]),
        }
    trace: dict[str, list] = {}
    cohort = None
    for ln in trace_lines:
        tool = str(ln["decision"]["tool"])
        trace.setdefault(str(ln["mode"]), []).append({
            "step": int(ln["step"]),
            "tool": tool,
            "note": _trace_note(tool, ln["observation"]),
        })
        if ln["mode"] == "with_critic" and tool == "locate_cohort":
            cohort = {
                "size": int(ln["observation"]["size"]),
                "share": float(ln["observation"]["share_of_online"]),
                "rule": str(ln["observation"]["rule"]),
            }
    return {
        "seed": int(metrics["config"]["seed"]),
        "n_users": int(metrics["config"]["n_users"]),
        "cohort": cohort,
        "modes": modes,
        "trace": trace,
    }


def _trace_note(tool: str, obs) -> str:
    """把轨迹观察压成一行报告口径的说明（合成世界可观测字段）。"""
    if tool == "detect_anomaly":
        return (f"对照基线互动环比 {_signed(obs['delta_pct'])}%（后半段 vs 前半段）"
                f"｜人口沉默占比 {_pct(obs['silent_share'])}")
    if tool == "locate_cohort":
        return f"{obs['size']} 人（占在线池 {_pct(obs['share_of_online'])}）；规则：{obs['rule']}"
    if tool == "analyze_cause":
        return (f"沉默中位 {_num(obs['silence_median'], 1)} 天｜资历中位 "
                f"{_thousands(obs['tenure_median'])} 天｜兴趣集中度 {_num(obs['concentration_median'], 3)}")
    if tool == "assess_risk":
        b = obs["bands"]
        return f"90–180 天 {b['90-180']} 人 · 180–365 天 {b['180-365']} 人 · 365+ 天 {b['365+']} 人"
    if tool == "allocate_interventions":
        a = obs["arms"]
        return (f"{obs['batch_size']} 人次 → 对照 {a['control']} / 推荐 rec {a['rec']} / "
                f"召回 recall {a['recall']}；否决 {obs['veto_count']}（{_pct(obs['veto_rate'])}）"
                f"｜批次均值 {_num(obs['mean_reward'], 4)}")
    if tool == "design_experiment":
        return f"{obs['grouping']}｜主指标 {obs['primary_metric']}｜触达 {obs['n_treated']} 人次"
    if tool == "finish":
        return "收工（干预已完成并通过校验）"
    return "—"


def _collect_real(root: Path) -> dict | None:
    if not all((root / rel).exists() for rel in REAL_INPUTS):
        return None
    l2 = _load_json(root / REAL_INPUTS[0])
    l3 = _load_json(root / REAL_INPUTS[1])
    raw_dir = root / "data" / "raw" / "user_profile"
    raw_files = len(list(raw_dir.glob("h_*.json"))) if raw_dir.is_dir() else None
    return {
        "l2": {
            "version": str(l2["feature_version"]),
            "n_users": int(l2["n_users"]),
            "events": int(l2["timeline"]["events"]),
            "truncated": int(l2["truncation_summary"]["users_truncated"]),
            "by_surface": {k: int(v) for k, v in l2["truncation_summary"]["by_surface"].items()},
            "errors": int(l2["users_with_errors"]),
            "fingerprint": str(l2["input_fingerprint"]),
        },
        "l3": {
            "version": str(l3["l3_version"]),
            "facts": int(l3["coverage_summary"]["facts_rows"]),
            "activity_bands": {k: int(v) for k, v in l3["activity"]["bands"].items()},
            "activity_mean": float(l3["activity"]["mean_score"]),
            "churn_stages": {k: int(v) for k, v in l3["churn"]["stages"].items()},
            "churn_mean_risk": float(l3["churn"]["mean_risk"]),
            "migration_available": int(l3["migration"]["genre_available"]),
            "migration_insufficient": int(l3["migration"]["insufficient_data"]),
            "migration_shift": float(l3["migration"]["mean_genre_shift"]),
            "calibration_status": str(l3["calibration_status"]),
        },
        "raw_files": raw_files,
    }


# ── 图表：Python 生成的 inline SVG（无 xmlns / 无脚本 / 无外链）──

def _svg_open(width: int, height: int) -> str:
    return f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">'


def bar_chart(items: list[dict], *, width: int = 720, height: int = 230,
              value_fmt=None) -> str:
    """竖排柱图：items = [{label, value, color}]；自动处理负值（零线为基线）。"""
    n = len(items)
    pad_l, pad_r, pad_t, pad_b = 12, 12, 28, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    vals = [float(it["value"]) for it in items]
    vmin, vmax = min([0.0] + vals), max([0.0] + vals)
    if vmax == vmin:
        vmax = vmin + 1.0
    span = (vmax - vmin) * 0.08
    vmin -= span
    vmax += span
    def y(v: float) -> float:
        return pad_t + (vmax - v) / (vmax - vmin) * plot_h
    y0 = y(0.0)
    slot = plot_w / n
    bw = min(56.0, slot * 0.5)
    parts = [_svg_open(width, height)]
    parts.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{width - pad_r}" y2="{y0:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
    for i, it in enumerate(items):
        v = float(it["value"])
        cx = pad_l + slot * (i + 0.5)
        x1, x2 = cx - bw / 2, cx + bw / 2
        yv = y(v)
        top, hgt = (yv, y0 - yv) if v >= 0 else (y0, yv - y0)
        parts.append(f'<rect x="{x1:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                     f'height="{max(hgt, 1.0):.1f}" rx="3" fill="{it.get("color", ACCENT)}"/>')
        text = value_fmt(v) if value_fmt else f"{v:.2f}"
        ty = top - 7 if v >= 0 else top + hgt + 14
        parts.append(f'<text x="{cx:.1f}" y="{ty:.1f}" text-anchor="middle" class="cv">{_esc(text)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height - 8}" text-anchor="middle" class="cl">{_esc(it["label"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(series: list[dict], *, width: int = 720, height: int = 250,
               y_fmt: str = "{:.1f}", x_ticks: int = 4, x_fmt: str | None = None) -> str:
    """折线图：series = [{name, points: [(x, y)], color, dashed?}]；右侧带图例。"""
    pad_l, pad_r, pad_t, pad_b = 44, 128, 20, 30
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmax = xmin + 1
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.08 or 1.0
    ymin, ymax = ymin - pad, ymax + pad
    def x(v: float) -> float:
        return pad_l + (v - xmin) / (xmax - xmin) * plot_w
    def y(v: float) -> float:
        return pad_t + (ymax - v) / (ymax - ymin) * plot_h
    parts = [_svg_open(width, height)]
    for i in range(5):
        v = ymax - (ymax - ymin) * i / 4
        yy = y(v)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{yy + 4:.1f}" text-anchor="end" class="cl">{_esc(y_fmt.format(v))}</text>')
    for i in range(x_ticks + 1):
        v = xmin + (xmax - xmin) * i / x_ticks
        xx = x(v)
        if x_fmt is not None:
            label = x_fmt.format(v)
        else:
            label = f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.2f}"
        parts.append(f'<text x="{xx:.1f}" y="{height - 8}" text-anchor="middle" class="cl">{_esc(label)}</text>')
    for s in series:
        pts = " ".join(f"{x(px):.1f},{y(py):.1f}" for px, py in s["points"])
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" '
                     f'stroke-width="2"{dash}/>')
    for i, s in enumerate(series):
        ly = pad_t + 8 + i * 18
        lx = pad_l + plot_w + 16
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" '
                     f'stroke="{s["color"]}" stroke-width="2"{dash}/>')
        parts.append(f'<text x="{lx + 24}" y="{ly + 4}" class="cl">{_esc(s["name"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def chart_block(svg: str, caption: str = "") -> str:
    cap = f'<p class="cap">{_esc(caption)}</p>' if caption else ""
    return f'<div class="chartwrap">{svg}{cap}</div>'


# ── 页面片段 ────────────────────────────────────────────────

def _kpi(value: str, label: str, note: str = "") -> str:
    note_html = f'<div class="kn">{_esc(note)}</div>' if note else ""
    return (f'<div class="kpi-item"><div class="kv">{_esc(value)}</div>'
            f'<div class="kl">{_esc(label)}</div>{note_html}</div>')


def _table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<table class="t"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def _header(stats: dict) -> str:
    return (
        '<header id="header">\n'
        '<h1>游戏社区用户行为洞察 Agent · 端到端报告</h1>\n'
        '<p class="sub">真实轨（用户现在什么状态）+ 模拟轨（干预下去会发生什么），'
        '全部数字来自各模块已落盘产出（聚合口径）</p>\n'
        f'<p class="meta">报告版本 <code>{_esc(REPORT_VERSION)}</code> · '
        f'输入指纹 <code>{_esc(stats["inputs_fingerprint"])}</code> · '
        '复现 <code>python -m report.build_report</code></p>\n'
        '</header>'
    )


def _story() -> str:
    return (
        '<section id="story">\n'
        '<h2>三问 · 两轨</h2>\n'
        '<div class="card">\n'
        '<p><strong>① 用户现在处于什么状态</strong> → 真实轨 L1 采集 → L2 特征 → L3 洞察'
        '（活跃度 / 兴趣迁移 / 流失风险）；<strong>② 怎么用 Agent 把洞察变成决策</strong> → '
        'harness 六环节（异常 → 人群 → 原因 → 风险 → 干预 → 实验）；'
        '<strong>③ 如果干预会发生什么</strong> → 模拟轨 S1 模拟器 → S2 Uplift → S3 Bandit → S4 接入 harness。</p>\n'
        '<p>真实轨用公开可观测行为痕给状态与风险；模拟轨在合成人群上验证干预决策的'
        '方法与边界。两条轨通过 harness 编排到一起：本报告按"真实轨 → S1 → S2 → S3 → S4"'
        '的顺序呈现各环节关键数字，最后一节是随结论一起引用的诚实边界。</p>\n'
        '</div>\n'
        '</section>'
    )


def _real_track(real: dict) -> str:
    l2, l3 = real["l2"], real["l3"]
    trunc = " · ".join(f"{k} {v}" for k, v in sorted(l2["by_surface"].items()))
    raw_card = (
        _kpi(_thousands(real["raw_files"]), "原始公开 JSON", "14 个公开数据面 / 人")
        if real["raw_files"] is not None else ""
    )
    kpis = (
        _kpi(_thousands(l2["n_users"]), "pilot 样本（人）")
        + raw_card
        + _kpi(_thousands(l2["events"]), "行为事件（时间线）")
        + _kpi(_thousands(l2["truncated"]), "截断用户", f"已标记 truncated_surfaces（{trunc}）")
        + _kpi(_thousands(l2["errors"]), "含采集错误用户", "wishlist 403 为隐私设置，不计入限流")
    )
    bands = l3["activity_bands"]
    activity_items = [
        {"label": "沉睡 dormant", "value": bands["dormant"], "color": MUTED},
        {"label": "低频 low", "value": bands["low"], "color": "#93c5fd"},
        {"label": "中频 mid", "value": bands["mid"], "color": "#60a5fa"},
        {"label": "高频 high", "value": bands["high"], "color": "#3b82f6"},
        {"label": "头部 top", "value": bands["top"], "color": ACCENT},
    ]
    churn = l3["churn_stages"]
    churn_items = [
        {"label": "活跃 active", "value": churn["active"], "color": GOOD},
        {"label": "沉默 30d", "value": churn["dormant_30"], "color": WARN},
        {"label": "沉默 60d", "value": churn["dormant_60"], "color": BAD},
        {"label": "沉默 90d+", "value": churn["dormant_90"], "color": DEEP},
    ]
    mig_den = l3["migration_available"] + l3["migration_insufficient"]
    return (
        '<section id="real-track">\n'
        f'<h2>真实轨（L1–L3）· 用户现在什么状态（{_kpi_ver(l2["version"], l3["version"])}）</h2>\n'
        f'<div class="kpi">{kpis}</div>\n'
        '<p class="note">公开可观测行为痕（评论 / 评分 / 收藏 / 关注 / 时间戳等），'
        '不含 App 埋点、曝光、停留、支付；未知时间的事件（time_kind=unknown）'
        '不参与任何时间序列与迁移计算。</p>\n'
        f'<div class="card"><h3>活跃度五档</h3>'
        f'{chart_block(bar_chart(activity_items, height=210, value_fmt=lambda v: f"{v:.0f}"), "")}'
        f'<p class="cap">活跃分均值 {_num(l3["activity_mean"], 2)}（0–100 档位口径，'
        f'calibration_status = {_esc(l3["calibration_status"])}）</p></div>\n'
        f'<div class="card"><h3>流失档（observed_inactivity 代理）</h3>'
        f'{chart_block(bar_chart(churn_items, height=210, value_fmt=lambda v: f"{v:.0f}"), "")}'
        f'<p class="cap">流失风险均值 {_num(l3["churn_mean_risk"], 2)}；'
        f'silent / churned 档为 0（阈值未经运营反馈校准，见文末边界）</p></div>\n'
        f'<div class="card"><h3>兴趣迁移</h3>'
        f'<p>可判定 {l3["migration_available"]} / {mig_den} 人'
        f'（{_pct(l3["migration_available"] / mig_den)}）｜平均迁移度 '
        f'{_num(l3["migration_shift"], 4)}｜数据不足 {l3["migration_insufficient"]} 人'
        '（迁移只在有标签轨迹的用户上计算）。</p></div>\n'
        f'<div class="card warn"><h3>校准状态</h3>'
        f'<p><code>calibration_status = {_esc(l3["calibration_status"])}</code>：'
        '流失 / 活跃阈值未经运营反馈校准，引用结论时必须带上该标记。'
        f'L2 输入指纹 <code>{_esc(l2["fingerprint"])}</code>、facts {_thousands(l3["facts"])} 行'
        '（Agent 唯一入口）。</p></div>\n'
        '</section>'
    )


def _kpi_ver(l2v: str, l3v: str) -> str:
    return f"{l2v} / {l3v}"


def _s1_section(s1: dict) -> str:
    arms = s1["arms"]
    items = [
        {"label": "对照（不触达）", "value": arms["control"]["reward"], "color": MUTED},
        {"label": "个性化推荐 rec", "value": arms["rec"]["reward"], "color": BAD},
        {"label": "沉默召回 recall", "value": arms["recall"]["reward"], "color": GOOD},
    ]
    kpis = (
        _kpi(_signed(arms["recall"]["reward"]), "召回臂净奖励", f"τ_struct {_signed(arms['recall']['tau'])}")
        + _kpi(_signed(arms["rec"]["reward"]), "推荐臂净奖励", "负收益：撑不起触达成本")
        + _kpi(_pct(s1["act30_gt0_share"]), "活跃（act_30d>0）占比", "沉默为主的人口结构")
    )
    return (
        '<section id="s1">\n'
        f'<h2>S1 · 用户模拟器：干预下去会发生什么（{s1["n_users"]} 人，seed {s1["seed"]}）</h2>\n'
        f'<div class="kpi">{kpis}</div>\n'
        f'<div class="card">{chart_block(bar_chart(items, value_fmt=_signed), "三臂净奖励均值（窗内互动增量 − 触达成本）")}</div>\n'
        '<p>净奖励 = 干预窗内互动增量 − 触达成本（rec 0.5 / recall 1.0，详见 simulator 参数）。'
        '召回臂在"沉默约 90 天"附近有钟形响应，整体净收益为正；推荐臂平均净收益为负——'
        '这条结构随后被 S2（推荐臂头部名额无正收益）与 S3（学习策略几乎不选 rec）反复验证。</p>\n'
        f'<p class="note">人口：{_esc(s1["population_path"])}（拟合行数 {s1["fit_rows"]}）｜'
        f'{s1["days"]} 天 · 奖励窗 {s1["window"]} 天；τ 为结构化真值 CATE（非真实业务效应）。</p>\n'
        '</section>'
    )


def _s2_section(s2: dict) -> str:
    rows = [
        [f["protocol"], f["arm"], f["learner"], _num(f["tau_hat"]), _num(f["tau_struct"]),
         _signed(f["bias"], 3), _num(f["spearman_struct"])]
        for f in s2["fidelity"]
    ]
    rct = s2["policy_reference"]["rct"]
    full = s2["policy_reference"]["full"]
    policy_rows = [
        [name, _num(rct["recall"][name]), _num(full["recall"][name])]
        for name in ("model_hgb", "model_ridge", "random", "oracle_struct", "oracle_ind")
    ]
    recall_items = [
        {"label": "HGB 模型", "value": rct["recall"]["model_hgb"], "color": ACCENT},
        {"label": "Ridge 模型", "value": rct["recall"]["model_ridge"], "color": VIOLET},
        {"label": "随机", "value": rct["recall"]["random"], "color": MUTED},
        {"label": "oracle 上界", "value": rct["recall"]["oracle_struct"], "color": INK},
    ]
    cal_series = [
        {"name": "τ̂（模型预测）", "points": [(c["bin"], c["hat"]) for c in s2["calibration"]], "color": ACCENT},
        {"name": "τ（真值）", "points": [(c["bin"], c["struct"]) for c in s2["calibration"]],
         "color": GOOD, "dashed": True},
    ]
    ratio = rct["recall"]["model_hgb"] / rct["recall"]["oracle_struct"]
    rec = rct["rec"]
    hgb = next(f for f in s2["fidelity"]
               if f["protocol"] == "rct" and f["arm"] == "recall" and f["learner"] == "hgb")
    rec_hgb = next(f for f in s2["fidelity"]
                   if f["protocol"] == "rct" and f["arm"] == "rec" and f["learner"] == "hgb")
    return (
        '<section id="s2">\n'
        f'<h2>S2 · Uplift：谁值得干预（{s2["n_users"]} 人，seed {s2["seed"]}，参考档 k={s2["reference_k_pct"]}%）</h2>\n'
        '<div class="card"><h3>保真度（rct / full 双协议，按用户切分）</h3>'
        + _table(["协议", "臂", "学习器", "τ̂ 均值", "τ 真值", "偏差", "Spearman(结构)"], rows)
        + '<p class="cap">训练 '
        + f'{_thousands(s2["n_train"])} / 评测 {_thousands(s2["n_holdout"])} 人；'
        'rec = 个性化推荐，recall = 沉默召回。full 协议信息更多（同臂全观测），保真度更高。</p></div>\n'
        f'<div class="card"><h3>召回臂策略价值 @ 头部 {s2["reference_k_pct"]}% 名额（rct）</h3>'
        + chart_block(bar_chart(recall_items, value_fmt=lambda v: f"{v:.2f}"),
                      "按 τ̂ 排序取头部名额后的实际召回收益；oracle_struct 为真值排序上界")
        + f'<p>模型方案 HGB {_num(rct["recall"]["model_hgb"])} = oracle 上界的 {_pct(ratio)}；'
        f'随机排序只有 {_num(rct["recall"]["random"])}——排序模型把名额花在了对的人身上。</p>'
        + _table([f"方案（recall @ k={s2['reference_k_pct']}%）", "rct", "full"], policy_rows)
        + f'<p class="cap">推荐臂 @ k={s2["reference_k_pct"]}%（rct）：HGB {_num(rec["model_hgb"])} / '
        f'Ridge {_num(rec["model_ridge"])} / 随机 {_num(rec["random"])} / oracle {_num(rec["oracle_struct"])}'
        '——模型救不回负期望的臂（与 S3"几乎不选 rec"互证）。</p></div>\n'
        f'<div class="card"><h3>校准（rct · recall · HGB 十分位）</h3>'
        + chart_block(line_chart(cal_series, y_fmt="{:.2f}", x_ticks=9),
                      "横轴 = 按 τ̂ 排序的十分位（1 低 → 10 高）；折线贴近 = 排序可信")
        + f'<p>τ̂ 均值 {_num(hgb["tau_hat"])}（真值 {_num(hgb["tau_struct"])}），'
        f'Spearman {_num(hgb["spearman_struct"])}；推荐臂 τ̂ {_num(rec_hgb["tau_hat"])}'
        f'（真值 {_num(rec_hgb["tau_struct"])}）——召回臂的效果远比推荐臂可排序。</p></div>\n'
        '</section>'
    )


def _s3_section(s3: dict) -> str:
    by_name = {p["policy"]: p for p in s3["policies"]}
    linucb = by_name["linucb"]
    random_p = by_name["random"]
    fixed = by_name["fixed_recall"]
    rows = [
        [p["policy"], _signed(p["mean_reward"], 3), _signed(p["head20"], 3),
         _signed(p["tail20"], 3), _thousands(p["cum_regret"]), _signed(p["audit_mean"], 3),
         _num(p["audit_frac"]), _pct(p["share_recall"], 0)]
        for p in s3["policies"]
    ]
    oracle_mean = linucb["oracle_mean"]
    oracle_total = oracle_mean * s3["n_online"]
    cur_series = [
        {"name": "LinUCB", "points": s3["curve"]["linucb"], "color": ACCENT},
        {"name": "Thompson", "points": s3["curve"]["thompson"], "color": VIOLET},
        {"name": "random", "points": s3["curve"]["random"], "color": MUTED},
        {"name": "固定 recall", "points": s3["curve"]["fixed_recall"], "color": GOOD},
        {"name": "oracle 上界", "points": [(x, oracle_mean) for x, _ in s3["curve"]["linucb"]],
         "color": BAD, "dashed": True},
    ]
    kpis = (
        _kpi(_pct(linucb["audit_frac"]), "审计 ÷ oracle（LinUCB）", "冻结评测：不再探索，只考知识")
        + _kpi(f"{_signed(linucb['head20'], 3)} → {_signed(linucb['tail20'], 3)}", "前 20 步 → 后 20 步", "学习信号")
        + _kpi(_signed(linucb["mean_reward"], 3), "在线均值", f"= random 的 {linucb['mean_reward'] / random_p['mean_reward']:.1f} 倍")
    )
    return (
        '<section id="s3">\n'
        f'<h2>S3 · Bandit：每次触达选哪条臂、能学多快（在线 {_thousands(s3["n_online"])} / '
        f'审计 {_thousands(s3["n_audit"])} 人）</h2>\n'
        f'<div class="kpi">{kpis}</div>\n'
        f'<div class="card">{chart_block(line_chart(cur_series, y_fmt="{:.1f}", x_fmt="{:.0f}"), "在线学习曲线（每 105 步采样，跳开第 0 步单点噪声）｜虚线 = oracle 上界终值 " + _num(oracle_mean))}</div>\n'
        '<div class="card"><h3>七策略对照（净奖励口径）</h3>'
        + _table(["策略", "在线均值", "head20", "tail20", "累计遗憾", "审计均值", "审计÷oracle", "recall 占比"], rows)
        + '</div>\n'
        '<p>学习确实发生：前 20 步均值 ≈ 0 → 后 20 步已贴近 oracle；两条学习策略几乎打平。'
        f'审计冻结后知识仍值 oracle 的 {_pct(linucb["audit_frac"])}'
        f'（固定召回 {_pct(fixed["audit_frac"])}、随机 {_pct(random_p["audit_frac"])}）——'
        f'收益不来自探索的运气；累计遗憾 {_thousands(linucb["cum_regret"])} ≈ oracle 总量'
        f'（{_thousands(oracle_total)}）的 {_pct(linucb["cum_regret"] / oracle_total, 0)}。</p>\n'
        '</section>'
    )


def _s4_section(s4: dict) -> str:
    wc = s4["modes"]["with_critic"]
    nc = s4["modes"]["no_critic"]
    trace_rows = [
        [str(t["step"]), t["tool"], t["note"]] for t in s4["trace"]["with_critic"]
    ]
    rows = [
        [name, _arms_text(m["arms"]), _num(m["batch_mean_reward"], 4),
         _pct(m["online_vs_oracle"]), _pct(m["audit_vs_oracle"]),
         f"{m['veto_count']}（{_pct(m['veto_rate'])}）"]
        for name, m in (("with_critic（带安全门）", wc), ("no_critic（不带）", nc))
    ]
    cohort = s4["cohort"]
    kpis = (
        _kpi(_thousands(cohort["size"]), "圈定人群（人）", f"占在线池 {_pct(cohort['share'])}")
        + _kpi(_thousands(wc["batch_size"]), "批次触达（人次）", _arms_text(wc["arms"]))
        + _kpi(_pct(wc["veto_rate"]), "否决率（带 Critic）", f"{wc['veto_count']} 人次被降级为对照")
        + _kpi(_pct(wc["audit_vs_oracle"]), "审计 ÷ oracle（带 Critic）", "合成世界审计池")
    )
    return (
        '<section id="s4">\n'
        f'<h2>S4 · 接 harness：Agent 决策循环里的干预与安全门（{_thousands(s4["n_users"])} 人，seed {s4["seed"]}）</h2>\n'
        f'<div class="kpi">{kpis}</div>\n'
        '<div class="card"><h3>循环轨迹（with_critic，七步）</h3>'
        + _table(["步", "工具", "关键观察（合成世界可观测口径）"], trace_rows)
        + f'<p class="cap">no_critic 轨迹同序同工具，仅第 5–6 步分配不同（{_arms_text(nc["arms"])}、零否决）。</p></div>\n'
        '<div class="card"><h3>带 / 不带 Critic 对照</h3>'
        + _table(["模式", "批次臂分布", "批次均值", "在线÷oracle", "审计÷oracle", "否决（率）"], rows)
        + '</div>\n'
        f'<p>安全门代价几乎为零：带 Critic 否决 {wc["veto_count"]} 人次（{_pct(wc["veto_rate"])}），'
        f'被否决的提议自动降级为对照（宁不打扰）；两模式批次均值 '
        f'{_num(wc["batch_mean_reward"], 4)} vs {_num(nc["batch_mean_reward"], 4)}、'
        f'审计÷oracle {_pct(wc["audit_vs_oracle"])} vs {_pct(nc["audit_vs_oracle"])}。'
        '证据门槛 min_obs = 3：所选臂至少 3 次观测、悲观下界优于对照才开价值门，'
        '避免单次观测"过度自信"提前放行。</p>\n'
        '</section>'
    )


def _arms_text(arms: dict) -> str:
    return f"对照 {arms['control']} / 推荐 {arms['rec']} / 召回 {arms['recall']}"


def _limits(real: dict | None) -> str:
    items = []
    if real:
        items.append(
            f'<li><strong>样本</strong>：{_thousands(real["l2"]["n_users"])} 人 pilot，来自'
            '"近期有公开活动"的用户 → 结论不代表全站；迁移仅 '
            f'{real["l3"]["migration_available"]} 人可判定。</li>'
        )
        items.append(
            '<li><strong>标签</strong>：单快照、无纵向回访 → 流失档是「observed_inactivity」'
            '（公开活动沉默代理），<strong>不是</strong>平台真实流失；'
            f'阈值 <code>calibration_status = {_esc(real["l3"]["calibration_status"])}</code>。</li>'
        )
    items.append(
        '<li><strong>行为痕 ≠ 埋点</strong>：只有评论 / 评分 / 收藏 / 关注 / 时间戳等公开痕迹；'
        '不含曝光、停留、支付等内部数据。</li>'
    )
    items.append(
        '<li><strong>时间线</strong>：time_kind=unknown 的事件不参与任何时间序列与迁移计算。</li>'
    )
    items.append(
        '<li><strong>模拟轨是方法验证</strong>：数字（召回净奖励、80.4%、78.8% 等）来自合成世界，'
        '不迁移到真实业务；各模块种子不同（S1 seed 7 / S2 seed 11 / S3–S4 seed 13），按各自口径展示。</li>'
    )
    items.append(
        '<li><strong>抽样波动</strong>：S3 审计池 1,800 人冻结评测，末位数字有抽样波动'
        '（学习信号已跨种子复核）。</li>'
    )
    items.append(
        '<li><strong>合规</strong>：只采集公开可见信息；限速，403/429 即停，不重试、不绕过权限；'
        'ip_location / device / gender 等只在聚合层出现；关注 / 粉丝的对方 id 只出计数。</li>'
    )
    return (
        '<section id="limits">\n'
        '<h2>诚实边界（随结论一起引用）</h2>\n'
        '<div class="card warn"><ul>' + "".join(items) + '</ul></div>\n'
        '</section>'
    )


def _footer() -> str:
    return (
        '<footer id="footer">\n'
        '<p>复现：<code>python -m report.build_report</code>（仓库根目录；合成轨缺失时'
        '先按各模块 README 重跑）。本页零依赖、零 CDN、离线直开，'
        '图表为 Python 生成的 inline SVG，仅含聚合数字。</p>\n'
        '<p>模块：'
        '<a href="../L1_data_source/">L1 采集</a> · <a href="../L2_features/">L2 特征</a> · '
        '<a href="../L3_insights/">L3 洞察</a> · <a href="../simulator/">simulator</a> · '
        '<a href="../uplift/">uplift</a> · <a href="../bandit/">bandit</a> · '
        '<a href="../harness/">harness</a> · <a href="../README.md">根 README</a></p>\n'
        '</footer>'
    )


# ── 渲染与产出 ──────────────────────────────────────────────

def sections_present(stats: dict) -> list[str]:
    """本报告实际渲染的章节序（真实轨缺失时不含 real-track）。"""
    secs = ["header", "story"]
    if stats["real"]:
        secs.append("real-track")
    secs += ["s1", "s2", "s3", "s4", "limits", "footer"]
    return secs


CSS = """
:root{--ink:#1f2937;--muted:#6b7280;--accent:#2563eb;--good:#059669;--bad:#dc2626;
--grid:#e5e7eb;--bg:#f8fafc;--card:#ffffff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.75 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 20px 56px}
h1{font-size:26px;line-height:1.35;margin:0 0 10px}
h2{font-size:19px;margin:34px 0 8px}
h3{font-size:15px;margin:0 0 10px}
p{margin:10px 0}
.sub{color:var(--muted);margin:0 0 6px}
.meta{color:var(--muted);font-size:13px;margin:0 0 8px}
code{background:#eef2ff;color:#3730a3;padding:1px 6px;border-radius:5px;font-size:12.5px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:12px;
padding:16px 18px;margin:14px 0}
.card.warn{border-left:4px solid var(--bad)}
.card ul{margin:6px 0 0;padding-left:20px}
.card li{margin:6px 0}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
gap:10px;margin:12px 0}
.kpi-item{background:var(--card);border:1px solid var(--grid);border-radius:10px;
padding:12px 14px}
.kpi-item .kv{font-size:20px;font-weight:700;letter-spacing:.2px}
.kpi-item .kl{font-size:12.5px;color:var(--muted);margin-top:2px}
.kpi-item .kn{font-size:12px;color:var(--muted);margin-top:4px}
.note{color:var(--muted);font-size:13px}
.cap{color:var(--muted);font-size:12.5px;margin:6px 0 0}
table.t{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 2px}
table.t th{text-align:left;color:var(--muted);font-weight:600;
border-bottom:2px solid var(--grid);padding:6px 9px;white-space:nowrap}
table.t td{padding:6px 9px;border-bottom:1px solid var(--grid);white-space:nowrap}
table.t tr:last-child td{border-bottom:none}
.chartwrap{margin:6px 0 2px}
svg.chart{width:100%;height:auto;display:block}
svg.chart text{font-family:inherit}
svg.chart .cl{fill:#6b7280;font-size:12px}
svg.chart .cv{fill:#1f2937;font-size:12px;font-weight:600}
footer{margin-top:40px;color:var(--muted);font-size:13px;
border-top:1px solid var(--grid);padding-top:14px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
"""


def render(stats: dict, sources: dict) -> str:
    """渲染完整 HTML（无脚本 / 无外链 / 无时间戳；逐字节稳定）。"""
    parts = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN">',
        '<head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>游戏社区用户行为洞察 Agent · 端到端报告</title>',
        f"<style>{CSS}</style>",
        '</head>',
        '<body>',
        '<main>',
        _header(stats),
        _story(),
    ]
    if stats["real"]:
        parts.append(_real_track(stats["real"]))
    parts += [
        _s1_section(stats["s1"]),
        _s2_section(stats["s2"]),
        _s3_section(stats["s3"]),
        _s4_section(stats["s4"]),
        _limits(stats["real"]),
        _footer(),
        '</main>',
        '</body>',
        '</html>',
    ]
    return "\n".join(parts) + "\n"


def run_report(root: Path = PROJECT_ROOT) -> dict:
    """跑一遍完整流程（收集 + 渲染，纯内存，deterministic）。"""
    stats, sources = collect(root)
    return {"stats": stats, "sources": sources, "html": render(stats, sources)}


def run_is_deterministic(first: dict, second: dict) -> bool:
    """双跑一致性：HTML 逐字节 + 输入快照 + 统计值全等。"""
    return (
        first["html"] == second["html"]
        and first["sources"] == second["sources"]
        and first["stats"] == second["stats"]
    )


def _highlights(stats: dict) -> dict:
    s1, s2 = stats["s1"], stats["s2"]
    by_name = {p["policy"]: p for p in stats["s3"]["policies"]}
    wc = stats["s4"]["modes"]["with_critic"]
    out = {
        "s1_recall_mean_reward": s1["arms"]["recall"]["reward"],
        "s1_rec_mean_reward": s1["arms"]["rec"]["reward"],
        "s2_recall_model_hgb_at_k20": s2["policy_reference"]["rct"]["recall"]["model_hgb"],
        "s2_recall_oracle_struct_at_k20": s2["policy_reference"]["rct"]["recall"]["oracle_struct"],
        "s3_linucb_audit_vs_oracle": by_name["linucb"]["audit_frac"],
        "s4_with_critic_veto_rate": wc["veto_rate"],
        "s4_with_critic_audit_vs_oracle": wc["audit_vs_oracle"],
    }
    if stats["real"]:
        out["real_track_users"] = stats["real"]["l2"]["n_users"]
        out["real_track_events"] = stats["real"]["l2"]["events"]
    return out


def write_outputs(run: dict, out_dir: Path, determinism: dict) -> dict:
    """落盘 index.html + _manifest.json（时间只进 manifest，HTML 逐字节稳定）。"""
    index_path = out_dir / "index.html"
    _write_text(index_path, run["html"])
    manifest = {
        "report_version": REPORT_VERSION,
        "generated_at": _now(),
        "inputs_fingerprint": run["stats"]["inputs_fingerprint"],
        "sources": run["sources"],
        "sections": sections_present(run["stats"]),
        "highlights": _highlights(run["stats"]),
        "outputs": {
            "index.html": {
                "bytes": index_path.stat().st_size,
                "sha256": _sha16(index_path.read_bytes()),
            }
        },
        "determinism": determinism,
    }
    _write_text(out_dir / "_manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


# ── CLI ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="S5 展示层：全链路产出 → 静态自包含 HTML 报告")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 report/）")
    ap.add_argument("--no-verify", action="store_true", help="跳过双跑一致性校验")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        first = run_report()
    except MissingInputsError as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 4
    except ValueError as exc:
        print(f"[stop] 参数无效：{exc}", file=sys.stderr)
        return 2

    determinism = {"checked": False}
    if not args.no_verify:
        second = run_report()
        same = run_is_deterministic(first, second)
        determinism = {"checked": True, "identical": bool(same)}
        if not same:
            print("[stop] 双跑不一致：存在非确定性来源，拒绝落盘", file=sys.stderr)
            return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_outputs(first, out_dir, determinism)

    stats = first["stats"]
    hp = manifest["highlights"]
    real_txt = "含" if stats["real"] else "不含"
    print(f"[info] 报告 {REPORT_VERSION}：sections {len(manifest['sections'])}（{real_txt}真实轨）"
          f"｜输入 {len(manifest['sources'])} 份｜输入指纹 {manifest['inputs_fingerprint']}")
    print(f"[info] 关键数字：S1 召回净奖励 {_signed(hp['s1_recall_mean_reward'])}"
          f"｜S3 审计÷oracle {_pct(hp['s3_linucb_audit_vs_oracle'])}"
          f"｜S4 否决率 {_pct(hp['s4_with_critic_veto_rate'])}")
    out_info = manifest["outputs"]["index.html"]
    print(f"[done] → {_rel(out_dir)}/index.html（{out_info['bytes']:,} B）＋ _manifest.json")
    if determinism["checked"]:
        print(f"[ok] 双跑一致性：{determinism['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
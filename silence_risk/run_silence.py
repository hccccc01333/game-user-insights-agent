#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 · ML 沉默预测正式版（silence_risk v1）：把「谁可能沉默」做成可复算、可验收的评测。

先说人话：
    对每个「近期活跃」的公开行为锚点 T，问：他未来 h 天内还会公开活动吗？
    基线 = 现有 recency heuristic：距最近一次公开事件的天数（gap_days，越大越危险）。
    挑战者 = LogReg / HistGBM（只用 T 及之前的公开事件特征）。
    纪律：只看**时间外推**（训旧段 → 测新段）的 AUC 是否**显著**高于基线——
    同期拆分 / 训练集 AUC 不算数（M1 达标线）。

两个评测协议（同一面板、同一特征、同一模型，只换切分）：
    group_cv{cv}  GroupKFold(cv) 按 uid 分组 OOF：防同一用户跨折泄漏（同期口径）
    time_extrap   训练 = 锚点 < split_until+1 天（+08），测试 = 其后（真·时间外推）

运行（在仓库根目录）：
    python -m silence_risk.run_silence
    python -m silence_risk.run_silence --no-verify --feature-set trunk_types

产出（默认 data/processed/silence/；仅聚合指标与 uid_hash，不含原始行为）：
    silence_metrics.json     配置 / 面板统计 / 双协议 × scorer 指标 / ΔAUC 显著性 / M1 裁决
    silence_calibration.csv  分位校准表（仅概率模型）
    silence_predictions.csv  逐样本分数长表（protocol/split/uid_hash/anchor_ts/label/score_*）
    _manifest.json           版本 / 输入指纹 / 输出指纹 / 双跑一致性
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import SILENCE_VERSION
from .evaluate import calibration_table, cluster_bootstrap_auc_diff, metrics
from .features import FEATURE_SETS, build_features, feature_columns
from .models import LEARNERS, hyperparams_snapshot, make_classifier
from .panel import PanelConfig, build_panel, load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMELINE = PROJECT_ROOT / "data" / "processed" / "user_features" / "timeline.jsonl"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "raw" / "user_profile" / "_index.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "silence"
TZ_CN = timezone(timedelta(hours=8))

GROUP_PROTOCOL = "group_cv"
TIME_PROTOCOL = "time_extrap"
PROTOCOLS = (GROUP_PROTOCOL, TIME_PROTOCOL)
HEURISTIC = "heuristic"
SCORERS = (HEURISTIC, *LEARNERS)
N_BOOT = 1000
PREDICTION_COLUMNS = (
    "protocol", "split", "uid_hash", "anchor_ts", "label", "left_truncated",
    "score_heuristic", "score_logreg", "score_hgb",
)
NOTES = (
    "口径：Future Public Inactivity Prediction（锚点 T；(T, T+h] 无 exact 事件即沉默），不叫 churn",
    "heuristic 基线 = gap_days（距最近一次公开事件天数，recency 规则），非概率 → 不上 Brier / 校准",
    "logreg 未使用 class_weight（与可行性快检的差异）：概率按基准率校准；排序 / AUC 不受影响",
    "特征只取自锚点 T 及之前的公开事件；产出只含聚合指标与 uid_hash，不含原始行为明细",
    "M1 达标线：只看时间外推，AUC 差（hgb vs recency）95% 聚类 bootstrap 下界 > 0",
)


# ── 协议与打分 ──────────────────────────────────────────────

def split_ts_from(split_until: str) -> int:
    """split_until（含）→ 边界 epoch：split_until+1 天 00:00（+08）。"""
    try:
        d = date.fromisoformat(split_until)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"split_until 需为 YYYY-MM-DD（当前 {split_until!r}）") from exc
    return int(datetime.combine(d + timedelta(days=1), time.min, tzinfo=TZ_CN).timestamp())


def heuristic_scores(X: pd.DataFrame) -> np.ndarray:
    """recency 基线分数：gap_days（越大越危险）。"""
    return X["gap_days"].to_numpy(dtype=float)


def group_cv_oof(
    X: pd.DataFrame, y, groups, cv: int, learner: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """GroupKFold(cv) 按 uid 分组 OOF 概率；返回 (oof, fold_ids[i]=行 i 所在测试折)。"""
    n = len(y)
    oof = np.full(n, np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    splitter = GroupKFold(n_splits=cv)
    for k, (tr_idx, te_idx) in enumerate(splitter.split(X, y, groups)):
        model = make_classifier(learner, seed).fit(X.iloc[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict_proba(X.iloc[te_idx])[:, 1]
        fold_ids[te_idx] = k
    if (fold_ids < 0).any():
        raise ValueError("group_cv：OOF 覆盖不完整")
    return oof, fold_ids


def protocol_scores(
    protocol: str, X: pd.DataFrame, y, groups, anchors, *, cv: int, split_ts: int, seed: int
) -> dict:
    """一个协议下全部 scorer 的分数。

    返回 {"eval_mask": bool 数组, "scores": {scorer: 分数数组}, "split": "oof"/"test",
          "fold_ids": 数组或 None}；time_extrap 只在 eval_mask 行有分数。
    """
    n = len(y)
    if protocol == GROUP_PROTOCOL:
        if len(np.unique(groups)) < cv:
            raise ValueError(f"group_cv：用户数（{len(np.unique(groups))}）少于折数（{cv}）")
        scores = {HEURISTIC: heuristic_scores(X)}
        fold_ids = None
        for learner in LEARNERS:
            oof, fold_ids = group_cv_oof(X, y, groups, cv, learner, seed)
            scores[learner] = oof
        return {"eval_mask": np.ones(n, dtype=bool), "scores": scores, "split": "oof", "fold_ids": fold_ids}
    if protocol == TIME_PROTOCOL:
        tr, te = anchors < split_ts, anchors >= split_ts
        if not tr.any():
            raise ValueError("time_extrap：训练侧为空（split_until 早于全部锚点）")
        if not te.any():
            raise ValueError("time_extrap：测试侧为空（split_until 晚于全部锚点）")
        for side, m in (("训练", tr), ("测试", te)):
            if len(np.unique(y[m])) < 2:
                raise ValueError(f"time_extrap：{side}侧类别不全（仅 {np.unique(y[m]).tolist()}），无法评估")
        scores = {HEURISTIC: heuristic_scores(X)}
        for learner in LEARNERS:
            model = make_classifier(learner, seed).fit(X.loc[tr], y[tr])
            prob = np.full(n, np.nan)
            prob[te] = model.predict_proba(X.loc[te])[:, 1]
            scores[learner] = prob
        return {"eval_mask": te, "scores": scores, "split": "test", "fold_ids": None}
    raise ValueError(f"未知协议：{protocol!r}（支持 {PROTOCOLS}）")


# ── 核心：跑一遍完整流程（纯内存，deterministic）────────────

def run_all(
    *,
    timeline_path: Path,
    index_path: Path,
    cfg: PanelConfig,
    cv: int = 5,
    split_until: str = "2025-12-31",
    top_frac: float = 0.10,
    seed: int = 7,
    feature_set: str = "trunk",
    n_boot: int = N_BOOT,
) -> dict:
    """面板 → 特征 → 双协议 × 三 scorer 评测 → ΔAUC / 消融 / 稳健性；同参同种子逐值一致。"""
    if isinstance(cv, bool) or not isinstance(cv, int) or cv < 2:
        raise ValueError(f"cv 需为 ≥2 的整数（当前 {cv!r}）")
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(f"top_frac 需在 (0,1] 内（当前 {top_frac}）")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f"n_boot 需为正整数（当前 {n_boot!r}）")
    cols_main = feature_columns(feature_set)  # 未知名在此显式报错
    split_ts = split_ts_from(split_until)

    timeline_path, index_path = Path(timeline_path), Path(index_path)
    timeline = load_jsonl(timeline_path)
    index = load_jsonl(index_path)
    panel, stats = build_panel(timeline, index, cfg)
    if panel.empty:
        raise ValueError("面板为空：没有满足口径的样本（检查输入路径 / 面板参数）")
    if panel["label"].nunique() < 2:
        raise ValueError("面板标签单一类别，无法评估")

    features = build_features(panel, timeline, cfg)
    y = panel["label"].to_numpy(dtype=int)
    groups = panel["uid_hash"].to_numpy()
    anchors = panel["anchor_ts"].to_numpy()
    left_trunc = panel["left_truncated"].to_numpy()
    X_main = features.loc[:, list(cols_main)]
    X_alt = None if feature_set == "trunk_types" else features.loc[:, list(FEATURE_SETS["trunk_types"])]

    protocols: dict[str, dict] = {}
    prediction_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    for proto in PROTOCOLS:
        key = f"{GROUP_PROTOCOL}{cv}" if proto == GROUP_PROTOCOL else TIME_PROTOCOL
        main = protocol_scores(proto, X_main, y, groups, anchors, cv=cv, split_ts=split_ts, seed=seed)
        mask = main["eval_mask"]
        res: dict = {
            "protocol": key,
            "split": main["split"],
            "n_eval": int(mask.sum()),
            "pos_rate": float(y[mask].mean()),
            "scorers": {},
            "deltas": {},
        }
        for scorer in SCORERS:
            score = main["scores"][scorer]
            res["scorers"][scorer] = metrics(
                y[mask], score[mask], probabilistic=scorer != HEURISTIC, top_frac=top_frac
            )
            if scorer != HEURISTIC:
                cal = calibration_table(y[mask], score[mask], bins=10)
                cal.insert(0, "protocol", key)
                cal.insert(1, "scorer", scorer)
                calibration_frames.append(cal)
        for learner in LEARNERS:
            res["deltas"][f"{learner}_vs_heuristic"] = cluster_bootstrap_auc_diff(
                y[mask], main["scores"][learner][mask], main["scores"][HEURISTIC][mask],
                groups[mask], n_boot=n_boot, seed=seed,
            )

        # 消融：主干 → 主干 + 类型占比（同一评测行）
        res["ablation_trunk_types"] = {}
        if X_alt is None:
            for learner in LEARNERS:
                auc = res["scorers"][learner]["roc_auc"]
                res["ablation_trunk_types"][learner] = {
                    "auc_trunk": auc, "auc_trunk_types": auc, "delta": 0.0,
                }
        else:
            alt = protocol_scores(proto, X_alt, y, groups, anchors, cv=cv, split_ts=split_ts, seed=seed)
            for learner in LEARNERS:
                auc_alt = metrics(
                    y[mask], alt["scores"][learner][mask], probabilistic=True, top_frac=top_frac
                )["roc_auc"]
                auc_trunk = res["scorers"][learner]["roc_auc"]
                res["ablation_trunk_types"][learner] = {
                    "auc_trunk": auc_trunk, "auc_trunk_types": auc_alt, "delta": auc_alt - auc_trunk,
                }

        # 稳健性：时间外推中"长特征窗完整"（非左截断）子集
        if proto == TIME_PROTOCOL:
            sub = mask & ~left_trunc
            if int(sub.sum()) >= 20 and len(np.unique(y[sub])) == 2:
                res["robust_no_left_trunc"] = {
                    "n_eval": int(sub.sum()),
                    "pos_rate": float(y[sub].mean()),
                    "scorers": {
                        s: metrics(y[sub], main["scores"][s][sub],
                                   probabilistic=s != HEURISTIC, top_frac=top_frac)
                        for s in SCORERS
                    },
                    "deltas": {
                        "hgb_vs_heuristic": cluster_bootstrap_auc_diff(
                            y[sub], main["scores"]["hgb"][sub], main["scores"][HEURISTIC][sub],
                            groups[sub], n_boot=n_boot, seed=seed,
                        )
                    },
                }
            else:
                res["robust_no_left_trunc"] = {
                    "skipped": "子集样本不足或类别单一", "n_eval": int(sub.sum()),
                }

        pred = pd.DataFrame(
            {
                "protocol": key,
                "split": main["split"],
                "uid_hash": groups,
                "anchor_ts": anchors,
                "label": y,
                "left_truncated": left_trunc,
            }
        )
        for scorer in SCORERS:
            pred[f"score_{scorer}"] = np.round(main["scores"][scorer], 6)
        if proto == TIME_PROTOCOL:
            pred = pred.loc[mask]
        prediction_frames.append(pred.loc[:, list(PREDICTION_COLUMNS)].reset_index(drop=True))
        protocols[key] = res

    predictions = pd.concat(prediction_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)

    scorer_verdicts = {
        learner: {
            **protocols[TIME_PROTOCOL]["deltas"][f"{learner}_vs_heuristic"],
            "passed": bool(protocols[TIME_PROTOCOL]["deltas"][f"{learner}_vs_heuristic"]["ci_low"] > 0),
        }
        for learner in LEARNERS
    }
    m1 = {
        "criterion": "时间外推（训旧 → 测新）**任一挑战者** AUC 显著 > recency 基线："
                     "ΔAUC（挑战者 vs heuristic(gap_days)）95% 聚类 bootstrap 下界 > 0",
        "scorers": scorer_verdicts,
        "passed_scorers": [l for l in LEARNERS if scorer_verdicts[l]["passed"]],
        "failed_scorers": [l for l in LEARNERS if not scorer_verdicts[l]["passed"]],
        "passed": any(scorer_verdicts[l]["passed"] for l in LEARNERS),
    }

    config = {
        "timeline": _rel(timeline_path),
        "index": _rel(index_path),
        "panel": cfg.snapshot(),
        "cv": int(cv),
        "split_until": split_until,
        "split_ts": int(split_ts),
        "top_frac": float(top_frac),
        "seed": int(seed),
        "feature_set": feature_set,
        "feature_columns": list(cols_main),
        "n_boot": int(n_boot),
        "protocols": [f"{GROUP_PROTOCOL}{cv}", TIME_PROTOCOL],
        "scorers": list(SCORERS),
    }
    inputs = {
        "timeline": {"path": _rel(timeline_path), "n_records": len(timeline),
                     "sha256_16": _sha16(timeline_path)},
        "index": {"path": _rel(index_path), "n_records": len(index),
                  "sha256_16": _sha16(index_path)},
    }
    return {
        "panel": panel,
        "features": features,
        "stats": stats,
        "predictions": predictions,
        "calibration": calibration,
        "protocols": protocols,
        "m1": m1,
        "config": config,
        "inputs": inputs,
    }


def run_is_deterministic(first: dict, second: dict) -> bool:
    """双跑一致性：结构化结果逐值相等 + 四个产出帧逐值相等。"""
    for key in ("stats", "protocols", "m1", "config", "inputs"):
        a = json.dumps(first[key], sort_keys=True, ensure_ascii=False, default=str)
        b = json.dumps(second[key], sort_keys=True, ensure_ascii=False, default=str)
        if a != b:
            return False
    for key in ("panel", "features", "predictions", "calibration"):
        if not first[key].equals(second[key]):
            return False
    return True


# ── 汇总与落盘 ──────────────────────────────────────────────

def _now() -> str:
    return datetime.now(TZ_CN).isoformat(timespec="seconds")


def _sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _rel(path: Path) -> str:
    """终端与产出只展示相对路径（避免本地绝对路径进入可复制出去的输出）。"""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return Path(path).name


def build_metrics(run: dict, determinism: dict) -> dict:
    """silence_metrics.json：主结果文件（配置 + 面板统计 + 双协议 + M1 裁决）。"""
    return {
        "silence_version": SILENCE_VERSION,
        "generated_at": _now(),
        "config": run["config"],
        "panel_stats": run["stats"],
        "models": hyperparams_snapshot(LEARNERS),
        "protocols": run["protocols"],
        "m1_verdict": run["m1"],
        "determinism": determinism,
        "notes": list(NOTES),
    }


def write_outputs(run: dict, out_dir: Path, determinism: dict) -> dict[str, dict]:
    """落盘四个产出：metrics / calibration / predictions / _manifest（含输出指纹）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "silence_metrics.json"
    metrics_path.write_text(
        json.dumps(build_metrics(run, determinism), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    frames = {
        "silence_calibration.csv": run["calibration"].round(4),
        "silence_predictions.csv": run["predictions"].round(6),
    }
    for name, df in frames.items():
        df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    outputs = {metrics_path.name: {"rows": None, "sha256": _sha16(metrics_path)}}
    for name, df in frames.items():
        outputs[name] = {"rows": int(len(df)), "sha256": _sha16(out_dir / name)}
    manifest = {
        "silence_version": SILENCE_VERSION,
        "generated_at": _now(),
        "config": run["config"],
        "panel_stats": run["stats"],
        "inputs": run["inputs"],
        "models": hyperparams_snapshot(LEARNERS),
        "outputs": outputs,
        "determinism": determinism,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return outputs


def format_report(run: dict) -> str:
    """终端报告：面板 → 每协议指标表 → ΔAUC → 消融 → 稳健性 → M1 裁决。"""
    cfg, st = run["config"], run["stats"]
    p = cfg["panel"]
    lines = [
        f"[info] 面板：样本 {st['n_samples']}｜用户 {st['n_users_used']}"
        f"｜沉默率 {st['silent_rate']:.1%}｜左截断 {st['left_truncated_share']:.1%}"
        f"（h={p['horizon']}d · step={p['step']}d · pre={p['pre_window']}d）",
        f"[info] 协议：{GROUP_PROTOCOL}{cfg['cv']}（按 uid 分组 OOF）+ {TIME_PROTOCOL}"
        f"（训 <{cfg['split_until']} → 测 ≥）｜feature_set={cfg['feature_set']}"
        f"｜seed={cfg['seed']}｜n_boot={cfg['n_boot']}",
    ]
    for key, res in run["protocols"].items():
        lines.append(f"[info] {key}（{res['split']}）评测 {res['n_eval']} 行"
                     f"｜沉默率 {res['pos_rate']:.1%}")
        lines.append(f"       {'scorer':<10}{'AUC':>8}{'PR-AUC':>9}{'Rec@10%':>9}{'Prec@10%':>9}{'Brier':>8}")
        for scorer in SCORERS:
            m = res["scorers"][scorer]
            brier = f"{m['brier']:.3f}" if m["brier"] is not None else "—"
            lines.append(
                f"       {scorer:<10}{m['roc_auc']:>8.3f}{m['pr_auc']:>9.3f}"
                f"{m['recall_top']:>9.3f}{m['precision_top']:>9.3f}{brier:>8}"
            )
        for name, d in res["deltas"].items():
            lines.append(
                f"       ΔAUC {name}: {d['delta']:+.3f}"
                f" [95%CI {d['ci_low']:+.3f}, {d['ci_high']:+.3f}] P(>0)={d['frac_positive']:.2f}"
            )
        ab = res["ablation_trunk_types"]
        lines.append("       消融（主干 → 主干+类型占比）："
                     + "｜".join(f"{l} ΔAUC {ab[l]['delta']:+.3f}" for l in LEARNERS))
        rb = res.get("robust_no_left_trunc")
        if rb is None:
            continue
        if "scorers" in rb:
            lines.append(
                f"       稳健（无左截断子集 n={rb['n_eval']}）："
                f"hgb {rb['scorers']['hgb']['roc_auc']:.3f}"
                f" vs recency {rb['scorers']['heuristic']['roc_auc']:.3f}"
                f"｜ΔAUC {rb['deltas']['hgb_vs_heuristic']['delta']:+.3f}"
            )
        else:
            lines.append(f"       稳健（无左截断子集）：跳过（{rb['skipped']}，n={rb['n_eval']}）")
    m1 = run["m1"]
    parts = []
    for learner in LEARNERS:
        d = m1["scorers"][learner]
        tag = "通过" if d["passed"] else "未通过"
        parts.append(
            f"{learner} ΔAUC={d['delta']:+.3f} 95%CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
            f"（P(>0)={d['frac_positive']:.2f}，{tag}）"
        )
    verdict = "达标" if m1["passed"] else "未达标"
    lines.append(f"[M1] 时间外推 vs recency（{cfg['split_until']} 切分）：" + "；".join(parts) + f" → {verdict}")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M1 沉默预测：公开行为痕上预测未来沉默，并回答能否显著优于 recency 基线")
    ap.add_argument("--timeline", default=str(DEFAULT_TIMELINE), help="L2 时间线 JSONL（默认 data/processed/user_features/timeline.jsonl）")
    ap.add_argument("--index", default=str(DEFAULT_INDEX), help="采集索引 JSONL（默认 data/raw/user_profile/_index.jsonl，取 fetched_at 做右截断）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 data/processed/silence）")
    ap.add_argument("--horizon", type=int, default=30, help="标签窗口天数（默认 30）")
    ap.add_argument("--step", type=int, default=30, help="锚点步长天数（默认 30）")
    ap.add_argument("--pre-window", type=int, default=30, help="预窗天数： (T−pre, T] 需有活动（默认 30）")
    ap.add_argument("--pred-window", type=int, default=180, help="最长特征窗天数，用于左截断标记（默认 180）")
    ap.add_argument("--min-history", type=int, default=30, help="首个锚点距首个事件的最小历史（默认 30）")
    ap.add_argument("--cv", type=int, default=5, help="GroupKFold 折数（默认 5）")
    ap.add_argument("--split-until", default="2025-12-31", help="时间外推切分：训练 ≤ 该日（默认 2025-12-31）")
    ap.add_argument("--top-frac", type=float, default=0.10, help="Top-k%% 指标的名额比例（默认 0.10）")
    ap.add_argument("--seed", type=int, default=7, help="随机种子（默认 7）")
    ap.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="trunk", help="特征集（默认 trunk）")
    ap.add_argument("--no-verify", action="store_true", help="跳过双跑一致性校验")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cfg = PanelConfig(
            horizon=args.horizon, step=args.step, pre_window=args.pre_window,
            pred_window=args.pred_window, min_history=args.min_history,
        )
    except ValueError as exc:
        print(f"[stop] 参数无效：{exc}", file=sys.stderr)
        return 2
    kwargs = dict(
        timeline_path=Path(args.timeline), index_path=Path(args.index), cfg=cfg,
        cv=args.cv, split_until=args.split_until, top_frac=args.top_frac,
        seed=args.seed, feature_set=args.feature_set,
    )
    try:
        run = run_all(**kwargs)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[stop] 运行失败：{exc}", file=sys.stderr)
        return 2

    determinism = {"checked": False}
    if not args.no_verify:
        run_again = run_all(**kwargs)
        same = run_is_deterministic(run, run_again)
        determinism = {"checked": True, "identical": bool(same)}
        if not same:
            print("[stop] 双跑不一致：存在非确定性来源，拒绝落盘", file=sys.stderr)
            return 3

    print(format_report(run))
    out_dir = Path(args.out)
    outputs = write_outputs(run, out_dir, determinism)
    body = " / ".join(
        f"{name} {v['rows'] if v['rows'] is not None else '—'} 行" for name, v in outputs.items()
    )
    print(f"[done] → {_rel(out_dir)}（{body}）")
    if determinism["checked"]:
        print(f"[ok] 双跑一致性：{determinism['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
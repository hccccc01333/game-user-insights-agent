#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 · 沉默预测模型落盘与推理封装（silence_artifact_v1）。

先说人话：
    run_silence.py 回答"这套建模方案值不值得信"（时间外推验收）；
    本模块回答"把它装成生产工件"——在全部合格面板行上重训一个推理模型，
    连同口径快照（特征列 / 面板参数 / 训练元数据 / 输入指纹）一起落盘，
    再提供单用户 / 批量打分接口，供 Agent 工具层（harness/tools）调用。

工件（pickle + 同名 .meta.json）：
    {"model": 已拟合分类器, "meta": 可读元数据}
    训练集指标仅供健全性检查——验收口径永远是 run_silence 的时间外推。

打分口径（与面板同源，不适用不硬打分）：
    · 观测终点 T：显式 as_of_ts > 该用户 fetched_at（有 index 时）> 全局 as_of；
    · 仅对"近期活跃"用户适用：(T−pre, T] 内有 exact 事件；
    · 且历史足够：T − 首个 exact 事件 ≥ min_history 天；
    · 不适用返回 applicable=False + 原因，score=None（口径诚实，供 Agent 判断）。

运行（在仓库根目录）：
    python -m silence_risk.artifact                      # 训练 + 落盘（默认 hgb/trunk_types）
    python -m silence_risk.artifact --learner logreg --feature-set trunk --no-verify

产出（默认 data/processed/silence/）：
    silence_model.pkl        推理工件（仅 uid_hash 相关特征，不含原始行为）
    silence_model.meta.json 口径快照 / 面板统计 / 健全性指标 / 确定性探针
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import SILENCE_VERSION
from .evaluate import metrics
from .features import build_features, feature_columns
from .models import make_classifier
from .panel import DAY, TZ_CN, PanelConfig, build_panel, group_exact_events, load_jsonl, parse_fetched_at
from .run_silence import DEFAULT_INDEX, DEFAULT_OUT, DEFAULT_TIMELINE, _now, _rel, _sha16

ARTIFACT_VERSION = "silence_art_v1"
MODEL_FILENAME = "silence_model.pkl"
DEFAULT_MODEL_PATH = DEFAULT_OUT / MODEL_FILENAME
DEFAULT_LEARNER = "hgb"
DEFAULT_FEATURE_SET = "trunk_types"
PROBE_N = 50  # 确定性探针：前 N 行概率哈希（双跑对拍用）

SCORE_COLUMNS = ("uid_hash", "as_of_ts", "applicable", "reason", "n_pre", "gap_days", "score")
NOTES = (
    "口径：Future Public Inactivity Prediction（锚点 T；(T, T+h] 无 exact 事件即沉默），不叫 churn",
    "工件 = 全量合格面板行重训的推理模型；训练集指标含乐观偏差，非验收口径",
    "验收口径（时间外推 AUC vs recency 基线）见 silence_metrics.json 与 run_silence.py",
    "打分只对近期活跃且历史足够的用户适用；产出仅 uid_hash 与聚合分数，不含原始行为",
)


# ── 训练与落盘 ──────────────────────────────────────────────

def train_artifact(
    *,
    timeline_path: Path,
    index_path: Path,
    cfg: PanelConfig,
    learner: str = DEFAULT_LEARNER,
    seed: int = 7,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> dict:
    """全量面板行上重训推理模型；返回 {"model", "meta", "features", "panel"}（同参同种子逐值一致）。"""
    cols = feature_columns(feature_set)  # 未知名在此显式报错
    timeline_path, index_path = Path(timeline_path), Path(index_path)
    timeline = load_jsonl(timeline_path)
    index = load_jsonl(index_path)
    panel, stats = build_panel(timeline, index, cfg)
    if panel.empty:
        raise ValueError("面板为空：没有满足口径的样本（检查输入路径 / 面板参数）")
    if panel["label"].nunique() < 2:
        raise ValueError("面板标签单一类别，无法训练分类模型")

    features = build_features(panel, timeline, cfg)
    X = features.loc[:, list(cols)]
    y = panel["label"].to_numpy(dtype=int)
    model = make_classifier(learner, seed).fit(X, y)
    sanity = metrics(y, model.predict_proba(X)[:, 1], probabilistic=True, top_frac=0.10)

    fetched = [ts for ts in (parse_fetched_at(r.get("fetched_at")) for r in index) if ts]
    as_of_ts = int(max(fetched)) if fetched else int(panel["anchor_ts"].max())
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "silence_version": SILENCE_VERSION,
        "created_at": _now(),
        "learner": learner,
        "seed": int(seed),
        "feature_set": feature_set,
        "columns": list(cols),
        "panel_cfg": cfg.snapshot(),
        "panel_stats": stats,
        "as_of_ts": as_of_ts,
        "n_train": int(len(panel)),
        "n_users": int(panel["uid_hash"].nunique()),
        "pos_rate": float(y.mean()),
        "anchor_range_ts": [int(panel["anchor_ts"].min()), int(panel["anchor_ts"].max())],
        "train_sanity": {**sanity, "note": "训练集指标（含乐观偏差）：仅健全性检查，非验收口径"},
        "inputs": {
            "timeline": {"path": _rel(timeline_path), "n_records": len(timeline),
                         "sha256_16": _sha16(timeline_path)},
            "index": {"path": _rel(index_path), "n_records": len(index),
                      "sha256_16": _sha16(index_path)},
        },
        "notes": list(NOTES),
    }
    return {"model": model, "meta": meta, "features": features, "panel": panel}


def probe_hash(features: pd.DataFrame, meta: dict, model, n: int = PROBE_N) -> str:
    """确定性探针：前 n 行概率（9 位小数）的哈希——同参双跑必须一致。"""
    X = features.head(n).loc[:, list(meta["columns"])]
    prob = np.round(np.asarray(model.predict_proba(X)[:, 1], dtype="<f8"), 9)
    return hashlib.sha256(prob.tobytes()).hexdigest()[:16]


def save_artifact(model, meta: dict, path: Path = DEFAULT_MODEL_PATH) -> dict:
    """pickle + 同名 .meta.json 落盘；返回文件信息（供 _manifest 用）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"model": model, "meta": meta}, f)
    meta_path = path.with_name(path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {
        "model": {"path": _rel(path), "bytes": path.stat().st_size},
        "meta": {"path": _rel(meta_path), "bytes": meta_path.stat().st_size},
    }


def load_artifact(path: Path = DEFAULT_MODEL_PATH) -> dict:
    """读工件（缺文件显式报错，避免下游拿到空模型）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"模型工件不存在：{path}（先运行 python -m silence_risk.artifact 训练落盘）"
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    meta = payload.get("meta") or {}
    if payload.get("model") is None or not meta.get("columns"):
        raise ValueError(f"工件损坏：缺少 model / meta.columns（{path}）")
    return payload


def _load_fetched(index_path: Path) -> dict[str, int]:
    """index → uid_hash 到 fetched_at（观测终点）的映射。"""
    out: dict[str, int] = {}
    for row in load_jsonl(index_path):
        uid = row.get("uid_hash")
        ts = parse_fetched_at(row.get("fetched_at"))
        if uid and ts:
            out[uid] = ts
    return out


# ── 推理封装 ────────────────────────────────────────────────

class SilencePredictor:
    """已训练工件的推理封装：批量 / 单用户打分（口径与面板同源）。

    fetched_at_by_uid：可选，用户观测终点表；缺省用全局 as_of_ts。
    显式传入 as_of_ts 时对全体用户统一取值（覆盖 fetched_at 表）。
    """

    def __init__(self, model, meta: dict, fetched_at_by_uid: dict | None = None):
        self.model = model
        self.meta = dict(meta)
        self.cfg = PanelConfig(**self.meta["panel_cfg"])
        self.columns = tuple(self.meta["columns"])
        self._fetched = dict(fetched_at_by_uid or {})

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH, index_path: Path | None = None) -> "SilencePredictor":
        """读工件；index_path 可选（给了才有逐用户 fetched_at 观测终点）。"""
        payload = load_artifact(path)
        fetched = _load_fetched(Path(index_path)) if index_path else {}
        return cls(payload["model"], payload["meta"], fetched)

    def default_as_of_ts(self) -> int:
        return int(self.meta["as_of_ts"])

    def _as_of_for(self, uid: str, as_of_ts: int | None) -> int:
        if as_of_ts is not None:
            return int(as_of_ts)
        return int(self._fetched.get(uid, self.meta["as_of_ts"]))

    def score_records(
        self, records: list[dict], as_of_ts: int | None = None, uids: list[str] | None = None
    ) -> pd.DataFrame:
        """批量打分：uids 缺省 = records 里全部用户（按 uid 排序）。

        固定列 SCORE_COLUMNS；不适用行 score=None 且给出 reason
        （no_exact_events / history_too_short / no_recent_activity）。
        """
        by_uid = group_exact_events(records)
        targets = sorted(by_uid) if uids is None else list(uids)
        pre_s, hist_s = self.cfg.pre_window * DAY, self.cfg.min_history * DAY
        rows: list[dict] = []
        pending: list[int] = []
        for uid in targets:
            t = self._as_of_for(str(uid), as_of_ts)
            events = by_uid.get(uid, [])
            reason, n_pre = None, 0
            if not events:
                reason = "no_exact_events"
            elif t - int(events[0]["event_ts"]) < hist_s:
                reason = "history_too_short"
            else:
                ts_arr = np.asarray([e["event_ts"] for e in events], dtype=np.int64)
                lo = int(np.searchsorted(ts_arr, t - pre_s, "right"))
                hi = int(np.searchsorted(ts_arr, t, "right"))
                n_pre = hi - lo
                if n_pre == 0:
                    reason = "no_recent_activity"
            rows.append({"uid_hash": str(uid), "as_of_ts": t, "applicable": reason is None,
                         "reason": reason, "n_pre": int(n_pre), "gap_days": None, "score": None})
            if reason is None:
                pending.append(len(rows) - 1)

        if pending:  # 仅对适用行构造特征（复用面板特征单一真源）
            panel = pd.DataFrame(
                [{"uid_hash": rows[i]["uid_hash"], "anchor_ts": rows[i]["as_of_ts"]} for i in pending]
            )
            feat = build_features(panel, records, self.cfg)
            prob = self.model.predict_proba(feat.loc[:, list(self.columns)])[:, 1]
            for pos, i in enumerate(pending):
                rows[i]["gap_days"] = round(float(feat["gap_days"].iloc[pos]), 6)
                rows[i]["score"] = round(float(prob[pos]), 6)
        return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))

    def score_user(self, records: list[dict], uid_hash: str, as_of_ts: int | None = None) -> dict:
        """单用户打分（JSON 可序列化 dict）；只对该用户的事件建特征。"""
        recs = [r for r in records if r.get("uid_hash") == uid_hash]
        row = self.score_records(recs, as_of_ts=as_of_ts, uids=[uid_hash]).iloc[0]
        return {
            "uid_hash": str(row["uid_hash"]),
            "as_of_ts": int(row["as_of_ts"]),
            "applicable": bool(row["applicable"]),
            "reason": None if row["reason"] is None else str(row["reason"]),
            "n_pre": int(row["n_pre"]),
            "gap_days": None if row["gap_days"] is None else float(row["gap_days"]),
            "score": None if row["score"] is None else float(row["score"]),
        }


# ── 报告与 CLI ──────────────────────────────────────────────

def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), TZ_CN).strftime("%Y-%m-%d")


def format_report(run: dict) -> str:
    meta, san = run["meta"], run["meta"]["train_sanity"]
    a0, a1 = meta["anchor_range_ts"]
    return "\n".join([
        f"[info] 训练集：样本 {meta['n_train']}｜用户 {meta['n_users']}"
        f"｜沉默率 {meta['pos_rate']:.1%}｜锚点 {_fmt_ts(a0)} → {_fmt_ts(a1)}",
        f"[info] learner={meta['learner']}｜feature_set={meta['feature_set']}"
        f"（{len(meta['columns'])} 列）｜seed={meta['seed']}｜as_of={_fmt_ts(meta['as_of_ts'])}",
        f"[info] 训练集健全性 AUC={san['roc_auc']:.3f}｜PR-AUC={san['pr_auc']:.3f}"
        "（含乐观偏差，非验收口径；验收看 run_silence 时间外推）",
    ])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M1 沉默预测：全量重训推理工件并落盘（口径快照 + 确定性探针）")
    ap.add_argument("--timeline", default=str(DEFAULT_TIMELINE), help="L2 时间线 JSONL（默认 data/processed/user_features/timeline.jsonl）")
    ap.add_argument("--index", default=str(DEFAULT_INDEX), help="采集索引 JSONL（默认 data/raw/user_profile/_index.jsonl，取 fetched_at 做观测终点）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="产出目录（默认 data/processed/silence）")
    ap.add_argument("--learner", choices=("logreg", "hgb"), default=DEFAULT_LEARNER, help="学习器（默认 hgb）")
    ap.add_argument("--feature-set", choices=("trunk", "trunk_types"), default=DEFAULT_FEATURE_SET, help="特征集（默认 trunk_types）")
    ap.add_argument("--seed", type=int, default=7, help="随机种子（默认 7）")
    ap.add_argument("--horizon", type=int, default=30, help="标签窗口天数（默认 30）")
    ap.add_argument("--step", type=int, default=30, help="锚点步长天数（默认 30）")
    ap.add_argument("--pre-window", type=int, default=30, help="预窗天数（默认 30）")
    ap.add_argument("--pred-window", type=int, default=180, help="最长特征窗天数（默认 180）")
    ap.add_argument("--min-history", type=int, default=30, help="首个锚点距首个事件的最小历史（默认 30）")
    ap.add_argument("--no-verify", action="store_true", help="跳过双跑确定性校验")
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
        learner=args.learner, seed=args.seed, feature_set=args.feature_set,
    )
    try:
        run = train_artifact(**kwargs)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[stop] 训练失败：{exc}", file=sys.stderr)
        return 2

    probe = probe_hash(run["features"], run["meta"], run["model"])
    determinism = {"checked": False}
    if not args.no_verify:
        again = train_artifact(**kwargs)
        same = probe_hash(again["features"], again["meta"], again["model"]) == probe
        determinism = {"checked": True, "identical": bool(same), "probe_sha16": probe}
        if not same:
            print("[stop] 双跑不一致：存在非确定性来源，拒绝落盘", file=sys.stderr)
            return 3
    run["meta"]["determinism"] = determinism
    run["meta"]["probe_sha16"] = probe

    print(format_report(run))
    files = save_artifact(run["model"], run["meta"], Path(args.out) / MODEL_FILENAME)
    print(f"[done] → {files['model']['path']}（{files['model']['bytes']} B）+ {files['meta']['path']}")
    if determinism["checked"]:
        print(f"[ok] 双跑确定性：{determinism['identical']}（探针 {probe}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
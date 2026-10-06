# -*- coding: utf-8 -*-
"""M1 沉默预测 · 冒烟与一致性测试（无外部依赖，纯断言）。

运行（在仓库根目录，任选其一）：
    python -m silence_risk.tests
    pytest silence_risk/tests.py

覆盖：
    1. 面板手算：锚点推进 / (T−pre,T] 左开右闭 / (T,T+h] 右闭 / 右截断 / 缺 index 用户
    2. 特征手算：n7..n180 与 gap_days、180d 类型占比（单一真源列序）
    3. 指标端点：完美分离 AUC=1 / 常数分 0.5 与 Brier 0.25 / Top-k 命中 / 校准表
    4. 协议完整性：group_cv 折内 uid 不跨 + OOF 全覆盖；time_extrap 切分不重叠且并集完整
    5. 无泄漏：特征列白名单；未来事件不影响既往锚点的特征
    6. 确定性：同参同种子端到端双跑一致；bootstrap 同种子可复现、换种子变；换 seed 模型分数变
    7. 边界：非法面板参数 / 特征集 / cv / top_frac / 切分两侧为空 / 单一类别 均显式报错
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import (
    CALIBRATION_COLUMNS, brier, calibration_table, cluster_bootstrap_auc_diff,
    metrics, precision_at_top, pr_auc, recall_at_top, roc_auc,
)
from .features import ALL_FEATURE_COLUMNS, FEATURE_SETS, build_features, feature_columns
from .models import make_classifier
from .panel import PanelConfig, build_panel
from .run_silence import group_cv_oof, run_all, run_is_deterministic

TZ_CN = timezone(timedelta(hours=8))
BASE = 1735660800  # 2025-01-01T00:00:00+08:00
D = 86400


# ── 构造工具 ────────────────────────────────────────────────

def _ev(uid: str, day: int, etype: str = "post") -> dict:
    return {
        "event_id": f"ev_{uid}_{day}",
        "uid_hash": uid,
        "event_type": etype,
        "event_ts": BASE + day * D,
        "time_kind": "exact",
    }


def _idx(uid: str, day: int) -> dict:
    return {"uid_hash": uid, "fetched_at": datetime.fromtimestamp(BASE + day * D, TZ_CN).isoformat()}


def _day(ts: int) -> int:
    return int((ts - BASE) / D)


def _handcrafted() -> tuple[list[dict], list[dict]]:
    """手算样本：A（结论 y0/y1 与左截断）/ C（T+h 右闭）/ E（T−pre 左开）/ 边界用户。"""
    timeline = [
        _ev("h_a", 0, "review"), _ev("h_a", 10, "post"), _ev("h_a", 35, "post"), _ev("h_a", 40, "badge"),
        _ev("h_c", 0, "review"), _ev("h_c", 10), _ev("h_c", 60),
        _ev("h_e", 0), _ev("h_e", 30),
        _ev("h_f", 0),            # fetched 太早（T+h > fetched）→ 无锚点
        _ev("h_g", 0),            # 有事件但缺 index → 不计入面板
        _ev("h_h", 0), _ev("h_h", 10),  # fetched 恰为 T+h → 边界保留
    ]
    index = [_idx("h_a", 200), _idx("h_c", 200), _idx("h_e", 200), _idx("h_f", 20), _idx("h_h", 60)]
    return timeline, index


@lru_cache(maxsize=1)
def _syn_files() -> tuple[Path, Path]:
    """合成时间线（deterministic）：活跃用户周期性活动 → y0 为主；静默用户中途停止 → y1。

    节奏：活跃用户每 40 天一个 5 天爆发（0..440）；静默用户爆发到 320 后再补一次 370，
    使 y1 锚点恰好落在时间外推的**训练侧**（T=330）与**测试侧**（T=390）。
    """
    etypes = ("post", "review", "badge", "wishlist", "follow_user")
    timeline: list[dict] = []
    index: list[dict] = []
    for i in range(24):
        uid = f"h_act{i:02d}"
        timeline += [_ev(uid, b + d, etypes[(b // 40) % 5]) for b in range(0, 441, 40) for d in range(5)]
        index.append(_idx(uid, 470))
    for i in range(12):
        uid = f"h_sil{i:02d}"
        bases = list(range(0, 321, 40)) + [370]
        timeline += [_ev(uid, b + d, etypes[(b // 40) % 5]) for b in bases for d in range(5)]
        index.append(_idx(uid, 470))
    tmp = Path(tempfile.mkdtemp(prefix="silence_test_"))
    for name, records in (("timeline.jsonl", timeline), ("_index.jsonl", index)):
        (tmp / name).write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in records) + "\n", encoding="utf-8"
        )
    return tmp / "timeline.jsonl", tmp / "_index.jsonl"


SYN_KW = dict(cv=3, split_until="2025-12-31", top_frac=0.10, seed=7, feature_set="trunk", n_boot=100)


@lru_cache(maxsize=1)
def _run() -> dict:
    timeline, index = _syn_files()
    return run_all(timeline_path=timeline, index_path=index, cfg=PanelConfig(), **SYN_KW)


# ── 1. 面板手算 ─────────────────────────────────────────────

def test_panel_handcrafted():
    timeline, index = _handcrafted()
    panel, stats = build_panel(timeline, index, PanelConfig())

    got = [
        (r.uid_hash, _day(r.anchor_ts), int(r.label), bool(r.left_truncated))
        for r in panel.itertuples(index=False)
    ]
    assert got == [
        ("h_a", 30, 0, True), ("h_c", 30, 0, True), ("h_e", 30, 1, True), ("h_h", 30, 1, True),
        ("h_a", 60, 1, True), ("h_c", 60, 1, True),
    ], f"面板行与手算不一致：{got}"
    assert list(panel.columns) == ["uid_hash", "anchor_ts", "label", "first_ts", "fetched_at_ts", "left_truncated"]

    assert stats["n_users_with_exact_events"] == 6
    assert stats["n_users_used"] == 4 and stats["n_users_missing_index"] == 1
    assert stats["n_samples"] == 6
    assert abs(stats["silent_rate"] - 4 / 6) < 1e-12
    assert stats["left_truncated_share"] == 1.0
    assert stats["anchors_by_year"] == {"2025": 6}
    assert stats["anchors_skipped_no_pre_activity"] >= 1

    # unknown 时间的事件不进时间序列
    mixed = timeline + [{"uid_hash": "h_a", "event_type": "post", "event_ts": None,
                         "time_kind": "unknown", "event_id": "ev_unknown"}]
    panel2, _ = build_panel(mixed, index, PanelConfig())
    assert panel2.equals(panel), "unknown 事件影响了面板"


# ── 2. 特征手算 ─────────────────────────────────────────────

def test_features_handcrafted():
    timeline, index = _handcrafted()
    panel, _ = build_panel(timeline, index, PanelConfig())
    feat = build_features(panel, timeline, PanelConfig())
    assert list(feat.columns) == ["uid_hash", "anchor_ts", *ALL_FEATURE_COLUMNS]

    def row(uid: str, day: int) -> pd.Series:
        m = feat[(feat["uid_hash"] == uid) & (feat["anchor_ts"] == BASE + day * D)]
        assert len(m) == 1, f"({uid}, {day}) 特征行不唯一"
        return m.iloc[0]

    r60 = row("h_a", 60)
    # 窗口均为左开右闭 (T−w, T]：n60 不含 D0，n90/n180 含 D0
    assert (r60["n7"], r60["n14"], r60["n30"], r60["n60"], r60["n90"], r60["n180"]) == (0, 0, 2, 3, 4, 4)
    assert r60["gap_days"] == 20.0
    assert r60["t180_review_share"] == 0.25 and r60["t180_post_share"] == 0.5
    assert r60["t180_badge_share"] == 0.25
    assert r60["t180_wishlist_share"] == 0.0 and r60["t180_follow_user_share"] == 0.0

    r30 = row("h_a", 30)
    assert (r30["n30"], r30["n180"], r30["gap_days"]) == (1, 2, 20.0)
    assert list(FEATURE_SETS["trunk_types"]) == list(ALL_FEATURE_COLUMNS)


# ── 3. 指标端点 ─────────────────────────────────────────────

def test_metrics_endpoints():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, s) == 1.0 and pr_auc(y, s) == 1.0
    assert recall_at_top(y, s, 0.5) == 1.0 and precision_at_top(y, s, 0.5) == 1.0

    const = np.full(4, 0.5)
    assert abs(roc_auc(y, const) - 0.5) < 1e-12
    assert abs(brier(y, const) - 0.25) < 1e-12

    m = metrics(y, s, probabilistic=True, top_frac=0.5)
    assert m["brier"] is not None and m["n"] == 4 and m["pos_rate"] == 0.5
    assert metrics(y, s, probabilistic=False, top_frac=0.5)["brier"] is None

    cal = calibration_table(y, s, bins=4)
    assert list(cal.columns) == list(CALIBRATION_COLUMNS)
    assert int(cal["n"].sum()) == 4 and ((cal["observed_rate"] >= 0).all())
    cal_const = calibration_table(y, const, bins=4)
    assert int(cal_const["n"].sum()) == 4 and len(cal_const) == 1, "常数分应落单箱"


# ── 4. 协议完整性 ───────────────────────────────────────────

def test_protocol_integrity():
    run = _run()
    panel, cfg = run["panel"], run["config"]
    y = panel["label"].to_numpy()
    g = panel["uid_hash"].to_numpy()
    X = run["features"].loc[:, list(FEATURE_SETS["trunk"])]

    # group_cv：同一 uid 的锚点必须在同一折（不跨折泄漏）
    oof, folds = group_cv_oof(X, y, g, 3, "logreg", 7)
    assert (folds >= 0).all(), "OOF 覆盖不完整"
    assert pd.DataFrame({"uid": g, "fold": folds}).groupby("uid")["fold"].nunique().max() == 1, \
        "同一用户的锚点被分进了不同折"

    pred = run["predictions"]
    key_cv = f"group_cv{cfg['cv']}"
    assert set(pred["protocol"]) == {key_cv, "time_extrap"}
    pcv = pred[pred["protocol"] == key_cv]
    assert len(pcv) == len(panel) and pcv["score_logreg"].notna().all()
    assert (pcv["split"] == "oof").all()
    assert np.allclose(pcv.sort_values(["anchor_ts", "uid_hash"])["label"].to_numpy(), y, atol=0), \
        "预测长表与面板行不对齐"

    # time_extrap：切分不重叠、并集完整、测试行都 ≥ 边界
    split_ts = cfg["split_ts"]
    te = panel["anchor_ts"].to_numpy() >= split_ts
    pte = pred[pred["protocol"] == "time_extrap"]
    assert (pte["anchor_ts"] >= split_ts).all() and len(pte) == int(te.sum())
    assert pte["score_heuristic"].notna().all() and pte["score_hgb"].notna().all()
    for res in run["protocols"].values():
        assert res["n_eval"] == int(res["scorers"]["hgb"]["n"]) > 0


# ── 5. 无泄漏 ───────────────────────────────────────────────

def test_no_leakage():
    timeline, index = _handcrafted()
    panel, _ = build_panel(timeline, index, PanelConfig())
    feat = build_features(panel, timeline, PanelConfig())
    for forbidden in ("label", "fetched_at_ts", "first_ts", "silent_rate"):
        assert forbidden not in feat.columns, f"特征帧混入非特征列 {forbidden}"
    assert set(feature_columns("trunk")) <= set(feat.columns)

    # 未来事件不得影响既往锚点的特征（特征只允许看 T 及之前；新增事件必须晚于全部面板锚点）
    future = timeline + [_ev("h_a", 180), _ev("h_a", 500), _ev("h_c", 181)]
    feat_future = build_features(panel, future, PanelConfig())
    key = ["uid_hash", "anchor_ts"]
    merged = feat.merge(feat_future, on=key, suffixes=("_old", "_new"))
    assert len(merged) == len(feat)
    for col in ALL_FEATURE_COLUMNS:
        assert np.allclose(merged[f"{col}_old"].to_numpy(float), merged[f"{col}_new"].to_numpy(float)), \
            f"未来事件改变了特征列 {col}（潜在泄漏）"


# ── 6. 确定性 ───────────────────────────────────────────────

def test_determinism():
    first = _run()
    second = run_all(timeline_path=_syn_files()[0], index_path=_syn_files()[1], cfg=PanelConfig(), **SYN_KW)
    assert run_is_deterministic(first, second), "同参同种子双跑不一致"

    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(30), np.ones(30)].astype(int)
    sa, sb = y + rng.normal(0, 0.5, 60), rng.normal(0, 0.5, 60)
    groups = [f"u{i}" for i in range(60)]
    d1 = cluster_bootstrap_auc_diff(y, sa, sb, groups, n_boot=100, seed=7)
    d2 = cluster_bootstrap_auc_diff(y, sa, sb, groups, n_boot=100, seed=7)
    d3 = cluster_bootstrap_auc_diff(y, sa, sb, groups, n_boot=100, seed=8)
    assert d1 == d2, "bootstrap 同种子不可复现"
    assert d1 != d3 and d1["ci_low"] <= d1["delta"] <= d1["ci_high"]

    # 类型占比特征非恒常（消融有意义）+ 消融结构完整（两学习器 × 两协议）
    share_cols = [c for c in first["features"].columns if c.startswith("t180_")]
    assert (first["features"].loc[:, share_cols].nunique() > 1).sum() >= 2, "类型占比恒常，消融无意义"
    for res in first["protocols"].values():
        ab = res["ablation_trunk_types"]
        assert set(ab) == {"logreg", "hgb"} and all("delta" in ab[l] for l in ab)


# ── 7. 边界与显式报错 ───────────────────────────────────────

def test_boundaries():
    def expect_error(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except ValueError:
            return
        raise AssertionError(f"{fn} 未按预期抛 ValueError")

    expect_error(PanelConfig, horizon=0)
    expect_error(PanelConfig, step=-30)
    expect_error(PanelConfig, pre_window=0)
    expect_error(feature_columns, "nope")
    expect_error(make_classifier, "svm", 1)

    timeline, index = _syn_files()
    base_kw = dict(timeline_path=timeline, index_path=index, cfg=PanelConfig(), **SYN_KW)
    expect_error(run_all, **{**base_kw, "feature_set": "nope"})
    expect_error(run_all, **{**base_kw, "cv": 1})
    expect_error(run_all, **{**base_kw, "top_frac": 0.0})
    expect_error(run_all, **{**base_kw, "split_until": "2024-01-01"})  # 训练侧为空
    expect_error(run_all, **{**base_kw, "split_until": "2026-06-01"})  # 测试侧为空
    expect_error(run_all, **{**base_kw, "n_boot": 0})

    expect_error(roc_auc, [0, 0, 0], [0.1, 0.2, 0.3])  # 单一类别
    expect_error(metrics, [0, 0], [0.1, 0.2], **{"probabilistic": True})
    expect_error(cluster_bootstrap_auc_diff, [0, 0, 0], [1, 0, 1], [0, 1, 0], ["a", "b", "a"], 10, 1)
    expect_error(calibration_table, [], [], 10)


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
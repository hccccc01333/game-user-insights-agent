# -*- coding: utf-8 -*-
"""S5 报告 · 冒烟与一致性测试（无外部依赖，纯断言）。

运行（在仓库根目录，任选其一）：
    python -m report.tests
    pytest report/tests.py

覆盖：
    1. 确定性：同输入双跑 HTML / 输入快照 / 统计值全等；两次落盘逐字节一致
    2. 自包含：无 script / 无 http(s) / 无外链；关键数字与源产出逐一对上
    3. 渲染忠实：篡改输入统计后 HTML 必须变化（证明读的是源而非常量）
    4. 章节结构：manifest.sections 与 HTML 中 id 一一对应
    5. 真实轨缺失：临时根目录（仅合成轨）自动跳过 real-track，报告仍生成
    6. 合成轨缺失：MissingInputsError 且带重跑指引（simulator / harness 命令）
    7. 隐私扫描：无盘符绝对路径 / uid_hash / h_ 前缀 / ev_ 前缀
    8. manifest：字段完整；outputs.index.html 的 bytes / sha256 与文件一致；
       时间戳只进 manifest（HTML 内不含 generated_at）
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from . import REPORT_VERSION
from .build_report import (
    PROJECT_ROOT, SYNTH_INPUTS, MissingInputsError, collect, render,
    run_is_deterministic, run_report, sections_present, write_outputs,
)


@lru_cache(maxsize=1)
def _base() -> dict:
    """真实仓库根目录跑一遍（多数用例的共享输入）。"""
    return run_report()


def _copy_synth(root: Path) -> None:
    """把合成轨 7 份产出复制进临时根目录（不含 data/processed 与 data/raw）。"""
    for rel in SYNTH_INPUTS:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / rel, dst)


# ── 1. 确定性 ───────────────────────────────────────────────

def test_determinism():
    a = run_report()
    b = run_report()
    assert run_is_deterministic(a, b), "同输入双跑不一致"
    assert a["html"] == b["html"], "HTML 非逐字节一致"


def test_write_byte_identical():
    a, b = _base(), run_report()
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        m1 = write_outputs(a, Path(t1), {"checked": True, "identical": True})
        m2 = write_outputs(b, Path(t2), {"checked": True, "identical": True})
        f1 = (Path(t1) / "index.html").read_bytes()
        f2 = (Path(t2) / "index.html").read_bytes()
        assert f1 == f2, "两次落盘的 HTML 不一致"
        m1.pop("generated_at")
        m2.pop("generated_at")
        assert m1 == m2, "两次落盘的 manifest（除 generated_at）不一致"


# ── 2. 自包含 + 数字对源 ────────────────────────────────────

def test_self_contained():
    text = _base()["html"].lower()
    assert "<script" not in text, "出现脚本"
    assert "http://" not in text and "https://" not in text, "出现外链"
    assert "src=" not in text, "出现外部资源引用"


def test_numbers_match_sources():
    text = _base()["html"]
    # S1：召回 / 推荐净奖励按源 manifest 现场格式化后必须出现
    s1 = json.loads((PROJECT_ROOT / SYNTH_INPUTS[0]).read_text(encoding="utf-8"))
    eff = s1["effects_summary"]
    assert f"{eff['recall']['mean_reward']:+.2f}" in text, "S1 召回净奖励未对上"
    assert f"{eff['rec']['mean_reward']:+.2f}" in text, "S1 推荐净奖励未对上"
    # S2：model_hgb / oracle_struct @k20 的比值（94.9%）
    s2 = json.loads((PROJECT_ROOT / SYNTH_INPUTS[1]).read_text(encoding="utf-8"))
    ref = s2["results"]["rct"]["policy_at_reference_k"]["recall"]
    assert f"{ref['model_hgb'] / ref['oracle_struct'] * 100:.1f}%" in text, "S2 策略比值未对上"
    # S3：linucb 审计 ÷ oracle（80.4%）
    with (PROJECT_ROOT / SYNTH_INPUTS[3]).open("r", encoding="utf-8-sig") as f:
        rows = [ln.split(",") for ln in f.read().splitlines()[1:] if ln.strip()]
    linucb = next(r for r in rows if r[0] == "linucb")
    assert f"{float(linucb[16]) * 100:.1f}%" in text, "S3 审计得分未对上"
    # S4：圈人 518 / 否决率 2.5%
    trace = [json.loads(ln) for ln in (PROJECT_ROOT / SYNTH_INPUTS[6])
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    cohort = next(ln for ln in trace
                  if ln["mode"] == "with_critic" and ln["decision"]["tool"] == "locate_cohort")
    assert str(cohort["observation"]["size"]) in text, "S4 圈人规模未对上"
    s4 = json.loads((PROJECT_ROOT / SYNTH_INPUTS[5]).read_text(encoding="utf-8"))
    veto_rate = s4["modes"]["with_critic"]["online"]["veto_rate"]
    assert f"{veto_rate * 100:.1f}%" in text, "S4 否决率未对上"
    # 真实轨：时间线事件数（千分位）
    l2 = json.loads((PROJECT_ROOT / "data/processed/user_features/_manifest.json")
                    .read_text(encoding="utf-8"))
    assert f"{l2['timeline']['events']:,}" in text, "L2 事件数未对上"


def test_render_reflects_stats():
    run = _base()
    stats = json.loads(json.dumps(run["stats"]))       # 深拷贝，避免污染缓存
    stats["s1"]["arms"]["recall"]["reward"] = 9.99
    tampered = render(stats, run["sources"])
    assert tampered != run["html"], "篡改输入统计后 HTML 未变化"
    assert "+9.99" in tampered, "篡改后的数字未进入 HTML"


# ── 3. 章节结构 ─────────────────────────────────────────────

def test_sections_present():
    run = _base()
    secs = sections_present(run["stats"])
    assert secs == ["header", "story", "real-track", "s1", "s2", "s3", "s4", "limits", "footer"]
    for sid in secs:
        assert f'id="{sid}"' in run["html"], f"缺少章节 {sid}"


# ── 4. 真实轨缺失 → 自动跳过 ────────────────────────────────

def test_missing_real_track_skips():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _copy_synth(root)
        run = run_report(root)
        assert run["stats"]["real"] is None, "临时根目录不应识别出真实轨"
        assert 'id="real-track"' not in run["html"], "真实轨缺失时应跳过该章节"
        assert "real-track" not in sections_present(run["stats"])
        assert "80.4%" in run["html"], "合成轨章节不应受影响"
        assert "9,521" not in run["html"], "真实轨数字不应出现"


# ── 5. 合成轨缺失 → 报错并给重跑指引 ────────────────────────

def test_missing_synth_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            collect(Path(tmp))
        except MissingInputsError as exc:
            msg = str(exc)
            assert "simulator.run_sim" in msg and "harness.run_agent" in msg, "缺少重跑指引"
        else:
            raise AssertionError("合成轨缺失应抛 MissingInputsError")


# ── 6. 隐私扫描 ─────────────────────────────────────────────

def test_privacy_scan():
    run = _base()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = write_outputs(run, Path(tmp), {"checked": False})
    blob = run["html"] + "\n" + json.dumps(manifest, ensure_ascii=False)
    assert not re.search(r"[A-Za-z]:[\\/]", blob), "出现盘符绝对路径"
    assert "uid_hash" not in blob, "出现 uid_hash 字段"
    assert not re.search(r"h_[0-9a-f]{14}", blob), "出现用户哈希文件名"
    assert not re.search(r"ev_[0-9a-f]{16}", blob), "出现事件哈希 id"


# ── 7. manifest 字段与指纹 ──────────────────────────────────

def test_manifest_fields():
    run = _base()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        manifest = write_outputs(run, out, {"checked": True, "identical": True})
        index = out / "index.html"
        assert index.exists() and (out / "_manifest.json").exists(), "产出文件缺失"
        assert manifest["report_version"] == REPORT_VERSION
        assert manifest["generated_at"] not in run["html"], "HTML 不应含时间戳"
        assert manifest["sections"] == sections_present(run["stats"])
        assert set(manifest["sources"]) == set(run["sources"]), "输入清单不一致"
        for rel, src in manifest["sources"].items():
            assert re.fullmatch(r"[0-9a-f]{16}", src["sha256"]), f"{rel} 指纹格式异常"
            disk = hashlib.sha256((PROJECT_ROOT / rel).read_bytes()).hexdigest()[:16]
            assert src["sha256"] == disk, f"{rel} 指纹与磁盘不一致"
        info = manifest["outputs"]["index.html"]
        data = index.read_bytes()
        assert info["bytes"] == len(data), "bytes 与文件不一致"
        assert info["sha256"] == hashlib.sha256(data).hexdigest()[:16], "sha256 与文件不一致"
        assert manifest["determinism"] == {"checked": True, "identical": True}


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
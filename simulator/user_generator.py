# -*- coding: utf-8 -*-
"""S1 模拟器 · 用户生成器（合成人口）。

先说人话：
    模拟器要有人。本模块造一批"看起来像真实社区成员"的合成用户，每个用户带
    四件可观测状态（干预效应的异质性来源）＋ 一件潜在节奏：
        act_30d               近 30 天事件数（观测）
        silence_days          沉默时长（观测）
        interest_concentration 兴趣集中度（观测，genre_top1_ratio 对应物）
        tenure_days           账号资历（观测）
        a_daily               潜在日活跃概率（驱动后续 rollout 的节奏参数）

两条路径（互为兜底）：
    fit   ：本地存在 L2 特征表（data/processed/user_features/features.csv）时，
            对上述四列做"分位数秩耦合抽样"，把合成人口的边缘分布对齐到真实
            观测形状；act_30d>0 的节奏由事件数反推 a ≈ act_30d / (30×活跃日
            平均条数)，act_30d=0 的用户落在很低但非零的"偶发冒泡"节奏带。
    default：没有本地数据（公开仓离线场景）用 params.pop_mix 内置默认分布，
            保证不依赖任何数据文件也能完整跑通。

诚实标注（v0 近似，README 同步声明）：
    · 逐列（边缘分布）对齐 + 同秩耦合，不是完整联合分布拟合；
    · act_30d 与 silence_days 共用同一分位秩 u（保留"越活跃越不沉默"的单调
      负相关），集中度与资历用独立秩——这是 v0 的已知简化；
    · 只"照分布重造"，不复制任何一条真实用户记录；产出 uid 一律 sim_ 前缀，
      与真实 uid_hash 永不混淆。
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .params import SIM, SimParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "user_features" / "features.csv"

# fit 路径所需的 L2 特征列（口径见 L2_features/feature_dict.py）
FIT_COLUMNS = ("act_30d", "recency_days", "genre_top1_ratio", "tenure_days")

# 随机流编号：0=人口生成，1=行为推进，2=响应噪声（env 侧使用）
STREAM_POP = 0


@dataclass
class SimUser:
    """一个合成用户的完整初始状态（观测四件 + 潜在节奏一件）。"""

    uid: str
    a_daily: float                # 潜在日活跃概率（rollout 的唯一节奏源）
    silence_days: float           # 干预时点沉默时长（天）
    act_30d: float                # 近 30 天事件数（效应异质性输入）
    interest_concentration: float # 兴趣集中度 ∈ [0,1]
    tenure_days: float            # 账号资历（天）


@dataclass
class Population:
    users: list[SimUser]
    meta: dict                    # 供 manifest：路径 / 指纹 / 实收分布统计


# ── 工具 ────────────────────────────────────────────────────

def _rank_sample(sorted_vals: np.ndarray, u: np.ndarray) -> np.ndarray:
    """按分位秩 u∈[0,1) 从排好序的经验分布取值（bootstrap 式，不插值）。

    这样取到的值严格来自真实观测的取值集合，不会产生"观测里没有的数"。
    """
    idx = np.clip(np.floor(u * len(sorted_vals)).astype(int), 0, len(sorted_vals) - 1)
    return sorted_vals[idx].astype(float)


def _fingerprint(path: Path) -> str:
    """输入指纹：与 L2 同口径（文件名:大小:mtime → sha256 前 16 位）。"""
    st = path.stat()
    h = hashlib.sha256(f"{path.name}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _rel(path: Path) -> str:
    """输出里不落本地绝对路径：能相对仓库根就相对，否则只留文件名。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _derive_a_daily(act30: np.ndarray, rng: np.random.Generator, p: SimParams) -> np.ndarray:
    """由 act_30d 反推潜在日活跃概率（口径：a ≈ act30 / (30 × 活跃日平均条数)）。

    act_30d=0 的用户无法反推（真实中他们是"极低频冒泡"群体），
    落在 quiet_a_band 的低节奏带内随机取值。
    """
    lo, hi = p.quiet_a_band
    quiet = rng.uniform(lo, hi, size=act30.shape)
    based = act30 / (30.0 * p.count_mean_active) * rng.uniform(*p.act_a_jitter, size=act30.shape)
    based = np.maximum(based, p.a_min_if_active)
    a = np.where(act30 > 0, based, quiet)
    return np.clip(a, 1e-4, p.a_ceiling)


def _realized_stats(users: list[SimUser]) -> dict:
    act = np.array([u.act_30d for u in users])
    sil = np.array([u.silence_days for u in users])
    conc = np.array([u.interest_concentration for u in users])
    ten = np.array([u.tenure_days for u in users])
    a = np.array([u.a_daily for u in users])
    return {
        "n": len(users),
        "act30_gt0_share": round(float((act > 0).mean()), 4),
        "act_30d_mean": round(float(act.mean()), 4),
        "silence_median": round(float(np.median(sil)), 2),
        "concentration_median": round(float(np.median(conc)), 4),
        "tenure_median": round(float(np.median(ten)), 1),
        "a_daily_mean": round(float(a.mean()), 5),
    }


# ── 两条生成路径 ────────────────────────────────────────────

def _generate_fit(n: int, rng: np.random.Generator, df: pd.DataFrame, p: SimParams):
    """分位数秩耦合抽样：返回 (act30, silence, conc, tenure, a_daily)。"""
    u = rng.uniform(0.0, 1.0, n)      # act_30d 秩（同一 u 反向用于 silence，保单调负相关）
    w = rng.uniform(0.0, 1.0, n)      # 集中度秩
    w2 = rng.uniform(0.0, 1.0, n)     # 资历秩
    act30 = np.round(_rank_sample(np.sort(df["act_30d"].to_numpy()), u))
    silence = _rank_sample(np.sort(df["recency_days"].to_numpy()), 1.0 - u)
    conc = _rank_sample(np.sort(df["genre_top1_ratio"].to_numpy()), w)
    tenure = _rank_sample(np.sort(df["tenure_days"].to_numpy()), w2)
    return act30, silence, conc, tenure, _derive_a_daily(act30, rng, p)


def _generate_default(n: int, rng: np.random.Generator, p: SimParams):
    """内置默认人口（公开仓离线路径，设计取值）：返回同上五项。"""
    shares = np.array([m[1] for m in p.pop_mix], dtype=float)
    shares = shares / shares.sum()
    pick = rng.choice(len(p.pop_mix), size=n, p=shares)  # 档位（dormant/silent/low/active）

    def col(j: int) -> np.ndarray:
        return np.array([p.pop_mix[i][j] for i in pick], dtype=float)

    act30 = np.round(rng.uniform(col(2), col(3), n))
    # act_30d>0 由事件数反推节奏；act_30d=0 按档位自己的低节奏带取值
    a_daily = np.where(act30 > 0, act30 / (30.0 * p.count_mean_active), rng.uniform(col(4), col(5), n))
    a_daily = np.clip(a_daily, 1e-4, p.a_ceiling)
    silence = rng.uniform(col(6), col(7), n)

    base, span, ba, bb = p.default_conc
    conc = base + span * rng.beta(ba, bb, n)
    mu, sigma = p.default_tenure_lognorm
    t_lo, t_hi = p.default_tenure_clip
    tenure = np.clip(rng.lognormal(mu, sigma, n), t_lo, t_hi)
    return act30, silence, conc, tenure, a_daily


def generate_users(
    n: int,
    seed: int,
    features_path: Path | str | None = DEFAULT_FEATURES,
    params: SimParams = SIM,
) -> Population:
    """生成 n 个合成用户。

    features_path=None → 强制默认分布；给了但文件缺失/列不可用 → 警告并兜底。
    """
    if n < 1:
        raise ValueError(f"用户数需 ≥ 1（当前 {n}）")
    rng = np.random.default_rng(np.random.SeedSequence([seed, STREAM_POP]))

    meta: dict = {"requested_n": n}
    act30 = silence = conc = tenure = a_daily = None

    path = Path(features_path) if features_path is not None else None
    fit_ok = False
    if path is not None:
        if path.exists():
            df = pd.read_csv(path)
            missing = [c for c in FIT_COLUMNS if c not in df.columns]
            if missing:
                meta["fallback_reason"] = f"特征表缺列：{','.join(missing)}"
            else:
                df = df[list(FIT_COLUMNS)].dropna()
                if len(df) == 0:
                    meta["fallback_reason"] = "特征表四列全空"
                else:
                    act30, silence, conc, tenure, a_daily = _generate_fit(n, rng, df, params)
                    meta["path"] = "fit"
                    meta["features_file"] = _rel(path)
                    meta["features_fingerprint"] = _fingerprint(path)
                    meta["fit_rows"] = int(len(df))
                    fit_ok = True
        else:
            meta["fallback_reason"] = f"未找到 {_rel(path)}"

    if not fit_ok:
        if "fallback_reason" in meta:
            print(f"[warn] {meta['fallback_reason']}，改用内置默认分布", file=sys.stderr)
        meta["path"] = "default"
        act30, silence, conc, tenure, a_daily = _generate_default(n, rng, params)

    users = [
        SimUser(
            uid=f"sim_{i:06d}",
            a_daily=float(a_daily[i]),
            silence_days=float(silence[i]),
            act_30d=float(act30[i]),
            interest_concentration=float(conc[i]),
            tenure_days=float(tenure[i]),
        )
        for i in range(n)
    ]
    meta["realized"] = _realized_stats(users)
    return Population(users=users, meta=meta)
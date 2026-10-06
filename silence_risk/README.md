# M1 · 沉默预测（silence_risk v1）

> **边界一句话**：L2 / L3 回答「用户现在是什么状态」，harness 决定「干预谁」，
> **M1 回答「只看公开行为痕，能否把『未来会不会沉默』预测得比 recency 规则更准——且经得起时间外推」**。
>
> 为什么需要它：L3 的 `churn_risk` 是启发式阈值（`calibration_status = uncalibrated`）；
> 干预队列需要一个**可校准、可复算、可验收**的风险分，而不是又一条拍脑袋规则。
>
> 所以这一层 **不做**：因果效果（那是 S1–S4 的活）、运营档位与动作（归 L3 / harness 的判断）、
> 真实流失标签（公开时间线只能观测"还有没有公开活动"）。

**本层决策（2026-10-06，已定稿）**
- 口径 **Future Public Inactivity Prediction**（公开沉默），不叫 churn：
  锚点 T 提问，y = (T, T+30d] 内无任何 exact 事件。
- 双模型：`logreg`（线性，概率可校准）+ `hgb`（非线性主力候选）；对照组 = `heuristic`（recency 规则 `gap_days`）。
- 双协议评测：`group_cv5`（按 uid 分组 OOF，同期口径）+ `time_extrap`（训旧 → 测新）。
- **M1 达标线只看时间外推**：任一挑战者的 ΔAUC 对 recency 的 95% 聚类 bootstrap 下界 > 0。

---

## 1. 数据流与模块

```
L2 timeline.jsonl（exact 事件）+ 采集索引（fetched_at）
        │
        ▼
silence_risk/
        ├── panel.py       (用户, 锚点 T) 面板：锚点推进 / 标签 / 右截断保护 / 左截断标记
        ├── features.py    多窗计数主干（n7..n180 + gap_days）+ 180d 类型占比（FEATURE_SETS 单一真源）
        ├── models.py      分类器工厂（logreg / hgb，超参单一真源 + 快照）
        ├── evaluate.py    AUC / PR-AUC / Top-k / Brier / 分位校准 + 聚类 bootstrap ΔAUC
        ├── run_silence.py CLI：双协议评测 → 落盘 → _manifest → 双跑一致性校验
        └── tests.py       冒烟与一致性测试（7 组：手算 / 端点 / 协议 / 无泄漏 / 确定性 / 边界）
        ▼
data/processed/silence/       （本地，不入公开仓；仅聚合指标与 uid_hash）
    ├── silence_metrics.json      配置 / 面板统计 / 双协议 × scorer 指标 / ΔAUC / M1 裁决
    ├── silence_calibration.csv   分位十等分校准表（仅概率模型）
    ├── silence_predictions.csv   逐样本分数长表（protocol/split/uid_hash/anchor_ts/score_*）
    └── _manifest.json            版本 / 输入指纹 / 输出指纹 / 双跑一致
```

**硬规则（四条）**
1. **只用锚点 T 及之前的 exact 事件**（无泄漏）；`unknown` 时间的事件不进任何时间序列——tests 用手算样例 + "未来事件不改既往特征"断言锁死。
2. **右截断保护**：T + 30d ≤ 采集时刻，标签窗口必须完整可见；左截断（T−180d 早于首个可见事件）单独标记，并做子集稳健性复核。
3. **按用户分组**评测（GroupKFold + 按 uid 聚类 bootstrap），防止同一用户多锚点造成的自相关假显著。
4. deterministic：同参同种子逐字节一致；CLI 默认**双跑一致性校验**，不一致拒绝落盘。

---

## 2. 面板与特征口径

| 项 | 取值 | 说明 |
|---|---|---|
| 锚点 T | 首事件 + 30d 起，每 30d 步进 | 只对"近期活跃"用户提问 |
| 样本条件 | (T−30d, T] 内有 ≥1 exact 事件 | 窗口统一**左开右闭** |
| 标签 y | 1 = (T, T+30d] 内无 exact 事件 | 右闭：恰在 T+30d 的事件算"有活动" |
| 特征主干 | `n7 n14 n30 n60 n90 n180` + `gap_days` | 多窗计数；`gap_days` = 距最近一次公开事件天数（即 recency 基线本身） |
| 类型块（可选） | `t180_{review,post,wishlist,badge,follow_user}_share` | 180d 事件类型占比；`--feature-set trunk_types` 启用 |

---

## 3. 评测与数字（400 人时间线，2026-10-06）

命令：`python -m silence_risk.run_silence`（面板 3,212 样本 / 360 用户 / 沉默率 60.8% / 左截断 16.2%）

| 协议（评测行） | scorer | AUC | PR-AUC | Recall@10% | Precision@10% | Brier |
|---|---|---|---|---|---|---|
| group_cv5 OOF（3,212） | recency | 0.602 | 0.680 | 0.115 | 0.701 | — |
| group_cv5 OOF | logreg | 0.660 | 0.715 | 0.121 | 0.738 | 0.219 |
| group_cv5 OOF | hgb | 0.618 | 0.678 | 0.113 | 0.685 | 0.245 |
| time_extrap test（928） | recency | 0.628 | 0.636 | 0.127 | 0.688 | — |
| time_extrap test | **logreg** | **0.695** | **0.697** | **0.139** | **0.753** | 0.229 |
| time_extrap test | hgb | 0.632 | 0.630 | 0.119 | 0.645 | 0.256 |

**读出来的四件事**
1. **M1 达标（logreg）**：时间外推 ΔAUC = +0.067（95% CI [+0.037, +0.095]，P(>0)=1.00）——ML 显著优于 recency；同期 OOF 只有 +0.058，说明这不是"同期拆分注水"。
2. **快检数字复现**：可行性快检（200 人，logreg）外推 0.697 → 正式版（400 人）0.695，口径落地无漂移。
3. **hgb 未达标**（+0.004，CI [−0.039, +0.045]）：当前超参在该数据量下过拟合；诊断实验显示强正则可到 ~0.68，但调参必须在训练侧内层 CV 做，留待 v2。
4. **类型占比无增量**（外推 ΔAUC 负）→ 不做复杂特征工程；长窗计数是信号主力。

---

## 4. 运行与产出

```bash
# 在仓库根目录执行（前置：L2 timeline.jsonl + 原始采集索引）
python -m silence_risk.run_silence                     # 默认：400 人口径 + 双跑校验
python -m silence_risk.run_silence --no-verify          # 快路径
python -m silence_risk.run_silence --feature-set trunk_types   # 消融用
python -m silence_risk.tests                            # 测试（也可 pytest silence_risk/tests.py）
```

CLI：`--timeline --index --out --horizon --step --pre-window --pred-window --min-history
--cv --split-until --top-frac --seed --feature-set --no-verify`。
退出码：参数 / 数据问题 `2`；双跑不一致 `3`（拒绝落盘）。

---

## 5. 与上下游的接口

| 上下游 | 用法 |
|---|---|
| L2 特征层 | 读 `timeline.jsonl`（仅 `time_kind=exact`）与采集索引 `fetched_at`；不做任何写回 |
| L3 / harness | `silence_metrics.json` 的结论（外推 AUC、校准表）可作为「风险分可信度」的证据引用；在线打分留给后续（需先把模型固化为可加载产物） |
| 干预实验 | 主指标口径（未来 30 天公开沉默率）与 [harness](../harness/) 的实验框架对齐——本层提供该口径的可复算评测 |

---

## 6. 已知局限（诚实清单）

1. **幸存者偏差**：种子池来自"近期发表过评价/动态"的用户 → 沉默率是"此类用户"口径，不代表全站。
2. **公开沉默 ≠ 流失**：只能观测公开行为痕，用户可能仍在使用只是不公开发言。
3. **可见历史 ≠ 完整历史**：分页可见范围限制（左截断 16.2% 已标记 + 子集复核）。
4. **单快照**：每个用户只有一次 `fetched_at`；采集节奏变化会改变右截断位置。
5. **模型族有限**：logreg / hgb 未做超参搜索（hgb 调参需内层 CV，v2）；维持"简单可信"优先。
6. **无在线接口**：本层只做离线评测，尚未产出可加载的模型文件（v2 再谈）。

---

*运行环境：Python ≥ 3.10，numpy / pandas / scikit-learn（见 requirements.txt）。改任何口径（面板 / 特征 / 模型 / 指标）前先升 `SILENCE_VERSION` 并重跑双跑校验。*
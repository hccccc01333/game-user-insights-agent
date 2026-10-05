# L3 · 洞察层（活跃度 / 兴趣迁移 / 流失风险 → Agent 输入契约）

> **项目目标**：用户行为洞察——利用 AI 建模分析用户行为习惯，**精准识别流失风险**并**制定自动化干预策略**。
>
> **边界一句话**：L2 回答「这个人是什么样的人、这些事发生在什么时候」；L3 回答「这个人**现在处于什么状态、可能往哪走、该不该干预**」；**「怎么办」是 Agent 的事**——L3 只把可引用的数字、标签与证据（facts）备好，不写策略。
>
> 所以这一层 **不做**：干预话术 / 触达动作 / 实验设计 / 决策循环——那些是 harness（Agent）的判断，L3 只交付**输入契约**。

**本轮决策（2026-10-04）**
- 推进方式：沿 L2 节奏——**先出设计文档 → 确认后写实现**。
- 三模型同批设计：**M1 活跃度 / M2 兴趣迁移 / M3 流失风险**（+ M4 = Agent 输入契约）。
- 流失用**多档风险梯度 30 · 60 · 90 天**；兴趣迁移**品类迁移 + 游戏流动都做**。
- 阈值来源一律标注「**未经运营反馈校准**」（见 §7）。
- Agent 层（Harness 组件语义、决策循环、LangChain/LangGraph 选型）设计决策归用户，L3 只提供输入契约与实现。

---

## 1. 层定位与数据流

```
L1  data/raw/user_profile/<uid_hash>.json          （只读，L3 永不触碰）
        │
        ▼
L2  data/processed/user_features/
        ├── features.csv / .parquet     用户级特征（200 × 99）
        └── timeline.jsonl              行为时间线（9521 事件）
        │   （只读；L3 不改写 L2 产物）
        ▼
L3_insights/
        ├── l3_schema.py        产出 schema 单一真源（字段 / 档位 / 锚点 / 版本）
        ├── app_tag_lookup.py   ★前置：app_id → tags 词表（补事件级品类，见 §3.2）
        ├── model_activity.py   M1 活跃度评分
        ├── model_churn.py      M3 流失风险分层
        ├── model_migration.py  M2 兴趣迁移（依赖 app_tag_lookup）
        └── build_facts.py      汇总三模型 → facts.jsonl（M4 输入契约）
        ▼
    data/processed/user_insights/
    ├── facts.jsonl           ★ Agent 唯一入口（一用户一行，只读）
    ├── activity.csv / churn.csv / migration.jsonl   三模型明细（可追溯）
    └── _manifest.json        版本 / 输入指纹 / 锚点快照 / 校准状态
        ▼
harness（Agent）——只读 facts + L2 特征，不碰 L1 原文
```

**硬规则**
1. **只读 L2，不改写**。L3 永不回写 `data/raw/` 与 `data/processed/user_features/`。
2. **deterministic + versioned + replayable**。同输入 + 同 `L3_VERSION` → 逐字节一致；换口径 = 升版本 + 重跑。
3. **锚点冻结**。所有归一化阈值来自本批 200 人分布并写入 `_manifest.anchors` 后冻结，跨批次可比（同 L2 词表纪律）。
4. **锁数纪律**。Agent 只见 facts 里的数字与标签；需要原文时按 id 走工具检索（reference, not copy）。

---

## 2. 四产出与 harness 六环节映射

harness 骨架的六个业务环节（[state.py](../harness/state.py) 注释）与本层产出的对应：

| harness 环节 | L3 供给 | 说明 |
|---|---|---|
| `anomaly` | M1 活跃度（绝对分 + 批内分位） | 异常活跃 / 异常沉默的定位 |
| `cohort` | M1 band + `tenure_bucket` + 类目画像 | 分群素材 |
| `cause` | M2 兴趣迁移（品类迁移 + 游戏流动） | 「为什么活跃/流失」的证据 |
| `risk` | M3 流失风险（stage + score + horizon） | 核心：精准识别流失风险 |
| `intervention` | M4 facts 契约（drivers + context） | 自动化干预策略的**输入** |
| `experiment` | 预留（本轮不产） | 后续 |

> L3 **不产** `intervention` 的策略内容，只产 `intervention` 的**素材**（风险驱动因子、偏好品类、近期关注游戏、情绪信号）。策略由 Agent 生成。

---

## 3. 三模型计算口径

### 3.0 通用：分位锚点归一化（锚点冻结）

所有子分先做**方向统一 + 重尾压缩 + 锚点归一**到 `[0,1]`，再按权重合成，最后 ×100。

```
norm(x;  cap, p_low) = clamp( (log1p(x) - log1p(p_low)) / (log1p(cap) - log1p(p_low)), 0, 1 )
```

- `cap` / `p_low` 取本批 200 人的分位（如 p95 / p10），**写入 `_manifest.anchors` 后冻结**。
- 反向指标（如 `recency_days`）先取反再归一：`norm_rev(x) = 1 - norm(x)`，或半衰形式 `1/(1+x/HALFLIFE)`（`HALFLIFE` 同样冻结）。
- 所有原始值**同时**输出到明细表，Agent 可引用真实数字而非只看分数（**透明度优先**）。

### 3.1 M1 活跃度评分（activity）

**问题**：给每个用户一个 0–100 的「当前活跃度」，可解释、可拆分、无标签也能算。

**输入列（来自 L2）**：`act_30d / act_90d / act_180d / decay_ratio / recency_days / content_total / content_per_year / played_app_count / distinct_tag_count / following_app_count / forum_count / fans_count / following_count / voteup_received / avg_ups_per_moment / max_ups / mutual_count`。

**四维子分（初始权重，待探索调整）**

| 子分 | 权重 | 主要输入 | 口径 |
|---|---|---|---|
| `activity_intensity` 近期产出强度 | 0.30 | `act_30d`,`act_90d`,`content_per_year` | 近端加权产出量归一 |
| `activity_recency` 近端新鲜度 | 0.35 | `recency_days`,`decay_ratio` | `0.65×`半衰衰减(`HALFLIFE=60d`) `+ 0.35×`decay_ratio 归一 |
| `activity_breadth` 参与广度 | 0.20 | `played_app_count`,`distinct_tag_count`,`following_app_count`,`forum_count` | 触达面归一 |
| `activity_social_influence` 社交与影响 | 0.15 | `fans_count`,`following_count`,`voteup_received`,`avg_ups_per_moment` | 关系 + 口碑归一 |

**产出**
- `activity_score` 0–100（锚点绝对分，跨批次可比）
- `activity_percentile` 0–100（**批内**分位，捕捉相对位置）
- `activity_band` ∈ `dormant / low / mid / high / top`（按批内分位切：<p25 / p25–p50 / p50–p75 / p75–p90 / >p90）
- `activity_drivers`：贡献最大的 2–3 个子分（Agent 可引用）

> **⚠️ 本批实测：`act_30d>0` 仅 23/200（11.5%），`recency_days` 中位 81 天**——「近 30 天活跃」层极薄，绝对分整体偏低是**数据事实而非模型 bug**；故必须同时给批内分位（`activity_percentile`）供分群。

### 3.2 M2 兴趣迁移（migration）

**两个子模型**（本轮都做）：

#### （a）品类迁移 genre migration
- **前置（缺口）**：时间线的 `review/post/wishlist` 事件有 `app_id` 但**无 tags**（实测：post 1621/1727、review 1106/1134、wishlist 37/37 有 `app_id`）。
- **解法（推荐）**：L3 侧新建 `app_tag_lookup.json`——从 L1 各用户 `following_app` / `favorite_app` 面（**携带 `app.tags`**）汇总 `app_id → tags[]` 词表并冻结。**无需重跑 L2**（app 品类是静态属性，非动作）。
  - 备选：把 tags 落进 timeline 是 **L2 口径变更**（需升 `FEATURE_VERSION`）——除非确需，不采用。
- **口径**：按 `event_ts` 排序，把用户事件切成**早期窗 vs 近期窗**（默认 `近期=近 180 天 exact 事件`，`早期=其之前`），分别聚合 tags 分布 → 
  - `genre_shift_score` ∈ [0,1]：两窗分布的 JS 散度（或有界化）。
  - `genre_entropy_delta`：`tag_entropy` 时间趋势（兴趣变散 / 变聚焦）。
  - `genre_from` / `genre_to`：份额变化最大的 1–2 个品类（迁移方向）。
- **门槛**：两窗均需 `≥ MIN_EVENTS(默认5)` 个带 tag 事件，否则置 `null` + `insufficient_data`（**不猜**）。

#### （b）游戏流动 game flow
- **输入**：时间线中带 `app_id` 的 `review/post/wishlist` 事件（`exact`）。
- **口径**（近窗口默认 180 天）：
  - `entered_games`：近窗首次出现、早期未见 → 新入场游戏数 / 列表。
  - `dropped_games`：早期活跃、近窗消失 → 流失游戏数 / 列表。
  - `returned_games`：早期有、中断 ≥ 90 天、近窗再现 → 回流游戏数。
  - `game_flow_net` = entered − dropped（正=扩张，负=收缩）。
- **产出**：`migration.jsonl`（一用户一行）。**与 M3 联动**：`dropped_games` 多 + `genre_shift_score` 高 → 作为流失的「原因证据」进入 facts 的 `cause` 槽。

### 3.3 M3 流失风险分层（churn）

**约束（关键）**：单快照、**无纵向流失标签** → 不能监督训练。因此采用**规则/信号驱动的综合风险评分 + 停更档位**，并**显式声明未经校准**。

**三件产出**
1. `churn_stage`（**停更档位**，30·60·90 梯度，硬标签，来自 `recency_days`）：

| stage | 条件 | 含义 |
|---|---|---|
| `churned` | `is_deactivated` 或 `is_deleted` | 已流失（硬判定，覆盖其余） |
| `silent` | `is_silent` | 平台标记沉默 |
| `dormant_90` | `recency_days > 90` | 停更 >90 天 |
| `dormant_60` | `60 < recency_days ≤ 90` | 停更 60–90 天 |
| `dormant_30` | `30 < recency_days ≤ 60` | 停更 30–60 天 |
| `active` | `recency_days ≤ 30` | 近 30 天有产出 |

2. `churn_risk_score` 0–100（综合，越高越危险）：

| 分量 | 权重 | 输入 | 方向 |
|---|---|---|---|
| `staleness` 停更 | 0.35 | `recency_days` | 越高越危 |
| `momentum` 动量衰减 | 0.25 | `1 - act_90d/max(act_180d,1)`（近 90 天在近 180 天中的占比反演） | 越低越危 |
| `sentiment` 情绪/口碑 | 0.15 | `dim_neg_rate_*` 均值、`review_score_mean` 偏低 | 差评/低分越危 |
| `social_erosion` 社交收缩 | 0.10 | `follower_ratio`、`fans_count`、`mutual_count` | 越低越危 |
| `account_flag` 账号标记 | 0.15 | `is_silent / is_deactivated / is_deleted` | 命中→置顶（overlay） |

3. `churn_horizon_days` ∈ `{90, 60, 30, none}`（**预警视界**）：按 `recency_days` 距 90 天沉默线的**剩余跑道**粗分档（`≤30→90`、`30–60→60`、`60–90→30`、`>90 或已流失→none`）。
   - 与 `churn_stage` 互补：stage 是**已发生多久**，horizon 是**还差多久**（规则驱动，非学习所得）。

**产出**：`churn.csv`；`churn_drivers`（贡献最大的 2–3 个分量）。

> **⚠️ 本批实测**：`recency_days` 中位 81、p75=86、`>90` 仅 25/200、`>30` 高达 177/200。→ 若**只按绝对天数**分级，会把 88% 用户打成风险。故：
> - `churn_stage` 保留（诚实、可解释）；
> - 但**风险排序主要靠 `churn_risk_score`**（融合动量/情绪/社交，非只停更）；
> - 且 `churn_stage` 边界**记录为待校准**，未来用运营反馈或真实回访标签上/下移。

---

## 4. 干预策略输入契约（M4 = facts.jsonl）

**这是 Agent 唯一入口**。一用户一行；**只读**；**不含 L1 原文**；**不含策略结论**（策略由 Agent 生成）。

```json
{
  "uid_hash": "h_xxxxxxxxxxxxxxxx",
  "facts_version": "l3v1",
  "as_of": 1791108404,
  "calibration_status": "uncalibrated",      // 未经运营反馈校准
  "quality": {
    "usable": true,
    "truncated_surfaces": "following_app",
    "account_status": "active",
    "wishlist_locked": false
  },
  "activity": {
    "score": 42.1, "percentile": 68, "band": "mid",
    "sub": { "intensity": 0.31, "recency": 0.55, "breadth": 0.72, "social_influence": 0.40 },
    "drivers": ["recency", "breadth"]
  },
  "churn": {
    "stage": "dormant_60", "risk_score": 71.4, "horizon_days": 60,
    "sub": { "staleness": 0.78, "momentum": 0.90, "sentiment": 0.30, "social_erosion": 0.55, "account_flag": 0 },
    "drivers": ["staleness", "momentum"]
  },
  "migration": {                              // S2 已实现；两窗各需 ≥5 带 tag 事件，否则 genre_*=null
    "window_days": 180,
    "genre_shift_score": 0.28, "genre_entropy_delta": -0.4,
    "genre_from": "角色扮演", "genre_to": "策略",
    "entered_games": ["app_id_a", "app_id_b"], "dropped_games": ["app_id_c"],
    "returned_games": [], "game_flow_net": 1,
    "insufficient_data": false,
    "n_tag_events_early": 6, "n_tag_events_recent": 8,   // 两窗规模（透明）
    "n_games_early": 5, "n_games_recent": 4
  },
  "context": {                                // 供 Agent 生成策略的素材（非结论）
    "top_genres": [["二次元", 0.31], ["角色扮演", 0.22]],
    "recent_app_ids": ["app_id_a"],
    "sentiment_neg_rate": 0.18,
    "tenure_bucket": "3-5y",
    "reference_events": ["ev_ab12...", "ev_cd34..."]   // 可回查的 L2 事件 id（不落原文）
  }
}
```

**契约承诺（Agent 可依赖）**
- `uid_hash` 唯一；`facts_version` / `as_of` / `calibration_status` 必有。
- 三模型字段**要么有值、要么显式 `null` + 原因**（`insufficient_data` / `quality.usable=false`），**绝不静默填 0**。
- 数值均为 L2 可复算结果或本层确定性派生；`reference_events` 只给 id，原文按 id 走工具检索。
- 敏感字段（`ip_location` / `device` / `gender`）**不进 facts 个体层**，仅聚合出现在报告层。

---

## 5. 阈值锚点与数据探索计划

### 5.1 本批 200 人实测分布（**初始档位锚点，未经校准**）

| 指标 | p10 | p25 | p50 | p75 | p90 | p95 | 备注 |
|---|---|---|---|---|---|---|---|
| `recency_days` | 20.7 | 70.2 | **81.1** | 86.3 | 91.3 | 94.6 | 双峰：少数极新 + 大多数 ~80 天 |
| `decay_ratio` | 0 | 0 | 0 | 0 | 0.111 | 0.423 | 近端活跃占比普遍为 0 |
| `act_30d` | 0 | 0 | 0 | 0 | 1.1 | 3.0 | 仅 23/200 非零 |
| `act_90d` | 0 | 2 | 2 | 4 | 10.2 | 24.2 | 175/200 非零 |
| `act_180d` | 2 | 2 | 3 | 6 | 16.1 | 27.1 | 198/200 非零 |
| `content_total` | 1 | 2 | 5 | 15 | 38.8 | 81 | 长尾至 5407 |
| `tag_entropy` | — | 4.61 | 5.26 | 5.66 | — | — | 兴趣偏分散 |
| `genre_top1_ratio` | — | 0.074 | 0.094 | 0.125 | — | — | 无强主品类 |
| `review_score_mean` | — | 2.97 | 3.80 | 5.00 | — | — | 走两极 |

**样本事实**：`fans_count>0` 91/200 · `following_count>0` 157/200 · `voteup_received>0` 164/200 · `account_status` 全 `active` · `is_silent` 全 `False` · 截断 7 人。

### 5.2 探索轮（定初始档位的可执行清单）
- **E1 活跃度分档**：对 `activity_score` 取 p25/p50/p75/p90 → 验证五档分布不塌缩。
- **E2 流失梯度**：扫 `recency_days` 各档人数（30/60/90），标注「若按绝对天数则 88% 入险」，据此**确认以 `risk_score` 排序为主**。
- **E3 兴趣迁移门槛**：统计两窗事件数分布 → 定 `MIN_EVENTS`；看 `genre_shift_score` 分布是否可分。
- **E4 外部锚（弱参考）**：`calibration/reddit_params.csv`（如周留存 `P(w+1|w)=21.4%`、`Gini=0.444`）作阈值的**外部合理性检查**，**不作主锚**（跨平台、跨口径）。
- **E5 稳定性**：对 `--limit 100` 子样重跑，看分数排序与全量一致率（锚点冻结后）。

#### S3 探索结果（2026-10-04）
| 项 | 实测 | 结论 |
|---|---|---|
| **E1 活跃度分档** | `activity_score` p25/p50/p75/p90 = 21.7 / 29.3 / 36.9 / 55.1；五档 49/50/50/30/21 | ✅ **不塌缩**，档位定案 |
| **E2 流失梯度** | 30·60·90 天数档 = active 23 / d30 20 / **d60 132** / d90 25；风险均分单调 19.1<35.3<44.7<71.7 | ✅ 确认**以 `risk_score` 排序为主**；天数档塌缩为数据事实，保留可解释性 |
| **E3 迁移门槛** | 两窗各≥5 → 21 人；≥3 → 38；≥2 → 93；`genre_shift_score` 21 人均值 0.571（可分） | ⏸ **建议保留 `MIN_EVENTS=5`**（质量优先，`cause` 槽由全量 `game_flow` 兜底）；放宽留待样本扩充后复议 |
| **E4 外部锚（弱）** | `content_total` Gini **0.903**、top1% 64.1%、top10% 87.6%（Reddit 参考 0.444 / 14.3% / 42.1%） | ⚠️ 分布**远比 Reddit 集中**——样本偏「近期活跃+含 5407 极端值」，仅作量级参考，**不作主锚** |
| **E5 稳定性** | `--limit` 会**重算锚点**，与全量不可比 | ⏸ 需给 `model_*.py` 加「锚点透传」才能做真正子样一致性；**本轮未做**（记为待办） |

> 结论：**不改口径、维持 `l3v1`**（E1/E2 达标；E3 保留 5；E5 待补锚点透传）。

---

## 6. 产出 schema 与落盘

```
data/processed/user_insights/
├── facts.jsonl          ★ Agent 唯一入口（一用户一行，见 §4）
├── activity.csv         M1 明细：score / percentile / band / 四子分 / drivers
├── churn.csv            M3 明细：stage / risk_score / horizon / 五分量 / drivers
├── migration.jsonl      M2 明细：genre_* + game_* + insufficient_data
├── app_tag_lookup.json  app_id → tags 冻结词表（M2 前置产物）
└── _manifest.json       {l3_version, generated_at, n_users, input_fingerprint,
                          anchors, calibration_status, coverage_summary,
                          app_tag_lookup, migration}
```

- `input_fingerprint` = 对 L2 `features.csv` + `timeline.jsonl` 的 sha256 → 判「是否需重跑」。
- `anchors` = §3.0 的所有冻结分位/权重/窗口/门槛 → 保证可复现、跨批可比。
- `coverage_summary` = 各模型有值/`insufficient_data`/`usable=false` 的人数 → 诚实交代覆盖。
- **Agent 侧接线**：`harness/facts.py` 只读加载 facts 并提供六环节默认映射（`FactsStore.stage_view()`）；决策语义不在该层（S4）。

---

## 7. 诚实边界（必须随产出引用）

1. **无流失标签**：单快照、无纵向回访 → `churn_risk_score` 是**信号合成排序**，**不是**校准过的概率；**阈值未经运营反馈校准**（`calibration_status=uncalibrated`）。
2. **样本偏置**：200 人、偏「近期有过评价/动态」的用户；`recency` 中位 81 天 → 结论不代表全站。
3. **截断 7 人**：其计数类特征偏低（`truncated_surfaces` 已标记）→ 影响 M1 广度分与 M2 流动，`quality` 已透出。
4. **时间线 unknown 事件**（`fans` 1950 条）不参与任何时间序列/迁移计算。
5. **品类迁移受词表约束**：冻结 top20（L2）+ `app_tag_lookup` 覆盖度 → 长尾/新品类缺失，迁移结论**在词表内**成立。
6. **合规**：`ip_location`/`device`/`gender` 不进 facts 个体层；`following_user.id`/`fans.id` 只出计数。

---

## 8. 版本 / 复现 / 运行

- `L3_VERSION`（`l3v1`）写入每行与 `_manifest`；口径 = `l3_schema.py`（同 L2 `feature_dict.py` 纪律：名字/权重/档位/锚点只定义一次）。
- **全量重算**：200 人级成本可忽略，不做逐 uid 断点；换口径 = 升版本 + 重跑，旧产物并存。
- 确定性：排序、窗口边界、锚点均显式固定；`as_of` 取 `_manifest.as_of`（与 L2 同源）。

```bash
# 在仓库根目录（建议顺序；M2 依赖前置词表）
python L3_insights/app_tag_lookup.py      # 前置：app_id → tags
python L3_insights/model_activity.py      # M1
python L3_insights/model_churn.py         # M3
python L3_insights/model_migration.py     # M2
python L3_insights/build_facts.py         # M4 汇总（依赖以上全部）
# 常用：--limit N 小样；--in-dir/--out-dir 改路径
```

---

## 9. 实现路线（分步，先闭环后深化）

| 步 | 内容 | 前置 | 验收 |
|---|---|---|---|
| S1 | `l3_schema.py` + `model_activity.py` + `model_churn.py` + `build_facts.py`（**活跃度 + 流失共享时间线信号，先出闭环**） | L2 就绪 | ✅ **2026-10-04 完成**：200 人 facts 落盘，字段无静默 0，重跑逐字节一致 |
| S2 | `app_tag_lookup.py` + `model_migration.py`（品类迁移 + 游戏流动） | S1 | ✅ **2026-10-04 完成**：migration 200 行；`app_tag_lookup` 覆盖 610/625=97.6%；品类迁移可用 21/200（`insufficient_data` 179 显式，见 §11） |
| S3 | 数据探索轮（§5.2）回填初始档位与权重 | S1–S2 | ✅ **2026-10-04 探索完成**：E1 分档不塌缩、E2 确认风险排序为主、E3 建议保留 `MIN_EVENTS=5`、E4 弱锚仅量级、E5 待补锚点透传；**维持 `l3v1`** |
| S4 | harness 接线（facts 读取契约层） | S1–S3 | ✅ **2026-10-04 完成契约层**：`harness/facts.py`（`load_facts()` / `FactsStore.stage_view()` 六环节默认映射 / `register_facts_tools()`）；**Agent 层决策语义仍归用户**（本层只读、不做判断） |

---

## 10. 已拍板决策（2026-10-04）

| # | 决策 | 结论 | 理由 / 代价 |
|---|---|---|---|
| A | 推进节奏 | ✅ 先文档 → 确认 → 实现（同 L2） | 口径先对齐，避免返工 |
| B | 三模型范围 | ✅ M1 活跃度 / M2 兴趣迁移 / M3 流失风险 同批设计 | 目标四产出一次对齐 |
| C | 流失梯度 | ✅ 多档 30·60·90（`churn_stage`）+ 综合 `risk_score` + `horizon` | 无标签下既诚实又可排序；代价 = 阈值未校准 |
| D | 兴趣迁移范围 | ✅ 品类迁移 + 游戏流动 都做 | 覆盖「兴趣」与「行为」两面 |
| E | 事件级品类缺口 | ✅ **L3 侧 `app_tag_lookup` 关联**，不改 L2 | 免升 `FEATURE_VERSION`；代价 = 多一张冻结词表 |
| F | facts 定位 | ✅ Agent **唯一入口**、只读、不含原文与策略 | 锁数纪律；策略由 Agent 生成 |

---

## 11. 实测（2026-10-04，S1 + S2：M1 + M2 + M3 + facts）

| 指标 | 数值 |
|---|---|
| facts | **200 行**（`activity`/`churn`/`migration` 各 200 有值；`usable=false` 0） |
| 活跃度分档 | dormant 49 · low 50 · mid 50 · high 30 · top 21｜均分 **32.47** |
| 流失档位 | active 23 · dormant_30 20 · dormant_60 **132** · dormant_90 25 · silent/churned 0 |
| 流失风险分 | min 4.83 · 中位 43.18 · max 85.0｜均分 44.16 |
| 风险—档位单调性 | active 19.1 < dormant_30 35.3 < dormant_60 44.7 < dormant_90 71.7 ✅ 方向正确 |
| context 覆盖 | top_genres 198 · recent_app_ids 199 · reference_events 200 |
| 确定性 | 连跑两遍 `activity.csv`/`churn.csv`/`migration.jsonl`/`facts.jsonl` **逐字节一致** ✅ |

**M2 实测（2026-10-04，S2）**

| 指标 | 数值 |
|---|---|
| `app_tag_lookup` | **5229 个 app** / 13870 tag 对；timeline 覆盖 **610/625 = 97.6%**（内容事件 2764） |
| migration 行数 | **200**（与 facts 行域对齐） |
| 品类迁移可用 | **21/200**；`insufficient_data=179`（两窗各需 ≥5 带 tag 事件，实测早期窗稀疏） |
| 两窗规模 | `n_tag_events_early` p50=1（≥5 仅 46/200）· `n_tag_events_recent` p50=3（≥5 66/200） |
| `genre_shift_score`（21 人） | min 0.068 · 均值 **0.571** · max 0.895（JS 散度，0=不变） |
| `genre_entropy_delta`（21 人） | min −2.31 · 均值 **−0.148** · max +1.06（略偏聚焦） |
| 游戏流动（全 200） | `game_flow_net` 正 **131** · 零 21 · 负 **48**；`entered`/`dropped` 均可用 |

**关键读数**
- `dormant_60`（停更 60–90 天）占 **132/200** → 与本批 `recency` 中位 81 天吻合；**风险排序必须靠 `churn_risk_score` 而非仅看天数**（否则 88% 同档）。
- 风险分量在多数用户上由 `staleness + momentum` 主导（drivers 印证）；`sentiment`/`social_erosion` 作区分项。
- **M2 覆盖不对称**：内容事件集中在近 180 天窗（`recency` 中位 81 天）→ **早期窗薄**，品类迁移只在 21 人满足两窗门槛；`game_flow` 不依赖两窗门槛、覆盖全量。故 facts 的 `cause` 槽应优先用 `game_flow_net`+`dropped_games`，`genre_*` 作增强项。
- **门槛待定（E3）**：`MIN_EVENTS` 由 5 放宽 → 3 得 38 人、2 得 93 人可算；放宽会引入小样本噪声，留待 S3 探索轮权衡。
- 迁移方向样本可解释（如 `买断制→二次元`、`策略→角色扮演`、`美少女→开放世界`）。

> ⚠️ 所有档位/权重/锚点为**首版初始值，未经运营反馈校准**（`calibration_status=uncalibrated`）；锚点已冻结进 `_manifest.anchors`，跨批次可比。
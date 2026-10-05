# L2 · 特征层（用户行为特征与行为时间线）

> **边界一句话**：L1 回答「这个人做了什么（原始记录）」，L2 回答「这个人是什么样的人（特征）＋这些事发生在什么时候（时间线）」。
>
> 所以这一层 **不做**：分层结论、异常判定、流失结论、人群聚类命名——那些是 L3 / Agent 的判断，L2 只把原始 JSON 压成**可被引用、可复算**的特征与事件。

**本轮决策（2026-10-04）**
- 推进方式：先出设计文档 → 确认后再写抽取器。
- 产出粒度：**用户级特征表 + 行为时间线**（两件产物）。
- 消费者：harness（Agent State / facts）。L2 产出的数字是 Agent 唯一能看到的用户侧数字（锁数纪律）。

---

## 1. 层定位与数据流

```
L1  data/raw/user_profile/<uid_hash>.json     200 人 × 14 面原始 JSON
        │   （只读，不改写原始域）
        ▼
L2_features/
        ├── extract_features.py      聚合特征：每用户一行（宽表）
        ├── extract_timeline.py      行为时间线：一事件一行
        └── feature_dict           特征字典（自动导出，口径可查）
        ▼
    data/processed/user_features/features.csv       用户级特征表（宽表）
    data/processed/user_features/timeline.jsonl     行为时间线（事件流）
    data/processed/user_features/_manifest.json     版本 / 输入指纹 / 参数
        ▼
L3 / Agent（harness）——只读特征与事件，不碰原文
```

**硬规则**
1. **只读 L1，不改写原始域**。抽取器永不回写 `data/raw/`。
2. **deterministic + versioned + replayable**。同一份 L1 输入 + 同一 `FEATURE_VERSION` 必须产出逐字节一致的结果；换口径 = 升版本 + 可重跑。
3. **产出只落分析域**（`data/processed/`），敏感字段分级（见 §7）。

---

## 2. 输入：L1 数据面与关键字段（实测，2026-10-04）

主文件结构：`{uid_hash, fetched_at, surfaces, errors}`，`surfaces` 14 个面。

| 面 | 记录数（最富用户） | 关键字段（实测） |
|---|---|---|
| `detail` | 27 键 | `stat`（50 个聚合字段，见 §3）、`created_days`、`gender`/`country`/`language`、`ip_location`/`loc`、`is_silent`/`is_deactivated`/`is_deleted`、`badge_info.custom_shows`、`wear_badges`、`show_setting` |
| `feed_review` | 12 | `moment.{publish_time, created_time, commented_time, device, app, stat, review, actions, user_actions}` |
| `feed_moment` | 13 | 同上，`moment.{topic, group, labels}` 可能为 null |
| `favorite_app` | 256 | `app.{id, title, tags, released_time, stat.rating, stat.vote_info}` |
| `favorite_moment` | 293 | `moment.{publish_time, topic, labels, group, app, stat}` |
| `favorite_collection` | 0 | 同名结构 |
| `favorite_hashtag` | 0 | 同上 |
| `favorite_event` | 0 | 同上 |
| `wishlist` | 1 | `{app_id, app, created_time, app_released_time}` |
| `badge` | 20 | `{id, title, level, time, status, unlock_tips}` |
| `following_app` | 382 | `{id, title, tags[], released_time, stat.{rating, vote_info, ...}}` |
| `following_user` | 800 | `{id, verified, follow_source, created_time, gender, is_silent, is_deactivated}` |
| `following_hashtag` | 0 | 话题关注 |
| `fans` | 474 | `{id, follow_source, is_deactivated, ...}` |

**嵌套字段细节（写特征时直接引用）**
- `moment.review` = `{id, score(1–5), played_spent, contents.{text,raw_text}, ratings[], stage, stage_label, source, total_played_spent}`
- `moment.review.ratings[]` = 四个维度 `{type: degree_of_freedom|gameplay|operation|visual_music, value: up|down, label}`
- `moment.stat` = `{ups, supports, feedback_status, mentioned_app_play_time}`
- `moment.actions` = `{repost, comment}`；`moment.user_actions` = `{repost, favorite, share, comment}`
- `app.tags[]` = `{id, value, uri, web_url}`（品类轴，如「角色扮演/日系/机甲/开放世界/二次元」）
- `app.stat.rating` = `{score, max, latest_score, latest_version_score, latest_review_count, latest_version_review_count}`
- `app.stat.vote_info` = `{"1":..,"2":..,"3":..,"4":..,"5":..}`（该游戏全平台评分分布）

---

## 3. 五维特征字典

字段命名：`snake_case`，维度前缀 `id_ / act_ / taste_ / rel_ / inf_`。
类型：`int / float / bool / cat / list / ts`。口径列给出**计算式或来源**，"缺失"列给出缺值策略。

### ① 身份 / 生命周期 `id_`

| 特征 | 类型 | 口径 | 来源 | 缺失 |
|---|---|---|---|---|
| `uid_hash` | str | 文件主键 | 文件名 | 必有 |
| `tenure_days` | int | `stat.created_days` | detail.stat | -1 |
| `tenure_bucket` | cat | 分档（阈值待定，见 §10） | tenure_days | `unknown` |
| `gender` | cat | `detail.gender` | detail | `unknown` |
| `country` / `language` | cat | `detail.country` / `detail.language` | detail | `unknown` |
| `ip_location` | cat | `detail.ip_location`（**敏感**，见 §7） | detail | `unknown` |
| `is_silent` / `is_deactivated` / `is_deleted` | bool | 原字段 | detail | False |
| `account_status` | cat | 由三标记派生：`active / silent / deactivated / deleted` | detail | `unknown` |
| `badge_count` | int | `stat.badges_count` | detail.stat | 0 |
| `badge_wear_count` | int | `len(wear_badges)` | detail | 0 |
| `first_seen_proxy_ts` | ts | `min(badge.time)`（近似，非真实注册） | badge | null |

### ② 活跃 / 参与强度 `act_`

| 特征 | 类型 | 口径 | 来源 | 缺失 |
|---|---|---|---|---|
| `review_count` | int | `stat.created_review_count` | detail.stat | 0 |
| `moment_count` | int | `stat.created_moment_count` | detail.stat | 0 |
| `post_count` | int | `stat.created_post_count` | detail.stat | 0 |
| `topic_count` / `video_count` | int | 同名 stat 字段 | detail.stat | 0 |
| `content_total` | int | review+moment+post+topic+video | 派生 | 0 |
| `content_per_year` | float | `content_total / max(tenure_days/365, 0.08)` | 派生 | 0 |
| `played_app_count` | int | `stat.played_app_count` | detail.stat | 0 |
| `playing_app_count` | int | `stat.playing_app_count` | detail.stat | 0 |
| `history_app_count` | int | `stat.history_app_count` | detail.stat | 0 |
| `played_spent_total` | int | `stat.played_spent`（分钟） | detail.stat | 0 |
| `reserved_count` | int | `stat.reserved_count` | detail.stat | 0 |
| `cloud_game_played_count` | int | 同名 stat | detail.stat | 0 |
| `recency_days` | float | `(now - max(publish_time)) / 86400` | 时间线 | null |
| `act_30d` / `act_90d` / `act_180d` | int | 时间线中窗口内事件数（评论+动态） | 时间线 | 0 |
| `decay_ratio` | float | `act_30d / max(act_180d,1)` 近端活跃占比 | 派生 | 0 |
| `active_hour_top` | cat | 时间线 `publish_time` 时段众数（0–23，按 UTC+8） | 时间线 | null |
| `weekend_ratio` | float | 周末事件 / 全部事件 | 时间线 | 0 |
| `device_top` | cat | `moment.device` 众数（**敏感**） | feed | `unknown` |

### ③ 兴趣 / 游戏图谱 `taste_`

| 特征 | 类型 | 口径 | 来源 | 缺失 |
|---|---|---|---|---|
| `following_app_count` | int | `stat.following_app_count` | detail.stat | 0 |
| `favorite_app_count` | int | `stat.favorite_app_count` | detail.stat | 0 |
| `wishlist_count` | int | `len(wishlist)`（**口径 A**） | wishlist | 0 |
| `want_app_count` | int | `stat.app_wishlist_count`（**口径 B**） | detail.stat | 0 |
| `distinct_tag_count` | int | `following_app` 去重 tag 数 | following_app | 0 |
| `tag_entropy` | float | tag 频次香农熵（兴趣分散度） | following_app | 0 |
| `genre_top1_ratio` | float | 最高频 tag 占比（兴趣集中度） | following_app | 0 |
| `tag_vector` | list | 前 N 品类占比（N 待定，见 §10） | following_app | [] |
| `avg_follow_rating` | float | `mean(app.stat.rating.score)` | following_app | null |
| `follow_rating_std` | float | 上述标准差 | following_app | null |
| `review_score_mean` / `_std` | float | 用户自己 review.score 均值/标准差 | feed_review | null |
| `review_score_dist` | list | score 1–5 计数 | feed_review | [] |
| `dim_neg_rate_<dim>` | float | 各维度 `value=down` 占比（4 个维度） | feed_review.ratings | null |
| `wishlist_recent_ratio` | float | 心愿单 created_time 近 1 年占比 | wishlist | null |

> `tag_vector` 是唯一的向量型特征；其余为标量，便于 L3 直接建模。

### ④ 社交 / 关系 `rel_`

| 特征 | 类型 | 口径 | 来源 | 缺失 |
|---|---|---|---|---|
| `following_count` | int | `stat.following_count` | detail.stat | 0 |
| `fans_count` | int | `stat.fans_count` | detail.stat | 0 |
| `follower_ratio` | float | `fans / max(following,1)` | 派生 | 0 |
| `verified_following_ratio` | float | 关注用户中 `verified` 非空占比 | following_user | null |
| `following_alive_ratio` | float | 关注用户中未注销占比 | following_user | null |
| `fans_alive_ratio` | float | 粉丝中未注销占比 | fans | null |
| `follow_source_top` | cat | `follow_source` 众数 | following_user | `unknown` |
| `mutual_count` | int | `following_user.id ∩ fans.id`（**需明文 id 现算哈希，只出计数**） | following_user+fans | null |
| `following_hashtag_count` | int | `stat.following_hashtag_count` | detail.stat | 0 |
| `following_developer_count` | int | 同名 stat | detail.stat | 0 |
| `forum_count` | int | `stat.forum_count` | detail.stat | 0 |

### ⑤ 内容影响力 / 口碑 `inf_`

| 特征 | 类型 | 口径 | 来源 | 缺失 |
|---|---|---|---|---|
| `voteup_received` | int | `stat.voteup_count` | detail.stat | 0 |
| `votefunny_received` | int | `stat.votefunny_count` | detail.stat | 0 |
| `be_voted_up_review` | int | `stat.be_voted_up_review_count` | detail.stat | 0 |
| `be_voted_up_moment` | int | `stat.be_voted_up_moment_count` | detail.stat | 0 |
| `be_favorited_count` | int | `stat.be_favorited_count` | detail.stat | 0 |
| `avg_ups_per_moment` | float | `mean(moment.stat.ups)` | feed_review+feed_moment | null |
| `max_ups` | int | `max(moment.stat.ups)` | feed | 0 |
| `interaction_speed_median` | float | `median(commented_time - publish_time)`（秒） | feed | null |
| `favorite_moment_count` | int | `stat.favorite_moment_count` | detail.stat | 0 |
| `purchased_app_count` | int | `stat.purchased_app_count` | detail.stat | 0 |
| `app_achievement_count` | int | `stat.app_achievement_count` | detail.stat | 0 |

> 完整的 50 个 stat 字段不必全进特征表；**未进表的字段保留在原始 JSON**，需要时可回查（reference, not copy）。

---

## 4. 行为时间线模型

**统一事件 schema**（`timeline.jsonl`，一事件一行）：

| 字段 | 说明 |
|---|---|
| `event_id` | `sha1(uid_hash + event_type + source_id)`，幂等键 |
| `uid_hash` | 用户主键 |
| `event_type` | `review / post / favorite_app / favorite_moment / favorite_collection / favorite_hashtag / favorite_event / wishlist / badge / follow_user / follow_app / follow_hashtag / fan` |
| `event_ts` | epoch 秒；无可靠时间则 null |
| `time_kind` | `exact / approx / unknown` |
| `app_id` / `app_name` | 关联游戏（可空） |
| `target_id` | moment_id / badge_id / 用户哈希（可空） |
| `subtype` | review 的 score、follow 的 source、favorite 的 type 等 |
| `metrics` | 轻量数字快照 `{score, ups, supports, played_spent}` |
| `source_surface` | 来自哪个数据面（可回溯） |

**各面时间可得性（关键口径）**

| 面 | 时间字段 | `time_kind` |
|---|---|---|
| `feed_review` / `feed_moment` | `publish_time`（主）/ `created_time` / `commented_time` | `exact` |
| `wishlist` | `created_time` | `exact` |
| `badge` | `time`（Unix 秒） | `exact` |
| `favorite_*`（快照面） | 无动作时间 | `unknown` |
| `following_user` | `created_time`（**实测=关注时间**，见 §10 D1） | `exact` |
| `fans` | 无时间 | `unknown` |
| `following_app` / `following_hashtag` | 无时间 | `unknown` |

> `time_kind` 是给下游 Agent 的诚实标记：`unknown` 的事件**不参与时间序列/衰减计算**，但可参与静态计数。

---

## 5. 产出 schema 与落盘

```
data/processed/user_features/
├── features.csv            用户级特征表（主键 uid_hash，每用户一行）
├── features.parquet        同上（parquet 版，pandas/pyarrow 已就绪，见 §10 D2）
├── feature_dict.csv        特征字典：name / dim / type / desc / source / missing_policy
├── timeline.jsonl          行为时间线（一事件一行）
├── favorite_snapshot.jsonl 收藏快照表（不进时间线，见 §10 C）
└── _manifest.json          {feature_version, generated_at, n_users, input_fingerprint, params, truncation_summary, tag_vocab}
```

- `input_fingerprint = sha256(sorted(uid_hash + mtime + size))`：换输入即可判「是否需重跑」。
- `truncation_summary`：逐用户汇总各面是否触达 `max_pages=10`（截断标记），供下游判断「计数类特征是否可信」。
- 特征字典**自动从代码导出**，不手写，避免文档与实现漂移。

---

## 6. 口径与数据质量规则（坑清单）

| # | 坑 | 处理 |
|---|---|---|
| 1 | `app.title` 部分缺失（部分 app 只有 id） | 以 `app_id` 为主键；`app_name` 允许 null |
| 2 | 快照面（favorite/following_app）无动作时间 | `time_kind=unknown`，不进时间线（`favorite_*` 另落收藏快照表，见 §10 C） |
| 3 | `following_user.created_time` 曾被疑为「账号创建时间」 | ✅ 已实测定论为**关注动作时间**（78/78 跨列表互异，见 §10 D1），按 `exact` 使用 |
| 4 | 心愿单双口径（`len(wishlist)` vs `stat.app_wishlist_count`） | 两列并存，命名区分 `wishlist_count`(A) / `want_app_count`(B)，差异记为质量指标 |
| 5 | `show_app_wishlist=false` → wishlist 403 | 记为「隐私锁定」而非采集失败；`wishlist_locked` bool 特征；不重试不绕过 |
| 6 | 重度用户触达 `max_pages` 截断 | `_manifest.truncation_summary` 标记；计数类特征标 `truncated=true` |
| 7 | `follow_source` / `verified` 大量为空 | 空值归 `unknown`，比例计算分母为「非空数」并同时输出覆盖率 |
| 8 | `gender` 大量为空 | 同上，不猜 |
| 9 | `moment.topic/group/labels` 为 null | 可空，不派生强制特征 |

---

## 7. PII 与合规（沿用项目三域纪律）

| 域 | 位置 | 允许内容 | 入库 |
|---|---|---|---|
| 原始域 | `data/raw/user_profile/*.json` | 明文标识、昵称、原文、`ip_location`/`device` | ❌ |
| 分析域 | `data/processed/user_features/` | 加盐哈希 `uid_hash` + 特征/事件（`ip_location`/`device`/`gender` 分级） | ✅ |
| 产出域 | 报告/看板 | 只有聚合（分布/占比） | ✅ |

- `ip_location` / `device` / `gender` 分敏感级：进特征表可，**出产出域必须转聚合**（分布，不落个体）。
- 时间线默认**不含原文**；如需文本，只留 `target_id` 引用，原文按 id 回查（reference, not copy）。
- `following_user.id` / `fans.id` 只用于现算交集计数，**不落哈希明文到产出**。

---

## 8. 版本 / 复现 / 断点续跑

- `FEATURE_VERSION`（`l2v1`，见 `feature_dict.py`）写入每行 / 每事件与 `_manifest`。
- **L2 为全量重算**：同输入 + 同版本 → 产出确定（词表排序、事件排序、基准时刻 `as_of = max(fetched_at)` 均已消除不确定性）；200 人级成本可忽略，不做逐 uid 断点。
  断点续跑由 L1 采集侧承担（`--force` / 跳过已采）。
- 换口径 = 改 `feature_dict.py` + 升 `FEATURE_VERSION` + 重跑；旧版本产物并存不覆盖。
- 输入审计：`errors` 非空的用户照常产出，`_manifest.users_with_errors` 记录人数；坏文件跳过并在 stderr 显式报告。

---

## 9. 与 Agent（harness）的接口

- L2 产出即 harness 的 facts 输入：`load_user_features()` 读 `features.csv` + `timeline.jsonl`。
- 对齐现有骨架的六个业务环节（[state.py](../harness/state.py) 注释）：`anomaly / cohort / cause / risk / intervention / experiment` 均可直接消费 `id_ / act_ / taste_ / rel_ / inf_` 五维标量。
- 锁数纪律：Agent 只见 L2 特征与聚合，不见 `data/raw/` 原文；需要原文时按 id 走工具检索。
- 显式降级：`_manifest` 缺 `truncation_summary` 或特征缺列 → 显式报「不可用：原因」，**绝不静默当 0**。

---

## 10. 已拍板决策（2026-10-04）

| # | 决策 | 结论 | 理由 / 代价 |
|---|---|---|---|
| D1 | `following_user.created_time` 语义 | ✅ **实测定论 = 关注动作时间**（`source=follow_time`），可从 `approx` 升级 | 决定性检验：78 个被多人关注的用户，其 `created_time` 在不同关注者列表中 **78/78 全部不同** → 排除「被关注者账号创建时间」，剩「关注动作发生时间」 |
| D2 | 产出格式 | ✅ CSV + parquet 双出 | pandas / pyarrow / numpy 均已就绪，无额外成本 |
| A | `tag_vector` | ✅ **A1 固定 top-N（N=20）+ 词表从 200 人频次生成后冻结** | 确定、可复现、跨批次可比；代价 = 长尾品类被截断、新品类进不来，需维护冻结词表 |
| B | `tenure_bucket` | ✅ **B1 等距按年**（`<1 / 1–3 / 3–5 / >5`） | 绝对语义、可解释、跨样本稳定；代价 = 本样本老兵扎堆、组内区分度弱 |
| C | `favorite_moment`（293 条/人、无时间） | ✅ **C3 单独落「收藏快照表」**，不进时间线 | 不污染时间线，同时保住收藏对象 / 关联游戏信息；代价 = 多一张表与 schema 维护 |
| D | 跨轨（内容锚点轨）数据 | ✅ **D1 暂不纳入**，只吃用户锚点轨 | L2 先纵向做深、闭环优先；代价 = 暂无「用户 × 内容」交叉特征，留 L3 按需接 |

**落地含义**
- `tenure_bucket` 分档常量：`<1y / 1–3y / 3–5y / >5y`（边界按 `tenure_days`：365 / 1095 / 1825）。
- `tag_vector`：固定 20 列 `tag_top01..tag_top20`，词表生成后写入 `_manifest.tag_vocab` 并冻结；换词表 = 升版本。
- 新增产物：`data/processed/user_features/favorite_snapshot.jsonl`（收藏快照表：`uid_hash / surface / app_id / target_id / title`，无时间）。
- `following_user.created_time` 在时间线中 `time_kind=exact`、`subtype=follow_time`。
- 延迟项（本轮未拍板、不影响实现）：`decay_ratio` / `content_per_year` 等派生量的分母口径——先按 §3 定义实现，视为 `l2v1` 的一部分，后续可升版本调整。

---

## 11. 目录与运行

```
L2_features/
├── README.md              本设计文档
├── feature_dict.py        特征字典（单一真源：列顺序 / 口径 / 导出）
├── extract_features.py    ① 用户级特征表（features.csv + parquet + feature_dict.csv + _manifest.json）
├── extract_timeline.py    ② 行为时间线 + 收藏快照表（timeline.jsonl + favorite_snapshot.jsonl）
└── tests/                 口径回归测试（待补：幂等 / 缺失 / 截断标记）
```

```bash
# 在仓库根目录执行（先①后②；②会向 _manifest.json 补充 timeline 段）
python L2_features/extract_features.py
python L2_features/extract_timeline.py

# 常用参数：--limit N 小样试跑；--in-dir/--out-dir 改路径；--no-parquet 跳过 parquet
```

---

## 12. 实测（2026-10-04，全量 200 用户）

| 指标 | 数值 |
|---|---|
| 特征表 | **200 行 × 99 列**（五维 79 列 + 品类向量 20 列），无重复 uid |
| 时间线 | **9521 事件**（exact 7571 / unknown 1950；unknown 全为 fans 无时间事件） |
| 收藏快照 | **2023 行**（favorite_app 1349 / favorite_moment 666 / hashtag 4 / event 4） |
| 事件类型分布 | follow_user 2505 · badge 2168 · fan 1950 · post 1727 · review 1134 · wishlist 37 |
| 截断用户 | **7**（following_app 3 / favorite_app 3 / following_user 2）——与 L1 收尾报告「残余缺口集中在 2 个关注 1000+ 的极端用户」一致 |
| 错误用户 | 7（wishlist 403 = 隐私锁定，`wishlist_locked` 已标记） |
| 资历分布 | `<1y` 36 · `1-3y` 48 · `3-5y` 56 · `>5y` 60 |
| 词表 top20 | 二次元 · 动作 · 角色扮演 · 单机 · 策略 · 休闲 · 冒险 · 养成 · 卡牌 · 多人联机 · 模拟 · 射击 · 开放世界 · 剧情 · 像素 · 美少女 · 模拟经营 · 高画质 · 益智 · 放置 |

**完整性抽检**：最重用户（14 面含 800 关注 / 474 粉丝 / 20 徽章 / 13 动态 / 12 评价 / 1 心愿）
共输出 **1320 事件，与逐面条数完全吻合**，无丢事件。

> ⚠️ 诚实边界：截断 7 人意味着其 `following_*` / `favorite_app` 计数类特征偏低（`truncated_surfaces` 已逐人标记）；
> 时间线对这批用户是「已采到的子集」而非全量——下游引用时须看该标记，**不得当全量使用**。

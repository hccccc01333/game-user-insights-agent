# 游戏社区用户行为洞察 Agent

> 用**公开可观测行为痕**（publicly observable behavioral traces）给游戏社区用户建模：
> 活跃度 / 兴趣迁移 / 流失风险 → 输出可引用的 `facts` 契约 → 交给自研 Agent Harness 编排干预决策。

这是一个端到端打通的用户行为洞察系统：数据管道 → 特征工程 → 建模洞察 → Agent 工程。
技术侧重 **Agent 工程**（自研 Harness、决策循环、工具 / 校验 / 状态契约的设计与取舍），
数据与 ML 作为其落地运行的支撑。

## 1. 项目回答的三个问题

| # | 问题 | 对应模块 |
|---|---|---|
| ① | 用户**现在处于什么状态**？ | L1 采集 → L2 特征 → L3 洞察（活跃度 / 兴趣迁移 / 流失风险） |
| ② | 怎么用 **Agent** 把洞察变成决策？ | [harness/](./harness/) 六环节闭环：异常 → 分群 → 归因 → 风险 → 干预 → 实验；真实主链（facts → 干预队列 + 实验分组）+ 离线评测沙箱（策略效果验证）双轨 |
| ③ | **如果干预**，会发生什么？ | 模拟环境轨（Roadmap S1–S5）：用户模拟器 → Uplift → Bandit → 接 Harness |

## 2. 一句话架构

```
真实数据轨（状态层）                          离线评测沙箱（策略效果验证，S1–S4）
┌──────────────────────────────┐             ┌──────────────────────────────────┐
│ L1 采集   TapTap 公开行为痕    │             │ S1 用户模拟器（生成器 + 响应函数） │
│ L2 特征   五维特征表 + 时间线   │             │ S2 Uplift 干预效果建模            │
│ L3 洞察   活跃 / 迁移 / 流失   │             │ S3 Bandit 策略学习                │
│          └→ facts.jsonl 契约  │             │ S4 接 harness 工具 + Critic       │
└───────────────┬───────────────┘             └───────────────┬──────────────────┘
                └──────────►  harness（自研 Agent Harness） ◄──┘
                    六环节：anomaly → cohort → cause → risk → intervention → experiment
```

**真实数据解决「用户现在是什么状态」，模拟环境解决「如果干预会发生什么」，Agent 把两者编排起来。**

分层承诺：L1 只存原始公开响应；L2 只压可复算特征与事件；L3 只出可引用的数字与标签（`facts.jsonl` 是 Agent 唯一入口）；
策略生成与决策循环归 harness。全程 deterministic + versioned（版本 / 锚点 / 输入指纹冻结在 `_manifest.json`，重跑逐字节一致）。

## 3. Quickstart

要求：Python ≥ 3.10；`pip install -r requirements.txt`

```bash
# 0) 演示 Agent Harness 主循环（离线：MockModel + 占位工具，无需数据）
python -m harness.loop

# 1) （可选）采集你自己的公开数据。需要浏览器里任取一个 TapTap webapiv2 请求的 X-UA，
#    与一个自定的哈希盐；两者通过环境变量或本地 .env 提供（不入库）：
#      TAPTAP_X_UA=xxx   TAPTAP_HASH_SALT=xxx
#    ① 种子池：从多款游戏评价里收公开用户 id（建议 5–8 款不同品类）
python L1_data_source/collectors/taptap/collect_seed_pool.py --app-ids <游戏id1>,<游戏id2>
#    ② 小批采集（先 5–10 人验证接口与限速，再日常慢跑）
python L1_data_source/collectors/taptap/run_batch.py --daily-users 10

# 2) 特征层（L2）：五维特征表 + 行为时间线
python L2_features/extract_features.py
python L2_features/extract_timeline.py

# 3) 洞察层（L3）：活跃度 / 兴趣迁移 / 流失风险 → facts 契约
python L3_insights/app_tag_lookup.py
python L3_insights/model_activity.py
python L3_insights/model_churn.py
python L3_insights/model_migration.py
python L3_insights/build_facts.py

# 4) 真实主链：L3 facts → Agent 六环节决策（干预队列 + 实验分组；需先跑 1–3）
python -m harness.run_real_agent --batch 50

# 5) （S1）用户模拟器：三臂干预实验 + 真值 CATE（离线可跑，不依赖任何数据文件）
python -m simulator.run_sim --no-fit --users 800     # 强制默认分布（公开仓路径）
python -m simulator.tests                            # 确定性 / 校准 / 边界测试

# 6) （S2）Uplift 干预效果建模：rct / full 双协议 + T-learner（离线可跑）
python -m uplift.run_uplift --no-fit --users 6000    # 训练 + 评测 + 落盘（默认双跑一致性校验）
python -m uplift.tests                               # 协议完整性 / 无泄漏 / 策略端点测试

# 7) （S3）Bandit 策略学习：LinUCB / Thompson + random / 固定臂 / oracle 参照（离线可跑）
python -m bandit.run_bandit --no-fit --users 6000    # 在线学习 + 审计冻结评测 + 落盘
python -m bandit.tests                               # 无泄漏 / 端点 / 学习信号测试

# 8) （S4）接 harness：Bandit 干预工具 + Critic 安全门，Agent 决策循环真调用（离线可跑）
python -m harness.run_agent --no-fit --users 6000    # 六环节 + 带/不带 Critic 双模式对照
python -m harness.tests                              # 零泄漏 / 门控 / 结构 / 真实主链 / 端到端测试

# 9) （S5）展示层：全链路产出汇总为静态自包含 HTML 报告（离线直开，含双跑校验）
python -m report.build_report                        # 生成 report/index.html + _manifest.json
python -m report.tests                               # 自包含 / 数字对源 / 隐私扫描测试
```

> L2 / L3 需要本地产出的原始数据（`data/raw/user_profile/`）。
> **本仓库不包含任何真实用户数据**：原始采集与加工产出均不进公开仓（隐私 + 平台条款），
> 只保留代码与聚合校准结果；S1 模拟器的合成数据也不随仓分发——任何人可用
> `python -m simulator.run_sim --no-fit --seed <种子>` 离线重生成（逐字节可复现）。

## 4. 实测结果（2026-10-04 批次：pilot 200 人）

| 环节 | 产出 | 关键数字 |
|---|---|---|
| L1 采集 | 每人 14 个公开数据面原始 JSON | 200 人，条目 ≈ 25,700；截断 7 人（已标记 `truncated_surfaces`） |
| L2 特征 | `features.csv`（200 × 99，五维 `id_/act_/taste_/rel_/inf_`）+ `timeline.jsonl` | 9,521 个行为事件（`time_kind = exact / approx / unknown`） |
| L3 洞察 | `facts.jsonl`（Agent 唯一入口） | 200 行；活跃度五档 49/50/50/30/21；流失档 active 23 · d30 20 · **d60 132** · d90 25 |
| 复现性 | 双跑逐字节一致；版本 / 锚点 / 输入指纹冻结 | `_manifest.json`（L2 / L3 各一份） |
| Harness | 六环节骨架 + facts 契约层 + **真实主链（facts→决策）** + **S4 真干预接入**（离线评测沙箱） | `python -m harness.run_real_agent`（200 人 facts：沉默占比 78.5%＝参考线 1.57 倍；人群 157 人；单批队列 50：control 43 / rec 7 / recall 0，剩 107 名额；实验 treatment 4 / holdout 3；双跑一致）/ `python -m harness.run_agent`（沙箱双模式演示：否决率 2.5%，审计得分 0.788 = oracle 的 78.8%） |
| S5 展示层 | [`report/index.html`](./report/index.html)（静态自包含，离线直开） | 9 节：真实轨 L1–L3 + S1–S4 关键数字 + 诚实边界；双跑逐字节一致（输入指纹冻结在 `_manifest.json`） |

## 5. 诚实边界（随结论一起引用）

- **样本**：200 人 pilot，来自「近期发表过评价 / 动态」的用户，`recency_days` 中位 81 天 → 结论不代表全站。
- **流失标签**：单快照、无纵向回访 → 命名 `observed_inactivity`（公开活动沉默代理），**不是**平台真实流失；
  阈值 `calibration_status = uncalibrated`（未经运营反馈校准）。
- **真实主链的干预臂**：来自启发式风险分档（`risk_band_heuristic_v0`，risk_score → recall / rec / control），
  **不是因果收益估计**；效果验证走离线评测沙箱（S1–S4）与实验框架，主指标（未来 30 天公开行为沉默率）待观察窗回填。
- **行为痕 ≠ 埋点**：只有发帖 / 评论 / 评分 / 收藏 / 关注 / 时间戳等公开痕迹；
  不含 App 埋点、曝光、停留、支付等内部数据——文中一律称「公开可观测行为痕」。
- **时间线**：`unknown` 时间的事件（如粉丝列表无时间戳）不参与任何时间序列 / 迁移计算。
- **合规**：只采集公开可见信息；限速，403/429 即停，不重试、不绕过权限；
  `ip_location` / `device` / `gender` 等只在聚合层出现；关注 / 粉丝的对方 id 只出计数。

## 6. 仓库结构

```
├── L1_data_source/collectors/taptap/   # 采集器（限速 / 断点续跑 / 403 不重试 / PII 哈希）
│   ├── crawl_user_profile.py           #   用户画像：14 个公开数据面
│   ├── collect_seed_pool.py            #   种子池：多游戏评价 → 公开用户 id 池
│   └── run_batch.py                    #   批次运行器：日预算慢跑 / 自动暂停 / 失败队列
├── L2_features/                        # 特征层：feature_dict 单一真源 + 特征表 + 时间线
├── L3_insights/                        # 洞察层：活跃度 M1 / 迁移 M2 / 流失 M3 → facts 契约
├── harness/                            # 自研 Agent Harness：六组件骨架 + 真实主链（facts→干预队列/实验分组）+ 沙箱接入层（bandit 工具 + Critic）
├── simulator/                          # S1 用户模拟器：合成人口 / 三臂效应 / 日步长配对环境 / 真值 CATE
├── uplift/                             # S2 Uplift：rct/full 双协议 + T-learner + 保真度/校准/策略价值评测
├── bandit/                             # S3 Bandit：部分反馈世界 + LinUCB/Thompson + 在线/审计两层评测
├── report/                             # S5 展示层：全链路产出 → 静态自包含 HTML 报告（零依赖 / 离线直开）
├── scripts/reddit_calibration.py      # Reddit 公开数据 → 模拟器参数校准（分布锚点，非主锚）
├── calibration/                       # 校准产出（聚合参数与报告；原始抓取不入库）
└── data/                              # 本地数据域（不入库）：raw / processed
```

## 7. 扩采节奏（到 1000–2000 人）

```bash
python L1_data_source/collectors/taptap/collect_seed_pool.py --app-ids <多款游戏>
python L1_data_source/collectors/taptap/run_batch.py --daily-users 20 --target-total 2000
```

- 每日小批 + 请求限速；**403/429 自动暂停整批**（wishlist 的 403 属用户隐私设置，不计入）。
- 台阶 **500 / 1000 / 2000 人**：跨台阶时运行器会提示——此时暂停扩采，重跑 L2 / L3，
  复盘截断率（`truncated_surfaces`）与分布漂移（锚点冻结，跨批可比）。

## 8. Roadmap（v2：模拟环境轨）

| 步 | 内容 | 状态 |
|---|---|---|
| v1 | 真实轨 L1–L3 + harness 契约层（本仓现状） | ✅ |
| S1 | 用户模拟器：合成人口 + 三臂效应（对照 / 个性化推荐 / 沉默召回）+ 配对奖励 + 真值 CATE | ✅ 本仓 [simulator/](./simulator/) |
| S2 | Uplift 干预效果建模（标签来自模拟器真值，避免循环论证） | ✅ 本仓 [uplift/](./uplift/) |
| S3 | Bandit 策略学习（LinUCB / Thompson；在线 + 审计两层评测） | ✅ 本仓 [bandit/](./bandit/) |
| S4 | 接入 harness：Bandit 干预工具 + Critic 安全门（批量分配 / 保守下界门 / 真圈人） | ✅ 本仓 [harness/](./harness/) |
| S5 | 展示层：全链路产出汇总为静态自包含 HTML 报告 | ✅ 本仓 [report/](./report/) |

## 9. 数据来源与边界（三类角色的分工）

| 来源 | 角色 | 边界 |
|---|---|---|
| 本仓 L1 采集（TapTap 公开行为痕） | 真实状态层的唯一数据来源 | 仅本地落盘，不入公开仓 |
| `calibration/`（Reddit 公开数据分布校准） | 模拟器参数的**弱参考锚**（发帖 / 评论频率、用户原型等） | 跨平台、跨口径，不作主锚；原始抓取不入库 |
| 论文口径引用（如 1-9-90 参与结构） | 机制设计的外部合理性检查 | 仅作量级参考 |

## License

[MIT](./LICENSE)
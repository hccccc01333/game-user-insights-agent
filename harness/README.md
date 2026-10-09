# S4 · 接入 harness：Bandit 干预工具 + Critic（让 Agent 决策循环真正调用）

> **边界一句话**：S3 的策略对象此前只活在离线评测脚本里，**S4 把它接进 harness 的决策循环**——六环节里的"干预"环节由 Agent 真调用：对已圈定人群批量分配触达臂（选臂 → 复核 → 结算 → 学习），Critic 作为安全门把关，收工前必须完成干预。同一套流程跑"带 / 不带 Critic"两条路径：一眼看出安全门的代价值不值。
>
> 为什么需要它：前三个 S 都在"离线账本"里做验证；harness 回答的是**工程问题**——策略怎么变成 Agent 能调用的工具、谁给乐观的 Actor 踩刹车、错误决策如何被校验拦住。
>
> 所以这一层 **不做**：换策略算法、换模拟口径、真实数据接入（演示全程离线合成世界，逐字节可复现）。

**本层决策（2026-10-05，已定稿）**
- **工具形态：批量分配**——`allocate_interventions(count)` 一次调用完成 count 人次完整链条，返回运营口径批次摘要（臂占比 / 净奖励 / 否决率 / 剩余名额）；逐人循环留在工具内部。
- **Critic 职责：保守下界门（LCB Gate）**——Critic 只用镜像观测流（不看真值、不看未观测臂）维护与 Actor 独立的后验，用"悲观估计"复核提议：所选臂连下界都不如对照 → 否决并降级 control（宁不打扰）。
- **演示口径：纯模拟世界**——S1 合成人口 + S3 部分反馈世界，离线可复现，不触任何真实数据。
- **真圈人**——cohort 不是写死的：从合成世界按可观测条件（`act_30d = 0 且 silence_days ≥ 90`）圈出沉默高风险人群，干预台只对该人群分配。

---

## 1. 数据流与模块

```
simulator.build_tables(...)   →   bandit.protocol.build_world(...)
        （S1 真值 CATE 表）            （部分反馈世界：observe 唯一反馈通道）
                │
                ▼
harness/  （六组件骨架 + S4 接入层）
        ├── state.py / registry.py / planner.py / model.py / verifier.py / loop.py
        │        v1 组件：决策信封 / 工具登记 / 前置条件 / 校验位 / 运行循环
        ├── bandit_tools.py  InterventionConsole：干预台（选臂→复核→结算→学习）+ 工具注册
        ├── critic.py        InterventionCritic：镜像后验 + 证据门槛 + 探索额度；
        │                    CriticVerifier：六环节字段契约 + 收工前干预确认
        ├── run_agent.py     六环节组装（真圈人）+ 带/不带 Critic 双模式 + CLI + 落盘
        ├── run_real_agent.py 真实主链（realv0）：facts → 干预队列 + 实验分组
        ├── tools/           8 业务工具（洞察链）+ InsightVerifier（业务断言）
        ├── run_insight_agent.py 洞察主链（insightv0）：模型驱动的动态决策循环
        ├── tracing.py       执行追踪旁路（耗时 / 用量；不进确定性轨迹）
        └── tests.py         零泄漏 / 门控行为 / 结构校验 / 前置条件 / 端到端确定性
                ▼
data/synthetic/harness/   （合成数据，gitignored；可随时按种子重生成）
    ├── s4_metrics.json   配置 / 世界口径 / Critic 口径 / 两模式汇总 / 审计对照
    ├── s4_batches.csv    逐人次分配日志（mode / proposed / final / gate / reward / oracle）
    ├── s4_trace.jsonl    两模式循环轨迹（每步：决策 → 观察 → 校验说明）
    └── _manifest.json    版本 / 模拟参数快照 / 输出指纹 / 双跑一致
```

**硬规则（都被 tests 测过）**
1. **零泄漏**：策略与 Critic 都只能通过 `world.observe` 拿到"所选臂"的净奖励；批次摘要**不含任何真值字段**（oracle 只进内部评测账本）。关键不变量测试：把"未观测臂"奖励全部篡改后同种子重跑，**决策逐值不变**。
2. **Critic 与 Actor 同源信息**：两者镜像同一份观测流，门 = 同一后验的悲观 / 乐观两个视角——叙事自洽，无第二数据源。
3. **收工把关**：`finish` 提名前必须有一次通过结构校验的干预批次（防"没做事就收工"）。
4. **deterministic**：同参同种子逐字节一致；CLI 默认**双跑一致性校验**，不一致拒绝落盘（退出码 3）。
5. 产出是**合成环境上的工程演示**，不是真实世界的策略结论。

---

## 2. 六环节流程（S4 版）

| # | 工具 | 本版实现口径 |
|---|---|---|
| 1 | `detect_anomaly` | 对照基线互动量的后半段 vs 前半段环比 + 人口沉默占比（合成人口口径） |
| 2 | `locate_cohort` | **真圈人**：按 `act_30d = 0 且 silence_days ≥ 90` 从在线池圈出人群，交给干预台 |
| 3 | `analyze_cause` | 人群状态中位数（沉默 / 资历 / 兴趣集中度） |
| 4 | `assess_risk` | 沉默时长分档（90–180 / 180–365 / 365+） |
| 5 | `allocate_interventions` | **真干预**：bandit 批量分配（含 Critic 安全门），可多次调用直至名额跑完 |
| 6 | `design_experiment` | 基于实际分配批次的 A/B 设计（分组 + 主指标 + 触达数） |

链路收窄：每步 `when` 前置条件串成链（异常 → 人群 → … → 干预 → 实验），名额跑完后干预工具自动退出候选；`MockModel` 从候选清单里挑"未用过"的工具，因此离线演示的步序天然就是六环节顺序。

---

## 3. Critic 门规则（本层的核心决策语义）

```
提议臂 == control            → 恒放行（不打扰无需复核）
n_obs[提议臂] < min_obs     → 价值门关闭（证据太薄，先看清）
lcb(提议臂) > lcb(control) → 放行（悲观估计也不亏于不触达）
否则若有本批探索额度        → 按额度放行（小额度试错）
额度用尽                    → 否决并降级 control（宁不打扰）
```

`lcb(臂) = μ̂ᵀx − β·√(xᵀA⁻¹x)`，镜像后验 `A = ridge·I + Σxxᵀ`、`b = Σrx`（与 Actor 同款岭回归，只吃 observe 流）。

三条设计线（都是踩过坑后定案的）：
- **证据门槛 `min_obs = 3`**：单次观测会让后验下界"过度自信"提前放行——每条臂至少被看过 3 次，价值门才生效（tests 锁死该行为）。
- **探索额度**：每批 `max(2, ⌈批 × 5%⌉)` 个试错名额——纯否决门会让被否决的臂永远拿不到观测，学习被"谨慎"饿死；额度按批重置。
- **否决统计**：`veto_count / veto_rate` 进批次摘要与循环轨迹——"带 / 不带 Critic"的对照，读的就是这个数。

---

## 4. 结果（fit 人口，6000 人 × 14 天，seed 13，在线 4200 / 审计 1800）

先说一句背景：六环节从 4200 名在线用户里圈出 **518 人**（12.3%）沉默高风险人群；本批名额 200（批 1 后剩 318 人可继续分配）。三臂真值均值 `control 0.000 / rec −0.359 / recall +1.459`。

| 模式 | 循环步数 | 分配人次 | 否决率 | 在线均值 | 在线 / oracle | 审计 / oracle | 探索 | 收工 |
|---|---|---|---|---|---|---|---|---|
| `with_critic` | 7 | 200 | **2.5%** | **+1.093** | **0.761** | **0.788** | 10 | ✅ |
| `no_critic` | 7 | 200 | 0% | +1.077 | 0.751 | 0.787 | 0 | ✅ |

**读出来的四件事**
1. **安全门的代价几乎为零**：6000 人批面下否决率只有 2.5%（5/200），在线均值反而略高（+1.093 vs +1.077）——被否决的都是"悲观估计也不如不打扰"的低确信用户；审计得分基本持平（0.788 vs 0.787），说明门没有伤害学到的知识。
2. **批 1（门最谨慎时）的结构**：`with_critic` 批内放行 162 个价值门 + 10 个探索 + 5 个否决；`no_critic` 无否决——对照可直接读数。
3. **圈人是真数据**：人群 518 人（`silence_median 94.1` 天、`tenure_median 1222` 天），批次名额与 `stream_remaining` 全部来自干预台账本；若名额超过人群规模会自动收窄（如 800 人小跑时批 1 只发 73 个名额给 73 人的人群）。
4. **小人口上门的成本更高**：800 人 no-fit 小跑时否决率 31.7%——人群小、探索观测少，门更长时间处于"证据不足"状态。这正是"带 / 不带 Critic"对照要暴露的取舍：人口越薄，谨慎越贵。

---

## 5. 运行与产出

```bash
# 在仓库根目录执行
python -m harness.run_agent                      # 默认：有 L2 用拟合人口 + 双跑校验
python -m harness.run_agent --no-fit --users 800 --batch 120   # 强制默认人口（离线快跑）
python -m harness.run_agent --policy thompson --min-obs 5      # 换策略 / 调门参数
python -m harness.tests                          # 测试（也可 pytest harness/tests.py）
```

CLI：`--users --days --window --seed --audit --policy --basis --batch --critic-beta --min-obs --explore-frac --explore-min --max-steps --out --features --no-fit --no-verify`。
退出码：参数无效 `2`；双跑不一致 `3`（拒绝落盘）。`--policy oracle` 显式拒绝（上界不可部署）。

产出（默认 `data/synthetic/harness/`，不入公开仓）：见 §1 文件清单；`_manifest.json` 冻结版本 / 模拟参数快照 / 世界口径 / Critic 口径 / 输出指纹与行数。

---

## 6. 与上下游的接口

| 上下游 | 用法 |
|---|---|
| S1 模拟器 | `build_tables` 提供合成人口与真值 CATE；`--no-fit` 路径不读任何本地文件 |
| S3 Bandit | `make_policy` 的策略对象（select/update）直接注册为干预工具的后端；`run_audit` 做冻结审计对照 |
| S2 Uplift | 预留：τ̂ 排序作为分配名额的离线先验（闭环学习留给后续版本） |
| tests | 零泄漏 / 门控 / 契约 / 端到端四组不变量，可作为后续版本回归基线 |

---

## 7. 洞察主链与 Agent 评测（P1 新增）

> 定位差别：`run_agent` / `run_real_agent` 是**固定工作流**（`when` 链驱动的确定性六环节）；`run_insight_agent` 是**模型驱动的动态决策**——工具调用路径不写死，由模型看目标、候选与观察逐轮决定继续 / 重试 / 改道 / 收工。

**8 个业务工具（`harness/tools/`）**：把 L1–L3 事实、沉默预测工件与离线沙箱封装成 Agent 可调用能力（参数 Schema + 权限串 + 口径诚实标注）。

| 工具 | 权限 | 说明 |
|---|---|---|
| `get_user_behavior` / `analyze_activity` / `analyze_interest_migration` | `facts:read` | 时间线只读摘要 / 活跃度画像 / 兴趣迁移信号（缺失显式 null） |
| `predict_silence_risk` | `model:predict` | 离线工件推理；不适用显式给原因（工件缺失则不注册） |
| `get_risk_cohort` | `facts:read` | 按档位 + 风险分圈人（产出确定性 cohort_id）；空结果显式报错 |
| `plan_intervention` | `intervention:plan` | 启发式分档队列（复用 `risk_band_heuristic_v0`；非因果收益估计） |
| `evaluate_strategy` | `sandbox:evaluate` | 离线合成沙箱审计评测（oracle 显式拒绝作为可部署策略） |
| `generate_insight_report` | `report:generate` | 汇总事实生成报告（证据字段 + 建议 + 口径 caveats） |

**决策循环（`insightv0`）**：目标 → 候选收窄（前置条件 + 权限）→ 模型决策（信封 `{"tool","args","reason","plan"}`）→ 执行（Schema 校验）→ `InsightVerifier` 业务断言 → 观察写回 → 下一轮。两条模型路径：`--model mock`（离线桩，同参双跑逐值对拍）与 `--model llm`（DeepSeek / OpenAI 兼容，`LLM_API_KEY`，超时退避重试 + 用量统计）。

**Agent 评测（`agent_eval/` v1）**：6 个场景（正常 / 数据缺失 / 工具超时 / 结果冲突 / 错误的风险解释 / 权限不足）× mock / llm 双模式；指标覆盖任务成功率、工具选择与参数准确率、步骤 · tokens · 耗时 · 成本、事实可靠性、错误恢复率、Critic 与权限拦截、稳定性（mock 双跑逐值一致）。

```bash
python -m harness.run_insight_agent                    # 洞察主链：真实工件 + mock 双跑
python -m harness.run_insight_agent --model llm        # 真实模型（需 LLM_API_KEY）
python -m agent_eval.regression                        # 6 场景 mock 回归（双跑对拍 + 硬期望门槛）
python -m agent_eval.regression --model llm --repeat 3 # 端到端能力评测（成功率的多次执行口径）
```

产出（`data/processed/harness_insight/` 与 `data/processed/agent_eval/`）：轨迹 / 产物 / 干预队列 / 报告 / 指标 / `_manifest.json`；仅 `uid_hash` 与派生字段，不含昵称、原文、ip/device。

---

## 8. 已知局限（诚实清单）

1. **两条链的模型口径不同**：六环节链（`run_agent` / `run_real_agent`）离线演示仍走 MockModel（挑未用过工具的规则模型，顺序由 `when` 链保证）；动态 LLM 决策在洞察主链（`run_insight_agent --model llm`）——真实模型的端到端评测需自备 `LLM_API_KEY`。
2. **校验深度分层**：`InsightVerifier` 已做业务断言（概率范围 / 分档闭合 / 口径矛盾 / 证据非空）；`CriticVerifier` 对干预批次做结构核对（计数求和 / 比例范围 / 字段契约），其余环节"结构完整即放行"。
3. **单批演示规模**：默认只跑批 1（200 人）就收工（`max_steps = 8`）；干预工具支持多批直至人群名额跑完（tests 覆盖），多批学习曲线留给后续版本。
4. **无疲劳、无延迟反馈**：沿用 S3 批量一次性口径；日粒度在线迭代是后续版本的事。
5. **无预算 / 公平约束**：每条臂成本已入净奖励，但无全局约束。
6. **合成世界**：所有数字只证明"工程链路可跑、安全门语义自洽"，不构成真实业务结论。
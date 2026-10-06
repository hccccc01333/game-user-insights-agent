"""harness —— 自研 Agent Harness。

这个包的定位：先把"组件的壳"和"最小可运行循环"搭出来，
每个组件的真实设计留在各文件的「待定设计点」里逐条定案；
S4 起，干预环节由"占位工具"升级为真调用（Bandit 策略 + Critic）。

组件地图（对应业务闭环：发现异常→定位人群→分析原因→识别风险→制定干预→设计实验）：

    state.py     AgentState    一次运行的全部状态（轨迹 + 产物）
    registry.py  ToolRegistry  Agent 可调用能力的登记处
    planner.py   Planner       每轮决定"下一步做什么"
    model.py     ModelAdapter  模型调用的统一入口（可 mock、可替换）
    verifier.py  Verifier      对每轮结果做校验与把关
    loop.py      run()         把以上组件串成的运行循环

S4 接入层（把 S1–S3 的模拟环境资产接进决策循环；定位 = **离线评测沙箱**：
回答"如果干预会发生什么"——策略效果在这里验证与对照，S1–S4 全程可复现）：

    critic.py      InterventionCritic / CriticVerifier  保守下界门 + 循环级校验
    bandit_tools.py InterventionConsole                 干预台：选臂→复核→结算→学习
    run_agent.py   六环节组装 + CLI（带 / 不带 Critic 双模式对照）
    tests.py       冒烟与一致性测试

真实主链（① L3 facts → Agent 决策：吃真实状态的只读快照，产出干预队列与
实验分组；干预臂为启发式风险分档，非因果估计）：

    run_real_agent.py  facts 六环节工具 + 干预规划 + RealChainVerifier + CLI（realv0）

运行演示（在仓库根目录执行）：
    python -m harness.loop            # v1 骨架：占位工具 + MockModel
    python -m harness.run_agent       # 沙箱：合成世界 + Bandit 干预工具 + Critic
    python -m harness.run_real_agent  # 真实主链：L3 facts → 干预队列 + 实验分组
    python -m harness.tests           # 测试
"""

# S4 版本：换口径（工具批量口径 / Critic 规则 / 六环节演示流程）必须升版本
HARNESS_VERSION = "s4v0"

__all__ = ["HARNESS_VERSION"]
"""harness —— 自研 Agent Harness（薄骨架）。

这个包的定位：先把"组件的壳"和"最小可运行循环"搭出来，
每个组件的真实设计留在各文件的「待定设计点」里逐条定案。

组件地图（对应业务闭环：发现异常→定位人群→分析原因→识别风险→制定干预→设计实验）：

    state.py     AgentState    一次运行的全部状态（轨迹 + 产物）
    registry.py  ToolRegistry  Agent 可调用能力的登记处
    planner.py   Planner       每轮决定"下一步做什么"
    model.py     ModelAdapter  模型调用的统一入口（可 mock、可替换）
    verifier.py  Verifier      对每轮结果做校验与把关
    loop.py      run()         把以上组件串成的运行循环

运行演示（在仓库根目录执行）：
    python -m harness.loop
"""
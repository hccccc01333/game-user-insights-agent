# -*- coding: utf-8 -*-
"""S1 用户模拟器包：合成人口 → 三臂干预效应 → 配对奖励（真值 CATE 的沙盒）。

模块分工（改任何口径，先看 params.py 的注释与 README 的"证据分级"表）：
    params.py         参数与口径单一真源（锚点 / 设计取值 / 派生值）
    user_generator.py 合成人口：L2 分位数秩耦合拟合 + 内置默认分布兜底
    response.py       效应模型：三臂真实 CATE（异质 + 疲劳 + 个体噪声）
    env.py            日步长环境：配对 rollout（共同随机数）
    reward.py         奖励契约：窗内增量 − 触达成本
    run_sim.py        CLI：跑实验、落盘、写 _manifest.json、双跑一致性校验

边界：产出全部为合成数据（uid 前缀 sim_），不包含任何真实用户记录。
"""
from .params import SIM_VERSION

__all__ = ["SIM_VERSION"]
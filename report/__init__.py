"""report —— S5 展示层：把全链路落盘产出汇总为一张静态自包含 HTML 报告。

这个包的定位：不改模型、不重跑任何环节，只把两条轨的产出读成**聚合统计**，
渲染成零依赖、零 CDN、离线直开的报告——真实轨（L1–L3）回答"用户现在什么
状态"，模拟轨（S1–S4）回答"干预下去会发生什么、怎么决定、谁踩刹车"。

    build_report.py  数据收集（各模块 _manifest / metrics / summary / curve / trace）
                     → inline SVG 渲染 → report/index.html + _manifest.json
    tests.py         双跑一致 / 自包含 / 节完整 / 数字对源 / 真实轨缺失跳过 / 隐私扫描
    index.html       产出（提交进仓；仅聚合数字，无逐用户隐私数据）

运行（在仓库根目录执行）：
    python -m report.build_report     # 生成报告（含双跑一致性校验）
    python -m report.tests            # 测试
"""

# S5 版本：换报告结构 / 图表口径 / 数据来源必须升版本
REPORT_VERSION = "s5v0"

__all__ = ["REPORT_VERSION"]
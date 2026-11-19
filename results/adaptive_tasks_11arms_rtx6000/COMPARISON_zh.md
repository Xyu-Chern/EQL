# 调参后的固定 EQL、Adaptive EQL 与在线 SAC-N

各格为最终评估回报的均值 ± 跨 seed 样本标准差。使用 seeds 201、202、203、204，每个 seed 训练 100,000 步、更新 9,796 次，最后评估 128 个 episode。各方法均用共享 replay 和 γ=0.98。EQL、Adaptive 和历史 SAC 使用 10 个 critic，A40 对照使用 10 或 100 个 critic。SAC 和 Adaptive 的学习率均为 0.001。

| 环境 | 固定 EQL（τ_Q, τ_a） | 调参后固定 EQL | 历史 SAC（N=10） | 固定 EQL − 历史 SAC（胜出 seeds） | 五档 Adaptive | 11 档 Adaptive | 在线 SAC-N（A40，N=10） | 在线 SAC-N（A40，N=100） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pendulum-v1 | (0.3, 0.7) | −151.72 ± 11.96 | −185.05 ± 37.71 | +33.33（4/4） | −153.24 ± 12.89 | 本轮未训练 | −185.05 ± 37.71 | −571.66 ± 378.89 |
| Reacher-misc | (0.1, 0.7) | 19.52 ± 0.88 | 46.43 ± 1.62 | −26.90（0/4） | 36.07 ± 4.20 | 37.24 ± 4.22 | 46.43 ± 1.62 | 36.72 ± 2.95 |
| Hopper-v5 | (0.3, 0.7) | 388.16 ± 29.27 | 341.23 ± 40.20 | +46.93（3/4） | 302.80 ± 51.04 | 337.39 ± 126.81 | 329.46 ± 51.46 | 279.92 ± 9.70 |
| Walker2d-v5 | (0.3, 0.5) | 555.00 ± 105.43 | 389.76 ± 33.52 | +165.24（4/4） | 317.57 ± 103.98 | 333.78 ± 65.90 | 404.89 ± 47.12 | 456.97 ± 32.20 |

Pendulum 采用后续同学习率 0.001 的九组 τ 搜索。表中配置在 seeds 201–204 上选出，并用这四个 seeds 报分。先前四环境搜索中，Pendulum 固定 EQL 的学习率是 0.0003，独立复测回报为 −584.67 ± 170.24。另三个环境表中的固定 EQL 均使用学习率 0.0003，先用 seeds 101、102 选 τ，再用 201–204 独立复测。[Pendulum 九组搜索](../pendulum_tau_grid/RESULTS_zh.md) · [四环境独立复测](../search_four/RESULTS_zh.md) · [所有方法逐 seed 对齐数据](all_methods_paired.csv)

11 档相对五档 Adaptive 的平均差依次为 Reacher +1.17（3/4）、Hopper +34.59（2/4）、Walker2d +16.21（2/4）。

三个环境的五档和 11 档使用同一张 Quadro RTX 6000 和相同的训练与评估程序。算法的唯一改动是将 actor expectile 候选从五档改为 0、0.1、…、1.0；两版均以 τ_Q=0.5 开始。Hopper 的 seed 203 从 361.77 升到 522.04，对平均差的贡献最大。四个 seeds 中仅两个提高。

历史 SAC 列取自四环境调参后的独立复测。每个环境都用搜索 seeds 101、102 选中学习率 0.001，再用 seeds 201–204 复测。已核对 16 份原始 JSON 的预算、评估回合数与平均回报。历史 SAC 和 A40 重跑使用同一个 10-critic 算法。当前归档中尚未找到四环境、同预算、同 seeds 的双 critic SAC 结果。Hopper 的 11 档均值比此次 A40 重跑高 7.94，比此前 SAC-10 低 3.84。[历史 SAC 逐 seed 记录](historical_sac10.csv) · [历史 SAC 实验协议](../search_four/protocol.json)

原始 JSON、逐 seed 模型与协议保存在各方法的环境子目录中。[11 档汇总](summary.csv) · [逐 seed 配对](paired_seeds.csv) · [在线 SAC-N 汇总](../online_sacn10_a40/summary.csv) · [三环境五档汇总](../adaptive_tasks_rtx6000/summary.csv) · [Pendulum 五档汇总](../adaptive_pendulum/summary.json)

[单张 A40 的 N=10 和 N=100 对照及逐 seed 验证](../online_sacn100_a40/COMPARISON_zh.md)

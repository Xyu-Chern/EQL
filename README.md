# EQL 强化学习实验

本仓库收录 **4 个算法家族**。连续控制实验使用 EQL、SAC 和 CGPO，早期 CartPole 实验还使用了 DQN。固定或自适应 τ、不同 replay 分袋方式和 critic 数量均作为家族内的配置统计。

| 算法家族 | 本仓库的实现与记录 |
|---|---|
| EQL | [固定 expectile](eql/algorithms/eql.py)、[分袋 replay](eql/algorithms/original_bag.py)、[自适应 τ](eql/algorithms/adaptive.py) 和 [仅校准 critic 的变体](eql/algorithms/calibration.py) |
| SAC | [实现](eql/algorithms/sac.py)和[在线 SAC-N 实验入口](experiments/sac_n.py)。Walker2d 复测包含 SAC-2 和 SAC-10，百万步主对照包含 SAC-10 和 SAC-100。 |
| CGPO | [训练入口](experiments/cgpo.py)、[集群实跑代码](provenance/cgpo_a100_v3/src/)和[两个环境的阶段性结果](results/cgpo_a100_v3/) |
| DQN | [早期 Rejax 入口](experiments/rejax_baseline.py)和[CartPole 回报曲线](figures/legacy/cartpole_dqn_returns.png) |

## 代码入口

| 需要做什么 | 入口 |
|---|---|
| 修改 EQL、SAC 或分袋变体 | [算法实现](eql/algorithms/)、[expectile 聚合](eql/expectiles.py)、[replay buffer](eql/replay.py) |
| 修改原生 MuJoCo 交互、评估或默认训练参数 | [环境适配与评估](eql/native.py)、[共享训练配置](eql/training.py) |
| 启动训练或复跑对照 | [实验入口](experiments/)，运行 `python -m experiments --help` 列出协议，运行 `python -m experiments walker_calibration --help` 查看参数。 |
| 汇总已有结果、重评或绘图 | [分析模块](analysis/) |
| 验证算法行为 | [测试](tests/)，从仓库根目录运行 `python -m pytest -q`。 |

从仓库根目录、已安装项目依赖的 Linux 或 WSL 环境运行。下面的命令在新目录保存一个 Pendulum 对照种子的结果，使用 GPU。

```bash
python -m experiments baseline --env Pendulum-v1 --method EQL --seeds 0 --output runs/pendulum-eql
```

百万步训练可用 `python -m experiments million --help` 查看环境、方法和输出参数。历史训练使用的代码版本与日志见 [provenance](provenance/)，对应的输出见 [results](results/)。CGPO 的入口复用归档的训练程序，需要其 PyTorch、MuJoCo v3 和 A100 环境。

## 实验清单

截至 2026-11-20，`results/` 中有 **18 组实验与分析**。每行对应一个结果目录。其中部分组包含多轮训练或重评。历史权重备份单独保存，早期仅有曲线的演示另列于[历史图表](figures/legacy/)。

| 结果目录 | 实验内容与规模 |
|---|---|
| [online_v1](results/online_v1/) | 分袋与 target 来源的四组实现对照，2 个环境 × 4 组 × 3 个种子。 |
| [original_comparison](results/original_comparison/) | 原配置的 EQL、分袋 EQL、SAC 基线，2 个环境 × 3 种方法 × 4 个种子。 |
| [search_four](results/search_four/) | 四环境超参搜索 72 次，选定配置后在独立种子上复测 48 次。 |
| [pendulum_sac_endpoint](results/pendulum_sac_endpoint/) | 在 Pendulum 上用 4 个种子比较两种零 expectile 端点处理，并与已有 SAC 对照。 |
| [pendulum_tau_grid](results/pendulum_tau_grid/) | 固定学习率搜索 9 组 τ，两个 EQL 版本共 72 条记录，含 4 条复用。 |
| [pendulum_capacity](results/pendulum_capacity/) | 3 种方法 × 2 个 critic 规模，测量显存和更新能力。 |
| [pendulum_episode_bags](results/pendulum_episode_bags/) | 整回合分袋的 4 次新训练，与已有的一步分袋、共享 replay 对照。 |
| [pendulum_critic_errors](results/pendulum_critic_errors/) | 3 个规模 × 3 种 replay × 4 个种子，比较 critic 误差和相关性。 |
| [adaptive_pendulum](results/adaptive_pendulum/) | 五档自适应 EQL 在 Pendulum 上的 4 个种子。 |
| [adaptive_tasks_rtx6000](results/adaptive_tasks_rtx6000/) | 五档自适应 EQL，另 3 个环境各 4 个种子。 |
| [adaptive_tasks_11arms_rtx6000](results/adaptive_tasks_11arms_rtx6000/) | 十一档自适应 EQL，3 个环境各 4 个种子。 |
| [online_sacn10_a40](results/online_sacn10_a40/) | 在线 SAC-10，4 个环境各 4 个种子。 |
| [online_sacn100_a40](results/online_sacn100_a40/) | 在线 SAC-100，同一张 A40 上与 SAC-10 配对。 |
| [ensemble_sacn_diagnostic](results/ensemble_sacn_diagnostic/) | 在已有模型上诊断 ensemble 的悲观聚合与动作价值差距。 |
| [million_hopper_walker](results/million_hopper_walker/) | Hopper-v5、Walker2d-v5 百万步续训的阶段记录。 |
| [million_hopper_walker_v3](results/million_hopper_walker_v3/) | 两个 v3 环境各跑 5 种配置、4 个种子，共 40 个百万步终点。相同权重的[确定性重评](results/million_hopper_walker_v3/deterministic_eval/)归在本组。 |
| [cgpo_a100_v3](results/cgpo_a100_v3/) | CGPO 的 seed 201 本地快照覆盖 Hopper-v3 至 245,000 步、Walker2d-v3 至 232,000 步，两者均有 200,000 步配对评估。 |
| [walker_calibration](results/walker_calibration/RESULTS_zh.md) | Walker2d-v5 仅校准 critic 的复测，8 个配置 × 4 个新种子，共 32 个十万步终点。 |

## 报告与复现

- [百万步主对照](results/million_hopper_walker_v3/README.md)给出 SAC-10、固定 EQL、自适应 EQL、一步分袋 EQL 和 SAC-100 的逐环境结果。
- [11 月 18 日实验报告](实验汇编/完整实验报告.md)与[早期九章实验](实验汇编/README.md)保存当时的结果和统计口径。
- [集群实跑代码与日志](provenance/)、[实验文件 SHA-256 清单](实验汇编/文件清单.csv)和[历史权重备份](results/million_hopper_walker_v3_backups/)用于核验。清单记录此版本 Git 对象中 `results/` 与 `provenance/` 的文件字节。
- [大型续训状态恢复说明](大文件/README.md)提供 Release 下载地址和解包校验方法。

# EQL、SAC 与 CGPO 连续控制实验

[打开完整实验报告](实验汇编/完整实验报告.md) · [查看早期九章实验](实验汇编/README.md) · [浏览逐种子原始数据](results/) · [下载续训状态](https://github.com/Xyu-Chern/EQL/releases/tag/rl-archive-2026-11-18-v2)

| 想看什么 | 从这里开始 |
|---|---|
| 两个 MuJoCo v3 环境的百万步结果 | [五种方法的学习曲线与终点](results/million_hopper_walker_v3/README.md) |
| 同一批策略改用确定性动作后能跑多少 | [确定性重评](results/million_hopper_walker_v3/deterministic_eval/README.md) |
| 自适应 τ 与固定 τ 的差异 | [四环境配对结果](results/adaptive_tasks_11arms_rtx6000/COMPARISON_zh.md) |
| 在线 SAC 增加 critic 的效果 | [同一张 A40 的 N=10、100 对照](results/online_sacn100_a40/COMPARISON_zh.md) |
| CGPO 的当前步数与原生评估 | [按环境和 seed 查看](results/cgpo_a100_v3/) |
| 复跑或核验 | [集群实跑代码和训练日志](provenance/) · [结果文件 SHA-256](实验汇编/文件清单.csv) · [大文件恢复说明](大文件/README.md) |

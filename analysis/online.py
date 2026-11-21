"""Seed-level descriptive summaries; no episodes-as-seeds or best-seed selection."""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t


def main():
    root=Path('results/online_v1')
    groups={'A':'Shared data / shared target','B':'Bags / shared target',
            'C':'Shared data / own target','D':'Bags / own target'}
    rows=[]
    effects=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,env in zip(axes,['Pendulum-v1','Reacher-misc']):
        final={}
        initial={}
        for g,label in groups.items():
            files=[root/env/f'{g}_seed{s}.json' for s in [0,1,2]]
            records=[json.loads(p.read_text()) for p in files]
            assert all(r['records'][-1]['step']==65536 for r in records)
            assert all(r['records'][-1]['updates']==16128 for r in records)
            curves=np.array([[x['return_mean'] for x in r['records']] for r in records])
            raw=np.stack([np.load(root/env/f'{g}_seed{s}.npz')['returns'] for s in [0,1,2]])
            assert np.isfinite(raw).all()
            np.testing.assert_allclose(raw.mean(axis=-1),curves,atol=1e-4)
            final[g]=curves[:,-1]
            initial[g]=curves[:,0]
            steps=np.array([x['step'] for x in records[0]['records']])
            mean=curves.mean(axis=0)
            sd=curves.std(axis=0,ddof=1)
            ax.plot(steps,mean,label=g+' '+label)
            ax.fill_between(steps,mean-sd,mean+sd,alpha=0.12)
            auc=np.trapz(curves,steps,axis=1)/(steps[-1]-steps[0])
            last=np.array([r['records'][-1]['disagreement'] for r in records])
            rows.append(dict(env=env,group=g,mean=float(final[g].mean()),sd=float(final[g].std(ddof=1)),
                seed0=float(final[g][0]),seed1=float(final[g][1]),seed2=float(final[g][2]),
                auc_mean=float(auc.mean()),disagreement_mean=float(last.mean()),
                training_seconds_mean=float(np.mean([r['training_seconds'] for r in records])),
                wall_seconds_total=float(sum(r['wall_seconds'] for r in records))))
        for g in ['B','C','D']:
            np.testing.assert_allclose(initial[g],initial['A'],atol=1e-5)
        for name,diff in [('B-A',final['B']-final['A']),('D-C',final['D']-final['C']),
                          ('C-A',final['C']-final['A']),('D-B',final['D']-final['B']),
                          ('interaction',(final['D']-final['C'])-(final['B']-final['A']))]:
            half=float(t.ppf(.975,2)*diff.std(ddof=1)/np.sqrt(3))
            effects.append(dict(env=env,contrast=name,mean=float(diff.mean()),
                                ci95_low=float(diff.mean()-half),ci95_high=float(diff.mean()+half),
                                paired_seed_differences=diff.tolist()))
        ax.set(title=env,xlabel='Environment transitions',ylabel='Evaluation return (higher is better)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Online EQL: 4 critics, 3 seeds/group; shading = +/- 1 seed SD',fontsize=12)
    fig.tight_layout()
    fig.savefig(root/'learning_curves.png',dpi=170)
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    telemetry=[]
    for file in root.glob('gpu_telemetry*.csv'):
        with file.open() as f:
            telemetry.extend(csv.DictReader(f))
    gpu=dict(peak_total_device_memory_MiB=max(float(r['memory_used_MiB_total_device']) for r in telemetry),
             peak_temperature_C=max(float(r['temperature_C']) for r in telemetry),
             sampled_seconds=max(float(r['unix_time']) for r in telemetry)-min(float(r['unix_time']) for r in telemetry),
             note='Total GPU memory includes Windows desktop and other processes; sampled every 5 seconds, not exact allocator peak.')
    summary=dict(rows=rows,paired_effects=effects,gpu=gpu,
                 total_run_wall_seconds=sum(r['wall_seconds_total'] for r in rows),
                 validation='24 runs present; 65536 steps and 16128 updates each; raw returns finite and match logs; same-seed initial evaluations match across groups.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 在线分袋 replay buffer 首轮实验结果','',
           '日期：2026-11-18。实际运行后汇总，非预测结果。', '',
           '## 环境与实现','',
           '- WSL2 已恢复；新建 Linux 环境 `/root/.venvs/eql-gpu`。JAX 0.4.38 在 `cuda:0` 上完成编译计算核验。',
           '- GPU：RTX 3070 Ti Laptop，8192 MiB。旧 Windows `.venv` 保留，未用于本轮训练。',
           '- 分袋版本为每环境 8 步 rollout fragment 整体随机分配，不是跨 batch 保持成员关系的完整 episode bootstrap。',
           '- 每片段分配给 1 个 critic；buffer 同时支持复制到多个不同 critic 和 replace 模式，主实验使用环形 replay。',
           '- 16 项 buffer/expectile/训练接入测试通过。',
           '- 四组均为修正后的同一个 Rejax SAC/EQL 骨架，A 不是未修改的原生 SAC；旧仓库截图不能直接作本表对照。','',
           '## 实验预算','',
           '每环境四组、每组 seeds 0/1/2；每次 65,536 次总环境交互、16,128 次网络更新。',
           '4 个并行环境、4 个 critic、每片段 8 步、每片段批次后 8 次更新；初始 1,024 次交互使用随机动作。',
           '每 critic batch=128，actor batch=128；两层 64 单元 MLP；gamma=0.99，Polyak=0.995，学习率 3e-4。',
           'tau_actor=tau_target=0.3，单动作采样。全局 replay=32,768 条；每袋=8,192 条；保留的交互时间窗口匹配。',
           '每 8,192 步评估一次，每次 64 个随机策略 episode，评估随机数与训练独立。',
           '最终回报取固定 65,536 步处；先在每个 seed 内对 episode 求均值，再跨 seed 汇总，不挑最好 seed 或最好时刻。','',
           '## 最终回报','',
           '以下 ± 为三个训练 seed 的样本标准差，不是置信区间。两个环境均为越高越好；负值不能取绝对值比较。','',
           '| 组别 | 数据 | target | Pendulum-v1 | Reacher-misc |',
           '|---|---|---|---|---|']
    for g in groups:
        vals=[next(r for r in rows if r['env']==e and r['group']==g) for e in ['Pendulum-v1','Reacher-misc']]
        data='全局共享' if g in ['A','C'] else '轨迹片段分袋'
        target='共享 expectile' if g in ['A','B'] else '自身 target critic'
        lines.append(f"| {g} | {data} | {target} | {vals[0]['mean']:.2f} ± {vals[0]['sd']:.2f} | {vals[1]['mean']:.2f} ± {vals[1]['sd']:.2f} |")
    lines += ['', '## 配对差异','',
              '按相同 seed 做差；95% 区间使用自由度 2 的配对 t 区间，样本很少且正态假设无法检验，不能视为稳健显著性证据。未做多重比较校正。','',
              '| 环境 | 比较 | 平均差 | 探索性 95% 区间 |','|---|---|---:|---|']
    for e in effects:
        lines.append(f"| {e['env']} | {e['contrast']} | {e['mean']:.2f} | [{e['ci95_low']:.2f}, {e['ci95_high']:.2f}] |")
    lines += ['', '## 学习过程与分歧','',
              'AUC 是按环境步数积分后除以总步数的平均学习曲线高度；分歧是在公共查询点上计算的 critic 输出标准差，不能冒充价值误差或校准质量。','',
              '| 环境 | 组别 | 平均曲线高度 | 末次平均分歧 | 纯训练秒/seed |','|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['env']} | {r['group']} | {r['auc_mean']:.2f} | {r['disagreement_mean']:.3f} | {r['training_seconds_mean']:.2f} |")
    lines += ['', '## 硬件与运行记录','',
              f"- 每 5 秒采样的整卡显存峰值：{gpu['peak_total_device_memory_MiB']:.0f} MiB / 8192 MiB；包括桌面和其他程序，不是训练独占显存，也不是瞬时精确峰值。",
              f"- 采样最高温度：{gpu['peak_temperature_C']:.0f}°C。",
              f"- 已完成 24 次训练的 run 内 wall time 合计 {summary['total_run_wall_seconds']:.1f} 秒，含各 run 的初始化/编译/评估/写盘；不含安装、测试、外层进程启动和同步卡住的排障时间。",
              '- 两次 JAX 就绪同步停顿的日志保留；改为逐数组同步、每组独立进程后继续。只重跑一个未完成的 seed，其已有评估点完全复现，没有按分数筛选。见 PROVENANCE.md。','',
              '## 结论边界','',
              '本轮可以验证实现能够训练，并提供这两个小型 Gymnax 状态输入任务上的四组描述性比较。',
              '不能据此证明完整 episode bagging 有效、真实 Q 偏差降低、普遍提升样本效率或达到 SOTA。',
              '没有扩大任务、种子和超参数范围；没有测真实 Q 或执行独立的误差校准实验。研究 insight 留给作者根据这些证据判断。','',
              '## 产物','',
              '- `summary.csv`：每组均值、seed 标准差、三个原始 seed 分数、曲线面积和计时。',
              '- `summary.json`：配对差异与硬件统计。',
              '- `learning_curves.png`：学习曲线，阴影为 ±1 个 seed 标准差。',
              '- 各环境目录：原始 JSON/JSONL、64 episode 级 NPZ、最终参数 checkpoint 与配置/source hash。',
              '- 复现命令与实现边界见[分袋实现与测试](../../实验汇编/附录/分袋实现与测试.md)。','']
    (root/'RESULTS_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()

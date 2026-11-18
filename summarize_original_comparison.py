import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/original_comparison')
envs=['Pendulum-v1','Reacher-misc']
methods=['SAC','EQL','EQL_bagging']
rows=[]; effects=[];wall=0.
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,env in zip(axes,envs):
    data={};initial={};hashes={}
    for method in methods:
        curves=[];seeds=[]
        manifest=json.loads((root/env/f'{method}_manifest.json').read_text())
        for name,digest in manifest['source_sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name
        for seed in range(4):
            stem=root/env/f'{method}_seed{seed}'
            run=json.loads(stem.with_suffix('.json').read_text())
            raw=np.load(stem.with_suffix('.npz'))
            assert np.isfinite(raw['returns']).all()
            steps=raw['steps']
            assert np.array_equal(steps,np.arange(0,100001,5000))
            assert raw['returns'].shape==(21,128)
            assert run['records'][-1]['step']==100000 and run['records'][-1]['updates']==9796
            means=raw['returns'].mean(axis=1)
            np.testing.assert_allclose(means,[r['return_mean'] for r in run['records']],rtol=1e-6)
            key=seed
            if key in initial:
                np.testing.assert_array_equal(raw['returns'][0],initial[key])
                assert run['initial_params_sha256']==hashes[key]
            else:initial[key]=raw['returns'][0];hashes[key]=run['initial_params_sha256']
            curves.append(means);seeds.append(float(means[-1]));wall+=run['wall_seconds']
        curves=np.asarray(curves);data[method]=np.asarray(seeds)
        mean=curves.mean(axis=0);sd=curves.std(axis=0,ddof=1)
        ax.plot(steps,mean,label=method.replace('_',' '));ax.fill_between(steps,mean-sd,mean+sd,alpha=.12)
        rows.append(dict(env=env,method=method,mean=float(np.mean(seeds)),sd=float(np.std(seeds,ddof=1)),
            seed0=seeds[0],seed1=seeds[1],seed2=seeds[2],seed3=seeds[3],
            auc_mean=float(np.trapz(curves,steps,axis=1).mean()/100000)))
    for left,right in [('EQL','SAC'),('EQL_bagging','EQL'),('EQL_bagging','SAC')]:
        d=data[left]-data[right];margin=3.182446305*d.std(ddof=1)/2
        effects.append(dict(env=env,contrast=f'{left} - {right}',mean=float(d.mean()),
            paired_seed_differences=d.tolist(),ci95=[float(d.mean()-margin),float(d.mean()+margin)]))
    ax.set(title=env,xlabel='Environment transitions',ylabel='Episode return (higher is better)')
    ax.grid(alpha=.2);ax.legend()
fig.suptitle('Original configuration: 10 critics, 4 seeds, 100,000 transitions; shading = ±1 seed SD')
fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=170);plt.close(fig)
with (root/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
gpu=list(csv.DictReader((root/'gpu_telemetry.csv').open()))
stats=dict(peak_sampled_memory_MiB=max(float(r['memory_used_MiB_total_device']) for r in gpu),
    peak_sampled_temperature_C=max(float(r['temperature_C']) for r in gpu))
result=dict(rows=rows,paired_effects=effects,gpu=stats,run_wall_seconds_total=wall,
    validation='24 runs; 100000 transitions and 9796 updates each; finite raw 128-episode returns; identical initial returns and parameters; source hashes match')
(root/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# 原配置三方法对比结果','',
'目标方法：EQL + replay buffer + bagging。以用户确认的 eql.py 为准，未修改原 eql.py/sac.py。',
'三个方法统一 10 环境、每环境每次 1 步、10 critics、100000 次交互、9796 次更新、4 个相同训练 keys。',
'学习率 3e-4、gamma=.98、Polyak=.99、batch=256、fill_buffer=2048、buffer_size=131072；EQL tau_a=.7/tau_Q=.3。',
'SAC 是用户 sac.py 的 10-critic min-SAC，不能标成标准双 critic SAC。',
'bagging 使用一步轨迹片段、每条数据复制到两个不同袋；每个 critic 只从自己的袋采样，target 使用完整 ensemble，actor 不变。',
'每袋容量保留 131072；并非完整 episode 分袋。具体配置来源和实现边界见 ../../ORIGINAL_COMPARISON.md。','',
'每个 seed 最终评估 128 个随机策略 episode；下表 ± 为四个训练 seeds 的样本标准差。固定最终 checkpoint，不挑最好结果。','',
'| 环境 | 方法 | 最终回报均值 ± SD | 四个 seed 的最终回报 |',
'|---|---|---:|---|']
for r in rows:
    scores=', '.join(f'{r[f"seed{i}"]:.2f}' for i in range(4))
    lines.append(f'| {r["env"]} | {r["method"]} | {r["mean"]:.2f} ± {r["sd"]:.2f} | {scores} |')
lines+=['','## 配对差异','',
'以下是按相同训练 key 配对的均值差和探索性 t 区间（df=3）；只有四个 seeds，未校正多重比较。','',
'| 环境 | 比较 | 均值差 | 探索性 95% 区间 |','|---|---|---:|---|']
for e in effects:lines.append(f'| {e["env"]} | {e["contrast"]} | {e["mean"]:.2f} | [{e["ci95"][0]:.2f}, {e["ci95"][1]:.2f}] |')
lines+=['','## 本轮观察','',
    'Pendulum：两个 EQL 版本在四个配对 seeds 上均优于 SAC。bagging 相对原 EQL 三次提升、一次下降，最终均值低 7.27；不能据此声称稳定提升。',
    'Reacher：SAC 在四个配对 seeds 上均优于两个 EQL 版本。bagging 相对原 EQL 两次提升、两次下降，最终均值高 0.30；改善幅度不足以支持稳定收益结论。',
    '本轮支持“分袋版已接入原 EQL 并完成同预算对比”，尚不支持“bagging 稳定改善原 EQL”。',
    '10 环境、每次采集后更新 1 次，是用户确认配置；相同总交互数下，更新次数约为原单环境设置的十分之一，不能直接拿本表和旧单环境图比较。']
lines+=['','## 核验与产物','',result['validation'],
    f'整卡显存采样峰值 {stats["peak_sampled_memory_MiB"]:.0f} MiB，最高温度 {stats["peak_sampled_temperature_C"]:.0f}°C（含桌面，5 秒采样）。',
    f'24 次运行内部耗时合计 {wall:.1f} 秒，包括各 run 的编译与评估，不包括安装/测试/外层进程启动。',
    '各环境目录含原始 JSON/JSONL、episode 级 NPZ、最终参数、manifest 和源码 SHA256；summary.csv/summary.json 为汇总。',
    '本轮保留原代码的 warmup policy、done/autoreset 语义和 IRLS 实现，未混用上一轮算法修正。',
    '结果仅覆盖这两个 Gymnax 环境及当前预算，不能直接推广到 MuJoCo Reacher 或完整 episode bagging。']
(root/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))

"""Audit all saved runs and summarize at the training-seed level."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from critic_error_study import ROOT, CLASSES, COUNTS, SEEDS, MC_REPS, evaluate_errors, make_panel

def write_csv(name, rows):
    with (ROOT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    rows=[];paired=[]; groups=[]
    text_labels={"shared":"不分袋","step":"一步分袋","episode":"完整 episode 分袋"}
    panel=make_panel()
    for count in COUNTS:
        data={}
        for method in CLASSES:
            records=[]
            for seed in SEEDS:
                folder=ROOT/str(count)/method/f"seed{seed}"
                train=json.loads((folder/"train.json").read_text())
                row=json.loads((folder/"metrics.json").read_text())
                assert hashlib.sha256((folder/"model.msgpack").read_bytes()).hexdigest()==train["model_sha256"]
                assert train["updates"]==9796 and train["steps"]==100000 and len(train["returns"])==128
                with np.load(folder/"diagnostics.npz") as d:
                    for p,v in zip(("theta","velocity","action"),panel):np.testing.assert_array_equal(d[p],v)
                    assert d["mc_soft_returns"].shape==(336,MC_REPS) and d["critic_q"].shape==(count,336)
                    metrics,arrays=evaluate_errors(d["critic_q"],d["mc_soft_returns"])
                    for k,v in metrics.items():np.testing.assert_allclose(row[k],v)
                    for k,v in arrays.items():np.testing.assert_allclose(d[k],v)
                    # Decompose same-direction mistakes into joint under/overestimates.
                    for name,mask in (("both_under",d["errors"] < 0),("both_over",d["errors"] > 0)):
                        n=mask.sum(axis=0)
                        row[name]=float(np.mean(n*(n-1)/(count*(count-1))))
                records.append(row);rows.append(row)
            data[method]=records
            item=dict(critics=count,method=method)
            for metric in ("correlation","raw_correlation","same_sign_fraction","both_under","both_over","mean_bias","critic_rmse","ensemble_mean_rmse","prediction_spread","median_mc_se","score"):
                col=[r[metric] for r in records]
                item[metric]=float(np.mean(col))
                item[metric+"_sd"]=float(np.std(col,ddof=1))
            groups.append(item)
        for m,base in (("step","shared"),("episode","shared"),("episode","step")):
            diffs=np.array([a["correlation"]-b["correlation"] for a,b in zip(data[m],data[base])])
            paired.append(dict(critics=count,comparison=f"{m} - {base}",mean_difference=diffs.mean(),
                               lower_seeds=int((diffs<0).sum()),**dict(zip([f"seed{s}" for s in SEEDS],diffs))))
    write_csv("by_seed.csv",rows);write_csv("summary.csv",groups);write_csv("paired.csv",paired)
    lines=["# Pendulum critic 误差相关性","",
           "对每个最终策略，在相同的 336 个 (s,a) 上强制执行首个动作，并用其冻结策略、冻结温度和 γ=0.98 向前模拟 200 步。测量目标是软策略 Q，熵项从第二个动作开始，log π 使用与训练相同的实现。角度等距取 16 个值，角速度取 −7、−4、−2、0、2、4、7，力矩取 −1.5、0、1.5。起始时钟均为零。","",
           "每个查询独立采样 256 次回报，分为两份 128 次的参考均值。对每个 critic，以预测减参考得到有符号误差；跨查询对误差中心化，计算跨两份参考的对称协方差，再归一化并平均非对角元素。这减少共用一个带噪标签造成的相关性上偏。普通 Pearson 和所有矩阵一并保存。同号率用全部 256 次的平均参考计算。","",
           "汇总先在每个训练 seed 内平均 critic 对，再对四个 seeds 等权平均。critic 对共享网络和数据；不把它们当作独立训练样本。10/20/50 个 critic 分别用 2/4/10 份复制，每袋仍覆盖 20% 交互。训练 100000 步、9796 次更新；学习率 0.001、τ_Q=0.1、τ_a=0.7。回报为此前协议下的普通 128 回合评估。","",
           "| critic 数 | replay | 误差相关 ± seed SD | 同号率 | 共同低估比例 | critic RMSE | 平均 Q 的 RMSE | 回报 ± seed SD |",
           "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in groups:
        lines.append(f"| {r['critics']} | {text_labels[r['method']]} | {r['correlation']:.4f} ± {r['correlation_sd']:.4f} | {r['same_sign_fraction']:.1%} | {r['both_under']:.1%} | {r['critic_rmse']:.2f} | {r['ensemble_mean_rmse']:.2f} | {r['score']:.2f} ± {r['score_sd']:.2f} |")
    lines+=["","## 各 seed 的配对相关性差值","","| critic 数 | 比较 | 201 | 202 | 203 | 204 | 均值 | 降低次数 |","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in paired:
        lines.append(f"| {p['critics']} | {p['comparison']} | "+" | ".join(f"{p[f'seed{s}']:+.5f}" for s in SEEDS)+f" | {p['mean_difference']:+.5f} | {p['lower_seeds']}/4 |")
    lines+=["","固定网格给每个状态动作相同权重，结果依赖这个分布；起始时钟为零，而训练观测不含时钟。各方法学习到的策略可能不同，相关性反映各自策略下的 critic 误差。扩大 ensemble 同时增加总计算量。","",
            "每个 seed 目录的 model.msgpack 包含 actor、在线与目标 critics、温度参数；train.json 含预算、配置、128 个回合回报和模型哈希；diagnostics.npz 含查询、全部原始模拟回报、Q 矩阵、逐点误差及两两相关矩阵；metrics.json 是单次运行的汇总。"]
    (ROOT/"RESULTS_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(1,3,figsize=(12,4.0),layout="constrained")
    for m,label in zip(CLASSES,["Shared","Step bags","Episode bags"]):
        sel=[g for g in groups if g["method"]==m]
        for ax,(var,title) in zip(axes,[("correlation","Critic error correlation"),("same_sign_fraction","Same-sign error fraction"),("score","Policy return")]):
            ax.errorbar([r["critics"] for r in sel],[r[var] for r in sel],yerr=[r[var+"_sd"] for r in sel],label=label,marker="o",capsize=3)
            ax.set(xlabel="Number of critics",title=title,xticks=COUNTS);ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(ROOT/"comparison.png",dpi=180)
    print("\n".join(lines[8:]),flush=True)
if __name__=="__main__":main()


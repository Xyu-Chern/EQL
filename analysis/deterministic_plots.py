"""Plot only the checkpoints actually re-evaluated with the deterministic actor."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from analysis.million_plots import METHODS, write_csv

TASKS=("Hopper-v3","Walker2d-v3")
SEEDS=(201,202,203,204)
SHORT=["SAC-10","EQL","Adaptive\nEQL","EQL +\nbagging","SAC-100"]


def plot(root):
    data={}
    for path in sorted((root/"scores").glob("*/*/seed*/points/*.json")):
        row=json.loads(path.read_bytes())
        key=(row["task"],row["method"],row["seed"],row["step"])
        assert path.parts[-5:-1]==(key[0],key[1],f"seed{key[2]}","points")
        assert key not in data and len(row["returns"])==len(row["lengths"])==128
        np.testing.assert_allclose([np.mean(row["returns"]),np.std(row["returns"])],
                                   [row["avg_ret"],row["std_ret"]])
        assert np.isfinite(row["returns"]).all() and all(1<=n<=1000 for n in row["lengths"])
        data[key]=row
    headers=["task","method","seed","step","avg_ret","std_ret","episodes","actor_sha256"]
    write_csv(root/"checkpoints.csv",headers,[[r[c] for c in headers] for r in data.values()])
    summary=[]
    with PdfPages(root/"deterministic_evaluation.pdf") as pdf:
        fig,axes=plt.subplots(1,2,figsize=(11,5.1),layout="constrained")
        for ax,task in zip(axes,TASKS):
            for i,(method,(_,color)) in enumerate(METHODS.items()):
                values=np.array([data[task,method,seed,1000000]["avg_ret"] for seed in SEEDS])
                mean,sd=values.mean(),values.std(ddof=1)
                ax.scatter(i+np.array([-.12,-.04,.04,.12]),values,alpha=.65,c=color,s=24)
                ax.errorbar(i,mean,yerr=sd,fmt="D",color=color,markeredgecolor="black",
                            markeredgewidth=.7,capsize=5,zorder=3)
                summary.append([task,method,*values,mean,sd])
            ax.set_title(task,loc="left",weight="bold")
            ax.set_xticks(range(len(METHODS)),SHORT)
            ax.set_xlim(-.5,len(METHODS)-.5)
            ax.set_ylabel("Average episode return")
            ax.grid(axis="y",alpha=.2)
            ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("Deterministic policies at 1 million environment steps",weight="bold")
        fig.supxlabel("Dots: individual seed means over 128 episodes. Diamonds and bars: mean ±1 sample SD across four seeds.",fontsize=9)
        fig.savefig(root/"deterministic_final.png",dpi=180)
        pdf.savefig(fig)
        plt.close(fig)
        for task in TASKS:
            fig,axes=plt.subplots(1,5,figsize=(14.5,4.7),sharey=True,layout="constrained")
            for ax,method,label in zip(axes,METHODS,SHORT):
                for j,seed in enumerate(SEEDS):
                    rows=sorted([r for k,r in data.items() if k[:3]==(task,method,seed)],key=lambda r:r["step"])
                    color=plt.get_cmap("tab10")(j)
                    x=np.array([r["step"] for r in rows])/1e6
                    y=[r["avg_ret"] for r in rows]
                    ax.errorbar(x,y,yerr=[r["std_ret"] for r in rows],fmt="none",
                                alpha=.28,color=color,lw=.9,capsize=2)
                    ax.scatter(x,y,marker=("o","s","^","x")[j],s=19,color=color)
                ax.set_title(label.replace("\n"," "))
                ax.set(xlim=(-.04,1.05),xticks=[0,.5,1])
                ax.grid(axis="y",alpha=.2)
                ax.spines[["top","right"]].set_visible(False)
            axes[0].set_ylabel("Average episode return")
            fig.legend(handles=[Line2D([],[],linestyle="none",marker=m,color=plt.get_cmap("tab10")(j),label=f"seed {s}")
                                for j,(m,s) in enumerate(zip(("o","s","^","x"),SEEDS))],
                       loc="outside upper center",ncols=4,frameon=False)
            fig.suptitle(f"{task} · available deterministic checkpoints",weight="bold")
            fig.supxlabel("Environment steps (millions). Each marker uses 128 episodes. Bars show ±1 episode SD. Missing weights have no point.",fontsize=9)
            fig.savefig(root/f"deterministic_checkpoints_{task}.png",dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    write_csv(root/"summary.csv",["task","method",*[f"seed{s}" for s in SEEDS],"mean","sample_sd"],summary)
    notes=["# v3 确定性策略重评","","## 100 万步终点","",
           "| 方法 | Hopper-v3 | Walker2d-v3 |","|---|---:|---:|"]
    for method,(label,_) in METHODS.items():
        vals=[r[-2:] for r in summary if r[1]==method]
        notes.append("| "+" | ".join([label,*[f"{m:.2f} ± {s:.2f}" for m,s in vals]])+" |")
    notes+=["","每个 seed 各执行 128 个完整 episode，表中 ± 为四个训练 seeds 之间的样本标准差。",
            "","![确定性策略终点对照](deterministic_final.png)","","## 可用中间点","",
            f"共重评 {len(data)} 个真实权重，其中 40 个是终点，其余 {len(data)-40} 个是中间点。图中每个标记对应一个 seed 的一个检查点，误差线是该策略在 128 个 episode 内的总体标准差。缺少权重的位置留空。",
            "","![Hopper 中间点](deterministic_checkpoints_Hopper-v3.png)","","![Walker2d 中间点](deterministic_checkpoints_Walker2d-v3.png)",
            "","## 评估口径","",
            "在 CPU 上读取 Rejax/Flax 的实际 actor 参数，以 tanh 变换后的高斯均值作为确定性动作，再裁剪到 [−1, 1]。直接使用训练所用的 Gymnasium 0.29.1 v3 环境和 MuJoCo 2.1.0。每个 episode 逐步执行，到 terminated 或 truncated 结束，累加未折扣的原始奖励。",
            "","沿用训练 seeds 201–204。每个 seed 经 NumPy master RNG 生成环境、动作和策略三个子种子，确定性动作无需使用策略种子。先给环境和动作空间设种子，再在 episode 开始时直接 reset()。每个名义评估点使用连续 128 次 reset 的初始状态。缺权重时只推进初始状态序列，从而保留各方法同一步数的配对。已逐 episode 校验该环境的 step 不消耗 reset 使用的随机数。",
            "","[全部评估点 CSV](checkpoints.csv) · [终点 CSV](summary.csv) · [图表 PDF](deterministic_evaluation.pdf)",
            "","每个 seed 目录中的 log.csv 使用 step、avg_ret、std_ret 三列，std_ret 按 128 个 episode 取 ddof=0。points 子目录保留逐 episode 的长度和回报。",""]
    (root/"README.md").write_text("\n".join(notes),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    plot(parser.parse_args().root)

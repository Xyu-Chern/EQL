"""Track author and matched CGPO evaluation beside the archived v3 experiments."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_million_native import METHODS


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def cgpo_series(ax,root,task,kind,label,color,style):
    all_seeds=[]
    for seed in (201,202,203,204):
        files=sorted((root/task/f"seed{seed}"/kind).glob("*.json"))
        if files:
            records=[json.loads(p.read_bytes()) for p in files]
            vals={int(r["step"]):r["avg_ret"] for r in records}
            ax.plot([s/1e6 for s in vals],list(vals.values()),color=color,ls=style,
                    marker=".",ms=3,lw=1.0,alpha=.56,label=f"{label} · {seed}")
            all_seeds.append(vals)
    if len(all_seeds)==4:
        steps=sorted(set.intersection(*(set(v) for v in all_seeds)))
        if steps:
            x=np.array(steps)/1e6
            y=np.array([[v[s] for s in steps] for v in all_seeds])
            mean,sd=y.mean(axis=0),y.std(axis=0,ddof=1)
            ax.plot(x,mean,color=color,ls=style,lw=2.5,label=f"{label} · 4-seed mean")
            ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.12)


def plot(cgpo,eql):
    previous=csv_rows(eql/"average_return.csv")
    det=csv_rows(eql/"deterministic_eval/checkpoints.csv")
    for task in ("Hopper-v3","Walker2d-v3"):
        fig,axes=plt.subplots(1,3,figsize=(15,5.4),layout="constrained")
        a,b,c=axes
        for method,(label,color) in METHODS.items():
            raw=[r for r in previous if r["task"]==task and r["method"]==method]
            a.plot([float(r["step"])/1e6 for r in raw],[float(r["mean"]) for r in raw],
                   color=color,lw=1.8,label=label)
            groups={}
            for row in det:
                if row["task"]==task and row["method"]==method:
                    groups.setdefault(int(row["step"]),[]).append(float(row["avg_ret"]))
            paired=[(s,np.mean(y)) for s,y in sorted(groups.items()) if len(y)==4]
            if paired:
                b.scatter([s/1e6 for s,_ in paired],[y for _,y in paired],s=30,
                          color=color,label=label,zorder=4)
        cgpo_series(b,cgpo,task,"eql_eval","CGPO matched","black","-")
        cgpo_series(c,cgpo,task,"native","CGPO author","#267BA5","-")
        cgpo_series(c,cgpo,task,"eql_eval","CGPO matched","#953979","--")
        for ax,title in zip(axes,["Previous online evaluation","Matched CPU evaluation","CGPO evaluation protocols"]):
            ax.set(xlim=(-.025,1.025),xlabel="Environment steps (millions)",title=title)
            ax.set_ylabel("Average episode return")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper center",bbox_to_anchor=(.5,-.17),frameon=False,fontsize=7,ncols=2)
        fig.suptitle(task,weight="bold")
        target=cgpo/f"cgpo_comparison_{task}.png"
        tmp=target.with_suffix(".tmp")
        fig.savefig(tmp,format="png",dpi=165)
        tmp.replace(target)
        plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("cgpo",type=Path)
    parser.add_argument("eql",type=Path)
    args=parser.parse_args()
    plot(args.cgpo,args.eql)

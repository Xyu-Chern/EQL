"""Verify and summarize the frozen search protocol from raw per-episode scores."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path("results/search_four")
    protocol=json.loads((root/"protocol.json").read_text())
    rows=[]
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,task in zip(axes,protocol["tasks"]):
        results={}
        for method in protocol["methods"]:
            folder=root/task/method
            selection=json.loads((folder/"selection.json").read_text())
            def load(phase,candidate,seed):
                record=json.loads((folder/f"{phase}_{candidate}_{seed}.json").read_text())
                assert (record["seed"],record["task"],record["method"],record["steps"],record["updates"])==(seed,task,method,100000,9796)
                returns=np.asarray(record["returns"])
                assert returns.shape==(128,) and np.isfinite(returns).all()
                np.testing.assert_allclose(returns.mean(),record["score"],rtol=1e-6)
                return record["score"]
            means=[np.mean([load("search",i,s) for s in protocol["search_seeds"]]) for i in range(3)]
            winner=int(np.argmax(means))
            np.testing.assert_allclose(means,selection["search_means"])
            assert winner==selection["winner"]
            scores=[load("confirm",winner,s) for s in protocol["confirmation_seeds"]]
            results[method]=np.array(scores)
            rows.append(dict(task=task,method=method,settings=json.dumps(selection["grid"][winner]),
                search_means=json.dumps(means),mean=np.mean(scores),sd=np.std(scores,ddof=1),scores=json.dumps(scores)))
        for j,s in enumerate(protocol["confirmation_seeds"]):
            ax.plot(range(3),[results[m][j] for m in protocol["methods"]],"o-",alpha=.4)
        ax.plot(range(3),[results[m].mean() for m in protocol["methods"]],"kD--",label="mean")
        ax.set_xticks(range(3),["SAC","EQL","EQL+bag"])
        ax.set_title(task);ax.grid(axis="y",alpha=.2)
        print(task,"paired bag-EQL",results["EQL_bagging"]-results["EQL"])
    axes[0].set_ylabel("Final episode return")
    fig.tight_layout();fig.savefig(root/"confirmation.png",dpi=160)
    with (root/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:print(row)

    lines=["# 四环境超参搜索与复测","",
           "每个任务、每种方法各用三个候选和两个搜索 seeds（101、102）。以 10 万步时 128 个随机策略评估回合的平均回报选出配置，再用独立训练 seeds 201–204 复测。每次训练含 9,796 次更新。", "",
           "| 环境 | 方法 | 选中参数 | 复测均值 ± seed 样本标准差 | 四个复测 seed 的回报 |",
           "|---|---|---|---:|---|"]
    for row in rows:
        scores=", ".join(f"{x:.2f}" for x in json.loads(row["scores"]))
        lines.append(f'| {row["task"]} | {row["method"]} | {row["settings"]} | {row["mean"]:.2f} ± {row["sd"]:.2f} | {scores} |')
    lines+=["","SAC 搜学习率 0.0003、0.0001、0.001。两个 EQL 都固定学习率 0.0003，依次搜索 (τ_actor, τ_Q) = (0.7,0.3)、(0.5,0.3)、(0.7,0.1)。分袋仍以单步为单位，每条进入两个不同袋，每个 critic 独立无放回采 256 条。actor 使用全局数据。","",
            "Pendulum-v1 和 Reacher-misc 使用 Gymnax。Hopper-v5 和 Walker2d-v5 使用 Gymnasium 1.4.0 和原生 MuJoCo 3.17.0。四种任务沿用上一轮的 done 掩码，对时间上限也停止 bootstrap。","",
            "各方法的搜索次数相同，SAC 与 EQL 搜索的参数种类不同。选中不同参数的两个 EQL 之间的差异包含调参和分袋两部分。两组共用同一参数时可以直接比较该参数下的分袋效果。每项仅有两个搜索 seeds 和四个复测 seeds。"]
    (root/"RESULTS_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()

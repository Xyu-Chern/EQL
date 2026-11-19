"""Validate the two complete experiments and plot paired-seed evaluation returns."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import numpy as np


METHODS = {"SAC10": ("SAC (10 critics)", "#0072B2"),
           "EQL": ("Tuned EQL", "#D55E00"),
           "Adaptive11": ("Adaptive EQL (11 arms)", "#009E73"),
           "EQL_bagging": ("Tuned EQL + bagging", "#AA4499"),
           "SAC100": ("SAC (100 critics)", "#5B5350")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def plot_progress(root):
    """Show each available seed independently while the four-seed groups fill."""
    for task in ("Hopper-v5", "Walker2d-v5"):
        fig, axes = plt.subplots(1, len(METHODS), figsize=(16, 4), sharey=True,
                                 layout="constrained")
        for ax, (method, (label, _)) in zip(axes, METHODS.items()):
            for seed in (201, 202, 203, 204):
                path = root/task/method/f"seed{seed}"/"curve.json"
                if path.exists():
                    records = read(path)["records"]
                    ax.plot([r["step"]/1e6 for r in records],
                            [r["mean_return"] for r in records],
                            color=plt.get_cmap("tab10")(seed-201), lw=1.5,
                            marker=".", ms=4, label=str(seed))
            ax.set(xlim=(-.02, 1.02), xticks=[0, .5, 1], title=label)
            ax.grid(axis="y", alpha=.23)
            if ax.lines:
                ax.legend(title="Seed", frameon=False, fontsize=8)
            else:
                ax.text(.5, .5, "Awaiting data", ha="center", transform=ax.transAxes)
        axes[0].set_ylabel("Average episode return (128 episodes)")
        fig.supxlabel("Environment steps (millions)")
        fig.suptitle(f"{task} | Training progress by seed")
        path = root/f"progress_{task}.png"
        tmp = path.with_suffix(".tmp")
        fig.savefig(tmp, format="png", dpi=125)
        tmp.replace(path)
        plt.close(fig)


def plot(root):
    steps = np.arange(0, 1000001, 50000)
    seeds = (201, 202, 203, 204)
    summary, curves = [], []
    with PdfPages(root/"average_return.pdf") as pdf:
        for task in ("Hopper-v5", "Walker2d-v5"):
            fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
            for method, (label, color) in METHODS.items():
                group = root/task/method
                protocol = read(group/"protocol.json")
                assert protocol["seeds"] == list(seeds) and protocol["eval_episodes"] == 128
                cfg = protocol["config"]
                assert cfg["total_timesteps"] == 1000000 and cfg["num_envs"] == 10
                assert cfg["num_critics"] == (100 if method == "SAC100" else 10)
                vals = []
                for seed in seeds:
                    run = group/f"seed{seed}"
                    final = read(run/"final.json")
                    assert final["steps"] == 1000000 and final["updates"] == 99796 and final["seed"] == seed
                    assert (run/"model.msgpack").stat().st_size > 0
                    hist = read(run/"curve.json")["records"]
                    np.testing.assert_array_equal([h["step"] for h in hist], steps)
                    returns = np.asarray([h["returns"] for h in hist])
                    assert returns.shape == (21,128) and np.isfinite(returns).all()
                    scores = returns.mean(axis=1)
                    np.testing.assert_allclose(scores, [h["mean_return"] for h in hist])
                    np.testing.assert_allclose(scores[-1], final["score"])
                    vals.append(scores)
                vals = np.asarray(vals)
                mean = vals.mean(axis=0)
                sd = vals.std(axis=0, ddof=1)
                ax.plot(steps, mean, color=color, lw=2, label=label)
                ax.fill_between(steps, mean-sd, mean+sd, color=color, alpha=.105, lw=0)
                summary.append([task,method,*vals[:,-1],mean[-1],sd[-1]])
                for i,step in enumerate(steps):
                    curves.append([task,method,step,*vals[:,i],mean[i],sd[i]])
            ax.set(xlim=(0,1000000), xlabel="Environment steps", ylabel="Average episode return")
            ax.set_title(task, loc="left", weight="bold")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x/1e6:g}M" if x else "0"))
            ax.grid(axis="y", alpha=.23)
            ax.spines[["top","right"]].set_visible(False)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01,1), frameon=False)
            fig.supxlabel("Lines show the mean across four training seeds. Shading shows ±1 sample SD.", fontsize=9)
            fig.savefig(root/f"average_return_{task}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    cols = [f"seed{s}" for s in seeds] + ["mean", "sample_sd"]
    write_csv(root/"summary.csv", ["task","method",*cols], summary)
    write_csv(root/"average_return.csv", ["task","method","step",*cols], curves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.progress:
        plot_progress(args.root)
    else:
        plot(args.root)

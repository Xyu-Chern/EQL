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


def write_report(root, tasks, summary):
    scores = {(row[0], row[1]): row for row in summary}
    labels = {"SAC10":"历史 SAC（10 critics）", "EQL":"固定 EQL",
              "Adaptive11":"Adaptive EQL（11 个 actor 候选）",
              "EQL_bagging":"固定 EQL + 一步分袋", "SAC100":"在线 SAC-N（100 critics）"}
    lines = ["# Hopper 与 Walker2d 百万步实验", "", "## 100 万步时的平均回报", "",
             "| 方法 | " + " | ".join(tasks) + " |", "|---|---:|---:|"]
    for method in METHODS:
        cells = []
        for task in tasks:
            mean, sd = scores[task,method][-2:]
            cells.append(f"{mean:.2f} ± {sd:.2f}")
        lines.append("| " + " | ".join([labels[method], *cells]) + " |")
    lines += ["", "每格给出四个训练 seeds（201–204）的均值和样本标准差。每个 seed 的分数取自 100 万步时的 128 个评估 episode。", "",
              "## 完整学习曲线", ""]
    for task in tasks:
        lines += [f"### {task}", "", f"![{task} 五方法平均回报](average_return_{task}.png)", ""]
    lines += ["每条曲线从第 0 步起，每 5 万环境步评估一次，共 21 个点。实线是同一组四个 seeds 的平均回报，阴影是跨 seed 的一个样本标准差。", "",
              "## 配置与数据", "",
              "所有方法均使用 10 个并行训练环境，每个 seed 共收集 100 万条 transition，完成 99,796 次 critic 更新。"]
    if tasks[0].endswith("-v3"):
        lines += ["", "本轮使用 Gymnasium 0.29.1 注册的 v3 环境、MuJoCo 2.1.0 和 mujoco-py 2.1.2.14。",
                  "", "固定 EQL 和分袋 EQL 沿用此前在 v5 选定的 EQL 配置。两者学习率均为 0.0003。Hopper 使用 τ_Q=0.3、τ_a=0.7，Walker2d 使用 τ_Q=0.3、τ_a=0.5。分袋版每条 transition 分入两个袋。SAC 两组和 Adaptive EQL 的学习率为 0.001。"]
    lines += ["", "[汇总 CSV](summary.csv) · [全部学习曲线与逐 seed 数据](average_return.csv) · [双页 PDF](average_return.pdf)", "",
              "每个环境和方法目录下的 `protocol.json` 保存配置与源文件哈希。各 seed 目录内的 `curve.json` 保存每个 episode 的原始回报和设备记录，`model.msgpack` 保存最终模型，`final.json` 保存终点回报和步数。", ""]
    (root/"README.md").write_text("\n".join(lines),encoding="utf-8")


def plot_progress(root, tasks):
    """Show each available seed independently while the four-seed groups fill."""
    for task in tasks:
        fig, axes = plt.subplots(1, len(METHODS), figsize=(16, 4), sharey=True,
                                 layout="constrained")
        overlay, comparison = plt.subplots(figsize=(9, 5.3), layout="constrained")
        for ax, (method, (label, color)) in zip(axes, METHODS.items()):
            histories = []
            for seed in (201, 202, 203, 204):
                path = root/task/method/f"seed{seed}"/"curve.json"
                if path.exists():
                    records = read(path)["records"]
                    histories.append(records)
                    ax.plot([r["step"]/1e6 for r in records],
                            [r["mean_return"] for r in records],
                            color=plt.get_cmap("tab10")(seed-201), lw=1.5,
                            marker=".", ms=4, label=str(seed))
            if len(histories) == 4:
                common = min(len(records) for records in histories)
                steps = [r["step"] for r in histories[0][:common]]
                assert all([r["step"] for r in records[:common]] == steps for records in histories)
                scores = np.asarray([[r["mean_return"] for r in records[:common]]
                                     for records in histories])
                mean, sd = scores.mean(axis=0), scores.std(axis=0, ddof=1)
                comparison.plot(steps, mean, color=color, lw=2, marker=".", ms=4, label=label)
                comparison.fill_between(steps, mean-sd, mean+sd, color=color, alpha=.105, lw=0)
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
        comparison.set(xlim=(0, 1000000), xlabel="Environment steps", ylabel="Average episode return")
        comparison.set_title(task, loc="left", weight="bold")
        comparison.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x/1e6:g}M" if x else "0"))
        comparison.grid(axis="y", alpha=.23)
        comparison.spines[["top","right"]].set_visible(False)
        if comparison.lines:
            comparison.legend(loc="upper left", bbox_to_anchor=(1.01,1), frameon=False)
        else:
            comparison.text(.5,.5,"Waiting for four paired seeds", ha="center", transform=comparison.transAxes)
        overlay.supxlabel("Each point uses all four seeds (mean ±1 sample SD). Lines end at their latest common evaluation.", fontsize=9)
        path = root/f"average_return_live_{task}.png"
        tmp = path.with_suffix(".tmp")
        overlay.savefig(tmp, format="png", dpi=160)
        tmp.replace(path)
        plt.close(overlay)


def plot(root, tasks):
    steps = np.arange(0, 1000001, 50000)
    seeds = (201, 202, 203, 204)
    summary, curves = [], []
    with PdfPages(root/"average_return.pdf") as pdf:
        for task in tasks:
            fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
            for method, (label, color) in METHODS.items():
                group = root/task/method
                protocol = read(group/"protocol.json")
                assert protocol["task"] == task
                if task.endswith("-v3"):
                    assert protocol["backend"]["mujoco_py"] == "2.1.2.14"
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
            caption = "Lines show the mean across four training seeds. Shading shows ±1 sample SD."
            if task.endswith("-v3"):
                caption += "\nFixed EQL and bagged EQL use the EQL settings selected on v5."
            fig.supxlabel(caption, fontsize=9)
            fig.savefig(root/f"average_return_{task}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    cols = [f"seed{s}" for s in seeds] + ["mean", "sample_sd"]
    write_csv(root/"summary.csv", ["task","method",*cols], summary)
    write_csv(root/"average_return.csv", ["task","method","step",*cols], curves)
    write_report(root,tasks,summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--version", choices=["3","5"], required=True)
    args = parser.parse_args()
    tasks = [f"{name}-v{args.version}" for name in ("Hopper","Walker2d")]
    if args.progress:
        plot_progress(args.root,tasks)
    else:
        plot(args.root,tasks)

"""Summarize complete paired Walker2d runs and plot same-seed return differences."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.stats import t


SEEDS = (1301, 1302, 1303, 1304)
PRIMARY = "AEQL-3e4"
DISPLAY = {
    PRIMARY: ("AEQL", "#007566"), "Mean-3e4": ("固定均值", "#53799d"),
    "EQL-555": ("原 555 配置", "#ba5b22"), "SAC10-1e3": ("SAC-10", "#8064a2"),
    "SAC2-1e3": ("SAC-2", "#454954"),
}
RATES = {name: .001 if name.endswith("1e3") else .0003 for name in
         [*DISPLAY, "AEQL-1e3", "Mean-1e3", "SAC10-3e4"]}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    rows, curves, windows = [], {}, []
    for name in RATES:
        curves[name] = []
        for seed in SEEDS:
            folder = root / name / f"seed{seed}"
            record, points = read(folder / "final.json"), read(folder / "curve.json")
            assert record["steps"] == 100000 and record["updates"] == 9796
            assert record["method"] == name and record["seed"] == seed
            assert [p["step"] for p in points] == list(range(0, 100001, 20000))
            assert all(len(p["returns"]) == (128 if p["step"] == 100000 else 32) for p in points)
            assert all(np.isclose(np.mean(p["returns"]), p["mean"]) for p in points)
            assert len(record["stochastic_returns"]) == 128
            assert np.isclose(np.mean(record["stochastic_returns"]), record["stochastic"])
            assert np.isclose(points[-1]["mean"], record["deterministic"])
            rows.append({k: record[k] for k in ("method","seed","deterministic","stochastic",
                                                "updates","gpu","gpu_uuid","final_tau","calibration_windows")})
            curves[name].append([p["mean"] for p in points])
            if name.startswith(("Mean", "AEQL")):
                events = read(folder / "calibration.json")
                assert len(events) == record["calibration_windows"] and len(events) > 0
                for ev in events:
                    assert np.asarray(ev["episode_losses"]).shape == (20, 4)
                    windows.append(dict(method=name, seed=seed, start=ev["start"], end=ev["end"],
                                        previous=ev["previous"], selected=ev["selected"],
                                        proposed=ev["proposed"], gain=ev["gain"], se=ev["standard_error"]))
    for seed in SEEDS:
        assert len({r["gpu_uuid"] for r in rows if r["seed"] == seed}) == 1
    csv_file(root / "seeds.csv", rows)
    csv_file(root / "windows.csv", windows)

    def values(method, metric):
        return np.array([r[metric] for r in rows if r["method"] == method])

    summary, pairs = [], []
    for name in RATES:
        for metric in ("deterministic", "stochastic"):
            scores = values(name, metric)
            summary.append(dict(method=name, lr=RATES[name], metric=metric, mean=scores.mean(),
                                seed_sd=scores.std(ddof=1)))
    for aeql in (PRIMARY, "AEQL-1e3"):
        for other in RATES:
            if other == aeql:
                continue
            for metric in ("deterministic", "stochastic"):
                d = values(aeql, metric) - values(other, metric)
                half_width = t.ppf(.975, len(d)-1) * d.std(ddof=1) / np.sqrt(len(d))
                pairs.append(dict(aeql=aeql, baseline=other, metric=metric, difference=d.mean(),
                                  wins=int((d > 0).sum()), low=d.mean()-half_width,
                                  high=d.mean()+half_width, **dict(zip(map(str,SEEDS),d))))
    csv_file(root / "summary.csv", summary)
    csv_file(root / "paired.csv", pairs)

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [f for f in ("Microsoft YaHei","Noto Sans CJK SC","SimHei")
                                       if f in fonts] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios":[1.16,1]})
    fig.subplots_adjust(left=.075, right=.98, bottom=.22, top=.78, wspace=.34)
    fig.suptitle("Walker2d-v5   |   10 万步的新种子复测", x=.075, y=.96, ha="left", fontsize=18)
    for name, (label, color) in DISPLAY.items():
        data = np.array(curves[name])
        ax.plot(np.arange(0, 101, 20), data.mean(0), marker="o", markersize=4, label=label,
                color=color, linewidth=2.8 if name == PRIMARY else 1.6)
    ax.set(xlabel="训练交互步数（千）", ylabel="确定性策略 · 原始回报", title="四个训练种子的平均学习曲线")
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=9)
    others = list(DISPLAY)[1:]
    bx.axvline(0, color="#666666", lw=1, linestyle="--")
    for i, other in enumerate(others):
        diffs = values(PRIMARY,"deterministic")-values(other,"deterministic")
        color = DISPLAY[other][1]
        bx.scatter(diffs, i+np.linspace(-.12,.12,4), s=35, color=color, alpha=.65)
        bx.scatter(diffs.mean(), i, s=69, marker="D", facecolors="white", edgecolors="#222", zorder=3)
    bx.set_yticks(range(len(others)), [DISPLAY[n][0] for n in others])
    bx.set_ylim(len(others)-.5, -.5)
    bx.set(xlabel="AEQL 减对照组的终点回报", title="配对差值  ·  右侧为 AEQL 胜出")
    for axis in (ax, bx):
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y" if axis is ax else "x", alpha=.15)
    fig.text(.075,.065,"种子 1301–1304。中间点每策略 32 回合，终点 128 回合。右图每个圆点对应一个种子，菱形为平均差。",fontsize=10)
    fig.text(.075,.022,"AEQL、固定均值与原 555 配置的学习率为 0.0003。两组 SAC 为 0.001。",fontsize=9, color="#51565b")
    fig.savefig(root / "walker_comparison.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4,4.6))
    fig.subplots_adjust(left=.085, right=.98, bottom=.19, top=.84, wspace=.32)
    for key, label, color in (("AEQL","AEQL","#007566"), ("Mean","固定均值","#53799d"),
                              ("SAC10","SAC-10","#8064a2")):
        ax.plot([0,1], [values(f"{key}-{suffix}","deterministic").mean() for suffix in ("3e4","1e3")],
                marker="o", linewidth=2.4, color=color, label=label)
    ax.set(xticks=[0,1], xticklabels=["0.0003","0.001"], xlim=(-.22,1.22),
           xlabel="学习率", ylabel="四个种子的平均确定性回报", title="10 万步终点对学习率的敏感度")
    ax.legend(frameon=False)
    for seed, color in zip(SEEDS, ("#147d70","#bb6933","#6a64ac","#3e6c95")):
        events = [r for r in windows if r["method"] == PRIMARY and r["seed"] == seed]
        times = [0] + [r["end"]/1000 for r in events] + [100]
        trajectory = [0] + [r["selected"] for r in events] + [events[-1]["selected"]]
        bx.step(times, trajectory, where="post", color=color, lw=1.5, alpha=.75)
        bx.scatter(times[1:-1], trajectory[1:-1], s=30, edgecolors=color, facecolors="none",
                   label=str(seed))
    bx.set(xlabel="训练交互步数（千）", ylabel="目标 critic 的 τ", ylim=(-.045,.56),
           yticks=[0,.1,.3,.5], title="主配置实际采用的目标 expectile")
    bx.legend(frameon=False, ncol=2, fontsize=9)
    for axis in (ax, bx):
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    fig.savefig(root / "walker_calibration.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)

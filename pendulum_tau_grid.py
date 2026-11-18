"""Paired Pendulum expectile sweep, using the archived SAC selection and exact zero endpoint."""
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import numpy as np
from eql import EQL
from original_bag_eql import OriginalBagEQL
from run_original_comparison import CONFIG, ready

ROOT = Path("results/pendulum_tau_grid")
SAC = Path("results/search_four/Pendulum-v1/SAC")
SEEDS = [201, 202, 203, 204]
GRID = list(itertools.product([0., 0.1, 0.3], [0., 0.5, 0.7]))
METHODS = {"EQL": EQL, "EQL_bagging": OriginalBagEQL}
COMMON = CONFIG | {"learning_rate": 1e-3}


def save(path, record):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False))
    temp.replace(path)


def file(method, q, a, seed):
    return ROOT / method / f"q{q}_a{a}_seed{seed}.json"


def read(path):
    return json.loads(path.read_text())


def reference(seed):
    run = read(SAC / f"confirm_2_{seed}.json")
    assert run["settings"] == {"learning_rate": 0.001} and run["seed"] == seed
    return run


def run(method, q, a):
    assert jax.devices()[0].platform == "gpu"
    cfg = COMMON | {"tau_Q": q, "tau_a": a}
    algo = METHODS[method].create(env="Pendulum-v1", **cfg)
    init = jax.jit(algo.init_state)
    def block(state):
        return jax.lax.fori_loop(0, 1000, lambda _, s: algo.train_iteration(s), state)
    train = jax.jit(block, donate_argnums=(0,))
    evaluate = jax.jit(lambda state, key: algo.eval_callback(algo, state, key)[1])
    for seed in SEEDS:
        path = file(method, q, a, seed)
        if path.exists():
            continue
        ref = reference(seed)
        if method == "EQL" and q == a == 0:
            source = Path(f"results/pendulum_sac_endpoint/exact/seed{seed}.json")
            record = read(source)
            assert record["config"] == cfg and record["returns"] == ref["returns"]
            record["source"] = str(source)
        else:
            state = init(jax.random.PRNGKey(seed))
            for _ in range(10):
                state = train(state)
                ready(state)
            assert int(state.global_step) == ref["steps"] == 100000
            assert int(state.critic_ts.step) == int(state.actor_ts.step) == ref["updates"] == 9796
            returns = np.asarray(evaluate(state, jax.random.PRNGKey(30000 + seed)))
            assert returns.shape == (128,) and np.isfinite(returns).all()
            record = dict(seed=seed,config=cfg,steps=100000,updates=9796,
                          score=float(returns.mean()),returns=returns.tolist(),source="new")
        record.update(method=method, environment="Pendulum-v1",sac=ref["score"])
        save(path, record)
        print(method,q,a,seed,record["score"],flush=True)


def summarize():
    lines = ["# Pendulum-v1 固定学习率的 expectile 搜索", "",
             "每组 10 万次交互、9796 次更新、10 个 critic。学习率 0.001，γ=0.98。训练 seeds 201–204，每个 seed 在终点评估 128 个 episode。正差值表示回报高于配对 SAC。分袋版每条一步片段进入两个袋。", "",
             "SAC 使用已保存的同 seed 终点评估；不分袋 (0,0) 复用已验证的逐 episode 等价结果。其余均为本轮训练。", ""]
    refs = np.array([reference(seed)["score"] for seed in SEEDS])
    lines.append(f"SAC 回报均值为 {refs.mean():.2f}，跨 seed 样本标准差为 {refs.std(ddof=1):.2f}。")
    rows = []
    for method in METHODS:
        lines.extend(["",f"## {method}","","| τ_Q | τ_a | 201 | 202 | 203 | 204 | 均值 ± SD | 平均差 | 胜出 seeds |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for q,a in GRID:
            runs = [read(file(method,q,a,s)) for s in SEEDS]
            for record,seed in zip(runs,SEEDS):
                assert record["config"] == COMMON | {"tau_Q":q,"tau_a":a}
                assert record["seed"]==seed and record["steps"]==100000 and record["updates"]==9796
                raw=np.asarray(record["returns"],dtype=np.float32)
                assert raw.shape==(128,) and np.isfinite(raw).all()
                np.testing.assert_allclose(raw.mean(),record["score"],rtol=1e-6)
            scores = np.array([r["score"] for r in runs])
            row = dict(method=method,tau_Q=q,tau_a=a,mean=scores.mean(),sd=scores.std(ddof=1),
                       paired_difference=(scores-refs).mean(),wins=int((scores>refs).sum()))
            rows.append(row | dict(zip([f"seed{s}" for s in SEEDS],scores)))
            cells = " | ".join(f"{v:.2f}" for v in scores)
            lines.append(f"| {q} | {a} | {cells} | {row['mean']:.2f} ± {row['sd']:.2f} | {row['paired_difference']:+.2f} | {row['wins']}/4 |")
    with (ROOT/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines.extend(["","候选排名与报告分数使用同一组四个 seeds，胜出表示本次网格上的观察值。"])
    (ROOT/"RESULTS_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def suite():
    ROOT.mkdir(parents=True,exist_ok=True)
    for m in METHODS: (ROOT/m).mkdir(exist_ok=True)
    protocol=dict(common=COMMON,grid_Q_then_a=GRID,seeds=SEEDS,methods=list(METHODS),
                  per_seed_episodes=128,score="final episode return mean",duplicates=2,
                  source_hash={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                               ["eql.py","original_bag_eql.py","bagging_buffer.py","pendulum_tau_grid.py"]})
    manifest=ROOT/"protocol.json"
    if manifest.exists() and read(manifest)!=json.loads(json.dumps(protocol)):
        raise RuntimeError("Output directory already contains a different protocol")
    save(manifest,protocol)
    for method in METHODS:
        for q,a in GRID:
            if not all(file(method,q,a,s).exists() for s in SEEDS):
                subprocess.run([sys.executable,"-u",__file__,method,str(q),str(a)],check=True)
    summarize()


if __name__=="__main__":
    if len(sys.argv)==1: suite()
    else: run(sys.argv[1],float(sys.argv[2]),float(sys.argv[3]))

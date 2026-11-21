"""Four-task, paired search. Run one task/method per process on the WSL GPU."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import numpy as np

from eql.algorithms.eql import EQL
from eql.algorithms.original_bag import OriginalBagEQL
from eql.algorithms.sac import SAC
from eql.native import NativeBag, NativeEQL, NativeSAC, NativeTask, native_evaluator
from eql.records import save
from eql.training import CONFIG, TASKS

ROOT = Path("results/search_four")
METHODS = {"SAC": SAC, "EQL": EQL, "EQL_bagging": OriginalBagEQL}
TUNE = [101, 102]
TEST = [201, 202, 203, 204]
EQL_GRID = [dict(tau_a=a,tau_Q=q) for a,q in [(0.7,0.3),(0.5,0.3),(0.7,0.1)]]
SAC_GRID = [dict(learning_rate=r) for r in [0.0003,0.0001,0.001]]


def run(task,method):
    grid=SAC_GRID if method=="SAC" else EQL_GRID
    folder=ROOT/task/method
    folder.mkdir(parents=True,exist_ok=True)
    native=task.endswith("-v5")
    cls={"SAC":NativeSAC,"EQL":NativeEQL,"EQL_bagging":NativeBag}[method] if native else METHODS[method]
    env=NativeTask(task) if native else task
    base=cls.create(env=env,**CONFIG)
    if native: evaluate=native_evaluator(task,base)
    else:
        evaluate_state=jax.jit(lambda s,k:base.eval_callback(base,s,k)[1])
    # Rejax captures the learning rate in its optimizer at init time. Build each
    # candidate via create/initialization so SAC's learning-rate search is real.
    def series(candidate,seeds,phase):
        paths=[folder/f"{phase}_{candidate}_{s}.json" for s in seeds]
        if all(p.exists() for p in paths): return [json.loads(p.read_text())["score"] for p in paths]
        algo=cls.create(env=env,**(CONFIG|grid[candidate]))
        init=jax.jit(algo.init_state)
        block=jax.jit(lambda s:jax.lax.fori_loop(0,1000,lambda _,v:algo.train_iteration(v),s),donate_argnums=(0,))
        scores=[]
        for seed,path in zip(seeds,paths):
            if path.exists():
                scores.append(json.loads(path.read_text())["score"]);continue
            state=init(jax.random.PRNGKey(seed))
            for _ in range(10):
                state=block(state)
                for x in jax.tree.leaves(state):
                    if hasattr(x,"block_until_ready"): x.block_until_ready()
            assert int(state.global_step)==100000 and int(state.critic_ts.step)==9796
            returns=evaluate(state.actor_ts.params,seed) if native else np.asarray(evaluate_state(state,jax.random.PRNGKey(30000+seed)))
            assert len(returns)==128 and np.isfinite(returns).all()
            score=float(np.mean(returns))
            save(path,dict(task=task,method=method,phase=phase,seed=seed,candidate=candidate,settings=grid[candidate],
                           steps=100000,updates=9796,returns=returns.tolist(),score=score))
            print(task,method,phase,candidate,seed,round(score,2),flush=True)
            scores.append(score)
        return scores
    means=[np.mean(series(i,TUNE,"search")) for i in range(len(grid))]
    best=int(np.argmax(means))
    save(folder/"selection.json",dict(grid=grid,search_means=means,winner=best))
    series(best,TEST,"confirm")


def suite():
    ROOT.mkdir(parents=True,exist_ok=True)
    save(ROOT/"protocol.json",dict(tasks=TASKS,methods=list(METHODS),common=CONFIG,search_seeds=TUNE,
            confirmation_seeds=TEST,eql_grid=EQL_GRID,sac_grid=SAC_GRID,duplicates=2,
            criterion="per-task per-method highest mean final 128-episode return across two search seeds, first in grid breaks ties"))
    for task in TASKS:
        for method in METHODS:
            subprocess.run([sys.executable,"-u","-m", "experiments.search",task,method],check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=TASKS)
    parser.add_argument("method", nargs="?", choices=METHODS)
    args = parser.parse_args()
    values = (args.task, args.method)
    if all(value is None for value in values):
        suite()
    elif any(value is None for value in values):
        parser.error("Provide all positional arguments to run one configuration.")
    else:
        assert jax.devices()[0].platform == "gpu"
        run(args.task, args.method)

"""Measure Pendulum training feasibility with larger critic ensembles in fresh processes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path("results/pendulum_capacity")


def train(method, count):
    import jax
    import numpy as np
    from run_original_comparison import CONFIG, ready, SAC, EQL, OriginalBagEQL
    config = CONFIG | dict(env="Pendulum-v1",num_critics=count,learning_rate=0.001)
    if method != "SAC": config.update(tau_a=0.7,tau_Q=0.3)
    if method == "EQL_bagging": config["num_duplicates"] = count//5
    algo = {"SAC":SAC,"EQL":EQL,"EQL_bagging":OriginalBagEQL}[method].create(**config)
    assert jax.devices()[0].platform == "gpu"
    state = jax.jit(algo.init_state)(jax.random.PRNGKey(201))
    step = jax.jit(lambda s:jax.lax.fori_loop(0,1000,lambda _,v:algo.train_iteration(v),s),donate_argnums=(0,))
    state = step(state)
    ready(state)
    assert int(state.global_step)==10000 and int(state.critic_ts.step)==796
    if method=="EQL_bagging": assert int(state.bags.count)==2000
    for leaf in jax.tree.leaves((state.actor_ts.params,state.critic_ts.params)):
        assert np.isfinite(np.asarray(leaf)).all()
    return config


def memory():
    result=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                          capture_output=True,text=True,check=True)
    return int(result.stdout.strip().splitlines()[0])


def suite():
    ROOT.mkdir(parents=True,exist_ok=True)
    records=[]
    for count in (20,50):
        for method in ("SAC","EQL","EQL_bagging"):
            initial=memory()
            peak=initial
            with (ROOT/f"{method}_{count}.log").open("w") as log:
                proc=subprocess.Popen([sys.executable,__file__,method,str(count)],stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    peak=max(peak,memory())
                    time.sleep(0.3)
            records.append(dict(method=method,critics=count,exit_code=proc.returncode,
                                whole_gpu_before_MiB=initial,whole_gpu_peak_sample_MiB=peak))
            (ROOT/"summary.json").write_text(json.dumps(records,indent=2))
            print(records[-1],flush=True)


if __name__=="__main__":
    if len(sys.argv)==1: suite()
    else: print(train(sys.argv[1],int(sys.argv[2])))

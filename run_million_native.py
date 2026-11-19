"""Long-horizon online comparison on native Gymnasium MuJoCo tasks."""
import argparse
import fcntl
import hashlib
import json
import os
import socket
import time
from importlib.metadata import version
from pathlib import Path

import jax
import mujoco
import numpy as np
from flax import serialization

from pendulum_tau_grid import COMMON, SEEDS
from run_adaptive_tasks import NativeAdaptiveEQL
from search_four import NativeBag, NativeEQL, NativeSAC, NativeTask, native_evaluator, save


METHODS = {"SAC10": NativeSAC, "EQL": NativeEQL, "Adaptive11": NativeAdaptiveEQL,
           "EQL_bagging": NativeBag, "SAC100": NativeSAC}
TASKS = ("Hopper-v3", "Walker2d-v3", "Hopper-v5", "Walker2d-v5")
EVAL_EVERY = 50000
TOTAL = 1000000


def native_snapshot(task):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    snapshots = []
    for wrapped in task.instances:
        env = wrapped.unwrapped
        if wrapped.spec.id.endswith("-v3"):
            data = env.sim.data
            snapshots.append(dict(legacy_state=env.sim.get_state().flatten().copy(),
                                  inputs={name:np.asarray(getattr(data,name)).copy()
                                          for name in ("qacc_warmstart","ctrl","qfrc_applied","xfrc_applied")},
                                  elapsed=int(wrapped._elapsed_steps),
                                  rng=json.dumps(env.np_random.bit_generator.state)))
            continue
        physics = np.empty(mujoco.mj_stateSize(env.model, spec))
        mujoco.mj_getState(env.model, env.data, physics, spec)
        snapshots.append(dict(physics=physics, elapsed=int(wrapped._elapsed_steps),
                              rng=json.dumps(env.np_random.bit_generator.state)))
    return snapshots


def restore_native(task, snapshots):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    for wrapped, snap in zip(task.instances, snapshots, strict=True):
        env = wrapped.unwrapped
        if "legacy_state" in snap:
            env.sim.set_state_from_flattened(snap["legacy_state"])
            for name, value in snap["inputs"].items():
                getattr(env.sim.data,name)[:] = value
            env.sim.forward()
            env.sim.data.qacc_warmstart[:] = snap["inputs"]["qacc_warmstart"]
        else:
            mujoco.mj_setState(env.model, env.data, snap["physics"], spec)
            mujoco.mj_forward(env.model, env.data)
            mujoco.mj_setState(env.model, env.data, snap["physics"], spec)
        wrapped._elapsed_steps = snap["elapsed"]
        env.np_random.bit_generator.state = json.loads(snap["rng"])


def write_bytes(path, data):
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def run(task, method, output):
    cfg = dict(COMMON, total_timesteps=TOTAL)
    if method in ("EQL", "EQL_bagging"):
        cfg.update(learning_rate=3e-4, tau_Q=.3, tau_a=.7 if task.startswith("Hopper-") else .5)
    if method=="EQL_bagging":
        cfg.update(num_duplicates=2)
    if method=="SAC100":
        cfg.update(num_critics=100)
    gpu = jax.devices()[0]
    assert gpu.platform=="gpu"
    folder = output/task/method
    folder.mkdir(parents=True, exist_ok=True)
    sources = ("run_million_native.py","sac.py","eql.py","search_four.py","adaptive_eql.py",
               "run_adaptive_tasks.py","original_bag_eql.py","bagging_buffer.py","pendulum_tau_grid.py",
               "run_original_comparison.py")
    env = NativeTask(task)
    assert all(e.spec.id == task for e in env.instances)
    backend = dict(gymnasium=version("gymnasium"),entry_point=env.instances[0].spec.entry_point)
    if task.endswith("-v3"):
        import mujoco_py
        sim = env.instances[0].unwrapped.sim
        assert isinstance(sim, mujoco_py.MjSim)
        backend.update(mujoco_py=version("mujoco-py"),mujoco_version=int(mujoco_py.functions.mj_version()))
        assert backend["mujoco_version"] == 210
    protocol = dict(task=task, method=method, config=cfg, seeds=SEEDS, eval_every=EVAL_EVERY,backend=backend,
                    eval_episodes=128, eval_protocol="search_four.native_evaluator, common seed at every evaluation",
                    actor_arms=[i/10 for i in range(11)] if method=="Adaptive11" else None,
                    source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
    with (folder/"protocol.lock").open("a") as lock:
        fcntl.lockf(lock,fcntl.LOCK_EX)
        mf=folder/"protocol.json"
        if mf.exists() and json.loads(mf.read_text())!=protocol:
            raise RuntimeError(f"Protocol mismatch in {folder}")
        if not mf.exists():
            save(mf,protocol)
    if all((folder/f"seed{s}"/"final.json").exists() for s in SEEDS):
        return
    algo = METHODS[method].create(env=env,**cfg)
    if method=="Adaptive11":
        assert list(algo.actor_arms)==protocol["actor_arms"]
    evaluate = native_evaluator(task,algo)
    init = jax.jit(algo.init_state)
    interval = 10000
    block = jax.jit(lambda state:jax.lax.fori_loop(0,interval//algo.num_envs,
                      lambda _,s:algo.train_iteration(s),state),donate_argnums=(0,))
    for seed in SEEDS:
        dest=folder/f"seed{seed}"
        dest.mkdir(exist_ok=True)
        with (dest/"run.lock").open("a") as lock:
            try:
                fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (dest/"final.json").exists():
                continue
            print(task,method,seed,gpu.device_kind,flush=True)
            state=init(jax.random.PRNGKey(seed))
            jax.block_until_ready(state)
            checkpoint=dest/"resume.msgpack"
            history=[]
            devices=[]
            if checkpoint.exists():
                snap=serialization.msgpack_restore(checkpoint.read_bytes())
                state=jax.device_put(serialization.from_state_dict(state,snap["training"]))
                restore_native(env,snap["native"])
                history=snap["history"]
                devices=snap["devices"]
                assert int(state.global_step)==history[-1]["step"]
                del snap
            devices.append(dict(node=socket.gethostname(),gpu=gpu.device_kind,
                                job=os.environ.get("SLURM_JOB_ID"),from_step=int(state.global_step)))
            start=int(state.global_step)
            for step in range(start,TOTAL+1,interval):
                if step>start:
                    state=block(state)
                if step==start and history:
                    continue
                assert int(state.global_step)==step
                if step%EVAL_EVERY==0:
                    scores=np.asarray(evaluate(state.actor_ts.params,seed))
                    assert scores.shape==(128,) and np.isfinite(scores).all()
                    record=dict(step=step,mean_return=float(scores.mean()),returns=scores.tolist(),
                                updates=int(state.critic_ts.step))
                    if method=="Adaptive11":
                        record.update(rounds=int(state.rounds),tau_Q=float(state.tau_q),
                                      actor_probabilities=np.asarray(algo.probabilities(state.logits)).tolist())
                    history.append(record)
                    save(dest/"curve.json",dict(seed=seed,devices=devices,records=history))
                    if step and step%100000==0 and step<TOTAL:
                        snap=dict(training=serialization.to_state_dict(jax.device_get(state)),
                                  native=native_snapshot(env),history=history,devices=devices)
                        write_bytes(checkpoint,serialization.msgpack_serialize(snap))
                        del snap
                    print(task,method,seed,step,round(float(scores.mean()),2),flush=True)
                save(dest/"progress.json",dict(step=step,node=socket.gethostname(),time=time.time()))
            assert len(history)==21 and int(state.actor_ts.step)==int(state.critic_ts.step)==int(state.alpha_ts.step)==99796
            weights=dict(actor=state.actor_ts.params,critics=state.critic_ts.params,
                         target_critics=state.critic_target_params,alpha=state.alpha_ts.params)
            write_bytes(dest/"model.msgpack",serialization.to_bytes(weights))
            save(dest/"final.json",dict(seed=seed,steps=TOTAL,updates=int(state.critic_ts.step),
                                        score=history[-1]["mean_return"],devices=devices))
            checkpoint.unlink(missing_ok=True)
    for wrapped in env.instances:
        wrapped.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("task",choices=TASKS)
    parser.add_argument("method",choices=METHODS)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    run(args.task,args.method,args.output)

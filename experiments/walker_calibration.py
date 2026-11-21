"""Paired 100k-step Walker2d-v5 comparison, with all controller samples in replay."""
import argparse
import json
import socket
import subprocess
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from eql.algorithms.calibration import (
    CriticCalibratedEQL,
    choose_tau,
    frozen_training_block,
    trajectory_losses,
)
from eql.native import NativeEQL, NativeSAC, NativeTask, native_evaluator
from eql.records import save, source_digest
from eql.training import CONFIG

TASK = "Walker2d-v5"
STEPS = 100_000
CHUNK = 100  # vector steps, 1,000 environment transitions
SEEDS = (1301, 1302, 1303, 1304)
METHODS = {
    "AEQL-3e4": (CriticCalibratedEQL, 10, 3e-4, 0., 0.),
    "Mean-3e4": (CriticCalibratedEQL, 10, 3e-4, .5, 0.),
    "EQL-555": (NativeEQL, 10, 3e-4, .3, .5),
    "SAC10-1e3": (NativeSAC, 10, 1e-3, None, None),
    "SAC2-1e3": (NativeSAC, 2, 1e-3, None, None),
    "AEQL-1e3": (CriticCalibratedEQL, 10, 1e-3, 0., 0.),
    "Mean-1e3": (CriticCalibratedEQL, 10, 1e-3, .5, 0.),
    "SAC10-3e4": (NativeSAC, 10, 3e-4, None, None),
}


def deterministic_actions(module, obs):
    loc = module.action_mean(module.features(obs))
    return (module.action_loc + module.action_scale * jnp.tanh(loc)).clip(-1, 1)


def deterministic_evaluator(algo):
    envs = [gym.make(TASK) for _ in range(128)]
    act = jax.jit(lambda params, x: algo.actor.apply(params, x, method=deterministic_actions))

    def evaluate(params, seed, count):
        rng = np.random.default_rng(seed + 70_000)
        obs = np.stack([e.reset(seed=int(k))[0] for e, k in
                        zip(envs, rng.integers(0, 2**32, count))]).astype(np.float32)
        live = np.ones(count, bool)
        rewards = np.zeros(count)
        while live.any():
            actions = np.asarray(act(params, obs))
            for i in np.flatnonzero(live):
                obs[i], reward, term, trunc, _ = envs[i].step(actions[i])
                rewards[i] += reward
                live[i] = not (term or trunc)
        return rewards

    return evaluate


def run(method, seeds, output):
    cls, n, lr, initial, actor_tau = METHODS[method]
    cfg = CONFIG | dict(num_critics=n, learning_rate=lr)
    if initial is not None:
        cfg.update(tau_Q=initial, tau_a=actor_tau)
    env = NativeTask(TASK)
    algo = cls.create(env=env, **cfg)
    gpu = jax.devices()[0]
    assert gpu.platform == "gpu" and not algo.normalize_observations and not algo.normalize_rewards
    assert version("gymnasium") == "1.4.0" and version("mujoco") == "3.17.0"
    matched = cls is CriticCalibratedEQL
    if matched:
        calibrate = frozen_training_block(algo, CHUNK)

    @jax.jit
    def live_block(state, current):
        active = algo.replace(tau_Q=current) if matched else algo
        return jax.lax.fori_loop(0, CHUNK, lambda _, s: active.train_iteration(s), state)

    init = jax.jit(algo.init_state)
    det = deterministic_evaluator(algo)
    stochastic = native_evaluator(TASK, algo)
    folder = output / method
    folder.mkdir(exist_ok=True, parents=True)
    sources = ("experiments/walker_calibration.py", "eql/algorithms/calibration.py", "eql/algorithms/eql.py", "eql/expectiles.py", "eql/algorithms/sac.py", "eql/native.py",
               "eql/training.py")
    protocol = dict(method=method, config=cfg, task=TASK, seeds=SEEDS, steps=STEPS,
                    calibration_starts=[5000,25000,45000,65000,85000], candidates=[0,.1,.3,.5],
                    source_hashes={f: source_digest(f)
                                   for f in sources},
                    packages={p: version(p) for p in ("gymnasium","mujoco","jax","rejax")})
    gpu_id = subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    for seed in seeds:
        path = folder / f"seed{seed}"
        path.mkdir(exist_ok=True)
        if (path / "final.json").exists():
            assert json.loads((path / "protocol.json").read_text()) == json.loads(json.dumps(protocol))
            continue
        save(path / "protocol.json", protocol)
        state = init(jax.random.PRNGKey(seed))
        tau = initial or 0.
        events, curve, traces = [], [], []
        snapshot = None
        for step in range(0, STEPS + 1, 1000):
            if step % 20_000 == 0:
                count = 128 if step == STEPS else 32
                returns = det(state.actor_ts.params, seed, count)
                assert np.isfinite(returns).all()
                curve.append(dict(step=step, mean=float(returns.mean()), returns=returns.tolist()))
                save(path / "curve.json", curve)
                print(method, seed, step, round(returns.mean(), 2), "tau", tau, flush=True)
            if step == STEPS:
                break
            if matched and snapshot is None and step % 20_000 == 5000:
                snapshot = (state.actor_ts.params, state.critic_target_params,
                            jnp.exp(state.alpha_ts.params["log_alpha"]))
                start = step
                traces = []
            if snapshot is None:
                state = live_block(state, jnp.float32(tau))
            else:
                state, trace = calibrate(state, *snapshot, jnp.float32(tau))
                traces.append(jax.device_get(trace))
                merged = tuple(np.concatenate(x) for x in zip(*traces))
                errors = trajectory_losses(merged, float(algo.gamma), algo.num_envs)
                if errors is not None:
                    select, row = choose_tau(errors, tau, np.random.default_rng([seed, start]))
                    previous = tau
                    if method.startswith("AEQL"):
                        tau = select
                    events.append(dict(start=start, end=step+1000, previous=previous,
                                       selected=tau, **row))
                    save(path / "calibration.json", events)
                    snapshot = None
        assert int(state.global_step) == STEPS and int(state.critic_ts.step) == 9796
        assert int(state.actor_ts.step) == int(state.alpha_ts.step) == 9796
        scores = stochastic(state.actor_ts.params, seed)
        assert scores.shape == (128,) and np.isfinite(scores).all()
        (path / "model.msgpack").write_bytes(serialization.to_bytes(dict(actor=state.actor_ts.params,
              critics=state.critic_ts.params, target=state.critic_target_params, alpha=state.alpha_ts.params)))
        save(path / "final.json", dict(method=method, seed=seed, steps=STEPS, updates=9796,
             deterministic=curve[-1]["mean"], stochastic=float(scores.mean()), stochastic_returns=scores.tolist(),
             final_tau=tau, calibration_windows=len(events), gpu=gpu.device_kind, gpu_uuid=gpu_id,
             node=socket.gethostname()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("seeds", type=int, nargs="+", choices=SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.method, args.seeds, args.output)

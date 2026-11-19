"""Run the repository's shared-replay, all-critic minimum SAC on one online task."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from flax.serialization import to_bytes

from pendulum_tau_grid import COMMON, SEEDS
from sac import SAC
from search_four import TASKS, NativeSAC, NativeTask, native_evaluator, save


def run(task, output):
    gpu = jax.devices()[0]
    if gpu.platform != "gpu":
        raise RuntimeError("GPU required for the comparison")
    native = task.endswith("-v5")
    algo = (NativeSAC.create(env=NativeTask(task), **COMMON) if native else
            SAC.create(env=task, **COMMON))
    folder = output / task
    folder.mkdir(parents=True, exist_ok=True)
    sources = ["sac.py", "search_four.py", "pendulum_tau_grid.py",
               "run_original_comparison.py", "run_online_sac_n.py"]
    protocol = dict(task=task, method="online SAC-N", config=COMMON, seeds=SEEDS,
                    evaluation_episodes=128, device=gpu.device_kind,
                    source_sha256={s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
    manifest = folder / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise RuntimeError(f"Protocol mismatch at {manifest}")
    save(manifest, protocol)
    init = jax.jit(algo.init_state)
    block = jax.jit(lambda s: jax.lax.fori_loop(0, 1000, lambda _, v: algo.train_iteration(v), s),
                    donate_argnums=(0,))
    if native:
        evaluate = native_evaluator(task, algo)
    else:
        evaluate = jax.jit(lambda s, k: algo.eval_callback(algo, s, k)[1])
    print(task, gpu.device_kind, COMMON, SEEDS, flush=True)
    for seed in SEEDS:
        path = folder / f"seed{seed}.json"
        if path.exists():
            continue
        state = init(jax.random.PRNGKey(seed))
        for step in range(1000 * algo.num_envs, algo.total_timesteps + 1, 1000 * algo.num_envs):
            state = block(state)
            assert int(state.global_step) == step
            print(task, seed, step, flush=True)
        assert int(state.global_step) == algo.total_timesteps == 100000
        assert int(state.actor_ts.step) == int(state.critic_ts.step) == int(state.alpha_ts.step) == 9796
        returns = (evaluate(state.actor_ts.params, seed) if native else
                   np.asarray(evaluate(state, jax.random.PRNGKey(30000 + seed))))
        assert returns.shape == (128,) and np.isfinite(returns).all()
        checkpoint = folder / f"seed{seed}.msgpack"
        temp = checkpoint.with_suffix(".tmp")
        temp.write_bytes(to_bytes(dict(actor=state.actor_ts.params, critics=state.critic_ts.params,
                                       target_critics=state.critic_target_params, alpha=state.alpha_ts.params)))
        temp.replace(checkpoint)
        save(path, dict(seed=seed, steps=int(state.global_step), updates=int(state.critic_ts.step),
                        returns=returns.tolist(), score=float(returns.mean())))
        print(task, seed, "finished", float(returns.mean()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.task, args.output)

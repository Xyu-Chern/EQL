"""Run adaptive shared-replay EQL on the three remaining online tasks."""
import argparse
import socket
from pathlib import Path

import jax
import numpy as np
from flax.serialization import to_bytes

from eql.algorithms.adaptive import AdaptiveEQL
from eql.native import NativeAdaptiveEQL, NativeTask, native_evaluator
from eql.records import read, save, source_digest
from eql.training import COMMON, SEEDS


def run(task, output):
    device = jax.devices()[0]
    assert device.platform == "gpu" and "RTX 6000" in device.device_kind
    native = task in ("Hopper-v5", "Walker2d-v5")
    algo = (NativeAdaptiveEQL.create(env=NativeTask(task), **COMMON) if native else
            AdaptiveEQL.create(env=task, **COMMON))
    folder = output / task
    folder.mkdir(parents=True, exist_ok=True)
    sources = ["eql/algorithms/adaptive.py", "experiments/adaptive_tasks.py", "eql/algorithms/eql.py", "eql/expectiles.py", "eql/algorithms/sac.py",
               "eql/native.py", "eql/records.py", "eql/training.py"]
    protocol = dict(task=task, config=COMMON, seeds=SEEDS, evaluation_episodes=128,
                    evaluator="eql.native.native_evaluator" if native else "Rejax eval_callback",
                    device=device.device_kind, controller="one complete post-warmup episode per slot per arm",
                    source_sha256={s: source_digest(s) for s in sources})
    manifest = folder / "protocol.json"
    if manifest.exists() and read(manifest) != protocol:
        raise RuntimeError(f"Protocol mismatch at {manifest}")
    save(manifest, protocol)
    if native:
        evaluate = native_evaluator(task, algo)
    else:
        evaluate = jax.jit(lambda state, key: algo.eval_callback(algo, state, key)[1])

    init = jax.jit(algo.init_state)
    iterations = 250
    block_steps = iterations * algo.num_envs
    assert COMMON["total_timesteps"] % block_steps == 0

    def iteration(state, _):
        new = algo.train_iteration(state)
        event = (new.rounds != state.rounds, new.global_step, state.arm, state.tau_q,
                 new.tau_q, new.last_bias, new.last_return, new.last_feedback)
        return new, event

    block = jax.jit(lambda state: jax.lax.scan(iteration, state, None, length=iterations),
                    donate_argnums=(0,))
    print(f"{socket.gethostname()} {device.device_kind} {task} seeds={SEEDS}", flush=True)
    for seed in SEEDS:
        path = folder / f"seed{seed}.json"
        if path.exists():
            assert read(path)["steps"] == COMMON["total_timesteps"]
            continue
        state = init(jax.random.PRNGKey(seed))
        history = []
        for step in range(block_steps, COMMON["total_timesteps"] + 1, block_steps):
            state, events = block(state)
            changed, steps, arms, old_q, new_q, bias, score, feedback = map(np.asarray, events)
            assert int(state.global_step) == step
            for i in np.flatnonzero(changed):
                assert np.isfinite([old_q[i], new_q[i], bias[i], score[i], feedback[i]]).all()
                assert 0 <= new_q[i] <= 1 and (new_q[i] - old_q[i]) * bias[i] <= 1e-6
                history.append(dict(step=int(steps[i]), tau_a=algo.actor_arms[int(arms[i])],
                                    tau_Q=float(old_q[i]), next_tau_Q=float(new_q[i]),
                                    bias=float(bias[i]), training_return=float(score[i]),
                                    feedback=float(feedback[i])))
            save(folder / f"progress_seed{seed}.json", dict(seed=seed, step=step, rounds=len(history)))
            if step == block_steps or step % 10000 == 0:
                print(f"{task} seed={seed} step={step} rounds={len(history)}", flush=True)
        assert int(state.actor_ts.step) == int(state.alpha_ts.step) == int(state.critic_ts.step) == 9796
        assert int(state.rounds) == len(history) and history
        returns = (evaluate(state.actor_ts.params, seed) if native else
                   np.asarray(evaluate(state, jax.random.PRNGKey(30000 + seed))))
        assert returns.shape == (128,) and np.isfinite(returns).all()
        checkpoint = folder / f"seed{seed}.msgpack"
        temp = checkpoint.with_suffix(".tmp")
        temp.write_bytes(to_bytes(dict(actor=state.actor_ts.params, critics=state.critic_ts.params,
                                       target_critics=state.critic_target_params, alpha=state.alpha_ts.params)))
        temp.replace(checkpoint)
        save(path, dict(seed=seed, steps=step, updates=int(state.actor_ts.step), rounds=history,
                        score=float(returns.mean()), returns=returns.tolist(), final_tau_Q=float(state.tau_q),
                        final_actor_probabilities=np.asarray(algo.probabilities(state.logits)).tolist()))
        print(f"{task} seed={seed} finished score={returns.mean():.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["Reacher-misc", "Hopper-v5", "Walker2d-v5"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.task, args.output)

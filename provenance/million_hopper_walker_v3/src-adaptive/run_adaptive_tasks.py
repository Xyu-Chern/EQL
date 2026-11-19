"""Run adaptive shared-replay EQL on the three remaining online tasks."""
import argparse
import hashlib
import socket
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import to_bytes
from rejax.algos.algorithm import register_init

from adaptive_eql import AdaptiveEQL
from eql import EQL, compute_expectile_on_axis
from pendulum_tau_grid import COMMON, SEEDS, read, save
from search_four import NativeTask, NativeTraining, native_evaluator


class NativeAdaptiveEQL(NativeTraining, AdaptiveEQL):
    def supports_episode_feedback(self):
        return self.env.name in ("Hopper-v3", "Walker2d-v3", "Hopper-v5", "Walker2d-v5")

    @register_init
    def init_episode_tracking(self, rng):
        return dict(age=jnp.zeros(self.num_envs, jnp.int32),
                    tracking=jnp.zeros(self.num_envs, bool),
                    completed=jnp.zeros(self.num_envs, bool))

    def collect_transitions(self, ts):
        _, action_key = jax.random.split(ts.rng)
        _, logp = self.actor.apply(ts.actor_ts.params, ts.last_obs, action_key,
                                   method="action_log_prob")
        collected, batch = EQL.collect_transitions(self, ts)
        # After a controller change, a trajectory already in flight must finish
        # and reset before this slot may contribute to the new arm.
        start = (ts.age == 0) & ~ts.completed & (ts.global_step >= self.fill_buffer)
        active = ts.tracking | start

        def predict():
            qs = self.vmap_critic(ts.critic_ts.params, batch.obs, batch.action)
            estimates = compute_expectile_on_axis(qs, ts.tau_q, axis=0, num_iters=self.num_iters)
            return jnp.where(start, estimates, ts.start_prediction)

        predicted = jax.lax.cond(start.any(), predict, lambda: ts.start_prediction)
        entropy = -jnp.exp(ts.alpha_ts.params["log_alpha"]) * logp
        increment = self.gamma ** ts.age * (batch.reward + jnp.where(start, 0., entropy))
        return collected.replace(
            start_prediction=predicted,
            soft_return=jnp.where(start, 0., ts.soft_return) + jnp.where(active, increment, 0.),
            raw_return=jnp.where(start, 0., ts.raw_return) + jnp.where(active, batch.reward, 0.),
            age=jnp.where(batch.done, 0, ts.age + 1),
            tracking=active & ~batch.done,
            completed=ts.completed | (active & batch.done),
        ), batch

    def train_iteration(self, ts):
        trained = EQL.train_iteration(self, ts)

        def finish(state):
            return self.finish_round(state).replace(completed=jnp.zeros_like(state.completed))

        return jax.lax.cond(trained.completed.all(), finish, lambda state: state, trained)


def run(task, output):
    device = jax.devices()[0]
    assert device.platform == "gpu" and "RTX 6000" in device.device_kind
    native = task in ("Hopper-v5", "Walker2d-v5")
    algo = (NativeAdaptiveEQL.create(env=NativeTask(task), **COMMON) if native else
            AdaptiveEQL.create(env=task, **COMMON))
    folder = output / task
    folder.mkdir(parents=True, exist_ok=True)
    sources = ["adaptive_eql.py", "run_adaptive_tasks.py", "eql.py", "sac.py",
               "search_four.py", "pendulum_tau_grid.py", "run_original_comparison.py"]
    protocol = dict(task=task, config=COMMON, seeds=SEEDS, evaluation_episodes=128,
                    evaluator="search_four native_evaluator" if native else "Rejax eval_callback",
                    device=device.device_kind, controller="one complete post-warmup episode per slot per arm",
                    source_sha256={s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
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

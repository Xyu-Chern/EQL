"""Run the adaptive shared-replay EQL under the archived Pendulum grid protocol."""
import argparse
import json
import os
from pathlib import Path

from eql.records import source_digest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import numpy as np
from flax.serialization import to_bytes

from eql.algorithms.adaptive import AdaptiveEQL
from eql.records import read, save
from eql.training import COMMON, SEEDS
from experiments._pendulum import grid_file, reference

OUT = Path("results/adaptive_pendulum")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    algo = AdaptiveEQL.create(env="Pendulum-v1", **COMMON)
    horizon = int(algo.env_params.max_steps_in_episode)
    batch_steps = horizon * algo.num_envs
    protocol = dict(env="Pendulum-v1", common=COMMON, seeds=SEEDS, horizon=horizon,
                    actor_arms=list(algo.actor_arms), initial_q=algo.initial_q, q_rate=algo.q_rate,
                    bandit_rate=algo.bandit_rate, exploration=algo.exploration, history_rounds=5,
                    alpha_init=1., target_entropy=float(algo.target_entropy), alpha_lr=float(algo.learning_rate),
                    training="shared replay; each arm held for one synchronous batch of episodes",
                    evaluation="128 final stochastic episodes, key=30000+seed; no controller feedback",
                    device=str(jax.devices()[0]), source_sha256={
                        f: source_digest(f) for f in
                        ("eql/algorithms/adaptive.py", "eql/algorithms/eql.py", "eql/expectiles.py", "experiments/adaptive_pendulum.py", "eql/training.py", "experiments/_pendulum.py")})
    manifest = OUT / "protocol.json"
    if manifest.exists() and read(manifest) != protocol:
        raise RuntimeError("Existing result directory belongs to another protocol")
    save(manifest, protocol)
    init = jax.jit(algo.init_state)
    train = jax.jit(lambda s: jax.lax.fori_loop(0, horizon, lambda _, v: algo.train_iteration(v), s),
                    donate_argnums=(0,))
    evaluate = jax.jit(lambda s, key: algo.eval_callback(algo, s, key)[1])
    for seed in SEEDS:
        path = OUT / f"seed{seed}.json"
        if path.exists():
            continue
        state = init(jax.random.PRNGKey(seed))
        records = []
        for step in range(batch_steps, COMMON["total_timesteps"] + 1, batch_steps):
            old_arm, old_q = int(state.arm), float(state.tau_q)
            state = train(state)
            probs = np.asarray(algo.probabilities(state.logits))
            q = float(state.tau_q)
            eligible = step - batch_steps >= algo.fill_buffer
            assert int(state.global_step) == step and np.all(np.asarray(state.env_state.time) == 0)
            assert 0 <= q <= 1 and np.isclose(probs.sum(), 1.)
            assert np.isfinite(np.asarray(state.soft_return)).all()
            bias = float(state.last_bias)
            if eligible:
                assert (q - old_q) * bias <= 1e-6
                assert np.isclose(np.asarray(state.start_prediction-state.soft_return).mean(), bias, atol=1e-4)
            records.append(dict(step=step, tau_a_used=algo.actor_arms[old_arm], tau_Q_used=old_q,
                                next_tau_Q=q, next_arm_probabilities=probs.tolist(), eligible=eligible,
                                bias=bias if eligible else None, feedback=float(state.last_feedback) if eligible else None,
                                training_return=float(np.asarray(state.raw_return).mean()),
                                alpha=float(np.exp(np.asarray(state.alpha_ts.params["log_alpha"])))))
        assert int(state.actor_ts.step) == int(state.critic_ts.step) == int(state.alpha_ts.step) == 9796
        assert int(state.rounds) == sum(row["eligible"] for row in records) == 48
        returns = np.asarray(evaluate(state, jax.random.PRNGKey(30000 + seed)))
        assert returns.shape == (128,) and np.isfinite(returns).all()
        checkpoint = OUT / f"seed{seed}.msgpack"
        checkpoint.write_bytes(to_bytes(dict(actor=state.actor_ts.params, critics=state.critic_ts.params,
                                             target_critics=state.critic_target_params, alpha=state.alpha_ts.params)))
        save(path, dict(seed=seed, score=float(returns.mean()), returns=returns.tolist(), rounds=records,
                        steps=int(state.global_step), updates=int(state.actor_ts.step),
                        final_tau_Q=q, final_actor_probabilities=probs.tolist(), final_alpha=records[-1]["alpha"]))
        print(f"seed {seed}: {returns.mean():.3f}", flush=True)
    scores = {}
    scores["adaptive"] = [read(OUT / f"seed{s}.json")["score"] for s in SEEDS]
    scores["fixed_Q0.3_a0.7"] = [read(grid_file("EQL", .3, .7, s))["score"] for s in SEEDS]
    scores["SAC"] = [reference(s)["score"] for s in SEEDS]
    summary = {name: dict(seeds=values, mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))
               for name, values in scores.items()}
    save(OUT / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()

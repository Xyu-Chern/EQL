import jax
import jax.numpy as jnp
import numpy as np

from adaptive_eql import AdaptiveEQL
from eql import EQL


def test_controllers_follow_feedback_and_keep_exploration():
    algo = AdaptiveEQL.create(env="Pendulum-v1", num_envs=2)
    state = algo.init_state(jax.random.PRNGKey(12))
    # Historical scores are negative. A less negative score should favor the arm.
    state = state.replace(rounds=jnp.int32(2), arm=jnp.int32(1),
                          return_history=jnp.array([-101., -99., 0., 0., 0.]),
                          raw_return=jnp.array([-70., -70.]), soft_return=jnp.zeros(2))
    for sign in (1, -1):
        sample = state.replace(start_prediction=jnp.full(2, 4. * sign),
                               raw_return=jnp.full(2, -100. + 30. * sign))
        next_state = algo.finish_round(sample)
        assert sign * (next_state.tau_q - sample.tau_q) < 0
        probabilities = algo.probabilities(next_state.logits)
        assert sign * (probabilities[1] - 1 / len(algo.actor_arms)) > 0
        assert np.all(np.asarray(probabilities) >= algo.exploration / len(algo.actor_arms))


def test_original_eql_update_and_alpha_match_before_adaptation():
    common = dict(env="Pendulum-v1", num_envs=2, num_critics=10,
                  batch_size=4, buffer_size=32, fill_buffer=4)
    adaptive = AdaptiveEQL.create(**common)
    key = jax.random.PRNGKey(18)
    new = adaptive.init_state(key)
    fixed = EQL.create(**common, tau_a=adaptive.actor_arms[int(new.arm)], tau_Q=adaptive.initial_q)
    old = fixed.init_state(key)
    assert not hasattr(new, "bags")
    for algo, state in ((fixed, old), (adaptive, new)):
        trained = jax.jit(lambda s: jax.lax.fori_loop(0, 5, lambda _, v: algo.train_iteration(v), s))(state)
        leaves = jax.tree.leaves((trained.critic_ts.params, trained.actor_ts.params,
                                  trained.alpha_ts.params, trained.replay_buffer.data))
        if algo is fixed:
            reference = leaves
        else:
            for expected, actual in zip(reference, leaves, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

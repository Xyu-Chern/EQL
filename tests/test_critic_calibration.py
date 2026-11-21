import jax
import jax.numpy as jnp
import numpy as np

from eql.algorithms.calibration import (
    CriticCalibratedEQL,
    choose_tau,
    frozen_training_block,
    trajectory_losses,
)
from eql.native import NativeSAC, NativeTask


def test_soft_return_omits_current_entropy_and_weights_episodes_equally():
    rewards = np.ones((8, 10))
    dones = np.zeros_like(rewards, bool)
    dones[[1, 3, 6]] = True
    bonus = np.full_like(rewards, 2)
    pred = np.zeros((8, 10, 4))
    # With gamma=1/2 the return labels in the two episodes are
    # [2.5, 1] and [3.25, 2.5, 1]. The first partial episode is discarded.
    losses = trajectory_losses((rewards, dones, bonus, pred), .5, 10)
    np.testing.assert_allclose(losses[::2], (2.5**2 + 1) / 2)
    np.testing.assert_allclose(losses[1::2], (3.25**2 + 2.5**2 + 1) / 3)
    assert trajectory_losses((rewards[:6], dones[:6], bonus[:6], pred[:6]), .5, 10) is None


def test_held_out_errors_can_veto_the_fit_winner():
    seed = 6
    losses = np.ones((20, 4))
    order = np.random.default_rng(seed).permutation(20)
    losses[order[:10], 2] = 0
    losses[order[10:], 2] = 2
    assert choose_tau(losses, 0., np.random.default_rng(seed))[0] == 0
    losses[:, 2] = 0
    assert choose_tau(losses, 0., np.random.default_rng(seed))[0] == .3


def test_zero_endpoint_and_frozen_behavior_use_actual_replay_actions():
    config = dict(num_envs=10, num_critics=2, fill_buffer=4, batch_size=4, buffer_size=64)
    key = jax.random.PRNGKey(91)
    task = NativeTask("Walker2d-v5")
    reference = NativeSAC.create(env=task, **config)
    sac = reference.init_state(key)
    sac = jax.jit(reference.train_iteration)(sac)
    jax.block_until_ready(sac)
    algo = CriticCalibratedEQL.create(env=task, tau_a=0., tau_Q=0., **config)
    state = algo.init_state(key)
    actor = state.actor_ts.params
    target = state.critic_target_params
    alpha = jnp.exp(state.alpha_ts.params["log_alpha"])
    block = frozen_training_block(algo, 1)
    nxt, trace = block(state, actor, target, alpha, jnp.float32(0))
    for name in ("actor_ts", "critic_ts", "alpha_ts"):
        for actual, expected in zip(jax.tree.leaves(getattr(nxt, name).params),
                                    jax.tree.leaves(getattr(sac, name).params), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    # A second iteration uses the initial snapshot after the live actor has updated.
    final, trace = block(nxt, actor, target, alpha, jnp.float32(.3))
    actions = final.replay_buffer.data.action[10:20]
    pred = algo.vmap_critic(target, nxt.last_obs, actions)
    np.testing.assert_allclose(trace[-1][0, :, 0], pred.min(0), atol=1e-5)
    np.testing.assert_allclose(trace[-1][0, :, -1], pred.mean(0), atol=1e-5)
    _, k = jax.random.split(nxt.rng)
    expected_actions, logp = algo.actor.apply(actor, nxt.last_obs, k, method="action_log_prob")
    np.testing.assert_allclose(actions, expected_actions, atol=1e-5)
    np.testing.assert_allclose(trace[2][0], -alpha * logp, atol=1e-5)
    for e in task.instances:
        e.close()

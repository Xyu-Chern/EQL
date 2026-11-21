"""Held-out trajectory calibration of the target critic for native continuous tasks."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from eql.expectiles import compute_expectile_on_axis
from eql.native import NativeEQL

CANDIDATES = (0.0, 0.1, 0.3, 0.5)


def aggregate(qs, tau, iterations):
    return jax.lax.cond(
        tau == 0.5, lambda: qs.mean(0),
        lambda: compute_expectile_on_axis(qs, tau, axis=0, num_iters=iterations),
    )


class CriticCalibratedEQL(NativeEQL):
    behavior_params: object = struct.field(pytree_node=True, default=None)

    def aggregate_target(self, qs):
        return aggregate(qs, self.tau_Q, self.num_iters)

    def update_actor(self, ts, mb):
        # Explicitly use the inherited SAC loss, independently of tau_Q.
        from rejax.algos.sac import SAC
        return SAC.update_actor(self, ts, mb)

    def collect_transitions(self, ts):
        if self.behavior_params is None:
            return super().collect_transitions(ts)
        frozen = ts.actor_ts.replace(params=self.behavior_params)
        collected, batch = super().collect_transitions(ts.replace(actor_ts=frozen))
        return collected.replace(actor_ts=ts.actor_ts), batch


def frozen_training_block(algo, length):
    @jax.jit
    def block(state, actor, target, alpha, tau):
        active = algo.replace(behavior_params=actor, tau_Q=tau)

        def train(s, _):
            _, key = jax.random.split(s.rng)
            action, logp = algo.actor.apply(actor, s.last_obs, key, method="action_log_prob")
            qs = algo.vmap_critic(target, s.last_obs, action)
            preds = jnp.stack([aggregate(qs, t, algo.num_iters) for t in CANDIDATES], axis=-1)
            nxt = active.train_iteration(s)
            indices = (s.replay_buffer.index + jnp.arange(algo.num_envs)) % s.replay_buffer.size
            data = nxt.replay_buffer.data
            return nxt, (data.reward[indices], data.done[indices], -alpha * logp, preds)

        return jax.lax.scan(train, state, None, length=length)

    return block


def trajectory_losses(trace, gamma, lanes):
    reward, done, entropy, predicted = trace
    losses = []
    for lane in range(lanes):
        # The first episode may have started with the previous behavior policy.
        ends = np.flatnonzero(done[:, lane])
        if len(ends) < 3:
            return None
        for left, right in zip(ends[:2] + 1, ends[1:3]):
            future = 0.0
            total = np.zeros(len(CANDIDATES))
            for t in range(right, left - 1, -1):
                soft_q = float(reward[t, lane]) + gamma * future
                total += (predicted[t, lane].astype(float) - soft_q) ** 2
                future = soft_q + float(entropy[t, lane])
            losses.append(total / (right - left + 1))
    return np.asarray(losses)


def choose_tau(losses, current, rng):
    if losses.shape != (20, len(CANDIDATES)) or not np.isfinite(losses).all():
        raise ValueError("Expected finite candidate losses for 20 complete trajectories")
    order = rng.permutation(len(losses))
    fit = losses[order[:10]].mean(0)
    old = CANDIDATES.index(current)
    best = old if fit[old] == fit.min() else int(fit.argmin())
    gain = losses[order[10:], old] - losses[order[10:], best]
    se = gain.std(ddof=1) / np.sqrt(len(gain))
    selected = CANDIDATES[best] if gain.mean() > se else current
    return selected, dict(fit=fit.tolist(), proposed=CANDIDATES[best], gain=float(gain.mean()),
                          standard_error=float(se), fit_indices=order[:10].tolist(),
                          episode_losses=losses.tolist())

"""Shared-replay EQL with episode feedback for synchronized Gymnax Pendulum."""
import jax
import jax.numpy as jnp
from flax import struct
from rejax.algos.algorithm import register_init

from eql.algorithms.eql import EQL
from eql.expectiles import compute_expectile_on_axis


class AdaptiveEQL(EQL):
    # Configuration-free call with respect to the two expectiles. Controller
    # constants are explicit so a run can be reproduced.
    actor_arms: tuple = struct.field(pytree_node=False, default=tuple(i / 10 for i in range(11)))
    initial_q: float = struct.field(pytree_node=False, default=.5)
    q_rate: float = struct.field(pytree_node=False, default=.05)
    bandit_rate: float = struct.field(pytree_node=False, default=.05)
    exploration: float = struct.field(pytree_node=False, default=.1)

    def supports_episode_feedback(self):
        return self.env.name in ("Pendulum-v1", "Reacher-misc")

    @register_init
    def init_base_state(self, rng):
        if not self.supports_episode_feedback() or self.normalize_rewards or self.normalize_observations:
            raise ValueError("Unsupported environment or normalization for episode feedback.")
        values = super().init_base_state(rng)
        # A separate RNG stream leaves EQL network, replay and action keys intact.
        key, draw = jax.random.split(jax.random.fold_in(rng, 4721))
        size = len(self.actor_arms)
        zeros = jnp.zeros(self.num_envs)
        values.update(
            bandit_key=key, arm=jax.random.randint(draw, (), 0, size),
            logits=jnp.zeros(size), tau_q=jnp.float32(self.initial_q),
            start_prediction=zeros, soft_return=zeros, raw_return=zeros,
            return_history=jnp.zeros(5), rounds=jnp.int32(0),
            last_bias=jnp.float32(0), last_return=jnp.float32(0),
            last_feedback=jnp.float32(0),
        )
        return values

    def probabilities(self, logits):
        return (1 - self.exploration) * jax.nn.softmax(logits) + self.exploration / len(self.actor_arms)

    def update_actor(self, ts, mb):
        active = self.replace(tau_a=jnp.asarray(self.actor_arms)[ts.arm])
        return EQL.update_actor(active, ts, mb)

    def update_critic(self, ts, mb):
        return EQL.update_critic(self.replace(tau_Q=ts.tau_q), ts, mb)

    def collect_transitions(self, ts):
        # Reuse the same sampling key as SAC.collect_transitions so logp is for
        # the action actually taken, including the original squashing convention.
        _, action_key = jax.random.split(ts.rng)
        _, logp = self.actor.apply(ts.actor_ts.params, ts.last_obs, action_key,
                                   method="action_log_prob")
        collected, batch = super().collect_transitions(ts)
        age = ts.env_state.time[0]
        first = age == 0

        def predict():
            qs = self.vmap_critic(ts.critic_ts.params, batch.obs, batch.action)
            return compute_expectile_on_axis(qs, ts.tau_q, axis=0, num_iters=self.num_iters)

        prediction = jax.lax.cond(first, predict, lambda: ts.start_prediction)
        # Soft Q conditions on action zero; entropy first enters at action one.
        entropy_bonus = -jnp.exp(ts.alpha_ts.params["log_alpha"]) * logp
        soft = batch.reward + jnp.where(first, 0., entropy_bonus)
        return collected.replace(
            start_prediction=prediction,
            soft_return=jnp.where(first, 0., ts.soft_return) + self.gamma ** age * soft,
            raw_return=jnp.where(first, 0., ts.raw_return) + batch.reward,
        ), batch

    def finish_round(self, ts):
        error = ts.start_prediction - ts.soft_return
        bias = error.mean()
        # Bounded step with the sign of the mean calibration error.
        q = jnp.clip(ts.tau_q - self.q_rate * bias / (jnp.abs(error).mean() + 1e-6), 0., 1.)

        score = ts.raw_return.mean()
        history = ts.return_history
        count = jnp.minimum(ts.rounds, history.size)
        valid = jnp.arange(history.size) < count
        mean = jnp.where(valid, history, 0.).sum() / jnp.maximum(1, count)
        variance = jnp.where(valid, (history - mean) ** 2, 0.).sum() / jnp.maximum(1, count)
        feedback = jnp.where(count >= 2, jnp.tanh((score - mean) / jnp.sqrt(variance + 1.)), 0.)
        probs = self.probabilities(ts.logits)
        logits = ts.logits.at[ts.arm].add(self.bandit_rate * feedback / probs[ts.arm])
        logits = logits - logits.max()
        key, draw = jax.random.split(ts.bandit_key)
        arm = jax.random.categorical(draw, jnp.log(self.probabilities(logits)))
        return ts.replace(tau_q=q, logits=logits, arm=arm, bandit_key=key,
                          return_history=history.at[ts.rounds % history.size].set(score),
                          rounds=ts.rounds + 1, last_bias=bias, last_return=score,
                          last_feedback=feedback)

    def train_iteration(self, ts):
        ts = super().train_iteration(ts)
        # Only credit complete episodes that started after replay warmup.
        episode_steps = self.num_envs * self.env_params.max_steps_in_episode
        eligible = (ts.env_state.time[0] == 0) & (ts.global_step - episode_steps >= self.fill_buffer)
        return jax.lax.cond(eligible, self.finish_round, lambda state: state, ts)

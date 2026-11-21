"""Bagging extension of the user's unmodified eql.py (continuous tasks)."""
import jax
import jax.numpy as jnp
import optax
from flax import struct
from rejax.algos.algorithm import register_init

from eql.algorithms.eql import EQL
from eql.expectiles import compute_expectile_on_axis
from eql.replay import BagReplayBuffer


class OriginalBagEQL(EQL):
    num_duplicates: int = struct.field(pytree_node=False, default=2)

    @register_init
    def initialize_replay_buffer(self, rng):
        values = super().initialize_replay_buffer(rng)
        example = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), values['replay_buffer'].data)
        values['bags'] = BagReplayBuffer.empty(example, self.num_critics, self.buffer_size)
        return values

    def sample_bags(self, bags, key):
        # Uniform sampling without replacement, as in original Rejax replay.
        # IID continuous random ranks + top-k avoid sorting the entire buffer.
        scores = jax.random.uniform(key, (self.num_critics, self.buffer_size))
        scores = jnp.where(jnp.arange(self.buffer_size) < bags.count, scores, -jnp.inf)
        indices = jax.lax.top_k(scores, self.batch_size)[1]
        return jax.tree.map(lambda x: x[jnp.arange(self.num_critics)[:, None], indices], bags.data)

    def update_bag_critic(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        n, b = mb.reward.shape
        obs = mb.next_obs.reshape((n*b, *mb.next_obs.shape[2:]))
        action, logprob = self.actor.apply(ts.actor_ts.params, obs, action_rng, method='action_log_prob')
        qs = self.vmap_critic(ts.critic_target_params, obs, action)
        next_q = compute_expectile_on_axis(qs, self.tau_Q, axis=0, num_iters=self.num_iters)
        q_target = (next_q - jnp.exp(ts.alpha_ts.params['log_alpha'])*logprob).reshape(n,b)
        target = mb.reward + self.gamma*(1-mb.done)*q_target
        def loss(params):
            pred = jax.vmap(self.critic.apply, in_axes=(0,0,0))(params, mb.obs, mb.action)
            return optax.l2_loss(pred, target).sum(axis=0).mean()
        grad = jax.grad(loss)(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grad))

    def train_iteration(self, ts):
        previous_step = ts.global_step
        ts, batch = self.collect_transitions(ts)  # original policy warmup/done semantics
        # Bag RNG is separate: adding bags does not consume the original training RNG.
        key = jax.random.fold_in(ts.rng, ts.global_step)
        bags, _ = ts.bags.add(jax.tree.map(lambda x:x[:,None], batch), key, self.num_duplicates)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch), bags=bags)
        def update(s):
            rng, sample_rng = jax.random.split(s.rng)
            actor_mb = s.replay_buffer.sample(self.batch_size, sample_rng)
            critic_mb = self.sample_bags(s.bags, jax.random.fold_in(sample_rng, 917))
            s = s.replace(rng=rng)
            s, lp = self.update_actor(s, actor_mb)  # inherited original EQL actor
            s = self.update_bag_critic(s, critic_mb)
            return self.update_alpha(s, lp)
        ts = jax.lax.cond(ts.global_step > self.fill_buffer,
            lambda s:jax.lax.fori_loop(0,self.num_epochs,lambda _,v:update(v),s), lambda s:s, ts)
        return ts.replace(critic_target_params=self.maybe_update_target_params(
            ts.critic_ts.params,ts.critic_target_params,ts.global_step,previous_step))

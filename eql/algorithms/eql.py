import jax
import jax.numpy as jnp
import optax
from flax import struct
from rejax.algos.sac import SAC

from eql.expectiles import compute_expectile_on_axis


class EQL(SAC):
    tau_a: float = struct.field(pytree_node=True, default=0.7)
    tau_Q: float = struct.field(pytree_node=True, default=0.3)
    num_iters: int = struct.field(pytree_node=False, default=10)

    def aggregate_target(self, qs):
        return compute_expectile_on_axis(qs, self.tau_Q, axis=0, num_iters=self.num_iters)

    def update_critic(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])
        def critic_loss_fn(params):
            if self.discrete:
                action_dist = self.actor.apply(
                    ts.actor_ts.params, mb.next_obs, method="_action_dist"
                )
                logprob = jnp.log(action_dist.probs)
                qs = self.vmap_critic(ts.critic_target_params, mb.next_obs)

                next_q = self.aggregate_target(qs)
                q_target = next_q - alpha * logprob
                q_target = jnp.sum(jnp.exp(logprob) * q_target, axis=1)
                qs = jax.vmap(
                    lambda *args: self.critic.apply(*args, method="take"),
                    in_axes=(0, None, None),
                )(params, mb.obs, mb.action)

            else:

                action, logprob = self.actor.apply(
                    ts.actor_ts.params,
                    mb.next_obs,
                    action_rng,
                    method="action_log_prob",
                )
                qs = self.vmap_critic(ts.critic_target_params, mb.next_obs, action)
                next_q = self.aggregate_target(qs)
                q_target = next_q - alpha * logprob
                qs = self.vmap_critic(params, mb.obs, mb.action)

            target = mb.reward + self.gamma * (1 - mb.done) * q_target
            losses = jax.vmap(lambda q: optax.l2_loss(q, target))(qs)
            critic_loss = losses.sum(axis=0).mean()

            return critic_loss

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts

    def update_actor(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def actor_loss_fn(actor_params):
            if self.discrete:
                logprob = jnp.log(
                    self.actor.apply(actor_params, mb.obs, method="_action_dist").probs
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs)
                Q_expectiles = compute_expectile_on_axis(
                    qs,
                    tau=self.tau_a,
                    axis=0,
                    num_iters=self.num_iters
                )
                loss_pi = alpha * logprob - Q_expectiles
                loss_pi = jnp.sum(jnp.exp(logprob) * loss_pi, axis=1)
                total_loss = loss_pi.mean()

            else:

                actions, logprob = self.actor.apply(
                    actor_params, mb.obs, action_rng, method="action_log_prob"
                )

                qs = self.vmap_critic(ts.critic_ts.params, mb.obs, actions)

                Q_expectiles = compute_expectile_on_axis(
                    qs,
                    tau=self.tau_a,
                    axis=0,
                    num_iters=self.num_iters
                )
                loss_pi = alpha * logprob - Q_expectiles

                total_loss = loss_pi.mean()

            return total_loss, logprob

        grads, logprob = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, logprob

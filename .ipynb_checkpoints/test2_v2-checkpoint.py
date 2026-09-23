from functools import partial
import jax
import jax.numpy as jnp
import optax
from rejax.algos.sac import SAC
from flax import struct
import chex


@partial(jax.jit, static_argnames=('axis', 'num_iters'))
def compute_expectile_on_axis(data, tau, axis, num_iters=10):
    """E^tau(x) = argmin_y E[ rho_tau(x - y) ] where rho_tau(u) = |tau - 1_{u < 0}| * u^2"""
    y = jnp.mean(data, axis=axis, keepdims=True)
    
    def body_fn(i, current_y):
        diff = data - current_y
        weight = jnp.where(diff < 0, 1.0 - tau, tau)
        numerator = jnp.sum(weight * data, axis=axis, keepdims=True)
        denominator = jnp.sum(weight, axis=axis, keepdims=True)
        return numerator / (denominator + 1e-6)
    
    final_y = jax.lax.fori_loop(0, num_iters, body_fn, y)
    return jnp.squeeze(final_y, axis=axis)


@partial(jax.jit, static_argnames=('num_iters',))
def compute_double_expectile(Q_values, tau_a, tau_Q, num_iters=10):
    """
    Q_values 形状: (batch, num_actions/action_ensemble, num_ensembles)
    - axis=2: 对 Critic Ensemble 计算 expectile (tau_Q)
    - axis=1: 对 Action 维度/样本 计算 expectile (tau_a)
    """
    Q_expectiles = compute_expectile_on_axis(
        Q_values, 
        tau=tau_Q, 
        axis=2,
        num_iters=num_iters
    )
    final_result = compute_expectile_on_axis(
        Q_expectiles,
        tau=tau_a,
        axis=1,
        num_iters=num_iters
    )
    return final_result


@struct.dataclass
class EQL(SAC):
    num_critics: int = struct.field(pytree_node=False, default=10)
    tau_a: float = struct.field(pytree_node=False, default=0.7)
    tau_Q: float = struct.field(pytree_node=False, default=0.7)
    num_iters: int = struct.field(pytree_node=False, default=10)
    action_ensemble: int = struct.field(pytree_node=False, default=10)
    actor_noise: float = struct.field(pytree_node=False, default=0.2)
    actor_noise_clip: float = struct.field(pytree_node=False, default=0.5)
    normalize_q_loss: bool = struct.field(pytree_node=False, default=False)
    eql_alpha: float = struct.field(pytree_node=False, default=10.0)  

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_dict = super().create_agent(config, env, env_params)
        return agent_dict

    @property
    def vmap_critic(self, method="__call__"):
        def apply(*args):
            return self.critic.apply(*args, method=method)

        if self.discrete:
            return jax.vmap(apply, in_axes=(0, None))
        return jax.vmap(apply, in_axes=(0, None, None))

    def update_critic(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def critic_loss_fn(params):
            if self.discrete:
                next_qs = self.vmap_critic(ts.critic_target_params, mb.next_obs)
                next_qs = jnp.transpose(next_qs, (1, 2, 0))  # (batch, num_actions, num_ensembles)

                next_q = compute_double_expectile(
                    next_qs,
                    tau_a=self.tau_a,
                    tau_Q=self.tau_Q,
                    num_iters=self.num_iters
                )

                qs = jax.vmap(
                    lambda *args: self.critic.apply(*args, method="take"),
                    in_axes=(0, None, None),
                )(params, mb.obs, mb.action)

            else:
                sample_rngs = jax.random.split(action_rng, self.action_ensemble)

                def sample_fn(key):
                    action, _ = self.actor.apply(
                        ts.actor_ts.params,
                        mb.next_obs,
                        key,
                        method="action_log_prob",
                    )
                    rng_noise, _ = jax.random.split(key)
                    noise = jnp.clip(
                        jax.random.normal(rng_noise, action.shape) * self.actor_noise,
                        -self.actor_noise_clip,
                        self.actor_noise_clip,
                    )
                    return jnp.clip(action + noise, -1.0, 1.0)

                actions = jax.vmap(sample_fn)(sample_rngs)
                actions = jnp.transpose(actions, (1, 0, 2))

                def compute_Q(a_slice):
                    return self.vmap_critic(ts.critic_target_params, mb.next_obs, a_slice)

                next_qs = jax.vmap(compute_Q, in_axes=1)(actions)
                next_qs = jnp.transpose(next_qs, (2, 0, 1))
                next_q = compute_double_expectile(
                    next_qs,
                    tau_a=self.tau_a,
                    tau_Q=self.tau_Q,
                    num_iters=self.num_iters
                )

                qs = self.vmap_critic(params, mb.obs, mb.action)

            target_q = mb.reward + self.gamma * (1 - mb.done) * next_q
            losses = jax.vmap(lambda q: optax.l2_loss(q, target_q))(qs)
            critic_loss = losses.sum(axis=0).mean()

            if self.normalize_q_loss:
                lam = 1.0 / jax.lax.stop_gradient(jnp.abs(qs).mean() + 1e-6)
                critic_loss = critic_loss * lam

            return critic_loss

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts

    def update_actor(self, ts, mb):
        rng, sample_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def actor_loss_fn(actor_params):
            if self.discrete:
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs)
                qs = jnp.transpose(qs, (1, 2, 0))

                Q_expectiles = compute_expectile_on_axis(
                    qs,
                    tau=self.tau_Q,
                    axis=2,
                    num_iters=self.num_iters
                )

                y_expanded = jnp.expand_dims(Q_expectiles, axis=-1)
                w = jnp.abs(self.tau_Q - (qs < y_expanded).astype(jnp.float32))
                w_tilde = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-6)
                w_tilde = jax.lax.stop_gradient(w_tilde)

                q_objective = jnp.sum(w_tilde * qs, axis=-1)

                action_logits = self.actor.apply(actor_params, mb.obs, method="_action_dist").logits
                log_probs = jax.nn.log_softmax(action_logits, axis=-1)
                probs = jax.nn.softmax(action_logits, axis=-1)

                actor_q_loss = -jnp.sum(probs * q_objective, axis=-1).mean()
                bc_loss = -jnp.sum(jax.nn.one_hot(mb.action, action_logits.shape[-1]) * log_probs, axis=-1).mean()
                
                total_loss = self.eql_alpha * actor_q_loss + bc_loss

            else:
                actions, _ = self.actor.apply(
                    actor_params,
                    mb.obs,
                    sample_rng,
                    method="action_log_prob"
                )

                noise = jnp.clip(
                    jax.random.normal(sample_rng, actions.shape) * self.actor_noise,
                    -self.actor_noise_clip,
                    self.actor_noise_clip,
                )
                actions = jnp.clip(actions + noise, -1.0, 1.0)

                qs = self.vmap_critic(ts.critic_ts.params, mb.obs, actions)
                Q_values = jnp.transpose(qs, (1, 0))

                Q_expectiles = compute_expectile_on_axis(
                    Q_values,
                    tau=self.tau_Q,
                    axis=1,
                    num_iters=self.num_iters
                )

                y_expanded = jnp.expand_dims(Q_expectiles, axis=-1)
                w = jnp.abs(self.tau_Q - (Q_values < y_expanded).astype(jnp.float32))

                w_tilde = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-6)
                w_tilde = jax.lax.stop_gradient(w_tilde)

                q_objective = jnp.sum(w_tilde * Q_values, axis=-1)

                bc_loss = jnp.square(actions - mb.action).sum(axis=-1).mean()
                total_loss = bc_loss - self.eql_alpha * q_objective.mean()

            return total_loss, {
                "actor_loss": total_loss,
                "bc_loss": bc_loss,
                "q_objective": q_objective.mean(),
            }

        grads, aux = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, None

    def update_alpha(self, ts, logprob):
        return ts


if __name__ == "__main__":
    import time
    from matplotlib import pyplot as plt

    env_str = "Pendulum-v1"
    TOTAL_TIMESTEPS = 100_000

    # 1. 设置实验参数变量
    tau_a_val = 0.7
    tau_Q_val = 0.3
    action_ens_val = 10
    q_ens_val = 50
    eql_alpha = 10.0

    # 实例化自定义 EQL 算法
    algo = EQL.create(
        env=env_str,
        learning_rate=3e-4,
        total_timesteps=TOTAL_TIMESTEPS,
        num_critics=q_ens_val,          # Q ensemble 数量
        action_ensemble=action_ens_val, # Action ensemble 数量
        tau_a=tau_a_val,
        tau_Q=tau_Q_val,
        eql_alpha=eql_alpha,
        gamma=0.98,                     # 预防 CartPole 崩溃的衰减因子
    )

    print(f"Algorithm instantiated: EQL on {env_str}")
    
    train_fn = jax.jit(algo.train)
    rng = jax.random.PRNGKey(0)

    start_time = time.time()
    train_state, evaluation = train_fn(rng)
    time_elapsed = time.time() - start_time

    sps = TOTAL_TIMESTEPS / time_elapsed
    print(f"Finished training in {time_elapsed:.2f} seconds ({sps:.2f} steps/second).")

    # 2. 绘制曲线并输出超参数细节
    ep_lengths, ep_returns = evaluation
    mean_return = ep_returns.mean(axis=-1)

    plt.figure(figsize=(9, 5.5))
    plt.plot(jnp.linspace(0, TOTAL_TIMESTEPS, len(mean_return)), mean_return, linewidth=1.5, color="tab:blue")
    plt.xlabel("Environment step")
    plt.ylabel("Episodic return")

    # 主标题与副标题（显示当前超参数）
    plt.suptitle(f"Training EQL on {env_str}", fontsize=14, fontweight="bold")
    plt.title(
        f"$\\tau_a$={tau_a_val} | $\\tau_Q$={tau_Q_val} | Action Ens={action_ens_val} | Q Ens={q_ens_val}",
        fontsize=11,
        color="darkslategrey"
    )

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # 3. 动态拼接文件名并保存图像
    save_filename = f"{env_str}_eql_taua{tau_a_val}_tauQ{tau_Q_val}_actEns{action_ens_val}_qEns{q_ens_val}_alpha{eql_alpha}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    print(f"图像已成功保存为: {save_filename}")
    plt.show()
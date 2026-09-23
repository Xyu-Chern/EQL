from functools import partial
import jax
import jax.numpy as jnp
import optax
from rejax.algos.sac import SAC
from flax import struct
import chex
from rejax.algos.sac import SAC

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


class EQL(SAC):

    num_critics: int = struct.field(pytree_node=False, default=10) # 任意数字，例如 10
    tau_a: chex.Scalar = struct.field(pytree_node=True, default=0.7)
    tau_Q: chex.Scalar = struct.field(pytree_node=True, default=0.7)
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
        '''
        它的 apply(params, obs) 只设计用来接收一套参数（即单个网络）。
        它根本不知道你传入的 params 里面其实装了 10 个 Q 网络的参数。
        '''
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
                next_qs = self.vmap_critic(ts.critic_target_params, mb.next_obs) # 它会一次性并行输出当前状态下所有 K 个动作的 Q 值  (num_ensembles, batch, num_actions)
                next_qs = jnp.transpose(next_qs, (1, 2, 0))  #  (batch, num_actions, num_ensembles)

                next_q = compute_double_expectile(
                    next_qs,
                    tau_a=self.tau_a,
                    tau_Q=self.tau_Q,
                    num_iters=self.num_iters
                )  #  (batch,)

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

                next_qs = jax.vmap(compute_Q, in_axes=1)(actions)   # (action_ensemble, num_ensembles, batch)
                next_qs = jnp.transpose(next_qs, (2, 0, 1))         #  (batch, action_ensemble, num_ensembles)
                next_q = compute_double_expectile(
                    next_qs,
                    tau_a=self.tau_a,
                    tau_Q=self.tau_Q,
                    num_iters=self.num_iters
                )     # (batch,)

                qs = self.vmap_critic(params, mb.obs, mb.action)      #  (num_ensembles, batch)

            target_q = mb.reward + self.gamma * (1 - mb.done) * next_q
            losses = jax.vmap(lambda q: optax.l2_loss(q, target_q))(qs)   # Target Q 扩维至 (1, batch) 以便广播到 (num_ensembles, batch)
            critic_loss = losses.sum(axis=0).mean()

            if self.normalize_q_loss:
                lam = 1.0 / jax.lax.stop_gradient(jnp.abs(qs).mean())
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
                # ---------------- 1. 离散动作空间 ----------------
                # 获取所有动作的 Q 值: (num_critics, batch, num_actions)
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs)
                # 转置为 (batch, num_actions, num_critics)
                qs = jnp.transpose(qs, (1, 2, 0))

                # 对 Critic Ensemble 维度 (axis=2) 计算 Expectile
                Q_expectiles = compute_expectile_on_axis(
                    qs,
                    tau=self.tau_Q,
                    axis=2,
                    num_iters=self.num_iters
                ) # (batch, num_actions)

                # 权重计算与包络定理截断
                y_expanded = jnp.expand_dims(Q_expectiles, axis=-1)
                w = jnp.abs(self.tau_Q - (qs < y_expanded).astype(jnp.float32))
                w_tilde = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-6)
                w_tilde = jax.lax.stop_gradient(w_tilde)

                # 根据包络定理计算加权 Q 目标: (batch, num_actions)
                q_objective = jnp.sum(w_tilde * qs, axis=-1)

                # 从 DiscretePolicy 中计算 log_prob 与 交叉熵 loss
                action_logits = self.actor.apply(actor_params, mb.obs, method="_action_dist").logits
                log_probs = jax.nn.log_softmax(action_logits, axis=-1)
                probs = jax.nn.softmax(action_logits, axis=-1)

                # Policy Loss: 期望 Q 值最大化 (交叉熵/加权) + BC Loss
                actor_q_loss = -jnp.sum(probs * q_objective, axis=-1).mean()
                bc_loss = -jnp.sum(jax.nn.one_hot(mb.action, self.critic.action_dim) * log_probs, axis=-1).mean()
                
                total_loss = self.eql_alpha * actor_q_loss + bc_loss

            else:
                # ---------------- 2. 连续动作空间 ----------------
                # 从 Actor 中通过采样获取当前 policy 预测的 actions
                actions, _ = self.actor.apply(
                    actor_params,
                    mb.obs,
                    sample_rng,
                    method="action_log_prob"
                )

                # 加入探查噪声 (Reparameterization Trick)
                noise = jnp.clip(
                    jax.random.normal(sample_rng, actions.shape) * self.actor_noise,
                    -self.actor_noise_clip,
                    self.actor_noise_clip,
                )
                actions = jnp.clip(actions + noise, -1.0, 1.0)

                # 获取当前所有 Q 网络的输出: (num_critics, batch)
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs, actions)
                # 转置为 (batch, num_critics)
                Q_values = jnp.transpose(qs, (1, 0))

                # 计算对 Critic 维度的 Expectile (axis=1)
                Q_expectiles = compute_expectile_on_axis(
                    Q_values,
                    tau=self.tau_Q,
                    axis=1,
                    num_iters=self.num_iters
                ) # (batch,)

                # 应用包络定理 (Envelope Theorem) 进行梯度截断 (stop_gradient)
                y_expanded = jnp.expand_dims(Q_expectiles, axis=-1)
                w = jnp.abs(self.tau_Q - (Q_values < y_expanded).astype(jnp.float32))

                # 归一化得到 w̃_i，确保 ∑ w̃_i = 1，并截断权重梯度
                w_tilde = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-6)
                w_tilde = jax.lax.stop_gradient(w_tilde)

                # 包络定理下对 Q_values 的加权期望目标
                q_objective = jnp.sum(w_tilde * Q_values, axis=-1)

                # 行为克隆损失 (Behavior Cloning Loss) 与 Total Loss
                bc_loss = jnp.square(actions - mb.action).sum(axis=-1).mean()
                total_loss = bc_loss - self.eql_alpha * q_objective.mean()

            return total_loss, {
                "actor_loss": total_loss,
                "bc_loss": bc_loss,
                "q_objective": q_objective.mean(),
            }

        # 计算梯度并更新 actor 的 TrainState (actor_ts)
        grads, aux = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, None

    def update_alpha(self, ts, logprob):
        # EQL 不需要 SAC 的动态温度更新，直接跳过并返回原 ts
        return ts

if __name__ == "__main__":
    import time
    from matplotlib import pyplot as plt

    env_str = "CartPole-v1"
    TOTAL_TIMESTEPS = 50_000

    # 实例化自定义 EQL 算法
    algo = EQL.create(
        env=env_str,
        learning_rate=3e-4,
        total_timesteps=TOTAL_TIMESTEPS,
        num_critics=10,        # 扩展到 10 个 Ensemble 网络
        tau_a=0.7,
        tau_Q=0.7,
        eql_alpha=1.0,
    )

    print(f"Algorithm instantiated: EQL on {env_str}")
    
    train_fn = jax.jit(algo.train)
    rng = jax.random.PRNGKey(0)

    start_time = time.time()
    train_state, evaluation = train_fn(rng)
    time_elapsed = time.time() - start_time

    sps = TOTAL_TIMESTEPS / time_elapsed
    print(f"Finished training in {time_elapsed:.2f} seconds ({sps:.2f} steps/second).")

    # 绘制曲线
    ep_lengths, ep_returns = evaluation
    mean_return = ep_returns.mean(axis=-1)

    plt.figure(figsize=(8, 5))
    plt.plot(jnp.linspace(0, TOTAL_TIMESTEPS, len(mean_return)), mean_return, linewidth=1.5)
    plt.xlabel("Environment step")
    plt.ylabel("Episodic return")
    plt.title(f"Training EQL on {env_str}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("cartpole_eql_returns.png", dpi=300, bbox_inches="tight")
    plt.show()

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
    tau = jnp.clip(tau, 0.0001, 0.99999)
    def body_fn(i, current_y):
        diff = data - current_y
        weight = jnp.where(diff <= 0, 1.0 - tau, tau)
        numerator = jnp.sum(weight * data, axis=axis, keepdims=True)
        denominator = jnp.sum(weight, axis=axis, keepdims=True)
        return numerator / (denominator + 1e-6)
    
    final_y = jax.lax.fori_loop(0, num_iters, body_fn, y)
    return jnp.squeeze(final_y, axis=axis)




class EQL(SAC):
    tau_a: float = struct.field(pytree_node=True, default=0.7)
    tau_Q: float = struct.field(pytree_node=True, default=0.3)
    num_iters: int = struct.field(pytree_node=False, default=10)

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
                
                next_q = compute_expectile_on_axis(
                    qs,
                    tau=self.tau_Q,
                    axis=0,
                    num_iters=self.num_iters
                )
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
                next_q = compute_expectile_on_axis(
                    qs,
                    tau=self.tau_Q,
                    axis=0,
                    num_iters=self.num_iters
                )
       #         print ("next_q",next_q.shape) 
       #         print ("logprob",logprob.shape) 
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



if __name__ == "__main__":
    import time
    from matplotlib import pyplot as plt
    import itertools
    import math

    env_str =  "CartPole-v1"  #

    TOTAL_TIMESTEPS = 100_000

    # 1. 设置实验参数变量
    action_ens_val = 1
    q_ens_val = 10

    num_seeds = 4  

    tau_a_values = [0.00, 0.25, 0.5, 0.75, 1.0]
    tau_Q_values = [0.00, 0.25, 0.5,0.75]
    # Generate the Cartesian product of hyperparameters.
    config_pairs = list(itertools.product(tau_a_values, tau_Q_values))
    # Convert list-of-dicts into dict-of-arrays.
    config_batch = jnp.asarray(config_pairs,dtype=jnp.float32)
    num_configs = len(config_pairs)
    print(f"Number of configurations: {num_configs}")
    print ("config_batch",config_batch.shape)

    def make_config(config):
        return EQL.create(
            env=env_str,
            learning_rate=3e-4,
            total_timesteps=TOTAL_TIMESTEPS,
            num_critics=q_ens_val,          # Q ensemble 数量
            tau_a=config[0],
            tau_Q=config[1],
            target_entropy_ratio=0.7,
            gamma=0.98,                     # 预防 CartPole 崩溃的衰减因子
        )

    eql = jax.vmap(make_config)(config_batch)
   # eql = [make_config(config) for config in config_batch]
    print ("eql",eql)
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, num_seeds)

    print(f"Algorithm instantiated: EQL on {env_str}")
    
    train_fn = jax.jit(EQL.train)
    train_fn = jax.vmap(train_fn, in_axes=(None, 0))  # Vmap across seeds
    train_fn = jax.vmap(train_fn, in_axes=(0, None))  # Vmap across configs

    start_time = time.time()
    train_state, evaluation = train_fn(eql,rngs)
    time_elapsed = time.time() - start_time

    sps = TOTAL_TIMESTEPS / time_elapsed
    print(f"Finished training in {time_elapsed:.2f} seconds ({sps:.2f} steps/second).")

    # 2. 绘制曲线并输出超参数细节import math
    import numpy as np
    ep_lengths, ep_returns = evaluation
    
    # Expected leading dimensions:
    # ep_returns.shape = (num_configs, num_seeds, num_evaluations, ...)
    print("ep_lengths shape:", ep_lengths.shape)
    print("ep_returns shape:", ep_returns.shape)
    
    if ep_returns.shape[0] != num_configs:
        raise ValueError(
        f"Expected config dimension {num_configs}, "
        f"but ep_returns.shape[0] is {ep_returns.shape[0]}."
    )
    
    if ep_returns.shape[1] != num_seeds:
        raise ValueError(
        f"Expected seed dimension {num_seeds}, "
        f"but ep_returns.shape[1] is {ep_returns.shape[1]}."
    )
    
    # If the final dimension contains multiple evaluation episodes
    # or parallel environments, average within each seed first.
    #
    # Result:
    # returns_per_seed.shape =
    # (num_configs, num_seeds, num_evaluations)
    if ep_returns.ndim == 4:
        returns_per_seed = ep_returns.mean(axis=-1)
    elif ep_returns.ndim == 3:
        returns_per_seed = ep_returns
    else:
        raise ValueError(
    "Expected ep_returns to have shape "
    "(configs, seeds, evaluations) or "
    "(configs, seeds, evaluations, episodes), "
    f"but got {ep_returns.shape}."
    )
    
    # Mean and standard error across the seed dimension, axis=1.
    mean_returns = returns_per_seed.mean(axis=1)
    
    seed_std = returns_per_seed.std(
        axis=1,
        ddof=1,
    )
    
    standard_error = seed_std / jnp.sqrt(num_seeds)
    
    # Approximate 95% normal confidence interval.
    confidence_radius = 1.96 * standard_error
    
    confidence_lower = mean_returns - confidence_radius
    confidence_upper = mean_returns + confidence_radius
    
    # Move arrays from JAX device memory to NumPy for Matplotlib.
    returns_per_seed_np = np.asarray(
    jax.device_get(returns_per_seed)
    )
    mean_returns_np = np.asarray(
    jax.device_get(mean_returns)
    )
    confidence_lower_np = np.asarray(
    jax.device_get(confidence_lower)
    )
    confidence_upper_np = np.asarray(
    jax.device_get(confidence_upper)
    )
    
    num_evaluations = mean_returns_np.shape[-1]
    
    environment_steps = np.linspace(
    0,
    TOTAL_TIMESTEPS,
    num_evaluations,
    )
    
    # Create an approximately square subplot grid.
    num_columns = len(tau_Q_values)
    num_rows = math.ceil(num_configs / num_columns)
    
    fig, axes = plt.subplots(
    num_rows,
    num_columns,
    figsize=(5.2 * num_columns, 3.8 * num_rows),
    sharex=True,
    sharey=True,
    squeeze=False,
    )
    
    axes = axes.flatten()
    
    for config_index, (tau_a, tau_Q) in enumerate(config_pairs):
        ax = axes[config_index]
    
        # Plot individual seed curves with low opacity.
        for seed_index in range(num_seeds):
            ax.plot(
                environment_steps,
                returns_per_seed_np[
                config_index,
                seed_index,
                ],
                color="tab:blue",
                linewidth=0.8,
                alpha=0.20,
            )
    
        # Plot the mean across seeds.
        ax.plot(
        environment_steps,
        mean_returns_np[config_index],
        color="tab:blue",
        linewidth=2.0,
        label="Seed mean",
        )
        
        # Plot the 95% confidence interval.
        ax.fill_between(
        environment_steps,
        confidence_lower_np[config_index],
        confidence_upper_np[config_index],
        color="tab:blue",
        alpha=0.20,
        linewidth=0,
        label="95% CI",
        )
        
        ax.set_title(
        rf"$\tau_a={tau_a}$, $\tau_Q={tau_Q}$",
        fontsize=11,
        fontweight="bold",
        )
    
    ax.grid(
    True,
    linestyle="--",
    alpha=0.4,
    )
    
    # Add a legend only to the first panel.
    if config_index == 0:
        ax.legend(
        loc="best",
        fontsize=9,
        )
        
    # Hide unused subplots.
    for unused_index in range(num_configs, len(axes)):
        axes[unused_index].set_visible(False)
    
    # Shared axis labels.
    fig.supxlabel(
    "Environment step",
    fontsize=12,
    )
    
    fig.supylabel(
    "Episodic return",
    fontsize=12,
    )
    
    fig.suptitle(
    (
    f"Training EQL on {env_str}\n"
    f"Action ensemble = {action_ens_val}, "
    f"Q ensemble = {q_ens_val}, "
    f"seeds = {num_seeds}"
    ),
    fontsize=15,
    fontweight="bold",
    )
    
    fig.tight_layout(
    rect=[0.02, 0.02, 1.0, 0.93]
    )
    
    save_filename = (
    f"{env_str}_eql_config_comparison_"
    f"actEns{action_ens_val}_"
    f"qEns{q_ens_val}_"
    f"seeds{num_seeds}.png"
    )
    
    fig.savefig(
    save_filename,
    dpi=300,
    bbox_inches="tight",
    )
    
    print(f"Plot saved as: {save_filename}")
    
    plt.show()
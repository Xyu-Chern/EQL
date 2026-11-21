import jax
import jax.numpy as jnp

from eql.algorithms.eql import EQL


def main():
    import itertools
    import math
    import time

    from matplotlib import pyplot as plt

    env_str =  "CartPole-v1"  #

    TOTAL_TIMESTEPS = 100_000

    action_ens_val = 1
    q_ens_val = 10

    num_seeds = 4

    tau_a_values = [0.00, 0.25, 0.5, 0.75, 1.0]
    tau_Q_values = [0.00, 0.25, 0.5,0.75]
    config_pairs = list(itertools.product(tau_a_values, tau_Q_values))
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

    import numpy as np
    ep_lengths, ep_returns = evaluation

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

    mean_returns = returns_per_seed.mean(axis=1)

    seed_std = returns_per_seed.std(
        axis=1,
        ddof=1,
    )

    standard_error = seed_std / jnp.sqrt(num_seeds)

    confidence_radius = 1.96 * standard_error

    confidence_lower = mean_returns - confidence_radius
    confidence_upper = mean_returns + confidence_radius

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

        ax.plot(
        environment_steps,
        mean_returns_np[config_index],
        color="tab:blue",
        linewidth=2.0,
        label="Seed mean",
        )

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

    if config_index == 0:
        ax.legend(
        loc="best",
        fontsize=9,
        )

    for unused_index in range(num_configs, len(axes)):
        axes[unused_index].set_visible(False)

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


if __name__ == "__main__":
    main()

import jax
import jax.numpy as jnp

from eql.algorithms.sac import SAC


def main():
    import time

    from matplotlib import pyplot as plt

    env_str = "Reacher-misc"
    TOTAL_TIMESTEPS = 100_000

    tau_a_val = 0.7
    tau_Q_val = 0.3
    action_ens_val = 1
    q_ens_val = 10
    eql_alpha = 10.0

    algo = SAC.create(
        env=env_str,
        learning_rate=3e-4,
        total_timesteps=TOTAL_TIMESTEPS,
        num_critics=q_ens_val,          # Q ensemble 数量
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

    ep_lengths, ep_returns = evaluation
    mean_return = ep_returns.mean(axis=-1)

    plt.figure(figsize=(9, 5.5))
    plt.plot(jnp.linspace(0, TOTAL_TIMESTEPS, len(mean_return)), mean_return, linewidth=1.5, color="tab:blue")
    plt.xlabel("Environment step")
    plt.ylabel("Episodic return")

    plt.suptitle(f"Training SAC on {env_str}", fontsize=14, fontweight="bold")
    plt.title(
        f"$Q Ens={q_ens_val}",
        fontsize=11,
        color="darkslategrey"
    )

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    save_filename = f"${env_str}_num_q_{q_ens_val}_SAC.png"
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    print(f"图像已成功保存为: {save_filename}")
    plt.show()


if __name__ == "__main__":
    main()

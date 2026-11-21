"""Run the early Rejax CartPole baseline and save its learning curve."""
import argparse
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from rejax import get_algo


def run(method, env, steps, output):
    settings = {"target_entropy_ratio": 0.7, "num_envs": 10, "num_epochs": 5} if method == "sac" else {}
    algo = get_algo(method).create(env=env, learning_rate=0.0003, **settings)
    algo = algo.replace(gamma=0.995, total_timesteps=steps)
    _, (_, returns) = jax.jit(algo.train)(jax.random.PRNGKey(0))
    means = np.asarray(returns).mean(axis=-1)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.linspace(0, steps, len(means)), means)
    axis.set(xlabel="Environment step", ylabel="Episodic return", title=f"{env} · {method.upper()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("dqn", "sac"))
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.method, args.env, args.steps, args.output or Path(f"{args.env}_{args.method}_returns.png"))

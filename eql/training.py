"""Shared settings for the original four-task comparison."""
import jax

CONFIG = dict(num_envs=10, num_critics=10, total_timesteps=100000, learning_rate=3e-4, gamma=0.98)
COMMON = CONFIG | {"learning_rate": 1e-3}
SEEDS = [201, 202, 203, 204]
TASKS = ["Pendulum-v1", "Reacher-misc", "Hopper-v5", "Walker2d-v5"]


def ready(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()

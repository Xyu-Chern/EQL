"""Episode-level bag assignments for synchronously resetting Gymnax Pendulum."""
import jax
import jax.numpy as jnp
from rejax.algos.algorithm import register_init
from gymnax.environments.classic_control.pendulum import Pendulum

from bagging_buffer import BagReplayBuffer
from original_bag_eql import OriginalBagEQL


class PendulumEpisodeBags(BagReplayBuffer):
    episode_key: jax.Array
    new_episode: jax.Array

    @classmethod
    def from_empty(cls, buffer):
        return cls(buffer.data, buffer.index, buffer.count, buffer.num_critics,
                   buffer.capacity, jax.random.PRNGKey(0), jnp.array(True))

    def add(self, fragments, key, num_duplicates=2, replace=False):
        if fragments.done.shape[1] != 1:
            raise ValueError("Pendulum episode replay expects one step per environment.")
        # Key determines the full balanced environment-to-critic assignment.
        # The terminal transition uses the current episode's key. The next
        # transition gets the fresh key provided by the original training loop.
        assignment_key = jnp.where(self.new_episode, key, self.episode_key)
        updated, assignment = super().add(fragments, assignment_key, num_duplicates, replace)
        return updated.replace(episode_key=assignment_key,
                               new_episode=jnp.all(fragments.done)), assignment


class EpisodeBagEQL(OriginalBagEQL):
    @register_init
    def initialize_replay_buffer(self, rng):
        if not isinstance(self.env, Pendulum):
            raise ValueError("Synchronized episode replay is specific to Gymnax Pendulum.")
        values = super().initialize_replay_buffer(rng)
        values["bags"] = PendulumEpisodeBags.from_empty(values["bags"])
        return values

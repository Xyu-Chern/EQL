"""Balanced trajectory-fragment bags from the teacher's whiteboard.

Input leaves: [num_envs, num_steps, ...]. Each fragment is assigned whole
to d distinct critics. This is fragment-level, NOT full-episode bootstrap.
All bags receive the same number of fragments per add. Sampling is with
replacement (standard replay); critics use independent PRNG keys.
"""
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct


def balanced_assignment(key, num_envs, num_critics, num_duplicates):
    if not 1 <= num_duplicates <= num_critics:
        raise ValueError('Require 1 <= num_duplicates <= num_critics')
    if num_envs * num_duplicates % num_critics:
        raise ValueError('Whole fragments require num_envs*num_duplicates divisible by num_critics')
    env_key, critic_key = jax.random.split(key)
    env_order = jax.random.permutation(env_key, num_envs)
    critic_order = jax.random.permutation(critic_key, num_critics)
    slots = jnp.arange(num_envs * num_duplicates).reshape(num_envs, num_duplicates)
    assigned = critic_order[slots % num_critics]
    return jnp.zeros((num_envs, num_duplicates), jnp.int32).at[env_order].set(assigned)


class BagReplayBuffer(struct.PyTreeNode):
    data: object
    index: jax.Array
    count: jax.Array
    num_critics: int = struct.field(pytree_node=False)
    capacity: int = struct.field(pytree_node=False)

    @classmethod
    def empty(cls, example, num_critics, capacity):
        if num_critics < 1 or capacity < 1:
            raise ValueError('Positive num_critics and capacity required')
        data = jax.tree.map(lambda x: jnp.zeros((num_critics, capacity, *x.shape), x.dtype), example)
        return cls(data, jnp.array(0, jnp.int32), jnp.array(0, jnp.int32), num_critics, capacity)

    @partial(jax.jit, static_argnames=('num_duplicates', 'replace'))
    def add(self, fragments, key, num_duplicates=1, replace=False):
        leaf = jax.tree.leaves(fragments)[0]
        num_envs, num_steps = leaf.shape[:2]
        assignment = balanced_assignment(key, num_envs, self.num_critics, num_duplicates)
        env_ids = jnp.repeat(jnp.arange(num_envs), num_duplicates)
        order = jnp.argsort(assignment.reshape(-1), stable=True)
        grouped_env = env_ids[order].reshape(self.num_critics, -1)
        n_add = grouped_env.shape[1] * num_steps
        if n_add > self.capacity:
            raise ValueError('One add exceeds bag capacity')
        def group(x):
            return x[grouped_env].reshape(self.num_critics, n_add, *x.shape[2:])
        incoming = jax.tree.map(group, fragments)
        start = jnp.array(0, jnp.int32) if replace else self.index
        indices = (start + jnp.arange(n_add)) % self.capacity
        data = jax.tree.map(lambda dest, src: dest.at[:, indices].set(src), self.data, incoming)
        count = jnp.array(n_add, jnp.int32) if replace else jnp.minimum(self.capacity, self.count + n_add)
        return self.replace(data=data, index=(start+n_add) % self.capacity, count=count), assignment

    @partial(jax.jit, static_argnames=('batch_size',))
    def sample(self, key, batch_size):
        # Caller must check ready() before sampling; JIT cannot raise on dynamic count.
        keys = jax.random.split(key, self.num_critics)
        indices = jax.vmap(lambda k: jax.random.randint(k, (batch_size,), 0, jnp.maximum(self.count, 1)))(keys)
        return jax.tree.map(lambda x: x[jnp.arange(self.num_critics)[:, None], indices], self.data)

    def ready(self, min_entries=1):
        return self.count >= min_entries


class GlobalReplayBuffer(struct.PyTreeNode):
    data: object
    index: jax.Array
    count: jax.Array
    capacity: int = struct.field(pytree_node=False)

    @classmethod
    def empty(cls, example, capacity):
        return cls(jax.tree.map(lambda x: jnp.zeros((capacity, *x.shape), x.dtype), example),
                   jnp.array(0, jnp.int32), jnp.array(0, jnp.int32), capacity)

    @jax.jit
    def add(self, batch):
        n_add = jax.tree.leaves(batch)[0].shape[0]
        if n_add > self.capacity:
            raise ValueError('One add exceeds global capacity')
        indices = (self.index + jnp.arange(n_add)) % self.capacity
        data = jax.tree.map(lambda x, v: x.at[indices].set(v), self.data, batch)
        return self.replace(data=data, index=(self.index+n_add) % self.capacity,
                            count=jnp.minimum(self.capacity, self.count+n_add))

    @partial(jax.jit, static_argnames=('batch_size',))
    def sample(self, key, batch_size):
        ids = jax.random.randint(key, (batch_size,), 0, jnp.maximum(self.count, 1))
        return jax.tree.map(lambda x: x[ids], self.data)

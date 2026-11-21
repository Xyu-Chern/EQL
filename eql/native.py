"""Gymnasium vector environments, native algorithm variants and policy evaluation."""
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces
from gymnax.environments.environment import EnvParams
from jax.experimental import io_callback
from rejax.algos.algorithm import register_init

from eql.algorithms.adaptive import AdaptiveEQL
from eql.algorithms.eql import EQL
from eql.algorithms.original_bag import OriginalBagEQL
from eql.algorithms.sac import SAC
from eql.expectiles import compute_expectile_on_axis


class NativeTask:
    """Native Gymnasium training environments, stepped in order via JAX IO callbacks."""
    def __init__(self, name):
        self.name = name
        self.instances = [gym.make(name) for _ in range(10)]
        self.shape = self.instances[0].observation_space.shape
        self.action_shape = self.instances[0].action_space.shape
        self.default_params = EnvParams(max_steps_in_episode=1000)
        self.obs_spec = jax.ShapeDtypeStruct((10,*self.shape),np.float32)

    def __deepcopy__(self, memo):
        return self

    def observation_space(self, params):
        return spaces.Box(-np.inf, np.inf,self.shape)

    def action_space(self, params):
        return spaces.Box(-1.,1.,self.action_shape)

    def reset_batch(self, keys):
        def reset(keys):
            return np.stack([e.reset(seed=int(k[0])<<32 | int(k[1]))[0]
                             for e,k in zip(self.instances,np.asarray(keys))]).astype(np.float32)
        return io_callback(reset,self.obs_spec,keys,ordered=True), jnp.zeros(10,jnp.int32)

    def step_batch(self, keys, state, actions, params):
        def step(keys, actions):
            observations, rewards, dones = [], [], []
            for env, key, action in zip(self.instances,np.asarray(keys),np.asarray(actions)):
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if done:
                    obs,_=env.reset(seed=int(key[0])<<32 | int(key[1]))
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
            return np.asarray(observations,np.float32), np.asarray(rewards,np.float32), np.asarray(dones,bool)
        specs=(self.obs_spec,jax.ShapeDtypeStruct((10,),np.float32),jax.ShapeDtypeStruct((10,),bool))
        obs,rewards,dones=io_callback(step,specs,keys,actions,ordered=True)
        return obs,state,rewards,dones,{}


class NativeTraining:
    @property
    def vmap_reset(self):
        return lambda keys,params:self.env.reset_batch(keys)

    @property
    def vmap_step(self):
        return self.env.step_batch


class NativeSAC(NativeTraining,SAC): pass
class NativeEQL(NativeTraining,EQL): pass
class NativeBag(NativeTraining,OriginalBagEQL): pass


def native_evaluator(name, algo):
    pool=[gym.make(name) for _ in range(128)]
    act=jax.jit(lambda params,obs,key:algo.actor.apply(params,obs,key,method="act"))
    def evaluate(params, seed):
        rng=np.random.default_rng(30000+seed)
        obs=np.asarray([env.reset(seed=int(s))[0] for s,env in zip(rng.integers(0,2**32,128),pool)],np.float32)
        alive=np.ones(128,bool)
        scores=np.zeros(128)
        key=jax.random.PRNGKey(40000+seed)
        for _ in range(1000):
            key, subkey=jax.random.split(key)
            actions=np.asarray(act(params,obs,subkey))
            for i in np.flatnonzero(alive):
                obs[i],rew,term,trunc,_=pool[i].step(actions[i])
                scores[i]+=rew
                alive[i]=not(term or trunc)
            if not alive.any(): break
        return scores
    return evaluate


class NativeAdaptiveEQL(NativeTraining, AdaptiveEQL):
    def supports_episode_feedback(self):
        return self.env.name in ("Hopper-v3", "Walker2d-v3", "Hopper-v5", "Walker2d-v5")

    @register_init
    def init_episode_tracking(self, rng):
        return dict(age=jnp.zeros(self.num_envs, jnp.int32),
                    tracking=jnp.zeros(self.num_envs, bool),
                    completed=jnp.zeros(self.num_envs, bool))

    def collect_transitions(self, ts):
        _, action_key = jax.random.split(ts.rng)
        _, logp = self.actor.apply(ts.actor_ts.params, ts.last_obs, action_key,
                                   method="action_log_prob")
        collected, batch = EQL.collect_transitions(self, ts)
        # After a controller change, a trajectory already in flight must finish
        # and reset before this slot may contribute to the new arm.
        start = (ts.age == 0) & ~ts.completed & (ts.global_step >= self.fill_buffer)
        active = ts.tracking | start

        def predict():
            qs = self.vmap_critic(ts.critic_ts.params, batch.obs, batch.action)
            estimates = compute_expectile_on_axis(qs, ts.tau_q, axis=0, num_iters=self.num_iters)
            return jnp.where(start, estimates, ts.start_prediction)

        predicted = jax.lax.cond(start.any(), predict, lambda: ts.start_prediction)
        entropy = -jnp.exp(ts.alpha_ts.params["log_alpha"]) * logp
        increment = self.gamma ** ts.age * (batch.reward + jnp.where(start, 0., entropy))
        return collected.replace(
            start_prediction=predicted,
            soft_return=jnp.where(start, 0., ts.soft_return) + jnp.where(active, increment, 0.),
            raw_return=jnp.where(start, 0., ts.raw_return) + jnp.where(active, batch.reward, 0.),
            age=jnp.where(batch.done, 0, ts.age + 1),
            tracking=active & ~batch.done,
            completed=ts.completed | (active & batch.done),
        ), batch

    def train_iteration(self, ts):
        trained = EQL.train_iteration(self, ts)

        def finish(state):
            return self.finish_round(state).replace(completed=jnp.zeros_like(state.completed))

        return jax.lax.cond(trained.completed.all(), finish, lambda state: state, trained)

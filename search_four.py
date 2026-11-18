"""Four-task, paired search. Run one task/method per process on the WSL GPU."""
import argparse
import csv
import json
import os
import sys
import subprocess
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces
from gymnax.environments.environment import EnvParams
from jax.experimental import io_callback
from eql import EQL
from sac import SAC
from original_bag_eql import OriginalBagEQL
from run_original_comparison import CONFIG

ROOT = Path("results/search_four")
TASKS = ["Pendulum-v1", "Reacher-misc", "Hopper-v5", "Walker2d-v5"]
METHODS = {"SAC": SAC, "EQL": EQL, "EQL_bagging": OriginalBagEQL}
TUNE = [101, 102]
TEST = [201, 202, 203, 204]
EQL_GRID = [dict(tau_a=a,tau_Q=q) for a,q in [(0.7,0.3),(0.5,0.3),(0.7,0.1)]]
SAC_GRID = [dict(learning_rate=r) for r in [0.0003,0.0001,0.001]]


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


def save(path, obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,allow_nan=False,indent=2))
    tmp.replace(path)


def run(task,method):
    grid=SAC_GRID if method=="SAC" else EQL_GRID
    folder=ROOT/task/method
    folder.mkdir(parents=True,exist_ok=True)
    native=task.endswith("-v5")
    cls={"SAC":NativeSAC,"EQL":NativeEQL,"EQL_bagging":NativeBag}[method] if native else METHODS[method]
    env=NativeTask(task) if native else task
    base=cls.create(env=env,**CONFIG)
    if native: evaluate=native_evaluator(task,base)
    else:
        evaluate_state=jax.jit(lambda s,k:base.eval_callback(base,s,k)[1])
    # Rejax captures the learning rate in its optimizer at init time. Build each
    # candidate via create/initialization so SAC's learning-rate search is real.
    def series(candidate,seeds,phase):
        paths=[folder/f"{phase}_{candidate}_{s}.json" for s in seeds]
        if all(p.exists() for p in paths): return [json.loads(p.read_text())["score"] for p in paths]
        algo=cls.create(env=env,**(CONFIG|grid[candidate]))
        init=jax.jit(algo.init_state)
        block=jax.jit(lambda s:jax.lax.fori_loop(0,1000,lambda _,v:algo.train_iteration(v),s),donate_argnums=(0,))
        scores=[]
        for seed,path in zip(seeds,paths):
            if path.exists():
                scores.append(json.loads(path.read_text())["score"]);continue
            state=init(jax.random.PRNGKey(seed))
            for _ in range(10):
                state=block(state)
                for x in jax.tree.leaves(state):
                    if hasattr(x,"block_until_ready"): x.block_until_ready()
            assert int(state.global_step)==100000 and int(state.critic_ts.step)==9796
            returns=evaluate(state.actor_ts.params,seed) if native else np.asarray(evaluate_state(state,jax.random.PRNGKey(30000+seed)))
            assert len(returns)==128 and np.isfinite(returns).all()
            score=float(np.mean(returns))
            save(path,dict(task=task,method=method,phase=phase,seed=seed,candidate=candidate,settings=grid[candidate],
                           steps=100000,updates=9796,returns=returns.tolist(),score=score))
            print(task,method,phase,candidate,seed,round(score,2),flush=True)
            scores.append(score)
        return scores
    means=[np.mean(series(i,TUNE,"search")) for i in range(len(grid))]
    best=int(np.argmax(means))
    save(folder/"selection.json",dict(grid=grid,search_means=means,winner=best))
    series(best,TEST,"confirm")


def suite():
    ROOT.mkdir(parents=True,exist_ok=True)
    save(ROOT/"protocol.json",dict(tasks=TASKS,methods=list(METHODS),common=CONFIG,search_seeds=TUNE,
            confirmation_seeds=TEST,eql_grid=EQL_GRID,sac_grid=SAC_GRID,duplicates=2,
            criterion="per-task per-method highest mean final 128-episode return across two search seeds, first in grid breaks ties"))
    for task in TASKS:
        for method in METHODS:
            subprocess.run([sys.executable,"-u",__file__,task,method],check=True)


if __name__=="__main__":
    if len(sys.argv)==1: suite()
    else:
        assert jax.devices()[0].platform=="gpu"
        run(*sys.argv[1:])

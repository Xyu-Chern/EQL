"""Re-evaluate saved Rejax actors on the CPU with sequential Gymnasium episodes."""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import serialization
from rejax.networks import SquashedGaussianPolicy


def mean_action(module, obs):
    loc = module.action_mean(module.features(obs[None, :]))
    return (module.action_loc + module.action_scale * jnp.tanh(loc[0])).clip(-1, 1)


def make_stream(task, seed, episodes):
    env_seed, action_seed, _ = map(int, np.random.default_rng(seed).integers(0, 2**32 - 1, 3))
    env = gym.make(task)
    assert env.spec.id == task and env.unwrapped.sim.__class__.__module__.startswith("mujoco_py")
    env.reset(seed=env_seed)
    env.action_space.seed(action_seed)
    # In these two v3 tasks, only reset() consumes environment randomness.
    # Keep each nominal checkpoint's episode stream aligned across methods,
    # even when another method has more saved intermediate actors.
    states = {}
    rng = env.unwrapped.np_random
    for step in range(0, 1000001, 50000):
        states[step] = copy.deepcopy(rng.bit_generator.state)
        if step < 1000000:
            for _ in range(episodes):
                env.reset()
    policy = SquashedGaussianPolicy(env.action_space.shape[0],(-1.,1.),(64,64),nn.relu,(-10,2))
    act = jax.jit(lambda params, obs: policy.apply(params,obs,method=mean_action))
    return env, states, act, env_seed, action_seed


def evaluate(stream, params, step, episodes):
    env, states, act, env_seed, action_seed = stream
    env.unwrapped.np_random.bit_generator.state = copy.deepcopy(states[step])
    lengths, returns = [], []
    params = jax.device_put(params)
    for _ in range(episodes):
        obs, _ = env.reset()
        start_rng = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)
        length, total = 0, 0.
        while True:
            action = np.asarray(act(params,np.asarray(obs,np.float32)))
            obs, reward, terminated, truncated, _ = env.step(action)
            length += 1
            total += float(reward)
            if terminated or truncated:
                break
        assert env.unwrapped.np_random.bit_generator.state == start_rng
        lengths.append(length)
        returns.append(total)
    assert np.isfinite(returns).all()
    return dict(step=step, avg_ret=float(np.mean(returns)), std_ret=float(np.std(returns,ddof=0)),
                lengths=lengths,returns=returns,env_seed=env_seed,env_action_seed=action_seed)


def save_log(dest):
    points = sorted((dest/"points").glob("*.json"))
    path = dest/"log.csv"
    with path.with_suffix(".tmp").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["step","avg_ret","std_ret"],extrasaction="ignore")
        writer.writeheader()
        for point in points:
            writer.writerow(json.loads(point.read_bytes()))
    path.with_suffix(".tmp").replace(path)


def run(root,episodes):
    assert jax.default_backend()=="cpu"
    out=root/"deterministic_eval"
    streams,seen={},set()
    while True:
        sources = [(p,"final") for p in sorted((root/"results").glob("*/*/seed*/model.msgpack"))]
        sources += [(p,"actor") for p in sorted((out/"extra_actors").glob("*/*/seed*/*.msgpack"))]
        sources += [(p,"resume") for p in sorted((out/"checkpoint_links").glob("*/*/seed*/*.msgpack"))]
        for p,kind in sources:
            if p in seen:
                continue
            task,method,seedname=p.parts[-4:-1]
            seed=int(seedname.removeprefix("seed"))
            payload=serialization.msgpack_restore(p.read_bytes())
            if kind=="resume":
                state=payload["training"]
                step=int(state["global_step"])
                actor=state["actor_ts"]["params"]
                del state
            elif kind=="actor":
                step=int(p.stem)
                actor=payload
            else:
                step=1000000
                actor=payload["actor"]
            del payload
            dest=out/"scores"/task/method/seedname
            (dest/"points").mkdir(parents=True,exist_ok=True)
            point=dest/"points"/f"{step:07d}.json"
            digest=hashlib.sha256()
            for a in jax.tree.leaves(actor):
                digest.update(np.asarray(a).tobytes())
            sha=digest.hexdigest()
            if point.exists():
                assert json.loads(point.read_bytes())["actor_sha256"]==sha
            else:
                key=(task,seed)
                if key not in streams:
                    streams[key]=make_stream(task,seed,episodes)
                rec=evaluate(streams[key],actor,step,episodes)
                rec.update(task=task,method=method,seed=seed,episodes=episodes,
                           actor_sha256=sha,source=str(p.relative_to(root)))
                point.with_suffix(".tmp").write_text(json.dumps(rec,allow_nan=False))
                point.with_suffix(".tmp").replace(point)
                save_log(dest)
                print(task,method,seed,step,round(rec["avg_ret"],2),flush=True)
            seen.add(p)
        if len(list((out/"scores").glob("*/*/seed*/points/1000000.json")))==40:
            if set(p for p in (out/"checkpoint_links").glob("*/*/seed*/*.msgpack")) <= seen:
                break
        time.sleep(8)
    for env,*_ in streams.values():
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--episodes",type=int,default=128)
    args=parser.parse_args()
    run(args.root,args.episodes)

"""Online author-recipe CGPO with native evaluation and resumable v3 experiments."""
import fcntl
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from main import readParser
from agent.cgpo import CGPO
from agent.replay_memory import ReplayMemory, DiffusionMemory


PARTS=("actor","actor_target","critic","critic_target","critic_v",
       "actor_optimizer","critic_optimizer","critic_v_optimizer")
STATS=("step","running_q_std","running_q_mean","entropy_alpha")
ARRAYS=("states","actions","next_states","masks","rewards")


def save(path,obj,writer=torch.save):
    tmp=path.with_suffix(".tmp")
    if writer is json.dump:
        with tmp.open("w") as f:
            json.dump(obj,f,allow_nan=False)
    else:
        writer(obj,tmp)
    tmp.replace(path)


def env_state(env):
    sim=env.unwrapped.sim
    return dict(state=sim.get_state().flatten(),elapsed=env._elapsed_steps,
                rng=env.unwrapped.np_random.bit_generator.state,
                actions=env.action_space.np_random.bit_generator.state,
                inputs={k:getattr(sim.data,k).copy() for k in ("qacc_warmstart","ctrl","qfrc_applied","xfrc_applied")})


def restore_env(env,info):
    sim=env.unwrapped.sim
    sim.set_state_from_flattened(info["state"])
    for k,v in info["inputs"].items():
        getattr(sim.data,k)[:]=v
    sim.forward()
    sim.data.qacc_warmstart[:]=info["inputs"]["qacc_warmstart"]
    env._elapsed_steps=info["elapsed"]
    env.unwrapped.np_random.bit_generator.state=info["rng"]
    env.action_space.np_random.bit_generator.state=info["actions"]


def checkpoint(dest,agent,memory,env,ev,step,obs,best):
    n=memory.capacity if memory.full else memory.idx
    snapshot=dict(step=step,obs=obs,best=best,env=env_state(env),eval_rng=ev.unwrapped.np_random.bit_generator.state,
                  replay={k:getattr(memory,k)[:n] for k in ARRAYS},idx=memory.idx,full=memory.full,
                  parts={k:getattr(agent,k).state_dict() for k in PARTS},stats={k:getattr(agent,k) for k in STATS},
                  cpu_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state())
    save(dest/"resume.pt",snapshot)


def restore(dest,agent,memory,env,ev):
    snap=torch.load(dest/"resume.pt",map_location="cpu")
    for k in PARTS:
        getattr(agent,k).load_state_dict(snap["parts"][k])
    for k,v in snap["stats"].items():
        setattr(agent,k,v.to(agent.device) if torch.is_tensor(v) else v)
    for k,v in snap["replay"].items():
        getattr(memory,k)[:len(v)]=v
    memory.idx,memory.full=snap["idx"],snap["full"]
    restore_env(env,snap["env"])
    ev.unwrapped.np_random.bit_generator.state=snap["eval_rng"]
    torch.set_rng_state(snap["cpu_rng"])
    torch.cuda.set_rng_state(snap["gpu_rng"])
    np.random.set_state(snap["numpy_rng"])
    return snap["step"],snap["obs"],snap["best"]


def evaluate(env,agent):
    scores,lengths=[],[]
    for _ in range(10):
        obs,_=env.reset()
        score,length=0.,0
        while True:
            action=agent.sample_action(obs,eval=True)
            obs,reward,terminated,truncated,_=env.step(action)
            score+=reward
            length+=1
            if terminated or truncated:
                break
        scores.append(float(score))
        lengths.append(length)
    return dict(returns=scores,lengths=lengths,avg_ret=float(np.mean(np.asarray(scores,np.float32))),
                std_ret=float(np.std(np.asarray(scores,np.float32))))


def weights(agent):
    return dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict())


def train(args,dest):
    assert args.aug and args.weighted and args.running_v
    assert args.env_name in ("Hopper-v3","Walker2d-v3") and args.num_steps==1000000
    gpu=torch.cuda.get_device_name(0)
    assert "A100" in gpu
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env,ev=(gym.make(args.env_name) for _ in range(2))
    for e in (env,ev):
        e.unwrapped.np_random,_=gym.utils.seeding.np_random(args.seed)
        e.action_space.seed(args.seed)
        assert e.spec.id==args.env_name and e.unwrapped.sim.__class__.__module__.startswith("mujoco_py")
    dims=(env.observation_space.shape[0],env.action_space.shape[0])
    device=torch.device("cuda:0")
    memory=ReplayMemory(*dims,1000000,device)
    agent=CGPO(args,dims[0],env.action_space,memory,DiffusionMemory(*dims,1000000,device),device)
    for name in ("native","policies"):
        (dest/name).mkdir(exist_ok=True)
    protocol=dict(source_commit="6958aca6e7faccb44f815b3b810192b8d21edd5e",args=vars(args),torch=torch.__version__,
                  gpu=gpu,gymnasium=gym.__version__,native_episodes=10,native_every=10000,eql_episodes=128,eql_every=50000)
    if (dest/"config.json").exists():
        assert json.loads((dest/"config.json").read_text())==protocol
    else:
        save(dest/"config.json",protocol,json.dump)
    obs,_=env.reset()
    first,best=0,-float("inf")
    if (dest/"resume.pt").exists():
        first,obs,best=restore(dest,agent,memory,env,ev)
    else:
        save(dest/"policies"/"0000000.pt",weights(agent))
    print(args.env_name,args.seed,gpu,"starting at",first,flush=True)
    for step in range(first+1,args.num_steps+1):
        action=env.action_space.sample() if step<=10000 else agent.sample_action(obs,eval=False)
        nxt,reward,terminated,truncated,_=env.step(action)
        done=terminated or truncated
        agent.append_memory(obs,action,reward,nxt,0. if done else args.gamma)
        if step>=10000:
            agent.train(step,1,batch_size=args.batch_size)
            agent.entropy_alpha=min(args.entropy_alpha,max(.002,args.entropy_alpha*(1-step/args.num_steps)))
        if step%10000==0:
            score=evaluate(ev,agent)
            save(dest/"native"/f"{step:07d}.json",dict(step=step,**score),json.dump)
            if score["avg_ret"]>best:
                best=score["avg_ret"]
                save(dest/"best.pt",weights(agent))
        if step%50000==0:
            save(dest/"policies"/f"{step:07d}.pt",weights(agent))
        obs=env.reset()[0] if done else nxt
        if step%5000==0:
            checkpoint(dest,agent,memory,env,ev,step,obs,best)
        if step%1000==0 or step in (1,100,10001,10010,10100):
            save(dest/"progress.json",dict(step=step,updates=agent.step,time=time.time()),json.dump)
            print(args.env_name,args.seed,step,agent.step,flush=True)
    assert agent.step==990001
    save(dest/"final.json",dict(steps=step,updates=agent.step,best_native_return=best),json.dump)
    env.close()
    ev.close()


if __name__=="__main__":
    args=readParser()
    dest=Path(os.environ["CGPO_RESULTS"])/args.env_name/f"seed{args.seed}"
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/"run.lock").open("a") as lock:
        fcntl.lockf(lock,fcntl.LOCK_EX)
        if not (dest/"final.json").exists():
            train(args,dest)

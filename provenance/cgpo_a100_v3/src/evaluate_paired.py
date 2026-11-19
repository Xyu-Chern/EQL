"""CPU-only CGPO evaluation on the same 128-episode streams as the EQL archive."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"

import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from agent.cgpo import CGPO


def evaluate_policy(source,dest,cfg):
    task=cfg.env_name
    seed=cfg.seed
    step=int(source.stem)
    env_seed,action_seed,policy_seed=map(int,np.random.default_rng(seed).integers(0,2**32-1,3))
    env=gym.make(task)
    env.reset(seed=env_seed)
    env.action_space.seed(action_seed)
    for _ in range((step//50000)*128):
        env.reset()
    agent=CGPO(cfg,env.observation_space.shape[0],env.action_space,None,None,torch.device("cpu"))
    weights=torch.load(source,map_location="cpu")
    agent.actor.load_state_dict(weights["actor"])
    agent.critic.load_state_dict(weights["critic"])
    agent.actor.eval()
    agent.critic.eval()
    noise_source=torch.Generator(device="cpu").manual_seed(policy_seed).get_state()

    @torch.inference_mode()
    def act(obs):
        # Reuse the same 32 complete DDPM noise paths at each observation.
        torch.set_rng_state(noise_source)
        return agent.sample_action(obs,eval=True).clip(-1,1)

    returns,lengths=[],[]
    for episode in range(128):
        obs,_=env.reset()
        if episode==0:
            assert np.array_equal(act(obs),act(obs))
        score,length=0.,0
        while True:
            obs,r,term,trunc,_=env.step(act(obs))
            score+=r
            length+=1
            if term or trunc:break
        returns.append(float(score))
        lengths.append(length)
    env.close()
    assert np.isfinite(returns).all()
    rec=dict(task=task,seed=seed,step=step,returns=returns,lengths=lengths,
             avg_ret=float(np.mean(returns)),std_ret=float(np.std(returns)),policy_seed=policy_seed,
             env_seed=env_seed,env_action_seed=action_seed,eval_sample=cfg.eval_sample,
             action_rule="fixed DDPM noise bank, author's critic argmax")
    dest.with_suffix(".tmp").write_text(json.dumps(rec,allow_nan=False))
    dest.with_suffix(".tmp").replace(dest)
    folder=dest.parent
    with (folder/"log.tmp").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["step","avg_ret","std_ret"],extrasaction="ignore")
        writer.writeheader()
        for p in sorted(folder.glob("*.json")):
            writer.writerow(json.loads(p.read_bytes()))
    (folder/"log.tmp").replace(folder/"log.csv")
    print(task,seed,step,round(rec["avg_ret"],2),flush=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    root=Path(os.environ["CGPO_RESULTS"])
    while True:
        for p in sorted(root.glob("*/seed*/policies/*.pt")):
            run=p.parents[1]
            out=run/"eql_eval"
            out.mkdir(exist_ok=True)
            dest=out/(p.stem+".json")
            if not dest.exists():
                cfg=SimpleNamespace(**json.loads((run/"config.json").read_bytes())["args"])
                evaluate_policy(p,dest,cfg)
        if len(list(root.glob("*/seed*/eql_eval/1000000.json")))==8:break
        time.sleep(12)

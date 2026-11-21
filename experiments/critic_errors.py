"""Fixed-budget Pendulum training and simulation-based critic error diagnostics."""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from eql.records import source_digest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import msgpack_restore, to_bytes
from gymnax.environments.classic_control.pendulum import EnvState

from eql.algorithms.episode_bag import EpisodeBagEQL
from eql.algorithms.eql import EQL
from eql.algorithms.original_bag import OriginalBagEQL
from eql.diagnostics import COUNTS, MC_REPS, ROOT, evaluate_errors, make_panel
from eql.records import read, save
from eql.training import COMMON, SEEDS, ready
from experiments._pendulum import grid_file

CLASSES = {"shared":EQL,"step":OriginalBagEQL,"episode":EpisodeBagEQL}
SOURCES = ["experiments/critic_errors.py", "eql/diagnostics.py", "experiments/_pendulum.py","eql/algorithms/eql.py", "eql/expectiles.py","eql/replay.py","eql/algorithms/original_bag.py","eql/algorithms/episode_bag.py","eql/training.py"]
MC_BLOCK = 16


def simulator(algo, panel):
    theta,velocity,action=map(jnp.asarray,panel)
    def rollout(actor,alpha,key):
        n=len(theta)*MC_BLOCK
        state=EnvState(theta=jnp.repeat(theta,MC_BLOCK),theta_dot=jnp.repeat(velocity,MC_BLOCK),
                       last_u=jnp.zeros(n),time=jnp.zeros(n,jnp.int32))
        step=jax.vmap(lambda s,a:algo.env.step_env(None,s,a,algo.env_params)[:3])
        # Q(s,a) conditions on the specified first action. Entropy starts at t=1.
        obs,state,total=step(state,jnp.repeat(action,MC_BLOCK)[:,None])
        def advance(t, carry):
            obs,state,total,power=carry
            act,logp=algo.actor.apply(actor,obs,jax.random.fold_in(key,t),method="action_log_prob")
            obs,state,reward=step(state,act)
            return obs,state,total+power*(reward-alpha*logp),power*algo.gamma
        _,_,total,_=jax.lax.fori_loop(1,200,advance,(obs,state,total,jnp.float32(algo.gamma)))
        return total.reshape((len(theta),MC_BLOCK))
    return jax.jit(rollout)


def archive_npz(path, **arrays):
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as stream: np.savez_compressed(stream,**arrays)
    tmp.replace(path)


def run(method,count):
    cfg=COMMON|dict(num_critics=count,tau_Q=0.1,tau_a=0.7)
    if method!="shared": cfg.update(num_duplicates=count//5)
    algo=CLASSES[method].create(env="Pendulum-v1",**cfg)
    assert jax.devices()[0].platform=="gpu"
    init=jax.jit(algo.init_state)
    train=jax.jit(lambda s:jax.lax.fori_loop(0,1000,lambda _,v:algo.train_iteration(v),s),donate_argnums=(0,))
    evaluate=jax.jit(lambda state,key:algo.eval_callback(algo,state,key)[1])
    panel=make_panel()
    rollout=simulator(algo,panel)
    theta,velocity,act=map(jnp.asarray,panel)
    obs=jnp.stack((jnp.cos(theta),jnp.sin(theta),velocity),axis=-1)
    predict=jax.jit(lambda p:algo.vmap_critic(p,obs,act[:,None]))
    for seed in SEEDS:
        folder=ROOT/str(count)/method/f"seed{seed}"
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/"metrics.json").exists(): continue
        model=folder/"model.msgpack"
        if not (folder/"train.json").exists():
            state=init(jax.random.PRNGKey(seed))
            for _ in range(10):
                state=train(state)
                ready(state)
            assert int(state.global_step)==100000 and int(state.actor_ts.step)==int(state.critic_ts.step)==9796
            if method!="shared": assert int(state.bags.count)==20000
            returns=np.asarray(evaluate(state,jax.random.PRNGKey(30000+seed)))
            assert returns.shape==(128,) and np.isfinite(returns).all()
            if count==10:
                old=(Path("results/pendulum_episode_bags")/f"seed{seed}.json" if method=="episode" else
                     grid_file("EQL" if method=="shared" else "EQL_bagging",0.1,0.7,seed))
                np.testing.assert_array_equal(returns,read(old)["returns"])
            params=dict(actor=state.actor_ts.params,critics=state.critic_ts.params,
                        target_critics=state.critic_target_params,alpha=state.alpha_ts.params)
            tmp=model.with_suffix(".tmp")
            tmp.write_bytes(to_bytes(params));tmp.replace(model)
            save(folder/"train.json",dict(config=cfg,seed=seed,updates=9796,steps=100000,
                          score=float(returns.mean()),returns=returns.tolist(),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest()))
            del state,params
        checkpoint=msgpack_restore(model.read_bytes())
        alpha=float(np.exp(checkpoint["alpha"]["log_alpha"]))
        q=np.asarray(predict(checkpoint["critics"]))
        key=jax.random.PRNGKey(89000+seed)
        mc=np.concatenate([np.asarray(rollout(checkpoint["actor"],alpha,jax.random.fold_in(key,i)))
                           for i in range(MC_REPS//MC_BLOCK)],axis=1)
        assert q.shape==(count,len(theta)) and mc.shape==(len(theta),MC_REPS)
        metrics,arrays=evaluate_errors(q,mc)
        archive_npz(folder/"diagnostics.npz",theta=panel[0],velocity=panel[1],action=panel[2],
                    obs=np.asarray(obs),critic_q=q,mc_soft_returns=mc,**arrays)
        metrics.update(method=method,critics=count,seed=seed,alpha=alpha,score=read(folder/"train.json")["score"])
        save(folder/"metrics.json",metrics)
        print(count,method,seed,"rho",round(metrics["correlation"],5),"return",round(metrics["score"],2),flush=True)


def suite():
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=dict(common=COMMON,tau_Q=0.1,tau_a=0.7,ensemble=COUNTS,seeds=SEEDS,
                  bag_copies=[2,4,10],coverage=0.2,mc_reps=MC_REPS,mc_block=MC_BLOCK,
                  probe="16 angles x 7 velocities x 3 actions, t=0; 200-step soft Q including entropy from next action onward",
                  source_sha256={f:source_digest(f) for f in SOURCES})
    if (ROOT/"protocol.json").exists(): assert read(ROOT/"protocol.json")==protocol
    save(ROOT/"protocol.json",protocol)
    for count in COUNTS:
        for method in CLASSES:
            subprocess.run([sys.executable,"-u","-m", "experiments.critic_errors",method,str(count)],check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", nargs="?", choices=CLASSES)
    parser.add_argument("count", nargs="?", type=int, choices=COUNTS)
    args = parser.parse_args()
    values = (args.method, args.count)
    if all(value is None for value in values):
        suite()
    elif any(value is None for value in values):
        parser.error("Provide all positional arguments to run one configuration.")
    else:
        run(args.method, args.count)

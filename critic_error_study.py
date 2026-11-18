"""Fixed-budget Pendulum training and simulation-based critic error diagnostics."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import to_bytes, msgpack_restore
from gymnax.environments.classic_control.pendulum import EnvState
from pendulum_tau_grid import COMMON, SEEDS, file as grid_file, read, save
from pendulum_episode_bags import EpisodeBagEQL
from original_bag_eql import OriginalBagEQL
from eql import EQL
from run_original_comparison import ready

ROOT = Path("results/pendulum_critic_errors")
CLASSES = {"shared":EQL,"step":OriginalBagEQL,"episode":EpisodeBagEQL}
COUNTS = [10,20,50]
SOURCES = ["critic_error_study.py","eql.py","bagging_buffer.py","original_bag_eql.py","pendulum_episode_bags.py","run_original_comparison.py"]
MC_BLOCK = 16
MC_REPS = 256


def make_panel():
    # Evaluate the same observation at episode start (full 200-step horizon).
    theta, velocity, action = np.meshgrid(np.arange(-8,8)*np.pi/8,
                              [-7.,-4.,-2.,0.,2.,4.,7.],[-1.5,0.,1.5],indexing="ij")
    return tuple(x.ravel().astype(np.float32) for x in (theta,velocity,action))


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


def evaluate_errors(q, mc):
    # Multiple critics share a noisy MC label, adding its error variance to
    # every raw covariance. Cross-covariance between independent MC halves
    # removes that common noise in expectation. Symmetrize before scaling.
    left,right=np.split(mc.astype(np.float64),2,axis=1)
    qa=q.astype(np.float64)
    err_a=qa-left.mean(1)
    err_b=qa-right.mean(1)
    errors=(err_a+err_b)/2
    a=err_a-err_a.mean(1,keepdims=True)
    b=err_b-err_b.mean(1,keepdims=True)
    cov=(a@b.T+b@a.T)/(2*(a.shape[1]-1))
    assert np.all(np.diag(cov)>0)
    std=np.sqrt(np.diag(cov))
    corrected=cov/(std[:,None]*std[None,:])
    raw=np.corrcoef(errors)
    pair=np.triu_indices(len(q),1)
    positive=(errors>0).sum(0)
    k=len(q)
    same=(positive*(positive-1)+(k-positive)*(k-positive-1))/(k*(k-1))
    metrics=dict(correlation=float(corrected[pair].mean()),raw_correlation=float(raw[pair].mean()),
                 same_sign_fraction=float(same.mean()),mean_bias=float(errors.mean()),
                 critic_rmse=float(np.sqrt(np.mean(errors**2))),
                 ensemble_mean_rmse=float(np.sqrt(np.mean(errors.mean(0)**2))),
                 prediction_spread=float(qa.std(0).mean()),
                 median_mc_se=float(np.median(mc.std(1,ddof=1)/np.sqrt(mc.shape[1]))))
    assert np.isfinite(list(metrics.values())).all()
    return metrics, dict(mc_reference=mc.mean(1),errors=errors,correlation_corrected=corrected,
                         correlation_raw=raw,cross_covariance=cov)


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
                  source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in SOURCES})
    if (ROOT/"protocol.json").exists(): assert read(ROOT/"protocol.json")==protocol
    save(ROOT/"protocol.json",protocol)
    for count in COUNTS:
        for method in CLASSES:
            subprocess.run([sys.executable,"-u",__file__,method,str(count)],check=True)


if __name__=="__main__":
    if len(sys.argv)==1:suite()
    else:run(sys.argv[1],int(sys.argv[2]))

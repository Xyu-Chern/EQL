"""Original-config three-way comparison; original source files are not edited."""
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np
from flax.serialization import to_bytes
from rejax.evaluate import evaluate

from eql.algorithms.eql import EQL
from eql.algorithms.original_bag import OriginalBagEQL
from eql.algorithms.sac import SAC
from eql.records import source_digest
from eql.training import CONFIG, ready


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--env',required=True)
    p.add_argument('--method',required=True,choices=['SAC','EQL','EQL_bagging'])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--output',default='results/original_comparison')
    a=p.parse_args()
    assert any(d.platform=='gpu' for d in jax.devices())
    assert a.steps%1000==0
    cls={'SAC':SAC,'EQL':EQL,'EQL_bagging':OriginalBagEQL}[a.method]
    config=dict(CONFIG,total_timesteps=a.steps)
    if a.method!='SAC': config.update(tau_a=0.7,tau_Q=0.3,num_iters=10)
    algo=cls.create(env=a.env,**config)
    out=Path(a.output)/a.env
    out.mkdir(parents=True,exist_ok=True)
    resolved={f.name:getattr(algo,f.name) for f in dataclasses.fields(algo)
              if isinstance(getattr(algo,f.name),(int,float,bool,str))}
    sources={f:source_digest(f) for f in
             ["eql/algorithms/eql.py", "eql/expectiles.py","eql/algorithms/sac.py","eql/algorithms/original_bag.py","eql/replay.py","experiments/baseline.py", "eql/training.py"]}
    manifest=dict(method=a.method,env=a.env,config=resolved,source_sha256=sources,
        training_keys='jax.random.split(jax.random.PRNGKey(0),4)',
        evaluation='128 stochastic episodes, independent common evaluation keys',
        bagging='num_duplicates=2; per-bag capacity=original buffer_size; per-critic batch=256; one-step fragments',
        original_semantics='original IRLS, policy warmup, done masking, Polyak, alpha and update order retained')
    (out/f'{a.method}_manifest.json').write_text(json.dumps(manifest,indent=2))
    init=jax.jit(algo.init_state)
    def block(s):
        return jax.lax.fori_loop(0,100,lambda _,v:algo.train_iteration(v),s)
    block=jax.jit(block,donate_argnums=(0,))
    ev=jax.jit(lambda s,k:evaluate(algo.make_act(s),k,algo.env,algo.env_params,128,algo.env_params.max_steps_in_episode))
    keys=jax.random.split(jax.random.PRNGKey(0),4)
    for seed in a.seeds:
        stem=out/f'{a.method}_seed{seed}'
        if stem.with_suffix('.json').exists(): raise RuntimeError(f'Refusing overwrite: {stem}')
        start=time.perf_counter()
        s=init(keys[seed]); ready(s)
        initial_hash=hashlib.sha256(to_bytes(dict(actor=s.actor_ts.params,critics=s.critic_ts.params))).hexdigest()
        records=[]; raw=[]; train_time=0.
        for k in range(a.steps//1000+1):
            if k:
                t=time.perf_counter(); s=block(s); ready(s); train_time+=time.perf_counter()-t
            # Evaluate every 5000 transitions, including 0 and exact final step.
            if k%5 and k*1000!=a.steps: continue
            _,returns=ev(s,jax.random.fold_in(jax.random.PRNGKey(10000+seed),k))
            returns=np.asarray(returns)
            params=jax.tree.leaves((s.actor_ts.params,s.critic_ts.params,s.alpha_ts.params))
            assert np.isfinite(returns).all() and all(np.isfinite(np.asarray(x)).all() for x in params)
            step=int(s.global_step)
            updates=int(s.actor_ts.step)
            assert step==k*1000
            assert updates==max(0,step//10-204)
            if a.method=='EQL_bagging': assert int(s.bags.count)==step//5
            rec=dict(step=step,updates=updates,return_mean=float(returns.mean()),alpha=float(np.exp(np.asarray(s.alpha_ts.params['log_alpha']))))
            records.append(rec);raw.append(returns)
            print(json.dumps(dict(method=a.method,env=a.env,seed=seed,**rec)),flush=True)
            with stem.with_suffix('.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        stem.with_suffix('.msgpack').write_bytes(to_bytes(dict(actor=s.actor_ts.params,critics=s.critic_ts.params,
            target_critics=s.critic_target_params,alpha=s.alpha_ts.params)))
        np.savez_compressed(stem.with_suffix('.npz'),returns=np.stack(raw),steps=[r['step'] for r in records])
        info=dict(method=a.method,env=a.env,seed=seed,training_key=np.asarray(keys[seed]).tolist(),initial_params_sha256=initial_hash,
            records=records,final_return=records[-1]['return_mean'],wall_seconds=time.perf_counter()-start,training_seconds_including_compile=train_time)
        stem.with_suffix('.json').write_text(json.dumps(info,indent=2))
        print('FINISHED '+json.dumps({k:v for k,v in info.items() if k!='records'}),flush=True)

if __name__=='__main__':main()

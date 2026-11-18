"""Sequential small online experiments, with raw seed data and checkpoints."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_default_matmul_precision', 'highest')
from flax.serialization import to_bytes
from rejax.evaluate import evaluate
from online_bag_eql import BagEQL


def synchronize(state):
    # Avoid batched GetReadyFuture over aliased state leaves on this older
    # JAX/WSL stack. Per-array synchronization has identical numerical semantics.
    for leaf in jax.tree.leaves(state):
        if hasattr(leaf, 'block_until_ready'):
            leaf.block_until_ready()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--env',default='Pendulum-v1',choices=['Pendulum-v1','Reacher-misc'])
    parser.add_argument('--steps',type=int,default=65536)
    parser.add_argument('--eval-every',type=int,default=8192)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--groups',nargs='+',default=['A','B','C','D'])
    parser.add_argument('--output',default='results/online_v1')
    args=parser.parse_args()
    assert any(d.platform=='gpu' for d in jax.devices()), 'GPU required for reported GPU experiment'
    assert args.steps%args.eval_every==0 and args.eval_every%32==0
    out=Path(args.output)/args.env
    out.mkdir(parents=True,exist_ok=True)
    modes={'A':('shared','shared'),'B':('bag','shared'),'C':('shared','own'),'D':('bag','own')}
    config=dict(num_envs=4,num_critics=4,num_steps=8,num_duplicates=1,buffer_size=32768,
        batch_size=128,fill_buffer=1024,num_epochs=8,hidden_layer_sizes=[64,64],
        learning_rate=3e-4,gamma=0.99,polyak=0.995,tau_actor=0.3,tau_target=0.3)
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
             ['run_online.py','online_bag_eql.py','bagging_buffer.py','expectile.py']}
    meta=dict(config=config,arguments=vars(args),jax=jax.__version__,devices=[str(d) for d in jax.devices()],
              source_sha256=sources,evaluation='64 stochastic-policy episodes, separate deterministic eval keys',
              unit='environment transitions across all 4 environments',fragment_semantics='fixed 8-step rollout per environment; not complete episodes',
              groups=modes)
    (out/('manifest_'+''.join(args.groups)+'.json')).write_text(json.dumps(meta,indent=2))
    for group in args.groups:
        data,target=modes[group]
        algo=BagEQL.create(env=args.env,**config,data_mode=data,target_mode=target)
        init=jax.jit(algo.init_state)
        def block(ts):
            return jax.lax.fori_loop(0,args.eval_every//32,lambda _,s:algo.train_iteration(s),ts)
        block=jax.jit(block,donate_argnums=(0,))
        def evaluation(ts,key):
            return evaluate(algo.make_act(ts),key,algo.env,algo.env_params,64,algo.env_params.max_steps_in_episode)
        evaluation=jax.jit(evaluation)
        for seed in args.seeds:
            stem=out/f'{group}_seed{seed}'
            if stem.with_suffix('.json').exists():
                print(f'SKIP {stem}',flush=True)
                continue
            started=time.perf_counter()
            state=init(jax.random.PRNGKey(seed))
            synchronize(state)
            compiled_at=time.perf_counter()
            compiled=block.lower(state).compile()
            compile_seconds=time.perf_counter()-compiled_at
            records=[]
            eval_returns=[]
            training_seconds=0.
            for k in range(args.steps//args.eval_every+1):
                if k:
                    t=time.perf_counter()
                    state=compiled(state)
                    synchronize(state)
                    training_seconds+=time.perf_counter()-t
                key=jax.random.fold_in(jax.random.PRNGKey(10000+seed),k)
                lengths,returns=evaluation(state,key)
                returns=np.asarray(returns)
                diag=np.asarray(state.diagnostics)
                if not np.isfinite(returns).all() or not np.isfinite(diag).all():
                    raise RuntimeError(f'Nonfinite values: {group}, {seed}, {k}')
                rec=dict(step=int(state.global_step),return_mean=float(returns.mean()),
                    return_std_episodes=float(returns.std()),updates=int(state.updates),
                    critic_loss=float(diag[0]),mean_q=float(diag[1]),disagreement=float(diag[2]),
                    alpha=float(diag[3]),training_seconds=training_seconds)
                records.append(rec)
                eval_returns.append(returns)
                print(json.dumps(dict(env=args.env,group=group,seed=seed,**rec)),flush=True)
                with stem.with_suffix('.jsonl').open('a') as f:
                    f.write(json.dumps(rec)+'\n')
            checkpoint={'actor':state.actor_ts.params,'critics':state.critic_ts.params,
                        'target_critics':state.critic_target_params,'alpha':state.alpha_ts.params}
            stem.with_suffix('.msgpack').write_bytes(to_bytes(checkpoint))
            np.savez_compressed(stem.with_suffix('.npz'),returns=np.stack(eval_returns),
                                steps=np.array([r['step'] for r in records]))
            info=dict(env=args.env,group=group,seed=seed,compile_seconds=compile_seconds,
                wall_seconds=time.perf_counter()-started,training_seconds=training_seconds,
                final_return=records[-1]['return_mean'],records=records)
            stem.with_suffix('.json').write_text(json.dumps(info,indent=2))
            print('FINISHED '+json.dumps({k:v for k,v in info.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from eql import EQL, compute_expectile_on_axis
from pendulum_tau_grid import COMMON, SEEDS
from itertools import combinations
device = jax.devices()[0]
assert device.platform == "gpu" and "A40" in device.device_kind
model = EQL.create(env="Pendulum-v1", **COMMON)
state = model.init_state(jax.random.PRNGKey(201))
params = np.asarray(next(v for v in jax.tree.leaves(state.critic_ts.params) if v.ndim == 3))
unique = len({p.tobytes() for p in params})
assert unique == model.num_critics == 10
rows=[]
for n in (10,20,50):
    for seed in SEEDS:
        folder = Path(f"results/pendulum_critic_errors/{n}/shared/seed{seed}")
        with np.load(folder/"diagnostics.npz") as d:
            q = jnp.asarray(d["critic_q"])
            target = d["mc_soft_returns"].mean(axis=1)
            coords = np.stack([d["theta"],d["velocity"]],axis=1)
            actions = np.asarray(d["action"])
        assert q.shape == (n,336)
        predictors={"min":q.min(0),"mean":q.mean(0),"q01":compute_expectile_on_axis(q,.1,0),"a07":compute_expectile_on_axis(q,.7,0)}
        estimates = {k:np.asarray(v) for k,v in predictors.items()}
        grouped = [np.flatnonzero((coords==s).all(axis=1)) for s in np.unique(coords,axis=0)]
        assert len(grouped)==112 and all(len(g)==3 and len(np.unique(actions[g]))==3 for g in grouped)
        agree = np.mean([g[np.argmax(estimates["min"][g])]==g[np.argmax(estimates["a07"][g])] for g in grouped])
        row={"n":n,"seed":seed,"rank_agreement":float(agree)}
        for k,v in estimates.items():
            row[k+"_bias"]=float(np.mean(v-target))
            row[k+"_rmse"]=float(np.sqrt(np.mean((v-target)**2)))
        rows.append(row)
print(json.dumps({"device":device.device_kind,"unique_init":unique,"inputs":"shared-replay fixed tau_Q=.1 tau_a=.7; 336 queries, 256 frozen-policy MC rollouts/query","rows":rows,"by_n":{n:{k:round(float(np.mean([a[k] for a in rows if a["n"]==n])),4) for k in rows[0] if k not in ["seed","n"]} for n in (10,20,50)}}))

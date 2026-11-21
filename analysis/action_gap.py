import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import msgpack_restore

from eql.algorithms.eql import EQL
from eql.training import COMMON, SEEDS


def main():
    rows = []
    device = jax.devices()[0]
    assert device.platform == "gpu" and "A40" in device.device_kind
    for n in [10, 20, 50]:
        algo = EQL.create(env="Pendulum-v1", **(COMMON | {"num_critics": n}))
        space = algo.env.action_space(algo.env_params)
        @jax.jit
        def probe(actor, critics, obs, key):
            obs = jnp.repeat(obs, 32, axis=0)
            policy_key, uniform_key = jax.random.split(key)
            policy = algo.actor.apply(actor, obs, policy_key, method="act")
            uniform = jax.random.uniform(uniform_key, policy.shape, minval=space.low, maxval=space.high)
            def penalty(actions):
                q = algo.vmap_critic(critics, obs, actions)
                return (q.mean(axis=0) - q.min(axis=0)).mean()
            return penalty(policy), penalty(uniform)
        for seed in SEEDS:
            folder = Path(f"results/pendulum_critic_errors/{n}/shared/seed{seed}")
            p = msgpack_restore((folder/"model.msgpack").read_bytes())
            with np.load(folder/"diagnostics.npz") as d:
                obs = np.unique(d["obs"], axis=0)
            assert obs.shape == (112,3)
            policy, uniform = map(float, probe(p["actor"], p["critics"], obs, jax.random.PRNGKey(73000+seed)))
            rows.append(dict(n=n,seed=seed,policy=policy,uniform=uniform,gap=uniform-policy))
    summary = {n:{k:round(float(np.mean([r[k] for r in rows if r["n"]==n])),4) for k in ("policy","uniform","gap")} for n in [10,20,50]}
    print(json.dumps(dict(device=device.device_kind,states=112,actions_per_state_per_distribution=32,rows=rows,by_n=summary)))


if __name__ == "__main__":
    main()

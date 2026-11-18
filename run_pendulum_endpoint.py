"""Run the EQL zero endpoint on the selected SAC settings and compare saved episode returns."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import numpy as np
from eql import EQL
from run_original_comparison import CONFIG, ready

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-clipped", action="store_true")
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    tau = 0.0001 if args.legacy_clipped else 0.0
    mode = "clipped" if args.legacy_clipped else "exact"
    reference = Path("results/search_four/Pendulum-v1/SAC")
    selection = json.loads((reference / "selection.json").read_text())
    winner = selection["winner"]
    config = CONFIG | selection["grid"][winner] | dict(tau_a=tau, tau_Q=tau)
    algo = EQL.create(env="Pendulum-v1", **config)
    init = jax.jit(algo.init_state)
    def block(state):
        return jax.lax.fori_loop(0, 1000, lambda _, s: algo.train_iteration(s), state)
    train = jax.jit(block, donate_argnums=(0,))
    evaluate = jax.jit(lambda s,k: algo.eval_callback(algo,s,k)[1])
    folder = Path("results/pendulum_sac_endpoint") / mode
    folder.mkdir(parents=True,exist_ok=True)
    for seed in [201,202,203,204]:
        sac = json.loads((reference / f"confirm_{winner}_{seed}.json").read_text())
        state = init(jax.random.PRNGKey(seed))
        for _ in range(10):
            state = train(state)
            ready(state)
        assert int(state.global_step)==sac["steps"] and int(state.critic_ts.step)==sac["updates"]
        returns = np.asarray(evaluate(state,jax.random.PRNGKey(30000+seed)))
        baseline = np.asarray(sac["returns"])
        assert returns.shape==baseline.shape==(128,) and np.isfinite(returns).all()
        record = dict(seed=seed,config=config,steps=sac["steps"],updates=sac["updates"],
                      score=float(returns.mean()),sac=sac["score"],episodes_equal=bool(np.array_equal(returns,baseline)),
                      max_abs_difference=float(np.abs(returns-baseline).max()),returns=returns.tolist())
        (folder/f"seed{seed}.json").write_text(json.dumps(record,indent=2))
        print(mode,seed,record["score"],record["sac"],record["episodes_equal"],flush=True)

if __name__=="__main__":
    main()

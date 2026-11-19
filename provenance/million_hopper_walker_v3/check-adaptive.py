import jax
from run_million_native import COMMON, TOTAL, NativeTask, NativeAdaptiveEQL
for name in ("Hopper-v3","Walker2d-v3"):
    algo=NativeAdaptiveEQL.create(env=NativeTask(name),**dict(COMMON,total_timesteps=TOTAL))
    state=jax.jit(algo.init_state)(jax.random.PRNGKey(7199))
    assert len(algo.actor_arms)==11 and algo.supports_episode_feedback()
    state=jax.jit(lambda s:jax.lax.fori_loop(0,450,lambda _,v:algo.train_iteration(v),s))(state)
    print(name, int(state.global_step), int(state.rounds), float(state.tau_q),flush=True)
    assert int(state.rounds)>0

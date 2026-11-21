"""Compare episode, step and shared replay on the paired Pendulum grid setting."""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rejax.buffers import Minibatch

from eql.algorithms.episode_bag import EpisodeBagEQL, PendulumEpisodeBags
from eql.records import read, save, verify_source
from eql.replay import BagReplayBuffer
from eql.training import COMMON, SEEDS, ready
from experiments._pendulum import GRID_RESULTS
from experiments._pendulum import grid_file as file

ROOT = Path("results/pendulum_episode_bags")
CONFIG = COMMON | dict(tau_Q=0.1, tau_a=0.7)
METHODS = ("EQL", "EQL_bagging", "episode")


def tagged_boundary_test():
    example = Minibatch(*(jnp.array(0) for _ in range(5)))
    bags = PendulumEpisodeBags.from_empty(BagReplayBuffer.empty(example, 10, 64))
    history = []
    for t in range(6):
        tags = jnp.arange(10)[:,None] + 100*t
        batch = Minibatch(tags,tags,tags,jnp.full((10,1),t%3==2),tags)
        bags, assignment = bags.add(batch,jax.random.PRNGKey(t+1))
        history.append(np.array(assignment))
        stored = np.asarray(bags.data.obs)[:,2*t:2*t+2]
        for env in range(10):
            found = np.flatnonzero((stored==100*t+env).any(axis=1))
            np.testing.assert_array_equal(found, np.sort(history[-1][env]))
            assert len(found)==2
    for start in (0,3):
        for t in (start+1,start+2):
            np.testing.assert_array_equal(history[t],history[start])
    assert not np.array_equal(history[0],history[3])


def audit_trajectories(state, horizon):
    # The 100k budget fits in both original buffers. Recover the two source
    # envs for each bag and verify every field for the entire 200-step episode.
    steps = int(state.global_step)
    envs = CONFIG["num_envs"]
    waves = steps // (horizon*envs)
    assert steps == waves*horizon*envs == int(state.replay_buffer.num_entries)
    per_bag = int(state.bags.count)
    assert per_bag == 2*waves*horizon
    global_data = {k:np.asarray(v[:steps]).reshape(waves,horizon,envs,*v.shape[1:])
                   for k,v in state.replay_buffer.data._asdict().items()}
    bag_data = {k:np.asarray(v[:,:per_bag]).reshape(envs,waves,horizon,2,*v.shape[2:])
                for k,v in state.bags.data._asdict().items()}
    done = global_data["done"]
    assert not done[:,:-1].any() and done[:,-1].all()
    pairs_by_wave = []
    for w in range(waves):
        pairs = [[] for _ in range(envs)]
        for bag in range(envs):
            for slot in range(2):
                source = np.flatnonzero((global_data["obs"][w,0] == bag_data["obs"][bag,w,0,slot]).all(axis=-1))
                assert len(source)==1
                env=int(source[0])
                pairs[env].append(bag)
                for field in global_data:
                    np.testing.assert_array_equal(bag_data[field][bag,w,:,slot],global_data[field][w,:,env])
        assert all(len(set(pair))==len(pair)==2 for pair in pairs)
        pairs_by_wave.append(pairs)
    return dict(verified_complete_episodes=waves*envs,entries_per_bag=per_bag,bag_pairs=pairs_by_wave)


def comparison():
    scores = {}
    for method in METHODS:
        values = []
        for seed in SEEDS:
            path = ROOT/f"seed{seed}.json" if method=="episode" else file(method,0.1,0.7,seed)
            run = read(path)
            assert run["config"]==CONFIG and run["seed"]==seed and run["steps"]==100000 and run["updates"]==9796
            raw = np.asarray(run["returns"],np.float32)
            assert raw.shape==(128,) and np.isfinite(raw).all()
            np.testing.assert_allclose(raw.mean(),run["score"],rtol=1e-6)
            values.append(run["score"])
        scores[method]=np.array(values)
    lines=["# Pendulum-v1 完整 episode 分袋对照","",
           "学习率 0.001，τ_Q=0.1，τ_a=0.7。三组均使用 10 个 critic、10 个环境、100000 次环境交互、9796 次网络更新、每 critic batch=256。两种分袋均为两个不同袋、无放回采样；仅完整 episode 组为本轮新跑。","",
           "| 方法 | seed 201 | seed 202 | seed 203 | seed 204 | 均值 ± seed 样本标准差 |",
           "|---|---:|---:|---:|---:|---:|"]
    for label,m in zip(("不分袋 EQL","一步分袋 EQL","完整 episode 分袋 EQL"),METHODS):
        v=scores[m]
        lines.append(f"| {label} | "+" | ".join(f"{i:.2f}" for i in v)+f" | {v.mean():.2f} ± {v.std(ddof=1):.2f} |")
    for m in METHODS[:2]:
        delta=scores["episode"]-scores[m]
        lines+=["",f"完整 episode 减去 {m}：配对差值为 "+", ".join(f"{x:+.2f}" for x in delta)+f"，均值 {delta.mean():+.2f}。"]
    lines+=["","每个 seed 的 500 条完整轨迹均根据原始 replay 回查了两个袋中全部 200 步的五个字段（含终止步）。每个袋含 20000 条交互。逐轨迹所属袋和 128 个评估回合的原始回报保存在 seed JSON 中。"]
    (ROOT/"RESULTS_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


def main():
    assert jax.devices()[0].platform=="gpu"
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=read(GRID_RESULTS/"protocol.json")
    for name,expected in protocol["source_hash"].items():
        verify_source(name, expected)
    tagged_boundary_test()
    print("Tagged boundary test passed",flush=True)
    algo=EpisodeBagEQL.create(env="Pendulum-v1",**CONFIG)
    init=jax.jit(algo.init_state)
    train=jax.jit(lambda s:jax.lax.fori_loop(0,1000,lambda _,v:algo.train_iteration(v),s),donate_argnums=(0,))
    evaluate=jax.jit(lambda s,k:algo.eval_callback(algo,s,k)[1])
    for seed in SEEDS:
        state=init(jax.random.PRNGKey(seed))
        for _ in range(10):
            state=train(state)
            ready(state)
        assert int(state.global_step)==100000 and int(state.actor_ts.step)==int(state.critic_ts.step)==9796
        audit=audit_trajectories(state,algo.env_params.max_steps_in_episode)
        returns=np.asarray(evaluate(state,jax.random.PRNGKey(30000+seed)))
        assert returns.shape==(128,) and np.isfinite(returns).all()
        score=float(returns.mean())
        save(ROOT/f"seed{seed}.json",dict(config=CONFIG,seed=seed,steps=100000,updates=9796,
             num_duplicates=2,score=score,returns=returns.tolist(),audit=audit))
        print(f"seed {seed}: return {score:.3f}, verified {audit['verified_complete_episodes']} episodes",flush=True)
    comparison()


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()

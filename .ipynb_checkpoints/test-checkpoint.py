# from rejax import get_algo

# env_str = "CartPole-v1"  # @param ["CartPole-v1", "Acrobot-v1", "Pendulum-v1"]
# algo_str = "dqn"  # @param ["ppo", "dqn", "sac", "td3"]

# # We make some changes to the default hyperparameters to get better plots.
# # All algorithms should work reasonably well on CartPole-v1 and Acrobot-1.
# CONFIGS = {
#     "sac": {"target_entropy_ratio": 0.7, "num_envs": 10, "num_epochs": 5},
#     "ppo": {"total_timesteps": 3e5},
#     "dqn": {},
#     "td3": {"total_timesteps": 5e4, "normalize_observations": True},
# }

# algo_cls = get_algo(algo_str)
# algo = algo_cls.create(env=env_str, learning_rate=0.0003, **CONFIGS.get(algo_str, {}))

# print(algo.config)

# algo = algo.replace(gamma=0.995,total_timesteps=1000000)
# print(f"New gamma: {algo.gamma}")


# import jax
# import time

# # Set training seed and jit train function
# rng = jax.random.PRNGKey(0)
# train_fn = jax.jit(algo.train)

# print("Starting to train")

# # Train!
# start = time.time()
# train_state, evaluation = train_fn(rng)
# time_elapsed = time.time() - start

# sps = algo.total_timesteps / time_elapsed
# print(f"Finished training in {time_elapsed:g} seconds ({sps:g} steps/second).")

# from matplotlib import pyplot as plt

# episode_lengths, episode_returns = evaluation
# plt.plot(jax.numpy.linspace(0, algo.total_timesteps, len(mean_return)), mean_return)
# plt.xn_return = episode_returns.mean(axis=1)

# plt.label("Environment step")
# plt.ylabel("Episodic return")
# plt.title(f"Training agent for {env_str} using {algo_str}")
# plt.show()

# plt.savefig("cartpole_dqn_returns.png", dpi=300, bbox_inches="tight")
# print("图像已保存为 cartpole_dqn_returns.png")







# import tqdm
# import time
# import jax
# import jax.numpy as jnp
# from matplotlib import pyplot as plt
# from rejax import get_algo
# from tqdm import tqdm

# # 1. 参数与算法初始化
# env_str = "CartPole-v1"
# algo_str = "sac"

# CONFIGS = {
#     "sac": {"target_entropy_ratio": 0.7, "num_envs": 10, "num_epochs": 5},
#     "ppo": {"total_timesteps": 3e5},
#     "dqn": {},
#     "td3": {"total_timesteps": 5e4, "normalize_observations": True},
# }

# algo_cls = get_algo(algo_str)
# algo = algo_cls.create(env=env_str, learning_rate=0.0003, **CONFIGS.get(algo_str, {}))

# # 修改配置
# algo = algo.replace(gamma=0.995, total_timesteps=1_000_000)
# print(f"Algorithm config: {algo.config}")
# print(f"New gamma: {algo.gamma}")

# # 2. 设置训练与进度条分块
# num_chunks = 100  # 将 100 万步分为 100 个 Chunk，每个 Chunk 训练 10,000 步
# steps_per_chunk = algo.total_timesteps // num_chunks
# chunk_algo = algo.replace(total_timesteps=steps_per_chunk)

# # JIT 编译单块训练函数
# train_fn = jax.jit(chunk_algo.train)

# rng = jax.random.PRNGKey(0)
# print("Starting to train...")
# start_time = time.time()

# # 3. 带 tqdm 进度条的训练循环
# returns_list = []
# lengths_list = []

# # 如果 algo 支持链式传入 state，可以分步更新；若直接单次运行，以下为通用分块处理逻辑：
# for i in tqdm(range(num_chunks), desc="Training Progress", unit="chunk"):
#     rng, step_rng = jax.random.split(rng)
#     train_state, evaluation = train_fn(step_rng)
    
#     ep_lengths, ep_returns = evaluation
#     returns_list.append(ep_returns)
#     lengths_list.append(ep_lengths)

# time_elapsed = time.time() - start_time
# sps = algo.total_timesteps / time_elapsed
# print(f"Finished training in {time_elapsed:.2f} seconds ({sps:.2f} steps/second).")

# # 4. 数据汇总与修正后的绘图逻辑
# all_returns = jnp.concatenate(returns_list, axis=0)  # 拼接评估结果
# mean_return = all_returns.mean(axis=-1)             # 修正计算平均 Return

# plt.figure(figsize=(8, 5))
# plt.plot(jnp.linspace(0, algo.total_timesteps, len(mean_return)), mean_return)
# plt.xlabel("Environment step")                      # 修正 plt.label -> plt.xlabel
# plt.ylabel("Episodic return")
# plt.title(f"Training agent for {env_str} using {algo_str}")
# plt.grid(True)

# # 修正保存顺序：先 Save 后 Show
# plt.savefig("cartpole_dqn_returns.png", dpi=300, bbox_inches="tight")
# print("图像已成功保存为 cartpole_dqn_returns.png")
# plt.show()





import time
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from rejax import get_algo


# 1. 参数与算法初始化
env_str = "CartPole-v1"
algo_str = "sac"  # 提示：若追求最高稳定度，可将 algo_str 改为 "dqn" 或 "ppo"

TOTAL_TIMESTEPS = 100_000  # 修改为 10 万步

CONFIGS = {
    "sac": {"target_entropy_ratio": 0.7, "num_envs": 10, "num_epochs": 5},
    "ppo": {"total_timesteps": 3e5},
    "dqn": {},
    "td3": {"total_timesteps": 5e4, "normalize_observations": True},
}

algo_cls = get_algo(algo_str)
algo = algo_cls.create(env=env_str, learning_rate=0.0003, **CONFIGS.get(algo_str, {}))

# 设置新的超参数
algo = algo.replace(gamma=0.995, total_timesteps=TOTAL_TIMESTEPS)
print(f"Algorithm config: {algo.config}")
print(f"New gamma: {algo.gamma}")

# 2. 一次性 JIT 编译训练（保证网络权重与 Replay Buffer 连续更新）
train_fn = jax.jit(algo.train)

rng = jax.random.PRNGKey(0)
print(f"Starting to train {algo_str.upper()} on {env_str} for {TOTAL_TIMESTEPS} steps...")

start_time = time.time()
# 底层并行运行全量 10 万步，无 Python 循环开销
train_state, evaluation = train_fn(rng)
time_elapsed = time.time() - start_time

sps = algo.total_timesteps / time_elapsed
print(f"Finished training in {time_elapsed:.2f} seconds ({sps:.2f} steps/second).")

# 3. 数据解析与修正后的绘图逻辑
ep_lengths, ep_returns = evaluation
mean_return = ep_returns.mean(axis=-1)  # 跨并行环境求平均回报

plt.figure(figsize=(8, 5))
plt.plot(jnp.linspace(0, algo.total_timesteps, len(mean_return)), mean_return, linewidth=1.5)
plt.xlabel("Environment step")
plt.ylabel("Episodic return")
plt.title(f"Training agent for {env_str} using {algo_str.upper()}")
plt.grid(True, linestyle="--", alpha=0.7)

# 动态指定文件名，避免硬编码
save_filename = f"cartpole_{algo_str}_returns.png"
plt.savefig(save_filename, dpi=300, bbox_inches="tight")
print(f"图像已成功保存为 {save_filename}")
plt.show()
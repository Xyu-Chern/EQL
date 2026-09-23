import pytest
import jax
import jax.numpy as jnp
# 假设我们后续封装的类名为 BaggingFlashbaxBuffer
# from bagging_buffer import make_bagging_flashbax_buffer


class TestBaggingFlashbaxBuffer:

    @pytest.fixture
    def setup_buffer(self):
        """初始化配置与 Key"""
        key = jax.random.PRNGKey(42)
        num_critics = 3
        buffer_size = 1000
        num_envs = 6
        num_steps = 10
        obs_dim = 4
        act_dim = 2
        per_bag_batch_size = 32

        # 示例数据模版 (Sample Example PyTree)
        fake_item = {
            "obs": jnp.zeros((obs_dim,)),
            "action": jnp.zeros((act_dim,)),
            "reward": jnp.zeros(()),
            "next_obs": jnp.zeros((obs_dim,)),
        }

        return {
            "key": key,
            "num_critics": num_critics,
            "buffer_size": buffer_size,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "per_bag_batch_size": per_bag_batch_size,
            "fake_item": fake_item,
        }

    def test_init_state_shape(self, setup_buffer):
        """测试1: 初始化后的 BufferState 维度是否包含 num_critics 轴"""
        cfg = setup_buffer
        # builder = make_bagging_flashbax_buffer(...)
        # state = builder.init(cfg["fake_item"])
        
        # 验证每个 Bag 的容量与索引状态维度为 (num_critics, ...)
        # assert state.experience['obs'].shape == (cfg["num_critics"], cfg["buffer_size"], 4)
        pass

    def test_vmap_add_distribution(self, setup_buffer):
        """测试2: 验证 add 操作能否通过 vmap 正确将 env 轨迹分发至各个 Bag"""
        cfg = setup_buffer
        key = cfg["key"]

        # 模拟生成 (num_envs, num_steps, obs_dim) 的轨迹数据
        batch_obs = jnp.ones((cfg["num_envs"], cfg["num_steps"], 4))
        # 故意给每个 env 赋不同的特征值，测试 Trajectory-level 完整性
        env_ids = jnp.arange(cfg["num_envs"])[:, None, None]
        batch_obs = batch_obs * env_ids

        batch_data = {"obs": batch_obs}

        # state = builder.add(state, batch_data)
        
        # 断言: 验证第 i 个 Critic 的 Bag 中接收到的轨迹是否完整且不交叉
        pass

    def test_jit_compatibility(self, setup_buffer):
        """测试3: 验证 add 和 sample 函数是否可以顺利被 jax.jit 编译"""
        cfg = setup_buffer
        key = cfg["key"]

        # @jax.jit
        # def step_fn(state, batch, key):
        #     state = builder.add(state, batch)
        #     batch = builder.sample(state, key)
        #     return state, batch

        # 运行一次以触发 JIT 编译，确保无 Tracer / Dynamic Shape 报错
        pass

    def test_independent_sampling_per_critic(self, setup_buffer):
        """测试4: 采样出来的 Batch 维度应为 (num_critics, per_bag_batch_size, ...)"""
        cfg = setup_buffer
        key = cfg["key"]

        # sampled_batch = builder.sample(state, key)
        # assert sampled_batch['obs'].shape == (cfg["num_critics"], cfg["per_bag_batch_size"], 4)
        pass
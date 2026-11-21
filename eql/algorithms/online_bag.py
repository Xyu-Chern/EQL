"""Online continuous-control EQL with teacher's fragment replay bags.

Reuses Rejax SAC network/optimizer/temperature definitions. New train loop
explicitly accounts for num_envs*num_steps transitions and per-critic batches.
"""
import jax
import jax.numpy as jnp
import optax
from flax import struct
from rejax.algos.algorithm import register_init
from rejax.algos.sac import SAC
from rejax.buffers import Minibatch

from eql.expectiles import expectile
from eql.replay import BagReplayBuffer, GlobalReplayBuffer


class BagEQL(SAC):
    num_steps: int = struct.field(pytree_node=False, default=8)
    num_duplicates: int = struct.field(pytree_node=False, default=1)
    data_mode: str = struct.field(pytree_node=False, default='bag')
    target_mode: str = struct.field(pytree_node=False, default='shared')
    tau_actor: float = struct.field(pytree_node=True, default=0.3)
    tau_target: float = struct.field(pytree_node=True, default=0.3)

    def train(self, rng=None, train_state=None):
        # SAC's inherited loop assumes ONE vector step per train_iteration.
        # This subclass collects num_steps, so count actual transitions instead.
        if train_state is None and rng is None:
            raise ValueError('rng or train_state is required')
        ts = train_state if train_state is not None else self.init_state(rng)
        stride = self.num_envs*self.num_steps
        if self.eval_freq % stride or self.total_timesteps % self.eval_freq:
            raise ValueError('Use exact multiples of rollout and evaluation sizes')
        initial = self.eval_callback(self, ts, ts.rng)
        def block(ts, _):
            ts = jax.lax.fori_loop(0,self.eval_freq//stride,lambda _,s:self.train_iteration(s),ts)
            return ts, self.eval_callback(self,ts,ts.rng)
        ts, ev = jax.lax.scan(block,ts,None,self.total_timesteps//self.eval_freq)
        if not self.skip_initial_evaluation:
            ev = jax.tree.map(lambda a,b:jnp.concatenate([a[None],b]),initial,ev)
        return ts, ev

    @register_init
    def initialize_replay_buffer(self, rng):
        if self.discrete:
            raise ValueError('This experiment entry point supports continuous actions only')
        if self.data_mode not in ('bag', 'shared') or self.target_mode not in ('shared', 'own'):
            raise ValueError('Invalid data_mode or target_mode')
        if self.buffer_size * self.num_duplicates % self.num_critics:
            raise ValueError('Require exactly matched replay horizons')
        example = Minibatch(jnp.zeros(self.obs_space.shape), jnp.zeros(self.action_space.shape),
                            jnp.array(0., jnp.float32), jnp.array(False), jnp.zeros(self.obs_space.shape))
        return {'replay_buffer': GlobalReplayBuffer.empty(example, self.buffer_size),
                'bags': BagReplayBuffer.empty(example, self.num_critics,
                            self.buffer_size*self.num_duplicates//self.num_critics),
                'diagnostics': jnp.zeros(4), 'updates': jnp.array(0, jnp.int32)}

    def collect_transitions(self, ts):
        # Use step_env to retain pre-autoreset next_obs and bootstrap time limits.
        # This entry point is validated on Gymnax Pendulum/Reacher time-limit tasks.
        rng, ak, sk, rk, warm = jax.random.split(ts.rng, 5)
        actions = self.actor.apply(ts.actor_ts.params, ts.last_obs, ak, method='act')
        random_actions = jax.random.uniform(warm, actions.shape,
                           minval=self.action_space.low, maxval=self.action_space.high)
        actions = jnp.where(ts.global_step < self.fill_buffer, random_actions, actions)
        next_obs, env_state, rewards, dones, _ = jax.vmap(self.env.step_env, in_axes=(0,0,0,None))(
            jax.random.split(sk, self.num_envs), ts.env_state, actions, self.env_params)
        reset_obs, reset_state = self.vmap_reset(jax.random.split(rk, self.num_envs), self.env_params)
        timeout = env_state.time >= self.env_params.max_steps_in_episode
        terminated = dones & ~timeout
        def choose(reset, current):
            shape = (self.num_envs,) + (1,)*(current.ndim-1)
            return jnp.where(dones.reshape(shape), reset, current)
        live_obs = choose(reset_obs, next_obs)
        live_state = jax.tree.map(choose, reset_state, env_state)
        mb = Minibatch(ts.last_obs, actions, rewards, terminated, next_obs)
        return ts.replace(rng=rng, last_obs=live_obs, env_state=live_state,
                          global_step=ts.global_step+self.num_envs), mb

    def critic_predictions(self, params, mb):
        return jax.vmap(self.critic.apply, in_axes=(0,0,0))(params, mb.obs, mb.action)

    def targets(self, ts, mb, key):
        n, b = mb.reward.shape
        obs = mb.next_obs.reshape((n*b, *mb.next_obs.shape[2:]))
        actions, lp = self.actor.apply(ts.actor_ts.params, obs, key, method='action_log_prob')
        if self.target_mode == 'shared':
            all_q = self.vmap_critic(ts.critic_target_params, obs, actions)
            next_q = expectile(all_q, self.tau_target).reshape(n,b)
        else:
            obs = obs.reshape(mb.next_obs.shape)
            actions = actions.reshape((n,b,*actions.shape[1:]))
            next_q = jax.vmap(self.critic.apply, in_axes=(0,0,0))(ts.critic_target_params, obs, actions)
        soft_q = next_q-jnp.exp(ts.alpha_ts.params['log_alpha'])*lp.reshape(n,b)
        return jax.lax.stop_gradient(mb.reward+self.gamma*(1-mb.done)*soft_q)

    def update_actor(self, ts, mb):
        rng, key = jax.random.split(ts.rng)
        alpha = jnp.exp(ts.alpha_ts.params['log_alpha'])
        def loss(params):
            a, lp = self.actor.apply(params, mb.obs, key, method='action_log_prob')
            qs = self.vmap_critic(ts.critic_ts.params, mb.obs, a)
            return (alpha*lp-expectile(qs, self.tau_actor)).mean(), lp
        (_, lp), grad = jax.value_and_grad(loss, has_aux=True)(ts.actor_ts.params)
        return ts.replace(rng=rng, actor_ts=ts.actor_ts.apply_gradients(grads=grad)), lp

    def update_bags(self, ts):
        rng, bk, actor_key, target_key = jax.random.split(ts.rng, 4)
        if self.data_mode == 'bag':
            mb = ts.bags.sample(bk, self.batch_size)
        else:
            mb = jax.vmap(lambda k: ts.replay_buffer.sample(k, self.batch_size))(
                jax.random.split(bk, self.num_critics))
        actor_mb = ts.replay_buffer.sample(actor_key, self.batch_size)
        ts = ts.replace(rng=rng)
        ts, lp = self.update_actor(ts, actor_mb)
        target = self.targets(ts, mb, target_key)
        def loss(params):
            pred = self.critic_predictions(params, mb)
            return optax.l2_loss(pred, target).mean(axis=1).sum(), pred
        (critic_loss, pred), grad = jax.value_and_grad(loss, has_aux=True)(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grad))
        ts = self.update_alpha(ts, lp)
        target_params = jax.tree.map(lambda p,t: (1-self.polyak)*p+self.polyak*t,
                                    ts.critic_ts.params, ts.critic_target_params)
        # Disagreement evaluated at COMMON queries, unlike per-bag predictions.
        query_q = self.vmap_critic(ts.critic_ts.params, actor_mb.obs, actor_mb.action)
        diag = jnp.array([critic_loss, pred.mean(), query_q.std(axis=0).mean(),
                          jnp.exp(ts.alpha_ts.params['log_alpha'])])
        return ts.replace(critic_target_params=target_params, diagnostics=diag, updates=ts.updates+1)

    def train_iteration(self, ts):
        ts, batch = jax.lax.scan(lambda state,_: self.collect_transitions(state), ts, None, self.num_steps)
        fragments = jax.tree.map(lambda x: jnp.swapaxes(x,0,1), batch)
        flat = jax.tree.map(lambda x: x.reshape((-1,*x.shape[2:])), batch)
        rng, bk = jax.random.split(ts.rng)
        bags, _ = ts.bags.add(fragments, bk, self.num_duplicates)
        ts = ts.replace(rng=rng, bags=bags, replay_buffer=ts.replay_buffer.add(flat))
        ready = (ts.global_step > self.fill_buffer) & ts.bags.ready(self.batch_size)
        return jax.lax.cond(ready,
            lambda s: jax.lax.fori_loop(0,self.num_epochs,lambda _,v:self.update_bags(v),s),
            lambda s:s,ts)

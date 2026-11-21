import jax
import jax.numpy as jnp
import numpy as np

from eql.algorithms.eql import EQL
from eql.algorithms.original_bag import OriginalBagEQL
from eql.algorithms.sac import SAC
from eql.training import CONFIG


def test_original_initialization_and_actor_preserved():
    models=[c.create(env='Pendulum-v1',**CONFIG) for c in (SAC,EQL,OriginalBagEQL)]
    states=[a.init_state(jax.random.PRNGKey(8)) for a in models]
    for s in states[1:]:
        for a,b in zip(jax.tree.leaves(states[0].actor_ts.params),jax.tree.leaves(s.actor_ts.params)):
            np.testing.assert_array_equal(a,b)
        for a,b in zip(jax.tree.leaves(states[0].critic_ts.params),jax.tree.leaves(s.critic_ts.params)):
            np.testing.assert_array_equal(a,b)
    assert OriginalBagEQL.update_actor is EQL.update_actor
    assert OriginalBagEQL.collect_transitions is EQL.collect_transitions
    assert models[2].tau_a==0.7 and models[2].tau_Q==0.3
    assert models[2].polyak==0.99 and models[2].batch_size==256

def test_bag_sampler_no_unwritten_or_duplicate_entries():
    a=OriginalBagEQL.create(env='Pendulum-v1',**CONFIG)
    s=a.init_state(jax.random.PRNGKey(0))
    data=s.bags.data._replace(reward=jnp.broadcast_to(jnp.arange(a.buffer_size),(10,a.buffer_size)))
    bags=s.bags.replace(data=data,count=jnp.array(410))
    mb=jax.jit(a.sample_bags)(bags,jax.random.PRNGKey(17))
    for row in np.asarray(mb.reward):
        assert len(set(row.tolist()))==256
        assert row.min()>=0 and row.max()<410

def test_first_update_keeps_original_budget():
    a=OriginalBagEQL.create(env='Pendulum-v1',**CONFIG)
    s=a.init_state(jax.random.PRNGKey(1))
    s=jax.jit(lambda s:jax.lax.fori_loop(0,205,lambda _,s:a.train_iteration(s),s))(s)
    assert int(s.global_step)==2050
    assert int(s.actor_ts.step)==1 and int(s.critic_ts.step)==1
    assert int(s.bags.count)==410
    assert int(s.replay_buffer.num_entries)==2050
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(s.actor_ts.params))

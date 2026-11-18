import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bagging_buffer import BagReplayBuffer, GlobalReplayBuffer, balanced_assignment
from expectile import expectile
from online_bag_eql import BagEQL


@pytest.mark.parametrize('n,e,d', [(2,2,1),(4,4,1),(4,4,2),(4,4,4),(3,6,2)])
def test_assignment_and_whole_fragments(n,e,d):
    values = jnp.arange(e*8).reshape(e,8)
    buf = BagReplayBuffer.empty({'id': jnp.array(0)}, n, 64)
    buf, assignments = buf.add({'id':values},jax.random.PRNGKey(5),d)
    a = np.asarray(assignments)
    for row in a:
        assert len(set(row)) == d
    np.testing.assert_array_equal(np.bincount(a.ravel(),minlength=n), np.full(n,e*d//n))
    for i in range(n):
        expected = np.sort(np.asarray(values)[np.any(a==i,axis=1)].ravel())
        np.testing.assert_array_equal(np.sort(np.asarray(buf.data['id'][i,:int(buf.count)])), expected)
    sample = buf.sample(jax.random.PRNGKey(2),128)['id']
    assert sample.shape == (n,128)
    for i in range(n):
        assert np.isin(sample[i], buf.data['id'][i,:int(buf.count)]).all()


def test_invalid_fragment_divisibility():
    # Transition count divisible but whole fragments cannot be balanced.
    with pytest.raises(ValueError):
        balanced_assignment(jax.random.PRNGKey(0),1,2,1)
    with pytest.raises(ValueError):
        balanced_assignment(jax.random.PRNGKey(0),4,2,3)


def test_wrap_replace_and_sampling_keys():
    buf = BagReplayBuffer.empty({'id':jnp.array(0)},2,8)
    for start in (0,8,16):
        buf,_=buf.add({'id':jnp.arange(start,start+8).reshape(2,4)},jax.random.PRNGKey(0))
    assert int(buf.count)==8 and int(buf.index)==4
    assert np.asarray(buf.data['id']).min()>=8
    a=buf.sample(jax.random.PRNGKey(7),128)
    b=buf.sample(jax.random.PRNGKey(7),128)
    np.testing.assert_array_equal(a['id'],b['id'])
    # Identical bags still use independent sample keys.
    buf,_=buf.add({'id':jnp.arange(8).reshape(2,4)},jax.random.PRNGKey(0),2,replace=True)
    sample=buf.sample(jax.random.PRNGKey(7),128)['id']
    assert not np.array_equal(sample[0],sample[1])
    buf,_=buf.add({'id':jnp.arange(100,108).reshape(2,4)},jax.random.PRNGKey(0),1,replace=True)
    assert int(buf.count)==4
    assert np.asarray(buf.sample(jax.random.PRNGKey(3),128)['id']).min()>=100


def test_global_replay_has_all_original_data():
    buf=GlobalReplayBuffer.empty({'id':jnp.array(0)},16)
    buf=buf.add({'id':jnp.arange(16)})
    np.testing.assert_array_equal(buf.data['id'], np.arange(16))
    buf=buf.add({'id':jnp.arange(16,20)})
    assert int(buf.count)==16
    assert np.asarray(buf.sample(jax.random.PRNGKey(4),128)['id']).min()>=4


def test_expectile_solution_and_gradient():
    q=jnp.array([-3.,-0.7,0.2,2.,8.])
    for t in (0.,0.1,0.3,0.5,0.9,1.):
        y=float(expectile(q,t))
        if 0<t<1:
            residual=jnp.where(q<y,1-t,t)*(q-y)
            assert abs(float(residual.sum()))<2e-6
        np.testing.assert_allclose(expectile(q+5,t),y+5,atol=2e-6)
    np.testing.assert_allclose(expectile(q,0.5),q.mean(),atol=1e-6)
    np.testing.assert_allclose(expectile(q,0.),q.min())
    np.testing.assert_allclose(expectile(q,1.),q.max())
    slopes=jnp.array([0.3,-0.4,0.2,0.7,-0.6])
    f=lambda a: expectile(q+a*slopes,0.3)
    fd=(f(0.001)-f(-0.001))/0.002
    np.testing.assert_allclose(jax.grad(f)(0.),fd,atol=2e-4)
    for t in (0.1,0.3,0.7):
        z=jnp.array([-2.,4.])
        np.testing.assert_allclose(expectile(z,t),z.mean()+(2*t-1)*3,atol=1e-6)
    assert float(expectile(jnp.ones(4)*7,0.3)) == pytest.approx(7,abs=1e-6)


@pytest.mark.parametrize('env', ['Pendulum-v1','Reacher-misc'])
def test_terminal_next_obs_and_bootstrap(env):
    algo=BagEQL.create(env=env,num_envs=4,num_critics=4,buffer_size=256,batch_size=8)
    state=algo.init_state(jax.random.PRNGKey(0))
    state=state.replace(env_state=state.env_state.replace(
        time=jnp.full_like(state.env_state.time,algo.env_params.max_steps_in_episode-1)))
    new, mb=jax.jit(algo.collect_transitions)(state)
    assert not np.asarray(mb.done).any()  # time limits bootstrap
    assert np.asarray(new.env_state.time==0).all()
    assert not np.array_equal(np.asarray(mb.next_obs),np.asarray(new.last_obs))


def test_critic_gradient_isolation():
    algo=BagEQL.create(env='Pendulum-v1',num_envs=4,num_critics=4,buffer_size=256,batch_size=8)
    state=algo.init_state(jax.random.PRNGKey(0))
    _,batch=jax.lax.scan(lambda s,_:algo.collect_transitions(s),state,None,8)
    batch=jax.tree.map(lambda x:jnp.swapaxes(x,0,1),batch)
    def loss(params):
        # Only critic 2's own supervised observations enter this loss.
        pred=algo.critic_predictions(params,batch)
        return ((pred[2]-1.)**2).mean()
    grads=jax.grad(loss)(state.critic_ts.params)
    for g in jax.tree.leaves(grads):
        np.testing.assert_array_equal(np.asarray(g)[[0,1,3]],0)
    assert sum(float(jnp.abs(g[2]).sum()) for g in jax.tree.leaves(grads))>0


@pytest.mark.parametrize('data,target',[('shared','shared'),('bag','shared'),('shared','own'),('bag','own')])
def test_train_iteration_modes(data,target):
    algo=BagEQL.create(env='Pendulum-v1',num_envs=4,num_critics=4,buffer_size=256,
        batch_size=8,fill_buffer=0,num_epochs=1,data_mode=data,target_mode=target)
    state=algo.init_state(jax.random.PRNGKey(3))
    new=jax.jit(algo.train_iteration)(state)
    assert int(new.global_step)==32 and int(new.updates)==1
    assert int(new.bags.count)==8 and int(new.replay_buffer.count)==32
    assert np.isfinite(np.asarray(new.diagnostics)).all()

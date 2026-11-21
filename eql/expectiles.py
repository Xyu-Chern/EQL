"""Empirical expectile aggregation with the sorted and original iterative estimators."""
from functools import partial

import jax
import jax.numpy as jnp


@jax.custom_jvp
def expectile(values, tau):
    """Aggregate axis 0. Endpoint extension is min/max, not endpoint argmin.

    At ties an equal-side convention selects an autodiff value; no assertion
    of classical differentiability at those boundaries is made.
    """
    z = jnp.sort(values, axis=0)
    n = z.shape[0]
    prefix = jnp.concatenate([jnp.zeros_like(z[:1]), jnp.cumsum(z, axis=0)], axis=0)
    k = jnp.arange(n+1).reshape((n+1,) + (1,)*(z.ndim-1))
    t = jnp.clip(tau, 1e-6, 1-1e-6)
    y = ((1-t)*prefix + t*(prefix[-1]-prefix)) / ((1-t)*k + t*(n-k))
    low = jnp.concatenate([jnp.full_like(z[:1], -jnp.inf), z], axis=0)
    high = jnp.concatenate([z, jnp.full_like(z[:1], jnp.inf)], axis=0)
    # Floating point roundoff near boundaries: pick the smallest violation.
    violation = jnp.maximum(low-y, 0) + jnp.maximum(y-high, 0)
    idx = jnp.argmin(violation, axis=0)
    out = jnp.take_along_axis(y, idx[None], axis=0)[0]
    return jnp.where(tau <= 0, z[0], jnp.where(tau >= 1, z[-1], out))


@expectile.defjvp
def expectile_jvp(primals, tangents):
    values, tau = primals
    dvalues, dtau = tangents
    y = expectile(values, tau)
    w = jnp.where(values < y, 1-tau, tau)
    total = jnp.maximum(w.sum(axis=0), 1e-12)
    dy = (w*dvalues).sum(axis=0)/total + jnp.abs(values-y).sum(axis=0)/total*dtau
    w_min = (values == values.min(axis=0)).astype(values.dtype)
    w_max = (values == values.max(axis=0)).astype(values.dtype)
    dy = jnp.where(tau <= 0, (w_min*dvalues).sum(axis=0)/w_min.sum(axis=0), dy)
    dy = jnp.where(tau >= 1, (w_max*dvalues).sum(axis=0)/w_max.sum(axis=0), dy)
    return y, dy


@partial(jax.jit, static_argnames=('axis', 'num_iters'))
def compute_expectile_on_axis(data, tau, axis, num_iters=10):
    """E^tau(x) = argmin_y E[ rho_tau(x - y) ] where rho_tau(u) = |tau - 1_{u < 0}| * u^2"""
    # The continuous extension at tau=0 is min, including SAC's gradient at ties.
    return jax.lax.cond(
        tau == 0,
        lambda: jnp.min(data, axis=axis),
        lambda: _irls_expectile(data, tau, axis, num_iters),
    )


def _irls_expectile(data, tau, axis, num_iters):
    y = jnp.mean(data, axis=axis, keepdims=True)
    tau = jnp.clip(tau, 0.0001, 0.99999)
    def body_fn(i, current_y):
        diff = data - current_y
        weight = jnp.where(diff <= 0, 1.0 - tau, tau)
        numerator = jnp.sum(weight * data, axis=axis, keepdims=True)
        denominator = jnp.sum(weight, axis=axis, keepdims=True)
        return numerator / (denominator + 1e-6)

    final_y = jax.lax.fori_loop(0, num_iters, body_fn, y)
    return jnp.squeeze(final_y, axis=axis)

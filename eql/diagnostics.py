"""Shared Pendulum evaluation panel and critic error statistics."""
from pathlib import Path

import numpy as np

ROOT = Path("results/pendulum_critic_errors")
COUNTS = [10,20,50]
MC_REPS = 256
REPLAY_METHODS = ("shared", "step", "episode")


def make_panel():
    # Evaluate the same observation at episode start (full 200-step horizon).
    theta, velocity, action = np.meshgrid(np.arange(-8,8)*np.pi/8,
                              [-7.,-4.,-2.,0.,2.,4.,7.],[-1.5,0.,1.5],indexing="ij")
    return tuple(x.ravel().astype(np.float32) for x in (theta,velocity,action))


def evaluate_errors(q, mc):
    # Multiple critics share a noisy MC label, adding its error variance to
    # every raw covariance. Cross-covariance between independent MC halves
    # removes that common noise in expectation. Symmetrize before scaling.
    left,right=np.split(mc.astype(np.float64),2,axis=1)
    qa=q.astype(np.float64)
    err_a=qa-left.mean(1)
    err_b=qa-right.mean(1)
    errors=(err_a+err_b)/2
    a=err_a-err_a.mean(1,keepdims=True)
    b=err_b-err_b.mean(1,keepdims=True)
    cov=(a@b.T+b@a.T)/(2*(a.shape[1]-1))
    assert np.all(np.diag(cov)>0)
    std=np.sqrt(np.diag(cov))
    corrected=cov/(std[:,None]*std[None,:])
    raw=np.corrcoef(errors)
    pair=np.triu_indices(len(q),1)
    positive=(errors>0).sum(0)
    k=len(q)
    same=(positive*(positive-1)+(k-positive)*(k-positive-1))/(k*(k-1))
    metrics=dict(correlation=float(corrected[pair].mean()),raw_correlation=float(raw[pair].mean()),
                 same_sign_fraction=float(same.mean()),mean_bias=float(errors.mean()),
                 critic_rmse=float(np.sqrt(np.mean(errors**2))),
                 ensemble_mean_rmse=float(np.sqrt(np.mean(errors.mean(0)**2))),
                 prediction_spread=float(qa.std(0).mean()),
                 median_mc_se=float(np.median(mc.std(1,ddof=1)/np.sqrt(mc.shape[1]))))
    assert np.isfinite(list(metrics.values())).all()
    return metrics, dict(mc_reference=mc.mean(1),errors=errors,correlation_corrected=corrected,
                         correlation_raw=raw,cross_covariance=cov)

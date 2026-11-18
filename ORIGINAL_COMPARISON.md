# Original configuration comparison

User-confirmed scope (2026-11-18): use `eql.py`, not the BC-loss variants;
10 parallel environments, one step per environment per collection, for all methods.
Target method: EQL + replay buffer + bagging. Baselines: user's SAC and EQL without bagging.

## Frozen settings

All methods: 10 critics, 100,000 total transitions, lr=3e-4, gamma=.98,
MLP=(64,64), ReLU, batch=256 per critic, actor batch=256,
buffer_size=131072, fill_buffer=2048, num_epochs=1, Polyak=.99,
target_update_freq=1, no observation/reward normalization, no gradient clipping.
The explicitly configured values come from original eql.py/sac.py; remaining
values are inherited Rejax 0.1.3 defaults. Four training keys are exactly
`jax.random.split(jax.random.PRNGKey(0),4)`, as in original eql.py.

EQL (both versions): tau_a=.7, tau_Q=.3, 10 IRLS iterations, original epsilon,
original automatic differentiation and entropy-temperature update.
SAC: original sac.py min over its 10 critics, not a canonical two-critic SAC.
There is no hyperparameter search or per-environment/per-method tuning.

## Bagging addition

Each collection supplies [10,1,...] fragments to [10,131072,...] bags.
num_duplicates=2: minimum integer giving each bag enough entries for the
original WITHOUT-replacement 256-sample update at step 2050 (410 entries).
With d=1 there would only be 205 entries. No fill_buffer or batch-size change
is used to resolve this constraint. Assignment is balanced and each copy goes
to a different critic. This is one-step-fragment bagging, NOT full-episode masks.
Actor samples original global data and uses the unchanged EQL objective.
Each critic learns from its own bag; targets use the complete target ensemble.
Per-bag capacity preserves the original buffer_size, as drawn in the image.
At this budget neither global nor bag buffers wrap, so all collected history
remains available. Allocated total memory differs, disclosed rather than hidden.
Per-bag sampling uses independent uniform random ranks and top-k, without
replacement; baseline replay uses the original Rejax sampler. Both sample
uniformly from written entries. The aggregate bag batch contains 10*256 entries,
matching the original 10 critics each trained on a 256-entry shared batch.

## Preservation and evaluation

Original eql.py and sac.py are imported without editing. Original collection
policy during warmup, done masking, environment autoreset, actor-before-critic
update and Polyak schedule are preserved for all methods. This comparison
does not mix in the previous run's time-limit bootstrap fix or exact expectile.
These original semantics remain possible research limitations.
Identical initial actor/critic parameters across methods are checked.
Custom outer loop stops at EXACTLY 100000 transitions, avoiding the inherited
train loop's rounding to eval_freq. All methods perform 9796 updates.
Every 5000 transitions: 128 stochastic-policy evaluation episodes, independent
of training keys and identical evaluation keys across methods. Report the
fixed final checkpoint, not best seed/checkpoint. Retain all four seeds.

## Reproduction

From this code directory in WSL:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false /root/.venvs/eql-gpu/bin/python -m pytest test_original_comparison.py -q
/root/.venvs/eql-gpu/bin/python -u run_original_suite.py
```
Each method/environment gets its own process. Outputs are isolated under
results/original_comparison; previous online_v1 results remain untouched.
Per-method manifests record resolved hyperparameters and source SHA256 hashes.

# Execution provenance

Initial launch ran Pendulum group A seeds 0,1,2 successfully. It then stalled while waiting for group B's initialized JAX state to become ready (Python/native stack inspected); group B had not started training.
Only the identified suite/training processes were terminated. Completed A outputs were kept unchanged.

The launcher was changed to one subprocess per environment/group, retaining sequential seeds within each process.
`run_online.py` changed only its manifest filename from `manifest.json` to `manifest_<groups>.json`; algorithm, buffer, expectile, seeds and hyperparameters were unchanged.

- Original A source hashes/configuration: `Pendulum-v1/manifest.json`.
- `Pendulum-v1/manifest_A.json` was emitted by the resumed invocation that SKIPPED completed A seeds, not a second training run.
- Other groups use their per-group manifest files.
- Initial logs and telemetry are retained with `_initial` suffix.
- No completed run was dropped, selected by score, or rerun for a better result.

A second wait occurred in Pendulum C seed 2 after step 24576, inside JAX's batched `block_until_ready(state)`. The runner now synchronizes each leaf separately instead of using the batched ready call; learning computations are unchanged.
The interrupted C seed 2 JSONL is preserved as `C_seed2_interrupted.jsonl`. This incomplete seed was restarted from its original seed, not selected by score. Its initial/8192/16384/24576 returns reproduced exactly. C seeds 0/1 and all completed A/B outputs were retained.
The first resumed telemetry is `gpu_telemetry_resume1.csv`; final telemetry is `gpu_telemetry.csv`.

Manifest clarification: the final resume also rewrote Pendulum manifest_B.json and manifest_C.json. B and C seeds 0/1 had already completed before the per-leaf synchronization change; C seed 2 used the updated wait. These manifests record the resumed invocation, not a claim that all retained seeds used the updated runner. Numerical algorithm, buffer, expectile and hyperparameters remained unchanged.

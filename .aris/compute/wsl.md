# WSL GPU environment specification

Ubuntu WSL2; RTX 3070 Ti Laptop, 8 GiB. Python 3.11.
Environment: /root/.venvs/eql-gpu (separate from Windows .venv).
Direct dependencies: requirements-gpu.txt + requirements-local.txt.
Source checkout: /mnt/c/Users/28449/Desktop/EQL/code.
GPU libraries installed in isolated venv through JAX cuda12 extra. Do not update Windows driver.
XLA_PYTHON_CLIENT_PREALLOCATE=false; one experiment at a time.

Verification from Windows PowerShell:
```
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/28449/Desktop/EQL/code && XLA_PYTHON_CLIENT_PREALLOCATE=false /root/.venvs/eql-gpu/bin/python -m tools.gpu_witness'
```

Status: compiled GPU witness passed on 2026-11-18; actual array device CudaDevice(id=0).
Fresh-agent execution of the documented witness also passed. No old JAX environment found in /root, /home, /opt, /usr/local.
Resolved versions are retained in requirements-gpu-lock.txt.
Buffer/integration validation: 16 tests passed with `python -m pytest tests/test_bagging_impl.py -q`.

Run both environments sequentially from Windows PowerShell:
```
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/28449/Desktop/EQL/code && XLA_PYTHON_CLIENT_PREALLOCATE=false /root/.venvs/eql-gpu/bin/python -u -m experiments.suite online'
```
Outputs: results/online_v1; 4 groups x 3 seeds x 2 environments, 65536 transitions/run.
Earlier Windows .venv remains available for CPU debugging; training uses this WSL environment.

# Local experiment environment specification

Windows x86-64, CPython 3.11, isolated `.venv` in this checkout.
Declarative direct dependencies: `requirements-local.txt`; freeze resolved dependencies after install.
Provider: local CPU JAX. RTX 3070 Ti Laptop 8192 MiB is present, but WSL2 Ubuntu fails with HCS_E_SERVICE_NOT_AVAILABLE. Do not label CPU runs GPU runs.
No paid compute, no remote upload, no system virtualisation changes.

Setup from this checkout in PowerShell:
```
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-local.txt
```

Acceptance requires a seeded compiled JAX kernel and the buffer tests, not imports alone.
Experiments run sequentially with bounded buffers and small MLPs.

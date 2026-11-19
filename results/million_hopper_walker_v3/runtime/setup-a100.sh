set -eu
umask 077
root=/gscratch/stf/xyuchern/eql-million-hopper-walker-v3
source ~/eql-gpu-activate
base=/tmp/eql-xyuchern-cb5f7e7-v1
localenv=$base/eql-v3-a100
mkdir -p "$localenv"/{deps,sysroot,mujoco210,source}
rsync -rt "$root/deps/" "$localenv/deps/"
rsync -a "$root/sysroot/" "$localenv/sysroot/"
rsync -a /sw/contrib/cse-src/mujoco/2.1.0/ "$localenv/mujoco210/"
tar -xzf "$root/code-base.tar.gz" -C "$localenv/source"
cp "$root"/src/*.py "$localenv/source/"
cp "$root/src-adaptive/run_adaptive_tasks.py" "$localenv/source/"
sed -e 's/fcntl.flock(/fcntl.lockf(/g' "$localenv/source/run_million_native.py" > "$localenv/source/run_million_native.py.new"
mv "$localenv/source/run_million_native.py.new" "$localenv/source/run_million_native.py"
cat > "$localenv/activate" <<'BASH'
source /tmp/eql-xyuchern-cb5f7e7-v1/venv/bin/activate
_eql_a100_root=/tmp/eql-xyuchern-cb5f7e7-v1/eql-v3-a100
export MUJOCO_PY_MUJOCO_PATH="$_eql_a100_root/mujoco210"
export MUJOCO_PY_FORCE_CPU=1 MUJOCO_GL=egl
export LD_LIBRARY_PATH="$_eql_a100_root/mujoco210/bin:$_eql_a100_root/sysroot/usr/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$_eql_a100_root/deps:${PYTHONPATH:-}"
export PATH="$_eql_a100_root/deps/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export EQL_V3_SOURCE="$_eql_a100_root/source"
BASH
source "$localenv/activate"
cd "$EQL_V3_SOURCE"
python "$root/probe.py"
python - <<'PY'
import jax, numpy as np
from importlib.metadata import version
from run_million_native import COMMON,NativeTask,NativeAdaptiveEQL
gpu=jax.devices()[0];print('device',gpu.device_kind,gpu.platform,flush=True)
assert gpu.platform=='gpu' and 'A100' in gpu.device_kind
algo=NativeAdaptiveEQL.create(env=NativeTask("Hopper-v3"),**COMMON)
s=jax.jit(algo.init_state)(jax.random.PRNGKey(3001))
s=jax.jit(lambda ts:jax.lax.fori_loop(0,230,lambda _,st:algo.train_iteration(st),ts))(s)
assert int(s.critic_ts.step)>0 and int(s.actor_ts.step)>0 and int(s.global_step)==2300
print('verified',int(s.global_step),'env steps',int(s.critic_ts.step),'critic updates',int(s.actor_ts.step),'actor updates')
for package in ('jax','jaxlib','rejax','gymnasium','mujoco-py','flax'):print(package,version(package))
PY
touch "$localenv/VERIFIED"

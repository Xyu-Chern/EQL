source ~/eql-gpu-activate
root=/gscratch/stf/xyuchern/eql-million-hopper-walker-v3
export MUJOCO_PY_MUJOCO_PATH=/sw/contrib/cse-src/mujoco/2.1.0
export MUJOCO_PY_FORCE_CPU=1
mesa=/sw/contrib/cardss-src/fltk_installation/INSTALLATION/mesa/23.0.0
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$root/sysroot/usr/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$root/sysroot/usr/lib64:$mesa/lib64:${LIBRARY_PATH:-}"
export CPATH="$root/sysroot/usr/include:$mesa/include:${CPATH:-}"
export PYTHONPATH="$root/deps:${PYTHONPATH:-}"
export PATH="$root/deps/bin:$PATH"
export JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export LDSHARED="$root/link-extension -shared -pthread"

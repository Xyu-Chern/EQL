import gymnasium as gym
import numpy as np
import mujoco_py
print("engine",mujoco_py.functions.mj_version(),gym.__version__,flush=True)
for name in ("Hopper-v3","Walker2d-v3"):
    e=gym.make(name)
    assert e.spec.id==name and isinstance(e.unwrapped.sim,mujoco_py.MjSim)
    o,_=e.reset(seed=201)
    obs,r,t,tr,_=e.step(np.zeros(e.action_space.shape))
    print(name, e.spec.entry_point,obs.shape,float(r),flush=True)
    e.close()

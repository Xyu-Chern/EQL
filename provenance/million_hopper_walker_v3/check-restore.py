import numpy as np
from run_million_native import NativeTask, native_snapshot, restore_native
for task in ("Hopper-v3","Walker2d-v3"):
    a,b = NativeTask(task),NativeTask(task)
    for i,(x,y) in enumerate(zip(a.instances,b.instances)):
        x.reset(seed=201+i)
        y.reset(seed=701+i)
    rng=np.random.default_rng(77)
    def actions():
        return rng.uniform(-.4,.4,size=(10,*a.action_shape))
    for _ in range(3):
        for x,u in zip(a.instances,actions()):
            x.step(u)
    restore_native(b,native_snapshot(a))
    for _ in range(5):
        for x,y,u in zip(a.instances,b.instances,actions()):
            sx=x.step(u); sy=y.step(u)
            np.testing.assert_allclose(sx[0],sy[0],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(sx[1],sy[1],atol=1e-12,rtol=1e-12)
            assert sx[2:4]==sy[2:4]
    print(task,"checkpoint trajectory matched",flush=True)

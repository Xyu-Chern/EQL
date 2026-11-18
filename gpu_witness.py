import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_default_matmul_precision', 'highest')

devices = jax.devices()
assert any(d.platform == 'gpu' for d in devices), devices
x = jax.random.normal(jax.random.PRNGKey(7), (128,128))
y = jax.jit(lambda z: z@z.T)(x)
y.block_until_ready()
np.testing.assert_allclose(np.asarray(y), np.asarray(x)@np.asarray(x).T, rtol=1e-3, atol=1e-3)
record = dict(jax=jax.__version__, devices=[str(d) for d in devices],
              array_device=str(y.devices()), trace=float(jnp.trace(y)), finite=bool(jnp.isfinite(y).all()))
Path('.aris/compute/gpu_witness.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))

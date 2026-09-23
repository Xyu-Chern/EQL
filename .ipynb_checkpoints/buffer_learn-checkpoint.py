import chex
import jax
import jax.numpy as jnp
import flashbax as fbx

# Setup fake devices - we use this later with `jax.pmap`.
DEVICE_COUNT_MOCK = 2
chex.set_n_cpu_devices(DEVICE_COUNT_MOCK)


# First define hyper-parameters of the buffer.
max_length = 32 # Maximum length of buffer (max number of experiences stored within the state).
min_length = 8 # Minimum number of experiences saved in the buffer state before we can sample.
sample_batch_size = 4 # Batch size of experience data sampled from the buffer.

add_sequences = False # Will we be adding data in sequences to the buffer?
add_batch_size = 6    # Will we be adding data in batches to the buffer? 
                      # It is possible to add data in both sequences and batches. 
                      # If adding data in batches, what is the batch size that is being added each time?

# Instantiate the flat buffer, which is a Dataclass of pure functions.
buffer = fbx.make_flat_buffer(max_length, min_length, sample_batch_size, add_sequences, add_batch_size)

fake_timestep = {"obs": jnp.array([5, 4]), "reward": jnp.array(1.0)} 
state = buffer.init(fake_timestep)
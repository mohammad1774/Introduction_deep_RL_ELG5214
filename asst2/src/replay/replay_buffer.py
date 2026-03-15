from typing import Dict 

import jax 
import jax.numpy as jnp

def init_buffer(capacity: int, obs_dim: int) -> Dict: 
    buffer = {
        "obs": jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        "actions": jnp.zeros((capacity,), dtype=jnp.int32),
        "rewards": jnp.zeros((capacity,), dtype=jnp.float32),
        "next_obs": jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        "dones": jnp.zeros((capacity,), dtype=jnp.bool_),
        "size": 0,
        "ptr": 0,
        "capacity": capacity
    }

    return buffer 

def add_transition(
        buffer: Dict,
        obs: jnp.ndarray,
        action: int,
        reward: float,
        next_obs: jnp.ndarray,
        done: bool
) -> Dict:
    """
    add transition to replay buffer"""

    ptr = buffer["ptr"]

    buffer["obs"] = buffer["obs"].at[ptr].set(obs)
    buffer["actions"] = buffer["actions"].at[ptr].set(action)
    buffer["rewards"] = buffer["rewards"].at[ptr].set(reward)
    buffer["next_obs"] = buffer["next_obs"].at[ptr].set(next_obs)
    buffer["dones"] = buffer["dones"].at[ptr].set(done)

    buffer["ptr"] = (ptr + 1) % buffer["capacity"]
    buffer["size"] = jnp.minimum(buffer["size"] + 1, buffer["capacity"])

    return buffer 

def sample_batch(
        buffer: Dict,
        key: jax.Array,
        batch_size: int
) -> Dict: 
    
    size = buffer["size"]
    indices = jax.random.randint(key, (batch_size,), 0, size)

    batch = {
        "obs": buffer["obs"][indices],
        "actions": buffer["actions"][indices],
        "rewards": buffer["rewards"][indices],
        "next_obs": buffer["next_obs"][indices],
        "dones": buffer["dones"][indices]
    }

    return batch



"""JAX-compatible random number generation utilities for simulation."""

import jax
import jax.numpy as jnp
import jax.random as jr


def create_prng_key(seed: int) -> jax.Array:
    """Create a PRNG key from seed.

    Parameters
    ----------
    seed : int
        Random seed

    Returns
    -------
    jax.Array
        PRNG key for JAX random operations
    """
    return jr.PRNGKey(seed)


def add_gaussian_noise(
    key: jax.Array,
    signal: jnp.ndarray,
    noise_std: float
) -> tuple[jax.Array, jnp.ndarray]:
    """Add Gaussian noise to signal using JAX random.

    Parameters
    ----------
    key : jax.Array
        PRNG key
    signal : jnp.ndarray
        Input signal
    noise_std : float
        Standard deviation of noise

    Returns
    -------
    tuple[jax.Array, jnp.ndarray]
        Updated PRNG key and noisy signal
    """
    key, subkey = jr.split(key)
    noise = jr.normal(subkey, signal.shape) * noise_std
    return key, signal + noise


def generate_random_walk_bias(
    key: jax.Array,
    n_samples: int,
    bias_drift_std: float,
    dt: float
) -> tuple[jax.Array, jnp.ndarray]:
    """Generate random walk bias using JAX random.

    Parameters
    ----------
    key : jax.Array
        PRNG key
    n_samples : int
        Number of samples
    bias_drift_std : float
        Standard deviation of bias drift per √second
    dt : float
        Time step in seconds

    Returns
    -------
    tuple[jax.Array, jnp.ndarray]
        Updated PRNG key and bias samples
    """
    key, subkey = jr.split(key)

    # Random walk noise: std scales with √dt
    walk_noise_std = bias_drift_std * jnp.sqrt(dt)
    walk_steps = jr.normal(subkey, (n_samples,)) * walk_noise_std

    # Cumulative sum to create random walk
    bias = jnp.cumsum(walk_steps)

    return key, bias


def interpolate_trajectory_jax(
    timestamps_in: jnp.ndarray,
    values_in: jnp.ndarray,
    timestamps_out: jnp.ndarray
) -> jnp.ndarray:
    """JAX-compatible linear interpolation.

    Parameters
    ----------
    timestamps_in : jnp.ndarray
        Input timestamps
    values_in : jnp.ndarray
        Input values to interpolate
    timestamps_out : jnp.ndarray
        Output timestamps

    Returns
    -------
    jnp.ndarray
        Interpolated values
    """
    return jnp.interp(timestamps_out, timestamps_in, values_in)


@jax.jit
def rotate_acceleration_jax(
    accel_body: jnp.ndarray,
    heading: jnp.ndarray
) -> jnp.ndarray:
    """Rotate acceleration from body frame to world frame using JAX.

    Parameters
    ----------
    accel_body : jnp.ndarray
        Acceleration in body frame [N, 2]
    heading : jnp.ndarray
        Heading angles [N]

    Returns
    -------
    jnp.ndarray
        Acceleration in world frame [N, 2]
    """
    cos_h = jnp.cos(heading)
    sin_h = jnp.sin(heading)

    # Rotation matrix for each timestamp
    R_xx = cos_h
    R_xy = -sin_h
    R_yx = sin_h
    R_yy = cos_h

    # Apply rotation
    accel_world_x = R_xx * accel_body[:, 0] + R_xy * accel_body[:, 1]
    accel_world_y = R_yx * accel_body[:, 0] + R_yy * accel_body[:, 1]

    return jnp.column_stack([accel_world_x, accel_world_y])
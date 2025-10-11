"""Shared utilities for filters (EKF, UKF) and smoothers.

This module contains host-side helper functions used by both filters and smoothers
to avoid code duplication.
"""

import jax.numpy as jnp
import numpy as np


def compute_imu_index_arrays(t_imu: np.ndarray, t_cam: np.ndarray) -> jnp.ndarray:
    """Build padded index arrays for IMU samples between camera frames.

    Parameters
    ----------
    t_imu : np.ndarray
        IMU timestamps (N_imu,) in seconds.
    t_cam : np.ndarray
        Camera timestamps (N_cam,) in seconds.

    Returns
    -------
    jnp.ndarray
        Index array (N_cam, max_imu_per_frame) of IMU indices; -1 indicates padding
        (no IMU sample). Returned as a JAX array for device use.

    Notes
    -----
    Host-side precomputation using NumPy avoids dynamic loop unrolling inside JIT.
    For each frame i, finds IMU indices in the half-open interval (t_cam[i-1], t_cam[i]].
    """
    n_cam = len(t_cam)
    all_indices = []

    # First pass: collect all valid index arrays to find max length
    for i in range(n_cam):
        if i == 0:
            # First frame: no IMU propagation
            valid_indices = np.array([], dtype=np.int32)
        else:
            # Find IMU samples in (t_prev, t_current]
            mask = (t_imu > t_cam[i - 1]) & (t_imu <= t_cam[i])
            valid_indices = np.nonzero(mask)[0]

        all_indices.append(valid_indices)

    # Compute max length from actual data
    max_imu_per_frame = max(len(idx) for idx in all_indices)

    # Second pass: pad all arrays to max length
    padded_indices = []
    for valid_indices in all_indices:
        indices = np.full(max_imu_per_frame, -1, dtype=np.int32)
        if len(valid_indices) > 0:
            indices[: len(valid_indices)] = valid_indices
        padded_indices.append(indices)

    # Convert to JAX array for device use
    return jnp.array(padded_indices, dtype=jnp.int32)

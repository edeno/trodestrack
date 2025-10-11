"""Shared utilities for filters (EKF, UKF) and smoothers.

This module contains host-side helper functions used by both filters and smoothers
to avoid code duplication.
"""

import jax.numpy as jnp
import numpy as np


def compute_imu_index_arrays(t_imu: np.ndarray, t_cam: np.ndarray) -> jnp.ndarray:
    """Build padded index arrays for IMU samples between camera frames.

    IMPORTANT: This is a HOST-SIDE precomputation, NOT JIT-traced.
    The function uses NumPy for the host-side loop to avoid JAX tracing surprises,
    then converts the final result to JAX for device use.

    This approach:
    - Avoids dynamic loop unrolling inside JIT (which would lock in n_cam)
    - Precomputes index arrays once rather than recomputing per filter call
    - Uses NumPy during construction to avoid device churn
    - Returns JAX array for seamless integration with JIT-compiled code

    Algorithm:
        For each camera frame i:
        - If i == 0: no IMU propagation (return all -1 padding)
        - Else: find IMU indices where t_imu in (t_cam[i-1], t_cam[i]]
        - Pad to max_imu_per_frame with -1 for invalid indices

    Args:
        t_imu: IMU timestamps (N_imu,) as NumPy array
        t_cam: Camera timestamps (N_cam,) as NumPy array

    Returns:
        jnp.ndarray: (N_cam, max_imu_per_frame) array of IMU indices
            where -1 indicates padding (no IMU sample)

    Example:
        >>> t_imu = np.array([0.0, 0.005, 0.010, 0.015, 0.020])
        >>> t_cam = np.array([0.0, 0.010, 0.020])
        >>> indices = compute_imu_index_arrays(t_imu, t_cam)
        >>> # Frame 0: [-1, -1] (no propagation)
        >>> # Frame 1: [0, 1] (IMU at 0.005, 0.010)
        >>> # Frame 2: [2, 3] (IMU at 0.015, 0.020)
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

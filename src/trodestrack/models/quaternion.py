"""Quaternion utilities for 6-DOF IMU orientation models.

Conventions
-----------
Quaternions use scalar-first order ``[qw, qx, qy, qz]`` and represent the
active rotation from body/sensor coordinates into world coordinates. Gyroscope
measurements are body-frame angular rates ``[ωx, ωy, ωz]`` in rad/s.
"""

from __future__ import annotations

import jax.numpy as jnp


def normalize_quaternion(
    quat: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Return a unit quaternion with scalar-first order.

    Parameters
    ----------
    quat : jnp.ndarray
        Quaternion array with shape ``(..., 4)`` in ``[qw, qx, qy, qz]`` order.
    eps : float, optional
        Minimum norm used to avoid division by zero.

    Returns
    -------
    jnp.ndarray
        Unit quaternion with shape ``(..., 4)``.
    """

    quat_arr = jnp.asarray(quat)
    quat_arr = quat_arr.astype(jnp.result_type(quat_arr, jnp.float32))
    norm = jnp.linalg.norm(quat_arr, axis=-1, keepdims=True)
    identity = jnp.zeros_like(quat_arr)
    identity = identity.at[..., 0].set(1.0)
    normalized = quat_arr / jnp.maximum(norm, jnp.asarray(eps, dtype=quat_arr.dtype))
    return jnp.where(norm > eps, normalized, identity)


def quaternion_conjugate(quat: jnp.ndarray) -> jnp.ndarray:
    """Return the quaternion conjugate.

    Parameters
    ----------
    quat : jnp.ndarray
        Quaternion array with shape ``(..., 4)``.

    Returns
    -------
    jnp.ndarray
        Conjugate quaternion with vector part sign-flipped.
    """

    quat_arr = jnp.asarray(quat)
    signs = jnp.array([1.0, -1.0, -1.0, -1.0], dtype=quat_arr.dtype)
    return quat_arr * signs


def quaternion_inverse(
    quat: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Return the multiplicative inverse of a quaternion.

    Parameters
    ----------
    quat : jnp.ndarray
        Quaternion array with shape ``(..., 4)``.
    eps : float, optional
        Minimum squared norm used to avoid division by zero.

    Returns
    -------
    jnp.ndarray
        Inverse quaternion. Degenerate zero-norm inputs return identity.
    """

    quat_arr = jnp.asarray(quat)
    quat_arr = quat_arr.astype(jnp.result_type(quat_arr, jnp.float32))
    norm_sq = jnp.sum(quat_arr * quat_arr, axis=-1, keepdims=True)
    identity = jnp.zeros_like(quat_arr)
    identity = identity.at[..., 0].set(1.0)
    inverse = quaternion_conjugate(quat_arr) / jnp.maximum(
        norm_sq,
        jnp.asarray(eps, dtype=quat_arr.dtype),
    )
    return jnp.where(norm_sq > eps, inverse, identity)


def quaternion_multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product of two scalar-first quaternions.

    Parameters
    ----------
    left : jnp.ndarray
        Left quaternion with shape ``(..., 4)``.
    right : jnp.ndarray
        Right quaternion with shape ``(..., 4)``.

    Returns
    -------
    jnp.ndarray
        Product ``left ⊗ right`` with shape ``(..., 4)``.
    """

    w1, x1, y1, z1 = jnp.moveaxis(jnp.asarray(left), -1, 0)
    w2, x2, y2, z2 = jnp.moveaxis(jnp.asarray(right), -1, 0)

    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quaternion_from_rotation_vector(
    rotation_vector: jnp.ndarray,
    *,
    small_angle_threshold: float = 1e-8,
) -> jnp.ndarray:
    """Convert a rotation vector to a unit quaternion.

    Parameters
    ----------
    rotation_vector : jnp.ndarray
        Axis-angle vector with shape ``(..., 3)``. Direction is the rotation
        axis and norm is the rotation angle in radians.
    small_angle_threshold : float, optional
        Threshold below which a Taylor approximation is used.

    Returns
    -------
    jnp.ndarray
        Unit quaternion with shape ``(..., 4)``.
    """

    rotvec = jnp.asarray(rotation_vector)
    angle = jnp.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    angle_safe = jnp.maximum(angle, jnp.asarray(small_angle_threshold, rotvec.dtype))

    scale_direct = jnp.sin(half_angle) / angle_safe
    scale_taylor = 0.5 - (angle * angle) / 48.0
    scale = jnp.where(angle > small_angle_threshold, scale_direct, scale_taylor)

    quat = jnp.concatenate([jnp.cos(half_angle), rotvec * scale], axis=-1)
    return normalize_quaternion(quat)


def integrate_body_gyro(
    quat_body_to_world: jnp.ndarray,
    omega_body: jnp.ndarray,
    dt: float | jnp.ndarray,
) -> jnp.ndarray:
    """Propagate body-to-world orientation with body-frame gyro readings.

    Parameters
    ----------
    quat_body_to_world : jnp.ndarray
        Current orientation quaternion, shape ``(..., 4)``.
    omega_body : jnp.ndarray
        Body-frame angular velocity, shape ``(..., 3)`` in rad/s.
    dt : float or jnp.ndarray
        Time step in seconds. May broadcast across leading dimensions.

    Returns
    -------
    jnp.ndarray
        Propagated unit quaternion, shape ``(..., 4)``.

    Notes
    -----
    For a body-to-world quaternion and body-frame angular velocity,
    ``q_next = q_current ⊗ exp(ω_body * dt)``.
    """

    q = normalize_quaternion(quat_body_to_world)
    rotvec = jnp.asarray(omega_body) * jnp.asarray(dt)[..., None]
    delta_q = quaternion_from_rotation_vector(rotvec)
    return normalize_quaternion(quaternion_multiply(q, delta_q))


def rotate_vector_body_to_world(
    quat_body_to_world: jnp.ndarray,
    vector_body: jnp.ndarray,
) -> jnp.ndarray:
    """Rotate body-frame vector(s) into world coordinates.

    Parameters
    ----------
    quat_body_to_world : jnp.ndarray
        Body-to-world unit quaternion, shape ``(..., 4)``.
    vector_body : jnp.ndarray
        Body-frame vector, shape ``(..., 3)``.

    Returns
    -------
    jnp.ndarray
        World-frame vector, shape ``(..., 3)``.
    """

    q = normalize_quaternion(quat_body_to_world)
    vector = jnp.asarray(vector_body)
    zeros = jnp.zeros(vector.shape[:-1] + (1,), dtype=vector.dtype)
    vector_quat = jnp.concatenate([zeros, vector], axis=-1)
    rotated = quaternion_multiply(
        quaternion_multiply(q, vector_quat),
        quaternion_conjugate(q),
    )
    return rotated[..., 1:]


def rotate_vector_world_to_body(
    quat_body_to_world: jnp.ndarray,
    vector_world: jnp.ndarray,
) -> jnp.ndarray:
    """Rotate world-frame vector(s) into body coordinates.

    Parameters
    ----------
    quat_body_to_world : jnp.ndarray
        Body-to-world unit quaternion, shape ``(..., 4)``.
    vector_world : jnp.ndarray
        World-frame vector, shape ``(..., 3)``.

    Returns
    -------
    jnp.ndarray
        Body-frame vector, shape ``(..., 3)``.
    """

    return rotate_vector_body_to_world(
        quaternion_conjugate(normalize_quaternion(quat_body_to_world)),
        vector_world,
    )


def quaternion_to_yaw(quat_body_to_world: jnp.ndarray) -> jnp.ndarray:
    """Extract ZYX yaw from a body-to-world quaternion.

    Parameters
    ----------
    quat_body_to_world : jnp.ndarray
        Quaternion array with shape ``(..., 4)``.

    Returns
    -------
    jnp.ndarray
        Yaw angle in radians with shape ``(...)``.
    """

    q = normalize_quaternion(quat_body_to_world)
    qw, qx, qy, qz = jnp.moveaxis(q, -1, 0)
    return jnp.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def quaternion_to_roll_pitch_yaw(
    quat_body_to_world: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract ZYX roll, pitch, and yaw from a body-to-world quaternion.

    Parameters
    ----------
    quat_body_to_world : jnp.ndarray
        Quaternion array with shape ``(..., 4)``.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(roll, pitch, yaw)`` in radians, each with shape ``(...)``.
    """

    q = normalize_quaternion(quat_body_to_world)
    qw, qx, qy, qz = jnp.moveaxis(q, -1, 0)

    roll = jnp.arctan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = jnp.arcsin(jnp.clip(sin_pitch, -1.0, 1.0))
    yaw = quaternion_to_yaw(q)
    return roll, pitch, yaw

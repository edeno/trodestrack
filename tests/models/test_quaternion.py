"""Tests for quaternion orientation utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.quaternion import (
    integrate_body_gyro,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_from_rotation_vector,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_roll_pitch_yaw,
    quaternion_to_yaw,
    rotate_vector_body_to_world,
    rotate_vector_world_to_body,
)


def test_normalize_quaternion_returns_unit_norm() -> None:
    quat = jnp.array([2.0, 0.0, 0.0, 0.0])

    normalized = normalize_quaternion(quat)

    np.testing.assert_allclose(normalized, [1.0, 0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(jnp.linalg.norm(normalized), 1.0, atol=1e-7)


def test_normalize_quaternion_zero_input_returns_identity() -> None:
    quat = jnp.zeros(4)

    normalized = normalize_quaternion(quat)

    np.testing.assert_allclose(normalized, [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_normalize_quaternion_integer_input_promotes_to_float() -> None:
    quat = jnp.array([1, 0, 0, 0])

    normalized = normalize_quaternion(quat)

    assert jnp.issubdtype(normalized.dtype, jnp.floating)
    np.testing.assert_allclose(normalized, [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_quaternion_multiply_by_conjugate_gives_identity() -> None:
    quat = quaternion_from_rotation_vector(jnp.array([0.2, -0.3, 0.4]))

    identity = quaternion_multiply(quat, quaternion_conjugate(quat))

    np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_rotation_vector_integer_input_promotes_to_float() -> None:
    quat = quaternion_from_rotation_vector(jnp.array([0, 0, 0]))

    assert jnp.issubdtype(quat.dtype, jnp.floating)
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_quaternion_inverse_handles_non_unit_quaternion() -> None:
    quat = jnp.array([2.0, 0.2, -0.4, 0.6])

    identity = quaternion_multiply(quat, quaternion_inverse(quat))

    np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_quaternion_inverse_zero_input_returns_identity() -> None:
    inverse = quaternion_inverse(jnp.zeros(4))

    np.testing.assert_allclose(inverse, [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_rotate_vector_body_to_world_known_yaw_rotation() -> None:
    quat = quaternion_from_rotation_vector(jnp.array([0.0, 0.0, np.pi / 2.0]))
    vector_body = jnp.array([1.0, 0.0, 0.0])

    vector_world = rotate_vector_body_to_world(quat, vector_body)

    np.testing.assert_allclose(vector_world, [0.0, 1.0, 0.0], atol=1e-6)


def test_world_to_body_rotation_inverts_body_to_world_rotation() -> None:
    quat = quaternion_from_rotation_vector(jnp.array([0.1, -0.2, 0.3]))
    vector_body = jnp.array([0.4, -0.5, 0.6])

    vector_world = rotate_vector_body_to_world(quat, vector_body)
    recovered_body = rotate_vector_world_to_body(quat, vector_world)

    np.testing.assert_allclose(recovered_body, vector_body, atol=1e-6)


def test_rotation_preserves_vector_norm() -> None:
    quat = quaternion_from_rotation_vector(jnp.array([0.7, -0.1, 0.2]))
    vector = jnp.array([1.2, -3.4, 5.6])

    rotated = rotate_vector_body_to_world(quat, vector)

    np.testing.assert_allclose(
        jnp.linalg.norm(rotated),
        jnp.linalg.norm(vector),
        rtol=1e-6,
        atol=1e-6,
    )


def test_integrate_body_gyro_constant_yaw_matches_expected_angle() -> None:
    quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    omega_body = jnp.array([0.0, 0.0, np.pi])

    next_quat = integrate_body_gyro(quat, omega_body, dt=0.5)

    np.testing.assert_allclose(quaternion_to_yaw(next_quat), np.pi / 2.0, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(next_quat), 1.0, atol=1e-6)


def test_integrate_body_gyro_repeated_steps_keep_unit_norm() -> None:
    quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    omega_body = jnp.array([0.2, -0.3, 0.4])

    for _ in range(1_000):
        quat = integrate_body_gyro(quat, omega_body, dt=0.001)

    np.testing.assert_allclose(jnp.linalg.norm(quat), 1.0, atol=1e-6)


def test_quaternion_to_roll_pitch_yaw_matches_single_axis_rotation() -> None:
    roll_quat = quaternion_from_rotation_vector(jnp.array([0.3, 0.0, 0.0]))
    pitch_quat = quaternion_from_rotation_vector(jnp.array([0.0, -0.2, 0.0]))
    yaw_quat = quaternion_from_rotation_vector(jnp.array([0.0, 0.0, 0.4]))

    roll, pitch, yaw = quaternion_to_roll_pitch_yaw(roll_quat)
    np.testing.assert_allclose([roll, pitch, yaw], [0.3, 0.0, 0.0], atol=1e-6)

    roll, pitch, yaw = quaternion_to_roll_pitch_yaw(pitch_quat)
    np.testing.assert_allclose([roll, pitch, yaw], [0.0, -0.2, 0.0], atol=1e-6)

    roll, pitch, yaw = quaternion_to_roll_pitch_yaw(yaw_quat)
    np.testing.assert_allclose([roll, pitch, yaw], [0.0, 0.0, 0.4], atol=1e-6)


def test_quaternion_utilities_support_batch_shapes() -> None:
    rotation_vectors = jnp.array(
        [
            [0.0, 0.0, np.pi / 2.0],
            [0.0, 0.0, -np.pi / 2.0],
        ]
    )
    vectors_body = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    quats = quaternion_from_rotation_vector(rotation_vectors)
    vectors_world = rotate_vector_body_to_world(quats, vectors_body)

    assert quats.shape == (2, 4)
    assert vectors_world.shape == (2, 3)
    np.testing.assert_allclose(
        vectors_world,
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        atol=1e-6,
    )


@pytest.mark.parametrize("rotation_angle", [0.0, 1e-10, 1e-9])
def test_rotation_vector_handles_zero_and_tiny_angles(rotation_angle: float) -> None:
    quat = quaternion_from_rotation_vector(jnp.array([rotation_angle, 0.0, 0.0]))

    assert jnp.isfinite(quat).all()
    np.testing.assert_allclose(jnp.linalg.norm(quat), 1.0, atol=1e-7)

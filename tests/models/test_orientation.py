"""Tests for standalone IMU orientation estimation."""

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.orientation import (
    OrientationEstimatorConfig,
    estimate_orientation,
)
from trodestrack.models.quaternion import (
    quaternion_from_rotation_vector,
    quaternion_multiply,
    rotate_vector_world_to_body,
)

GRAVITY = 9.80665


def test_stationary_tilt_converges_to_gravity_direction() -> None:
    t_imu = np.linspace(0.0, 8.0, 801)
    true_quat = quaternion_multiply(
        quaternion_from_rotation_vector(jnp.array([0.0, -0.25, 0.0])),
        quaternion_from_rotation_vector(jnp.array([0.35, 0.0, 0.0])),
    )
    gravity_body = np.asarray(
        rotate_vector_world_to_body(true_quat, jnp.array([0.0, 0.0, GRAVITY]))
    )
    gyro = np.zeros((t_imu.size, 3))
    accel = np.tile(gravity_body, (t_imu.size, 1))

    result = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        config=OrientationEstimatorConfig(
            initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            initial_gyro_bias_rad_s=np.zeros(3),
        ),
    )

    estimated_gravity_body = np.asarray(
        rotate_vector_world_to_body(
            jnp.asarray(result.quaternions[-1]),
            jnp.array([0.0, 0.0, GRAVITY]),
        )
    )
    np.testing.assert_allclose(estimated_gravity_body, gravity_body, atol=0.05)
    assert result.diagnostics.gravity_update_fraction > 0.99
    np.testing.assert_allclose(
        np.linalg.norm(result.quaternions, axis=1),
        1.0,
        atol=2e-7,
    )


def test_default_initial_orientation_matches_stationary_gravity() -> None:
    t_imu = np.linspace(0.0, 1.0, 101)
    true_quat = quaternion_multiply(
        quaternion_from_rotation_vector(jnp.array([0.0, -0.25, 0.0])),
        quaternion_from_rotation_vector(jnp.array([0.35, 0.0, 0.0])),
    )
    gravity_body = np.asarray(
        rotate_vector_world_to_body(true_quat, jnp.array([0.0, 0.0, GRAVITY]))
    )
    gyro = np.zeros((t_imu.size, 3))
    accel = np.tile(gravity_body, (t_imu.size, 1))

    result = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        config=OrientationEstimatorConfig(
            initial_gyro_bias_rad_s=np.zeros(3),
            accel_correction_gain=0.0,
        ),
    )

    estimated_gravity_body = np.asarray(
        rotate_vector_world_to_body(
            jnp.asarray(result.quaternions[0]),
            jnp.array([0.0, 0.0, GRAVITY]),
        )
    )
    np.testing.assert_allclose(estimated_gravity_body, gravity_body, atol=1e-6)


def test_yaw_rotation_tracks_gyro_without_roll_pitch_drift() -> None:
    yaw_rate = 0.3
    t_imu = np.linspace(0.0, 10.0, 1_001)
    gyro = np.zeros((t_imu.size, 3))
    gyro[:, 2] = yaw_rate
    accel = np.tile([0.0, 0.0, GRAVITY], (t_imu.size, 1))

    result = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        config=OrientationEstimatorConfig(initial_gyro_bias_rad_s=np.zeros(3)),
    )

    expected_yaw = (yaw_rate * t_imu[-1] + np.pi) % (2.0 * np.pi) - np.pi
    np.testing.assert_allclose(result.yaw[-1], expected_yaw, atol=0.01)
    np.testing.assert_allclose(result.roll, 0.0, atol=0.01)
    np.testing.assert_allclose(result.pitch, 0.0, atol=0.01)
    assert result.diagnostics.quaternion_norm_min > 1.0 - 2e-7
    assert result.diagnostics.quaternion_norm_max < 1.0 + 2e-7


def test_default_initial_gyro_bias_does_not_erase_slow_yaw() -> None:
    yaw_rate = 0.1
    t_imu = np.linspace(0.0, 10.0, 1_001)
    gyro = np.zeros((t_imu.size, 3))
    gyro[:, 2] = yaw_rate
    accel = np.tile([0.0, 0.0, GRAVITY], (t_imu.size, 1))

    result = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
    )

    np.testing.assert_allclose(result.yaw[-1], yaw_rate * t_imu[-1], atol=0.01)
    np.testing.assert_allclose(result.gyro_bias_rad_s[-1], 0.0, atol=1e-12)


def test_high_linear_acceleration_is_gated_out() -> None:
    t_imu = np.linspace(0.0, 4.0, 401)
    gyro = np.zeros((t_imu.size, 3))
    accel = np.tile([0.0, 0.0, GRAVITY], (t_imu.size, 1))
    accel[100:180, 0] = 4.0

    result = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        config=OrientationEstimatorConfig(initial_gyro_bias_rad_s=np.zeros(3)),
    )

    assert not result.gravity_update_mask[100:180].any()
    assert result.gravity_update_mask[:90].all()
    assert result.gravity_update_mask[190:].all()


def test_camera_yaw_correction_reduces_gyro_bias_drift() -> None:
    true_yaw_rate = 0.25
    gyro_bias_z = 0.04
    t_imu = np.linspace(0.0, 12.0, 1_201)
    t_cam = np.linspace(0.0, 12.0, 361)
    true_heading_cam = true_yaw_rate * t_cam
    led1 = np.column_stack(
        [-0.02 * np.cos(true_heading_cam), -0.02 * np.sin(true_heading_cam)]
    )
    led2 = -led1

    gyro = np.zeros((t_imu.size, 3))
    gyro[:, 2] = true_yaw_rate + gyro_bias_z
    accel = np.tile([0.0, 0.0, GRAVITY], (t_imu.size, 1))

    no_camera = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        config=OrientationEstimatorConfig(initial_gyro_bias_rad_s=np.zeros(3)),
    )
    with_camera = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        t_cam=t_cam,
        led1=led1,
        led2=led2,
        config=OrientationEstimatorConfig(initial_gyro_bias_rad_s=np.zeros(3)),
    )

    true_yaw_final = (true_yaw_rate * t_imu[-1] + np.pi) % (2.0 * np.pi) - np.pi
    no_camera_error = abs(
        (no_camera.yaw[-1] - true_yaw_final + np.pi) % (2 * np.pi) - np.pi
    )
    camera_error = abs(
        (with_camera.yaw[-1] - true_yaw_final + np.pi) % (2 * np.pi) - np.pi
    )

    assert camera_error < 0.25 * no_camera_error
    assert 0.25 < with_camera.diagnostics.camera_yaw_update_fraction < 0.35
    assert with_camera.diagnostics.yaw_camera_rmse_rad is not None


def test_camera_speed_gate_supports_two_camera_samples() -> None:
    t_imu = np.linspace(0.0, 1.0, 101)
    t_cam = np.array([0.0, 1.0])
    gyro = np.zeros((t_imu.size, 3))
    accel = np.tile([0.0, 0.0, GRAVITY], (t_imu.size, 1))
    led1 = np.array([[0.0, 0.0], [0.01, 0.0]])
    led2 = led1 + np.array([0.04, 0.0])

    result = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        t_cam=t_cam,
        led1=led1,
        led2=led2,
        config=OrientationEstimatorConfig(
            initial_gyro_bias_rad_s=np.zeros(3),
            camera_speed_threshold_m_s=0.05,
        ),
    )

    assert np.isfinite(result.quaternions).all()


def _stationary_orientation_inputs(
    pitch_deg: float, duration_s: float = 5.0, fs_imu: float = 100.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stationary IMU inputs at a given true pitch.

    Returns ``(t_imu, gyro, accel, true_quat)`` such that
    ``estimate_orientation`` should recover a body-to-world quaternion close
    to ``true_quat``. Roll and yaw are held at zero.
    """

    n = round(duration_s * fs_imu) + 1
    t_imu = np.linspace(0.0, duration_s, n)
    pitch_rad = np.deg2rad(pitch_deg)
    true_quat = quaternion_from_rotation_vector(jnp.array([0.0, pitch_rad, 0.0]))
    gravity_body = np.asarray(
        rotate_vector_world_to_body(true_quat, jnp.array([0.0, 0.0, GRAVITY]))
    )
    gyro = np.zeros((n, 3))
    accel = np.tile(gravity_body, (n, 1))
    return t_imu, gyro, accel, np.asarray(true_quat)


@pytest.mark.parametrize("pitch_deg", [88.0, 92.0])
def test_orientation_near_pitch_singularity_keeps_quaternion_unit_norm(
    pitch_deg: float,
) -> None:
    """Stationary headstage near the ±90° gimbal-lock region.

    The Euler-angle pitch singularity sits at ±90°. The internal quaternion
    representation must stay unit-norm regardless of how poorly Euler-angle
    yaw is conditioned there, so we assert ``||q|| == 1 ± 2e-7`` on every
    timestep and check that yaw is finite (no division-by-zero NaN).
    """

    t_imu, gyro, accel, true_quat = _stationary_orientation_inputs(pitch_deg)

    result = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        config=OrientationEstimatorConfig(
            initial_quaternion=true_quat,
            initial_gyro_bias_rad_s=np.zeros(3),
        ),
    )

    norms = np.linalg.norm(result.quaternions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=2e-7)
    assert np.isfinite(result.yaw).all()

    pitch_recovered_deg = np.rad2deg(result.pitch[-1])
    # Pitch is measured as arcsin and wraps within [-90, 90]; for an input
    # of 92° the recovered value sits near +88° (the symmetric image of the
    # singularity). Compare via sin to avoid that branch ambiguity.
    np.testing.assert_allclose(
        np.sin(result.pitch[-1]),
        np.sin(np.deg2rad(pitch_deg)),
        atol=np.sin(np.deg2rad(3.0)),
    )
    # Sanity-check that we didn't accidentally land on the opposite hemisphere.
    assert abs(pitch_recovered_deg) > 80.0


def test_orientation_through_pitch_singularity_does_not_nan() -> None:
    """Pitch sweep -89° → +89° (10 s, 100 Hz) must not produce NaN outputs.

    Crossing the ±90° region with a finite pitch rate is the realistic
    failure mode for an inverted headstage. None of the public outputs
    (quaternion, roll, pitch, yaw) may NaN at any timestep.
    """

    fs_imu = 100.0
    duration_s = 10.0
    n = round(duration_s * fs_imu) + 1
    t_imu = np.linspace(0.0, duration_s, n)
    pitch_start = np.deg2rad(-89.0)
    pitch_end = np.deg2rad(89.0)
    pitch_rate = (pitch_end - pitch_start) / duration_s

    gyro = np.zeros((n, 3))
    gyro[:, 1] = pitch_rate  # constant body-y rotation
    # Accelerometer measures gravity rotated into the body frame at the
    # current pitch; build the truth quaternion at each sample.
    pitch_t = pitch_start + pitch_rate * t_imu
    accel = np.zeros((n, 3))
    for i, pitch_i in enumerate(pitch_t):
        q_i = quaternion_from_rotation_vector(jnp.array([0.0, float(pitch_i), 0.0]))
        accel[i] = np.asarray(
            rotate_vector_world_to_body(q_i, jnp.array([0.0, 0.0, GRAVITY]))
        )

    initial_quat = quaternion_from_rotation_vector(
        jnp.array([0.0, float(pitch_start), 0.0])
    )
    result = estimate_orientation(
        t_imu=t_imu,
        gyro_xyz=gyro,
        accel_xyz=accel,
        config=OrientationEstimatorConfig(
            initial_quaternion=np.asarray(initial_quat),
            initial_gyro_bias_rad_s=np.zeros(3),
        ),
    )

    assert np.isfinite(result.quaternions).all()
    assert np.isfinite(result.roll).all()
    assert np.isfinite(result.pitch).all()
    assert np.isfinite(result.yaw).all()

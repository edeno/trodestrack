from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter, predict_step
from trodestrack.models.filter_common import (
    FilterState,
    dynamics_function,
    initialize_state,
    measurement_function,
)
from trodestrack.models.quaternion import (
    quaternion_from_rotation_vector,
    quaternion_to_roll_pitch_yaw,
    quaternion_to_yaw,
)
from trodestrack.models.state_layout import get_layout
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter


def test_initialize_state_vision_only_layout() -> None:
    layout = get_layout("vision_only")

    # Two frames, one valid, one invalid
    led1 = jnp.array([[0.1, 0.2], [jnp.nan, jnp.nan]], dtype=jnp.float32)
    led2 = jnp.array([[0.14, 0.2], [jnp.nan, jnp.nan]], dtype=jnp.float32)
    mask = jnp.array([True, False])

    state = initialize_state(
        led1, led2, mask, dt_cam=jnp.array(1 / 30.0), led_distance=0.04, layout=layout
    )

    # Expect 5D state for vision_only
    assert state.mean.shape[0] == layout.n == 5

    # Position mapped to layout indices
    assert np.isfinite(np.array(state.mean[layout.pos_idx[0]])).all()
    assert np.isfinite(np.array(state.mean[layout.pos_idx[1]])).all()


def test_measurement_function_respects_layout_indices() -> None:
    layout = get_layout("2d_cam_3d_imu")

    # Build a state with known position and heading at layout indices
    n = layout.n
    m = jnp.zeros(n)
    px_i, py_i = layout.pos_idx[0], layout.pos_idx[1]
    h_i = layout.heading_idx  # type: ignore[assignment]
    m = m.at[px_i].set(1.0)
    m = m.at[py_i].set(2.0)
    m = m.at[h_i].set(0.0)  # cos=1, sin=0

    z = measurement_function(m, 0.04, layout)
    # LED positions should be symmetric around (x,y) along x-axis
    assert np.allclose(np.array(z[0]), 1.0 - 0.02, atol=1e-6)
    assert np.allclose(np.array(z[2]), 1.0 + 0.02, atol=1e-6)
    assert np.allclose(np.array(z[1]), 2.0, atol=1e-6)
    assert np.allclose(np.array(z[3]), 2.0, atol=1e-6)


def test_measurement_function_uses_quaternion_yaw() -> None:
    layout = get_layout("2d_cam_6dof_imu_orientation")
    n = layout.n
    m = jnp.zeros(n)
    m = m.at[layout.pos_idx[0]].set(1.0)
    m = m.at[layout.pos_idx[1]].set(2.0)
    quat_idx = jnp.array(layout.heading_idx)
    yaw = jnp.pi / 2
    quat = jnp.array([jnp.cos(yaw / 2.0), 0.0, 0.0, jnp.sin(yaw / 2.0)])
    m = m.at[quat_idx].set(quat)

    z = measurement_function(m, 0.04, layout)

    np.testing.assert_allclose(np.asarray(z), [1.0, 1.98, 1.0, 2.02], atol=1e-6)


def test_dynamics_6dof_orientation_does_not_translate_from_accel_by_default() -> None:
    layout = get_layout("2d_cam_6dof_imu_orientation")
    state = jnp.zeros(layout.n)
    quat_idx = jnp.array(layout.heading_idx)
    state = state.at[layout.pos_idx[0]].set(1.0)
    state = state.at[layout.pos_idx[1]].set(2.0)
    state = state.at[layout.vel_idx[0]].set(0.3)
    state = state.at[layout.vel_idx[1]].set(-0.1)
    state = state.at[quat_idx].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    imu = jnp.array([0.0, 0.0, 0.2, 50.0, -40.0, 20.0])

    next_state = dynamics_function(
        state,
        imu,
        dt=0.5,
        damping=0.0,
        layout=layout,
        enable_experimental_accel_translation=False,
    )

    np.testing.assert_allclose(
        np.asarray(next_state[jnp.array(layout.pos_idx)]),
        [1.15, 1.95],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(next_state[jnp.array(layout.vel_idx)]),
        [0.3, -0.1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(quaternion_to_yaw(next_state[quat_idx])),
        0.1,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(next_state[quat_idx])),
        1.0,
        atol=1e-7,
    )


def test_dynamics_6dof_orientation_requires_explicit_accel_translation() -> None:
    layout = get_layout("2d_cam_6dof_imu_orientation")
    state = jnp.zeros(layout.n)
    quat_idx = jnp.array(layout.heading_idx)
    state = state.at[quat_idx].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    imu = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 9.81])

    no_translation = dynamics_function(
        state,
        imu,
        dt=0.1,
        damping=0.0,
        layout=layout,
        enable_experimental_accel_translation=False,
    )
    with_translation = dynamics_function(
        state,
        imu,
        dt=0.1,
        damping=0.0,
        layout=layout,
        enable_experimental_accel_translation=True,
    )

    np.testing.assert_allclose(
        np.asarray(no_translation[jnp.array(layout.vel_idx)]),
        [0.0, 0.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(with_translation[jnp.array(layout.vel_idx)]),
        [0.1, 0.0],
        atol=1e-6,
    )


def test_ekf_gravity_update_corrects_quaternion_roll_pitch() -> None:
    layout = get_layout("2d_cam_6dof_imu_orientation")
    quat_idx = jnp.array(layout.heading_idx)
    initial_quat = quaternion_from_rotation_vector(jnp.array([0.35, -0.25, 0.0]))
    mean = jnp.zeros(layout.n)
    mean = mean.at[quat_idx].set(initial_quat)
    state = FilterState(mean=mean, cov=jnp.eye(layout.n) * 0.5)
    imu = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.81])

    next_state = predict_step(
        state,
        imu,
        dt_imu=0.01,
        config=EKFConfig(
            state_mode="2d_cam_6dof_imu_orientation",
            use_gravity_orientation_update=True,
            gravity_orientation_measurement_noise=1e-4,
            enable_zupt=False,
        ),
        layout=layout,
    )

    initial_roll, initial_pitch, _ = quaternion_to_roll_pitch_yaw(initial_quat)
    next_roll, next_pitch, _ = quaternion_to_roll_pitch_yaw(next_state.mean[quat_idx])
    assert abs(float(next_roll)) < abs(float(initial_roll))
    assert abs(float(next_pitch)) < abs(float(initial_pitch))


def test_ekf_smoke_vision_only_shapes() -> None:
    ekf_config = EKFConfig(
        state_mode="vision_only", led_distance=0.04, use_heading_measurement=False
    )

    # Small 3-frame scenario
    t_cam = np.array([0.0, 0.0333, 0.0666], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.0666, 15, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 3), dtype=np.float32)

    # Simple forward motion in camera
    Z1 = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    mask = np.array([True, True, True])

    filter_result = extended_kalman_filter(
        ekf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )

    # Shapes reflect 5D state
    assert filter_result.filtered_means.shape == (3, get_layout("vision_only").n)
    assert filter_result.predicted_means.shape == (3, get_layout("vision_only").n)


def test_ekf_vision_only_ignores_imu_during_dropout() -> None:
    ekf_config = EKFConfig(
        state_mode="vision_only",
        led_distance=0.04,
        use_heading_measurement=False,
        enable_zupt=False,
        use_mahalanobis_gating=False,
    )

    t_cam = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    t_imu = np.array([0.0, 0.05, 0.1, 0.15, 0.2], dtype=np.float32)
    U_imu = np.tile(np.array([[1.0, 2.0, -3.0]], dtype=np.float32), (5, 1))

    Z1 = np.array([[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    Z2[1:] = np.nan
    mask = np.array([True, False, False])

    filter_result = extended_kalman_filter(
        ekf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )
    layout = get_layout("vision_only")

    # Initial velocity is zero because only one valid camera frame is available.
    # Nonzero IMU samples must not move position or heading in vision-only mode.
    np.testing.assert_allclose(
        np.asarray(filter_result.filtered_means[:, layout.pos_idx]),
        np.array([[0.02, 0.0], [0.02, 0.0], [0.02, 0.0]]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(filter_result.filtered_means[:, layout.heading_idx]),
        np.zeros(3),
        atol=1e-6,
    )


def test_ukf_smoke_2d_cam_3d_imu_shapes() -> None:
    ukf_config = UKFConfig(
        state_mode="2d_cam_3d_imu", led_distance=0.04, use_heading_measurement=False
    )

    # Small 3-frame scenario
    t_cam = np.array([0.0, 0.0333, 0.0666], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.0666, 15, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 3), dtype=np.float32)

    # Simple forward motion in camera
    Z1 = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    mask = np.array([True, True, True])

    filter_result = unscented_kalman_filter(
        ukf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )

    # Shapes reflect 10D state
    assert filter_result.filtered_means.shape == (3, get_layout("2d_cam_3d_imu").n)
    assert filter_result.predicted_means.shape == (3, get_layout("2d_cam_3d_imu").n)


def test_ekf_smoke_2d_cam_6dof_imu_orientation_shapes() -> None:
    ekf_config = EKFConfig(
        state_mode="2d_cam_6dof_imu_orientation",
        led_distance=0.04,
        use_heading_measurement=True,
        enable_zupt=False,
        use_mahalanobis_gating=False,
    )

    t_cam = np.array([0.0, 0.0333, 0.0666], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.0666, 15, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_imu[:, 5] = 9.81

    Z1 = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    mask = np.array([True, True, True])

    filter_result = extended_kalman_filter(
        ekf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )
    layout = get_layout("2d_cam_6dof_imu_orientation")
    quat_idx = np.array(layout.heading_idx)

    assert filter_result.filtered_means.shape == (3, layout.n)
    assert filter_result.predicted_means.shape == (3, layout.n)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(filter_result.filtered_means[:, quat_idx]), axis=1),
        1.0,
        atol=1e-5,
    )


def test_ekf_2d_cam_6dof_orientation_stays_finite_during_dropout() -> None:
    ekf_config = EKFConfig(
        state_mode="2d_cam_6dof_imu_orientation",
        led_distance=0.04,
        use_heading_measurement=True,
        enable_zupt=False,
        use_mahalanobis_gating=False,
    )
    t_cam = np.linspace(0.0, 0.4, 5, dtype=np.float32)
    t_imu = np.linspace(0.0, 0.4, 41, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_imu[:, 2] = 0.5
    U_imu[:, 5] = 9.81

    Z1 = np.column_stack([0.02 * t_cam, np.zeros_like(t_cam)]).astype(np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    Z1[2:4] = np.nan
    Z2[2:4] = np.nan
    mask = np.array([True, True, False, False, True])

    filter_result = extended_kalman_filter(
        ekf_config,
        t_imu,
        U_imu,
        t_cam,
        Z1,
        Z2,
        mask,
        initial_state=None,
        conf_cam=None,
    )
    layout = get_layout("2d_cam_6dof_imu_orientation")
    quat_idx = np.array(layout.heading_idx)

    assert np.isfinite(np.asarray(filter_result.filtered_means)).all()
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(filter_result.filtered_means[:, quat_idx]), axis=1),
        1.0,
        atol=1e-5,
    )

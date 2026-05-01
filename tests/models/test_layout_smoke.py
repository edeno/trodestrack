from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from trodestrack.models.ekf import (
    EKFConfig,
    _chi2_threshold_active,
    _gravity_direction_prediction,
    extended_kalman_filter,
    extended_kalman_filter_3d,
    predict_step,
)
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
    rotate_vector_body_to_world,
    rotate_vector_world_to_body,
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


def test_dynamics_3d_cam_6dof_imu_integrates_full_3d_accel() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    state = jnp.zeros(layout.n)
    quat_idx = jnp.array(layout.heading_idx)
    state = state.at[jnp.array(layout.pos_idx)].set(jnp.array([1.0, 2.0, 3.0]))
    state = state.at[jnp.array(layout.vel_idx)].set(jnp.array([0.1, 0.2, -0.1]))
    state = state.at[quat_idx].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    imu = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0, 12.81])

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
        [0.1, 0.2, -0.1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(no_translation[jnp.array(layout.pos_idx)]),
        [1.01, 2.02, 2.99],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(with_translation[jnp.array(layout.vel_idx)]),
        [0.2, 0.4, 0.2],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(with_translation[jnp.array(layout.pos_idx)]),
        [1.015, 2.03, 3.005],
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


def test_ekf_gravity_gate_uses_bias_corrected_gyro_norm() -> None:
    layout = get_layout("2d_cam_6dof_imu_orientation")
    quat_idx = jnp.array(layout.heading_idx)
    initial_quat = quaternion_from_rotation_vector(jnp.array([0.35, -0.25, 0.0]))
    gyro_bias = jnp.array([0.3, 0.0, 0.0])
    mean = jnp.zeros(layout.n)
    mean = mean.at[quat_idx].set(initial_quat)
    mean = mean.at[jnp.array(layout.bias_gyro_idx)].set(gyro_bias)
    state = FilterState(mean=mean, cov=jnp.eye(layout.n) * 0.5)
    imu = jnp.concatenate([gyro_bias, jnp.array([0.0, 0.0, 9.81])])

    next_state = predict_step(
        state,
        imu,
        dt_imu=0.01,
        config=EKFConfig(
            state_mode="2d_cam_6dof_imu_orientation",
            use_gravity_orientation_update=True,
            gravity_orientation_measurement_noise=1e-4,
            gravity_gyro_norm_threshold_rad_s=0.2,
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


def test_ekf_3d_cam_6dof_imu_recovers_synthetic_pose() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    led_offsets = np.array(
        [
            [-0.03, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.0, 0.025, 0.02],
        ],
        dtype=np.float32,
    )
    true_position = jnp.array([1.0, -0.5, 0.25])
    true_quat = quaternion_from_rotation_vector(jnp.array([0.08, -0.06, 0.12]))
    z_leds_one = true_position[None, :] + rotate_vector_body_to_world(
        true_quat, jnp.asarray(led_offsets)
    )
    t_cam = np.linspace(0.0, 0.2, 7, dtype=np.float32)
    t_imu = np.linspace(0.0, 0.2, 41, dtype=np.float32)
    z_leds = np.repeat(np.asarray(z_leds_one)[None, :, :], t_cam.shape[0], axis=0)
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    gravity_body = rotate_vector_world_to_body(
        true_quat,
        jnp.array([0.0, 0.0, 9.81]),
    )
    U_imu[:, 3:6] = np.asarray(gravity_body)

    result = extended_kalman_filter_3d(
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            measurement_noise_pos=1e-8,
            enable_experimental_accel_translation=True,
            enable_zupt=False,
            use_mahalanobis_gating=False,
        ),
        t_imu,
        U_imu,
        t_cam,
        z_leds,
        led_offsets,
    )

    quat_idx = np.array(layout.heading_idx)
    assert result.filtered_means.shape == (t_cam.shape[0], layout.n)
    assert np.isfinite(np.asarray(result.filtered_means)).all()
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(result.filtered_means[:, quat_idx]), axis=1),
        1.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.filtered_means[-1, np.array(layout.pos_idx)]),
        np.asarray(true_position),
        atol=2e-3,
    )
    final_quat = result.filtered_means[-1, quat_idx]
    orientation_error = 2.0 * jnp.arccos(
        jnp.clip(jnp.abs(jnp.dot(final_quat, true_quat)), -1.0, 1.0)
    )
    assert float(orientation_error) < 5e-3


def test_dynamics_3d_quaternion_preserves_calibrated_gravity_direction() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    state = jnp.zeros(layout.n)
    state = state.at[jnp.array(layout.heading_idx)].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    gravity_body = jnp.array([1.0, 0.0, 9.76])
    imu = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 9.76])

    next_state = dynamics_function(
        state,
        imu,
        dt=0.1,
        damping=0.0,
        layout=layout,
        gravity_body=gravity_body,
        enable_experimental_accel_translation=True,
    )

    np.testing.assert_allclose(
        np.asarray(next_state[jnp.array(layout.vel_idx)]),
        0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(next_state[jnp.array(layout.pos_idx)]),
        0.0,
        atol=1e-7,
    )


def test_gravity_direction_update_preserves_calibrated_tilt() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    state = jnp.zeros(layout.n)
    state = state.at[jnp.array(layout.heading_idx)].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    gravity_world = jnp.array([0.5, 0.0, 9.79])

    prediction = _gravity_direction_prediction(
        state,
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            imu_gravity_body=tuple(float(x) for x in np.asarray(gravity_world)),
        ),
        layout=layout,
    )

    expected = gravity_world / jnp.linalg.norm(gravity_world)
    np.testing.assert_allclose(
        np.asarray(prediction),
        np.asarray(expected),
        atol=1e-7,
    )
    assert float(prediction[0]) > 0.04


def test_predict_step_3d_quaternion_couples_bias_covariance() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    mean = jnp.zeros(layout.n)
    mean = mean.at[jnp.array(layout.heading_idx)].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    cov = jnp.eye(layout.n)
    imu = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.81])

    predicted = predict_step(
        FilterState(mean=mean, cov=cov),
        imu,
        0.01,
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            enable_experimental_accel_translation=True,
            use_gravity_orientation_update=False,
        ),
        layout=layout,
    )

    quat_vec_idx = np.array(layout.heading_idx[1:])
    gyro_bias_idx = np.array(layout.bias_gyro_idx)
    vel_idx = np.array(layout.vel_idx)
    accel_bias_idx = np.array(layout.bias_accel_idx)
    assert np.any(
        np.abs(np.asarray(predicted.cov[quat_vec_idx[:, None], gyro_bias_idx])) > 0.0
    )
    assert np.any(
        np.abs(np.asarray(predicted.cov[vel_idx[:, None], accel_bias_idx])) > 0.0
    )


def test_ekf_3d_iekf_iterations_do_not_recount_linear_measurement() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    mean = jnp.zeros(layout.n)
    mean = mean.at[jnp.array(layout.heading_idx)].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    cov = jnp.eye(layout.n)
    t_cam = np.array([0.0], dtype=np.float32)
    t_imu = np.array([0.0, 0.01], dtype=np.float32)
    U_imu = np.zeros((2, 6), dtype=np.float32)
    U_imu[:, 5] = 9.81
    z_leds = np.zeros((1, 3, 3), dtype=np.float32)
    led_offsets = np.zeros((3, 3), dtype=np.float32)

    common_kwargs = dict(
        t_imu=t_imu,
        U_imu=U_imu,
        t_cam=t_cam,
        Z_cam_leds=z_leds,
        led_offsets_body=led_offsets,
        initial_state=FilterState(mean=mean, cov=cov),
    )
    result_one = extended_kalman_filter_3d(
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            measurement_noise_pos=0.01,
            num_iter=1,
            enable_zupt=False,
            use_mahalanobis_gating=False,
            use_gravity_orientation_update=False,
        ),
        **common_kwargs,
    )
    result_three = extended_kalman_filter_3d(
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            measurement_noise_pos=0.01,
            num_iter=3,
            enable_zupt=False,
            use_mahalanobis_gating=False,
            use_gravity_orientation_update=False,
        ),
        **common_kwargs,
    )

    np.testing.assert_allclose(
        np.asarray(result_three.filtered_covariances[-1]),
        np.asarray(result_one.filtered_covariances[-1]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_ekf_3d_camera_gating_rejects_large_outlier() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    led_offsets = np.array(
        [
            [-0.03, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.0, 0.025, 0.02],
        ],
        dtype=np.float32,
    )
    true_position = jnp.array([1.0, -0.5, 0.25])
    true_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    z_leds_one = true_position[None, :] + rotate_vector_body_to_world(
        true_quat, jnp.asarray(led_offsets)
    )
    z_leds = np.repeat(np.asarray(z_leds_one)[None, :, :], 2, axis=0)
    z_leds[1] += np.array([0.5, -0.4, 0.3], dtype=np.float32)
    t_cam = np.array([0.0, 0.1], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.1, 11, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_imu[:, 5] = 9.81

    result = extended_kalman_filter_3d(
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            measurement_noise_pos=1e-5,
            use_mahalanobis_gating=True,
            enable_zupt=False,
            use_gravity_orientation_update=False,
        ),
        t_imu,
        U_imu,
        t_cam,
        z_leds,
        led_offsets,
    )

    np.testing.assert_allclose(
        np.asarray(result.filtered_means[-1, np.array(layout.pos_idx)]),
        np.asarray(true_position),
        atol=2e-2,
    )


def test_ekf_3d_zupt_updates_velocity_when_enabled() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    mean = jnp.zeros(layout.n)
    mean = mean.at[jnp.array(layout.heading_idx)].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    mean = mean.at[jnp.array(layout.vel_idx)].set(jnp.array([0.01, -0.01, 0.005]))
    cov = jnp.eye(layout.n) * 0.1
    t_cam = np.array([0.0], dtype=np.float32)
    t_imu = np.array([0.0, 0.01], dtype=np.float32)
    U_imu = np.zeros((2, 6), dtype=np.float32)
    U_imu[:, 5] = 9.81
    z_leds = np.full((1, 3, 3), np.nan, dtype=np.float32)
    led_offsets = np.array(
        [
            [-0.03, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.0, 0.025, 0.02],
        ],
        dtype=np.float32,
    )

    result = extended_kalman_filter_3d(
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            enable_zupt=True,
            zupt_velocity_threshold=0.1,
            zupt_measurement_noise=1e-8,
            use_gravity_orientation_update=False,
        ),
        t_imu,
        U_imu,
        t_cam,
        z_leds,
        led_offsets,
        initial_state=FilterState(mean=mean, cov=cov),
    )

    np.testing.assert_allclose(
        np.asarray(result.filtered_means[-1, np.array(layout.vel_idx)]),
        0.0,
        atol=1e-4,
    )


def test_ekf_3d_chi2_threshold_supports_more_than_three_leds() -> None:
    from scipy.stats import chi2

    threshold = _chi2_threshold_active(12, 0.997, dtype=jnp.float32)

    np.testing.assert_allclose(
        np.asarray(threshold),
        chi2.ppf(0.997, df=12),
        rtol=1e-6,
    )


def _noncollinear_3d_led_offsets() -> np.ndarray:
    return np.array(
        [
            [-0.04, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.0, 0.03, 0.02],
        ],
        dtype=np.float32,
    )


def test_ekf_3d_recovers_injected_imu_biases_on_synthetic_data() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    led_offsets = _noncollinear_3d_led_offsets()
    true_position = jnp.array([0.0, 0.0, 0.2])
    true_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    true_gyro_bias = np.array([0.02, -0.015, 0.025], dtype=np.float32)
    true_accel_bias = np.array([0.12, -0.08, 0.10], dtype=np.float32)
    t_cam = np.linspace(0.0, 2.0, 8, dtype=np.float32)
    t_imu = np.linspace(0.0, 2.0, 80, dtype=np.float32)
    z_one = true_position[None, :] + rotate_vector_body_to_world(
        true_quat,
        jnp.asarray(led_offsets),
    )
    z_leds = np.repeat(np.asarray(z_one)[None, :, :], t_cam.shape[0], axis=0)
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_imu[:, :3] = true_gyro_bias
    U_imu[:, 3:6] = np.array([0.0, 0.0, 9.81], dtype=np.float32) + true_accel_bias

    mean = jnp.zeros(layout.n)
    mean = mean.at[jnp.array(layout.pos_idx)].set(
        true_position + jnp.array([0.05, -0.03, 0.02])
    )
    mean = mean.at[jnp.array(layout.heading_idx)].set(true_quat)
    cov = jnp.eye(layout.n) * 0.5
    cov = cov.at[
        jnp.array(layout.bias_gyro_idx),
        jnp.array(layout.bias_gyro_idx),
    ].set(0.2**2)
    cov = cov.at[
        jnp.array(layout.bias_accel_idx),
        jnp.array(layout.bias_accel_idx),
    ].set(0.5**2)

    result = extended_kalman_filter_3d(
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            measurement_noise_pos=1e-5,
            enable_experimental_accel_translation=True,
            enable_zupt=False,
            use_mahalanobis_gating=False,
            use_gravity_orientation_update=False,
            process_noise_gyro_bias=1e-7,
            process_noise_accel_bias=1e-6,
        ),
        t_imu,
        U_imu,
        t_cam,
        z_leds,
        led_offsets,
        initial_state=FilterState(mean=mean, cov=cov),
    )

    final_state = np.asarray(result.filtered_means[-1])
    gyro_bias_error = np.linalg.norm(
        final_state[np.array(layout.bias_gyro_idx)] - true_gyro_bias
    )
    accel_bias_error = np.linalg.norm(
        final_state[np.array(layout.bias_accel_idx)] - true_accel_bias
    )
    assert gyro_bias_error < 5e-3
    assert accel_bias_error < 6e-2


def test_ekf_3d_dropout_comparison_camera_gyro_vs_accel_enabled() -> None:
    layout = get_layout("3d_cam_6dof_imu")
    led_offsets = _noncollinear_3d_led_offsets()
    t_cam = np.linspace(0.0, 0.8, 9, dtype=np.float32)
    t_imu = np.linspace(0.0, 0.8, 81, dtype=np.float32)
    yaw_rate = 0.4
    accel_world = np.array([0.7, -0.25, 0.35], dtype=np.float32)
    initial_position = np.array([0.0, 0.0, 0.25], dtype=np.float32)
    initial_velocity = np.array([0.08, -0.02, 0.03], dtype=np.float32)

    def yaw_quaternion(times: np.ndarray) -> jnp.ndarray:
        yaw = jnp.asarray(yaw_rate * times)
        return jnp.stack(
            [
                jnp.cos(0.5 * yaw),
                jnp.zeros_like(yaw),
                jnp.zeros_like(yaw),
                jnp.sin(0.5 * yaw),
            ],
            axis=-1,
        )

    quat_cam = yaw_quaternion(t_cam)
    quat_imu = yaw_quaternion(t_imu)
    true_positions = (
        initial_position[None, :]
        + initial_velocity[None, :] * t_cam[:, None]
        + 0.5 * accel_world[None, :] * t_cam[:, None] ** 2
    )
    z_leds = np.stack(
        [
            np.asarray(
                position[None, :]
                + rotate_vector_body_to_world(quat, jnp.asarray(led_offsets))
            )
            for position, quat in zip(true_positions, quat_cam, strict=True)
        ]
    ).astype(np.float32)
    mask_leds = np.ones((t_cam.shape[0], led_offsets.shape[0]), dtype=bool)
    mask_leds[3:7] = False
    z_leds[~mask_leds] = np.nan
    dropout_frames = ~mask_leds.any(axis=1)

    U_accel = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_accel[:, :3] = np.array([0.0, 0.0, yaw_rate], dtype=np.float32)
    U_gyro = U_accel.copy()
    U_camera = U_accel.copy()
    U_camera[:, :3] = 0.0
    for idx, quat in enumerate(quat_imu):
        gravity_body = rotate_vector_world_to_body(
            quat,
            jnp.array([0.0, 0.0, 9.81]),
        )
        specific_force_body = rotate_vector_world_to_body(
            quat,
            jnp.asarray(np.array([0.0, 0.0, 9.81], dtype=np.float32) + accel_world),
        )
        U_gyro[idx, 3:6] = np.asarray(gravity_body)
        U_camera[idx, 3:6] = np.asarray(gravity_body)
        U_accel[idx, 3:6] = np.asarray(specific_force_body)

    mean = jnp.zeros(layout.n)
    mean = mean.at[jnp.array(layout.pos_idx)].set(jnp.asarray(initial_position))
    mean = mean.at[jnp.array(layout.vel_idx)].set(jnp.asarray(initial_velocity))
    mean = mean.at[jnp.array(layout.heading_idx)].set(quat_cam[0])
    initial_state = FilterState(mean=mean, cov=jnp.eye(layout.n) * 0.05)

    def run_filter(U_imu: np.ndarray, *, use_accel_translation: bool) -> float:
        result = extended_kalman_filter_3d(
            EKFConfig(
                state_mode="3d_cam_6dof_imu",
                measurement_noise_pos=1e-5,
                enable_experimental_accel_translation=use_accel_translation,
                enable_zupt=False,
                use_mahalanobis_gating=False,
                use_gravity_orientation_update=False,
                adaptive_q_during_dropout=False,
            ),
            t_imu,
            U_imu,
            t_cam,
            z_leds,
            led_offsets,
            mask_cam_leds=mask_leds,
            initial_state=initial_state,
        )
        estimated_positions = np.asarray(
            result.filtered_means[:, np.array(layout.pos_idx)]
        )
        return float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            estimated_positions[dropout_frames]
                            - true_positions[dropout_frames]
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )

    rmse_camera_only = run_filter(U_camera, use_accel_translation=False)
    rmse_gyro_only = run_filter(U_gyro, use_accel_translation=False)
    rmse_accel_enabled = run_filter(U_accel, use_accel_translation=True)

    assert rmse_accel_enabled < 0.25 * rmse_camera_only
    assert rmse_accel_enabled < 0.25 * rmse_gyro_only
    assert rmse_camera_only > 0.04
    assert rmse_gyro_only > 0.04

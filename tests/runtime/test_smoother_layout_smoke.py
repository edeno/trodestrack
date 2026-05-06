from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from trodestrack.models.ekf import (
    EKF3DResult,
    EKFConfig,
    extended_kalman_filter,
    extended_kalman_filter_3d,
    predict_step,
)
from trodestrack.models.filter_common import FilterState, state_yaw, symmetrize
from trodestrack.models.process_noise import assemble_Q
from trodestrack.models.quaternion import rotate_vector_body_to_world
from trodestrack.models.state_layout import get_layout
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.runtime.offline import (
    _transition_mean_and_jacobian,
    rts_smoother,
    sigma_point_smoother,
)


def _tiny_synthetic_sequence():
    # Three camera frames, evenly spaced; IMU at higher rate
    t_cam = np.array([0.0, 0.0333, 0.0666], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.0666, 15, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 3), dtype=np.float32)
    # Straight motion along x, LEDs 4 cm apart on x-axis
    Z1 = np.array([[0.00, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    mask = np.array([True, True, True])
    return t_cam, t_imu, U_imu, Z1, Z2, mask


def _all_nan_dropout_sequence():
    layout = get_layout("2d_full")
    t_cam = np.array([0.0, 0.05, 0.1], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.1, 11, dtype=np.float32)
    u_imu = np.zeros((t_imu.shape[0], 3), dtype=np.float32)
    z1 = np.full((t_cam.shape[0], 2), np.nan, dtype=np.float32)
    z2 = np.full((t_cam.shape[0], 2), np.nan, dtype=np.float32)
    initial_state = FilterState(
        mean=jnp.zeros(layout.n, dtype=jnp.float32),
        cov=jnp.eye(layout.n, dtype=jnp.float32) * 1e-4,
    )
    return t_cam, t_imu, u_imu, z1, z2, initial_state


def test_rts_smoother_smoke_vision_only_layout():
    t_cam, t_imu, U_imu, Z1, Z2, mask = _tiny_synthetic_sequence()

    ekf_config = EKFConfig(
        state_mode="vision_only", led_distance=0.04, use_heading_measurement=False
    )
    filter_result = extended_kalman_filter(
        ekf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )
    smoother_result = rts_smoother(filter_result, ekf_config, t_imu, U_imu, t_cam)

    assert filter_result.filtered_means.shape[1] == 5
    assert smoother_result.smoothed_means.shape[1] == 5


def test_rts_smoother_uses_filter_usable_vision_mask_for_all_nan_leds():
    t_cam, t_imu, u_imu, z1, z2, initial_state = _all_nan_dropout_sequence()
    mask_true = np.ones(t_cam.shape[0], dtype=bool)
    mask_false = np.zeros(t_cam.shape[0], dtype=bool)
    ekf_config = EKFConfig(
        state_mode="2d_full",
        led_distance=0.04,
        enable_zupt=False,
        adaptive_q_during_dropout=True,
        dropout_q_pos_multiplier=1000.0,
        dropout_q_vel_multiplier=1000.0,
        dropout_q_bias_multiplier=0.0,
        freeze_bias_during_blackout=True,
        reduce_imu_noise_during_blackout=False,
        use_heading_measurement=False,
    )

    filter_result = extended_kalman_filter(
        ekf_config,
        t_imu,
        u_imu,
        t_cam,
        z1,
        z2,
        mask_true,
        initial_state=initial_state,
    )
    smoother_mask_true = rts_smoother(
        filter_result, ekf_config, t_imu, u_imu, t_cam, mask_cam=mask_true
    )
    smoother_mask_false = rts_smoother(
        filter_result, ekf_config, t_imu, u_imu, t_cam, mask_cam=mask_false
    )

    np.testing.assert_array_equal(
        np.asarray(filter_result.usable_vision_mask), mask_false
    )
    np.testing.assert_allclose(
        np.asarray(smoother_mask_true.smoothed_covariances),
        np.asarray(smoother_mask_false.smoothed_covariances),
        rtol=1e-6,
        atol=1e-8,
    )


def test_sigma_point_smoother_smoke_2d_cam_3d_imu_layout():
    t_cam, t_imu, U_imu, Z1, Z2, mask = _tiny_synthetic_sequence()

    ukf_config = UKFConfig(
        state_mode="2d_cam_3d_imu", led_distance=0.04, use_heading_measurement=False
    )
    filter_result = unscented_kalman_filter(
        ukf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )
    smoother_result = sigma_point_smoother(
        filter_result, ukf_config, t_imu, U_imu, t_cam
    )

    assert filter_result.filtered_means.shape[1] == 10
    assert smoother_result.smoothed_means.shape[1] == 10


def test_sigma_point_smoother_uses_filter_usable_vision_mask_for_all_nan_leds():
    t_cam, t_imu, u_imu, z1, z2, initial_state = _all_nan_dropout_sequence()
    mask_true = np.ones(t_cam.shape[0], dtype=bool)
    mask_false = np.zeros(t_cam.shape[0], dtype=bool)
    ukf_config = UKFConfig(
        state_mode="2d_full",
        led_distance=0.04,
        enable_zupt=False,
        adaptive_q_during_dropout=True,
        dropout_q_pos_multiplier=1000.0,
        dropout_q_vel_multiplier=1000.0,
        dropout_q_bias_multiplier=0.0,
        freeze_bias_during_blackout=True,
        reduce_imu_noise_during_blackout=False,
        use_heading_measurement=False,
    )

    filter_result = unscented_kalman_filter(
        ukf_config,
        t_imu,
        u_imu,
        t_cam,
        z1,
        z2,
        mask_true,
        initial_state=initial_state,
    )
    smoother_mask_true = sigma_point_smoother(
        filter_result, ukf_config, t_imu, u_imu, t_cam, mask_cam=mask_true
    )
    smoother_mask_false = sigma_point_smoother(
        filter_result, ukf_config, t_imu, u_imu, t_cam, mask_cam=mask_false
    )

    np.testing.assert_array_equal(
        np.asarray(filter_result.usable_vision_mask), mask_false
    )
    np.testing.assert_allclose(
        np.asarray(smoother_mask_true.smoothed_covariances),
        np.asarray(smoother_mask_false.smoothed_covariances),
        rtol=1e-6,
        atol=1e-8,
    )


def test_ukf_layout_no_hardcoded_8d():
    # Ensure UKF works with 10D without indexing errors.
    t_cam, t_imu, U_imu, Z1, Z2, mask = _tiny_synthetic_sequence()
    ukf_config = UKFConfig(
        state_mode="2d_cam_3d_imu", led_distance=0.04, use_heading_measurement=True
    )
    filter_result = unscented_kalman_filter(
        ukf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )
    assert filter_result.filtered_means.shape == (3, 10)
    assert filter_result.predicted_covariances.shape == (3, 10, 10)


def test_rts_smoother_smoke_3d_cam_6dof_imu_layout():
    layout = get_layout("3d_cam_6dof_imu")
    t_cam = np.array([0.0, 0.05, 0.1], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.1, 11, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_imu[:, 5] = 9.81
    led_offsets = np.array(
        [
            [-0.03, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.0, 0.025, 0.02],
        ],
        dtype=np.float32,
    )
    position = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    z_leds_one = position[None, :] + led_offsets
    z_leds = np.repeat(z_leds_one[None, :, :], t_cam.shape[0], axis=0)

    ekf_config = EKFConfig(
        state_mode="3d_cam_6dof_imu",
        measurement_noise_pos=1e-6,
        enable_experimental_accel_translation=True,
        enable_zupt=False,
        use_gravity_orientation_update=False,
        use_mahalanobis_gating=False,
    )
    filter_result = extended_kalman_filter_3d(
        ekf_config,
        t_imu,
        U_imu,
        t_cam,
        z_leds,
        led_offsets,
    )
    smoother_result = rts_smoother(
        filter_result,
        ekf_config,
        t_imu,
        U_imu,
        t_cam,
    )

    quat_idx = np.array(layout.heading_idx)
    assert smoother_result.smoothed_means.shape == (t_cam.shape[0], layout.n)
    assert smoother_result.smoothed_covariances.shape == (
        t_cam.shape[0],
        layout.n,
        layout.n,
    )
    assert np.isfinite(np.asarray(smoother_result.smoothed_means)).all()
    assert np.isfinite(np.asarray(smoother_result.smoothed_covariances)).all()
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(smoother_result.smoothed_means[:, quat_idx]), axis=1),
        1.0,
        atol=1e-5,
    )


def test_rts_smoother_3d_quaternion_reduces_injected_midpoint_position_error():
    layout = get_layout("3d_cam_6dof_imu")
    t_cam = np.linspace(0.0, 0.4, 5, dtype=np.float32)
    t_imu = t_cam.copy()
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_imu[:, 5] = 9.81

    true_states = np.zeros((t_cam.shape[0], layout.n), dtype=np.float32)
    pos_idx = np.array(layout.pos_idx)
    vel_idx = np.array(layout.vel_idx)
    quat_idx = np.array(layout.heading_idx)
    p0 = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    velocity = np.array([0.2, 0.05, -0.1], dtype=np.float32)
    true_states[:, pos_idx] = p0 + t_cam[:, None] * velocity
    true_states[:, vel_idx] = velocity
    true_states[:, quat_idx] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    filtered_means = true_states.copy()
    filtered_means[2, pos_idx] += np.array([0.2, -0.1, 0.05], dtype=np.float32)
    filtered_covs = np.repeat(
        (np.eye(layout.n, dtype=np.float32) * 0.05)[None, :, :],
        t_cam.shape[0],
        axis=0,
    )
    filter_result = EKF3DResult(
        filtered_means=jnp.asarray(filtered_means),
        filtered_covariances=jnp.asarray(filtered_covs),
        # Synthetic shortcut: RTS re-propagates predictions internally.
        predicted_means=jnp.asarray(filtered_means),
        predicted_covariances=jnp.asarray(filtered_covs),
        marginal_loglik=0.0,
    )

    ekf_config = EKFConfig(
        state_mode="3d_cam_6dof_imu",
        enable_experimental_accel_translation=False,
        enable_zupt=False,
        use_gravity_orientation_update=False,
        use_mahalanobis_gating=False,
    )
    smoother_result = rts_smoother(
        filter_result,
        ekf_config,
        t_imu,
        U_imu,
        t_cam,
    )

    smoothed_means = np.asarray(smoother_result.smoothed_means)
    filtered_midpoint_error = np.linalg.norm(
        filtered_means[2, pos_idx] - true_states[2, pos_idx]
    )
    smoothed_midpoint_error = np.linalg.norm(
        smoothed_means[2, pos_idx] - true_states[2, pos_idx]
    )

    assert smoothed_midpoint_error < 0.1 * filtered_midpoint_error
    np.testing.assert_allclose(
        np.linalg.norm(smoothed_means[:, quat_idx], axis=1),
        1.0,
        atol=1e-5,
    )


def test_rts_smoother_3d_quaternion_double_cover_residual_alignment():
    layout = get_layout("3d_cam_6dof_imu")
    t_cam = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    t_imu = t_cam.copy()
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_imu[:, 5] = 9.81

    pos_idx = np.array(layout.pos_idx)
    vel_idx = np.array(layout.vel_idx)
    quat_idx = np.array(layout.heading_idx)
    quat = np.array([0.9238795, 0.0, 0.0, 0.38268343], dtype=np.float32)
    filtered_means = np.zeros((t_cam.shape[0], layout.n), dtype=np.float32)
    filtered_means[:, pos_idx] = np.array(
        [
            [0.0, 0.0, 0.2],
            [0.1, 0.0, 0.2],
            [0.2, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    filtered_means[:, vel_idx] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    filtered_means[:, quat_idx] = np.stack([quat, -quat, quat])
    filtered_covs = np.repeat(
        (np.eye(layout.n, dtype=np.float32) * 0.05)[None, :, :],
        t_cam.shape[0],
        axis=0,
    )
    filter_result = EKF3DResult(
        filtered_means=jnp.asarray(filtered_means),
        filtered_covariances=jnp.asarray(filtered_covs),
        # Synthetic shortcut: RTS re-propagates predictions internally.
        predicted_means=jnp.asarray(filtered_means),
        predicted_covariances=jnp.asarray(filtered_covs),
        marginal_loglik=0.0,
    )

    smoother_result = rts_smoother(
        filter_result,
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            enable_experimental_accel_translation=False,
            enable_zupt=False,
            use_gravity_orientation_update=False,
            use_mahalanobis_gating=False,
        ),
        t_imu,
        U_imu,
        t_cam,
    )

    smoothed_means = np.asarray(smoother_result.smoothed_means)
    smoothed_quats = smoothed_means[:, quat_idx]
    assert np.isfinite(smoothed_means).all()
    np.testing.assert_allclose(
        np.linalg.norm(smoothed_quats, axis=1),
        1.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(smoothed_quats, filtered_means[:, quat_idx], atol=1e-5)
    np.testing.assert_allclose(
        np.abs(np.sum(smoothed_quats[:-1] * smoothed_quats[1:], axis=1)),
        1.0,
        atol=1e-5,
    )


def test_rts_smoother_3d_quaternion_perfect_input_idempotent():
    layout = get_layout("3d_cam_6dof_imu")
    t_cam = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    t_imu = t_cam.copy()
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    U_imu[:, 5] = 9.81

    pos_idx = np.array(layout.pos_idx)
    vel_idx = np.array(layout.vel_idx)
    quat_idx = np.array(layout.heading_idx)
    velocity = np.array([0.2, -0.05, 0.1], dtype=np.float32)
    filtered_means = np.zeros((t_cam.shape[0], layout.n), dtype=np.float32)
    filtered_means[:, pos_idx] = (
        np.array([0.1, -0.2, 0.3], dtype=np.float32)[None, :]
        + t_cam[:, None] * velocity[None, :]
    )
    filtered_means[:, vel_idx] = velocity
    filtered_means[:, quat_idx] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    filtered_covs = np.repeat(
        (np.eye(layout.n, dtype=np.float32) * 0.02)[None, :, :],
        t_cam.shape[0],
        axis=0,
    )
    filter_result = EKF3DResult(
        filtered_means=jnp.asarray(filtered_means),
        filtered_covariances=jnp.asarray(filtered_covs),
        # Synthetic shortcut: RTS re-propagates predictions internally.
        predicted_means=jnp.asarray(filtered_means),
        predicted_covariances=jnp.asarray(filtered_covs),
        marginal_loglik=0.0,
    )

    smoother_result = rts_smoother(
        filter_result,
        EKFConfig(
            state_mode="3d_cam_6dof_imu",
            enable_experimental_accel_translation=False,
            enable_zupt=False,
            use_gravity_orientation_update=False,
            use_mahalanobis_gating=False,
        ),
        t_imu,
        U_imu,
        t_cam,
    )

    smoothed_means = np.asarray(smoother_result.smoothed_means)
    smoothed_cov_traces = np.trace(
        np.asarray(smoother_result.smoothed_covariances),
        axis1=1,
        axis2=2,
    )
    assert np.isfinite(smoothed_means).all()
    assert np.isfinite(smoothed_cov_traces).all()
    np.testing.assert_allclose(
        np.asarray(smoother_result.smoothed_covariances),
        np.swapaxes(np.asarray(smoother_result.smoothed_covariances), 1, 2),
        atol=1e-6,
    )
    np.testing.assert_allclose(smoothed_means, filtered_means, atol=2e-5)
    np.testing.assert_allclose(
        np.linalg.norm(smoothed_means[:, quat_idx], axis=1),
        1.0,
        atol=1e-5,
    )


def test_rts_3d_transition_jacobian_accel_bias_uses_linearization_quaternion():
    layout = get_layout("3d_cam_6dof_imu")
    quat_idx = np.array(layout.heading_idx)
    pos_idx = np.array(layout.pos_idx)
    vel_idx = np.array(layout.vel_idx)
    bias_accel_idx = np.array(layout.bias_accel_idx)
    dt = jnp.asarray(0.1, dtype=jnp.float32)

    state_mean = jnp.zeros(layout.n, dtype=jnp.float32)
    state_mean = state_mean.at[quat_idx].set(jnp.array([1.0, 0.0, 0.0, 0.0]))

    yaw_quarter_turn = jnp.array(
        [
            jnp.cos(jnp.pi / 4.0),
            0.0,
            0.0,
            jnp.sin(jnp.pi / 4.0),
        ],
        dtype=jnp.float32,
    )
    linearization_mean = state_mean.at[quat_idx].set(yaw_quarter_turn)
    u_imu = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.81], dtype=jnp.float32)

    _, F_x = _transition_mean_and_jacobian(
        state_mean,
        linearization_mean,
        u_imu,
        dt,
        ekf_config=EKFConfig(
            state_mode="3d_cam_6dof_imu",
            enable_experimental_accel_translation=True,
        ),
        layout=layout,
    )

    rotation_world_from_body = rotate_vector_body_to_world(
        yaw_quarter_turn,
        jnp.eye(3, dtype=jnp.float32),
    ).T
    expected_vel_bias_block = -dt * rotation_world_from_body
    expected_pos_bias_block = -0.5 * dt**2 * rotation_world_from_body
    np.testing.assert_allclose(
        np.asarray(F_x[np.ix_(pos_idx, bias_accel_idx)]),
        np.asarray(expected_pos_bias_block),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(F_x[np.ix_(vel_idx, bias_accel_idx)]),
        np.asarray(expected_vel_bias_block),
        atol=1e-6,
    )


def test_rts_3d_transition_matches_ekf_prediction_covariance():
    layout = get_layout("3d_cam_6dof_imu")
    quat_idx = np.array(layout.heading_idx)
    mean = jnp.zeros(layout.n, dtype=jnp.float32)
    mean = mean.at[jnp.array(layout.pos_idx)].set(jnp.array([0.1, -0.2, 0.3]))
    mean = mean.at[jnp.array(layout.vel_idx)].set(jnp.array([0.4, -0.1, 0.2]))
    mean = mean.at[jnp.array(layout.heading_idx)].set(
        jnp.array([0.9689124, 0.0, 0.0, 0.24740396])
    )
    mean = mean.at[jnp.array(layout.bias_gyro_idx)].set(jnp.array([0.01, -0.02, 0.015]))
    mean = mean.at[jnp.array(layout.bias_accel_idx)].set(jnp.array([0.03, -0.04, 0.02]))
    cov = jnp.eye(layout.n, dtype=jnp.float32) * 0.05
    u_imu = jnp.array([0.03, -0.01, 0.2, 0.4, -0.1, 9.9], dtype=jnp.float32)
    dt = jnp.asarray(0.02, dtype=jnp.float32)
    config = EKFConfig(
        state_mode="3d_cam_6dof_imu",
        enable_experimental_accel_translation=True,
        adaptive_q_during_dropout=False,
        use_gravity_orientation_update=False,
    )

    ekf_pred = predict_step(
        FilterState(mean=mean, cov=cov),
        u_imu,
        dt,
        config,
        has_vision=False,
        layout=layout,
    )
    rts_mean_pred, F_x = _transition_mean_and_jacobian(
        mean,
        mean,
        u_imu,
        dt,
        ekf_config=config,
        layout=layout,
    )
    Q = assemble_Q(
        config,
        theta=state_yaw(rts_mean_pred, layout),
        dt=dt,
        n=layout.n,
        has_vision=False,
        dtype=mean.dtype,
        orientation_quaternion=rts_mean_pred[quat_idx],
    )
    rts_cov_pred = symmetrize(F_x @ cov @ F_x.T + Q)
    # This isolates Q because EKF and RTS both use the shared F_x helper.
    ekf_q_contribution = ekf_pred.cov - symmetrize(F_x @ cov @ F_x.T)

    np.testing.assert_allclose(
        np.asarray(rts_mean_pred),
        np.asarray(ekf_pred.mean),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(ekf_q_contribution),
        np.asarray(Q),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(rts_cov_pred),
        np.asarray(ekf_pred.cov),
        rtol=1e-6,
        atol=1e-6,
    )

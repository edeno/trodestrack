"""Tests for offline IMU calibration diagnostics."""

import numpy as np

from trodestrack.qa.imu_calibration import (
    camera_body_acceleration,
    compute_led_heading,
    diagnose_accel_axis_signs,
    estimate_accel_gravity_body,
    estimate_gyro_bias,
    estimate_stationary_mask,
    lagged_linear_fit,
)


def test_stationary_bias_and_gravity_estimates_recover_known_values() -> None:
    t_imu = np.linspace(0.0, 10.0, 1_001)
    gyro_bias = 0.037
    gravity_body = np.array([0.4, 3.9, -8.4])

    gyro_z = np.full_like(t_imu, gyro_bias)
    gyro_z[700:] = 0.8
    accel_xyz = np.tile(gravity_body, (t_imu.size, 1))
    accel_xyz[700:, 0] += 2.0

    stationary = estimate_stationary_mask(
        t_imu,
        gyro_z,
        gyro_threshold=0.2,
    )

    assert stationary.sum() == 700
    np.testing.assert_allclose(estimate_gyro_bias(gyro_z, stationary), gyro_bias)
    np.testing.assert_allclose(
        estimate_accel_gravity_body(accel_xyz, stationary),
        gravity_body,
    )


def test_lagged_linear_fit_recovers_lag_scale_and_bias() -> None:
    t_source = np.linspace(0.0, 20.0, 2_001)
    t_target = np.linspace(1.0, 19.0, 901)
    true_lag = 0.12
    source_signal = np.sin(2.0 * np.pi * 0.4 * (t_source - true_lag))
    target_signal = 0.25 + 1.7 * np.sin(2.0 * np.pi * 0.4 * t_target)
    candidate_lags = np.linspace(-0.3, 0.3, 121)

    fit = lagged_linear_fit(
        t_source,
        source_signal,
        t_target,
        target_signal,
        candidate_lags,
    )

    assert abs(fit.lag_s - true_lag) <= 0.01
    assert fit.correlation > 0.99
    np.testing.assert_allclose(fit.slope, 1.7, rtol=1e-2)
    np.testing.assert_allclose(fit.intercept, 0.25, atol=1e-3)


def test_accel_axis_sign_diagnostics_identify_swaps_and_signs() -> None:
    t_imu = np.linspace(0.0, 10.0, 1_001)
    t_cam = np.linspace(0.5, 9.5, 271)
    body_x = np.sin(2.0 * np.pi * 0.5 * t_imu)
    body_y = np.cos(2.0 * np.pi * 0.3 * t_imu)
    accel_kinematic = np.column_stack(
        [
            0.1 * np.sin(t_imu),
            -body_x,
            body_y,
        ]
    )
    camera_body = np.column_stack(
        [
            np.interp(t_cam, t_imu, body_x),
            np.interp(t_cam, t_imu, body_y),
        ]
    )

    diagnostics = diagnose_accel_axis_signs(
        t_imu,
        accel_kinematic,
        t_cam,
        camera_body,
        candidate_lags_s=np.array([0.0]),
    )

    assert diagnostics[0].target_axis == "body_x"
    assert diagnostics[0].imu_axis == "y"
    assert diagnostics[0].sign == -1
    assert diagnostics[0].correlation > 0.99

    assert diagnostics[1].target_axis == "body_y"
    assert diagnostics[1].imu_axis == "z"
    assert diagnostics[1].sign == 1
    assert diagnostics[1].correlation > 0.99


def test_heading_and_camera_acceleration_handle_missing_led_frames() -> None:
    t_cam = np.linspace(0.0, 1.0, 21)
    led1 = np.column_stack([t_cam, np.zeros_like(t_cam)])
    led2 = led1 + np.array([0.04, 0.0])
    led1[8:11] = np.nan
    led2[9:12] = np.nan

    heading = compute_led_heading(led1, led2)
    accel = camera_body_acceleration(t_cam, led1, led2, smoothing_sigma_s=0.05)

    assert np.isfinite(heading).all()
    assert np.isfinite(accel).all()
    np.testing.assert_allclose(heading, 0.0, atol=1e-7)


def test_accel_axis_sign_diagnostics_use_one_to_one_axis_assignment() -> None:
    t_imu = np.linspace(0.0, 10.0, 1_001)
    t_cam = np.linspace(0.5, 9.5, 271)
    shared = np.sin(2.0 * np.pi * 0.4 * t_imu)
    body_y_true = np.cos(2.0 * np.pi * 0.3 * t_imu)
    accel_kinematic = np.column_stack(
        [
            shared,
            body_y_true,
            0.1 * np.sin(t_imu),
        ]
    )
    camera_body = np.column_stack(
        [
            np.interp(t_cam, t_imu, shared),
            0.95 * np.interp(t_cam, t_imu, shared)
            + 0.25 * np.interp(t_cam, t_imu, body_y_true),
        ]
    )

    diagnostics = diagnose_accel_axis_signs(
        t_imu,
        accel_kinematic,
        t_cam,
        camera_body,
        candidate_lags_s=np.array([0.0]),
    )

    assert diagnostics[0].imu_axis != diagnostics[1].imu_axis

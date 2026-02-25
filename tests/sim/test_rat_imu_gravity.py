"""Tests for gravity and tilt in rat_imu simulation."""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.sim.rat_imu import (
    RatIMUSimConfig,
    compute_gravity_in_tilted_frame,
    simulate_rat_imu,
)


def test_compute_gravity_in_tilted_frame_zero_tilt():
    """Test that zero tilt gives zero gravity in horizontal plane."""
    g_x, g_y = compute_gravity_in_tilted_frame(0.0, 0.0, 9.80665)

    assert np.isclose(g_x, 0.0, atol=1e-10)
    assert np.isclose(g_y, 0.0, atol=1e-10)


def test_compute_gravity_in_tilted_frame_pitch_only():
    """Test that pitch tilt projects gravity onto x-axis."""
    pitch = np.deg2rad(5.0)  # 5 degree pitch
    g = 9.80665

    g_x, g_y = compute_gravity_in_tilted_frame(0.0, pitch, g)

    # g_x = g * sin(pitch)
    expected_gx = g * np.sin(pitch)
    assert np.isclose(g_x, expected_gx, rtol=1e-10)
    assert np.isclose(g_y, 0.0, atol=1e-10)


def test_compute_gravity_in_tilted_frame_roll_only():
    """Test that roll tilt projects gravity onto y-axis."""
    roll = np.deg2rad(5.0)  # 5 degree roll
    g = 9.80665

    g_x, g_y = compute_gravity_in_tilted_frame(roll, 0.0, g)

    # g_y = -g * sin(roll) * cos(0) = -g * sin(roll)
    expected_gy = -g * np.sin(roll)
    assert np.isclose(g_x, 0.0, atol=1e-10)
    assert np.isclose(g_y, expected_gy, rtol=1e-10)


def test_compute_gravity_in_tilted_frame_small_angle_approx():
    """Test small angle approximation for tilt."""
    # For small angles: g_x ≈ g * θ_p, g_y ≈ -g * θ_r
    roll = np.deg2rad(2.0)
    pitch = np.deg2rad(3.0)
    g = 9.80665

    g_x, g_y = compute_gravity_in_tilted_frame(roll, pitch, g)

    # Small angle approximation
    approx_gx = g * pitch
    approx_gy = -g * roll

    # Should be very close for small angles
    assert np.isclose(g_x, approx_gx, rtol=0.01)  # Within 1%
    assert np.isclose(g_y, approx_gy, rtol=0.01)


def test_stationary_rat_measures_gravity():
    """Test that stationary rat with zero inertial accel still measures specific force."""
    # Create a very simple scenario: stationary rat, no motion
    config = RatIMUSimConfig(
        duration_s=5.0,
        fs_imu=200.0,
        fs_cam=30.0,
        # Very low process noise to keep rat nearly stationary
        sigma_yaw_rate=0.01,
        sigma_a_fwd=0.01,
        sigma_a_lat=0.01,
        vel_drag=10.0,  # High drag to kill motion quickly
        # Small tilt
        imu_tilt_roll_deg=2.0,
        imu_tilt_pitch_deg=3.0,
    )

    sim = simulate_rat_imu(config, seed=42)

    # Extract data
    accel_meas = sim["U_imu"][:, 1:3]  # Specific force measurements
    accel_truth = sim["accel_body_truth"]  # Inertial acceleration

    # After initial transient, rat should be nearly stationary
    # Use last 50% of data
    half = len(accel_truth) // 2
    accel_meas_late = accel_meas[half:]
    accel_truth_late = accel_truth[half:]

    # Mean inertial acceleration should be small (nearly stationary)
    mean_inertial = np.mean(accel_truth_late, axis=0)
    assert np.linalg.norm(mean_inertial) < 0.5  # Less than 0.5 m/s²

    # But mean specific force should be non-zero (includes gravity)
    mean_sf = np.mean(accel_meas_late, axis=0)
    sf_magnitude = np.linalg.norm(mean_sf)

    # Should be on order of gravity component (~0.5-1 m/s² for small tilts)
    assert sf_magnitude > 0.2  # At least 0.2 m/s²
    assert sf_magnitude < 2.0  # But not too large


def test_specific_force_vs_inertial_accel_difference():
    """Test that specific force and inertial accel differ by gravity term."""
    config = RatIMUSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        imu_tilt_roll_deg=3.0,
        imu_tilt_pitch_deg=2.0,
    )

    sim = simulate_rat_imu(config, seed=123)

    accel_meas = sim["U_imu"][:, 1:3]  # Specific force
    accel_truth = sim["accel_body_truth"]  # Inertial accel
    bias_x = sim["bias_accel_x"]
    bias_y = sim["bias_accel_y"]

    # Remove bias and noise from measurements
    accel_meas_unbiased = accel_meas - np.column_stack([bias_x, bias_y])

    # Difference should be due to gravity component
    # specific_force = inertial_accel - g_body
    diff = accel_meas_unbiased - accel_truth

    # Instantaneous difference magnitude should be on order of gravity
    # (not the mean, because gravity rotates with heading)
    diff_magnitudes = np.linalg.norm(diff, axis=1)
    mean_diff_magnitude = np.mean(diff_magnitudes)

    # Expected gravity magnitude in tilted frame
    tilt_roll_rad = np.deg2rad(config.imu_tilt_roll_deg)
    tilt_pitch_rad = np.deg2rad(config.imu_tilt_pitch_deg)
    g_x, g_y = compute_gravity_in_tilted_frame(
        tilt_roll_rad, tilt_pitch_rad, config.gravity
    )
    expected_g_magnitude = np.hypot(g_x, g_y)

    # Mean instantaneous magnitude should be close to expected gravity magnitude
    assert np.isclose(mean_diff_magnitude, expected_g_magnitude, rtol=0.3)


def test_gravity_rotates_with_heading():
    """Test that gravity rotates with rat's heading."""
    # Create scenario where rat does a full rotation
    config = RatIMUSimConfig(
        duration_s=5.0,
        fs_imu=200.0,
        sigma_yaw_rate=np.deg2rad(100.0),  # Strong yaw rate
        sigma_a_fwd=0.01,  # Minimal translation
        sigma_a_lat=0.01,
        vel_drag=5.0,
        imu_tilt_roll_deg=5.0,  # Larger tilt for clearer signal
        imu_tilt_pitch_deg=5.0,
    )

    sim = simulate_rat_imu(config, seed=456)

    # Extract heading and specific force
    X = sim["X_truth"]
    theta = X[:, 4]  # heading
    accel_meas = sim["U_imu"][:, 1:3]
    accel_truth = sim["accel_body_truth"]

    # Remove biases
    bias_x = sim["bias_accel_x"]
    bias_y = sim["bias_accel_y"]
    sf_unbiased = accel_meas - np.column_stack([bias_x, bias_y])

    # Gravity component in body frame should vary with heading
    # Compute expected gravity in body frame at each timestep
    tilt_roll_rad = np.deg2rad(config.imu_tilt_roll_deg)
    tilt_pitch_rad = np.deg2rad(config.imu_tilt_pitch_deg)
    g_imu_x, g_imu_y = compute_gravity_in_tilted_frame(
        tilt_roll_rad, tilt_pitch_rad, config.gravity
    )

    # Rotate gravity to body frame at each timestep
    c = np.cos(theta)
    s = np.sin(theta)
    g_body_x = c * g_imu_x + s * g_imu_y
    g_body_y = -s * g_imu_x + c * g_imu_y

    # Gravity component = inertial_accel - specific_force
    g_estimated_x = accel_truth[:, 0] - sf_unbiased[:, 0]
    g_estimated_y = accel_truth[:, 1] - sf_unbiased[:, 1]

    # Should correlate with expected gravity (allowing for noise)
    # Check that magnitude varies with heading (std > 0)
    assert np.std(g_body_x) > 0.1  # Should vary significantly
    assert np.std(g_body_y) > 0.1

    # Mean of estimated gravity should be close to mean of expected
    # (averaged over all headings, some components cancel)
    mean_g_est = np.mean([g_estimated_x, g_estimated_y], axis=1)
    mean_g_exp = np.mean([g_body_x, g_body_y], axis=1)

    # Should be within reasonable tolerance given noise
    assert np.allclose(mean_g_est, mean_g_exp, atol=0.5)


def test_default_tilt_is_small_but_nonzero():
    """Test that default configuration has small but non-zero tilt."""
    config = RatIMUSimConfig()

    assert config.imu_tilt_roll_deg > 0
    assert config.imu_tilt_pitch_deg > 0
    assert config.imu_tilt_roll_deg < 10  # Reasonable range
    assert config.imu_tilt_pitch_deg < 10


def test_zero_tilt_option():
    """Test that user can set zero tilt if desired."""
    config = RatIMUSimConfig(
        duration_s=2.0,
        imu_tilt_roll_deg=0.0,
        imu_tilt_pitch_deg=0.0,
    )

    sim = simulate_rat_imu(config, seed=42)

    # With zero tilt and averaging over many headings,
    # mean gravity effect should be very small
    accel_meas = sim["U_imu"][:, 1:3]
    accel_truth = sim["accel_body_truth"]
    bias_x = sim["bias_accel_x"]
    bias_y = sim["bias_accel_y"]

    sf_unbiased = accel_meas - np.column_stack([bias_x, bias_y])
    g_component = accel_truth - sf_unbiased

    # Mean gravity component should be near zero
    mean_g = np.mean(g_component, axis=0)
    assert np.linalg.norm(mean_g) < 0.2  # Very small


# =============================================================================
# Filter Integration Tests: 3D IMU with Gravity Compensation
# =============================================================================


def test_filter_3d_imu_accepts_4_element_input():
    """Test that EKF/UKF can process 4-element IMU input [ω_z, fx, fy, fz]."""
    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

    # Create minimal synthetic data with 3D IMU
    duration_s = 2.0
    fs_imu = 200.0
    fs_cam = 30.0
    T_imu = int(duration_s * fs_imu)
    T_cam = int(duration_s * fs_cam)

    t_imu = np.arange(T_imu) / fs_imu
    t_cam = np.arange(T_cam) / fs_cam

    # 3D IMU: [ω_z, fx, fy, fz] - stationary rat reading gravity
    U_imu = np.zeros((T_imu, 4))
    U_imu[:, 3] = 9.81  # fz reads gravity at rest

    # Camera measurements: stationary at origin
    Z_cam_led1 = np.tile([0.5, 0.5], (T_cam, 1))
    Z_cam_led2 = np.tile([0.55, 0.5], (T_cam, 1))
    mask_cam = np.ones(T_cam, dtype=bool)

    # Run filter with 10D state (2D camera + 3D IMU)
    config = EKFConfig(state_mode="2d_cam_3d_imu")

    result = extended_kalman_filter(
        ekf_config=config,
        t_imu=t_imu,
        U_imu=U_imu,
        t_cam=t_cam,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
    )

    # Should complete without errors
    assert result.filtered_means.shape == (T_cam, 10)
    assert result.filtered_covariances.shape == (T_cam, 10, 10)


def test_filter_3d_imu_gravity_compensation_at_rest():
    """Test that filter correctly handles gravity at rest (vz remains ~0).

    At rest, the IMU reads [0, 0, 9.81] in fz. After gravity compensation,
    the vertical velocity should remain near zero.
    """
    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

    # Create stationary scenario with 3D IMU
    duration_s = 5.0
    fs_imu = 200.0
    fs_cam = 30.0
    T_imu = int(duration_s * fs_imu)
    T_cam = int(duration_s * fs_cam)

    t_imu = np.arange(T_imu) / fs_imu
    t_cam = np.arange(T_cam) / fs_cam

    # 3D IMU: stationary rat
    U_imu = np.zeros((T_imu, 4))
    U_imu[:, 3] = 9.81  # fz reads gravity

    # Camera: stationary at [0.5, 0.5]
    Z_cam_led1 = np.tile([0.5, 0.5], (T_cam, 1))
    Z_cam_led2 = np.tile([0.55, 0.5], (T_cam, 1))
    mask_cam = np.ones(T_cam, dtype=bool)

    config = EKFConfig(state_mode="2d_cam_3d_imu")

    result = extended_kalman_filter(
        ekf_config=config,
        t_imu=t_imu,
        U_imu=U_imu,
        t_cam=t_cam,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
    )

    # Extract vertical velocity (index 4 in LAYOUT_2D_CAM_3D_IMU)
    vz = result.filtered_means[:, 4]

    # After convergence (last 50%), vz should remain near zero
    half = len(vz) // 2
    vz_late = vz[half:]

    # Mean should be close to zero (gravity compensated)
    assert np.abs(np.mean(vz_late)) < 0.05, (
        f"vz mean {np.mean(vz_late):.4f} should be ~0"
    )

    # Std should be small (no vertical motion)
    assert np.std(vz_late) < 0.1, f"vz std {np.std(vz_late):.4f} should be small"


def test_filter_3d_imu_detects_vertical_acceleration():
    """Test that filter detects vertical acceleration during jumping.

    When fz > 9.81, the rat is accelerating upward. The filter should
    estimate increasing vz.
    """
    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

    duration_s = 3.0
    fs_imu = 200.0
    fs_cam = 30.0
    T_imu = int(duration_s * fs_imu)
    T_cam = int(duration_s * fs_cam)

    t_imu = np.arange(T_imu) / fs_imu
    t_cam = np.arange(T_cam) / fs_cam

    # 3D IMU: simulate jump at t=1s
    U_imu = np.zeros((T_imu, 4))
    U_imu[:, 3] = 9.81  # Default: gravity

    # Jump impulse: 1.0s to 1.5s, accel = 2 m/s² above gravity
    jump_start = int(1.0 * fs_imu)
    jump_end = int(1.5 * fs_imu)
    U_imu[jump_start:jump_end, 3] = 9.81 + 2.0  # 11.81 m/s²

    # Camera: stationary horizontal position
    Z_cam_led1 = np.tile([0.5, 0.5], (T_cam, 1))
    Z_cam_led2 = np.tile([0.55, 0.5], (T_cam, 1))
    mask_cam = np.ones(T_cam, dtype=bool)

    config = EKFConfig(state_mode="2d_cam_3d_imu")

    result = extended_kalman_filter(
        ekf_config=config,
        t_imu=t_imu,
        U_imu=U_imu,
        t_cam=t_cam,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
    )

    # Extract vertical velocity
    vz = result.filtered_means[:, 4]

    # Before jump: vz near zero
    t_cam_arr = t_cam
    before_jump = vz[t_cam_arr < 0.9]
    assert np.abs(np.mean(before_jump)) < 0.1, "vz should be ~0 before jump"

    # During/after jump: vz should increase
    during_jump = vz[(t_cam_arr >= 1.0) & (t_cam_arr < 1.6)]

    # Should show positive velocity (accelerating upward)
    assert np.mean(during_jump) > 0.2, (
        f"vz during jump {np.mean(during_jump):.3f} should be positive"
    )


def test_filter_3d_imu_reduced_drift_during_occlusion():
    """Test that 3D IMU reduces drift compared to 2D IMU during occlusions.

    PRD requirement (Section 4): Vision dropout tolerance → ≤3.5 m drift after 5s IMU-only.

    This test compares:
    1. 2D IMU (3-element: [ω_z, fx, fy]) - baseline
    2. 3D IMU (4-element: [ω_z, fx, fy, fz]) - should reduce drift

    The 3D mode with gravity compensation should maintain better velocity estimates
    during occlusions, leading to reduced positional drift.
    """
    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

    # Scenario: rat moving, then 3-second vision blackout
    duration_s = 8.0
    fs_imu = 200.0
    fs_cam = 30.0
    T_imu = int(duration_s * fs_imu)
    T_cam = int(duration_s * fs_cam)

    t_imu = np.arange(T_imu) / fs_imu
    t_cam = np.arange(T_cam) / fs_cam

    # Ground truth: rat moving at 0.3 m/s in x-direction
    true_vx = 0.3
    true_x = 0.5 + true_vx * t_cam  # Starting at x=0.5
    true_y = np.full_like(t_cam, 0.5)

    # Camera measurements (valid until blackout)
    Z_cam_led1 = np.column_stack([true_x, true_y])
    Z_cam_led2 = np.column_stack([true_x + 0.05, true_y])

    # Mask: all valid except 3s blackout from t=3s to t=6s
    mask_cam = np.ones(T_cam, dtype=bool)
    blackout_start_idx = int(3.0 * fs_cam)
    blackout_end_idx = int(6.0 * fs_cam)
    mask_cam[blackout_start_idx:blackout_end_idx] = False

    # --- Run 1: 2D IMU (baseline) ---
    U_imu_2d = np.zeros((T_imu, 3))
    # Simulate constant velocity in body frame (fx = 0 after initialization)
    U_imu_2d[:, 1] = 0.0  # fx
    U_imu_2d[:, 2] = 0.0  # fy

    config_2d = EKFConfig(state_mode="2d_full")  # LAYOUT_DEFAULT (8D)

    result_2d = extended_kalman_filter(
        ekf_config=config_2d,
        t_imu=t_imu,
        U_imu=U_imu_2d,
        t_cam=t_cam,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
    )

    # --- Run 2: 3D IMU (with gravity compensation) ---
    U_imu_3d = np.zeros((T_imu, 4))
    U_imu_3d[:, 1] = 0.0  # fx
    U_imu_3d[:, 2] = 0.0  # fy
    U_imu_3d[:, 3] = 9.81  # fz reads gravity

    config_3d = EKFConfig(state_mode="2d_cam_3d_imu")  # LAYOUT_2D_CAM_3D_IMU (10D)

    result_3d = extended_kalman_filter(
        ekf_config=config_3d,
        t_imu=t_imu,
        U_imu=U_imu_3d,
        t_cam=t_cam,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
    )

    # Measure drift at end of blackout (t=6s)
    end_blackout_idx = blackout_end_idx

    # 2D drift
    x_2d = result_2d.filtered_means[:, 0]
    y_2d = result_2d.filtered_means[:, 1]
    drift_2d = np.hypot(
        x_2d[end_blackout_idx] - true_x[end_blackout_idx],
        y_2d[end_blackout_idx] - true_y[end_blackout_idx],
    )

    # 3D drift
    x_3d = result_3d.filtered_means[:, 0]
    y_3d = result_3d.filtered_means[:, 1]
    drift_3d = np.hypot(
        x_3d[end_blackout_idx] - true_x[end_blackout_idx],
        y_3d[end_blackout_idx] - true_y[end_blackout_idx],
    )

    print("\nDrift after 3s occlusion:")
    print(f"  2D IMU: {drift_2d:.3f} m")
    print(f"  3D IMU: {drift_3d:.3f} m")

    # Both should meet PRD requirement (≤3.5 m for 5s, so ~2.1 m for 3s)
    # But 3D should be better or comparable
    assert drift_3d <= 2.5, f"3D IMU drift {drift_3d:.3f}m exceeds tolerance"

    # 3D should be ≤ 2D (or at least not significantly worse)
    # Allow small tolerance for numerical differences
    assert drift_3d <= drift_2d * 1.1, (
        f"3D IMU drift {drift_3d:.3f}m should not be worse than 2D {drift_2d:.3f}m"
    )


def test_filter_3d_imu_backward_compatible_with_2d():
    """Test that 3-element IMU input still works (backward compatibility).

    Users should be able to use 3-element IMU [ω_z, fx, fy] with the filter,
    and it should behave identically to the old 2D mode.
    """
    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

    duration_s = 2.0
    fs_imu = 200.0
    fs_cam = 30.0
    T_imu = int(duration_s * fs_imu)
    T_cam = int(duration_s * fs_cam)

    t_imu = np.arange(T_imu) / fs_imu
    t_cam = np.arange(T_cam) / fs_cam

    # 2D IMU: [ω_z, fx, fy]
    U_imu_2d = np.zeros((T_imu, 3))

    # Camera: stationary
    Z_cam_led1 = np.tile([0.5, 0.5], (T_cam, 1))
    Z_cam_led2 = np.tile([0.55, 0.5], (T_cam, 1))
    mask_cam = np.ones(T_cam, dtype=bool)

    config = EKFConfig(state_mode="2d_full")  # LAYOUT_DEFAULT (8D)

    # Should not crash with 3-element IMU
    result = extended_kalman_filter(
        ekf_config=config,
        t_imu=t_imu,
        U_imu=U_imu_2d,
        t_cam=t_cam,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
    )

    assert result.filtered_means.shape == (T_cam, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

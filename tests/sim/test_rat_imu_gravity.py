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
    g_x, g_y = compute_gravity_in_tilted_frame(tilt_roll_rad, tilt_pitch_rad, config.gravity)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

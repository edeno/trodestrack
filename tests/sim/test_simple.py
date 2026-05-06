"""Tests for simple analytic simulations.

Tests verify:
1. Ground truth matches analytic solutions
2. IMU measurements are physically correct
3. Camera measurements track position
4. Noise characteristics are as specified
5. Data structure consistency
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Standard test configuration with low noise."""
    return SimpleSimConfig(
        duration_s=5.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.0001,
        accel_noise_density=0.03,
        gyro_bias_std=0.001,
        accel_bias_std=0.01,
        cam_noise_std=0.002,
        cam_dropout_prob=0.1,
    )


# =============================================================================
# Test: simulate_stationary
# =============================================================================


def test_stationary_returns_correct_keys(config):
    """Test that stationary simulation returns all expected keys."""
    sim = simulate_stationary(config, seed=42)

    expected_keys = {
        "t_imu",
        "t_cam_exp",
        "t_cam_obs",
        "X_truth",
        "U_imu",
        "Z_cam_led1",
        "Z_cam_led2",
        "mask_cam",
        "mask_led1",  # Individual LED masks
        "mask_led2",  # Individual LED masks
        "confidence_led1",
        "confidence_led2",
        "led1_truth_cam",  # Ground truth LED positions before noise/swaps/reflections
        "led2_truth_cam",  # Ground truth LED positions before noise/swaps/reflections
        "swap_applied",  # Artifact tracking
        "led_reflection_applied",  # Artifact tracking
        "bias_gyro",
        "bias_accel_x",
        "bias_accel_y",
        "yaw_rate_truth",
        "accel_world_truth",
        "accel_body_truth",
        "config",
    }

    assert set(sim.keys()) == expected_keys


def test_stationary_position_constant(config):
    """Test that position remains constant in stationary simulation."""
    position = np.array([0.5, 0.5])
    sim = simulate_stationary(config, position=position, seed=42)

    X = sim["X_truth"]

    # All positions should be exactly equal to initial position
    assert np.allclose(X[:, 0], position[0])
    assert np.allclose(X[:, 1], position[1])


def test_stationary_velocity_zero(config):
    """Test that velocity is zero in stationary simulation."""
    sim = simulate_stationary(config, seed=42)

    X = sim["X_truth"]

    # All velocities should be exactly zero
    assert np.allclose(X[:, 2], 0.0)
    assert np.allclose(X[:, 3], 0.0)


def test_stationary_heading_constant(config):
    """Test that heading remains constant in stationary simulation."""
    heading = 0.5  # rad
    sim = simulate_stationary(config, heading=heading, seed=42)

    X = sim["X_truth"]

    # All headings should be exactly equal to initial heading
    assert np.allclose(X[:, 4], heading)


def test_stationary_gyro_near_zero(config):
    """Test that gyro measures near-zero angular velocity (bias + noise)."""
    sim = simulate_stationary(config, seed=42)

    gyro = sim["U_imu"][:, 0]
    bias = sim["bias_gyro"]

    # Remove bias, should be small noise
    gyro_unbiased = gyro - bias

    # Should be within 5 sigma of zero (99.9999% confidence)
    dt = 1.0 / config.fs_imu
    sigma = config.gyro_noise_density / np.sqrt(dt)
    assert np.abs(np.mean(gyro_unbiased)) < 5 * sigma / np.sqrt(len(gyro))
    assert np.std(gyro_unbiased) < 2 * sigma  # Within factor of 2


def test_stationary_accel_measures_gravity(config):
    """Test that accelerometer measures specific force (≈0 for level-mounted, stationary IMU)."""
    heading = np.pi / 6  # 30 degrees
    sim = simulate_stationary(config, heading=heading, seed=42)

    accel_x = sim["U_imu"][:, 1]
    accel_y = sim["U_imu"][:, 2]

    # Expected: specific force = 0 (no motion, gravity is along Z for level mounting)
    expected_x = 0.0
    expected_y = 0.0

    # Remove bias
    bias_x = sim["bias_accel_x"]
    bias_y = sim["bias_accel_y"]

    mean_x = np.mean(accel_x - bias_x)
    mean_y = np.mean(accel_y - bias_y)

    # Should match expected within noise tolerance
    dt = 1.0 / config.fs_imu
    sigma = config.accel_noise_density / np.sqrt(dt)

    assert np.abs(mean_x - expected_x) < 3 * sigma / np.sqrt(len(accel_x))
    assert np.abs(mean_y - expected_y) < 3 * sigma / np.sqrt(len(accel_y))


def test_stationary_camera_near_position(config):
    """Test that camera measurements are near true position."""
    position = np.array([0.5, 0.5])
    sim = simulate_stationary(config, position=position, seed=42)

    Z = sim["Z_cam_led1"]
    mask = sim["mask_cam"]

    # Valid measurements should be near true position
    Z_valid = Z[mask]

    mean_pos = np.mean(Z_valid, axis=0)
    std_pos = np.std(Z_valid, axis=0)

    # Mean should be close to true position
    assert np.allclose(mean_pos, position, atol=3 * config.cam_noise_std)

    # Std should match noise level
    assert np.allclose(std_pos, config.cam_noise_std, rtol=0.3)


# =============================================================================
# Test: simulate_constant_velocity
# =============================================================================


def test_constant_velocity_position_linear(config):
    """Test that position follows linear trajectory."""
    initial = np.array([0.1, 0.1])
    velocity = np.array([0.2, 0.1])

    sim = simulate_constant_velocity(
        config, initial_position=initial, velocity=velocity, seed=42
    )

    t = sim["t_imu"]
    X = sim["X_truth"]

    # Position should be: p(t) = p0 + v*t
    expected_x = initial[0] + velocity[0] * t
    expected_y = initial[1] + velocity[1] * t

    assert np.allclose(X[:, 0], expected_x, atol=1e-10)
    assert np.allclose(X[:, 1], expected_y, atol=1e-10)


def test_constant_velocity_velocity_constant(config):
    """Test that velocity remains constant."""
    velocity = np.array([0.2, 0.1])

    sim = simulate_constant_velocity(config, velocity=velocity, seed=42)

    X = sim["X_truth"]

    assert np.allclose(X[:, 2], velocity[0])
    assert np.allclose(X[:, 3], velocity[1])


def test_constant_velocity_heading_aligned(config):
    """Test that heading aligns with velocity direction."""
    velocity = np.array([0.2, 0.1])

    sim = simulate_constant_velocity(config, velocity=velocity, seed=42)

    X = sim["X_truth"]
    expected_heading = np.arctan2(velocity[1], velocity[0])

    assert np.allclose(X[:, 4], expected_heading)


def test_constant_velocity_gyro_near_zero(config):
    """Test that gyro measures near-zero (no rotation)."""
    sim = simulate_constant_velocity(config, seed=42)

    gyro = sim["U_imu"][:, 0]
    bias = sim["bias_gyro"]
    gyro_unbiased = gyro - bias

    # Should be small noise
    dt = 1.0 / config.fs_imu
    sigma = config.gyro_noise_density / np.sqrt(dt)
    assert np.abs(np.mean(gyro_unbiased)) < 5 * sigma / np.sqrt(len(gyro))


def test_constant_velocity_camera_tracks_position(config):
    """Test that camera measurements track linear motion."""
    initial = np.array([0.1, 0.1])
    velocity = np.array([0.2, 0.1])

    sim = simulate_constant_velocity(
        config, initial_position=initial, velocity=velocity, seed=42
    )

    t_cam = sim["t_cam_exp"]
    Z = sim["Z_cam_led1"]
    mask = sim["mask_cam"]

    # Expected position at camera times
    expected_x = initial[0] + velocity[0] * t_cam
    expected_y = initial[1] + velocity[1] * t_cam

    # Check valid measurements are close to expected
    Z_valid = Z[mask]
    expected_valid = np.column_stack([expected_x[mask], expected_y[mask]])

    # Should be within a few sigma
    diff = Z_valid - expected_valid
    assert np.abs(np.mean(diff)) < 3 * config.cam_noise_std / np.sqrt(len(Z_valid))


# =============================================================================
# Test: simulate_circular
# =============================================================================


def test_circular_position_on_circle(config):
    """Test that position stays on circle."""
    center = np.array([0.5, 0.5])
    radius = 0.3

    sim = simulate_circular(config, center=center, radius=radius, seed=42)

    X = sim["X_truth"]
    x = X[:, 0]
    y = X[:, 1]

    # Distance from center should equal radius
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    assert np.allclose(distance, radius, atol=1e-10)


def test_circular_velocity_tangent(config):
    """Test that velocity is tangent to circle."""
    center = np.array([0.5, 0.5])
    radius = 0.3
    omega = 1.0

    sim = simulate_circular(
        config, center=center, radius=radius, angular_velocity=omega, seed=42
    )

    X = sim["X_truth"]
    x = X[:, 0]
    y = X[:, 1]
    vx = X[:, 2]
    vy = X[:, 3]

    # Velocity magnitude should be v = r*ω
    speed = np.sqrt(vx**2 + vy**2)
    expected_speed = radius * omega
    assert np.allclose(speed, expected_speed, atol=1e-10)

    # Velocity should be perpendicular to radius vector
    # (x - x_c, y - y_c) · (vx, vy) = 0
    radial_x = x - center[0]
    radial_y = y - center[1]
    dot_product = radial_x * vx + radial_y * vy

    assert np.allclose(dot_product, 0.0, atol=1e-10)


def test_circular_heading_tangent(config):
    """Test that heading points tangent to circle."""
    center = np.array([0.5, 0.5])
    radius = 0.3
    omega = 1.0

    sim = simulate_circular(
        config, center=center, radius=radius, angular_velocity=omega, seed=42
    )

    X = sim["X_truth"]
    vx = X[:, 2]
    vy = X[:, 3]
    heading = X[:, 4]

    # Heading should align with velocity direction
    expected_heading = np.arctan2(vy, vx)

    # Handle wrapping
    diff = np.arctan2(
        np.sin(heading - expected_heading), np.cos(heading - expected_heading)
    )
    assert np.allclose(diff, 0.0, atol=1e-10)


def test_circular_gyro_measures_omega(config):
    """Test that gyro measures constant angular velocity."""
    omega = 1.0
    sim = simulate_circular(config, angular_velocity=omega, seed=42)

    gyro = sim["U_imu"][:, 0]
    bias = sim["bias_gyro"]

    # Remove bias
    gyro_unbiased = gyro - bias

    # Mean should be close to omega
    mean_gyro = np.mean(gyro_unbiased)

    dt = 1.0 / config.fs_imu
    sigma = config.gyro_noise_density / np.sqrt(dt)

    assert np.abs(mean_gyro - omega) < 3 * sigma / np.sqrt(len(gyro))


def test_circular_accel_has_centripetal(config):
    """Test that accelerometer measures centripetal acceleration."""
    center = np.array([0.5, 0.5])
    radius = 0.3
    omega = 1.0

    sim = simulate_circular(
        config, center=center, radius=radius, angular_velocity=omega, seed=42
    )

    # World frame centripetal acceleration magnitude: a_c = r*ω²
    expected_accel_mag = radius * omega**2

    accel_world = sim["accel_world_truth"]
    accel_mag = np.sqrt(accel_world[:, 0] ** 2 + accel_world[:, 1] ** 2)

    assert np.allclose(accel_mag, expected_accel_mag, atol=1e-10)


def test_circular_camera_tracks_circle(config):
    """Test that camera measurements track circular motion."""
    center = np.array([0.5, 0.5])
    radius = 0.3
    omega = 1.0

    sim = simulate_circular(
        config, center=center, radius=radius, angular_velocity=omega, seed=42
    )

    Z = sim["Z_cam_led1"]
    mask = sim["mask_cam"]

    # Valid measurements should be near circle
    Z_valid = Z[mask]

    distance = np.sqrt(
        (Z_valid[:, 0] - center[0]) ** 2 + (Z_valid[:, 1] - center[1]) ** 2
    )

    # Should be close to radius (within a few sigma)
    mean_dist = np.mean(distance)
    assert np.abs(mean_dist - radius) < 3 * config.cam_noise_std


# =============================================================================
# Test: Data structure consistency
# =============================================================================


@pytest.mark.parametrize(
    "sim_func,kwargs",
    [
        (simulate_stationary, {}),
        (simulate_constant_velocity, {}),
        (simulate_circular, {}),
    ],
)
def test_array_shapes_consistent(config, sim_func, kwargs):
    """Test that all array shapes are consistent."""
    sim = sim_func(config, seed=42, **kwargs)

    T_imu = int(config.duration_s * config.fs_imu)
    T_cam = int(config.duration_s * config.fs_cam)

    assert sim["t_imu"].shape == (T_imu,)
    assert sim["t_cam_exp"].shape == (T_cam,)
    assert sim["t_cam_obs"].shape == (T_cam,)
    assert sim["X_truth"].shape == (T_imu, 5)
    assert sim["U_imu"].shape == (T_imu, 3)
    assert sim["Z_cam_led1"].shape == (T_cam, 2)
    assert sim["mask_cam"].shape == (T_cam,)
    assert sim["yaw_rate_truth"].shape == (T_imu,)
    assert sim["accel_world_truth"].shape == (T_imu, 2)
    assert sim["accel_body_truth"].shape == (T_imu, 2)


@pytest.mark.parametrize(
    "sim_func,kwargs",
    [
        (simulate_stationary, {}),
        (simulate_constant_velocity, {}),
        (simulate_circular, {}),
    ],
)
def test_time_vectors_monotonic(config, sim_func, kwargs):
    """Test that time vectors are strictly increasing."""
    sim = sim_func(config, seed=42, **kwargs)

    assert np.all(np.diff(sim["t_imu"]) > 0)
    assert np.all(np.diff(sim["t_cam_exp"]) > 0)


@pytest.mark.parametrize(
    "sim_func,kwargs",
    [
        (simulate_stationary, {}),
        (simulate_constant_velocity, {}),
        (simulate_circular, {}),
    ],
)
def test_mask_is_boolean(config, sim_func, kwargs):
    """Test that mask is boolean array."""
    sim = sim_func(config, seed=42, **kwargs)

    assert sim["mask_cam"].dtype == bool


@pytest.mark.parametrize(
    "sim_func,kwargs",
    [
        (simulate_stationary, {}),
        (simulate_constant_velocity, {}),
        (simulate_circular, {}),
    ],
)
def test_dropout_rate_approximately_correct(config, sim_func, kwargs):
    """Test that dropout rate matches configuration."""
    sim = sim_func(config, seed=42, **kwargs)

    mask = sim["mask_cam"]
    dropout_rate = 1 - mask.mean()

    # Should be within 3 sigma of expected (binomial distribution)
    n = len(mask)
    p = config.cam_dropout_prob
    sigma = np.sqrt(p * (1 - p) / n)

    assert np.abs(dropout_rate - p) < 3 * sigma


# =============================================================================
# Test: Reproducibility
# =============================================================================


@pytest.mark.parametrize(
    "sim_func,kwargs",
    [
        (simulate_stationary, {}),
        (simulate_constant_velocity, {}),
        (simulate_circular, {}),
    ],
)
def test_seed_reproducibility(config, sim_func, kwargs):
    """Test that same seed produces identical results."""
    sim1 = sim_func(config, seed=123, **kwargs)
    sim2 = sim_func(config, seed=123, **kwargs)

    assert np.allclose(sim1["X_truth"], sim2["X_truth"])
    assert np.allclose(sim1["U_imu"], sim2["U_imu"])
    # equal_nan=True since simple sims now NaN-out dropped frames in
    # Z_cam_led1 to match the rat_imu convention; the NaN positions are
    # deterministic from the seed and should match between runs.
    assert np.allclose(sim1["Z_cam_led1"], sim2["Z_cam_led1"], equal_nan=True)
    assert np.array_equal(sim1["mask_cam"], sim2["mask_cam"])


@pytest.mark.parametrize(
    "sim_func,kwargs",
    [
        (simulate_stationary, {}),
        (simulate_constant_velocity, {}),
        (simulate_circular, {}),
    ],
)
def test_different_seeds_produce_different_results(config, sim_func, kwargs):
    """Test that different seeds produce different noise realizations."""
    sim1 = sim_func(config, seed=123, **kwargs)
    sim2 = sim_func(config, seed=456, **kwargs)

    # Ground truth should be the same
    assert np.allclose(sim1["X_truth"], sim2["X_truth"])

    # But measurements should differ (due to noise). Compare only finite
    # samples — np.allclose returns False on NaN even when both arrays
    # have identical NaN positions, so the raw `not np.allclose` would
    # spuriously pass once the simple sims NaN-out dropped frames.
    assert not np.allclose(sim1["U_imu"], sim2["U_imu"])
    finite_both = np.isfinite(sim1["Z_cam_led1"]) & np.isfinite(sim2["Z_cam_led1"])
    assert finite_both.any(), "expected at least one finite-vs-finite LED1 sample"
    assert not np.allclose(
        sim1["Z_cam_led1"][finite_both], sim2["Z_cam_led1"][finite_both]
    )


def test_simple_short_duration_with_too_few_samples_is_rejected():
    """SimpleSimConfig must reject durations producing < 2 IMU/camera samples.

    The simple simulators compute counts as ``int(duration_s * fs_*)``;
    very small positive durations previously produced empty camera
    streams, which crashed downstream in prepare_video_data with an
    IndexError on nearest-neighbor indexing.
    """
    # 0.02s @ fs_cam=30 → 0 camera samples (was: empty camera stream).
    with pytest.raises(ValueError, match=r"need at least 2 of each"):
        SimpleSimConfig(duration_s=0.02, fs_imu=100.0, fs_cam=30.0)
    # 0.05s @ fs_cam=30 → 1 camera sample (still < 2).
    with pytest.raises(ValueError, match=r"need at least 2 of each"):
        SimpleSimConfig(duration_s=0.05, fs_imu=100.0, fs_cam=30.0)
    # 0.1s @ fs_cam=30 → 3 camera samples (boundary, must accept).
    cfg = SimpleSimConfig(duration_s=0.1, fs_imu=100.0, fs_cam=30.0)
    assert cfg.duration_s == 0.1

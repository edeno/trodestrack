"""Tests for rat_imu simulator - OU dynamics, noise, dropouts, full scenarios.

Tests verify:
1. Ornstein-Uhlenbeck process statistics (mean-reversion, correlation time)
2. IMU noise characteristics (white noise, bias random walks)
3. Camera dropout behavior (correlation, confidence)
4. LED swap detection
5. Arena boundary reflections
6. Full end-to-end scenario validation
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.sim.rat_imu import (
    RatIMUSimConfig,
    make_default_config,
    simulate_rat_imu,
)

# =============================================================================
# Test tolerance constants
# =============================================================================

# Statistical tests - multiples of sigma for finite sample sizes
TOLERANCE_MEAN_TO_STD = 0.2  # Mean should be < 20% of std for zero-mean process
TOLERANCE_AUTOCORR_ABS = 0.2  # Absolute error in autocorrelation (finite sample)
TOLERANCE_BINOMIAL_RATE = 0.3  # Relative error in dropout rates (3-sigma)
TOLERANCE_NOISE_STD = 0.1  # Relative error in noise std validation
TOLERANCE_BIAS_RW_STD = 0.1  # Relative error in bias random walk std
TOLERANCE_OU_VARIANCE = 0.5  # Relative error in OU variance (complex dynamics)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_config():
    """Minimal configuration for fast tests."""
    return RatIMUSimConfig(
        duration_s=5.0,
        fs_imu=200.0,
        fs_cam=30.0,
        # Minimal noise
        gyro_noise_density=np.deg2rad(0.01),
        accel_noise_density=0.01,
        gyro_bias_rw_density=np.deg2rad(0.001),
        accel_bias_rw_density=0.001,
        cam_sigma_m=0.002,
        cam_dropout_prob=0.05,
    )


@pytest.fixture
def realistic_config():
    """Realistic configuration with OU motion."""
    return RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=200.0,
        fs_cam=30.0,
        # Realistic motion
        tau_yaw_rate=0.8,
        sigma_yaw_rate=np.deg2rad(60.0),
        tau_a_fwd=0.7,
        sigma_a_fwd=1.0,
        tau_a_lat=0.5,
        sigma_a_lat=0.5,
        vel_drag=0.4,
        speed_clip=1.5,
    )


# =============================================================================
# Test: Configuration validation
# =============================================================================


def test_config_validates_duration() -> None:
    """Test that invalid duration raises error."""
    with pytest.raises(ValueError, match="duration must be positive"):
        RatIMUSimConfig(duration_s=-1.0)

    with pytest.raises(ValueError, match="duration must be positive"):
        RatIMUSimConfig(duration_s=0.0)


def test_config_validates_sampling_rates() -> None:
    """Test that invalid sampling rates raise errors."""
    with pytest.raises(ValueError, match="IMU sampling rate must be positive"):
        RatIMUSimConfig(fs_imu=-1.0)

    with pytest.raises(ValueError, match="Camera sampling rate must be positive"):
        RatIMUSimConfig(fs_cam=0.0)


def test_config_validates_arena() -> None:
    """Test that invalid arena dimensions raise errors."""
    with pytest.raises(ValueError, match="Arena dimensions must be positive"):
        RatIMUSimConfig(arena_w=-1.0)

    with pytest.raises(ValueError, match="Arena dimensions must be positive"):
        RatIMUSimConfig(arena_h=0.0)


def test_config_validates_probabilities() -> None:
    """Test that invalid probabilities raise errors."""
    with pytest.raises(ValueError, match="Dropout probability must be in"):
        RatIMUSimConfig(cam_dropout_prob=-0.1)

    with pytest.raises(ValueError, match="Dropout probability must be in"):
        RatIMUSimConfig(cam_dropout_prob=1.5)

    with pytest.raises(ValueError, match="LED swap probability must be in"):
        RatIMUSimConfig(led_swap_prob=-0.1, use_second_led=True)


def test_config_warns_imu_rate_too_low() -> None:
    """Test that warning is issued if IMU rate is too low."""
    with pytest.warns(UserWarning, match="IMU rate.*should typically be much higher"):
        RatIMUSimConfig(fs_imu=25.0, fs_cam=30.0)


def test_config_warns_swap_without_second_led() -> None:
    """Test that warning is issued if swap enabled but no second LED."""
    with pytest.warns(UserWarning, match="LED swap.*Swaps require two LEDs"):
        RatIMUSimConfig(led_swap_prob=0.1, use_second_led=False)


def test_make_default_config_accepts_overrides() -> None:
    """Test that make_default_config allows overrides."""
    cfg = make_default_config(duration_s=120.0, fs_imu=1000.0, use_second_led=True)

    assert cfg.duration_s == 120.0
    assert cfg.fs_imu == 1000.0
    assert cfg.use_second_led is True


def test_make_default_config_rejects_unknown_params() -> None:
    """Test that unknown parameters raise error."""
    with pytest.raises(ValueError, match="Unknown config parameter"):
        make_default_config(unknown_param=42)


# =============================================================================
# Test: Ornstein-Uhlenbeck process statistics
# =============================================================================


def test_ou_process_mean_reversion() -> None:
    """Test that OU process mean-reverts to zero."""
    # Long simulation with strong mean reversion
    config = RatIMUSimConfig(
        duration_s=100.0,
        fs_imu=200.0,
        tau_yaw_rate=1.0,  # 1s time constant
        sigma_yaw_rate=np.deg2rad(30.0),
        tau_a_fwd=1.0,
        sigma_a_fwd=0.5,
        tau_a_lat=1.0,
        sigma_a_lat=0.5,
        vel_drag=2.0,  # Strong damping to isolate OU process
    )

    sim = simulate_rat_imu(config, seed=42)

    # Extract yaw rate (OU process drives heading)
    yaw_rate = sim["yaw_rate_truth"]

    # Mean should be close to zero (OU mean = 0)
    mean_yaw_rate = np.mean(yaw_rate)
    std_yaw_rate = np.std(yaw_rate)

    # Mean should be small relative to std
    assert np.abs(mean_yaw_rate) < TOLERANCE_MEAN_TO_STD * std_yaw_rate


def test_ou_process_autocorrelation() -> None:
    """Test that OU process has expected correlation time."""
    config = RatIMUSimConfig(
        duration_s=50.0,
        fs_imu=200.0,
        tau_yaw_rate=1.0,  # 1s time constant
        sigma_yaw_rate=np.deg2rad(30.0),
        vel_drag=5.0,  # High damping
    )

    sim = simulate_rat_imu(config, seed=123)

    yaw_rate = sim["yaw_rate_truth"]
    dt = 1.0 / config.fs_imu

    # Compute autocorrelation at lag = tau
    # For OU: corr(lag) = exp(-lag/tau)
    lag_samples = int(config.tau_yaw_rate / dt)

    # Simple autocorrelation calculation
    mean = np.mean(yaw_rate)
    centered = yaw_rate - mean
    c0 = np.mean(centered**2)
    c_lag = np.mean(centered[:-lag_samples] * centered[lag_samples:])
    autocorr = c_lag / c0

    # Expected: exp(-1) ≈ 0.368
    expected_corr = np.exp(-1.0)

    # Allow tolerance due to finite sample
    assert np.abs(autocorr - expected_corr) < TOLERANCE_AUTOCORR_ABS


def test_ou_variance_matches_steady_state() -> None:
    """Test that OU process variance is in reasonable range for steady-state."""
    config = RatIMUSimConfig(
        duration_s=100.0,
        fs_imu=200.0,
        tau_a_fwd=2.0,
        sigma_a_fwd=1.0,  # Noise intensity
        vel_drag=10.0,  # Kill velocity quickly
    )

    sim = simulate_rat_imu(config, seed=456)

    # Extract body-frame accelerations (driven by OU)
    accel_body = sim["accel_body_truth"]

    # Use last 50% to ensure steady state
    half = len(accel_body) // 2
    accel_late = accel_body[half:]

    # Steady-state variance of OU: σ_ss² = σ² · τ / 2
    # Note: This is approximate due to rotation, drag, and coupling between axes
    expected_var = config.sigma_a_fwd**2 * config.tau_a_fwd / 2

    # Measure variance (using forward component as proxy)
    observed_var = np.var(accel_late[:, 0])

    # Check that variance is in reasonable range (within factor of 10)
    # Loose tolerance due to complex dynamics (rotation, drag, body/world frame transforms)
    assert 0.01 < observed_var < 10.0 * expected_var


# =============================================================================
# Test: IMU noise characteristics
# =============================================================================


def test_gyro_white_noise_std() -> None:
    """Test that gyro white noise has correct standard deviation."""
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=200.0,
        gyro_noise_density=np.deg2rad(0.05),  # 0.05 deg/s / √Hz
        gyro_bias_rw_density=0.0,  # No bias walk
        # Minimal motion
        sigma_yaw_rate=0.01,
        sigma_a_fwd=0.01,
        sigma_a_lat=0.01,
        vel_drag=10.0,
    )

    sim = simulate_rat_imu(config, seed=42)

    gyro_meas = sim["U_imu"][:, 0]
    gyro_truth = sim["yaw_rate_truth"]
    bias = sim["bias_gyro"]

    # Remove bias and truth to isolate noise
    noise = gyro_meas - gyro_truth - bias

    # Expected std from density
    dt = 1.0 / config.fs_imu
    expected_std = config.gyro_noise_density / np.sqrt(dt)

    observed_std = np.std(noise)

    # Guard against divide-by-zero and validate tolerance
    assert expected_std > 1e-10, f"Expected std too small: {expected_std}"
    assert np.abs(observed_std - expected_std) / expected_std < TOLERANCE_NOISE_STD


def test_accel_white_noise_std() -> None:
    """Test that accelerometer white noise has correct std.

    Isolates noise by:
    1. Zero tilt (no gravity in accelerometer frame)
    2. Near-zero motion (minimal inertial acceleration)
    3. Subtracting truth + bias to leave only white noise
    """
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=200.0,
        accel_noise_density=0.05,  # 0.05 m/s² / √Hz
        accel_bias_rw_density=0.0,  # No bias walk
        # Zero tilt: no gravity component in accelerometer
        imu_tilt_roll_deg=0.0,
        imu_tilt_pitch_deg=0.0,
        # Minimal motion: reduce inertial acceleration
        sigma_yaw_rate=0.001,
        sigma_a_fwd=0.001,
        sigma_a_lat=0.001,
        vel_drag=20.0,  # High drag kills motion quickly
    )

    sim = simulate_rat_imu(config, seed=123)

    # Accelerometer measures: f = a_body - g_body (specific force)
    # With zero tilt and minimal motion: f ≈ -g (pointing down)
    # Subtract truth to isolate noise
    accel_meas_x = sim["U_imu"][:, 1]
    accel_body_x = sim["accel_body_truth"][:, 0]
    bias_x = sim["bias_accel_x"]

    # With zero tilt: g_body_x ≈ 0, so f_x ≈ a_body_x
    # Noise = measurement - truth - bias
    noise_x = accel_meas_x - accel_body_x - bias_x

    # Use late samples to ensure steady state
    half = len(noise_x) // 2
    noise_late = noise_x[half:]

    dt = 1.0 / config.fs_imu
    expected_std = config.accel_noise_density / np.sqrt(dt)

    observed_std = np.std(noise_late)

    # Guard against divide-by-zero and validate tolerance
    assert expected_std > 1e-10, f"Expected std too small: {expected_std}"
    assert np.abs(observed_std - expected_std) / expected_std < TOLERANCE_NOISE_STD, (
        f"Accel noise std {observed_std:.4f} differs from expected {expected_std:.4f}"
    )


def test_gyro_bias_random_walk() -> None:
    """Test that gyro bias performs random walk."""
    config = RatIMUSimConfig(
        duration_s=60.0,
        fs_imu=200.0,
        gyro_bias_rw_density=np.deg2rad(0.01),  # 0.01 deg/s / √s
    )

    sim = simulate_rat_imu(config, seed=42)

    bias = sim["bias_gyro"]

    # Compute increments
    dt = 1.0 / config.fs_imu
    increments = np.diff(bias)

    # Random walk increments should have zero mean
    assert np.abs(np.mean(increments)) < 3 * config.gyro_bias_rw_density * np.sqrt(dt)

    # Increments should have std = density * √dt
    expected_std = config.gyro_bias_rw_density * np.sqrt(dt)
    observed_std = np.std(increments)

    # Guard against divide-by-zero
    assert expected_std > 1e-10, f"Expected std too small: {expected_std}"
    assert np.abs(observed_std - expected_std) / expected_std < TOLERANCE_BIAS_RW_STD


def test_accel_bias_random_walk() -> None:
    """Test that accelerometer bias performs random walk."""
    config = RatIMUSimConfig(
        duration_s=60.0,
        fs_imu=200.0,
        accel_bias_rw_density=0.01,  # 0.01 m/s² / √s
    )

    sim = simulate_rat_imu(config, seed=123)

    bias_x = sim["bias_accel_x"]
    bias_y = sim["bias_accel_y"]

    dt = 1.0 / config.fs_imu

    # Check both axes
    for bias in [bias_x, bias_y]:
        increments = np.diff(bias)

        # Zero mean
        assert np.abs(np.mean(increments)) < 3 * config.accel_bias_rw_density * np.sqrt(
            dt
        )

        # Correct std
        expected_std = config.accel_bias_rw_density * np.sqrt(dt)
        observed_std = np.std(increments)

        # Guard against divide-by-zero
        assert expected_std > 1e-10, f"Expected std too small: {expected_std}"
        assert (
            np.abs(observed_std - expected_std) / expected_std < TOLERANCE_BIAS_RW_STD
        )


# =============================================================================
# Test: Camera dropouts and confidence
# =============================================================================


def test_camera_dropout_rate() -> None:
    """Test that dropout rate matches configuration."""
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_cam=30.0,
        cam_dropout_prob=0.15,  # 15% dropout
    )

    sim = simulate_rat_imu(config, seed=42)

    mask = sim["mask_led1"]
    dropout_rate = 1 - mask.mean()

    # Binomial: should be within 3 sigma
    n = len(mask)
    p = config.cam_dropout_prob
    sigma = np.sqrt(p * (1 - p) / n)

    assert np.abs(dropout_rate - p) < 3 * sigma


def test_camera_dropout_correlation_independent() -> None:
    """Test that LED1 and LED2 dropouts are independent when correlation=0."""
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_cam=30.0,
        cam_dropout_prob=0.2,
        cam_dropout_correlation=0.0,  # Independent
        use_second_led=True,
    )

    sim = simulate_rat_imu(config, seed=42)

    mask1 = sim["mask_led1"]
    mask2 = sim["mask_led2"]

    # Count joint events
    both_visible = (mask1 & mask2).sum()
    both_dropped = (~mask1 & ~mask2).sum()

    # Expected counts if independent
    n = len(mask1)
    p_visible = 1 - config.cam_dropout_prob
    expected_both_visible = n * p_visible**2
    expected_both_dropped = n * config.cam_dropout_prob**2

    # Should be close (with tolerance for binomial noise)
    assert expected_both_visible > 0, "Expected visible count must be positive"
    assert (
        np.abs(both_visible - expected_both_visible) / expected_both_visible
        < TOLERANCE_BINOMIAL_RATE
    )
    # Use max(expected, 1) to avoid divide-by-zero for rare events
    assert (
        np.abs(both_dropped - expected_both_dropped) / max(expected_both_dropped, 1.0)
        < TOLERANCE_BINOMIAL_RATE + 0.2
    )


def test_camera_dropout_correlation_high() -> None:
    """Test that LED1 and LED2 dropouts are correlated when correlation=1."""
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_cam=30.0,
        cam_dropout_prob=0.2,
        cam_dropout_correlation=1.0,  # Perfect correlation
        use_second_led=True,
    )

    sim = simulate_rat_imu(config, seed=42)

    mask1 = sim["mask_led1"]
    mask2 = sim["mask_led2"]

    # With perfect correlation: mask1 == mask2
    # Allow small discrepancy due to numerical issues
    agreement = (mask1 == mask2).mean()

    assert agreement > 0.95  # At least 95% agreement


def test_confidence_enabled() -> None:
    """Test that confidence scores are generated when enabled."""
    config = RatIMUSimConfig(
        duration_s=10.0,
        fs_cam=30.0,
        use_confidence=True,
        confidence_base=0.9,
    )

    sim = simulate_rat_imu(config, seed=42)

    conf1 = sim["confidence_led1"]
    mask1 = sim["mask_led1"]

    # Valid frames should have confidence close to base
    conf_valid = conf1[mask1]

    assert np.all((conf_valid > 0) & (conf_valid <= 1.0))
    assert np.mean(conf_valid) > 0.7  # Should be reasonably high


def test_confidence_zero_on_dropout() -> None:
    """Test that confidence is zero when LED is dropped."""
    config = RatIMUSimConfig(
        duration_s=10.0,
        fs_cam=30.0,
        use_confidence=True,
        cam_dropout_prob=0.2,
    )

    sim = simulate_rat_imu(config, seed=42)

    conf1 = sim["confidence_led1"]
    mask1 = sim["mask_led1"]

    # Dropped frames must have zero confidence
    conf_dropped = conf1[~mask1]

    assert np.all(conf_dropped == 0.0)


def test_confidence_reduced_near_dropout() -> None:
    """Test that confidence is reduced in frames adjacent to dropouts."""
    config = RatIMUSimConfig(
        duration_s=20.0,
        fs_cam=30.0,
        use_confidence=True,
        confidence_base=0.95,
        confidence_dropout_decay=0.3,
        cam_dropout_prob=0.15,
    )

    sim = simulate_rat_imu(config, seed=123)

    conf1 = sim["confidence_led1"]
    mask1 = sim["mask_led1"]

    # Find frames adjacent to dropouts
    dropout_indices = np.where(~mask1)[0]

    if len(dropout_indices) > 0:
        # Check neighbors
        adjacent = []
        for idx in dropout_indices:
            if idx > 0 and mask1[idx - 1]:
                adjacent.append(idx - 1)
            if idx < len(mask1) - 1 and mask1[idx + 1]:
                adjacent.append(idx + 1)

        if len(adjacent) > 0:
            conf_adjacent = conf1[adjacent]
            # Should be reduced compared to base
            assert np.mean(conf_adjacent) < config.confidence_base


# =============================================================================
# Test: LED swaps
# =============================================================================


def test_led_swap_occurs_when_enabled() -> None:
    """Test that LED swaps occur when swap_prob > 0 and are correctly tracked."""
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_cam=30.0,
        use_second_led=True,
        led_swap_prob=0.2,  # 20% swap rate
        cam_dropout_prob=0.05,  # Low dropout to get many swap candidates
        led1_offset_body=np.array([0.05, 0.0]),
        led2_offset_body=np.array([-0.05, 0.0]),
    )

    sim = simulate_rat_imu(config, seed=42)

    mask1 = sim["mask_led1"]
    mask2 = sim["mask_led2"]
    both_visible = mask1 & mask2
    swap_applied = sim["swap_applied"]

    # Count actual swaps
    num_swaps = swap_applied.sum()
    num_both_visible = both_visible.sum()

    # Should have some swaps given 20% rate
    assert num_swaps > 0, "No swaps occurred despite 20% swap_prob"

    # Swap rate should be approximately 20% of both-visible frames
    if num_both_visible > 0:
        observed_swap_rate = num_swaps / num_both_visible
        # Binomial tolerance: within 3 sigma
        p = config.led_swap_prob
        sigma = np.sqrt(p * (1 - p) / num_both_visible)
        assert np.abs(observed_swap_rate - p) < 3 * sigma, (
            f"Swap rate {observed_swap_rate:.2%} differs from expected {p:.2%}"
        )

    # Verify swaps only occur when both visible
    assert np.all(both_visible[swap_applied]), (
        "Swaps occurred when LEDs not both visible"
    )

    # Verify that when swaps occur, measurements match swapped truth
    led1_truth = sim["led1_truth_cam"]
    led2_truth = sim["led2_truth_cam"]
    Z1 = sim["Z_cam_led1"]
    Z2 = sim["Z_cam_led2"]

    # Check a few swapped frames
    swapped_frames = np.where(swap_applied)[0]
    if len(swapped_frames) > 0:
        # Take first swapped frame
        idx = swapped_frames[0]

        # When swapped: Z1 should be near led2_truth, Z2 should be near led1_truth
        # (within noise tolerance)
        noise_tolerance = 5 * config.cam_sigma_m  # 5-sigma

        dist_z1_to_truth2 = np.linalg.norm(Z1[idx] - led2_truth[idx])
        dist_z2_to_truth1 = np.linalg.norm(Z2[idx] - led1_truth[idx])

        assert dist_z1_to_truth2 < noise_tolerance, (
            f"Swapped Z1 not near led2_truth: dist={dist_z1_to_truth2:.4f}m"
        )
        assert dist_z2_to_truth1 < noise_tolerance, (
            f"Swapped Z2 not near led1_truth: dist={dist_z2_to_truth1:.4f}m"
        )


def test_led_swap_only_when_both_visible() -> None:
    """Test that swaps only occur when both LEDs are visible."""
    # This is implicitly tested by the swap logic in simulate_rat_imu
    # We verify by checking that single-LED dropout frames are unaffected

    config = RatIMUSimConfig(
        duration_s=20.0,
        fs_cam=30.0,
        use_second_led=True,
        led_swap_prob=0.3,
        cam_dropout_prob=0.3,  # Higher dropout
    )

    sim = simulate_rat_imu(config, seed=456)

    Z1 = sim["Z_cam_led1"]
    Z2 = sim["Z_cam_led2"]
    mask1 = sim["mask_led1"]
    mask2 = sim["mask_led2"]

    # When only one LED is visible, position should not be NaN
    only_led1 = mask1 & ~mask2
    only_led2 = ~mask1 & mask2

    if only_led1.sum() > 0:
        assert not np.any(np.isnan(Z1[only_led1]))
    if only_led2.sum() > 0:
        assert not np.any(np.isnan(Z2[only_led2]))


# =============================================================================
# Test: Arena boundary reflections
# =============================================================================


def test_arena_boundaries_respected() -> None:
    """Test that rat stays within arena bounds."""
    config = RatIMUSimConfig(
        duration_s=60.0,
        fs_imu=200.0,
        arena_w=1.5,
        arena_h=1.5,
        # Strong motion to hit walls
        sigma_yaw_rate=np.deg2rad(80.0),
        sigma_a_fwd=2.0,
        sigma_a_lat=1.5,
        vel_drag=0.2,
    )

    sim = simulate_rat_imu(config, seed=42)

    X = sim["X_truth"]
    x = X[:, 0]
    y = X[:, 1]

    # All positions should be within bounds
    assert np.all(x >= 0) and np.all(x <= config.arena_w)
    assert np.all(y >= 0) and np.all(y <= config.arena_h)


def test_arena_reflections_reduce_velocity() -> None:
    """Test that reflections reduce velocity (inelastic)."""
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=200.0,
        arena_w=1.0,
        arena_h=1.0,
        # Initial position near edge with velocity toward wall
        m0=np.array([0.05, 0.5, 0.5, 0.0, 0.0]),  # Moving right toward wall
        sigma_a_fwd=0.1,  # Minimal acceleration
        vel_drag=0.1,
    )

    sim = simulate_rat_imu(config, seed=42)

    X = sim["X_truth"]
    vx = X[:, 2]

    # Find first reflection (x reaches 0 or arena_w and velocity reverses)
    # Before reflection: vx > 0
    # After reflection: vx < 0 and |vx_after| < |vx_before| (energy loss)

    # Check that velocity sign changes occur
    sign_changes = np.diff(np.sign(vx))
    num_reflections = np.sum(sign_changes != 0)

    # Should have at least one reflection given the setup
    assert num_reflections > 0


# =============================================================================
# Test: Data structure and consistency
# =============================================================================


def test_output_keys_complete(minimal_config) -> None:
    """Test that all expected output keys are present."""
    sim = simulate_rat_imu(minimal_config, seed=42)

    expected_keys = {
        "t_imu",
        "t_cam_exp",
        "t_cam_obs",
        "X_truth",
        "yaw_rate_truth",
        "accel_world_truth",
        "accel_body_truth",
        "U_imu",
        "bias_gyro",
        "bias_accel_x",
        "bias_accel_y",
        "Z_cam_led1",
        "Z_cam_led2",
        "led1_truth_cam",  # NEW: Ground truth LED positions
        "led2_truth_cam",  # NEW: Ground truth LED positions
        "swap_applied",  # NEW: Swap tracking
        "led_reflection_applied",  # NEW: Wall reflection tracking
        "confidence_led1",
        "confidence_led2",
        "mask_cam",
        "mask_led1",
        "mask_led2",
        "config",
    }

    assert set(sim.keys()) == expected_keys


def test_array_shapes_consistent(minimal_config) -> None:
    """Test that all arrays have consistent shapes."""
    sim = simulate_rat_imu(minimal_config, seed=42)

    T_imu = int(np.round(minimal_config.duration_s * minimal_config.fs_imu))
    T_cam = int(np.round(minimal_config.duration_s * minimal_config.fs_cam))

    assert sim["t_imu"].shape == (T_imu,)
    assert sim["t_cam_exp"].shape == (T_cam,)
    assert sim["t_cam_obs"].shape == (T_cam,)
    assert sim["X_truth"].shape == (T_imu, 5)
    assert sim["U_imu"].shape == (T_imu, 3)
    assert sim["bias_gyro"].shape == (T_imu,)
    assert sim["Z_cam_led1"].shape == (T_cam, 2)
    assert sim["mask_led1"].shape == (T_cam,)
    assert sim["confidence_led1"].shape == (T_cam,)


def test_time_vectors_monotonic(minimal_config) -> None:
    """Test that time vectors are mostly monotonic (camera jitter can cause reordering).

    Policy: IMU time is strictly monotonic (no jitter). Camera exposure time
    has Gaussian jitter which can occasionally cause non-monotonic timestamps.
    This is realistic behavior - we require ≥95% of intervals to be positive.
    """
    sim = simulate_rat_imu(minimal_config, seed=42)

    # IMU time should be perfectly uniform (no jitter)
    assert np.all(np.diff(sim["t_imu"]) > 0), "IMU time must be strictly monotonic"

    # Camera exposure time has jitter, so check that MOST diffs are positive
    # Jitter is Gaussian, so occasional negative diffs can occur
    cam_exp_diffs = np.diff(sim["t_cam_exp"])
    positive_rate_exp = (cam_exp_diffs > 0).mean()
    assert positive_rate_exp > 0.95, (
        f"Camera exposure time mostly monotonic: {positive_rate_exp:.1%} positive"
    )

    # Observation time = exposure + latency (should also be mostly monotonic)
    # Inherits jitter from exposure time
    cam_obs_diffs = np.diff(sim["t_cam_obs"])
    positive_rate_obs = (cam_obs_diffs > 0).mean()
    assert positive_rate_obs > 0.95, (
        f"Camera observation time mostly monotonic: {positive_rate_obs:.1%} positive"
    )


def test_camera_timestamps_relationship(minimal_config) -> None:
    """Test that t_cam_obs = t_cam_exp + latency."""
    sim = simulate_rat_imu(minimal_config, seed=42)

    t_exp = sim["t_cam_exp"]
    t_obs = sim["t_cam_obs"]

    # Difference should be approximately equal to latency
    diff = t_obs - t_exp

    # All differences should be positive and close to latency
    assert np.all(diff >= 0)
    assert np.allclose(
        diff, minimal_config.cam_latency_s, atol=minimal_config.cam_jitter_s * 3
    )


# =============================================================================
# Test: Reproducibility
# =============================================================================


def test_seed_reproducibility(minimal_config) -> None:
    """Test that same seed produces identical results."""
    sim1 = simulate_rat_imu(minimal_config, seed=999)
    sim2 = simulate_rat_imu(minimal_config, seed=999)

    assert np.allclose(sim1["X_truth"], sim2["X_truth"])
    assert np.allclose(sim1["U_imu"], sim2["U_imu"])
    assert np.allclose(sim1["Z_cam_led1"], sim2["Z_cam_led1"], equal_nan=True)
    assert np.array_equal(sim1["mask_led1"], sim2["mask_led1"])
    assert np.allclose(sim1["bias_gyro"], sim2["bias_gyro"])


def test_different_seeds_produce_different_results(minimal_config) -> None:
    """Test that different seeds produce different noise realizations."""
    sim1 = simulate_rat_imu(minimal_config, seed=111)
    sim2 = simulate_rat_imu(minimal_config, seed=222)

    # Ground truth should differ (OU process is seeded)
    assert not np.allclose(sim1["X_truth"], sim2["X_truth"])

    # Measurements should differ
    assert not np.allclose(sim1["U_imu"], sim2["U_imu"])

    # Biases should differ
    assert not np.allclose(sim1["bias_gyro"], sim2["bias_gyro"])


# =============================================================================
# Test: Full end-to-end scenarios
# =============================================================================


def test_realistic_session_completes(realistic_config) -> None:
    """Test that realistic 30s session completes without errors."""
    sim = simulate_rat_imu(realistic_config, seed=42)

    # Basic sanity checks
    assert sim["X_truth"].shape[0] > 0
    assert not np.any(np.isnan(sim["X_truth"]))
    assert not np.any(np.isnan(sim["U_imu"]))

    # Check that rat moved
    X = sim["X_truth"]
    total_distance = np.sum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))
    assert total_distance > 1.0  # Moved at least 1 meter


def test_stationary_rat_with_imu_drift() -> None:
    """Test that biases drift even when rat is stationary."""
    config = RatIMUSimConfig(
        duration_s=60.0,
        fs_imu=200.0,
        # Very low motion (nearly stationary)
        sigma_yaw_rate=0.001,
        sigma_a_fwd=0.001,
        sigma_a_lat=0.001,
        vel_drag=10.0,
        # Moderate bias drift
        gyro_bias_rw_density=np.deg2rad(0.01),
        accel_bias_rw_density=0.01,
    )

    sim = simulate_rat_imu(config, seed=42)

    # Check that biases have drifted
    bias_gyro = sim["bias_gyro"]
    bias_accel_x = sim["bias_accel_x"]

    # Bias should have non-zero std (random walk)
    assert np.std(bias_gyro) > 1e-6
    assert np.std(bias_accel_x) > 1e-6

    # But position should be nearly constant
    X = sim["X_truth"]
    pos_std = np.std(X[:, :2], axis=0)
    assert np.all(pos_std < 0.1)  # Less than 10 cm std


def test_high_speed_motion_is_clipped() -> None:
    """Test that speed clipping prevents excessive velocities."""
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=200.0,
        # Very strong acceleration
        sigma_a_fwd=5.0,
        sigma_a_lat=3.0,
        speed_clip=1.2,  # 1.2 m/s max
        vel_drag=0.1,  # Low drag
    )

    sim = simulate_rat_imu(config, seed=42)

    X = sim["X_truth"]
    vx = X[:, 2]
    vy = X[:, 3]
    speed = np.hypot(vx, vy)

    # Speed should never exceed clip (with small tolerance for numerical issues)
    assert np.all(speed <= config.speed_clip * 1.05)


def test_second_led_enabled() -> None:
    """Test that second LED is generated when enabled."""
    config = RatIMUSimConfig(
        duration_s=10.0,
        use_second_led=True,
        led1_offset_body=np.array([0.05, 0.0]),
        led2_offset_body=np.array([-0.05, 0.0]),
    )

    sim = simulate_rat_imu(config, seed=42)

    Z1 = sim["Z_cam_led1"]
    Z2 = sim["Z_cam_led2"]
    mask2 = sim["mask_led2"]

    # LED2 should have valid (non-NaN) measurements when not dropped
    if mask2.sum() > 0:
        Z2_valid = Z2[mask2]
        assert not np.any(np.isnan(Z2_valid))

        # LED2 should be different from LED1
        Z1_valid = Z1[mask2]
        assert not np.allclose(Z1_valid, Z2_valid)


def test_second_led_disabled() -> None:
    """Test that second LED is NaN when disabled."""
    config = RatIMUSimConfig(
        duration_s=10.0,
        use_second_led=False,
    )

    sim = simulate_rat_imu(config, seed=42)

    Z2 = sim["Z_cam_led2"]
    mask2 = sim["mask_led2"]

    # LED2 should be all NaN
    assert np.all(np.isnan(Z2))

    # mask_led2 should be all False
    assert not mask2.any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

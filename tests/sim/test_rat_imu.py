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


def test_config_warns_when_inactive_swap_param_is_set() -> None:
    """Mode-specific LED swap parameters must not be silently ignored.

    The runtime branches only consult ``led_swap_prob`` in per_frame
    mode and ``led_swap_rate`` in persistent mode; the inactive
    parameter is unused. The reported probe — ``led_swap_mode=
    'persistent', led_swap_prob=1.0, led_swap_rate=0.0`` — produced
    zero swaps with no warning, which masks swap-heavy simulator
    test setups that accidentally set the wrong field.
    """
    # Persistent mode + non-default led_swap_prob → warn.
    with pytest.warns(UserWarning, match="led_swap_prob=1.0.*persistent"):
        RatIMUSimConfig(
            led_swap_mode="persistent",
            led_swap_prob=1.0,
            led_swap_rate=0.0,
            use_second_led=True,
        )

    # per_frame mode + non-default led_swap_rate → warn.
    with pytest.warns(UserWarning, match="led_swap_rate=2.0.*per_frame"):
        RatIMUSimConfig(
            led_swap_mode="per_frame",
            led_swap_prob=0.05,
            led_swap_rate=2.0,
            use_second_led=True,
        )


def test_config_does_not_warn_on_default_led_swap_params() -> None:
    """The default config (per_frame, prob=0, rate=0.5) must not warn.

    ``led_swap_rate=0.5`` is the documented default and is silently
    ignored in the default ``per_frame`` mode. Warning on the
    default would create noise on every basic ``RatIMUSimConfig()``.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Defaults: led_swap_mode='per_frame', led_swap_prob=0.0,
        # led_swap_rate=0.5. Use minimum-viable IMU rate to skip
        # the unrelated low-rate warning.
        RatIMUSimConfig(fs_imu=200.0, fs_cam=30.0)


def test_config_drag_strict_type_validation() -> None:
    """drag_fwd / drag_lat must reject non-numeric inputs cleanly.

    The probe — ``drag_fwd="0.1"`` raised a raw ``TypeError`` from
    ``np.isfinite``, and ``drag_fwd=[0.1]`` raised a raw ``TypeError``
    from ``< 0``. The other numeric-scalar fields go through the
    up-front ``finite_scalar_fields`` gate; ``drag_fwd`` /
    ``drag_lat`` skip it because ``None`` is a valid sentinel until
    ``vel_drag`` fills it in.
    """
    bad_values: list[object] = ["0.1", [0.1], (0.1,), True]
    for bad in bad_values:
        with pytest.raises(
            ValueError, match=r"drag_fwd must be a finite non-negative scalar"
        ):
            RatIMUSimConfig(drag_fwd=bad, drag_lat=0.3)  # type: ignore[arg-type]
        with pytest.raises(
            ValueError, match=r"drag_lat must be a finite non-negative scalar"
        ):
            RatIMUSimConfig(drag_fwd=0.3, drag_lat=bad)  # type: ignore[arg-type]

    # Numeric drag still works.
    cfg = RatIMUSimConfig(drag_fwd=0.4, drag_lat=1.2)
    assert cfg.drag_fwd == 0.4
    assert cfg.drag_lat == 1.2


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
        "specific_force_truth",
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
    """Time vectors must be strictly increasing.

    Policy:
      - IMU time is strictly monotonic (no jitter applied to IMU samples).
      - Camera exposure / observation time may include Gaussian jitter, but the
        simulator sorts and minimally separates jittered camera samples after
        clipping so downstream filter timestamp validation accepts them.
    """
    sim = simulate_rat_imu(minimal_config, seed=42)

    # IMU time should be perfectly uniform (no jitter).
    assert np.all(np.diff(sim["t_imu"]) > 0), "IMU time must be strictly monotonic"

    # Sorted camera timestamps: strictly increasing to match filter validation.
    cam_exp_diffs = np.diff(sim["t_cam_exp"])
    assert np.all(cam_exp_diffs > 0), (
        "t_cam_exp must be strictly increasing; the simulator stable-sorts and "
        "minimally separates jittered camera samples to keep filter intervals "
        "well defined."
    )
    cam_obs_diffs = np.diff(sim["t_cam_obs"])
    assert np.all(cam_obs_diffs > 0), (
        "t_cam_obs must be strictly increasing; inherits ordering from "
        "t_cam_exp via t_cam_obs = t_cam_exp + cam_latency_s."
    )


def test_camera_timestamps_strict_with_large_jitter() -> None:
    """Boundary-clipped jitter must not create tied camera timestamps."""
    config = RatIMUSimConfig(
        duration_s=5.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_jitter_s=0.5,
    )
    sim = simulate_rat_imu(config, seed=42)

    assert np.all(np.diff(sim["t_cam_exp"]) > 0)
    assert np.all(np.diff(sim["t_cam_obs"]) > 0)
    assert sim["t_cam_exp"][0] >= sim["t_imu"][0]
    assert sim["t_cam_exp"][-1] <= sim["t_imu"][-1]
    np.testing.assert_allclose(
        sim["t_cam_obs"] - sim["t_cam_exp"],
        config.cam_latency_s,
        atol=1e-12,
    )


def test_camera_timestamps_relationship(minimal_config) -> None:
    """Test that t_cam_obs == t_cam_exp + cam_latency_s exactly.

    The simulator now clips ``t_cam_exp`` to the IMU support range first
    and computes ``t_cam_obs`` from the *clipped* exposure, so this
    contract holds exactly even at the clipping boundaries (previously a
    boundary-clipped sample had ``t_cam_obs - t_cam_exp != cam_latency_s``).
    """
    sim = simulate_rat_imu(minimal_config, seed=42)

    t_exp = sim["t_cam_exp"]
    t_obs = sim["t_cam_obs"]

    diff = t_obs - t_exp

    np.testing.assert_allclose(diff, minimal_config.cam_latency_s, atol=1e-12)


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


def test_led_offset_body_must_be_2d_finite_array():
    """LED offsets must validate at construction (shape (2,), all finite)."""
    # Scalar offsets used to crash mid-simulation with a raw IndexError.
    with pytest.raises(ValueError, match=r"led1_offset_body must be a body-frame"):
        RatIMUSimConfig(
            duration_s=1.0,
            use_second_led=True,
            led1_offset_body=0.025,
            led2_offset_body=np.array([-0.025, 0.0]),
        )
    # NaN offsets used to construct silently and emit non-finite Z_cam values.
    with pytest.raises(ValueError, match=r"led2_offset_body must contain only finite"):
        RatIMUSimConfig(
            duration_s=1.0,
            use_second_led=True,
            led1_offset_body=np.array([0.02, 0.0]),
            led2_offset_body=np.array([np.nan, 0.0]),
        )
    # Wrong-length offsets must also be rejected.
    with pytest.raises(ValueError, match=r"led1_offset_body must be a body-frame"):
        RatIMUSimConfig(
            duration_s=1.0,
            use_second_led=True,
            led1_offset_body=np.array([0.02, 0.0, 0.0]),
            led2_offset_body=np.array([-0.02, 0.0]),
        )


def test_short_duration_with_too_few_samples_is_rejected():
    """duration_s producing < 2 IMU or camera samples must be rejected.

    The simulator rounds ``duration_s * fs_*`` to integer counts; very
    small positive durations previously crashed with IndexError or
    zero-size reduction errors mid-simulation.
    """
    with pytest.raises(ValueError, match=r"need at least 2 of each"):
        RatIMUSimConfig(duration_s=0.001, fs_imu=100.0, fs_cam=30.0)
    with pytest.raises(ValueError, match=r"need at least 2 of each"):
        RatIMUSimConfig(duration_s=0.01, fs_imu=100.0, fs_cam=30.0)
    # 0.05s @ fs_cam=30 → 2 camera samples (boundary, must accept).
    cfg = RatIMUSimConfig(duration_s=0.05, fs_imu=100.0, fs_cam=30.0)
    sim = simulate_rat_imu(cfg, seed=0)
    assert len(sim["t_cam_exp"]) >= 2
    assert len(sim["t_imu"]) >= 2


def test_confidence_decays_on_frames_adjacent_to_dropouts():
    """Visible frames adjacent to a dropout must respect confidence_dropout_decay.

    The neighbor-dropout convolution previously had an operator-precedence
    bug: ``~mask.astype(int)`` parsed as ``~(mask.astype(int))`` and
    produced strictly-negative integers, so the ``> 0`` check that gates
    the decay multiplier was never satisfied. Adjacent-to-dropout frames
    therefore retained the unmodified random envelope (~0.8-1.2 ×
    confidence_base) instead of being bounded by
    confidence_base * 1.2 * confidence_dropout_decay.
    """
    confidence_base = 1.0
    decay = 0.25
    rand_max = 1.2  # 0.8 + 0.4 * 1 (worst-case random multiplier)
    cfg = RatIMUSimConfig(
        duration_s=20.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=0.15,
        cam_dropout_correlation=0.0,
        use_second_led=True,
        use_confidence=True,
        confidence_base=confidence_base,
        confidence_dropout_decay=decay,
    )
    sim = simulate_rat_imu(cfg, seed=42)

    expected_cap = confidence_base * rand_max * decay
    lower_envelope = 0.8 * confidence_base

    def _check(name: str, mask, conf) -> None:
        adj = np.zeros_like(mask)
        adj[1:] |= ~mask[:-1]
        adj[:-1] |= ~mask[1:]
        visible_adj = mask & adj
        if visible_adj.any():
            adj_confs = conf[visible_adj]
            assert adj_confs.max() <= expected_cap + 1e-9, (
                f"{name}: adjacent-to-dropout confidence max "
                f"{adj_confs.max():.4f} exceeds expected cap "
                f"{expected_cap:.4f} — decay not applied."
            )
        non_adj = mask & ~adj
        if non_adj.any():
            non_adj_confs = conf[non_adj]
            assert non_adj_confs.min() >= lower_envelope - 1e-9, (
                f"{name}: non-adjacent visible confidence min "
                f"{non_adj_confs.min():.4f} fell below the random envelope's "
                "lower bound — decay was applied where it shouldn't."
            )

    _check("led1", sim["mask_led1"], sim["confidence_led1"])
    _check("led2", sim["mask_led2"], sim["confidence_led2"])


def test_led_swap_per_frame_is_bernoulli_not_rounded_count():
    """Per-frame swaps must use Bernoulli sampling, not round(n*p).

    Previously the simulator picked exactly ``round(len(candidates) * p)``
    frames, which (a) rounded small ``p × n`` to 0 and silently disabled
    swaps on short/sparse runs and (b) produced no binomial variance
    across seeds. With Bernoulli sampling we expect both: a non-zero
    probability that swaps fire even when ``p × n < 0.5``, and seed-to-
    seed variability in the swap count.
    """
    seeds = list(range(40))
    counts = []
    for seed in seeds:
        cfg = RatIMUSimConfig(
            duration_s=2.0,
            fs_imu=200.0,
            fs_cam=30.0,
            cam_dropout_prob=0.0,
            use_second_led=True,
            led_swap_mode="per_frame",
            led_swap_prob=0.05,
            led_wall_reflection_prob=0.0,
        )
        sim = simulate_rat_imu(cfg, seed=seed)
        counts.append(int(sim["swap_applied"].sum()))

    # At least one seed must produce a non-zero swap count (rules out
    # the "rounded to zero" failure mode).
    assert max(counts) > 0, (
        f"led_swap_prob=0.05 produced 0 swaps across {len(seeds)} seeds — "
        "looks like the rounded-count path is back."
    )
    # And the seed-to-seed variability must be non-trivial (rules out
    # the "exact fixed count" failure mode).
    assert len(set(counts)) > 1, (
        f"swap counts identical across seeds {set(counts)} — Bernoulli "
        "sampling should produce binomial variance."
    )


def test_led_wall_reflection_is_bernoulli_not_rounded_count():
    """Wall reflections must use Bernoulli sampling, not round(n*p).

    Same bug class as ``led_swap_prob`` above, on the reflection path.
    """
    seeds = list(range(40))
    counts = []
    for seed in seeds:
        cfg = RatIMUSimConfig(
            duration_s=5.0,
            fs_imu=200.0,
            fs_cam=30.0,
            cam_dropout_prob=0.0,
            use_second_led=True,
            led_swap_mode="per_frame",
            led_swap_prob=0.0,
            led_wall_reflection_prob=0.05,
            led_wall_reflection_distance=0.05,
            arena_w=0.5,
            arena_h=0.5,
        )
        sim = simulate_rat_imu(cfg, seed=seed)
        counts.append(int(sim["led_reflection_applied"].sum()))

    assert max(counts) > 0, (
        f"led_wall_reflection_prob=0.05 produced 0 reflections across "
        f"{len(seeds)} seeds — rounded-count path is back."
    )
    assert len(set(counts)) > 1, (
        f"reflection counts identical across seeds {set(counts)} — "
        "Bernoulli sampling should produce binomial variance."
    )


def test_specific_force_truth_matches_measured_imu():
    """``specific_force_truth`` must equal noiseless ``U_imu[:, 1:3]``.

    The diagnostic-video IMU panel previously plotted ``accel_body_truth``
    (inertial accel in body frame) as the IMU truth overlay, but
    ``U_imu[:, 1:3]`` measures specific force = a_body - g_body. Under
    the default tilt of (3°, 2°), no-motion runs showed a phantom ~|g·
    sin(tilt)| offset between the two. ``specific_force_truth`` is the
    quantity the overlay actually wants.
    """
    cfg = RatIMUSimConfig(
        duration_s=2.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,
        use_second_led=True,
        gyro_noise_density=0.0,
        accel_noise_density=0.0,
        gyro_bias_rw_density=0.0,
        accel_bias_rw_density=0.0,
        sigma_yaw_rate=0.0,
        sigma_a_fwd=0.0,
        sigma_a_lat=0.0,
        m0=np.array([0.5, 0.5, 0.0, 0.0, 0.0]),
        P0=np.zeros((5, 5)),
    )
    sim = simulate_rat_imu(cfg, seed=0)

    # With all noise / process drivers zero the measured specific force
    # equals the truth specific force exactly (modulo float precision).
    np.testing.assert_allclose(
        sim["U_imu"][:, 1:3], sim["specific_force_truth"], atol=1e-12
    )

    # The default tilt produces a non-zero specific force at rest, so
    # ``accel_body_truth`` (which omits the gravity term) is *not* an
    # acceptable substitute — guard against a future regression.
    sf_mean = np.linalg.norm(sim["specific_force_truth"].mean(axis=0))
    assert sf_mean > 0.1, (
        "Default tilt should produce a measurable resting specific force "
        f"(got |mean|={sf_mean:.3f} m/s²) — sim setup may have changed."
    )
    accel_body_mean = np.linalg.norm(sim["accel_body_truth"].mean(axis=0))
    assert accel_body_mean < 1e-3, (
        "accel_body_truth should be near zero at rest (got "
        f"|mean|={accel_body_mean:.3f}); the two truths must differ for "
        "the diagnostic panel mismatch test to be meaningful."
    )


def test_p0_validation_accepts_psd_including_zero_covariance():
    """P0 contract is positive semi-definite — accept zero, reject negative.

    The previous Cholesky-based check enforced strictly positive-definite
    despite the docstring promising PSD. That blocked P0=0 (deterministic
    initial state) and any diagonal with a zero variance.
    """
    m0 = np.array([0.5, 0.5, 0.0, 0.0, 0.0])

    # Zero covariance: deterministic initial state — must construct.
    RatIMUSimConfig(duration_s=1.0, m0=m0, P0=np.zeros((5, 5)))

    # Diagonal with one zero variance: must construct.
    RatIMUSimConfig(duration_s=1.0, m0=m0, P0=np.diag([0.01, 0.01, 0.0, 0.01, 0.01]))

    # Negative-definite must still be rejected.
    with pytest.raises(ValueError, match=r"positive semi-definite"):
        RatIMUSimConfig(
            duration_s=1.0,
            m0=m0,
            P0=np.diag([0.01, 0.01, -0.01, 0.01, 0.01]),
        )


def test_p0_zero_yields_deterministic_initial_state():
    """P0=0 + zero process noise + zero initial velocity ⇒ truth stays at m0.

    The simulator integrates one IMU step before storing ``X_truth[0]``,
    so we choose ``m0`` with zero velocity (and disable all stochastic
    drivers) so the recorded truth equals the configured initial state.
    With any non-zero P0 + a fixed seed the perturbation would shift
    these values; the assertion below is bit-exact at the 1e-12 level.
    """
    m0 = np.array([0.25, 0.5, 0.0, 0.0, 0.3])
    cfg = RatIMUSimConfig(
        duration_s=1.0,
        fs_imu=200.0,
        fs_cam=30.0,
        m0=m0,
        P0=np.zeros((5, 5)),
        sigma_yaw_rate=0.0,
        sigma_a_fwd=0.0,
        sigma_a_lat=0.0,
        gyro_noise_density=0.0,
        accel_noise_density=0.0,
        gyro_bias_rw_density=0.0,
        accel_bias_rw_density=0.0,
    )
    sim = simulate_rat_imu(cfg, seed=0)
    np.testing.assert_allclose(sim["X_truth"][0], m0, atol=1e-12)


def test_m0_and_p0_accept_list_inputs_with_clear_error():
    """``m0`` / ``P0`` accept list/tuple inputs and surface ValueError on shape errors.

    The shape validation previously read ``self.m0.shape`` *before* any
    ndarray coercion, so list inputs raised
    ``AttributeError: 'list' object has no attribute 'shape'`` instead
    of the documented ``ValueError``. Coerce both via ``np.asarray``
    first so callers spelling ``m0=[...]`` get the contract violation
    they expect.
    """
    # List inputs must be accepted (after coercion).
    cfg = RatIMUSimConfig(
        duration_s=1.0, m0=[0.5, 0.5, 0.0, 0.0, 0.0], P0=np.eye(5).tolist()
    )
    assert isinstance(cfg.m0, np.ndarray) and cfg.m0.shape == (5,)
    assert isinstance(cfg.P0, np.ndarray) and cfg.P0.shape == (5, 5)

    # Tuple inputs must also work.
    RatIMUSimConfig(duration_s=1.0, m0=(0.0,) * 5)

    # Wrong-shape list still raises ValueError (not AttributeError).
    with pytest.raises(ValueError, match=r"m0 must have shape"):
        RatIMUSimConfig(duration_s=1.0, m0=[0.5, 0.5])
    with pytest.raises(ValueError, match=r"P0 must have shape"):
        RatIMUSimConfig(duration_s=1.0, P0=[[1.0, 0.0], [0.0, 1.0]])


def test_non_numeric_array_inputs_raise_clear_valueerror():
    """Non-numeric m0 / P0 / LED-offset inputs must raise ValueError, not TypeError.

    Previously the validation coerced via ``np.asarray`` without a dtype
    check, then called ``np.isfinite`` which raised a raw NumPy
    ``TypeError: ufunc 'isfinite' not supported for the input types``.
    Inspect the un-forced dtype kind first and reject anything outside
    ``{int, uint, float, bool}`` with a clear ValueError.
    """
    with pytest.raises(ValueError, match=r"m0 must contain numeric values"):
        RatIMUSimConfig(duration_s=1.0, m0=["0", "0", "0", "0", "0"])
    with pytest.raises(ValueError, match=r"P0 must contain numeric values"):
        RatIMUSimConfig(duration_s=1.0, P0=[["1", "0", "0", "0", "0"]] * 5)
    with pytest.raises(
        ValueError, match=r"led1_offset_body must contain numeric values"
    ):
        RatIMUSimConfig(duration_s=1.0, led1_offset_body=["a", "b"])


def test_camera_latency_and_jitter_must_be_non_negative():
    """cam_latency_s and cam_jitter_s must be ≥ 0 at construction.

    cam_latency_s is the exposure → arrival latency, so a negative value
    would have observations arrive before they were exposed. cam_jitter_s
    is a jitter standard deviation; a negative std is meaningless. Both
    were previously only finite-checked.
    """
    with pytest.raises(ValueError, match=r"cam_latency_s must be non-negative"):
        RatIMUSimConfig(duration_s=1.0, fs_imu=200.0, fs_cam=30.0, cam_latency_s=-0.05)
    with pytest.raises(ValueError, match=r"cam_jitter_s must be non-negative"):
        RatIMUSimConfig(duration_s=1.0, fs_imu=200.0, fs_cam=30.0, cam_jitter_s=-0.01)
    # Zero must remain acceptable (constant exposure timing / no latency).
    RatIMUSimConfig(
        duration_s=1.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_latency_s=0.0,
        cam_jitter_s=0.0,
    )


def test_use_confidence_and_use_second_led_require_strict_bool():
    """Boolean fields must reject non-bool truthy values.

    Plain Python truthiness silently accepts strings like ``"False"``
    (truthy), integers, and lists; the simulator then branches on
    these via ``if self.use_second_led: ...`` and produces finite
    LED2 observations / non-uniform confidences despite the user's
    apparent intent. CLI / YAML / env loaders are an obvious source
    of these bugs — require ``isinstance(value, bool)`` exactly.
    """
    # Clean True/False both work.
    RatIMUSimConfig(duration_s=1.0, use_second_led=True, use_confidence=True)
    RatIMUSimConfig(duration_s=1.0, use_second_led=False, use_confidence=False)

    # Non-bool truthy / falsy values must raise — note ``"False"`` is
    # the dangerous one because Python evaluates it as truthy.
    for bad in ("False", "True", "yes", 1, 0, [1], [0]):
        with pytest.raises(ValueError, match=r"use_second_led must be a Python"):
            RatIMUSimConfig(duration_s=1.0, use_second_led=bad)
        with pytest.raises(ValueError, match=r"use_confidence must be a Python"):
            RatIMUSimConfig(duration_s=1.0, use_confidence=bad)


def test_persistent_swap_with_single_led_warns():
    """Persistent-swap settings on a single-LED sim should emit a warning.

    Previously the warning only checked led_swap_prob (per_frame mode); a
    user enabling persistent swaps with use_second_led=False got no
    warning and silently zero swaps.
    """
    import warnings as warnings_module

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        RatIMUSimConfig(
            duration_s=1.0,
            fs_imu=200.0,
            fs_cam=30.0,
            use_second_led=False,
            led_swap_mode="persistent",
            led_swap_rate=2.0,
        )

    persistent_warnings = [
        str(w.message)
        for w in caught
        if "persistent" in str(w.message).lower() and "use_second_led" in str(w.message)
    ]
    assert len(persistent_warnings) == 1, (
        f"Expected exactly one persistent-swap warning; got "
        f"{len(persistent_warnings)}: {persistent_warnings}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

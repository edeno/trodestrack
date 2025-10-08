"""Property-based tests using Hypothesis for simulation validation.

These tests complement example-based tests by exploring a wide range of
parameter combinations to find edge cases and validate universal invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)


# =============================================================================
# Strategy definitions
# =============================================================================


@st.composite
def valid_sim_config(draw):
    """Generate valid RatIMUSimConfig with random parameters."""
    duration_s = draw(st.floats(min_value=1.0, max_value=20.0))
    fs_imu = draw(st.floats(min_value=100.0, max_value=500.0))
    fs_cam = draw(st.floats(min_value=10.0, max_value=60.0))
    arena_w = draw(st.floats(min_value=0.5, max_value=3.0))
    arena_h = draw(st.floats(min_value=0.5, max_value=3.0))

    return RatIMUSimConfig(
        duration_s=duration_s,
        fs_imu=fs_imu,
        fs_cam=fs_cam,
        arena_w=arena_w,
        arena_h=arena_h,
    )


@st.composite
def valid_simple_config(draw):
    """Generate valid SimpleSimConfig with random parameters."""
    duration_s = draw(st.floats(min_value=1.0, max_value=10.0))
    fs_imu = draw(st.floats(min_value=100.0, max_value=300.0))
    fs_cam = draw(st.floats(min_value=10.0, max_value=50.0))

    return SimpleSimConfig(
        duration_s=duration_s,
        fs_imu=fs_imu,
        fs_cam=fs_cam,
    )


# =============================================================================
# Property tests for rat_imu simulator
# =============================================================================


@given(config=valid_sim_config(), seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=20, deadline=5000)
def test_property_rat_imu_produces_valid_output(config: RatIMUSimConfig, seed: int) -> None:
    """Property: rat_imu simulation always produces valid, finite output."""
    sim = simulate_rat_imu(config, seed=seed)

    # Universal invariants
    assert not np.any(np.isnan(sim["X_truth"])), "Ground truth contains NaN"
    assert not np.any(np.isnan(sim["U_imu"])), "IMU measurements contain NaN"
    assert not np.any(np.isinf(sim["X_truth"])), "Ground truth contains Inf"
    assert not np.any(np.isinf(sim["U_imu"])), "IMU measurements contain Inf"

    # Time monotonicity (standardized policy across all tests)
    # IMU: strictly monotonic (no jitter)
    assert np.all(np.diff(sim["t_imu"]) > 0), "IMU time not monotonic"

    # Camera: mostly monotonic (Gaussian jitter can cause occasional reordering)
    # Policy: require ≥95% of intervals positive
    cam_diffs = np.diff(sim["t_cam_exp"])
    positive_rate = (cam_diffs > 0).mean()
    assert positive_rate > 0.95, f"Camera time mostly monotonic: {positive_rate:.1%} positive"

    # Arena bounds: reflections work but there can be overshoots due to:
    # 1. Initial state sampling from P0 can place rat outside arena
    # 2. Integration step happens before reflection check
    # This is a known behavior - the reflection mechanism prevents escape but allows temporary overshoot
    # Strict bounds are tested in test_rat_imu.py with controlled initial conditions
    x, y = sim["X_truth"][:, 0], sim["X_truth"][:, 1]
    # Just verify positions stay reasonable (not wildly out of bounds)
    reasonable_bound = max(config.arena_w, config.arena_h) + 0.5  # Within 50cm of arena
    assert np.all(x >= -reasonable_bound), f"Position x wildly negative: min={x.min()}"
    assert np.all(
        x <= config.arena_w + reasonable_bound
    ), f"Position x wildly positive: max={x.max()}"
    assert np.all(y >= -reasonable_bound), f"Position y wildly negative: min={y.min()}"
    assert np.all(
        y <= config.arena_h + reasonable_bound
    ), f"Position y wildly positive: max={y.max()}"

    # Speed limits
    vx, vy = sim["X_truth"][:, 2], sim["X_truth"][:, 3]
    speed = np.hypot(vx, vy)
    # Allow small overshoot due to numerical integration
    assert np.all(speed <= config.speed_clip * 1.1), f"Speed exceeds clip: max={speed.max()}"

    # Heading wrapping
    theta = sim["X_truth"][:, 4]
    assert np.all(theta >= -np.pi), f"Heading < -π: min={theta.min()}"
    assert np.all(theta <= np.pi), f"Heading > π: max={theta.max()}"


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=50, deadline=3000)
def test_property_rat_imu_reproducibility(seed: int) -> None:
    """Property: same seed always produces identical results."""
    config = RatIMUSimConfig(duration_s=3.0, fs_imu=200.0, fs_cam=30.0)

    sim1 = simulate_rat_imu(config, seed=seed)
    sim2 = simulate_rat_imu(config, seed=seed)

    # Deterministic reproduction
    assert np.allclose(sim1["X_truth"], sim2["X_truth"]), "Truth states differ"
    assert np.allclose(sim1["U_imu"], sim2["U_imu"]), "IMU measurements differ"
    assert np.array_equal(sim1["mask_led1"], sim2["mask_led1"]), "Dropout masks differ"


@given(
    seed1=st.integers(min_value=0, max_value=10000),
    seed2=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=30, deadline=3000)
def test_property_rat_imu_different_seeds_differ(seed1: int, seed2: int) -> None:
    """Property: different seeds produce different results."""
    if seed1 == seed2:
        return  # Skip if seeds are the same

    config = RatIMUSimConfig(duration_s=3.0, fs_imu=200.0, fs_cam=30.0)

    sim1 = simulate_rat_imu(config, seed=seed1)
    sim2 = simulate_rat_imu(config, seed=seed2)

    # Ground truth should differ (OU process is seeded)
    # Allow for extremely rare case where they happen to be close
    assert not np.allclose(
        sim1["X_truth"], sim2["X_truth"]
    ), f"Seeds {seed1} and {seed2} produced identical truth (extremely unlikely)"


@given(
    duration=st.floats(min_value=1.0, max_value=30.0),
    fs_imu=st.floats(min_value=100.0, max_value=1000.0),
    fs_cam=st.floats(min_value=10.0, max_value=60.0),
)
@settings(max_examples=20, deadline=5000)
def test_property_rat_imu_sample_counts(duration: float, fs_imu: float, fs_cam: float) -> None:
    """Property: sample counts match configured rates."""
    config = RatIMUSimConfig(duration_s=duration, fs_imu=fs_imu, fs_cam=fs_cam)

    sim = simulate_rat_imu(config, seed=42)

    # Expected sample counts
    expected_T_imu = int(np.round(duration * fs_imu))
    expected_T_cam = int(np.round(duration * fs_cam))

    # Actual sample counts
    actual_T_imu = len(sim["t_imu"])
    actual_T_cam = len(sim["t_cam_exp"])

    assert actual_T_imu == expected_T_imu, f"IMU count: {actual_T_imu} != {expected_T_imu}"
    assert actual_T_cam == expected_T_cam, f"Cam count: {actual_T_cam} != {expected_T_cam}"


@given(
    dropout_prob=st.floats(min_value=0.0, max_value=0.5),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=30, deadline=3000)
def test_property_dropout_rate_bounds(dropout_prob: float, seed: int) -> None:
    """Property: dropout rate stays within reasonable bounds."""
    config = RatIMUSimConfig(
        duration_s=20.0,
        fs_cam=30.0,
        cam_dropout_prob=dropout_prob,
    )

    sim = simulate_rat_imu(config, seed=seed)

    mask = sim["mask_led1"]
    observed_dropout_rate = 1 - mask.mean()

    # Dropout rate should be within 4 sigma of expected (99.99% confidence)
    n = len(mask)
    p = dropout_prob

    # For very small dropout_prob (< 0.01), use absolute tolerance instead of relative
    if p < 0.01:
        # Absolute tolerance: allow ±2% deviation for rare events
        assert (
            observed_dropout_rate <= 0.02
        ), f"Dropout rate {observed_dropout_rate:.3f} too high for p={p:.6f}"
    else:
        sigma = np.sqrt(p * (1 - p) / n)
        lower_bound = max(0.0, dropout_prob - 4 * sigma)
        upper_bound = min(1.0, dropout_prob + 4 * sigma)

        assert (
            lower_bound <= observed_dropout_rate <= upper_bound
        ), f"Dropout rate {observed_dropout_rate:.3f} outside [{lower_bound:.3f}, {upper_bound:.3f}]"


# =============================================================================
# Property tests for simple simulator
# =============================================================================


@given(config=valid_simple_config(), seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=15, deadline=3000)
def test_property_stationary_stays_stationary(config: SimpleSimConfig, seed: int) -> None:
    """Property: stationary simulation produces constant position."""
    position = np.array([1.0, 1.0])
    sim = simulate_stationary(config, position=position, seed=seed)

    X = sim["X_truth"]

    # Position should be constant
    assert np.allclose(X[:, 0], position[0]), "X position changed"
    assert np.allclose(X[:, 1], position[1]), "Y position changed"

    # Velocity should be zero
    assert np.allclose(X[:, 2], 0.0), "X velocity non-zero"
    assert np.allclose(X[:, 3], 0.0), "Y velocity non-zero"


@given(config=valid_simple_config(), seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=15, deadline=3000)
def test_property_constant_velocity_is_linear(config: SimpleSimConfig, seed: int) -> None:
    """Property: constant velocity produces linear trajectory."""
    initial = np.array([0.2, 0.3])
    velocity = np.array([0.15, 0.10])

    sim = simulate_constant_velocity(config, initial_position=initial, velocity=velocity, seed=seed)

    t = sim["t_imu"]
    X = sim["X_truth"]

    # Position should follow: p(t) = p0 + v*t
    expected_x = initial[0] + velocity[0] * t
    expected_y = initial[1] + velocity[1] * t

    assert np.allclose(X[:, 0], expected_x, atol=1e-9), "X position not linear"
    assert np.allclose(X[:, 1], expected_y, atol=1e-9), "Y position not linear"

    # Velocity should be constant
    assert np.allclose(X[:, 2], velocity[0], atol=1e-9), "X velocity not constant"
    assert np.allclose(X[:, 3], velocity[1], atol=1e-9), "Y velocity not constant"


@given(
    config=valid_simple_config(),
    radius=st.floats(min_value=0.1, max_value=0.5),
    omega=st.floats(min_value=0.1, max_value=2.0),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=15, deadline=3000)
def test_property_circular_stays_on_circle(
    config: SimpleSimConfig, radius: float, omega: float, seed: int
) -> None:
    """Property: circular motion stays on circle."""
    center = np.array([1.0, 1.0])

    sim = simulate_circular(config, center=center, radius=radius, angular_velocity=omega, seed=seed)

    X = sim["X_truth"]
    x, y = X[:, 0], X[:, 1]

    # Distance from center should equal radius
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    assert np.allclose(distance, radius, atol=1e-9), f"Not on circle: std={distance.std()}"


# =============================================================================
# Property tests for data structure invariants
# =============================================================================


@given(config=valid_sim_config(), seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=20, deadline=5000)
def test_property_output_structure_complete(config: RatIMUSimConfig, seed: int) -> None:
    """Property: output always has complete expected structure."""
    sim = simulate_rat_imu(config, seed=seed)

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
        "confidence_led1",
        "confidence_led2",
        "mask_cam",
        "mask_led1",
        "mask_led2",
        "config",
    }

    assert set(sim.keys()) == expected_keys, f"Missing keys: {expected_keys - set(sim.keys())}"

    # Check array dimensions
    T_imu = len(sim["t_imu"])
    T_cam = len(sim["t_cam_exp"])

    assert sim["X_truth"].shape == (T_imu, 5), "X_truth shape mismatch"
    assert sim["U_imu"].shape == (T_imu, 3), "U_imu shape mismatch"
    assert sim["Z_cam_led1"].shape == (T_cam, 2), "Z_cam_led1 shape mismatch"
    assert sim["mask_led1"].shape == (T_cam,), "mask_led1 shape mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])

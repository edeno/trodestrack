"""Tests for quality assurance metrics module.

Tests cover:
- RMSE computation for position, velocity, heading
- NEES and NIS consistency checks
- Residual autocorrelation (whiteness test)
- Dropout drift measurement
- Edge cases: NaN handling, singular covariances, empty inputs
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from trodestrack.models.state_layout import StateLayout
from trodestrack.qa.metrics import (
    chi2_ci95,
    compute_dropout_drift,
    compute_heading_error,
    compute_heading_rmse,
    compute_nees,
    compute_nees_stats,
    compute_nis,
    compute_nis_stats,
    compute_position_rmse,
    compute_residual_autocorrelation,
    compute_velocity_rmse,
)

# =============================================================================
# Position RMSE Tests
# =============================================================================


def test_position_rmse_perfect_match():
    """Perfect estimate should give zero RMSE."""
    true_pos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    est_pos = true_pos.copy()

    rmse = compute_position_rmse(true_pos, est_pos)
    assert_allclose(rmse, 0.0, atol=1e-10)


def test_position_rmse_known_error():
    """Known error should give expected RMSE."""
    # True positions at origin
    true_pos = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    # Estimates offset by (0.03, 0.04) -> Euclidean error = 0.05 m = 5 cm
    est_pos = np.array([[0.03, 0.04], [0.03, 0.04], [0.03, 0.04]])

    rmse = compute_position_rmse(true_pos, est_pos)
    # RMSE = sqrt(mean((0.05)^2)) = 0.05 m
    assert_allclose(rmse, 0.05, atol=1e-10)


def test_position_rmse_with_mask():
    """Mask should exclude invalid samples."""
    true_pos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # Last sample has large error
    est_pos = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])

    # With mask excluding last sample, RMSE should be zero
    mask = np.array([True, True, False])
    rmse = compute_position_rmse(true_pos, est_pos, valid_mask=mask)
    assert_allclose(rmse, 0.0, atol=1e-10)

    # Without mask, RMSE should be large
    rmse_no_mask = compute_position_rmse(true_pos, est_pos)
    assert rmse_no_mask > 4.0  # Should be ~6.5 m


def test_position_rmse_with_nans():
    """NaNs should be automatically filtered out."""
    true_pos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    est_pos = np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0]])

    # Should compute RMSE only on valid samples (first and third)
    rmse = compute_position_rmse(true_pos, est_pos)
    assert_allclose(rmse, 0.0, atol=1e-10)


def test_position_rmse_shape_mismatch():
    """Mismatched shapes should raise ValueError."""
    true_pos = np.array([[0.0, 0.0], [1.0, 1.0]])
    est_pos = np.array([[0.0, 0.0]])

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_position_rmse(true_pos, est_pos)


def test_position_rmse_wrong_dimension():
    """Non-2D positions should raise ValueError."""
    true_pos = np.array([[0.0, 0.0, 0.0]])  # 3D
    est_pos = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match=r"positions of shape \(N, 2\)"):
        compute_position_rmse(true_pos, est_pos)


def test_position_rmse_one_d_input():
    """1-D positions should raise ValueError, not IndexError."""
    true_pos = np.array([1, 2])
    est_pos = np.array([1, 2])

    with pytest.raises(ValueError, match=r"positions of shape \(N, 2\)"):
        compute_position_rmse(true_pos, est_pos)


def test_position_rmse_no_valid_samples():
    """All invalid samples should raise ValueError."""
    true_pos = np.array([[0.0, 0.0], [1.0, 1.0]])
    est_pos = np.array([[np.nan, 0.0], [1.0, np.nan]])

    with pytest.raises(ValueError, match="No valid samples"):
        compute_position_rmse(true_pos, est_pos)


# =============================================================================
# Velocity RMSE Tests
# =============================================================================


def test_velocity_rmse_perfect_match():
    """Perfect velocity estimate should give zero RMSE."""
    true_vel = np.array([[0.1, 0.2], [0.3, 0.4]])
    est_vel = true_vel.copy()

    rmse = compute_velocity_rmse(true_vel, est_vel)
    assert_allclose(rmse, 0.0, atol=1e-10)


def test_velocity_rmse_known_error():
    """Known velocity error should give expected RMSE."""
    true_vel = np.array([[0.0, 0.0], [0.0, 0.0]])
    # Error magnitude = sqrt(0.03^2 + 0.04^2) = 0.05 m/s
    est_vel = np.array([[0.03, 0.04], [0.03, 0.04]])

    rmse = compute_velocity_rmse(true_vel, est_vel)
    assert_allclose(rmse, 0.05, atol=1e-10)


def test_velocity_rmse_with_mask():
    """Mask should exclude invalid velocity samples."""
    true_vel = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    est_vel = np.array([[0.1, 0.1], [0.2, 0.2], [999.0, 999.0]])

    mask = np.array([True, True, False])
    rmse = compute_velocity_rmse(true_vel, est_vel, valid_mask=mask)
    assert_allclose(rmse, 0.0, atol=1e-10)


def test_velocity_rmse_one_d_input():
    """1-D velocities should raise ValueError, not IndexError."""
    true_vel = np.array([1, 2])
    est_vel = np.array([1, 2])

    with pytest.raises(ValueError, match=r"velocities of shape \(N, 2\)"):
        compute_velocity_rmse(true_vel, est_vel)


# =============================================================================
# Heading Error Tests
# =============================================================================


def test_heading_error_perfect_match():
    """Perfect heading estimate should give zero error."""
    true_heading = np.array([0.0, np.pi / 2, np.pi])
    est_heading = true_heading.copy()

    error = compute_heading_error(true_heading, est_heading)
    assert_allclose(error, 0.0, atol=1e-10)


def test_heading_error_known_error():
    """Known heading error should be computed correctly with wrapping."""
    # True: [0, π/2, π], Est: [0.1, π/2+0.1, π-0.1]
    # Errors: [0.1, 0.1, -0.1] rad
    # Mean absolute error = 0.1 rad ≈ 5.73°
    true_heading = np.array([0.0, np.pi / 2, np.pi])
    est_heading = np.array([0.1, np.pi / 2 + 0.1, np.pi - 0.1])

    error = compute_heading_error(true_heading, est_heading)
    assert_allclose(error, 0.1, atol=1e-10)  # Returns radians (SI unit)


def test_heading_error_wrapping():
    """Heading error should handle angle wrapping correctly."""
    # True: 0.1 rad, Est: -0.1 rad (or 2π - 0.1)
    # Wrapped difference should be 0.2 rad (shortest path)
    true_heading = np.array([0.1])
    est_heading = np.array([-0.1])

    error = compute_heading_error(true_heading, est_heading)
    assert_allclose(error, 0.2, atol=1e-10)  # Returns radians (SI unit)

    # Test wrap-around at ±π
    true_heading = np.array([np.pi - 0.1])
    est_heading = np.array([-np.pi + 0.1])

    error = compute_heading_error(true_heading, est_heading)
    # Shortest path: 0.2 rad (not 2π - 0.2)
    assert_allclose(error, 0.2, atol=1e-10)  # Returns radians (SI unit)


def test_heading_rmse():
    """RMSE for heading should handle angle wrapping."""
    true_heading = np.array([0.0, np.pi / 2, np.pi])
    est_heading = np.array([0.1, np.pi / 2 + 0.1, np.pi - 0.1])

    rmse = compute_heading_rmse(true_heading, est_heading)
    # Errors: [0.1, 0.1, -0.1] rad -> RMSE = sqrt(mean(0.01, 0.01, 0.01)) = 0.1 rad
    assert_allclose(rmse, 0.1, atol=1e-10)


# =============================================================================
# NEES Tests
# =============================================================================


def test_nees_consistent_filter():
    """NEES should be ~state_dim for a consistent filter."""
    np.random.seed(42)
    state_dim = 5
    N = 100

    # Generate errors consistent with covariance
    cov = np.eye(state_dim) * 0.1
    errors = np.random.multivariate_normal(np.zeros(state_dim), cov, size=N)

    true_state = np.zeros((N, state_dim))
    est_state = true_state + errors
    cov_est = np.stack([cov] * N)

    nees = compute_nees(true_state, est_state, cov_est)

    # Mean NEES should be approximately equal to state dimension
    mean_nees = np.mean(nees)
    # Allow generous tolerance due to finite sample size
    assert 3.0 < mean_nees < 7.0, (
        f"Mean NEES {mean_nees:.2f} outside expected range [3, 7]"
    )


def test_nees_overconfident_filter():
    """NEES should be high when covariance is underestimated."""
    np.random.seed(42)
    state_dim = 2
    N = 50

    # True error larger than estimated covariance (filter is overconfident)
    true_cov = np.eye(state_dim) * 1.0  # Large error
    est_cov = np.eye(state_dim) * 0.1  # Small covariance (overconfident)

    errors = np.random.multivariate_normal(np.zeros(state_dim), true_cov, size=N)
    true_state = np.zeros((N, state_dim))
    est_state = true_state + errors
    cov_est = np.stack([est_cov] * N)

    nees = compute_nees(true_state, est_state, cov_est)

    # NEES should be much higher than state_dim (overconfident)
    mean_nees = np.mean(nees)
    assert mean_nees > 10.0, (
        f"Expected high NEES for overconfident filter, got {mean_nees:.2f}"
    )


def test_nees_shape_mismatch():
    """NEES should raise error on shape mismatch."""
    true_state = np.zeros((10, 5))
    est_state = np.zeros((10, 5))
    cov_est = np.zeros((10, 4, 4))  # Wrong dimension

    with pytest.raises(ValueError, match="Covariance shape mismatch"):
        compute_nees(true_state, est_state, cov_est)


def test_nees_singular_covariance():
    """NEES should return inf for singular covariance."""
    true_state = np.array([[0.0, 0.0]])
    est_state = np.array([[1.0, 1.0]])
    cov_est = np.array([[[0.0, 0.0], [0.0, 0.0]]])  # Singular

    nees = compute_nees(true_state, est_state, cov_est)
    assert np.isinf(nees[0])


def test_nees_with_angle_wrapping():
    """NEES should correctly handle angle wrapping for heading states.

    This is a regression test for the bug where heading errors weren't wrapped,
    causing huge NEES values when angles wrap through 0°/360°.
    """
    # State: [x, heading_rad]
    # Simulate heading wrapping through 0° (359° → 0° → 1°)
    true_state = np.array(
        [
            [1.0, np.deg2rad(359)],  # 359°
            [1.0, np.deg2rad(360)],  # 360° = 0°
            [1.0, np.deg2rad(1)],  # 1°
        ]
    )

    est_state = np.array(
        [
            [1.0, np.deg2rad(359)],  # Perfect match
            [1.0, np.deg2rad(0)],  # Perfect match (360° = 0°)
            [1.0, np.deg2rad(1)],  # Perfect match
        ]
    )

    # Small covariance (filter is confident)
    cov = np.array(
        [
            [[0.01, 0], [0, 0.01]],  # σ_x = 0.1 m, σ_θ = 0.1 rad ≈ 5.7°
            [[0.01, 0], [0, 0.01]],
            [[0.01, 0], [0, 0.01]],
        ]
    )

    # Compute NEES with angle wrapping at index 1
    nees = compute_nees(true_state, est_state, cov, heading_idx=1)

    # All errors are zero (perfect tracking), so NEES should be ~0
    # Definitely should NOT be huge like 1315 (the bug behavior)
    mean_nees = np.mean(nees)
    max_nees = np.max(nees)

    assert mean_nees < 1.0, (
        f"Mean NEES {mean_nees:.2f} too high (should be ~0 for perfect tracking)"
    )
    assert max_nees < 2.0, f"Max NEES {max_nees:.2f} too high (bug: angle not wrapped)"

    # The bug would cause middle sample to have NEES ≈ 3948 (from 360° error)
    # With fix, middle sample should have NEES ≈ 0
    assert nees[1] < 1.0, (
        f"Middle sample NEES {nees[1]:.2f} indicates angle wrapping bug"
    )


def test_nees_with_layout_auto_detection():
    """NEES should automatically extract heading index from layout parameter.

    This tests the user-friendly API where users pass the layout and don't
    need to remember to specify heading_idx manually.
    """
    # Create a 2D state layout (similar to "2d_full" mode)
    # State: [x, y, vx, vy, heading, bias_gyro, bias_ax, bias_ay] (8D)
    layout = StateLayout(
        n=8,
        pos_idx=(0, 1),
        vel_idx=(2, 3),
        heading_idx=4,  # Heading at index 4
        bias_gyro_idx=(5,),
        bias_accel_idx=(6, 7),
    )

    # Simulate heading wrapping through 0° (359° → 0° → 1°)
    # Full 8D state with heading at index 4
    true_state = np.array(
        [
            [1.0, 2.0, 0.1, 0.2, np.deg2rad(359), 0.0, 0.0, 0.0],  # 359°
            [1.0, 2.0, 0.1, 0.2, np.deg2rad(360), 0.0, 0.0, 0.0],  # 360° = 0°
            [1.0, 2.0, 0.1, 0.2, np.deg2rad(1), 0.0, 0.0, 0.0],  # 1°
        ]
    )

    est_state = np.array(
        [
            [1.0, 2.0, 0.1, 0.2, np.deg2rad(359), 0.0, 0.0, 0.0],  # Perfect match
            [1.0, 2.0, 0.1, 0.2, np.deg2rad(0), 0.0, 0.0, 0.0],  # Perfect match
            [1.0, 2.0, 0.1, 0.2, np.deg2rad(1), 0.0, 0.0, 0.0],  # Perfect match
        ]
    )

    # Small covariance (filter is confident)
    cov = np.array([np.eye(8) * 0.01] * 3)

    # Pass layout parameter instead of heading_idx (user-friendly API)
    nees = compute_nees(true_state, est_state, cov, layout=layout)

    # All errors are zero (perfect tracking), so NEES should be ~0
    mean_nees = np.mean(nees)
    max_nees = np.max(nees)

    assert mean_nees < 1.0, f"Mean NEES {mean_nees:.2f} too high (should be ~0)"
    assert max_nees < 2.0, f"Max NEES {max_nees:.2f} too high"

    # Middle sample with angle wrapping should still have NEES ≈ 0
    assert nees[1] < 1.0, (
        f"Middle sample NEES {nees[1]:.2f} indicates layout didn't work"
    )


# =============================================================================
# NIS Tests
# =============================================================================


def test_nis_consistent_filter():
    """NIS should be ~measurement_dim for a consistent filter."""
    np.random.seed(42)
    meas_dim = 2
    N = 100

    # Generate innovations consistent with covariance
    cov = np.eye(meas_dim) * 0.1
    innovations = np.random.multivariate_normal(np.zeros(meas_dim), cov, size=N)
    cov_est = np.stack([cov] * N)

    nis = compute_nis(innovations, cov_est)

    # Mean NIS should be approximately equal to measurement dimension
    mean_nis = np.mean(nis)
    assert 1.0 < mean_nis < 3.0, (
        f"Mean NIS {mean_nis:.2f} outside expected range [1, 3]"
    )


def test_nis_singular_covariance():
    """NIS should return inf for singular innovation covariance."""
    innovations = np.array([[1.0, 1.0]])
    cov = np.array([[[0.0, 0.0], [0.0, 0.0]]])  # Singular

    nis = compute_nis(innovations, cov)
    assert np.isinf(nis[0])


# =============================================================================
# NEES/NIS Stats Tests
# =============================================================================


def test_nees_stats():
    """NEES stats should compute chi-squared bounds correctly."""
    np.random.seed(42)
    state_dim = 5
    nees = np.random.chisquare(df=state_dim, size=100)

    stats = compute_nees_stats(nees, state_dim)

    # Check all required keys present
    assert "mean" in stats
    assert "std" in stats
    assert "chi2_lower" in stats
    assert "chi2_upper" in stats
    assert "pct_in_bounds" in stats
    assert "confidence" in stats
    assert stats["confidence"] == 0.95  # Default confidence

    # Mean should be approximately state_dim for chi-squared(state_dim)
    assert 4.0 < stats["mean"] < 6.0

    # Most samples should be within 95% CI (allow some variance)
    assert stats["pct_in_bounds"] > 85.0


def test_nis_stats():
    """NIS stats should compute chi-squared bounds correctly."""
    np.random.seed(42)
    meas_dim = 4
    nis = np.random.chisquare(df=meas_dim, size=100)

    stats = compute_nis_stats(nis, meas_dim)

    # Check all required keys present
    assert "mean" in stats
    assert "std" in stats
    assert "chi2_lower" in stats
    assert "chi2_upper" in stats
    assert "pct_in_bounds" in stats
    assert "confidence" in stats
    assert stats["confidence"] == 0.95  # Default confidence

    # Mean should be approximately meas_dim for chi-squared(meas_dim)
    assert 3.0 < stats["mean"] < 5.0


def test_chi2_ci95():
    """Chi-squared 95% CI should match known values."""
    # For df=2: 95% CI ≈ [0.051, 7.378]
    lower, upper = chi2_ci95(df=2)
    assert_allclose(lower, 0.051, atol=0.01)
    assert_allclose(upper, 7.378, atol=0.01)

    # For df=4: 95% CI ≈ [0.484, 11.143]
    lower, upper = chi2_ci95(df=4)
    assert_allclose(lower, 0.484, atol=0.01)
    assert_allclose(upper, 11.143, atol=0.01)


def test_chi2_bounds_95():
    """chi2_bounds should match chi2_ci95 for 95% confidence."""
    from trodestrack.qa.metrics import chi2_bounds

    # Should match chi2_ci95 exactly
    for df in [2, 4, 5, 8]:
        lower_old, upper_old = chi2_ci95(df=df)
        lower_new, upper_new = chi2_bounds(df=df, confidence=0.95)
        assert_allclose(lower_new, lower_old, rtol=1e-10)
        assert_allclose(upper_new, upper_old, rtol=1e-10)


def test_chi2_bounds_different_confidences():
    """chi2_bounds should work with different confidence levels."""
    from trodestrack.qa.metrics import chi2_bounds

    # 90% CI should be narrower than 95% CI
    lower_90, upper_90 = chi2_bounds(df=4, confidence=0.90)
    lower_95, upper_95 = chi2_bounds(df=4, confidence=0.95)
    lower_99, upper_99 = chi2_bounds(df=4, confidence=0.99)

    # Wider confidence → wider interval
    assert lower_90 > lower_95
    assert lower_95 > lower_99
    assert upper_90 < upper_95
    assert upper_95 < upper_99


def test_within_envelope_basic():
    """within_envelope should correctly identify values in/out of bounds."""
    from trodestrack.qa.metrics import within_envelope

    # Generate chi-squared samples
    np.random.seed(42)
    df = 4
    values = np.random.chisquare(df, size=1000)

    # With 95% confidence, expect ~95% within bounds
    pct = within_envelope(values, df=df, confidence=0.95)
    assert 0.93 < pct < 0.97, f"Expected ~95%, got {pct * 100:.1f}%"

    # With 99% confidence, expect ~99% within bounds
    pct_99 = within_envelope(values, df=df, confidence=0.99)
    assert 0.97 < pct_99 < 1.0, f"Expected ~99%, got {pct_99 * 100:.1f}%"
    assert pct_99 > pct, "99% envelope should contain more values than 95%"


def test_within_envelope_edge_cases():
    """within_envelope should handle edge cases correctly."""
    from trodestrack.qa.metrics import within_envelope

    # All zeros → 0% within bounds (below lower threshold)
    values = np.zeros(100)
    pct = within_envelope(values, df=4, confidence=0.95)
    assert pct == 0.0

    # All very large values → 0% within bounds (above upper threshold)
    values = np.full(100, 1000.0)
    pct = within_envelope(values, df=4, confidence=0.95)
    assert pct == 0.0

    # Values at mean of chi-squared → 100% within bounds
    values = np.full(100, 4.0)  # Mean of chi-squared(4) = 4
    pct = within_envelope(values, df=4, confidence=0.95)
    assert pct == 1.0


# =============================================================================
# Residual Autocorrelation Tests
# =============================================================================


def test_autocorrelation_white_noise():
    """White noise should have near-zero autocorrelation at lag > 0."""
    np.random.seed(42)
    white_noise = np.random.randn(200)

    acf = compute_residual_autocorrelation(white_noise, max_lag=10)

    # Lag 0 should be 1.0
    assert_allclose(acf[0], 1.0, atol=1e-10)

    # Higher lags should be near zero (95% bounds: ±1.96/sqrt(N) ≈ ±0.14)
    # Use looser bound for robustness
    assert np.all(np.abs(acf[1:]) < 0.2)


def test_autocorrelation_correlated_signal():
    """Correlated signal should show non-zero autocorrelation."""
    np.random.seed(42)
    # AR(1) process: x[t] = 0.8 * x[t-1] + noise
    N = 200
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = 0.8 * x[t - 1] + 0.3 * np.random.randn()

    acf = compute_residual_autocorrelation(x, max_lag=5)

    # Lag 0 should be 1.0
    assert_allclose(acf[0], 1.0, atol=1e-10)

    # Lag 1 should show strong positive correlation (~0.8)
    assert acf[1] > 0.5


def test_autocorrelation_multivariate():
    """Multivariate residuals should compute ACF per dimension."""
    np.random.seed(42)
    residuals = np.random.randn(100, 3)

    acf = compute_residual_autocorrelation(residuals, max_lag=5)

    # Should return (3, 6) array: 3 dimensions × 6 lags (0-5)
    assert acf.shape == (3, 6)

    # Each dimension should have lag-0 correlation = 1.0
    assert_allclose(acf[:, 0], 1.0, atol=1e-10)


def test_autocorrelation_constant_input():
    """Constant input should return ACF[0]=1, rest=NaN."""
    constant = np.ones(50)

    acf = compute_residual_autocorrelation(constant, max_lag=3)

    # Lag 0 should be 1.0
    assert_allclose(acf[0], 1.0, atol=1e-10)

    # Higher lags should be NaN (zero variance)
    assert np.all(np.isnan(acf[1:]))


# =============================================================================
# Dropout Drift Tests
# =============================================================================


def test_dropout_drift_no_dropout():
    """No dropout should return None values."""
    t = np.linspace(0, 10, 100)
    positions = np.column_stack([t * 0.1, np.zeros_like(t)])
    mask = np.ones(100, dtype=bool)  # All valid (no dropout)

    result = compute_dropout_drift(positions, mask, t, min_duration_s=5.0)

    assert result["drift_m"] is None
    assert result["duration_s"] is None
    assert result["start_idx"] is None
    assert result["end_idx"] is None


def test_dropout_drift_too_short():
    """Dropout shorter than min_duration should return None."""
    t = np.linspace(0, 10, 100)
    positions = np.column_stack([t * 0.1, np.zeros_like(t)])

    # Create 2s dropout (shorter than 5s minimum)
    mask = (t < 3.0) | (t >= 5.0)

    result = compute_dropout_drift(positions, mask, t, min_duration_s=5.0)

    assert result["drift_m"] is None


def test_dropout_drift_known_drift():
    """Known dropout with constant velocity should give expected drift."""
    # 10s trajectory at 0.1 m/s in x-direction
    t = np.linspace(0, 10, 1000)
    positions = np.column_stack([t * 0.1, np.zeros_like(t)])

    # Dropout from t=3s to t=8s (5s duration)
    mask = (t < 3.0) | (t >= 8.0)

    result = compute_dropout_drift(positions, mask, t, min_duration_s=4.0)

    # Drift should be ~0.5 m (5s * 0.1 m/s)
    assert result["drift_m"] is not None
    assert 0.45 < result["drift_m"] < 0.55

    # Duration should be ~5s
    assert result["duration_s"] is not None
    assert 4.8 < result["duration_s"] < 5.2


def test_dropout_drift_multiple_blocks():
    """Should return first qualifying dropout block."""
    t = np.linspace(0, 20, 2000)
    positions = np.column_stack([t * 0.1, np.zeros_like(t)])

    # Two dropout blocks: [2-4s] (too short) and [10-16s] (long enough)
    mask = (t < 2.0) | ((t >= 4.0) & (t < 10.0)) | (t >= 16.0)

    result = compute_dropout_drift(positions, mask, t, min_duration_s=5.0)

    # Should return second block (10-16s, ~6s duration, ~0.6m drift)
    assert result["drift_m"] is not None
    assert 0.55 < result["drift_m"] < 0.65
    assert 5.8 < result["duration_s"] < 6.2


def test_dropout_drift_shape_mismatch():
    """Mismatched input shapes should raise ValueError."""
    positions = np.zeros((100, 2))
    mask = np.zeros(50, dtype=bool)  # Wrong size
    t = np.zeros(100)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_dropout_drift(positions, mask, t)


def test_dropout_drift_rejects_non_1d_mask():
    """``valid_mask`` must be 1D (N,); column / row vectors must be rejected.

    Previously the function only checked ``shape[0]`` and then ran
    ``np.diff(..., axis=-1)`` over the mask. A column-vector mask
    (a common shape coming out of column-vector loaders or one-hot
    conversions) silently bypassed the dropout detection and returned
    "no qualifying dropout" for what was actually a real dropout —
    masking a PRD-relevant drift failure.
    """
    t = np.linspace(0, 10, 100)
    positions = np.column_stack([t * 0.1, np.zeros_like(t)])
    mask_1d = (t < 3.0) | (t >= 8.0)  # 5 s dropout

    # Sanity: 1D path still detects the dropout.
    result_1d = compute_dropout_drift(positions, mask_1d, t, min_duration_s=4.0)
    assert result_1d["duration_s"] is not None

    # Column-vector mask: must reject, not silently return None.
    with pytest.raises(ValueError, match=r"valid_mask must be 1-D"):
        compute_dropout_drift(positions, mask_1d.reshape(-1, 1), t, min_duration_s=4.0)

    # Row-vector mask: same rejection.
    with pytest.raises(ValueError, match=r"valid_mask must be 1-D"):
        compute_dropout_drift(positions, mask_1d.reshape(1, -1), t, min_duration_s=4.0)


# =============================================================================
# Integration Tests
# =============================================================================


def test_metrics_workflow_integration():
    """Test full QA workflow: generate data, compute all metrics."""
    np.random.seed(42)
    N = 100
    state_dim = 5
    meas_dim = 2

    # Generate synthetic ground truth and estimates
    true_state = np.random.randn(N, state_dim) * 0.1
    errors = np.random.randn(N, state_dim) * 0.05
    est_state = true_state + errors

    # Generate covariances
    cov_est = np.stack([np.eye(state_dim) * 0.05**2] * N)

    # Compute NEES
    nees = compute_nees(true_state, est_state, cov_est)
    nees_stats = compute_nees_stats(nees, state_dim)

    assert nees.shape == (N,)
    assert "mean" in nees_stats

    # Compute position RMSE (first 2 dimensions)
    pos_rmse = compute_position_rmse(true_state[:, :2], est_state[:, :2])
    assert pos_rmse > 0

    # Compute velocity RMSE (next 2 dimensions)
    vel_rmse = compute_velocity_rmse(true_state[:, 2:4], est_state[:, 2:4])
    assert vel_rmse > 0

    # Compute heading error (last dimension)
    heading_error = compute_heading_error(true_state[:, 4], est_state[:, 4])
    assert heading_error >= 0

    # Generate innovations and compute NIS
    innovations = np.random.randn(N, meas_dim) * 0.1
    innov_cov = np.stack([np.eye(meas_dim) * 0.1**2] * N)
    nis = compute_nis(innovations, innov_cov)
    nis_stats = compute_nis_stats(nis, meas_dim)

    assert nis.shape == (N,)
    assert "mean" in nis_stats

    # Check residual autocorrelation
    residuals = innovations[:, 0]
    acf = compute_residual_autocorrelation(residuals, max_lag=5)
    assert acf.shape == (6,)
    assert_allclose(acf[0], 1.0, atol=1e-10)

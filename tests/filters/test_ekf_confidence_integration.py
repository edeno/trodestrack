"""Test EKF top-level confidence integration (PRD robustness requirement).

This module tests that camera confidence scores are properly wired through
the extended_kalman_filter() API and affect measurement noise scaling.

PRD Reference:
    Section 13: "DLC confidence → measurement noise scaling"
"""

import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.simple import SimpleSimConfig, simulate_stationary


def test_confidence_parameter_exists() -> None:
    """Test that extended_kalman_filter accepts conf_cam parameter.

    This is the first test - verifying the API signature exists.
    Currently FAILS because conf_cam parameter doesn't exist.
    """
    # Simulate stationary scenario
    config_sim = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
    sim = simulate_stationary(config=config_sim, seed=42)

    config = EKFConfig(state_mode="2d_full")

    # Create confidence array (all high confidence)
    conf_cam = np.ones((len(sim["t_cam_exp"]), 4))

    # This should NOT raise TypeError about unexpected keyword argument
    result = extended_kalman_filter(
        ekf_config=config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
        conf_cam=conf_cam,  # NEW PARAMETER
    )

    assert result is not None


def test_low_confidence_increases_uncertainty() -> None:
    """Test that low confidence increases position uncertainty.

    PRD requirement: Low-confidence measurements should be trusted less,
    resulting in larger posterior covariance.

    Currently FAILS because conf_cam is not wired through.
    """
    # Simulate stationary scenario
    config_sim = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
    sim = simulate_stationary(config=config_sim, seed=42)

    config = EKFConfig(state_mode="2d_full")
    N_cam = len(sim["t_cam_exp"])

    # Run 1: High confidence (default)
    conf_high = np.ones((N_cam, 4))
    result_high = extended_kalman_filter(
        ekf_config=config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
        conf_cam=conf_high,
    )

    # Run 2: Low confidence (10% confidence = 10x larger R)
    conf_low = 0.1 * np.ones((N_cam, 4))
    result_low = extended_kalman_filter(
        ekf_config=config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
        conf_cam=conf_low,
    )

    # Position variance should be larger with low confidence
    # Extract final position covariance (x, y components)
    P_high_final = result_high.filtered_covariances[-1][:2, :2]
    P_low_final = result_low.filtered_covariances[-1][:2, :2]

    var_high = np.trace(P_high_final)
    var_low = np.trace(P_low_final)

    # Low confidence should result in >2x larger position variance
    assert var_low > 2.0 * var_high, (
        f"Low confidence should increase variance: "
        f"high={var_high:.6f}, low={var_low:.6f}, ratio={var_low / var_high:.2f}"
    )


def test_confidence_none_defaults_to_high() -> None:
    """Test that conf_cam=None defaults to high confidence (backward compat).

    Currently FAILS because conf_cam parameter doesn't exist.
    """
    # Simulate stationary scenario
    config_sim = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
    sim = simulate_stationary(config=config_sim, seed=42)

    config = EKFConfig(state_mode="2d_full")

    # Run with conf_cam=None (should default to all 1.0)
    result_none = extended_kalman_filter(
        ekf_config=config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
        conf_cam=None,  # Default behavior
    )

    # Run with explicit high confidence
    conf_high = np.ones((len(sim["t_cam_exp"]), 4))
    result_explicit = extended_kalman_filter(
        ekf_config=config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
        conf_cam=conf_high,
    )

    # Results should be identical (within numerical precision)
    np.testing.assert_allclose(
        result_none.filtered_means,
        result_explicit.filtered_means,
        rtol=1e-10,
        atol=1e-12,
        err_msg="conf_cam=None should match explicit high confidence",
    )


def test_varying_confidence_across_frames() -> None:
    """Test that confidence can vary frame-by-frame without breaking filter.

    Simulates realistic scenario where DLC confidence varies over time
    (e.g., occlusions, motion blur).

    Note: In a stationary scenario, variance decreases over time as the
    filter converges. Low confidence means measurements are trusted less,
    so variance decreases MORE SLOWLY (not increases). This test verifies
    the filter handles varying confidence gracefully.
    """
    # Simulate stationary scenario
    config_sim = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
    sim = simulate_stationary(config=config_sim, seed=42)

    config = EKFConfig(state_mode="2d_full")
    N_cam = len(sim["t_cam_exp"])

    # Create time-varying confidence: drops in middle of sequence
    conf_cam = np.ones((N_cam, 4))
    # Middle third has low confidence (simulating occlusion)
    start_idx = N_cam // 3
    end_idx = 2 * N_cam // 3
    conf_cam[start_idx:end_idx, :] = 0.1

    result = extended_kalman_filter(
        ekf_config=config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
        conf_cam=conf_cam,
    )

    # Filter should not diverge - all covariances finite
    assert np.all(
        np.isfinite(result.filtered_covariances)
    ), "Filter diverged with varying confidence"

    # Variance should remain bounded (not explode)
    variances = np.array([np.trace(P[:2, :2]) for P in result.filtered_covariances])
    assert np.max(variances) < 1.0, f"Variance exploded: max={np.max(variances):.4f} m²"

    # Filter should converge overall (final variance < initial variance)
    assert (
        variances[-1] < variances[0]
    ), f"Filter should converge: initial={variances[0]:.6f}, final={variances[-1]:.6f}"


def test_confidence_affects_log_likelihood() -> None:
    """Test that confidence affects marginal log-likelihood.

    Low-confidence measurements should contribute less to log-likelihood
    (larger measurement noise → smaller penalty for innovation).

    Currently FAILS because conf_cam is not wired through.
    """
    # Simulate stationary scenario
    config_sim = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
    sim = simulate_stationary(config=config_sim, seed=42)

    config = EKFConfig(state_mode="2d_full")
    N_cam = len(sim["t_cam_exp"])

    # High confidence
    conf_high = np.ones((N_cam, 4))
    result_high = extended_kalman_filter(
        ekf_config=config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
        conf_cam=conf_high,
    )

    # Low confidence
    conf_low = 0.1 * np.ones((N_cam, 4))
    result_low = extended_kalman_filter(
        ekf_config=config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
        conf_cam=conf_low,
    )

    # Log-likelihoods should differ
    # Low confidence affects both log(det(S)) and innovation terms
    ll_high = result_high.marginal_loglik
    ll_low = result_low.marginal_loglik

    # Log-likelihoods should differ substantially (>5% relative difference)
    # Direction depends on innovation magnitude vs covariance scaling
    # What matters is that confidence scaling affects the likelihood
    rel_diff = abs(ll_high - ll_low) / abs(ll_high)
    assert rel_diff > 0.05, (
        f"Confidence should affect log-likelihood (>5% change): "
        f"ll_high={ll_high:.2f}, ll_low={ll_low:.2f}, rel_diff={rel_diff:.2%}"
    )

"""Test UKF partial observation handling (CR-2 from FULL_REVIEW.md).

This module tests that the UKF correctly handles single-LED observations
without spuriously reducing covariance for missing measurements.

Critical bug fix: UKF was computing Kalman gain from full 4×4 innovation
covariance even when only 1 LED was valid, causing overconfident estimates.
"""

import numpy as np
import pytest

from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu


def _prepare_ukf_inputs(sim):
    """Extract UKF input parameters from simulation output."""
    return {
        "t_imu": sim["t_imu"],
        "U_imu": sim["U_imu"],
        "t_cam": sim["t_cam_exp"],  # Note: simulation uses "t_cam_exp"
        "Z_cam_led1": sim["Z_cam_led1"],
        "Z_cam_led2": sim["Z_cam_led2"],
        "mask_cam": sim["mask_cam"],
    }


def test_ukf_single_led_no_spurious_covariance_reduction() -> None:
    """UKF with single LED should not reduce covariance for missing LED.

    Critical test for CR-2: Verify that when only LED1 is observed, the
    filter does NOT reduce uncertainty for LED2 dimensions.

    The fix uses lifted subspace operators (same as EKF) to compute the
    Kalman gain in the active 2D subspace, preventing spurious information
    gain from missing observations.
    """
    # Create scenario with both LEDs enabled
    # Use rat_imu simulation to get realistic dual-LED tracking
    sim_config = RatIMUSimConfig(
        duration_s=5.0,
        use_second_led=True,  # Enable both LEDs
        led_swap_prob=0.0,  # No swaps
        cam_dropout_prob=0.0,  # No dropouts
    )
    sim = simulate_rat_imu(config=sim_config, seed=42)
    config = UKFConfig()

    # Run with both LEDs (baseline)
    result_both = unscented_kalman_filter(config, **_prepare_ukf_inputs(sim))

    # Run with only LED1 (set LED2 to NaN)
    sim_led1_only = dict(sim)
    sim_led1_only["Z_cam_led2"] = np.full_like(sim["Z_cam_led2"], np.nan)
    result_led1 = unscented_kalman_filter(config, **_prepare_ukf_inputs(sim_led1_only))

    # Critical test for CR-2: Single LED should have higher final uncertainty than dual-LED
    # This verifies that the filter doesn't spuriously reduce covariance for missing observations
    final_pos_var_both = result_both.filtered_covariances[-1, 0, 0]
    final_pos_var_led1 = result_led1.filtered_covariances[-1, 0, 0]

    # Single LED should be less confident (larger final variance)
    assert final_pos_var_led1 > final_pos_var_both, (
        f"Single-LED should have higher final uncertainty than dual-LED: {final_pos_var_led1} > {final_pos_var_both}"
    )

    # Single LED should not become overconfident (variance should remain reasonable)
    # With proper subspace handling, variance should stay above a reasonable threshold
    # For 5mm measurement noise, we expect at least 0.01 mm² variance
    assert final_pos_var_led1 > 1e-5, (
        f"Single-LED should not become overconfident (CR-2 bug check): {final_pos_var_led1} > 1e-5"
    )

    # Both filters should remain stable (no NaN/Inf)
    assert np.all(np.isfinite(result_both.filtered_means)), (
        "Dual-LED: means should be finite"
    )
    assert np.all(np.isfinite(result_both.filtered_covariances)), (
        "Dual-LED: covariances should be finite"
    )
    assert np.all(np.isfinite(result_led1.filtered_means)), (
        "Single-LED: means should be finite"
    )
    assert np.all(np.isfinite(result_led1.filtered_covariances)), (
        "Single-LED: covariances should be finite"
    )


def test_ukf_led2_only_symmetry() -> None:
    """Verify LED2-only produces similar results to LED1-only (symmetry check)."""
    sim_config = RatIMUSimConfig(
        duration_s=5.0,
        use_second_led=True,
        led_swap_prob=0.0,
        cam_dropout_prob=0.0,
    )
    sim = simulate_rat_imu(config=sim_config, seed=42)
    config = UKFConfig()

    # LED1-only
    sim_led1 = dict(sim)
    sim_led1["Z_cam_led2"] = np.full_like(sim["Z_cam_led2"], np.nan)
    result_led1 = unscented_kalman_filter(config, **_prepare_ukf_inputs(sim_led1))

    # LED2-only
    sim_led2 = dict(sim)
    sim_led2["Z_cam_led1"] = np.full_like(sim["Z_cam_led1"], np.nan)
    result_led2 = unscented_kalman_filter(config, **_prepare_ukf_inputs(sim_led2))

    # Final position uncertainties should be similar (not identical due to different geometries)
    final_var_led1 = result_led1.filtered_covariances[-1, 0, 0]
    final_var_led2 = result_led2.filtered_covariances[-1, 0, 0]

    # Allow 20% difference due to potential asymmetry in initial conditions
    assert np.isclose(final_var_led1, final_var_led2, rtol=0.2), (
        "LED1-only and LED2-only should have similar final uncertainties"
    )


def test_ukf_alternating_leds_maintains_stability() -> None:
    """Test that alternating between LED1 and LED2 maintains stable covariance."""
    sim_config = RatIMUSimConfig(
        duration_s=10.0,
        use_second_led=True,
        led_swap_prob=0.0,
        cam_dropout_prob=0.0,
    )
    sim = simulate_rat_imu(config=sim_config, seed=42)
    config = UKFConfig()

    # Create alternating pattern: LED1 for even frames, LED2 for odd frames
    sim_alt = dict(sim)

    Z_led1 = sim["Z_cam_led1"].copy()
    Z_led2 = sim["Z_cam_led2"].copy()

    # Mask alternating frames
    Z_led1[1::2, :] = np.nan  # LED1 missing on odd frames
    Z_led2[0::2, :] = np.nan  # LED2 missing on even frames

    sim_alt["Z_cam_led1"] = Z_led1
    sim_alt["Z_cam_led2"] = Z_led2

    result = unscented_kalman_filter(config, **_prepare_ukf_inputs(sim_alt))

    # Covariance should not explode or collapse
    covs = result.filtered_covariances[:, 0, 0]  # Position x variance

    # Check no explosion
    assert np.all(covs < 1.0), "Covariance should not explode with alternating LEDs"

    # Check no collapse (overconfidence)
    # With proper subspace handling, uncertainty should stay reasonable
    assert np.all(covs[10:] > 1e-5), (
        "Covariance should not collapse to near-zero (overconfidence)"
    )

    # Check stability (no oscillations)
    # Variance should be relatively smooth (no huge jumps)
    cov_diffs = np.abs(np.diff(covs))
    assert np.percentile(cov_diffs, 95) < 0.1, (
        "Covariance should be stable (no large oscillations)"
    )


def test_ukf_gradual_led_dropout() -> None:
    """Test UKF handles gradual transition from dual-LED to single-LED."""
    sim_config = RatIMUSimConfig(
        duration_s=10.0,
        use_second_led=True,
        led_swap_prob=0.0,
        cam_dropout_prob=0.0,
    )
    sim = simulate_rat_imu(config=sim_config, seed=42)
    config = UKFConfig()

    # Start with both LEDs, then drop LED2 halfway through
    sim_dropout = dict(sim)
    n_cam = len(sim["t_cam_exp"])  # Use t_cam_exp (standard key) not t_cam
    dropout_frame = n_cam // 2

    Z_led2 = sim["Z_cam_led2"].copy()
    Z_led2[dropout_frame:, :] = np.nan
    sim_dropout["Z_cam_led2"] = Z_led2

    result = unscented_kalman_filter(config, **_prepare_ukf_inputs(sim_dropout))

    # Before dropout: dual-LED (lower uncertainty)
    pre_dropout_var = result.filtered_covariances[dropout_frame - 1, 0, 0]

    # After dropout: single-LED (higher uncertainty)
    post_dropout_var = result.filtered_covariances[
        dropout_frame + 5, 0, 0
    ]  # Give it a few frames

    # Uncertainty should increase after LED2 drops out
    assert post_dropout_var > pre_dropout_var, (
        "Uncertainty should increase when transitioning from dual-LED to single-LED"
    )

    # Filter should remain stable (no NaN/Inf)
    assert np.all(np.isfinite(result.filtered_means)), "Means should remain finite"
    assert np.all(np.isfinite(result.filtered_covariances)), (
        "Covariances should remain finite"
    )


def test_ukf_no_leds_skips_update() -> None:
    """Test that UKF skips measurement update when no LEDs are valid.

    When all LED observations are NaN (complete dropout), the filter should
    run in prediction-only mode and remain stable. Covariance should grow
    due to process noise accumulation.
    """
    sim_config = RatIMUSimConfig(
        duration_s=5.0,
        use_second_led=True,
        led_swap_prob=0.0,
        cam_dropout_prob=0.0,
    )
    sim = simulate_rat_imu(config=sim_config, seed=42)
    # Disable gating for this edge case (all measurements are NaN)
    config = UKFConfig(
        use_mahalanobis_gating=False,
        enable_zupt=True,
        zupt_velocity_threshold=0.05,
        zupt_measurement_noise=1e-4,
    )

    # Create scenario with no LED observations (complete dropout)
    sim_no_leds = dict(sim)
    sim_no_leds["Z_cam_led1"] = np.full_like(sim["Z_cam_led1"], np.nan)
    sim_no_leds["Z_cam_led2"] = np.full_like(sim["Z_cam_led2"], np.nan)

    result = unscented_kalman_filter(config, **_prepare_ukf_inputs(sim_no_leds))
    result_no_zupt = unscented_kalman_filter(
        UKFConfig(use_mahalanobis_gating=False, enable_zupt=False),
        **_prepare_ukf_inputs(sim_no_leds),
    )

    # With no measurements, covariance should grow (prediction-only)
    initial_var = result.filtered_covariances[0, 0, 0]
    final_var = result.filtered_covariances[-1, 0, 0]

    assert final_var > initial_var, (
        "Covariance should grow with prediction-only (no measurement updates)"
    )

    np.testing.assert_allclose(
        result.filtered_covariances[0, 2, 2],
        result_no_zupt.filtered_covariances[0, 2, 2],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.filtered_covariances[0, 3, 3],
        result_no_zupt.filtered_covariances[0, 3, 3],
        atol=1e-7,
    )

    # Filter should remain stable
    assert np.all(np.isfinite(result.filtered_means)), "Means should remain finite"
    assert np.all(np.isfinite(result.filtered_covariances)), (
        "Covariances should remain finite"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

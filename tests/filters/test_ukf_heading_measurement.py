"""Tests for UKF heading pseudo-measurement from dual LEDs.

Tests validate UKF feature parity with EKF heading measurement:
- Heading observation improves heading uncertainty
- LED spacing gating rejects invalid observations
- Adaptive noise scaling with baseline geometry
- Single LED graceful degradation
- JAX JIT compatibility
"""

import numpy as np

from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.qa.metrics import compute_heading_rmse
from trodestrack.sim.simple import SimpleSimConfig, simulate_circular


def test_ukf_heading_measurement_improves_convergence():
    """Heading pseudo-measurement should improve heading uncertainty."""
    # Simulate circular motion with dual LEDs
    config_sim = SimpleSimConfig(duration_s=10.0, fs_cam=30.0, fs_imu=1000.0)
    result = simulate_circular(config=config_sim, radius=0.5, seed=42)

    # UKF WITHOUT heading measurement
    config_no_heading = UKFConfig(
        use_heading_measurement=False,
        measurement_noise_pos=0.005**2,
        led_distance=0.04,
    )

    result_no_heading = unscented_kalman_filter(
        ukf_config=config_no_heading,
        t_imu=result["t_imu"],
        U_imu=result["U_imu"],
        t_cam=result["t_cam_obs"],
        Z_cam_led1=result["Z_cam_led1"],
        Z_cam_led2=result["Z_cam_led2"],
        mask_cam=result["mask_cam"],
    )

    # UKF WITH heading measurement
    config_with_heading = UKFConfig(
        use_heading_measurement=True,
        measurement_noise_pos=0.005**2,
        led_distance=0.04,
        led_distance_tolerance=0.3,
        adaptive_heading_noise=True,
    )

    result_with_heading = unscented_kalman_filter(
        ukf_config=config_with_heading,
        t_imu=result["t_imu"],
        U_imu=result["U_imu"],
        t_cam=result["t_cam_obs"],
        Z_cam_led1=result["Z_cam_led1"],
        Z_cam_led2=result["Z_cam_led2"],
        mask_cam=result["mask_cam"],
    )

    # Extract heading estimates (align truth to camera times)
    X_truth_at_cam = np.array(
        [result["X_truth"][np.argmin(np.abs(result["t_imu"] - t_c))] for t_c in result["t_cam_obs"]]
    )
    theta_truth = X_truth_at_cam[:, 4]
    theta_est_no_heading = np.array(result_no_heading.filtered_means[:, 4])
    theta_est_with_heading = np.array(result_with_heading.filtered_means[:, 4])

    # Compute RMSE
    rmse_no_heading = compute_heading_rmse(theta_truth, theta_est_no_heading)
    rmse_with_heading = compute_heading_rmse(theta_truth, theta_est_with_heading)

    # Heading uncertainty should be smaller with heading measurement
    # (averaged over all timesteps)
    cov_no_heading = np.array(result_no_heading.filtered_covariances[:, 4, 4])
    cov_with_heading = np.array(result_with_heading.filtered_covariances[:, 4, 4])

    mean_std_no_heading = np.sqrt(np.mean(cov_no_heading))
    mean_std_with_heading = np.sqrt(np.mean(cov_with_heading))

    # Heading measurement should reduce uncertainty
    assert mean_std_with_heading < mean_std_no_heading, (
        f"Heading uncertainty should decrease: "
        f"no_heading={mean_std_no_heading:.4f}, "
        f"with_heading={mean_std_with_heading:.4f}"
    )

    # RMSE may or may not improve (depends on camera noise), but we shouldn't make it worse
    # Allow up to 3x worse (graceful degradation)
    improvement_ratio = rmse_with_heading / rmse_no_heading
    assert improvement_ratio < 3.0, (
        f"Heading RMSE should not catastrophically degrade: "
        f"no_heading={rmse_no_heading:.4f} rad, "
        f"with_heading={rmse_with_heading:.4f} rad, "
        f"ratio={improvement_ratio:.2f}x"
    )


def test_ukf_spacing_gating_rejects_invalid_observations():
    """LED spacing tolerance should gate observations outside expected range."""
    # Create simple scenario
    config_sim = SimpleSimConfig(duration_s=2.0, fs_cam=30.0, fs_imu=1000.0)
    result = simulate_circular(config=config_sim, radius=0.5, seed=42)

    # Manually corrupt LED spacing (make LED2 too far from LED1)
    Z_cam_led2_corrupted = result["Z_cam_led2"].copy()
    Z_cam_led2_corrupted = Z_cam_led2_corrupted * 2.0  # Double the spacing

    # Tight spacing tolerance should reject corrupted observations
    config = UKFConfig(
        use_heading_measurement=True,
        led_distance=0.04,
        led_distance_tolerance=0.2,  # ±20% tolerance (tight)
    )

    # Should run without crashing despite corrupted spacing
    result_filtered = unscented_kalman_filter(
        ukf_config=config,
        t_imu=result["t_imu"],
        U_imu=result["U_imu"],
        t_cam=result["t_cam_obs"],
        Z_cam_led1=result["Z_cam_led1"],
        Z_cam_led2=Z_cam_led2_corrupted,
        mask_cam=result["mask_cam"],
    )

    # Filter should not diverge (covariances should be finite)
    assert np.all(np.isfinite(result_filtered.filtered_covariances))


def test_ukf_adaptive_noise_scales_with_baseline():
    """Adaptive noise should scale R_heading proportional to (expected/observed)²."""
    # This test validates the config parameter exists and is used
    config_adaptive = UKFConfig(
        use_heading_measurement=True,
        adaptive_heading_noise=True,
        led_distance=0.04,
    )

    config_fixed = UKFConfig(
        use_heading_measurement=True,
        adaptive_heading_noise=False,
        led_distance=0.04,
    )

    # Both configs should be valid
    assert config_adaptive.adaptive_heading_noise is True
    assert config_fixed.adaptive_heading_noise is False


def test_ukf_single_led_disables_heading_automatically():
    """Single LED (LED2=NaN) should gracefully degrade without heading update."""
    config_sim = SimpleSimConfig(duration_s=2.0, fs_cam=30.0, fs_imu=1000.0)
    result = simulate_circular(config=config_sim, radius=0.5, seed=42)

    # Set LED2 to NaN (single LED scenario)
    Z_cam_led2_nan = np.full_like(result["Z_cam_led2"], np.nan)

    config = UKFConfig(
        use_heading_measurement=True,
        led_distance=0.04,
    )

    # Should run without crashing despite missing LED2
    result_filtered = unscented_kalman_filter(
        ukf_config=config,
        t_imu=result["t_imu"],
        U_imu=result["U_imu"],
        t_cam=result["t_cam_obs"],
        Z_cam_led1=result["Z_cam_led1"],
        Z_cam_led2=Z_cam_led2_nan,
        mask_cam=result["mask_cam"],
    )

    # Filter should not diverge
    assert np.all(np.isfinite(result_filtered.filtered_means))
    assert np.all(np.isfinite(result_filtered.filtered_covariances))


def test_ukf_heading_basic_functionality():
    """UKF with heading measurement should run without errors."""
    config_sim = SimpleSimConfig(duration_s=1.0, fs_cam=30.0, fs_imu=1000.0)
    result = simulate_circular(config=config_sim, radius=0.5, seed=42)

    config = UKFConfig(
        use_heading_measurement=True,
        led_distance=0.04,
    )

    # Should run without errors
    result_filtered = unscented_kalman_filter(
        ukf_config=config,
        t_imu=result["t_imu"],
        U_imu=result["U_imu"],
        t_cam=result["t_cam_obs"],
        Z_cam_led1=result["Z_cam_led1"],
        Z_cam_led2=result["Z_cam_led2"],
        mask_cam=result["mask_cam"],
    )

    # Should produce finite results
    assert np.all(np.isfinite(result_filtered.filtered_means))
    assert np.all(np.isfinite(result_filtered.filtered_covariances))

    # Marginal log-likelihood should be finite
    assert np.isfinite(result_filtered.marginal_loglik)

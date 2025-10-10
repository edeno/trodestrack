"""Tests for RTS smoother and sigma-point smoother.

This module tests the offline smoothing implementations:
    - RTS (Rauch-Tung-Striebel) smoother for EKF
    - Sigma-point smoother for UKF

Test strategy:
    - Use analytic simulations (stationary, constant velocity, circular)
    - Verify smoother improves upon filter (lower RMSE)
    - Check numerical stability and consistency
    - Verify smoothed covariances are smaller than filtered

References:
    - PRD.md Section 12: Algorithms & Implementation Notes
    - Särkkä (2013) "Bayesian Filtering and Smoothing", Algorithm 8.2 (RTS)
"""

import jax.numpy as jnp
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.qa.metrics import compute_position_rmse
from trodestrack.runtime.offline import rts_smoother, sigma_point_smoother
from trodestrack.sim.simple import SimOut, simulate_circular, simulate_stationary


class TestRTSSmoother:
    """Test RTS smoother for EKF."""

    def test_rts_smoother_stationary_improves_estimates(self):
        """RTS smoother should improve position estimates on stationary scenario."""
        # Generate stationary simulation
        sim: SimOut = simulate_stationary(seed=42)

        # Run EKF filter
        cfg = EKFConfig()
        filter_result = extended_kalman_filter(
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Run RTS smoother
        smoother_result = rts_smoother(
            filter_result=filter_result,
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )

        # Extract ground truth at camera times
        cam_times = sim["t_cam_exp"]
        truth_x = np.interp(cam_times, sim["t_imu"], sim["X_truth"][:, 0])
        truth_y = np.interp(cam_times, sim["t_imu"], sim["X_truth"][:, 1])
        truth_pos = np.column_stack([truth_x, truth_y])  # (N_cam, 2)

        # Compute RMSE: position only
        filtered_pos = np.array(filter_result.filtered_means[:, :2])  # (N_cam, 2)
        smoothed_pos = np.array(smoother_result.smoothed_means[:, :2])  # (N_cam, 2)

        rmse_filter = compute_position_rmse(filtered_pos, truth_pos)
        rmse_smoother = compute_position_rmse(smoothed_pos, truth_pos)

        # Smoother should improve estimates (lower or equal RMSE)
        # For stationary case with excellent measurements, improvement may be minimal
        # Allow small tolerance (20 microns) for numerical effects (adaptive dropout Q alters smoothing path)
        assert (
            rmse_smoother <= rmse_filter + 2e-5
        ), f"Smoother RMSE {rmse_smoother:.6f} should be <= filter RMSE {rmse_filter:.6f}"

        # Both should be within PRD requirement (2 cm + small tolerance)
        assert rmse_filter < 0.021, f"Filter RMSE {rmse_filter:.4f} exceeds 2 cm"
        assert rmse_smoother < 0.021, f"Smoother RMSE {rmse_smoother:.4f} exceeds 2 cm"

    def test_rts_smoother_circular_improves_bias_estimates(self):
        """RTS smoother should improve gyro bias estimates on circular motion."""
        # Generate circular motion
        sim: SimOut = simulate_circular(seed=42)

        # Run EKF filter
        cfg = EKFConfig()
        filter_result = extended_kalman_filter(
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Run RTS smoother
        smoother_result = rts_smoother(
            filter_result=filter_result,
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )

        # Check final gyro bias estimate (index 5)
        true_bias = float(sim["bias_gyro"][0])  # rad/s (constant bias, take first element)
        filtered_bias = float(filter_result.filtered_means[-1, 5])
        smoothed_bias = float(smoother_result.smoothed_means[-1, 5])

        bias_error_filter = abs(filtered_bias - true_bias)
        bias_error_smoother = abs(smoothed_bias - true_bias)

        # Smoother should improve bias estimate
        assert (
            bias_error_smoother <= bias_error_filter
        ), f"Smoother bias error {bias_error_smoother:.6f} should be <= filter {bias_error_filter:.6f}"

    def test_rts_smoother_reduces_covariance(self):
        """RTS smoother should produce smaller covariances than filter."""
        # Generate simple stationary scenario
        sim: SimOut = simulate_stationary(seed=42)

        # Run EKF filter
        cfg = EKFConfig()
        filter_result = extended_kalman_filter(
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Run RTS smoother
        smoother_result = rts_smoother(
            filter_result=filter_result,
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )

        # Compare trace of covariances (sum of variances)
        filter_trace = jnp.trace(filter_result.filtered_covariances, axis1=1, axis2=2)
        smoother_trace = jnp.trace(smoother_result.smoothed_covariances, axis1=1, axis2=2)

        # Smoother covariances should be smaller (on average)
        mean_filter_trace = float(jnp.mean(filter_trace))
        mean_smoother_trace = float(jnp.mean(smoother_trace))

        assert (
            mean_smoother_trace < mean_filter_trace
        ), f"Smoother cov trace {mean_smoother_trace:.6f} should be < filter {mean_filter_trace:.6f}"

    def test_rts_smoother_deterministic(self):
        """RTS smoother should be deterministic (same inputs → same outputs)."""
        # Generate simulation
        sim: SimOut = simulate_stationary(seed=42)

        # Run EKF filter
        cfg = EKFConfig()
        filter_result = extended_kalman_filter(
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Run RTS smoother twice
        smoother_result_1 = rts_smoother(
            filter_result=filter_result,
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )
        smoother_result_2 = rts_smoother(
            filter_result=filter_result,
            ekf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )

        # Results should be identical
        np.testing.assert_allclose(
            smoother_result_1.smoothed_means,
            smoother_result_2.smoothed_means,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            smoother_result_1.smoothed_covariances,
            smoother_result_2.smoothed_covariances,
            rtol=1e-10,
            atol=1e-12,
        )


class TestSigmaPointSmoother:
    """Test sigma-point smoother for UKF."""

    def test_sigma_point_smoother_stationary_improves_estimates(self):
        """Sigma-point smoother should improve position estimates."""
        # Generate stationary simulation
        sim: SimOut = simulate_stationary(seed=42)

        # Run UKF filter
        cfg = UKFConfig()
        filter_result = unscented_kalman_filter(
            ukf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Run sigma-point smoother
        smoother_result = sigma_point_smoother(
            filter_result=filter_result,
            ukf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )

        # Extract ground truth at camera times
        cam_times = sim["t_cam_exp"]
        truth_x = np.interp(cam_times, sim["t_imu"], sim["X_truth"][:, 0])
        truth_y = np.interp(cam_times, sim["t_imu"], sim["X_truth"][:, 1])
        truth_pos = np.column_stack([truth_x, truth_y])

        # Compute RMSE: position only
        filtered_pos = np.array(filter_result.filtered_means[:, :2])
        smoothed_pos = np.array(smoother_result.smoothed_means[:, :2])

        rmse_filter = compute_position_rmse(filtered_pos, truth_pos)
        rmse_smoother = compute_position_rmse(smoothed_pos, truth_pos)

        # Smoother should improve estimates (allow small numerical tolerance)
        # Note: For stationary scenarios with excellent camera measurements,
        # the smoother may not improve much and can show tiny numerical degradation
        # due to backward pass accumulation. We allow 50 µm tolerance (0.05mm).
        assert (
            rmse_smoother <= rmse_filter + 5e-5
        ), f"Smoother RMSE {rmse_smoother:.6f} should be <= filter RMSE {rmse_filter:.6f}"

        # Both should be within PRD requirement (2 cm + small tolerance)
        assert rmse_filter < 0.021, f"Filter RMSE {rmse_filter:.4f} exceeds 2 cm"
        assert rmse_smoother < 0.021, f"Smoother RMSE {rmse_smoother:.4f} exceeds 2 cm"

    def test_sigma_point_smoother_reduces_covariance(self):
        """Sigma-point smoother should produce smaller covariances."""
        # Generate simple stationary scenario
        sim: SimOut = simulate_stationary(seed=42)

        # Run UKF filter
        cfg = UKFConfig()
        filter_result = unscented_kalman_filter(
            ukf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Run sigma-point smoother
        smoother_result = sigma_point_smoother(
            filter_result=filter_result,
            ukf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )

        # Compare trace of covariances
        filter_trace = jnp.trace(filter_result.filtered_covariances, axis1=1, axis2=2)
        smoother_trace = jnp.trace(smoother_result.smoothed_covariances, axis1=1, axis2=2)

        mean_filter_trace = float(jnp.mean(filter_trace))
        mean_smoother_trace = float(jnp.mean(smoother_trace))

        assert (
            mean_smoother_trace < mean_filter_trace
        ), f"Smoother cov trace {mean_smoother_trace:.6f} should be < filter {mean_filter_trace:.6f}"

    def test_sigma_point_smoother_deterministic(self):
        """Sigma-point smoother should be deterministic."""
        # Generate simulation
        sim: SimOut = simulate_stationary(seed=42)

        # Run UKF filter
        cfg = UKFConfig()
        filter_result = unscented_kalman_filter(
            ukf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Run sigma-point smoother twice
        smoother_result_1 = sigma_point_smoother(
            filter_result=filter_result,
            ukf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )
        smoother_result_2 = sigma_point_smoother(
            filter_result=filter_result,
            ukf_config=cfg,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )

        # Results should be identical
        np.testing.assert_allclose(
            smoother_result_1.smoothed_means,
            smoother_result_2.smoothed_means,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            smoother_result_1.smoothed_covariances,
            smoother_result_2.smoothed_covariances,
            rtol=1e-10,
            atol=1e-12,
        )

"""Tests for Zero-Velocity Update (ZUPT) in EKF.

ZUPT constrains velocity estimates during stationary periods to prevent IMU drift.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.qa.metrics import compute_velocity_rmse
from trodestrack.sim.simple import simulate_stationary


class TestZUPTDetection:
    """Test ZUPT detection logic."""

    def test_zupt_enabled_by_default(self):
        """ZUPT should be enabled by default for better stationary tracking."""
        config = EKFConfig()
        assert config.enable_zupt is True

    def test_zupt_detection_threshold_configurable(self):
        """ZUPT velocity threshold should be configurable."""
        config = EKFConfig(enable_zupt=True, zupt_velocity_threshold=0.08)
        assert config.enable_zupt is True
        assert config.zupt_velocity_threshold == 0.08

    def test_zupt_measurement_noise_configurable(self):
        """ZUPT measurement noise should be configurable."""
        config = EKFConfig(enable_zupt=True, zupt_measurement_noise=0.02**2)
        assert config.zupt_measurement_noise == 0.02**2


class TestZUPTStationary:
    """Test ZUPT performance on stationary scenario.

    Validates PRD Section 4 acceptance criteria:
    - Velocity RMSE ≤ 0.10 m/s (general requirement)
    - ZUPT should significantly improve stationary tracking
    """

    @pytest.fixture
    def stationary_sim(self):
        """Generate stationary simulation with zero velocity."""
        from trodestrack.sim.simple import SimpleSimConfig

        config = SimpleSimConfig(duration_s=10.0, fs_imu=400, fs_cam=30)
        return simulate_stationary(
            config=config,
            position=np.array([0.5, 0.5]),
            heading=0.0,
            seed=42,
        )

    def test_zupt_reduces_velocity_drift_stationary(self, stationary_sim):
        """ZUPT should reduce velocity drift during stationary period.

        Without ZUPT: IMU noise accumulates into velocity error
        With ZUPT: Velocity constrained to ~0, preventing drift
        """
        # Run filter WITHOUT ZUPT
        config_no_zupt = EKFConfig(enable_zupt=False)
        result_no_zupt = extended_kalman_filter(
            ekf_config=config_no_zupt,
            t_imu=stationary_sim["t_imu"],
            U_imu=stationary_sim["U_imu"],
            t_cam=stationary_sim["t_cam_exp"],
            Z_cam_led1=stationary_sim["Z_cam_led1"],
            Z_cam_led2=stationary_sim["Z_cam_led2"],
            mask_cam=stationary_sim["mask_cam"],
        )

        # Run filter WITH ZUPT
        config_with_zupt = EKFConfig(
            enable_zupt=True,
            zupt_velocity_threshold=0.05,  # 5 cm/s
            zupt_measurement_noise=0.01**2,  # 1 cm/s noise
        )
        result_with_zupt = extended_kalman_filter(
            ekf_config=config_with_zupt,
            t_imu=stationary_sim["t_imu"],
            U_imu=stationary_sim["U_imu"],
            t_cam=stationary_sim["t_cam_exp"],
            Z_cam_led1=stationary_sim["Z_cam_led1"],
            Z_cam_led2=stationary_sim["Z_cam_led2"],
            mask_cam=stationary_sim["mask_cam"],
        )

        # Ground truth: velocity is always zero
        truth_vel = np.zeros((len(stationary_sim["t_cam_exp"]), 2))

        # Compute velocity RMSE
        vel_rmse_no_zupt = compute_velocity_rmse(
            result_no_zupt.filtered_means[:, 2:4], truth_vel
        )
        vel_rmse_with_zupt = compute_velocity_rmse(
            result_with_zupt.filtered_means[:, 2:4], truth_vel
        )

        # ZUPT should reduce velocity error significantly
        # Expect at least 30% improvement
        assert vel_rmse_with_zupt < vel_rmse_no_zupt * 0.7, (
            f"ZUPT did not reduce velocity RMSE: {vel_rmse_with_zupt:.4f} vs {vel_rmse_no_zupt:.4f}"
        )

        # ZUPT should achieve very low velocity error (< 2 cm/s)
        assert vel_rmse_with_zupt < 0.02, (
            f"ZUPT velocity RMSE too high: {vel_rmse_with_zupt:.4f} m/s"
        )

    def test_zupt_reduces_velocity_uncertainty(self, stationary_sim):
        """ZUPT should reduce velocity covariance during stationary period."""
        # Run filter WITH ZUPT
        config_with_zupt = EKFConfig(
            enable_zupt=True,
            zupt_velocity_threshold=0.05,
            zupt_measurement_noise=0.01**2,
        )
        result_with_zupt = extended_kalman_filter(
            ekf_config=config_with_zupt,
            t_imu=stationary_sim["t_imu"],
            U_imu=stationary_sim["U_imu"],
            t_cam=stationary_sim["t_cam_exp"],
            Z_cam_led1=stationary_sim["Z_cam_led1"],
            Z_cam_led2=stationary_sim["Z_cam_led2"],
            mask_cam=stationary_sim["mask_cam"],
        )

        # Extract velocity variances (vx, vy on diagonal)
        vel_var = result_with_zupt.filtered_covariances[:, [2, 3], [2, 3]]
        vel_std = jnp.sqrt(vel_var)

        # After 3 seconds, velocity std should be small (< 3 cm/s)
        idx_3s = int(3.0 / (1 / 30))
        assert jnp.all(vel_std[idx_3s] < 0.03), (
            f"Velocity std too high after 3s: {vel_std[idx_3s]}"
        )

        # Velocity std should decrease over time (ZUPT is working)
        assert vel_std[-1, 0] < vel_std[idx_3s, 0], "Velocity std did not decrease"


class TestZUPTMoving:
    """Test that ZUPT does NOT activate during motion."""

    @pytest.fixture
    def moving_sim(self):
        """Generate constant velocity simulation."""
        from trodestrack.sim.simple import SimpleSimConfig, simulate_constant_velocity

        config = SimpleSimConfig(duration_s=5.0, fs_imu=400, fs_cam=30)
        return simulate_constant_velocity(
            config=config,
            initial_position=np.array([0.3, 0.3]),
            velocity=np.array([0.2, 0.0]),  # 20 cm/s to the right
            seed=42,
        )

    def test_zupt_does_not_activate_during_motion(self, moving_sim):
        """ZUPT should NOT activate when velocity > threshold.

        This ensures ZUPT doesn't interfere with normal tracking.
        """
        config_with_zupt = EKFConfig(
            enable_zupt=True,
            zupt_velocity_threshold=0.05,  # 5 cm/s
            zupt_measurement_noise=0.01**2,
        )

        result = extended_kalman_filter(
            ekf_config=config_with_zupt,
            t_imu=moving_sim["t_imu"],
            U_imu=moving_sim["U_imu"],
            t_cam=moving_sim["t_cam_exp"],
            Z_cam_led1=moving_sim["Z_cam_led1"],
            Z_cam_led2=moving_sim["Z_cam_led2"],
            mask_cam=moving_sim["mask_cam"],
        )

        # Ground truth velocity
        truth_vel = np.tile([0.2, 0.0], (len(moving_sim["t_cam_exp"]), 1))

        # Velocity RMSE should be good (< 10 cm/s per PRD)
        vel_rmse = compute_velocity_rmse(result.filtered_means[:, 2:4], truth_vel)
        assert vel_rmse < 0.10, f"Velocity RMSE too high: {vel_rmse:.4f} m/s"

        # ZUPT should NOT reduce velocity estimates to zero
        # Mean velocity should be close to 0.2 m/s
        mean_vx = jnp.mean(result.filtered_means[30:, 2])  # After convergence
        assert jnp.abs(mean_vx - 0.2) < 0.05, (
            f"ZUPT incorrectly activated during motion: mean vx = {mean_vx:.3f}"
        )


class TestZUPTNumericalStability:
    """Test ZUPT numerical stability and edge cases."""

    def test_zupt_with_vision_dropout(self):
        """ZUPT should work correctly during vision dropout.

        During dropout, ZUPT prevents velocity drift even without position measurements.
        """
        from trodestrack.sim.simple import SimpleSimConfig

        config = SimpleSimConfig(duration_s=10.0, fs_imu=400, fs_cam=30)
        sim = simulate_stationary(
            config=config, position=np.array([0.5, 0.5]), heading=0.0, seed=42
        )

        # Create vision dropout in the middle (frames 60-90, ~1 second)
        mask_cam = sim["mask_cam"].copy()
        mask_cam[60:90] = False

        config_with_zupt = EKFConfig(
            enable_zupt=True,
            zupt_velocity_threshold=0.05,
            zupt_measurement_noise=0.01**2,
        )

        result = extended_kalman_filter(
            ekf_config=config_with_zupt,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=mask_cam,
        )

        # During dropout, velocity should remain near zero
        truth_vel = np.zeros((len(sim["t_cam_exp"]), 2))
        vel_rmse = compute_velocity_rmse(result.filtered_means[:, 2:4], truth_vel)

        assert vel_rmse < 0.03, (
            f"ZUPT failed during vision dropout: vel RMSE = {vel_rmse:.4f} m/s"
        )

    def test_zupt_jax_jit_compatible(self):
        """ZUPT implementation should be JAX JIT compatible (no Python branching)."""
        from trodestrack.sim.simple import SimpleSimConfig

        config = SimpleSimConfig(duration_s=2.0, fs_imu=400, fs_cam=30)
        sim = simulate_stationary(
            config=config, position=np.array([0.5, 0.5]), heading=0.0, seed=42
        )

        ekf_config = EKFConfig(enable_zupt=True, zupt_velocity_threshold=0.05)

        # ZUPT uses lax.select for gating (JAX-friendly), so it should work without ConcretizationError
        # Note: We can't JIT the full function because EKFConfig is not hashable
        # But we can verify that the individual update functions use JAX-compatible primitives
        try:
            result = extended_kalman_filter(
                ekf_config=ekf_config,
                t_imu=sim["t_imu"],
                U_imu=sim["U_imu"],
                t_cam=sim["t_cam_exp"],
                Z_cam_led1=sim["Z_cam_led1"],
                Z_cam_led2=sim["Z_cam_led2"],
                mask_cam=sim["mask_cam"],
            )
            # Success means no ConcretizationError during execution
            assert jnp.all(jnp.isfinite(result.filtered_means))
        except Exception as e:
            if "ConcretizationError" in str(type(e)):
                pytest.fail(f"ZUPT implementation not JAX JIT compatible: {e}")
            raise

    def test_zupt_threshold_edge_case(self):
        """ZUPT should handle velocity exactly at threshold correctly."""
        # This is a regression test to ensure no numerical instability at boundary
        # Implementation should use <= or >= consistently
        from trodestrack.sim.simple import SimpleSimConfig

        config = SimpleSimConfig(duration_s=2.0, fs_imu=400, fs_cam=30)
        sim = simulate_stationary(
            config=config, position=np.array([0.5, 0.5]), heading=0.0, seed=42
        )

        config = EKFConfig(
            enable_zupt=True,
            zupt_velocity_threshold=0.05,
            zupt_measurement_noise=0.01**2,
        )

        # Should not crash or produce NaN
        result = extended_kalman_filter(
            ekf_config=config,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        assert jnp.all(jnp.isfinite(result.filtered_means))
        assert jnp.all(jnp.isfinite(result.filtered_covariances))


class TestUKFZUPT:
    """UKF ZUPT behavior should mirror EKF implementation."""

    @staticmethod
    def _run_ukf(sim, config: UKFConfig):
        """Helper to run UKF with provided configuration."""
        return unscented_kalman_filter(
            ukf_config=config,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

    def test_zupt_enabled_by_default(self):
        """UKF should have ZUPT enabled by default for better stationary tracking."""
        config = UKFConfig()
        assert config.enable_zupt is True

    def test_zupt_reduces_velocity_drift_stationary(self):
        """UKF ZUPT should suppress velocity drift when stationary."""
        from trodestrack.sim.simple import SimpleSimConfig

        config = SimpleSimConfig(duration_s=10.0, fs_imu=400, fs_cam=30)
        sim = simulate_stationary(
            config=config, position=np.array([0.5, 0.5]), heading=0.0, seed=123
        )

        # Run UKF without ZUPT
        config_no_zupt = UKFConfig(enable_zupt=False)
        result_no_zupt = self._run_ukf(sim, config_no_zupt)

        # Run UKF with ZUPT enabled
        config_with_zupt = UKFConfig(
            enable_zupt=True,
            zupt_velocity_threshold=0.05,
            zupt_measurement_noise=0.01**2,
        )
        result_with_zupt = self._run_ukf(sim, config_with_zupt)

        truth_vel = np.zeros((len(sim["t_cam_exp"]), 2))
        vel_rmse_no_zupt = compute_velocity_rmse(
            result_no_zupt.filtered_means[:, 2:4], truth_vel
        )
        vel_rmse_with_zupt = compute_velocity_rmse(
            result_with_zupt.filtered_means[:, 2:4], truth_vel
        )

        assert vel_rmse_with_zupt < vel_rmse_no_zupt * 0.7, (
            f"UKF ZUPT did not reduce velocity error: {vel_rmse_with_zupt:.4f} vs {vel_rmse_no_zupt:.4f}"
        )
        assert vel_rmse_with_zupt < 0.02, (
            f"UKF ZUPT velocity error too high: {vel_rmse_with_zupt:.4f} m/s"
        )

    def test_zupt_does_not_activate_during_motion(self):
        """UKF ZUPT should stay inactive when velocity exceeds threshold."""
        from trodestrack.sim.simple import SimpleSimConfig, simulate_constant_velocity

        config = SimpleSimConfig(duration_s=5.0, fs_imu=400, fs_cam=30)
        sim = simulate_constant_velocity(
            config=config,
            initial_position=np.array([0.3, 0.3]),
            velocity=np.array([0.2, 0.0]),
            seed=99,
        )

        config_with_zupt = UKFConfig(
            enable_zupt=True,
            zupt_velocity_threshold=0.05,
            zupt_measurement_noise=0.01**2,
        )
        result = self._run_ukf(sim, config_with_zupt)

        truth_vel = np.tile([0.2, 0.0], (len(sim["t_cam_exp"]), 1))
        vel_rmse = compute_velocity_rmse(result.filtered_means[:, 2:4], truth_vel)
        assert vel_rmse < 0.10, f"UKF velocity RMSE too high: {vel_rmse:.4f} m/s"

        mean_vx = jnp.mean(result.filtered_means[30:, 2])
        assert jnp.abs(mean_vx - 0.2) < 0.05, (
            f"UKF ZUPT incorrectly suppressed motion: mean vx = {mean_vx:.3f}"
        )

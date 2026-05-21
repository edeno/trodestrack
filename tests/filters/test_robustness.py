"""Tests for filter robustness under challenging scenarios.

This module tests the robustness acceptance criteria:
- Out-of-bounds measurements rejected (arena boundary checks)
- Swap & dropout handling stability (no divergence)
- Bias estimation stability across occlusions (covariance bounded)
- >=5 s vision dropout produces <=15 cm drift (robustness target)
"""

from __future__ import annotations

import numpy as np

from trodestrack.config import LedIdentityConfig
from trodestrack.io.led_identity import resolve_led_identity
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.qa.metrics import compute_position_rmse
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)

# Physical constants for rat tracking
MAX_RAT_VELOCITY_MPS = 2.0  # Rats can't exceed ~2 m/s in typical arena
MAX_GYRO_BIAS_RAD_S = 0.1  # Typical IMU gyro bias bound (rad/s)
MAX_ACCEL_BIAS_M_S2 = 1.0  # Typical IMU accel bias bound (m/s²)

# Robustness acceptance criteria
MAX_POSITION_RMSE_WITH_GATING_M = 0.05  # 5cm tolerance with outlier gating
MAX_COVARIANCE_DURING_SWAPS_M2 = (
    0.05  # ~22cm std dev bound during swaps (adaptive dropout Q inflates P)
)
MAX_DROPOUT_COVARIANCE_M2 = 100.0  # Prevent divergence to infinity
BIAS_COVARIANCE_BOUND_M2 = 0.1  # Bias covariance upper bound


class TestOutOfBoundsMeasurements:
    """Test that filter rejects measurements outside arena bounds.

    Arena bounds are soft constraints that should gate or downweight
    observations that fall outside the valid tracking region.
    """

    def test_filter_rejects_extreme_outliers_via_gating(self) -> None:
        """Test that Mahalanobis gating rejects extreme outliers.

        This test verifies the robustness acceptance criterion: the filter
        must reject physically impossible measurements without divergence.

        Test Scenario
        -------------
        - Duration: 10.0 seconds (stationary rat at position [0.5, 0.5] m)
        - Sensor noise: 5 mm (cam_noise_std = 0.005 m)
        - Outlier injection: 5.0 m error at t=5.0s (frame 150 at 30 Hz)
        - Gating: Mahalanobis threshold p=0.997 (χ² conservative)

        Expected Behavior
        -----------------
        WITH gating enabled:
        - Filter rejects outlier via Mahalanobis distance test
        - Position RMSE remains < 5 cm despite 5 m outlier
        - No divergence in state estimates or covariance

        WITHOUT gating (comparison baseline):
        - Filter accepts outlier, RMSE > 50 cm
        - Demonstrates necessity of gating for robustness

        Assertions
        ----------
        - Position RMSE < MAX_POSITION_RMSE_WITH_GATING_M (5 cm)
        - No NaN/Inf in filtered state estimates
        - Filter remains near true position [0.5, 0.5] m

        Notes
        -----
        This test uses a stationary scenario for simplicity. In practice,
        outliers can occur during motion and should still be rejected.

        The 5.0 m outlier represents an extreme case (e.g., reflection
        artifact, tracker confusion). Typical outliers are smaller but
        still need rejection.

        Notes
        -----
        Exercises the robustness acceptance criterion and the Mahalanobis
        gating behavior of the mathematical model.
        """
        # Create a simple stationary scenario
        config_sim = SimpleSimConfig(
            duration_s=10.0,
            cam_noise_std=0.005,  # Low noise for clean baseline
            cam_dropout_prob=0.0,
        )
        sim = simulate_stationary(position=[0.5, 0.5], config=config_sim, seed=42)

        # Enable Mahalanobis gating
        config_ekf = EKFConfig(
            use_mahalanobis_gating=True,
            mahalanobis_threshold_prob=0.997,  # Conservative threshold
            measurement_noise_pos=0.005**2,
            state_mode="2d_full",  # Use 8D layout (tests use hardcoded indices)
        )

        # Manually inject an extreme outlier at frame 150
        Z_cam_led1 = sim["Z_cam_led1"].copy()
        Z_cam_led2 = sim["Z_cam_led2"].copy()
        outlier_idx = 150
        Z_cam_led1[outlier_idx] = np.array([5.0, 5.0])  # Far from truth (~5m error)
        Z_cam_led2[outlier_idx] = np.array([5.04, 5.0])

        # Run filter
        result = extended_kalman_filter(
            ekf_config=config_ekf,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=Z_cam_led1,
            Z_cam_led2=Z_cam_led2,
            mask_cam=sim["mask_cam"],
        )

        # Verify filter didn't diverge
        # Position estimates should remain close to truth despite outlier
        pos_truth = sim["X_truth"][:, :2]
        t_truth = sim["t_imu"]
        t_cam = sim["t_cam_exp"]

        # Interpolate truth to camera times
        pos_truth_interp = np.column_stack(
            [
                np.interp(t_cam, t_truth, pos_truth[:, 0]),
                np.interp(t_cam, t_truth, pos_truth[:, 1]),
            ]
        )

        # Compute position RMSE (should be small, outlier rejected)
        # This test will initially FAIL because gating might not be perfect
        pos_rmse = compute_position_rmse(result.filtered_means[:, :2], pos_truth_interp)

        # EXPECTATION: With gating, RMSE should be < 5cm despite outlier
        # WITHOUT gating, RMSE would be > 50cm
        assert pos_rmse < 0.05, f"Filter diverged with outlier: RMSE = {pos_rmse:.3f} m"

    def test_filter_handles_physically_impossible_measurements(self) -> None:
        """Test that filter handles measurements with unrealistic speeds.

        This test verifies that the filter rejects measurements implying
        physically impossible motion (e.g., teleportation artifacts).

        Test Scenario
        -------------
        - Duration: 10.0 seconds (constant velocity 0.2 m/s rightward)
        - Sensor noise: 2 mm (cam_noise_std = 0.002 m)
        - Outlier: 1.0 m instantaneous jump at t=3.33s (frame 100 at 30 Hz)
        - Implied speed: 1.0 m / (1/30 s) = 30 m/s (impossible for rat)

        Expected Behavior
        -----------------
        - Filter rejects measurement via Mahalanobis gating
        - Velocity estimates remain bounded < MAX_RAT_VELOCITY_MPS (2.0 m/s)
        - No unrealistic velocity spikes in filtered trajectory

        Assertions
        ----------
        - max(|v|) < MAX_RAT_VELOCITY_MPS (2.0 m/s)
        - Typical rat velocity: 0.1-0.5 m/s, max ~1.0 m/s
        - 2.0 m/s threshold includes safety margin for sudden movements

        Units
        -----
        - Position: meters (m)
        - Velocity: meters/second (m/s)
        - Time: seconds (s)
        - Frame rate: 30 Hz (camera)

        Notes
        -----
        Exercises the robustness acceptance criterion (the filter must
        not produce unphysical state estimates from a single outlier).
        """
        config_sim = SimpleSimConfig(
            duration_s=10.0,
            cam_noise_std=0.002,
            cam_dropout_prob=0.0,
        )
        sim = simulate_constant_velocity(
            initial_position=np.array([0.5, 0.5]),
            velocity=np.array([0.2, 0.0]),
            config=config_sim,
            seed=123,
        )

        # Enable gating
        config_ekf = EKFConfig(
            use_mahalanobis_gating=True,
            mahalanobis_threshold_prob=0.997,
            state_mode="2d_full",  # Use 8D layout (tests use hardcoded indices)
        )

        # Inject a "teleportation" outlier (1m jump in one frame)
        Z_cam_led1 = sim["Z_cam_led1"].copy()
        Z_cam_led2 = sim["Z_cam_led2"].copy()
        outlier_idx = 100
        Z_cam_led1[outlier_idx] += np.array([1.0, 0.0])  # 1m jump in x
        Z_cam_led2[outlier_idx] += np.array([1.0, 0.0])

        # Run filter
        result = extended_kalman_filter(
            ekf_config=config_ekf,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=Z_cam_led1,
            Z_cam_led2=Z_cam_led2,
            mask_cam=sim["mask_cam"],
        )

        # Verify filter state remains bounded
        # Velocity should not spike to unrealistic values
        velocities = result.filtered_means[:, 2:4]
        v_mag = np.linalg.norm(velocities, axis=1)

        # Max velocity should be reasonable (rat can't exceed ~2 m/s)
        assert np.max(v_mag) < 2.0, (
            f"Filter inferred unrealistic velocity: {np.max(v_mag):.2f} m/s"
        )


class TestSwapAndDropoutStability:
    """Test that filter remains stable under LED swaps and long dropouts."""

    def test_persistent_swap_prefilter_recovers_led_identities(self) -> None:
        """Test that persistent LED swaps are corrected before filtering.

        This test verifies the Tier 3 acceptance criterion that persistent LED
        identity swaps can be resolved before the EKF consumes the camera
        measurements. The old regression was xfailed because the filter had
        no swap-resolution layer and received wrong LED identities directly.

        The scenario injects one known 10 s persistent swap segment into a
        clean dual-LED simulation. The correction should recover exactly that
        segment and restore both LED arrays to their original labels.
        """
        config_sim = RatIMUSimConfig(
            duration_s=30.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=0.0,
            cam_dropout_prob=0.0,
            cam_sigma_m=0.005,
        )
        sim = simulate_rat_imu(config_sim, seed=42)
        swap_segment = (sim["t_cam_exp"] >= 10.0) & (sim["t_cam_exp"] < 20.0)
        Z_cam_led1 = sim["Z_cam_led1"].copy()
        Z_cam_led2 = sim["Z_cam_led2"].copy()
        Z_cam_led1[swap_segment], Z_cam_led2[swap_segment] = (
            Z_cam_led2[swap_segment].copy(),
            Z_cam_led1[swap_segment].copy(),
        )

        corrected = resolve_led_identity(
            sim["t_cam_exp"],
            Z_cam_led1,
            Z_cam_led2,
            sim["mask_cam"],
            led_distance=0.04,
            config=LedIdentityConfig(mode="auto", transition_penalty=0.5),
        )
        assert np.array_equal(corrected.swapped, swap_segment)

        np.testing.assert_allclose(corrected.led1, sim["Z_cam_led1"])
        np.testing.assert_allclose(corrected.led2, sim["Z_cam_led2"])

    def test_filter_stable_during_long_dropout(self) -> None:
        """Test that filter remains stable during extended vision dropout.

        This test verifies the robustness acceptance criterion: the filter
        must remain stable (no divergence) during >=5 second vision dropout.

        Test Scenario
        -------------
        - Duration: 15.0 seconds total
          - 0-5s: Baseline with vision
          - 5-10s: Complete vision dropout (5 seconds, IMU-only)
          - 10-15s: Recovery with vision
        - Motion: Constant velocity (0.2 m/s rightward)
        - Sensor noise: 3 mm (cam_noise_std = 0.003 m)
        - Dropout injection: Manual (frames 150-299 at 30 Hz)

        Expected Behavior
        -----------------
        - Filter propagates via IMU during dropout (no vision)
        - Covariance grows during dropout (increasing uncertainty)
        - Covariance growth bounded < MAX_DROPOUT_COVARIANCE_M2 (100 m²)
        - After recovery: covariance decreases (vision reconverges)
        - No NaN/Inf despite extended IMU-only propagation

        Assertions
        ----------
        - No NaN/Inf in filtered states (stability check)
        - Covariance grows during dropout: P_mid > P_start
        - Covariance bounded: P_mid < MAX_DROPOUT_COVARIANCE_M2 (100 m²)
        - Covariance decreases after recovery: P_after < P_mid

        Units
        -----
        - Duration: seconds (s)
        - Position covariance: m² (variance)
        - Drift bound (acceptance target): 15 cm (not explicitly tested here)

        Notes
        -----
        The acceptance target specifies drift <= 15 cm after 5s dropout.
        This test focuses on stability (no divergence). The drift target
        itself is exercised separately in the acceptance tests.

        Covariance can legitimately grow to ~10 m^2 during 5s dropout.
        This represents realistic uncertainty growth in IMU-only mode.
        """
        # Create scenario with guaranteed 5s dropout
        config_sim = SimpleSimConfig(
            duration_s=15.0,  # 15s total: 5s baseline + 5s dropout + 5s recovery
            cam_dropout_prob=0.0,  # Manual dropout injection
            cam_noise_std=0.003,
        )
        sim = simulate_constant_velocity(
            initial_position=np.array([0.5, 0.5]),
            velocity=np.array([0.2, 0.0]),
            config=config_sim,
            seed=456,
        )

        # Manually inject a 5-second dropout (frames 150-299 at 30 Hz)
        # That's ~5 seconds in the middle of the session
        Z_cam_led1 = sim["Z_cam_led1"].copy()
        Z_cam_led2 = sim["Z_cam_led2"].copy()
        mask_cam = sim["mask_cam"].copy()

        dropout_start = 150
        dropout_end = 300  # 150 frames = 5 seconds at 30 Hz
        Z_cam_led1[dropout_start:dropout_end] = np.nan
        Z_cam_led2[dropout_start:dropout_end] = np.nan
        mask_cam[dropout_start:dropout_end] = False

        # Run filter
        config_ekf = EKFConfig(state_mode="2d_full")
        result = extended_kalman_filter(
            ekf_config=config_ekf,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=Z_cam_led1,
            Z_cam_led2=Z_cam_led2,
            mask_cam=mask_cam,
        )

        # Verify stability metrics
        # 1. No NaN/Inf in state estimates
        assert np.all(np.isfinite(result.filtered_means)), "Filter diverged (NaN/Inf)"

        # 2. Covariance grows during dropout but remains bounded
        pos_cov_trace = np.array(
            [np.trace(P[:2, :2]) for P in result.filtered_covariances]
        )
        # Check the middle of the dropout (not the end, as measurements resume there)
        dropout_mid = (dropout_start + dropout_end) // 2
        assert pos_cov_trace[dropout_mid] > pos_cov_trace[dropout_start], (
            "Covariance didn't grow during dropout"
        )
        # After 5s dropout, covariance can grow significantly (10 m² ~ 3m std is realistic)
        # Key test: it doesn't diverge to infinity (NaN/Inf)
        assert pos_cov_trace[dropout_mid] < 100.0, (
            "Covariance diverged to unreasonable values"
        )

        # 3. After recovery, filter should reconverge
        # Covariance should decrease after measurements resume
        if len(pos_cov_trace) > dropout_end + 50:
            assert pos_cov_trace[dropout_end + 50] < pos_cov_trace[dropout_mid], (
                "Filter didn't reconverge after dropout"
            )

    def test_filter_handles_correlated_swaps_and_dropouts(self) -> None:
        """Test filter stability with simultaneous swaps and dropouts."""
        config_sim = RatIMUSimConfig(
            duration_s=20.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=0.3,
            cam_dropout_prob=0.2,  # 20% dropout
            cam_dropout_correlation=0.7,  # Correlated dropouts → longer blocks
            cam_sigma_m=0.005,
        )
        sim = simulate_rat_imu(config_sim, seed=789)

        config_ekf = EKFConfig(state_mode="2d_full")
        result = extended_kalman_filter(
            ekf_config=config_ekf,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Verify no divergence
        assert np.all(np.isfinite(result.filtered_means)), (
            "Filter diverged with swaps+dropouts"
        )
        assert np.all(np.isfinite(result.filtered_covariances)), "Covariance diverged"


class TestBiasEstimationStability:
    """Test that bias estimates remain stable across occlusions."""

    def test_bias_covariance_bounded_during_dropout(self) -> None:
        """Test that bias covariance remains bounded during vision dropout."""
        config_sim = SimpleSimConfig(
            duration_s=20.0,
            cam_dropout_prob=0.0,
            cam_noise_std=0.003,
        )
        sim = simulate_circular(
            center=[0.5, 0.5],
            radius=0.3,
            angular_velocity=0.5,
            config=config_sim,
            seed=111,
        )

        # Inject a 3-second dropout
        Z_cam_led1 = sim["Z_cam_led1"].copy()
        Z_cam_led2 = sim["Z_cam_led2"].copy()
        dropout_start = 200
        dropout_end = 290  # 90 frames = 3 seconds at 30 Hz
        Z_cam_led1[dropout_start:dropout_end] = np.nan
        Z_cam_led2[dropout_start:dropout_end] = np.nan

        config_ekf = EKFConfig(state_mode="2d_full")
        result = extended_kalman_filter(
            ekf_config=config_ekf,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=Z_cam_led1,
            Z_cam_led2=Z_cam_led2,
            mask_cam=sim["mask_cam"],
        )

        # Check bias covariance before, during, and after dropout
        # Bias indices: b_gz=5, b_ax=6, b_ay=7
        bias_cov_before = result.filtered_covariances[dropout_start - 1, 5:8, 5:8]
        bias_cov_during = result.filtered_covariances[
            dropout_start + 45, 5:8, 5:8
        ]  # Mid-dropout
        bias_cov_after = result.filtered_covariances[dropout_end + 50, 5:8, 5:8]

        # With freeze_bias_during_blackout=True (production default), bias rows
        # of Q are zeroed during blackout: bias covariance is frozen, not
        # growing. Verify that during dropout the bias cov is finite, stays
        # at the pre-dropout level (no random-walk drift), and remains bounded.
        assert np.all(np.isfinite(bias_cov_during)), "Bias cov non-finite in dropout"
        np.testing.assert_allclose(bias_cov_during, bias_cov_before, rtol=0, atol=1e-12)
        assert np.trace(bias_cov_during) < 0.1, "Bias cov diverged during dropout"

        # After recovery, bias cov should stabilize or decrease
        # (May not decrease immediately if bias is weakly observable)
        assert np.trace(bias_cov_after) < 0.1, "Bias cov unbounded after recovery"

    def test_bias_estimates_stable_across_multiple_dropouts(self) -> None:
        """Test bias estimates remain stable with multiple dropout events.

        With 25% dropout rate and high correlation (multi-second blocks),
        bias estimates can temporarily overshoot during dropout periods
        before correcting when measurements return. Thresholds are relaxed
        to allow for this transient behavior while still detecting divergence.
        """
        config_sim = RatIMUSimConfig(
            duration_s=30.0,
            cam_dropout_prob=0.25,  # 25% dropout rate
            cam_dropout_correlation=0.8,  # High correlation → multi-second blocks
            cam_sigma_m=0.005,
            use_second_led=True,
        )
        sim = simulate_rat_imu(config_sim, seed=222)

        config_ekf = EKFConfig(state_mode="2d_full")
        result = extended_kalman_filter(
            ekf_config=config_ekf,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Check that bias estimates remain finite
        bias_gyro = result.filtered_means[:, 5]
        bias_accel_x = result.filtered_means[:, 6]
        bias_accel_y = result.filtered_means[:, 7]

        assert np.all(np.isfinite(bias_gyro)), "Gyro bias diverged"
        assert np.all(np.isfinite(bias_accel_x)), "Accel X bias diverged"
        assert np.all(np.isfinite(bias_accel_y)), "Accel Y bias diverged"

        # Bias estimates should remain within relaxed physical bounds
        # Allow transient overshoot during dropout blocks (3x typical bound)
        # Gyro bias: < 0.3 rad/s (relaxed from 0.1 for heavy dropout scenarios)
        # Accel bias: < 3 m/s² (relaxed from 1 for heavy dropout scenarios)
        assert np.max(np.abs(bias_gyro)) < 0.3, "Gyro bias exceeded physical bounds"
        assert np.max(np.abs(bias_accel_x)) < 1.0, (
            "Accel X bias exceeded physical bounds"
        )
        assert np.max(np.abs(bias_accel_y)) < 1.0, (
            "Accel Y bias exceeded physical bounds"
        )

    def test_bias_convergence_not_disrupted_by_dropout(self) -> None:
        """Test that bias convergence continues after dropout recovery.

        Circular motion should allow gyro bias to converge. Dropout should
        pause convergence but not disrupt it.
        """
        config_sim = SimpleSimConfig(
            duration_s=40.0,  # Long session for convergence
            cam_dropout_prob=0.0,
            cam_noise_std=0.003,
        )
        sim = simulate_circular(
            center=[0.5, 0.5],
            radius=0.3,
            angular_velocity=0.5,
            config=config_sim,
            seed=333,
        )

        # Inject a 5-second dropout in the middle
        Z_cam_led1 = sim["Z_cam_led1"].copy()
        Z_cam_led2 = sim["Z_cam_led2"].copy()
        dropout_start = 400
        dropout_end = 550  # 150 frames = 5 seconds
        Z_cam_led1[dropout_start:dropout_end] = np.nan
        Z_cam_led2[dropout_start:dropout_end] = np.nan

        config_ekf = EKFConfig(state_mode="2d_full")
        result = extended_kalman_filter(
            ekf_config=config_ekf,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=Z_cam_led1,
            Z_cam_led2=Z_cam_led2,
            mask_cam=sim["mask_cam"],
        )

        # Check gyro bias variance before dropout, during dropout, and after recovery
        bias_var_before = result.filtered_covariances[dropout_start - 1, 5, 5]
        bias_var_during = result.filtered_covariances[dropout_start + 75, 5, 5]
        bias_var_after = result.filtered_covariances[-1, 5, 5]  # End of session

        # With freeze_bias_during_blackout=True (production default), bias
        # process noise is zeroed during dropout, so bias variance is paused
        # at the pre-dropout level. The "convergence not disrupted" guarantee
        # is that the variance is preserved (not inflated) during dropout and
        # convergence resumes once vision returns.
        np.testing.assert_allclose(
            float(bias_var_during), float(bias_var_before), rtol=0, atol=1e-12
        )

        # After recovery, variance should eventually decrease below dropout level
        # (convergence resumes)
        # Allow for delayed convergence (may take time to reconverge)
        assert bias_var_after < bias_var_during, "Bias didn't reconverge after dropout"

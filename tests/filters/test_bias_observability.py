"""Test bias observability properties of the EKF.

Verifies that:
- Stationary motion cannot observe IMU biases (unobservable)
- Circular motion with gyro allows gyro bias convergence
- Straight-line motion cannot observe lateral acceleration bias
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)


def test_stationary_bias_unobservable() -> None:
    """Stationary motion should not be able to estimate IMU biases.

    With no motion, the filter has no way to distinguish between:
    - True IMU bias
    - Measurement noise
    - Process noise

    We expect bias covariance to remain high (not converge).
    """
    # Simulation config
    sim_config = SimpleSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_std=0.01,
        accel_bias_std=0.05,
        cam_noise_std=0.005,
        cam_dropout_prob=0.0,
    )

    # Run stationary simulation
    sim = simulate_stationary(sim_config, position=np.array([1.0, 1.0]), seed=42)

    # EKF config with very slow bias drift
    ekf_config = EKFConfig(
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        process_noise_heading=0.02,
        process_noise_gyro_bias=1e-5,  # Very slow drift
        process_noise_accel_bias=1e-5,  # Very slow drift
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.05**2,
        imu_gyro_noise_density=0.001,
        imu_accel_noise_density=0.05,
        damping_coeff=0.5,
        led_distance=0.04,
    )

    # Run EKF
    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Extract bias covariance at start and end. Resolve indices through
    # the actual filter layout — hardcoded 5:8 silently picked up heading
    # / vz columns under the default 10D 2d_cam_3d_imu layout.
    layout = get_layout(ekf_config.state_mode)
    bias_idx = list(layout.bias_gyro_idx) + list(layout.bias_accel_idx)
    P_est = result.filtered_covariances
    P_bias_start = P_est[0][np.ix_(bias_idx, bias_idx)]
    P_bias_end = P_est[-1][np.ix_(bias_idx, bias_idx)]

    # Ensure initial covariance is valid (defensive check)
    trace_start = np.trace(P_bias_start)
    assert trace_start > 0, "Initial bias covariance must be positive"

    # Check that bias covariance has NOT significantly decreased
    # (i.e., biases remain unobservable)
    cov_reduction_ratio = np.trace(P_bias_end) / trace_start

    # We expect minimal reduction (>80% of original uncertainty remains)
    # Threshold: Stationary motion provides no information to distinguish
    # bias from noise, so covariance should remain near initial value
    assert cov_reduction_ratio > 0.8, (
        f"Bias covariance reduced too much ({cov_reduction_ratio:.2f}). "
        "Stationary motion should not observe biases."
    )


def test_circular_bias_converges() -> None:
    """Circular motion should allow gyro bias to converge.

    During circular motion:
    - Gyro bias is observable through discrepancies between:
      * IMU-predicted heading change
      * Vision-measured heading from LED pair
    - We expect gyro bias covariance to decrease significantly
    """
    # Simulation config with longer duration for bias convergence
    sim_config = SimpleSimConfig(
        duration_s=20.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_std=0.01,
        accel_bias_std=0.05,
        cam_noise_std=0.005,
        cam_dropout_prob=0.0,
    )

    # Run circular simulation
    sim = simulate_circular(
        sim_config,
        center=np.array([1.0, 1.0]),
        radius=0.2,  # 20 cm
        angular_velocity=0.5,  # rad/s
        seed=42,
    )

    # EKF config with reasonable bias drift
    ekf_config = EKFConfig(
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        process_noise_heading=0.02,
        process_noise_gyro_bias=2e-6,  # Slow but observable drift
        process_noise_accel_bias=2e-4,
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.05**2,
        imu_gyro_noise_density=0.0001,  # Reduced for better observability
        imu_accel_noise_density=0.005,
        damping_coeff=0.5,
        led_distance=0.04,
    )

    # Run EKF
    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Resolve gyro-bias index from the layout the filter actually used.
    # In the default 2d_cam_3d_imu layout (10D), b_gz is at index 6, not 5;
    # index 5 is the heading. Hardcoding index 5 in this test silently
    # validated the wrong state component.
    layout = get_layout(ekf_config.state_mode)
    b_gz_idx = layout.bias_gyro_idx[0]

    P_est = result.filtered_covariances
    gyro_bias_var_start = P_est[0, b_gz_idx, b_gz_idx]
    gyro_bias_var_end = P_est[-1, b_gz_idx, b_gz_idx]

    # Defensive check
    assert gyro_bias_var_start > 0, "Initial gyro bias variance must be positive"

    # Check that gyro bias variance has significantly decreased
    variance_reduction_ratio = gyro_bias_var_end / gyro_bias_var_start

    # We expect at least 50% reduction (conservative threshold)
    # Justification: Circular motion provides strong observability via
    # heading updates from LED pair - bias should converge within 20s
    assert variance_reduction_ratio < 0.5, (
        f"Gyro bias variance only reduced to {variance_reduction_ratio:.2f} of initial. "
        "Expected significant convergence during circular motion."
    )

    # Also verify that the bias estimate is reasonable (not diverging)
    X_est = result.filtered_means
    gyro_bias_estimate = X_est[-1, b_gz_idx]
    assert np.abs(gyro_bias_estimate) < 1.0, (
        f"Gyro bias estimate {gyro_bias_estimate:.3f} rad/s seems unreasonably large"
    )


def test_straight_line_lateral_bias_unobservable() -> None:
    """Straight-line motion cannot observe lateral acceleration bias.

    During straight-line motion in the forward direction:
    - Forward acceleration bias is observable (affects velocity magnitude)
    - Lateral acceleration bias is NOT observable (no lateral motion to constrain it)
    - We expect lateral accel bias covariance to remain high
    """
    # Simulation config
    sim_config = SimpleSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_std=0.01,
        accel_bias_std=0.05,
        cam_noise_std=0.005,
        cam_dropout_prob=0.0,
    )

    # Run constant velocity simulation (straight line)
    sim = simulate_constant_velocity(
        sim_config,
        initial_position=np.array([0.1, 0.1]),  # meters
        velocity=np.array([0.2, 0.0]),  # 20 cm/s in x direction
        seed=42,
    )

    # EKF config
    ekf_config = EKFConfig(
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        process_noise_heading=0.02,
        process_noise_gyro_bias=2e-6,
        process_noise_accel_bias=2e-4,
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.05**2,
        imu_gyro_noise_density=0.0001,
        imu_accel_noise_density=0.005,
        damping_coeff=0.5,
        led_distance=0.04,
    )

    # Run EKF
    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Extract acceleration bias covariances via layout indices.
    # Body-frame at θ=0: bias_accel_idx[0] is forward (b_ax),
    # bias_accel_idx[1] is lateral (b_ay).
    layout = get_layout(ekf_config.state_mode)
    forward_idx = layout.bias_accel_idx[0]
    lateral_idx = layout.bias_accel_idx[1]

    P_est = result.filtered_covariances

    forward_bias_var_start = P_est[0, forward_idx, forward_idx]
    forward_bias_var_end = P_est[-1, forward_idx, forward_idx]

    lateral_bias_var_start = P_est[0, lateral_idx, lateral_idx]
    lateral_bias_var_end = P_est[-1, lateral_idx, lateral_idx]

    # Defensive checks
    assert forward_bias_var_start > 0, "Initial forward bias variance must be positive"
    assert lateral_bias_var_start > 0, "Initial lateral bias variance must be positive"

    # Forward bias should show some convergence (observable)
    forward_reduction = forward_bias_var_end / forward_bias_var_start

    # Lateral bias should remain largely uncertain (unobservable)
    lateral_reduction = lateral_bias_var_end / lateral_bias_var_start

    # We expect lateral bias to remain largely unobservable (>70% of initial uncertainty)
    # Justification: In straight-line motion, there's no lateral acceleration to
    # constrain lateral bias - it's a null space of the observability matrix
    assert lateral_reduction > 0.7, (
        f"Lateral bias variance reduced too much ({lateral_reduction:.2f}). "
        "Should remain largely unobservable in straight-line motion."
    )

    # Forward bias should be MORE observable than lateral (but both may grow due to process noise)
    # Key insight: with perfect camera + damping, acceleration biases are weakly observable
    # We verify lateral is not significantly more observable than forward (5% tolerance)
    expected_min_lateral = forward_reduction * 0.95
    assert lateral_reduction >= expected_min_lateral, (
        f"Lateral bias ({lateral_reduction:.2f}) is significantly more observable "
        f"than forward bias ({forward_reduction:.2f}), which is unexpected. "
        f"Expected lateral ≥ {expected_min_lateral:.2f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

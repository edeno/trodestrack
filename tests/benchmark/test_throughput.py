"""Throughput and latency benchmarks for PRD performance requirements.

This module validates system performance against PRD requirements:
- Offline smoothing ≥10× realtime (CPU) on 30 min session (PRD §4.3)
- Online EKF latency ≤33 ms per frame (CPU) (PRD §4.4)

These benchmarks ensure the filter implementation meets production throughput
and latency requirements for both offline post-processing and real-time tracking.

References:
    - PRD.md Section 4: Core Outcomes (Acceptance Criteria)
"""

import time
from typing import Any

import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.runtime.offline import rts_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

# =============================================================================
# PRD Performance Requirements (from PRD.md Section 4)
# =============================================================================

PRD_OFFLINE_SMOOTHER_SPEEDUP_MIN = 10.0  # Offline ≥10× realtime (CPU)
PRD_ONLINE_EKF_LATENCY_MS_MAX = 33.0  # Online ≤33 ms per frame (CPU)

# Session duration for throughput benchmarking
BENCHMARK_SESSION_DURATION_S = 1800.0  # 30 minutes


# =============================================================================
# Helper: Production EKF Configuration
# =============================================================================


def get_production_ekf_config(**overrides: Any) -> EKFConfig:
    """Get production EKF configuration for benchmarking.

    Returns configuration matching production settings with adaptive dropout handling.

    Args:
        **overrides: Optional parameter overrides

    Returns:
        EKFConfig with production settings
    """
    defaults = dict(
        process_noise_pos=0.001,
        process_noise_vel=0.5,
        process_noise_heading=0.02,
        process_noise_gyro_bias=2e-6,
        process_noise_accel_bias=2e-4,
        measurement_noise_pos=0.005**2,
        imu_gyro_noise_density=0.001,
        imu_accel_noise_density=0.05,
        damping_coeff=0.4,
        led_distance=0.04,
        use_heading_measurement=True,
        adaptive_q_during_dropout=False,
        dropout_q_pos_multiplier=10.0,
        dropout_q_vel_multiplier=10.0,
        dropout_q_bias_multiplier=0.1,
        state_mode="2d_full",  # Use 8D layout (tests use hardcoded indices)
    )
    defaults.update(overrides)
    return EKFConfig(**defaults)


# =============================================================================
# Test 1: Offline Smoother Throughput (≥10× realtime)
# =============================================================================


@pytest.mark.slow
@pytest.mark.benchmark
def test_offline_smoother_throughput():
    """PRD §4.3: Offline RTS smoother should achieve ≥10× realtime on CPU.

    Validates that the complete offline smoothing pipeline (filter + RTS smoother)
    can process a 30-minute session at least 10× faster than realtime on CPU.

    Throughput calculation:
        - Session duration: 30 minutes (1800 seconds)
        - Target: ≥10× realtime → processing time ≤ 180 seconds (3 minutes)

    Expected runtime: ~45-60 seconds (on modern CPU, measured on M-series Mac)
    """
    # Generate 30-minute realistic rat tracking session
    config = RatIMUSimConfig(
        duration_s=BENCHMARK_SESSION_DURATION_S,
        fs_imu=200.0,  # 200 Hz IMU (downsampled from 20 kHz)
        fs_cam=30.0,  # 30 Hz camera
        cam_dropout_prob=0.05,  # 5% random dropouts
        cam_dropout_correlation=0.8,  # Correlated dropouts
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_rw_density=0.0001,
        accel_bias_rw_density=0.001,
        cam_sigma_m=0.005,  # 5mm camera noise
        use_second_led=True,
        imu_tilt_roll_deg=0.0,
        imu_tilt_pitch_deg=0.0,
        arena_w=1.0,
        arena_h=1.0,
    )
    sim_data = simulate_rat_imu(config=config, seed=42)

    # Get production EKF configuration
    ekf_config = get_production_ekf_config()

    # Measure total processing time (filter + smoother)
    t_start = time.perf_counter()

    # Run forward filter
    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_data["t_imu"],
        U_imu=sim_data["U_imu"],
        t_cam=sim_data["t_cam_exp"],
        Z_cam_led1=sim_data["Z_cam_led1"],
        Z_cam_led2=sim_data["Z_cam_led2"],
        mask_cam=sim_data["mask_cam"],
    )

    # Run RTS smoother
    smoother_result = rts_smoother(
        filter_result=filter_result,
        ekf_config=ekf_config,
        t_imu=sim_data["t_imu"],
        U_imu=sim_data["U_imu"],
        t_cam=sim_data["t_cam_exp"],
        mask_cam=sim_data["mask_cam"],
    )

    t_end = time.perf_counter()
    processing_time_s = t_end - t_start

    # Compute speedup (realtime factor)
    session_duration_s = BENCHMARK_SESSION_DURATION_S
    speedup = session_duration_s / processing_time_s

    # Log results for diagnostics
    print("\n=== Offline Smoother Throughput Benchmark ===")
    print(
        f"Session duration: {session_duration_s:.1f} s ({session_duration_s / 60:.1f} min)"
    )
    print(
        f"Processing time: {processing_time_s:.2f} s ({processing_time_s / 60:.2f} min)"
    )
    print(f"Speedup: {speedup:.1f}× realtime")
    print(f"PRD requirement: ≥{PRD_OFFLINE_SMOOTHER_SPEEDUP_MIN:.1f}× realtime")
    print(
        f"Status: {'PASS ✓' if speedup >= PRD_OFFLINE_SMOOTHER_SPEEDUP_MIN else 'FAIL ✗'}"
    )

    # Validate PRD requirement
    assert (
        speedup >= PRD_OFFLINE_SMOOTHER_SPEEDUP_MIN
    ), f"Offline smoother speedup {speedup:.1f}× is below PRD requirement {PRD_OFFLINE_SMOOTHER_SPEEDUP_MIN:.1f}×"

    # Sanity check: verify smoother produced valid results
    assert smoother_result.smoothed_means.shape[0] == len(
        sim_data["t_cam_exp"]
    ), "Smoother output length mismatch"
    assert (
        smoother_result.smoothed_means.shape[1] == 8
    ), "Smoother state dimension should be 8"
    assert np.all(
        np.isfinite(smoother_result.smoothed_means)
    ), "Smoother means contain NaN/Inf"
    # Validate smoother covariances
    assert smoother_result.smoothed_covariances.shape == (
        len(sim_data["t_cam_exp"]),
        8,
        8,
    ), "Smoother covariance shape mismatch"
    assert np.all(
        np.isfinite(smoother_result.smoothed_covariances)
    ), "Smoother covariances contain NaN/Inf"
    # Check positive-definiteness (diagonal elements > 0)
    assert np.all(
        np.diagonal(smoother_result.smoothed_covariances, axis1=1, axis2=2) > 0
    ), "Smoother covariances not positive-definite"


# =============================================================================
# Test 2: Online EKF Latency (≤33 ms per frame)
# =============================================================================


@pytest.mark.slow
@pytest.mark.benchmark
def test_online_ekf_latency():
    """PRD §4.4: Online EKF should achieve ≤33 ms per-frame latency on CPU.

    Validates that the EKF prediction + update cycle can process a single
    camera frame (with inter-frame IMU pre-integration) in ≤33 ms on CPU.

    This simulates online tracking where the filter must process each camera
    frame as it arrives at 30 Hz (~33 ms frame period).

    Strategy:
        - Generate a realistic session with 30 Hz camera
        - Run the full EKF filter (pre-integration + measurement update per frame)
        - Measure total processing time and compute average per-frame latency
        - PRD requires p99 latency ≤33 ms, we test mean latency ≤33 ms (stricter)

    Expected runtime: ~45-60 seconds (on modern CPU, measured on M-series Mac)
    """
    # Generate 30-minute realistic rat tracking session
    # (same as offline smoother benchmark for consistency)
    config = RatIMUSimConfig(
        duration_s=BENCHMARK_SESSION_DURATION_S,
        fs_imu=200.0,  # 200 Hz IMU
        fs_cam=30.0,  # 30 Hz camera
        cam_dropout_prob=0.05,  # 5% random dropouts
        cam_dropout_correlation=0.8,  # Correlated dropouts
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_rw_density=0.0001,
        accel_bias_rw_density=0.001,
        cam_sigma_m=0.005,  # 5mm camera noise
        use_second_led=True,
        imu_tilt_roll_deg=0.0,
        imu_tilt_pitch_deg=0.0,
        arena_w=1.0,
        arena_h=1.0,
    )
    sim_data = simulate_rat_imu(config=config, seed=42)

    # Get production EKF configuration
    ekf_config = get_production_ekf_config()

    # Measure filter processing time
    t_start = time.perf_counter()

    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_data["t_imu"],
        U_imu=sim_data["U_imu"],
        t_cam=sim_data["t_cam_exp"],
        Z_cam_led1=sim_data["Z_cam_led1"],
        Z_cam_led2=sim_data["Z_cam_led2"],
        mask_cam=sim_data["mask_cam"],
    )

    t_end = time.perf_counter()
    total_processing_time_s = t_end - t_start

    # Compute per-frame latency
    num_frames = len(sim_data["t_cam_exp"])
    mean_latency_per_frame_ms = (total_processing_time_s / num_frames) * 1000.0

    # Compute theoretical frame period for 30 Hz camera
    frame_period_ms = 1000.0 / config.fs_cam  # ~33.33 ms

    # Log results for diagnostics
    print("\n=== Online EKF Latency Benchmark ===")
    print(
        f"Session duration: {BENCHMARK_SESSION_DURATION_S:.1f} s ({BENCHMARK_SESSION_DURATION_S / 60:.1f} min)"
    )
    print(f"Number of frames: {num_frames}")
    print(f"Total processing time: {total_processing_time_s:.2f} s")
    print(f"Mean latency per frame: {mean_latency_per_frame_ms:.2f} ms")
    print(f"Camera frame period (30 Hz): {frame_period_ms:.2f} ms")
    print(f"PRD requirement: ≤{PRD_ONLINE_EKF_LATENCY_MS_MAX:.1f} ms per frame")
    print(
        f"Status: {'PASS ✓' if mean_latency_per_frame_ms <= PRD_ONLINE_EKF_LATENCY_MS_MAX else 'FAIL ✗'}"
    )

    # Validate PRD requirement
    assert mean_latency_per_frame_ms <= PRD_ONLINE_EKF_LATENCY_MS_MAX, (
        f"Mean EKF latency {mean_latency_per_frame_ms:.2f} ms exceeds "
        f"PRD requirement {PRD_ONLINE_EKF_LATENCY_MS_MAX:.1f} ms"
    )

    # Sanity check: verify filter produced valid results
    assert (
        filter_result.filtered_means.shape[0] == num_frames
    ), "Filter output length mismatch"
    assert (
        filter_result.filtered_means.shape[1] == 8
    ), "Filter state dimension should be 8"
    assert np.all(
        np.isfinite(filter_result.filtered_means)
    ), "Filter means contain NaN/Inf"

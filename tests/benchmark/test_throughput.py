"""Throughput and latency benchmarks for PRD performance requirements.

This module validates system performance against PRD requirements:
- Offline smoothing ≥10× realtime (CPU) on 30 min session (PRD §4.3)
- Online EKF latency ≤33 ms per frame (CPU) (PRD §4.4) — measured here
  as amortized mean per-frame time (total / num_frames) over a single
  JIT'd ``lax.scan`` batch, which is a *necessary* but not sufficient
  condition for the per-frame requirement. Per-frame tail / p99 latency
  is not measured by this suite (the filter is not driven from a
  streaming ingest loop), so this is a throughput-style proxy for the
  forward-only "online" CLI rather than a streaming / real-time
  guarantee.

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
    assert speedup >= PRD_OFFLINE_SMOOTHER_SPEEDUP_MIN, (
        f"Offline smoother speedup {speedup:.1f}× is below PRD requirement {PRD_OFFLINE_SMOOTHER_SPEEDUP_MIN:.1f}×"
    )

    # Sanity check: verify smoother produced valid results
    assert smoother_result.smoothed_means.shape[0] == len(sim_data["t_cam_exp"]), (
        "Smoother output length mismatch"
    )
    assert smoother_result.smoothed_means.shape[1] == 8, (
        "Smoother state dimension should be 8"
    )
    assert np.all(np.isfinite(smoother_result.smoothed_means)), (
        "Smoother means contain NaN/Inf"
    )
    # Validate smoother covariances
    assert smoother_result.smoothed_covariances.shape == (
        len(sim_data["t_cam_exp"]),
        8,
        8,
    ), "Smoother covariance shape mismatch"
    assert np.all(np.isfinite(smoother_result.smoothed_covariances)), (
        "Smoother covariances contain NaN/Inf"
    )
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
    """PRD §4 Online: end-to-end latency ≤33 ms per frame (EKF on CPU).

    Validates that the EKF can keep up with a 30 Hz camera over a long
    session: total wall-clock processing time divided by frame count
    must stay below the 33 ms frame period.

    Caveat: this is an amortized / mean per-frame check, not a per-frame
    tail-latency check. The filter runs as a single JIT-compiled
    ``lax.scan`` over the full session, so individual scan steps are not
    timed and slow tail frames cannot be detected here. Per-frame
    distribution and p99 measurement require an unrolled / online-loop
    harness; this test verifies only that the average frame budget is
    met. For the cited PRD ≤33 ms requirement, the mean is a
    *necessary*, not sufficient, condition.

    Strategy:
        - Generate a realistic 30-min session with 30 Hz camera.
        - Run the full EKF filter (pre-integration + measurement update per frame).
        - Measure total processing time and assert mean per-frame ≤33 ms.

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
    print(f"Mean latency per frame (amortized): {mean_latency_per_frame_ms:.2f} ms")
    print(f"Camera frame period (30 Hz): {frame_period_ms:.2f} ms")
    print(
        f"PRD requirement: ≤{PRD_ONLINE_EKF_LATENCY_MS_MAX:.1f} ms per frame "
        "(this test checks MEAN only — necessary, not sufficient)"
    )
    print(
        f"Status: {'PASS ✓' if mean_latency_per_frame_ms <= PRD_ONLINE_EKF_LATENCY_MS_MAX else 'FAIL ✗'}"
    )

    # Validate PRD ≤33 ms requirement at the mean (necessary condition).
    # Tail / p99 verification requires an unrolled per-frame harness and is
    # not covered by this test.
    assert mean_latency_per_frame_ms <= PRD_ONLINE_EKF_LATENCY_MS_MAX, (
        f"Mean EKF latency {mean_latency_per_frame_ms:.2f} ms exceeds "
        f"PRD requirement {PRD_ONLINE_EKF_LATENCY_MS_MAX:.1f} ms (mean check)"
    )

    # Sanity check: verify filter produced valid results
    assert filter_result.filtered_means.shape[0] == num_frames, (
        "Filter output length mismatch"
    )
    assert filter_result.filtered_means.shape[1] == 8, (
        "Filter state dimension should be 8"
    )
    assert np.all(np.isfinite(filter_result.filtered_means)), (
        "Filter means contain NaN/Inf"
    )

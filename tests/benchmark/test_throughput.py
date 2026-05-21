"""Throughput and latency benchmarks for the performance acceptance targets.

This module validates system performance against the project's targets:
- Offline smoothing >=10x realtime (CPU) on a 30 min session
- Online EKF latency <=33 ms per frame (CPU) - measured here as amortized
  mean per-frame time (total / num_frames) over a single JIT'd ``lax.scan``
  batch, which is a *necessary* but not sufficient condition for the
  per-frame target. Per-frame tail / p99 latency is not measured by this
  suite (the filter is not driven from a streaming ingest loop), so this
  is a throughput-style proxy for the forward-only "online" CLI rather
  than a streaming / real-time guarantee.
"""

import time
from typing import Any

import jax
import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.runtime.offline import rts_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu


def _block_until_ready(result: Any) -> Any:
    """Force JAX dispatch to complete on every array leaf in ``result``.

    JAX execution is asynchronous: ``extended_kalman_filter`` and
    ``rts_smoother`` return ``ArrayImpl`` futures that the host gets back
    immediately while the underlying XLA computation is still running.
    Stopping the timer before materializing those leaves measures
    dispatch latency rather than completed compute, so headline
    throughput / latency numbers can under-report the real cost. Mirror
    the existing helper used in ``test_ekf_3d_core_jit.py`` to walk the
    pytree and call ``block_until_ready`` on every JAX array leaf.
    """
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result


def _assert_cpu_backend() -> None:
    """Assert the active JAX backend is CPU and report it.

    The acceptance floors are explicitly described as CPU targets ("≥10×
    realtime on CPU", "≤33 ms per frame on CPU"). Without this gate, a
    runner with ``JAX_PLATFORMS=cuda`` (or a local machine where jaxlib
    finds an accelerator) would silently use the GPU and the printed
    numbers would misrepresent what the floors actually cover. Run with
    ``JAX_PLATFORMS=cpu pytest -m benchmark`` to enforce.
    """
    backend = jax.default_backend()
    print(f"   JAX backend: {backend}")
    if backend != "cpu":
        raise RuntimeError(
            f"Throughput benchmarks claim CPU floors but jax.default_backend()"
            f" is {backend!r}. Set ``JAX_PLATFORMS=cpu`` (or "
            "``JAX_PLATFORM_NAME=cpu``) before running, or skip these tests "
            "on accelerator-equipped machines."
        )


# =============================================================================
# Performance Requirements (acceptance criteria)
# =============================================================================

TARGET_OFFLINE_SMOOTHER_SPEEDUP_MIN = 10.0  # Offline ≥10× realtime (CPU)
TARGET_ONLINE_EKF_LATENCY_MS_MAX = 33.0  # Online ≤33 ms per frame (CPU)

# Session duration for throughput benchmarking
BENCHMARK_SESSION_DURATION_S = 1800.0  # 30 minutes


# =============================================================================
# Helper: Production EKF Configuration
# =============================================================================


def get_benchmark_ekf_config(**overrides: Any) -> EKFConfig:
    """EKF configuration for the throughput benchmark.

    Sets sensor / dynamics fields (process noise, measurement noise,
    IMU noise densities, damping, LED spacing, heading measurement) to
    values matching the simulator below. **Internal toggles**
    (``adaptive_q_during_dropout``, ``freeze_bias_during_blackout``,
    ``reduce_imu_noise_during_blackout`` and the dropout-Q multipliers)
    are NOT overridden — the benchmark therefore exercises the same
    dropout-adaptive path users get from ``EKFConfig()`` defaults
    (``adaptive_q_during_dropout=True`` etc).

    The ``state_mode`` is forced to ``"2d_full"`` (8D) because
    ``simulate_rat_imu`` emits a 3-channel ``U_imu`` ``(yaw_rate,
    accel_x, accel_y)`` that matches the 8D layout's
    ``(b_gz, b_ax, b_ay)`` bias slot. The user-facing default
    ``"2d_cam_3d_imu"`` (10D) needs a 6-channel IMU input that this
    simulator does not produce; throughput on that layout would
    require a different sim and may differ from what's measured here.

    The headline floors checked by these tests therefore cover the
    synthetic 2D ``simulate_rat_imu`` path only. The YAML real-data
    workflow (``trodestrack online --config session.yaml``) is not
    covered: it runs through additional preprocessing (parquet
    loading, sample-and-hold removal, IMU calibration diagnostics,
    LED identity correction) and an optional vision-only safety
    check that roughly doubles filter wall-clock when enabled. The
    README, ``docs/index.md``, and ``docs/TROUBLESHOOTING.md``
    repeat this scope caveat next to the headline numbers.
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
        # Forced because the simulator only emits 3-channel U_imu —
        # see the docstring above. All other fields below this line
        # would override production defaults and are intentionally
        # NOT set so the benchmark mirrors what users actually run.
        state_mode="2d_full",
    )
    defaults.update(overrides)
    return EKFConfig(**defaults)


# Backward-compatible alias for any external callers / scripts.
get_production_ekf_config = get_benchmark_ekf_config


# =============================================================================
# Test 1: Offline Smoother Throughput (≥10× realtime)
# =============================================================================


@pytest.mark.slow
@pytest.mark.benchmark
def test_offline_smoother_throughput():
    """Offline RTS smoother should achieve >=10x realtime on CPU (acceptance target).

    Validates that the complete offline smoothing pipeline (filter + RTS smoother)
    can process a 30-minute session at least 10× faster than realtime on CPU.

    Throughput calculation:
        - Session duration: 30 minutes (1800 seconds)
        - Target: ≥10× realtime → processing time ≤ 180 seconds (3 minutes)

    Expected runtime: ~45-60 seconds (on modern CPU, measured on M-series Mac)
    """
    _assert_cpu_backend()
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

    # Measure total processing time (filter + smoother). Block on each
    # JAX result inside the timed interval so we measure completed
    # compute, not async dispatch — see ``_block_until_ready`` docstring.
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
    _block_until_ready(filter_result)

    # Run RTS smoother
    smoother_result = rts_smoother(
        filter_result=filter_result,
        ekf_config=ekf_config,
        t_imu=sim_data["t_imu"],
        U_imu=sim_data["U_imu"],
        t_cam=sim_data["t_cam_exp"],
        mask_cam=sim_data["mask_cam"],
    )
    _block_until_ready(smoother_result)

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
    print(f"Target: ≥{TARGET_OFFLINE_SMOOTHER_SPEEDUP_MIN:.1f}× realtime")
    print(
        f"Status: {'PASS ✓' if speedup >= TARGET_OFFLINE_SMOOTHER_SPEEDUP_MIN else 'FAIL ✗'}"
    )

    # Validate acceptance target
    assert speedup >= TARGET_OFFLINE_SMOOTHER_SPEEDUP_MIN, (
        f"Offline smoother speedup {speedup:.1f}× is below acceptance target {TARGET_OFFLINE_SMOOTHER_SPEEDUP_MIN:.1f}×"
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
    """Online EKF end-to-end latency <=33 ms per frame on CPU (acceptance target).

    Validates that the EKF can keep up with a 30 Hz camera over a long
    session: total wall-clock processing time divided by frame count
    must stay below the 33 ms frame period.

    Caveat: this is an amortized / mean per-frame check, not a per-frame
    tail-latency check. The filter runs as a single JIT-compiled
    ``lax.scan`` over the full session, so individual scan steps are not
    timed and slow tail frames cannot be detected here. Per-frame
    distribution and p99 measurement require an unrolled / online-loop
    harness; this test verifies only that the average frame budget is
    met. For the cited <=33 ms target, the mean is a *necessary*, not
    sufficient, condition.

    Strategy:
        - Generate a realistic 30-min session with 30 Hz camera.
        - Run the full EKF filter (pre-integration + measurement update per frame).
        - Measure total processing time and assert mean per-frame ≤33 ms.

    Expected runtime: ~45-60 seconds (on modern CPU, measured on M-series Mac)
    """
    _assert_cpu_backend()
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

    # Measure filter processing time. Block on the JAX result inside the
    # timed interval so we measure completed compute, not async dispatch.
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
    _block_until_ready(filter_result)

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
        f"Target: ≤{TARGET_ONLINE_EKF_LATENCY_MS_MAX:.1f} ms per frame "
        "(this test checks MEAN only — necessary, not sufficient)"
    )
    print(
        f"Status: {'PASS ✓' if mean_latency_per_frame_ms <= TARGET_ONLINE_EKF_LATENCY_MS_MAX else 'FAIL ✗'}"
    )

    # Validate the <=33 ms acceptance target at the mean (necessary condition).
    # Tail / p99 verification requires an unrolled per-frame harness and is
    # not covered by this test.
    assert mean_latency_per_frame_ms <= TARGET_ONLINE_EKF_LATENCY_MS_MAX, (
        f"Mean EKF latency {mean_latency_per_frame_ms:.2f} ms exceeds "
        f"acceptance target {TARGET_ONLINE_EKF_LATENCY_MS_MAX:.1f} ms (mean check)"
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

"""Baseline orientation-estimator performance and parity fixtures."""

from __future__ import annotations

import time

import numpy as np
import pytest

from trodestrack.models.orientation import (
    OrientationEstimatorConfig,
    estimate_orientation,
)

GRAVITY = 9.80665


def make_orientation_benchmark_inputs(
    *,
    duration_s: float = 60.0,
    fs_imu: float = 200.0,
    fs_cam: float = 30.0,
) -> dict[str, np.ndarray]:
    """Create deterministic IMU/camera inputs for orientation refactor parity."""

    t_imu = np.arange(0.0, duration_s + 0.5 / fs_imu, 1.0 / fs_imu)
    t_cam = np.arange(0.0, duration_s + 0.5 / fs_cam, 1.0 / fs_cam)
    yaw_rate = 0.22
    gyro_bias_z = 0.025
    gyro_xyz = np.zeros((t_imu.shape[0], 3), dtype=float)
    gyro_xyz[:, 2] = yaw_rate + gyro_bias_z
    accel_xyz = np.tile([0.0, 0.0, GRAVITY], (t_imu.shape[0], 1))

    heading = yaw_rate * t_cam
    half_led_distance = 0.02
    led1 = np.column_stack(
        [
            -half_led_distance * np.cos(heading),
            -half_led_distance * np.sin(heading),
        ]
    )
    led2 = -led1

    return {
        "t_imu": t_imu,
        "gyro_xyz": gyro_xyz,
        "accel_xyz": accel_xyz,
        "t_cam": t_cam,
        "led1": led1,
        "led2": led2,
    }


def _run_orientation_baseline(inputs: dict[str, np.ndarray]):
    return estimate_orientation(
        t_imu=inputs["t_imu"],
        gyro_xyz=inputs["gyro_xyz"],
        accel_xyz=inputs["accel_xyz"],
        t_cam=inputs["t_cam"],
        led1=inputs["led1"],
        led2=inputs["led2"],
        config=OrientationEstimatorConfig(
            initial_gyro_bias_rad_s=np.zeros(3),
            camera_speed_threshold_m_s=0.05,
        ),
    )


def test_orientation_estimator_baseline_outputs_are_stable() -> None:
    """Pin deterministic outputs for orientation refactor parity."""

    inputs = make_orientation_benchmark_inputs(duration_s=12.0)

    result = _run_orientation_baseline(inputs)

    assert result.quaternions.shape == (inputs["t_imu"].shape[0], 4)
    assert result.gyro_bias_rad_s.shape == (inputs["t_imu"].shape[0], 3)
    assert np.isfinite(result.quaternions).all()
    assert np.isfinite(result.gyro_bias_rad_s).all()
    np.testing.assert_allclose(
        np.linalg.norm(result.quaternions, axis=1),
        1.0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.yaw[-1],
        2.642939567565918,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.gyro_bias_rad_s[-1],
        [0.0, 0.0, 0.00014809130516368896],
        atol=1e-7,
    )
    assert result.diagnostics.gravity_update_fraction == 0.0
    assert 0.14 < result.diagnostics.camera_yaw_update_fraction < 0.16
    assert result.diagnostics.yaw_camera_rmse_rad is not None
    assert result.diagnostics.yaw_camera_rmse_rad < 0.12


@pytest.mark.benchmark
def test_orientation_estimator_scan_throughput_baseline() -> None:
    """Record warmed scan throughput for the orientation estimator."""

    inputs = make_orientation_benchmark_inputs(duration_s=60.0)

    _run_orientation_baseline(inputs)
    start = time.perf_counter()
    result = _run_orientation_baseline(inputs)
    elapsed_s = time.perf_counter() - start
    throughput_hz = inputs["t_imu"].shape[0] / elapsed_s

    print("\n=== Orientation Estimator Scan Throughput ===")
    print(f"IMU samples: {inputs['t_imu'].shape[0]}")
    print(f"Elapsed: {elapsed_s:.3f} s")
    print(f"Throughput: {throughput_hz:.0f} IMU samples/s")

    assert np.isfinite(result.quaternions).all()
    assert throughput_hz > 100_000.0

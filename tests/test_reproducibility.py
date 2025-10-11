"""Tests for reproducibility and determinism.

This module verifies that:
1. Simulations are deterministic with fixed seeds
2. Filters produce identical results with same inputs
3. Version pinning is in place (uv.lock exists)

Test cases ensure that research results can be reproduced exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.runtime.offline import rts_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.simple import SimpleSimConfig, simulate_circular, simulate_stationary


def test_simulation_determinism_rat_imu():
    """Verify that rat_imu simulations are deterministic with fixed seed."""
    config = RatIMUSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)

    # Run simulation twice with same seed
    sim1 = simulate_rat_imu(config=config, seed=42)
    sim2 = simulate_rat_imu(config=config, seed=42)

    # All arrays should be identical
    np.testing.assert_array_equal(sim1["t_imu"], sim2["t_imu"])
    np.testing.assert_array_equal(sim1["t_cam_exp"], sim2["t_cam_exp"])
    np.testing.assert_array_equal(sim1["X_truth"], sim2["X_truth"])
    np.testing.assert_array_equal(sim1["U_imu"], sim2["U_imu"])
    np.testing.assert_array_equal(sim1["Z_cam_led1"], sim2["Z_cam_led1"])
    np.testing.assert_array_equal(sim1["Z_cam_led2"], sim2["Z_cam_led2"])
    np.testing.assert_array_equal(sim1["mask_cam"], sim2["mask_cam"])


def test_simulation_determinism_simple_stationary():
    """Verify that simple stationary simulations are deterministic with fixed seed."""
    config = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)

    # Run simulation twice with same seed
    sim1 = simulate_stationary(config=config, seed=42)
    sim2 = simulate_stationary(config=config, seed=42)

    # All arrays should be identical
    np.testing.assert_array_equal(sim1["t_imu"], sim2["t_imu"])
    np.testing.assert_array_equal(sim1["t_cam_exp"], sim2["t_cam_exp"])
    np.testing.assert_array_equal(sim1["X_truth"], sim2["X_truth"])
    np.testing.assert_array_equal(sim1["U_imu"], sim2["U_imu"])
    np.testing.assert_array_equal(sim1["Z_cam_led1"], sim2["Z_cam_led1"])
    np.testing.assert_array_equal(sim1["mask_cam"], sim2["mask_cam"])


def test_simulation_determinism_simple_circular():
    """Verify that simple circular simulations are deterministic with fixed seed."""
    config = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)

    # Run simulation twice with same seed
    sim1 = simulate_circular(config=config, seed=42)
    sim2 = simulate_circular(config=config, seed=42)

    # All arrays should be identical
    np.testing.assert_array_equal(sim1["t_imu"], sim2["t_imu"])
    np.testing.assert_array_equal(sim1["t_cam_exp"], sim2["t_cam_exp"])
    np.testing.assert_array_equal(sim1["X_truth"], sim2["X_truth"])
    np.testing.assert_array_equal(sim1["U_imu"], sim2["U_imu"])
    np.testing.assert_array_equal(sim1["Z_cam_led1"], sim2["Z_cam_led1"])
    np.testing.assert_array_equal(sim1["mask_cam"], sim2["mask_cam"])


def test_simulation_different_seeds_produce_different_results():
    """Verify that different seeds produce different results."""
    config = RatIMUSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)

    # Run simulation with different seeds
    sim1 = simulate_rat_imu(config=config, seed=42)
    sim2 = simulate_rat_imu(config=config, seed=123)

    # IMU timestamps should be identical (deterministic grid)
    np.testing.assert_array_equal(sim1["t_imu"], sim2["t_imu"])

    # But stochastic elements should differ (at least one of these should be different)
    # Note: t_cam_exp can differ due to dropouts, X_truth differs due to OU dynamics
    arrays_differ = (
        not np.array_equal(sim1["X_truth"], sim2["X_truth"])
        or not np.array_equal(sim1["U_imu"], sim2["U_imu"])
        or not np.array_equal(sim1["Z_cam_led1"], sim2["Z_cam_led1"])
        or not np.array_equal(sim1["t_cam_exp"], sim2["t_cam_exp"])
    )
    assert arrays_differ, "Different seeds should produce different results"


def test_filter_determinism():
    """Verify that EKF filtering is deterministic with same inputs."""
    # Generate deterministic test data
    config = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
    sim = simulate_stationary(config=config, seed=42)

    # Configure filter
    ekf_config = EKFConfig(
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        measurement_noise_pos=0.005**2,
        imu_gyro_noise_density=0.001,
        imu_accel_noise_density=0.05,
    )

    # Run filter twice
    result1 = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    result2 = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Results should be identical
    np.testing.assert_array_equal(result1.filtered_means, result2.filtered_means)
    np.testing.assert_array_equal(result1.filtered_covariances, result2.filtered_covariances)
    assert result1.marginal_loglik == result2.marginal_loglik


def test_smoother_determinism():
    """Verify that RTS smoothing is deterministic with same inputs."""
    # Generate deterministic test data
    config = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
    sim = simulate_stationary(config=config, seed=42)

    # Configure filter
    ekf_config = EKFConfig(
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        measurement_noise_pos=0.005**2,
        imu_gyro_noise_density=0.001,
        imu_accel_noise_density=0.05,
    )

    # Run filter once
    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Run smoother twice
    smoother1 = rts_smoother(
        filter_result=filter_result,
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        num_iter=1,
    )

    smoother2 = rts_smoother(
        filter_result=filter_result,
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        num_iter=1,
    )

    # Results should be identical
    np.testing.assert_array_equal(smoother1.smoothed_means, smoother2.smoothed_means)
    np.testing.assert_array_equal(smoother1.smoothed_covariances, smoother2.smoothed_covariances)
    assert smoother1.marginal_loglik == smoother2.marginal_loglik


def test_uv_lock_exists():
    """Verify that uv.lock exists for version pinning."""
    # Check from project root
    lock_file = Path(__file__).parent.parent / "uv.lock"
    assert (
        lock_file.exists()
    ), "uv.lock file not found. Run 'uv sync' to generate lock file for reproducibility."

    # Verify it's not empty
    assert lock_file.stat().st_size > 0, "uv.lock file is empty"


def test_python_version_file_exists():
    """Verify that .python-version exists for Python version pinning."""
    # Check from project root
    version_file = Path(__file__).parent.parent / ".python-version"
    assert (
        version_file.exists()
    ), ".python-version file not found. This file pins the Python version for reproducibility."

    # Read and verify format
    version = version_file.read_text().strip()
    assert version.startswith("3."), f"Python version should start with '3.', got: {version}"


def test_ci_workflow_exists():
    """Verify that CI workflow exists with required checks."""
    # Check from project root
    ci_file = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
    assert ci_file.exists(), "CI workflow file (.github/workflows/ci.yml) not found"

    # Read workflow content
    ci_content = ci_file.read_text()

    # Verify required checks are present
    assert "mypy" in ci_content, "CI workflow must include mypy type checking"
    assert "ruff" in ci_content, "CI workflow must include ruff linting"
    assert "pytest" in ci_content, "CI workflow must include pytest testing"
    assert "black" in ci_content, "CI workflow must include black formatting check"


@pytest.mark.parametrize("seed", [0, 42, 123, 999])
def test_reproducibility_across_multiple_seeds(seed):
    """Verify reproducibility holds for various seed values."""
    config = SimpleSimConfig(duration_s=1.0, fs_imu=200.0, fs_cam=30.0)

    # Run simulation twice with same seed
    sim1 = simulate_stationary(config=config, seed=seed)
    sim2 = simulate_stationary(config=config, seed=seed)

    # Results should be identical
    np.testing.assert_array_equal(sim1["X_truth"], sim2["X_truth"])
    np.testing.assert_array_equal(sim1["U_imu"], sim2["U_imu"])
    np.testing.assert_array_equal(sim1["Z_cam_led1"], sim2["Z_cam_led1"])


def test_end_to_end_reproducibility():
    """Verify end-to-end reproducibility: simulation -> filter -> smoother."""
    # Fixed seed for entire pipeline
    seed = 42

    # Run full pipeline twice
    def run_pipeline():
        # Simulation
        config = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
        sim = simulate_stationary(config=config, seed=seed)

        # Filter
        ekf_config = EKFConfig(
            process_noise_pos=0.02,
            process_noise_vel=2.0,
            measurement_noise_pos=0.005**2,
        )
        filter_result = extended_kalman_filter(
            ekf_config=ekf_config,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Smoother
        smoother_result = rts_smoother(
            filter_result=filter_result,
            ekf_config=ekf_config,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
        )

        return smoother_result.smoothed_means, smoother_result.smoothed_covariances

    # Run twice
    means1, covs1 = run_pipeline()
    means2, covs2 = run_pipeline()

    # Results should be identical
    np.testing.assert_array_equal(means1, means2)
    np.testing.assert_array_equal(covs1, covs2)

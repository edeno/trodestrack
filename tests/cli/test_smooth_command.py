"""Tests for trodestrack smooth CLI command.

This module tests the smooth subcommand functionality which runs offline
smoothing on simulation data and saves results to disk.

Test cases:
1. Test help message displays correctly
2. Test smooth command with minimal config (stationary scenario)
3. Test output directory creation and file structure
4. Test error handling for missing/invalid files
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trodestrack import main


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def synthetic_data_files(temp_output_dir):
    """Generate minimal synthetic data files for testing.

    Creates:
        - t_imu.txt: IMU timestamps
        - U_imu.txt: IMU measurements [ω_z, f_x, f_y]
        - t_cam.txt: Camera timestamps
        - Z_cam_led1.txt: LED1 positions [x, y]
        - Z_cam_led2.txt: LED2 positions [x, y]
        - mask_cam.txt: Camera validity mask
    """
    # Generate 2 seconds of data
    duration_s = 2.0
    fs_imu = 200.0
    fs_cam = 30.0

    # IMU data (200 Hz)
    n_imu = int(duration_s * fs_imu)
    t_imu = np.linspace(0, duration_s, n_imu)
    # Simple stationary scenario: small noise around zero
    U_imu = np.column_stack(
        [
            np.random.randn(n_imu) * 0.001,  # gyro: small noise
            np.random.randn(n_imu) * 0.01,  # accel_x: small noise
            np.random.randn(n_imu) * 0.01,  # accel_y: small noise
        ]
    )

    # Camera data (30 Hz)
    n_cam = int(duration_s * fs_cam)
    t_cam = np.linspace(0, duration_s, n_cam)
    # Stationary at origin with LED spacing 0.04 m
    Z_cam_led1 = np.tile([0.5, 0.5], (n_cam, 1)) + np.random.randn(n_cam, 2) * 0.005
    Z_cam_led2 = (
        np.tile([0.54, 0.5], (n_cam, 1)) + np.random.randn(n_cam, 2) * 0.005
    )  # 4 cm forward
    mask_cam = np.ones(n_cam, dtype=bool)

    # Save to files
    input_dir = temp_output_dir / "input"
    input_dir.mkdir()

    np.savetxt(input_dir / "t_imu.txt", t_imu)
    np.savetxt(input_dir / "U_imu.txt", U_imu)
    np.savetxt(input_dir / "t_cam.txt", t_cam)
    np.savetxt(input_dir / "Z_cam_led1.txt", Z_cam_led1)
    np.savetxt(input_dir / "Z_cam_led2.txt", Z_cam_led2)
    np.savetxt(input_dir / "mask_cam.txt", mask_cam, fmt="%d")

    return input_dir


def test_smooth_help_message():
    """Test that smooth --help displays usage information."""
    with patch("sys.argv", ["trodestrack", "smooth", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        # Help should exit with code 0
        assert exc_info.value.code == 0


def test_smooth_command_creates_output_directory(synthetic_data_files, temp_output_dir):
    """Test that smooth command creates output directory structure."""
    input_dir = synthetic_data_files
    output_dir = temp_output_dir / "run1"

    # Run smooth command
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "smooth",
            "--imu-timestamps",
            str(input_dir / "t_imu.txt"),
            "--imu-measurements",
            str(input_dir / "U_imu.txt"),
            "--camera-timestamps",
            str(input_dir / "t_cam.txt"),
            "--led1-positions",
            str(input_dir / "Z_cam_led1.txt"),
            "--led2-positions",
            str(input_dir / "Z_cam_led2.txt"),
            "--camera-mask",
            str(input_dir / "mask_cam.txt"),
            "--output-dir",
            str(output_dir),
        ],
    ):
        main()

    # Check output directory was created
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_smooth_command_saves_required_outputs(synthetic_data_files, temp_output_dir):
    """Test that smooth command saves all required output files."""
    input_dir = synthetic_data_files
    output_dir = temp_output_dir / "run1"

    # Run smooth command
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "smooth",
            "--imu-timestamps",
            str(input_dir / "t_imu.txt"),
            "--imu-measurements",
            str(input_dir / "U_imu.txt"),
            "--camera-timestamps",
            str(input_dir / "t_cam.txt"),
            "--led1-positions",
            str(input_dir / "Z_cam_led1.txt"),
            "--led2-positions",
            str(input_dir / "Z_cam_led2.txt"),
            "--camera-mask",
            str(input_dir / "mask_cam.txt"),
            "--output-dir",
            str(output_dir),
        ],
    ):
        main()

    # Check required outputs exist
    assert (output_dir / "smoothed_means.txt").exists()
    assert (output_dir / "smoothed_covariances.txt").exists()
    assert (output_dir / "filtered_means.txt").exists()
    assert (output_dir / "filtered_covariances.txt").exists()
    assert (output_dir / "marginal_loglik.txt").exists()

    # Verify data shapes are correct
    smoothed_means = np.loadtxt(output_dir / "smoothed_means.txt")
    n_cam = len(np.loadtxt(input_dir / "t_cam.txt"))
    assert smoothed_means.shape[0] == n_cam
    assert smoothed_means.shape[1] == 10  # Default 10D state (2d_cam_3d_imu)


def test_smooth_command_missing_input_file(temp_output_dir):
    """Test that smooth command handles missing input files gracefully."""
    output_dir = temp_output_dir / "run1"

    # Run smooth command with non-existent files
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "smooth",
            "--imu-timestamps",
            str(temp_output_dir / "nonexistent_t_imu.txt"),
            "--imu-measurements",
            str(temp_output_dir / "nonexistent_U_imu.txt"),
            "--camera-timestamps",
            str(temp_output_dir / "nonexistent_t_cam.txt"),
            "--led1-positions",
            str(temp_output_dir / "nonexistent_led1.txt"),
            "--output-dir",
            str(output_dir),
        ],
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
        # Should exit with error code
        assert exc_info.value.code != 0


@pytest.fixture
def ten_second_session(temp_output_dir):
    """Synthetic 10 s session with a gentle yaw rotation.

    Long enough to exercise prediction / update cycles past the initial
    transient (which is where covariance asymmetries from numerical noise
    would first surface). LED spacing is constant and matches
    ``led_distance=0.04``, so the heading pseudo-measurement is well-posed
    on every frame.
    """

    duration_s = 10.0
    fs_imu = 200.0
    fs_cam = 30.0
    rng = np.random.default_rng(0)

    n_imu = int(duration_s * fs_imu)
    t_imu = np.linspace(0.0, duration_s, n_imu)
    yaw_rate = 0.1  # rad/s
    U_imu = np.column_stack(
        [
            np.full(n_imu, yaw_rate) + rng.standard_normal(n_imu) * 1e-3,
            rng.standard_normal(n_imu) * 1e-2,
            rng.standard_normal(n_imu) * 1e-2,
        ]
    )

    n_cam = int(duration_s * fs_cam)
    t_cam = np.linspace(0.0, duration_s, n_cam)
    heading = yaw_rate * t_cam
    center = np.column_stack([np.full(n_cam, 0.5), np.full(n_cam, 0.5)])
    half_spacing = 0.02
    led1 = center - half_spacing * np.column_stack([np.cos(heading), np.sin(heading)])
    led2 = center + half_spacing * np.column_stack([np.cos(heading), np.sin(heading)])
    led1 += rng.standard_normal(led1.shape) * 0.003
    led2 += rng.standard_normal(led2.shape) * 0.003
    mask = np.ones(n_cam, dtype=bool)

    input_dir = temp_output_dir / "input_long"
    input_dir.mkdir()
    np.savetxt(input_dir / "t_imu.txt", t_imu)
    np.savetxt(input_dir / "U_imu.txt", U_imu)
    np.savetxt(input_dir / "t_cam.txt", t_cam)
    np.savetxt(input_dir / "Z_cam_led1.txt", led1)
    np.savetxt(input_dir / "Z_cam_led2.txt", led2)
    np.savetxt(input_dir / "mask_cam.txt", mask, fmt="%d")
    return input_dir


def _assert_outputs_are_finite_and_psd(
    means_path: Path, cov_path: Path, *, rtol: float = 1e-8
) -> None:
    """Shared finiteness + symmetry + PD check for smooth/online outputs."""

    means = np.loadtxt(means_path)
    flat_covs = np.loadtxt(cov_path)
    n_cam = means.shape[0]
    state_dim = means.shape[1]
    assert flat_covs.shape == (n_cam, state_dim * state_dim)
    covs = flat_covs.reshape(n_cam, state_dim, state_dim)

    assert np.all(np.isfinite(means)), "means contain NaN/Inf"
    assert np.all(np.isfinite(covs)), "covariances contain NaN/Inf"

    # Symmetry within rtol on every frame.
    asym = np.abs(covs - np.swapaxes(covs, 1, 2))
    max_sym_violation = float(asym.max())
    cov_scale = float(np.abs(covs).max())
    assert max_sym_violation <= rtol * max(cov_scale, 1.0), (
        f"covariance asymmetry {max_sym_violation:.3e} exceeds "
        f"rtol*scale={rtol * cov_scale:.3e}"
    )

    # Positive definiteness (smallest eigenvalue > 0 on every frame).
    sym_covs = 0.5 * (covs + np.swapaxes(covs, 1, 2))
    eigs = np.linalg.eigvalsh(sym_covs)
    min_eig = float(eigs.min())
    assert min_eig > 0.0, f"covariance not PD: min eigenvalue {min_eig:.3e}"


def test_smooth_command_outputs_are_finite_and_psd(ten_second_session, temp_output_dir):
    """Smoothed means + covariances must stay finite and PSD across a 10 s run.

    Probes silent numerical-stability failures (negative eigenvalues from
    Joseph-form drift, NaN propagation through gated updates) that the
    existing CLI tests don't check because they only assert file existence.
    """

    output_dir = temp_output_dir / "run_smooth_finite"
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "smooth",
            "--imu-timestamps",
            str(ten_second_session / "t_imu.txt"),
            "--imu-measurements",
            str(ten_second_session / "U_imu.txt"),
            "--camera-timestamps",
            str(ten_second_session / "t_cam.txt"),
            "--led1-positions",
            str(ten_second_session / "Z_cam_led1.txt"),
            "--led2-positions",
            str(ten_second_session / "Z_cam_led2.txt"),
            "--camera-mask",
            str(ten_second_session / "mask_cam.txt"),
            "--output-dir",
            str(output_dir),
        ],
    ):
        main()

    _assert_outputs_are_finite_and_psd(
        output_dir / "smoothed_means.txt",
        output_dir / "smoothed_covariances.txt",
    )


def test_smooth_command_with_filter_config(synthetic_data_files, temp_output_dir):
    """Test smooth command with custom filter configuration."""
    input_dir = synthetic_data_files
    output_dir = temp_output_dir / "run1"

    # Run with custom process noise
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "smooth",
            "--imu-timestamps",
            str(input_dir / "t_imu.txt"),
            "--imu-measurements",
            str(input_dir / "U_imu.txt"),
            "--camera-timestamps",
            str(input_dir / "t_cam.txt"),
            "--led1-positions",
            str(input_dir / "Z_cam_led1.txt"),
            "--led2-positions",
            str(input_dir / "Z_cam_led2.txt"),
            "--camera-mask",
            str(input_dir / "mask_cam.txt"),
            "--output-dir",
            str(output_dir),
            "--process-noise-pos",
            "0.05",
            "--process-noise-vel",
            "3.0",
            "--damping-coeff",
            "0.5",
        ],
    ):
        main()

    # Should complete successfully
    assert (output_dir / "smoothed_means.txt").exists()

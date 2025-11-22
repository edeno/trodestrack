"""Tests for trodestrack online CLI command.

This module tests the online subcommand functionality which runs real-time
filtering (EKF forward pass only, no smoothing).

Test cases:
1. Test help message displays correctly
2. Test online command with minimal config (stationary scenario)
3. Test output directory creation and file structure
4. Test error handling for missing/invalid files
5. Test streaming mode (process frame-by-frame)
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


def test_online_help_message():
    """Test that online --help displays usage information."""
    with patch("sys.argv", ["trodestrack", "online", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        # Help should exit with code 0
        assert exc_info.value.code == 0


def test_online_command_creates_output_directory(synthetic_data_files, temp_output_dir):
    """Test that online command creates output directory structure."""
    input_dir = synthetic_data_files
    output_dir = temp_output_dir / "run1"

    # Run online command
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "online",
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


def test_online_command_saves_required_outputs(synthetic_data_files, temp_output_dir):
    """Test that online command saves all required output files (filter only, no smoother)."""
    input_dir = synthetic_data_files
    output_dir = temp_output_dir / "run1"

    # Run online command
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "online",
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

    # Check required outputs exist (filter only, no smoother)
    assert (output_dir / "filtered_means.txt").exists()
    assert (output_dir / "filtered_covariances.txt").exists()
    assert (output_dir / "marginal_loglik.txt").exists()

    # Smoother outputs should NOT exist for online mode
    assert not (output_dir / "smoothed_means.txt").exists()
    assert not (output_dir / "smoothed_covariances.txt").exists()

    # Verify data shapes are correct
    filtered_means = np.loadtxt(output_dir / "filtered_means.txt")
    n_cam = len(np.loadtxt(input_dir / "t_cam.txt"))
    assert filtered_means.shape[0] == n_cam
    assert filtered_means.shape[1] == 10  # Default 10D state (2d_cam_3d_imu)


def test_online_command_missing_input_file(temp_output_dir):
    """Test that online command handles missing input files gracefully."""
    output_dir = temp_output_dir / "run1"

    # Run online command with non-existent files
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "online",
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


def test_online_command_with_filter_config(synthetic_data_files, temp_output_dir):
    """Test online command with custom filter configuration."""
    input_dir = synthetic_data_files
    output_dir = temp_output_dir / "run1"

    # Run with custom process noise
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "online",
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
    assert (output_dir / "filtered_means.txt").exists()

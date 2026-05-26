"""Tests for the ``trodestrack filter`` CLI command.

The ``filter`` subcommand runs the EKF in forward-pass-only mode (no
backward smoothing) as a batch over complete IMU/camera/LED arrays loaded
from disk. These tests cover the batch end-to-end behaviour.

Test cases:
1. Help message displays correctly.
2. Filter command with minimal config (stationary scenario), end-to-end batch run.
3. Output directory creation and file structure.
4. Error handling for missing/invalid input files.
5. Argument parsing fall-through to ``EKFConfig`` defaults.
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


def test_filter_help_message():
    """Test that filter --help displays usage information."""
    with patch("sys.argv", ["trodestrack", "filter", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        # Help should exit with code 0
        assert exc_info.value.code == 0


def test_filter_command_creates_output_directory(
    synthetic_data_files, temp_output_dir, smooth_filter_io_args
):
    """Test that filter command creates output directory structure."""
    output_dir = temp_output_dir / "run1"
    argv = [
        "trodestrack",
        "filter",
        *smooth_filter_io_args(synthetic_data_files, output_dir),
    ]
    with patch("sys.argv", argv):
        main()

    # Check output directory was created
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_filter_command_saves_required_outputs(
    synthetic_data_files, temp_output_dir, smooth_filter_io_args
):
    """Test that filter command saves all required output files (filter only, no smoother)."""
    output_dir = temp_output_dir / "run1"
    argv = [
        "trodestrack",
        "filter",
        *smooth_filter_io_args(synthetic_data_files, output_dir),
    ]
    with patch("sys.argv", argv):
        main()

    # Check required outputs exist (filter only, no smoother)
    assert (output_dir / "filtered_means.txt").exists()
    assert (output_dir / "filtered_covariances.txt").exists()
    assert (output_dir / "marginal_loglik.txt").exists()

    # Smoother outputs should NOT exist for the filter-only command
    assert not (output_dir / "smoothed_means.txt").exists()
    assert not (output_dir / "smoothed_covariances.txt").exists()

    # Verify data shapes are correct
    filtered_means = np.loadtxt(output_dir / "filtered_means.txt")
    n_cam = len(np.loadtxt(synthetic_data_files / "t_cam.txt"))
    assert filtered_means.shape[0] == n_cam
    assert filtered_means.shape[1] == 10  # Default 10D state (2d_cam_3d_imu)


def test_filter_command_missing_input_file(temp_output_dir):
    """Test that filter command handles missing input files gracefully.

    Doesn't reuse ``smooth_filter_io_args`` because that helper expects
    a fully populated input directory; here we deliberately point flags
    at nonexistent paths so the CLI must reject them.
    """
    output_dir = temp_output_dir / "run1"

    # Run filter command with non-existent files
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "filter",
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

    Mirrors ``test_smooth_command.ten_second_session`` so the two
    finiteness tests use identical inputs (and therefore detect the same
    regressions in the shared filter core).
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


def test_filter_command_outputs_are_finite_and_psd(
    ten_second_session,
    temp_output_dir,
    smooth_filter_io_args,
    assert_outputs_are_finite_and_psd,
):
    """Filter-only outputs must stay finite and PSD across a 10 s run.

    Same probe as the smooth-command equivalent: silent numerical
    failures (negative eigenvalues from Joseph-form drift, NaN
    propagation) that the existing CLI tests don't catch because they
    only assert file existence.
    """

    output_dir = temp_output_dir / "run_filter_finite"
    argv = [
        "trodestrack",
        "filter",
        *smooth_filter_io_args(ten_second_session, output_dir),
    ]
    with patch("sys.argv", argv):
        main()

    assert_outputs_are_finite_and_psd(
        output_dir / "filtered_means.txt",
        output_dir / "filtered_covariances.txt",
    )


def test_filter_command_with_filter_config(
    synthetic_data_files, temp_output_dir, smooth_filter_io_args
):
    """Test filter command with custom filter configuration."""
    output_dir = temp_output_dir / "run1"
    argv = [
        "trodestrack",
        "filter",
        *smooth_filter_io_args(synthetic_data_files, output_dir),
        "--process-noise-pos",
        "0.05",
        "--process-noise-vel",
        "3.0",
        "--damping-coeff",
        "0.5",
    ]
    with patch("sys.argv", argv):
        main()

    # Should complete successfully
    assert (output_dir / "filtered_means.txt").exists()


# --- CLI error-path coverage -------------------------------------------------
#
# The filter command has 482 LoC and very thin error-path coverage. These
# tests probe the most likely user-visible failure modes that previously
# would have gone silent or produced opaque tracebacks: NaN in IMU rows,
# non-monotonic timestamps, mismatched-row inputs, wrong column count,
# malformed YAML config, and YAML referencing missing input files.


@pytest.fixture
def malformed_data_files(synthetic_data_files):
    """Re-yields the synthetic_data_files fixture for per-test mutation."""
    return synthetic_data_files


def _capture_filter_error(argv: list[str]) -> tuple[int, str]:
    """Run the CLI and return (exit_code, captured_stderr)."""
    import contextlib
    import io

    err_buf = io.StringIO()
    code: int | None = None
    with patch("sys.argv", argv), contextlib.redirect_stderr(err_buf):
        try:
            main()
            code = 0
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 1
    assert code is not None
    return code, err_buf.getvalue()


def test_filter_command_rejects_nan_in_imu_measurements(
    malformed_data_files, temp_output_dir, smooth_filter_io_args
):
    """A single NaN row in U_imu would poison every downstream state."""
    U_imu = np.loadtxt(malformed_data_files / "U_imu.txt")
    U_imu[42, 0] = np.nan
    np.savetxt(malformed_data_files / "U_imu.txt", U_imu)

    output_dir = temp_output_dir / "run_nan"
    argv = [
        "trodestrack",
        "filter",
        *smooth_filter_io_args(malformed_data_files, output_dir),
    ]
    code, err = _capture_filter_error(argv)
    assert code != 0
    assert "IMU measurements" in err and "non-finite" in err, err


def test_filter_command_rejects_nonmonotonic_imu_timestamps(
    malformed_data_files, temp_output_dir, smooth_filter_io_args
):
    """Non-monotonic timestamps would produce negative dt and NaN outputs."""
    t_imu = np.loadtxt(malformed_data_files / "t_imu.txt")
    t_imu[100] = t_imu[99] - 0.01  # decreasing step
    np.savetxt(malformed_data_files / "t_imu.txt", t_imu)

    output_dir = temp_output_dir / "run_nonmonotonic"
    argv = [
        "trodestrack",
        "filter",
        *smooth_filter_io_args(malformed_data_files, output_dir),
    ]
    code, err = _capture_filter_error(argv)
    assert code != 0
    assert "IMU timestamps" in err and "strictly increasing" in err, err


def test_filter_command_rejects_wrong_imu_column_count(
    malformed_data_files, temp_output_dir, smooth_filter_io_args
):
    """U_imu must have 3, 4, or 6 columns; anything else is a clear error."""
    U_imu = np.loadtxt(malformed_data_files / "U_imu.txt")
    # Strip to 2 columns (invalid layout).
    np.savetxt(malformed_data_files / "U_imu.txt", U_imu[:, :2])

    output_dir = temp_output_dir / "run_bad_cols"
    argv = [
        "trodestrack",
        "filter",
        *smooth_filter_io_args(malformed_data_files, output_dir),
    ]
    code, err = _capture_filter_error(argv)
    assert code != 0
    assert "IMU measurements" in err, err


def test_filter_command_rejects_mismatched_led1_shape(
    malformed_data_files, temp_output_dir, smooth_filter_io_args
):
    """LED1 row count must equal n_cam; mismatch is a clear error."""
    led1 = np.loadtxt(malformed_data_files / "Z_cam_led1.txt")
    # Drop the last 5 rows so shape no longer matches t_cam.
    np.savetxt(malformed_data_files / "Z_cam_led1.txt", led1[:-5])

    output_dir = temp_output_dir / "run_led1_mismatch"
    argv = [
        "trodestrack",
        "filter",
        *smooth_filter_io_args(malformed_data_files, output_dir),
    ]
    code, err = _capture_filter_error(argv)
    assert code != 0
    assert "LED1" in err and "shape" in err, err


def test_filter_command_rejects_malformed_yaml_config(tmp_path: Path) -> None:
    """A YAML config file with broken syntax must produce a clear error."""
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("inputs: [unterminated\n  - foo\n")  # invalid YAML

    output_dir = tmp_path / "out"
    argv = [
        "trodestrack",
        "filter",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    code, err = _capture_filter_error(argv)
    assert code != 0, err


def test_filter_command_rejects_yaml_with_missing_input_file(tmp_path: Path) -> None:
    """A YAML config referencing a nonexistent input file must error clearly."""
    output_dir = tmp_path / "out"
    config_path = tmp_path / "missing.yaml"
    config_path.write_text(
        f"""
inputs:
  format: prepared_arrays
  imu_timestamps: {tmp_path.name}/does_not_exist_t_imu.txt
  imu_measurements: {tmp_path.name}/does_not_exist_U_imu.txt
  camera_timestamps: {tmp_path.name}/does_not_exist_t_cam.txt
  led1_positions: {tmp_path.name}/does_not_exist_led1.txt
  led2_positions: {tmp_path.name}/does_not_exist_led2.txt
filter:
  state_mode: vision_only
outputs:
  output_dir: {output_dir.name}
""".lstrip()
    )

    argv = ["trodestrack", "filter", "--config", str(config_path)]
    code, err = _capture_filter_error(argv)
    assert code != 0, err

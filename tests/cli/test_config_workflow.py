"""CLI tests for YAML-configured online/smooth workflows."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trodestrack import main


def _write_prepared_config(tmp_path: Path, *, command: str) -> Path:
    input_dir = tmp_path / f"input_{command}"
    output_dir = tmp_path / f"out_{command}"
    input_dir.mkdir()
    t_imu = np.linspace(0.0, 0.5, 51)
    t_cam = np.linspace(0.0, 0.5, 16)
    center = np.column_stack([0.02 * t_cam, np.zeros_like(t_cam)])
    led1 = center - np.array([0.02, 0.0])
    led2 = center + np.array([0.02, 0.0])
    np.savetxt(input_dir / "t_imu.txt", t_imu)
    np.savetxt(input_dir / "U_imu.txt", np.zeros((len(t_imu), 3)))
    np.savetxt(input_dir / "t_cam.txt", t_cam)
    np.savetxt(input_dir / "led1.txt", led1)
    np.savetxt(input_dir / "led2.txt", led2)
    np.savetxt(input_dir / "mask.txt", np.ones(len(t_cam)), fmt="%d")
    config_path = tmp_path / f"{command}.yaml"
    config_path.write_text(
        f"""
inputs:
  format: prepared_arrays
  imu_timestamps: {input_dir.name}/t_imu.txt
  imu_measurements: {input_dir.name}/U_imu.txt
  camera_timestamps: {input_dir.name}/t_cam.txt
  led1_positions: {input_dir.name}/led1.txt
  led2_positions: {input_dir.name}/led2.txt
  camera_mask: {input_dir.name}/mask.txt
filter:
  state_mode: vision_only
outputs:
  output_dir: {output_dir.name}
  run_safety_checks: true
""".lstrip()
    )
    return config_path


def _write_arthur_config(tmp_path: Path, *, safety_max_speed_mps: float) -> Path:
    data_dir = tmp_path / "arthur_input"
    output_dir = tmp_path / "out_unsafe"
    data_dir.mkdir()
    t_imu = np.linspace(0.0, 0.5, 51)
    t_cam = np.linspace(0.0, 0.5, 16)
    pd.DataFrame(
        {
            "time": t_imu,
            "Headstage_GyroX": np.zeros_like(t_imu),
            "Headstage_GyroY": np.zeros_like(t_imu),
            "Headstage_GyroZ": np.linspace(0.0, 1.0, len(t_imu)),
            "Headstage_AccelX": np.zeros_like(t_imu),
            "Headstage_AccelY": np.zeros_like(t_imu),
            "Headstage_AccelZ": np.zeros_like(t_imu),
        }
    ).to_parquet(data_dir / "imu.parquet")
    center = np.column_stack([0.02 * t_cam, np.zeros_like(t_cam)])
    led1 = (center - np.array([0.02, 0.0])) / 0.01
    led2 = (center + np.array([0.02, 0.0])) / 0.01
    pd.DataFrame(
        {
            "time": t_cam,
            "xloc": led1[:, 0],
            "yloc": led1[:, 1],
            "xloc2": led2[:, 0],
            "yloc2": led2[:, 1],
        }
    ).to_parquet(data_dir / "position.parquet")
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(
        f"""
inputs:
  format: spikegadgets_trodes
  imu_file: {data_dir.name}/imu.parquet
  position_file: {data_dir.name}/position.parquet
imu:
  run_calibration: false
camera:
  meters_per_pixel: 0.01
filter:
  state_mode: 2d_full
outputs:
  output_dir: {output_dir.name}
  run_safety_checks: true
  safety_max_speed_mps: {safety_max_speed_mps}
""".lstrip()
    )
    return config_path


def test_online_config_writes_filter_outputs(tmp_path: Path) -> None:
    """``trodestrack online --config`` produces the expected output files."""

    config_path = _write_prepared_config(tmp_path, command="online")

    with patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]):
        main()

    output_dir = tmp_path / "out_online"
    filtered = np.loadtxt(output_dir / "filtered_means.txt")
    assert filtered.shape == (16, 5)
    assert (output_dir / "session_diagnostics.json").exists()
    assert not (output_dir / "smoothed_means.txt").exists()


def test_config_output_dir_flag_overrides_yaml(tmp_path: Path) -> None:
    """``--output-dir`` should override the YAML ``outputs.output_dir``."""

    config_path = _write_prepared_config(tmp_path, command="override")
    override_dir = tmp_path / "override_run"

    with patch(
        "sys.argv",
        [
            "trodestrack",
            "online",
            "--config",
            str(config_path),
            "--output-dir",
            str(override_dir),
        ],
    ):
        main()

    assert (override_dir / "filtered_means.txt").exists()
    assert not (tmp_path / "out_override" / "filtered_means.txt").exists()


def test_smooth_config_writes_filter_and_smoother_outputs(tmp_path: Path) -> None:
    """``trodestrack smooth --config`` runs the smoother over config inputs."""

    config_path = _write_prepared_config(tmp_path, command="smooth")

    with patch("sys.argv", ["trodestrack", "smooth", "--config", str(config_path)]):
        main()

    output_dir = tmp_path / "out_smooth"
    filtered = np.loadtxt(output_dir / "filtered_means.txt")
    smoothed = np.loadtxt(output_dir / "smoothed_means.txt")
    assert filtered.shape == (16, 5)
    assert smoothed.shape == (16, 5)
    assert (output_dir / "session_diagnostics.json").exists()


def test_spikegadgets_vision_only_config_skips_real_data_safety(
    tmp_path: Path,
) -> None:
    """Vision-only real-data configs should not run fused-output checks."""

    config_path = _write_arthur_config(tmp_path, safety_max_speed_mps=1e-9)
    config_path.write_text(
        config_path.read_text().replace(
            "state_mode: 2d_full", "state_mode: vision_only"
        )
    )

    with patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]):
        main()

    output_dir = tmp_path / "out_unsafe"
    assert (output_dir / "filtered_means.txt").exists()
    metadata = (output_dir / "metadata.txt").read_text()
    assert "disabled" in metadata
    assert "Mahalanobis gating: False" in metadata


def test_config_fused_safety_failure_is_friendly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Implausible Arthur-style fused runs fail before trusted output."""

    config_path = _write_arthur_config(tmp_path, safety_max_speed_mps=1e-9)

    with (
        patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert "physically implausible" in capsys.readouterr().err
    assert (tmp_path / "out_unsafe" / "session_diagnostics.json").exists()


def test_config_bad_imu_calibration_fails_with_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid IMU calibration should stop before writing trusted outputs."""

    config_path = _write_arthur_config(tmp_path, safety_max_speed_mps=3.0)
    config_path.write_text(
        config_path.read_text().replace("imu:\n  run_calibration: false\n", "")
    )

    with (
        patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert "IMU calibration diagnostics failed" in capsys.readouterr().err
    output_dir = tmp_path / "out_unsafe"
    assert (output_dir / "session_diagnostics.json").exists()
    assert not (output_dir / "filtered_means.txt").exists()

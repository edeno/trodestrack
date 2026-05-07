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
  safety_min_dual_led_frames: 1
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


def _write_orientation_config(tmp_path: Path, *, command: str) -> tuple[Path, Path]:
    input_dir = tmp_path / f"input_orientation_{command}"
    output_dir = tmp_path / f"out_orientation_{command}"
    input_dir.mkdir()
    t_imu = np.linspace(0.0, 0.5, 51)
    t_cam = np.linspace(0.0, 0.5, 16)
    center = np.column_stack([0.02 * t_cam, np.zeros_like(t_cam)])
    led1 = center - np.array([0.02, 0.0])
    led2 = center + np.array([0.02, 0.0])
    U_imu = np.zeros((len(t_imu), 6))
    U_imu[:, 5] = 9.81
    np.savetxt(input_dir / "t_imu.txt", t_imu)
    np.savetxt(input_dir / "U_imu.txt", U_imu)
    np.savetxt(input_dir / "t_cam.txt", t_cam)
    np.savetxt(input_dir / "led1.txt", led1)
    np.savetxt(input_dir / "led2.txt", led2)
    np.savetxt(input_dir / "mask.txt", np.ones(len(t_cam)), fmt="%d")
    config_path = tmp_path / f"orientation_{command}.yaml"
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
  state_mode: 2d_cam_6dof_imu_orientation
  enable_experimental_accel_translation: false
outputs:
  output_dir: {output_dir.name}
  run_safety_checks: true
""".lstrip()
    )
    return config_path, output_dir


def test_online_config_runs_orientation_fused_mode(tmp_path: Path) -> None:
    """``online --config`` should wire 6-channel IMU into the orientation EKF."""

    config_path, output_dir = _write_orientation_config(tmp_path, command="online")

    with patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]):
        main()

    filtered = np.loadtxt(output_dir / "filtered_means.txt")
    assert filtered.shape == (16, 14)


def test_smooth_config_runs_orientation_fused_mode(tmp_path: Path) -> None:
    """``smooth --config`` should run the orientation layout through RTS."""

    config_path, output_dir = _write_orientation_config(tmp_path, command="smooth")

    with patch("sys.argv", ["trodestrack", "smooth", "--config", str(config_path)]):
        main()

    filtered = np.loadtxt(output_dir / "filtered_means.txt")
    smoothed = np.loadtxt(output_dir / "smoothed_means.txt")
    assert filtered.shape == (16, 14)
    assert smoothed.shape == (16, 14)


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
    assert "Dual-LED frames: 0" in metadata
    assert "Deviation frames: 0" in metadata
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


def test_online_config_writes_camera_arrays_and_npz_bundle(tmp_path: Path) -> None:
    """Config-driven online run must persist t_cam, LEDs, mask, and the npz bundle.

    Programmatic consumers need the camera-frame side-data (timestamps,
    post-correction LEDs, validity mask) alongside the filter output;
    previously only ``filtered_means.txt`` and the flattened
    covariance were written, forcing users to re-run the loader to
    recover the inputs.
    """

    config_path = _write_prepared_config(tmp_path, command="online")

    with patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]):
        main()

    output_dir = tmp_path / "out_online"
    assert (output_dir / "t_cam.txt").exists()
    assert (output_dir / "Z_cam_led1.txt").exists()
    assert (output_dir / "Z_cam_led2.txt").exists()
    assert (output_dir / "mask_cam.txt").exists()

    bundle = np.load(output_dir / "filter_outputs.npz")
    assert set(bundle.files) >= {
        "t_cam",
        "Z_cam_led1",
        "Z_cam_led2",
        "mask_cam",
        "filtered_means",
        "filtered_covariances",
        "marginal_loglik",
    }
    assert bundle["filtered_means"].shape == (16, 5)


def test_smooth_config_npz_bundle_includes_smoother(tmp_path: Path) -> None:
    """``smooth --config`` must augment the npz bundle with smoother arrays."""

    config_path = _write_prepared_config(tmp_path, command="smooth")

    with patch("sys.argv", ["trodestrack", "smooth", "--config", str(config_path)]):
        main()

    output_dir = tmp_path / "out_smooth"
    bundle = np.load(output_dir / "filter_outputs.npz")
    assert set(bundle.files) >= {
        "filtered_means",
        "smoothed_means",
        "smoothed_covariances",
        "smoother_marginal_loglik",
    }
    assert bundle["smoothed_means"].shape == (16, 5)


def test_config_missing_input_file_does_not_print_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config whose YAML is valid but points at a missing input file
    must surface the file error *without* the run-started banner.

    The earlier deferral only moved the banner past
    ``load_session_config``. ``load_session`` (which actually opens
    the parquet/text files and reads parquet columns) ran *after*
    the banner, so file-not-found and column-missing errors were
    still framed under the header. Pin the contract: nothing
    prefixed by ``trodestrack — config-driven run`` may appear in
    captured output before the friendly stderr error fires.
    """

    # Valid YAML that points at a nonexistent IMU timestamps file.
    config_path = tmp_path / "missing_inputs.yaml"
    config_path.write_text(
        """
inputs:
  format: prepared_arrays
  imu_timestamps: nonexistent_t_imu.txt
  imu_measurements: nonexistent_U_imu.txt
  camera_timestamps: nonexistent_t_cam.txt
  led1_positions: nonexistent_led1.txt
filter:
  state_mode: vision_only
outputs:
  output_dir: out
""".lstrip()
    )

    with (
        patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "trodestrack — config-driven run" not in combined, (
        f"Banner printed before input-file error; got:\n{combined!r}"
    )


def test_config_missing_parquet_column_does_not_print_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A spikegadgets_trodes config whose parquet is missing a required
    column must surface the column error *without* the run-started banner.

    The earlier banner placement also covered ``load_session_config``
    only, which left ``_require_columns`` (run inside
    ``_load_spikegadgets_trodes``) firing *under* the header. The
    bug originally encompassed both prepared-array file errors and
    parquet-column errors; the prior regression test only pinned
    the file case. This test pins the column case explicitly.
    """

    data_dir = tmp_path / "missing_column_input"
    data_dir.mkdir()
    t_imu = np.linspace(0.0, 0.5, 51)
    t_cam = np.linspace(0.0, 0.5, 16)
    # Drop the ``Headstage_AccelZ`` column the IMU loader requires.
    pd.DataFrame(
        {
            "time": t_imu,
            "Headstage_GyroX": np.zeros_like(t_imu),
            "Headstage_GyroY": np.zeros_like(t_imu),
            "Headstage_GyroZ": np.zeros_like(t_imu),
            "Headstage_AccelX": np.zeros_like(t_imu),
            "Headstage_AccelY": np.zeros_like(t_imu),
        }
    ).to_parquet(data_dir / "imu.parquet")
    pd.DataFrame(
        {
            "time": t_cam,
            "xloc": np.zeros_like(t_cam),
            "yloc": np.zeros_like(t_cam),
            "xloc2": np.zeros_like(t_cam),
            "yloc2": np.zeros_like(t_cam),
        }
    ).to_parquet(data_dir / "position.parquet")
    config_path = tmp_path / "missing_column.yaml"
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
  output_dir: out
""".lstrip()
    )

    with (
        patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Headstage_AccelZ" in combined, (
        f"Expected the missing-column error to mention the column; got:\n{combined!r}"
    )
    assert "trodestrack — config-driven run" not in combined, (
        f"Banner printed before parquet column error; got:\n{combined!r}"
    )


def test_config_safety_check_raise_writes_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run_real_data_safety_check`` raising must still dump session_diagnostics.

    The two ``raise ValueError`` sites in ``io/session.py`` (no
    finite dual-LED frame; no finite fused/vision pair) used to
    bubble out without writing diagnostics, leaving the user with
    only the friendly stderr line and no JSON to inspect. The
    not-passed branch already wrote diagnostics; this test pins
    the symmetric raise path so a regression would be caught.
    """

    config_path = _write_arthur_config(tmp_path, safety_max_speed_mps=3.0)
    output_dir = tmp_path / "out_unsafe"

    # Force the safety check to raise by replacing the camera arrays
    # with an all-NaN dual-LED stream after Arthur-style construction.
    # The simplest way to trigger the second raise (line 166 of
    # session.py) is to mock ``run_real_data_safety_check`` to raise
    # directly; that's faithful to the contract being pinned and
    # avoids depending on real-data corner-case data shapes.
    from unittest.mock import MagicMock

    from trodestrack.cli import config_workflow

    real_check = config_workflow.run_real_data_safety_check
    raised_message = (
        "Safety check requires at least one finite fused/vision position pair."
    )
    mocked = MagicMock(side_effect=ValueError(raised_message))
    config_workflow.run_real_data_safety_check = mocked
    try:
        with (
            patch("sys.argv", ["trodestrack", "online", "--config", str(config_path)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
    finally:
        config_workflow.run_real_data_safety_check = real_check

    assert exc_info.value.code == 1
    assert raised_message in capsys.readouterr().err
    assert (output_dir / "session_diagnostics.json").exists()


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

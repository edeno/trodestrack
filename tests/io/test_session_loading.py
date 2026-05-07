"""Tests for config-driven session loading and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trodestrack.config import load_session_config
from trodestrack.io import load_session, write_session_diagnostics


def test_load_arthur_style_parquets_removes_sample_hold_and_projects_imu(tmp_path):
    """Arthur/Trodes parquet layout loads into filter-ready arrays."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    imu_path = data_dir / "imu.parquet"
    position_path = data_dir / "position.parquet"
    pd.DataFrame(
        {
            "time": [100.0, 100.005, 100.010, 100.015, 100.020],
            "Headstage_GyroX": [1, 1, 1, 1, 1],
            "Headstage_GyroY": [2, 2, 2, 2, 2],
            "Headstage_GyroZ": [0, 0, 10, 10, 20],
            "Headstage_AccelX": [100, 100, 110, 110, 120],
            "Headstage_AccelY": [200, 200, 210, 210, 220],
            "Headstage_AccelZ": [300, 300, 310, 310, 320],
        }
    ).to_parquet(imu_path)
    pd.DataFrame(
        {
            "time": [100.0, 100.033, 100.066],
            "xloc": [10.0, 11.0, 12.0],
            "yloc": [20.0, 20.0, 20.0],
            "xloc2": [14.0, 15.0, 16.0],
            "yloc2": [20.0, 20.0, 20.0],
        }
    ).to_parquet(position_path)
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
camera:
  meters_per_pixel: 0.01
filter:
  state_mode: 2d_cam_3d_imu
outputs:
  run_safety_checks: false
""".lstrip()
    )

    session = load_session(load_session_config(config_path))

    assert session.t_imu.shape == (3,)
    assert session.U_imu.shape == (3, 4)
    assert session.Z_cam_led1.shape == (3, 2)
    assert session.Z_cam_led2.shape == (3, 2)
    assert session.mask_cam.tolist() == [True, True, True]
    assert session.led_distance == pytest.approx(0.04)
    assert session.diagnostics["loader"]["sample_hold_strategy"] == "gyro_z_change"


def test_prepared_arrays_session_loads_and_writes_diagnostics(tmp_path):
    """Existing prepared text arrays continue to load through the config API."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    np.savetxt(input_dir / "t_imu.txt", [0.0, 0.01, 0.02])
    np.savetxt(input_dir / "U_imu.txt", np.zeros((3, 3)))
    np.savetxt(input_dir / "t_cam.txt", [0.0, 0.033])
    np.savetxt(input_dir / "led1.txt", [[0.0, 0.0], [0.01, 0.0]])
    np.savetxt(input_dir / "led2.txt", [[0.04, 0.0], [0.05, 0.0]])
    np.savetxt(input_dir / "mask.txt", [1, 1], fmt="%d")
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: prepared_arrays
  imu_timestamps: input/t_imu.txt
  imu_measurements: input/U_imu.txt
  camera_timestamps: input/t_cam.txt
  led1_positions: input/led1.txt
  led2_positions: input/led2.txt
  camera_mask: input/mask.txt
outputs:
  output_dir: run
""".lstrip()
    )

    session = load_session(load_session_config(config_path))
    write_session_diagnostics(session, tmp_path / "run")

    assert session.diagnostics["loader"]["format"] == "prepared_arrays"
    assert (tmp_path / "run" / "session_diagnostics.json").exists()


def test_arthur_loader_reports_missing_required_columns(tmp_path):
    """Malformed Arthur parquet inputs fail with actionable column names."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "time": [0.0, 0.01],
            "Headstage_GyroX": [0, 0],
            "Headstage_GyroY": [0, 0],
            "Headstage_GyroZ": [0, 1],
            "Headstage_AccelX": [0, 0],
            "Headstage_AccelY": [0, 0],
            "Headstage_AccelZ": [0, 0],
        }
    ).to_parquet(data_dir / "imu.parquet")
    pd.DataFrame(
        {
            "time": [0.0, 0.033],
            "xloc": [10.0, 11.0],
            "yloc": [20.0, 20.0],
            "xloc2": [14.0, 15.0],
        }
    ).to_parquet(data_dir / "position.parquet")
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
""".lstrip()
    )

    with pytest.raises(ValueError, match="yloc2"):
        load_session(load_session_config(config_path))

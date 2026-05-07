"""Tests for YAML session configuration loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trodestrack.config import SessionConfig, load_session_config


def test_load_session_config_resolves_paths_relative_to_yaml(tmp_path):
    """Relative input/output paths resolve from the YAML file location."""

    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
outputs:
  output_dir: runs/session-a
""".lstrip()
    )

    config = load_session_config(config_path)

    assert config.inputs.imu_file == tmp_path / "data" / "imu.parquet"
    assert config.inputs.position_file == tmp_path / "data" / "position.parquet"
    assert config.outputs.output_dir == tmp_path / "runs" / "session-a"


def test_prepared_arrays_config_requires_filter_ready_paths():
    """Prepared-array configs must include the existing per-file inputs."""

    with pytest.raises(ValidationError, match="missing required path"):
        SessionConfig.model_validate({"inputs": {"format": "prepared_arrays"}})


def test_spikegadgets_trodes_config_requires_arthur_parquets():
    """Arthur-style real-data configs require IMU and position parquet paths."""

    with pytest.raises(ValidationError, match="imu_file, position_file"):
        SessionConfig.model_validate({"inputs": {"format": "spikegadgets_trodes"}})


def test_session_config_rejects_typo_fields():
    """Strict YAML schemas should fail fast on misspelled sections."""

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SessionConfig.model_validate(
            {
                "inputs": {
                    "format": "prepared_arrays",
                    "imu_timestamps": "t_imu.txt",
                    "imu_measurements": "U_imu.txt",
                    "camera_timestamps": "t_cam.txt",
                    "led1_positions": "led1.txt",
                },
                "output": {"output_dir": "run"},
            }
        )


def test_vision_only_config_defaults_mahalanobis_gating_off():
    """Config-driven camera-only runs should keep tracking fast real motion."""

    config = SessionConfig.model_validate(
        {
            "inputs": {
                "format": "prepared_arrays",
                "imu_timestamps": "t_imu.txt",
                "imu_measurements": "U_imu.txt",
                "camera_timestamps": "t_cam.txt",
                "led1_positions": "led1.txt",
            },
            "filter": {"state_mode": "vision_only"},
        }
    )

    assert config.filter.to_ekf_kwargs()["use_mahalanobis_gating"] is False


def test_vision_only_config_respects_explicit_mahalanobis_gating():
    """Users can still opt into Mahalanobis gating in camera-only configs."""

    config = SessionConfig.model_validate(
        {
            "inputs": {
                "format": "prepared_arrays",
                "imu_timestamps": "t_imu.txt",
                "imu_measurements": "U_imu.txt",
                "camera_timestamps": "t_cam.txt",
                "led1_positions": "led1.txt",
            },
            "filter": {
                "state_mode": "vision_only",
                "use_mahalanobis_gating": True,
            },
        }
    )

    assert config.filter.to_ekf_kwargs()["use_mahalanobis_gating"] is True


def test_orientation_fused_config_passes_ekf_kwargs():
    """YAML configs should expose the 6-DOF orientation fused path."""

    config = SessionConfig.model_validate(
        {
            "inputs": {
                "format": "prepared_arrays",
                "imu_timestamps": "t_imu.txt",
                "imu_measurements": "U_imu.txt",
                "camera_timestamps": "t_cam.txt",
                "led1_positions": "led1.txt",
            },
            "filter": {
                "state_mode": "2d_cam_6dof_imu_orientation",
                "imu_gravity_body": [0.0, 0.0, 9.80665],
                "enable_experimental_accel_translation": False,
                "use_gravity_orientation_update": True,
                "gravity_orientation_measurement_noise": 0.0025,
            },
        }
    )

    kwargs = config.filter.to_ekf_kwargs()
    assert kwargs["state_mode"] == "2d_cam_6dof_imu_orientation"
    assert kwargs["imu_gravity_body"] == (0.0, 0.0, 9.80665)
    assert kwargs["enable_experimental_accel_translation"] is False
    assert kwargs["use_gravity_orientation_update"] is True
    assert kwargs["gravity_orientation_measurement_noise"] == 0.0025


def test_led_identity_initial_state_config():
    """Users can provide an initial LED identity prior for global swaps."""

    config = SessionConfig.model_validate(
        {
            "inputs": {
                "format": "prepared_arrays",
                "imu_timestamps": "t_imu.txt",
                "imu_measurements": "U_imu.txt",
                "camera_timestamps": "t_cam.txt",
                "led1_positions": "led1.txt",
            },
            "led_identity": {
                "mode": "auto",
                "initial_state": "swapped",
            },
        }
    )

    assert config.led_identity.initial_state == "swapped"

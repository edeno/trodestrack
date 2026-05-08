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
    assert config.filter.to_ekf_kwargs()["enable_zupt"] is False


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


def test_vision_only_config_respects_explicit_zupt():
    """Users can still opt into ZUPT in camera-only configs."""

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
                "enable_zupt": True,
            },
        }
    )

    assert config.filter.to_ekf_kwargs()["enable_zupt"] is True


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


def test_zupt_visual_context_hold_frames_passes_ekf_kwargs():
    """YAML configs expose bounded ZUPT visual-context carry."""

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
                "zupt_visual_context_hold_frames": 0,
            },
        }
    )

    assert config.filter.to_ekf_kwargs()["zupt_visual_context_hold_frames"] == 0


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


def test_led_identity_rejects_non_positive_speed():
    """LED identity speed scale must not allow divide-by-zero transition costs."""

    with pytest.raises(ValidationError, match="greater than 0"):
        SessionConfig.model_validate(
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
                    "max_speed_mps": 0.0,
                },
            }
        )


def test_imu_calibration_thresholds_are_configurable():
    """Real-data calibration gates should expose documented thresholds."""

    config = SessionConfig.model_validate(
        {
            "inputs": {
                "format": "spikegadgets_trodes",
                "imu_file": "imu.parquet",
                "position_file": "position.parquet",
            },
            "imu": {
                "calibration_min_yaw_correlation": 0.2,
                "calibration_max_horizontal_gravity_mps2": 0.75,
                "calibration_min_accel_axis_correlation_for_translation": 0.6,
            },
        }
    )

    assert config.imu.calibration_min_yaw_correlation == 0.2
    assert config.imu.calibration_max_horizontal_gravity_mps2 == 0.75
    assert config.imu.calibration_min_accel_axis_correlation_for_translation == 0.6


_PREPARED_INPUTS = {
    "format": "prepared_arrays",
    "imu_timestamps": "t_imu.txt",
    "imu_measurements": "U_imu.txt",
    "camera_timestamps": "t_cam.txt",
    "led1_positions": "led1.txt",
}


def test_imu_axis_map_must_have_six_canonical_keys():
    """``axis_map`` partial / extra keys raise at schema time, not deep in loader.

    The loader indexes ``axis_map`` with the six canonical names
    directly; a partial map (e.g. only ``gyro_z``) used to surface
    as ``Unexpected error: 'gyro_x'`` from a raw KeyError. The
    schema-level model_validator now rejects partial / extra maps
    with a clean Pydantic message.
    """

    with pytest.raises(ValidationError, match=r"axis_map.*missing keys"):
        SessionConfig.model_validate(
            {
                "inputs": _PREPARED_INPUTS,
                "imu": {"axis_map": {"gyro_z": "Headstage_GyroZ"}},
            }
        )

    with pytest.raises(ValidationError, match=r"axis_map.*unknown keys"):
        SessionConfig.model_validate(
            {
                "inputs": _PREPARED_INPUTS,
                "imu": {
                    "axis_map": {
                        "gyro_x": "GyroX",
                        "gyro_y": "GyroY",
                        "gyro_z": "GyroZ",
                        "accel_x": "AccelX",
                        "accel_y": "AccelY",
                        "accel_z": "AccelZ",
                        "extra_axis": "Extra",
                    }
                },
            }
        )


def test_imu_axis_signs_must_have_six_canonical_keys():
    """Partial ``axis_signs`` silently defaulted to +1 via .get; reject up front."""

    with pytest.raises(ValidationError, match=r"axis_signs.*missing keys"):
        SessionConfig.model_validate(
            {
                "inputs": _PREPARED_INPUTS,
                "imu": {"axis_signs": {"gyro_x": -1.0}},
            }
        )


def test_imu_axis_signs_values_must_be_sign_flips():
    """``axis_signs`` is a sign flip; non -1/+1 values silently rescale.

    Probe: ``gyro_z: 0.0`` zeroed out yaw rate,
    ``accel_z: 2.0`` doubled apparent gravity, ``gyro_x: -0.5``
    halved and flipped — all silently. ``axis_signs`` must restrict
    values to -1.0 / 1.0; for scale changes the user has
    ``gyro_scale_dps_per_lsb`` and ``accel_scale_g_per_lsb``.
    """

    full_signs = dict.fromkeys(
        ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"), 1.0
    )
    for axis, bad in [("gyro_z", 0.0), ("accel_z", 2.0), ("gyro_x", -0.5)]:
        signs = dict(full_signs)
        signs[axis] = bad
        with pytest.raises(ValidationError, match=r"axis_signs values must be"):
            SessionConfig.model_validate(
                {
                    "inputs": _PREPARED_INPUTS,
                    "imu": {"axis_signs": signs},
                }
            )

    # -1 / +1 still validate.
    SessionConfig.model_validate(
        {
            "inputs": _PREPARED_INPUTS,
            "imu": {"axis_signs": {**full_signs, "gyro_z": -1.0}},
        }
    )


def test_camera_partial_confidence_columns_rejected():
    """One confidence column without the other was silently ignored.

    ``_load_leds`` only builds ``conf_cam`` when both columns are
    configured. Probe: ``confidence_led1_column: led1_likelihood``
    alone left ``session.conf_cam = None`` so the user-specified
    column had no effect. The schema now requires both or neither.
    """

    for partial in (
        {"confidence_led1_column": "led1_likelihood"},
        {"confidence_led2_column": "led2_likelihood"},
    ):
        with pytest.raises(
            ValidationError, match=r"confidence_led1_column.*confidence_led2_column"
        ):
            SessionConfig.model_validate(
                {
                    "inputs": _PREPARED_INPUTS,
                    "camera": partial,
                }
            )

    # Both set — validates.
    SessionConfig.model_validate(
        {
            "inputs": _PREPARED_INPUTS,
            "camera": {
                "confidence_led1_column": "led1_likelihood",
                "confidence_led2_column": "led2_likelihood",
            },
        }
    )

    # Both None (default) — also validates.
    SessionConfig.model_validate({"inputs": _PREPARED_INPUTS})


def test_camera_meters_per_pixel_must_be_positive():
    """``meters_per_pixel <= 0`` collapses or mirrors trajectory; reject."""

    for bad in (0.0, -0.0022):
        with pytest.raises(ValidationError, match=r"meters_per_pixel"):
            SessionConfig.model_validate(
                {
                    "inputs": _PREPARED_INPUTS,
                    "camera": {"meters_per_pixel": bad},
                }
            )


def test_imu_sensor_scales_must_be_positive():
    """Negative gyro/accel scales silently mirror; reject at schema time."""

    for field, bad in [
        ("gyro_scale_dps_per_lsb", -0.061),
        ("accel_scale_g_per_lsb", -0.000061),
        ("gravity_mps2", -9.81),
        ("gyro_scale_dps_per_lsb", 0.0),
    ]:
        with pytest.raises(ValidationError, match=field):
            SessionConfig.model_validate(
                {
                    "inputs": _PREPARED_INPUTS,
                    "imu": {field: bad},
                }
            )


def test_outputs_safety_thresholds_must_be_positive():
    """Non-positive safety thresholds make the gate meaningless or unusable."""

    for field, bad in [
        ("safety_envelope_multiplier", 0.0),
        ("safety_envelope_multiplier", -1.0),
        ("safety_max_speed_mps", -1.0),
        ("safety_max_position_deviation_m", 0.0),
        ("safety_p95_position_deviation_m", -0.1),
        ("safety_extra_range_m", -0.5),  # only ge=0 is allowed
    ]:
        with pytest.raises(ValidationError, match=field):
            SessionConfig.model_validate(
                {
                    "inputs": _PREPARED_INPUTS,
                    "outputs": {field: bad},
                }
            )

    # ``safety_extra_range_m`` allows zero (extra-range disabled is
    # legitimate); confirm the boundary still validates.
    SessionConfig.model_validate(
        {"inputs": _PREPARED_INPUTS, "outputs": {"safety_extra_range_m": 0.0}}
    )

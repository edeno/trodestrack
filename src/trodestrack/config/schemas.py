"""Pydantic schemas for YAML-driven trodestrack sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputsConfig(BaseModel):
    """Input file locations and format selection."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["prepared_arrays", "spikegadgets_trodes"] = "prepared_arrays"

    # Existing prepared-array workflow.
    imu_timestamps: Path | None = None
    imu_measurements: Path | None = None
    camera_timestamps: Path | None = None
    led1_positions: Path | None = None
    led2_positions: Path | None = None
    camera_mask: Path | None = None

    # Real-data parquet workflow.
    imu_file: Path | None = None
    position_file: Path | None = None

    @model_validator(mode="after")
    def _validate_required_paths(self) -> InputsConfig:
        if self.format == "prepared_arrays":
            missing = [
                name
                for name in (
                    "imu_timestamps",
                    "imu_measurements",
                    "camera_timestamps",
                    "led1_positions",
                )
                if getattr(self, name) is None
            ]
        else:
            missing = [
                name
                for name in ("imu_file", "position_file")
                if getattr(self, name) is None
            ]
        if missing:
            raise ValueError(
                f"inputs.format={self.format!r} is missing required path(s): "
                f"{', '.join(missing)}."
            )
        return self


_CANONICAL_IMU_AXES = (
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
)


class IMUConfig(BaseModel):
    """IMU preprocessing controls for real-data sessions."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["2d", "3d"] = "3d"
    sample_hold_strategy: Literal["gyro_z_change", "any_axis_change", "none"] = (
        "gyro_z_change"
    )
    time_offset_s: float = 0.0

    # Sensor scales must be strictly positive. Negative scale would
    # silently mirror the trajectory while ``axis_signs`` still
    # appears to read +1; the documented way to flip an axis is
    # ``axis_signs``.
    gyro_scale_dps_per_lsb: float = Field(default=0.061, gt=0.0)
    accel_scale_g_per_lsb: float = Field(default=0.000061, gt=0.0)
    gravity_mps2: float = Field(default=9.80665, gt=0.0)

    axis_map: dict[str, str] = Field(
        default_factory=lambda: {
            "gyro_x": "Headstage_GyroX",
            "gyro_y": "Headstage_GyroY",
            "gyro_z": "Headstage_GyroZ",
            "accel_x": "Headstage_AccelX",
            "accel_y": "Headstage_AccelY",
            "accel_z": "Headstage_AccelZ",
        }
    )
    axis_signs: dict[str, float] = Field(
        default_factory=lambda: {
            "gyro_x": 1.0,
            "gyro_y": 1.0,
            "gyro_z": 1.0,
            "accel_x": 1.0,
            "accel_y": 1.0,
            "accel_z": 1.0,
        }
    )

    run_calibration: bool = True
    require_calibration_for_fusion: bool = True
    calibration_min_yaw_correlation: float = 0.1
    calibration_max_horizontal_gravity_mps2: float = 1.0
    calibration_min_accel_axis_correlation_for_translation: float = 0.5

    @model_validator(mode="after")
    def _validate_axis_keys(self) -> IMUConfig:
        # ``_convert_imu_to_si`` indexes ``axis_map`` with the
        # canonical six keys directly; a partial map (e.g. only
        # ``gyro_z``) used to fail deep in the loader with a raw
        # ``KeyError: 'gyro_x'`` and surface as ``Unexpected
        # error`` via the friendly_cli_errors decorator. Reject at
        # schema time so the user gets a clean Pydantic message.
        # ``axis_signs`` is queried with ``.get(name, 1.0)`` so a
        # partial map there silently defaults to +1 — also reject so
        # users are explicit about every axis.
        for field_name in ("axis_map", "axis_signs"):
            mapping = getattr(self, field_name)
            missing = [name for name in _CANONICAL_IMU_AXES if name not in mapping]
            extra = [name for name in mapping if name not in _CANONICAL_IMU_AXES]
            if missing or extra:
                detail = []
                if missing:
                    detail.append(f"missing keys: {', '.join(missing)}")
                if extra:
                    detail.append(f"unknown keys: {', '.join(extra)}")
                raise ValueError(
                    f"imu.{field_name} must contain exactly the six canonical "
                    f"axis names ({', '.join(_CANONICAL_IMU_AXES)}); "
                    f"got {' and '.join(detail)}."
                )

        # ``axis_signs`` values are multiplied directly into IMU
        # channels by ``_convert_imu_to_si``. The field name and
        # documented intent restrict it to a sign flip (-1 or +1);
        # arbitrary floats silently rescale or zero out a channel
        # (probe: ``gyro_z: 0.0`` deletes yaw rate, ``accel_z: 2.0``
        # doubles the apparent gravity). Use ``axis_map`` to point
        # at a different column or scale upstream — not ``axis_signs``.
        bad_signs = {
            name: value
            for name, value in self.axis_signs.items()
            if value not in (-1.0, 1.0)
        }
        if bad_signs:
            detail = ", ".join(f"{name}={value!r}" for name, value in bad_signs.items())
            raise ValueError(
                "imu.axis_signs values must be -1.0 or 1.0 (the field is a "
                "sign flip, not a scale factor); use imu.gyro_scale_dps_per_lsb "
                f"or imu.accel_scale_g_per_lsb for scale changes. Got {detail}."
            )

        return self


class CameraConfig(BaseModel):
    """Camera/LED preprocessing controls for real-data sessions."""

    model_config = ConfigDict(extra="forbid")

    # ``meters_per_pixel`` scales LED pixel coordinates into world
    # meters. Zero collapses every frame onto the origin and
    # negative values silently mirror the trajectory while
    # preserving positive LED spacing — both produce plausible-
    # looking but wrong fused output.
    meters_per_pixel: float = Field(default=0.0022, gt=0.0)
    time_offset_s: float = 0.0
    led1_x_column: str = "xloc"
    led1_y_column: str = "yloc"
    led2_x_column: str = "xloc2"
    led2_y_column: str = "yloc2"
    confidence_led1_column: str | None = None
    confidence_led2_column: str | None = None

    @model_validator(mode="after")
    def _validate_confidence_columns(self) -> CameraConfig:
        # ``_load_leds`` only builds ``conf_cam`` when both LED
        # confidence columns are configured; setting just one used
        # to silently drop the column with no warning, so a user
        # who set ``confidence_led1_column: led1_likelihood`` got
        # ``session.conf_cam = None`` and the EKF ran without any
        # confidence weighting. Require both or neither so partial
        # config fails loudly at schema time.
        cols = (self.confidence_led1_column, self.confidence_led2_column)
        if any(c is not None for c in cols) and not all(c is not None for c in cols):
            raise ValueError(
                "camera.confidence_led1_column and confidence_led2_column "
                "must both be set or both be None; partial confidence "
                "configuration is silently ignored by the loader. Got "
                f"led1={self.confidence_led1_column!r}, "
                f"led2={self.confidence_led2_column!r}."
            )
        return self


class FilterConfig(BaseModel):
    """Subset of EKFConfig exposed in YAML."""

    model_config = ConfigDict(extra="forbid")

    state_mode: Literal[
        "vision_only",
        "2d_full",
        "2d_cam_3d_imu",
        "2d_cam_6dof_imu_orientation",
    ] = "2d_cam_3d_imu"
    led_distance: float | None = None
    use_heading_measurement: bool | None = None
    process_noise_pos: float | None = None
    process_noise_vel: float | None = None
    process_noise_heading: float | None = None
    process_noise_gyro_bias: float | None = None
    process_noise_accel_bias: float | None = None
    measurement_noise_pos: float | None = None
    measurement_noise_heading: float | None = None
    imu_gyro_noise_density: float | None = None
    imu_accel_noise_density: float | None = None
    imu_gravity_body: tuple[float, float, float] | None = None
    damping_coeff: float | None = None
    use_mahalanobis_gating: bool | None = None
    enable_experimental_accel_translation: bool | None = None
    use_gravity_orientation_update: bool | None = None
    gravity_orientation_measurement_noise: float | None = None
    gravity_accel_magnitude_tolerance_m_s2: float | None = None
    gravity_gyro_norm_threshold_rad_s: float | None = None
    enable_zupt: bool | None = None
    zupt_velocity_threshold: float | None = None
    zupt_measurement_noise: float | None = None
    zupt_gyro_threshold_rad_s: float | None = None
    zupt_accel_threshold_m_s2: float | None = None
    zupt_camera_stationary_window_frames: int | None = None
    zupt_visual_context_hold_frames: int | None = None

    def to_ekf_kwargs(self, *, led_distance: float | None = None) -> dict[str, object]:
        """Return keyword arguments suitable for EKFConfig."""

        data = self.model_dump(exclude_none=True)
        if self.state_mode == "vision_only" and self.use_mahalanobis_gating is None:
            data["use_mahalanobis_gating"] = False
        data["led_distance"] = (
            self.led_distance if self.led_distance is not None else led_distance
        )
        return data


class OutputsConfig(BaseModel):
    """Output directory and config-run validation controls."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("trodestrack_run")
    write_diagnostics: bool = True
    run_safety_checks: bool = True
    # Safety thresholds gate physically implausible fused output.
    # Non-positive thresholds either accept anything (rendering the
    # check meaningless) or reject everything (blocking valid runs);
    # both indicate a config bug.
    safety_envelope_multiplier: float = Field(default=3.0, gt=0.0)
    safety_extra_range_m: float = Field(default=0.5, ge=0.0)
    safety_max_speed_mps: float = Field(default=3.0, gt=0.0)
    safety_max_position_deviation_m: float = Field(default=0.5, gt=0.0)
    safety_p95_position_deviation_m: float = Field(default=0.25, gt=0.0)
    # Minimum dual-LED frame count required to estimate the camera
    # midpoint envelope. Sessions with fewer dual-LED frames cannot
    # produce a meaningful camera-range bound for the
    # ``safety_envelope_multiplier`` gate; fail fast rather than
    # passing on a near-zero envelope.
    safety_min_dual_led_frames: int = Field(default=20, ge=1)


class LedIdentityConfig(BaseModel):
    """Persistent LED identity correction controls."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "auto"] = "none"
    initial_state: Literal["auto", "original", "swapped"] = "auto"
    transition_penalty: float = Field(default=2.0, ge=0.0)
    gyro_weight: float = Field(default=0.0, ge=0.0)
    max_speed_mps: float = Field(default=3.0, gt=0.0)


class SessionConfig(BaseModel):
    """Top-level YAML config for one trodestrack run."""

    model_config = ConfigDict(extra="forbid")

    inputs: InputsConfig
    imu: IMUConfig = Field(default_factory=IMUConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    led_identity: LedIdentityConfig = Field(default_factory=LedIdentityConfig)


def load_session_config(path: str | Path) -> SessionConfig:
    """Load a YAML session config, resolving relative paths from the config file."""

    config_path = Path(path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    config = SessionConfig.model_validate(raw)
    return _resolve_paths(config, base_dir=config_path.parent)


def _resolve_paths(config: SessionConfig, *, base_dir: Path) -> SessionConfig:
    def resolve(path: Path | None) -> Path | None:
        if path is None or path.is_absolute():
            return path
        return base_dir / path

    inputs = config.inputs.model_copy(
        update={
            name: resolve(getattr(config.inputs, name))
            for name in (
                "imu_timestamps",
                "imu_measurements",
                "camera_timestamps",
                "led1_positions",
                "led2_positions",
                "camera_mask",
                "imu_file",
                "position_file",
            )
        }
    )
    outputs = config.outputs.model_copy(
        update={"output_dir": resolve(config.outputs.output_dir)}
    )
    return config.model_copy(update={"inputs": inputs, "outputs": outputs})

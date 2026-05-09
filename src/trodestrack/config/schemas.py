"""Pydantic schemas for YAML-driven trodestrack sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrodesNativeConfig(BaseModel):
    """Native Trodes ``.videoPositionTracking`` + PTP timestamps inputs."""

    model_config = ConfigDict(extra="forbid")

    position_tracking_file: Path
    # PTP-synced per-frame timestamps. v1 supports PTP only; the loader
    # rejects ``cameraHWFrameCount`` and plain ``videoTimeStamps``
    # variants because clock-stitching them needs sample-rate logic
    # that's out of scope.
    camera_timestamps_file: Path


class DLCKeypointsConfig(BaseModel):
    """DeepLabCut HDF5 keypoint inputs."""

    model_config = ConfigDict(extra="forbid")

    h5_file: Path
    led1_bodypart: str
    led2_bodypart: str
    likelihood_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # ``meta_pickle`` synthesizes camera timestamps from the sibling
    # DLC ``_meta.pickle`` (``fps``, ``nframes``); ``trodes_hw_sync``
    # joins against an external ``*.videoTimeStamps.cameraHWSync``;
    # ``timestamp_file`` reads a 1-D float array.
    timestamps_source: Literal["meta_pickle", "trodes_hw_sync", "timestamp_file"] = (
        "meta_pickle"
    )
    camera_timestamps_file: Path | None = None
    timestamp_file: Path | None = None
    apply_crop_offset: bool = True

    @model_validator(mode="after")
    def _validate_timestamps_source(self) -> DLCKeypointsConfig:
        if self.timestamps_source == "trodes_hw_sync" and (
            self.camera_timestamps_file is None
        ):
            raise ValueError(
                "dlc_keypoints.timestamps_source='trodes_hw_sync' requires "
                "camera_timestamps_file."
            )
        if self.timestamps_source == "timestamp_file" and self.timestamp_file is None:
            raise ValueError(
                "dlc_keypoints.timestamps_source='timestamp_file' requires "
                "timestamp_file."
            )
        return self


class NWBLEDSourceConfig(BaseModel):
    """How to locate the LED position container inside an NWB file."""

    model_config = ConfigDict(extra="forbid")

    container: Literal["auto", "trodes_position", "ndx_pose"] = "auto"
    # For ``trodes_position``: SpatialSeries names under the Position
    # container. For ``ndx_pose``: bodypart names under PoseEstimation.
    # Loader-time auto-detection picks defaults when None.
    led1_series_name: str | None = None
    led2_series_name: str | None = None
    led1_bodypart: str | None = None
    led2_bodypart: str | None = None
    likelihood_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_paired_names(self) -> NWBLEDSourceConfig:
        # Series names and bodypart names are pairs: the loader needs
        # both halves to address the pair, and a half-set config used
        # to silently fall through to auto-detect on the missing side
        # (so ``led1_series_name="x"`` paired with the writer-default
        # LED2 quietly loaded the wrong second series).
        for name1, name2 in (
            ("led1_series_name", "led2_series_name"),
            ("led1_bodypart", "led2_bodypart"),
        ):
            v1, v2 = getattr(self, name1), getattr(self, name2)
            if (v1 is None) != (v2 is None):
                raise ValueError(
                    f"NWBLEDSourceConfig: {name1} and {name2} must both "
                    f"be set or both be None (got {name1}={v1!r}, "
                    f"{name2}={v2!r}); the loader resolves the LED "
                    "pair atomically and a half-set config used to "
                    "silently fall back to auto-detect on the missing "
                    "side."
                )
        return self


class NWBDIOToTTLConfig(BaseModel):
    """Map NWB ``behavioral_events`` TimeSeries names to TTL source ids."""

    model_config = ConfigDict(extra="forbid")

    # Edges come straight from the int8 0/1 stream — 1 → "rise",
    # 0 → "fall". The first sample is the initial level (not a
    # transition); the loader drops it.
    name_to_source_id: dict[str, int]


class NWBConfig(BaseModel):
    """NWB file + auto-detected position container + optional extras."""

    model_config = ConfigDict(extra="forbid")

    nwb_file: Path
    led_source: NWBLEDSourceConfig = Field(default_factory=NWBLEDSourceConfig)
    # Post-hoc re-calibration. Wins over file-stored ``conversion``
    # and over ``camera.meters_per_pixel`` per the
    # ``pixels_to_meters`` precedence ladder.
    meters_per_pixel_override: float | None = None
    # When set, the NWB ``processing["behavior"]["behavioral_events"]``
    # TimeSeries are assembled into the EKF/UKF event channel.
    # Phase 1 schema hook only; the loader lands in Phase 4c.
    dio_to_ttl: NWBDIOToTTLConfig | None = None


class InputsConfig(BaseModel):
    """Input file locations and format selection."""

    model_config = ConfigDict(extra="forbid")

    format: Literal[
        "prepared_arrays",
        "spikegadgets_trodes",
        "trodes_native",
        "dlc_keypoints",
        "nwb",
    ] = "prepared_arrays"

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

    # Native-loader workflows. Each format is selected via
    # ``format=...`` and configured by the matching nested block.
    trodes_native: TrodesNativeConfig | None = None
    dlc_keypoints: DLCKeypointsConfig | None = None
    nwb: NWBConfig | None = None

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
            if missing:
                raise ValueError(
                    f"inputs.format='prepared_arrays' is missing required "
                    f"path(s): {', '.join(missing)}."
                )
        elif self.format == "spikegadgets_trodes":
            missing = [
                name
                for name in ("imu_file", "position_file")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"inputs.format='spikegadgets_trodes' is missing "
                    f"required path(s): {', '.join(missing)}."
                )
        else:
            # Native loaders: ``format`` value matches the nested
            # config attribute name on InputsConfig.
            if getattr(self, self.format) is None:
                raise ValueError(
                    f"inputs.format={self.format!r} is missing the required "
                    f"'inputs.{self.format}' configuration block."
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


@dataclass(frozen=True)
class EventLocationSource:
    """Resolved geometry the EKF event-update model consumes per source.

    Every TTL event source (beam break, zone trigger, RFID reader) collapses to
    a 2D point measurement at ``anchor`` with anisotropic 2x2 covariance.
    Per-source-type distinctions (geometry math, default edge) live in the
    spec classes; the model is unaware of the original source type.
    """

    source_id: int
    anchor: np.ndarray  # (2,) world meters
    covariance: np.ndarray  # (2, 2) world-frame measurement covariance, PSD
    label: str | None = None
    source_type: str = "unknown"


def _isotropic_event_source(
    *,
    source_id: int,
    center: tuple[float, float],
    sigma: float,
    label: str | None,
    source_type: str,
) -> EventLocationSource:
    return EventLocationSource(
        source_id=source_id,
        anchor=np.asarray(center, dtype=float),
        covariance=(sigma**2) * np.eye(2, dtype=float),
        label=label,
        source_type=source_type,
    )


class BeamSpec(BaseModel):
    """A beam-break source.

    Computes anchor (midpoint of emitter/receiver) and an anisotropic
    covariance aligned with the beam: ``σ_perp`` perpendicular to the beam
    (default the IR beam-width scale) and ``σ_along = max(σ_perp, L/√12)``
    along the beam, where ``L`` is the emitter-receiver distance. Short beams
    collapse to an isotropic point fix; long beams become weakly along-beam
    constraints.
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    emitter: tuple[float, float]
    receiver: tuple[float, float]
    sigma_perp_m: float = Field(default=0.005, gt=0.0)
    active_edge: Literal["rise", "fall"] = "fall"
    label: str | None = None

    def to_event_source(self) -> EventLocationSource:
        emitter = np.asarray(self.emitter, dtype=float)
        receiver = np.asarray(self.receiver, dtype=float)
        anchor = 0.5 * (emitter + receiver)
        delta = receiver - emitter
        beam_length = float(np.linalg.norm(delta))

        sigma_along = max(self.sigma_perp_m, beam_length / np.sqrt(12.0))

        # Beam tangent (along) and normal (perp) unit vectors. Use a
        # safe default for the degenerate L=0 case (any orthonormal basis
        # is fine; the resulting R is isotropic).
        tangent = delta / beam_length if beam_length > 0.0 else np.array([1.0, 0.0])
        normal = np.array([-tangent[1], tangent[0]])

        # Rotation matrix mapping event-local axes (perp, along) to world.
        rot = np.column_stack([normal, tangent])
        diag = np.diag([self.sigma_perp_m**2, sigma_along**2])
        covariance = rot @ diag @ rot.T

        return EventLocationSource(
            source_id=self.id,
            anchor=anchor,
            covariance=covariance,
            label=self.label,
            source_type="beam",
        )


class ZoneTriggerSpec(BaseModel):
    """A point-trigger source (nose poke, lever press, gate)."""

    model_config = ConfigDict(extra="forbid")

    id: int
    center: tuple[float, float]
    sigma_m: float = Field(default=0.02, gt=0.0)
    active_edge: Literal["rise", "fall"] = "rise"
    label: str | None = None

    def to_event_source(self) -> EventLocationSource:
        return _isotropic_event_source(
            source_id=self.id,
            center=self.center,
            sigma=self.sigma_m,
            label=self.label,
            source_type="zone",
        )


class RFIDReaderSpec(BaseModel):
    """An RFID reader source.

    ``effective_radius_m`` is the detection range; treated as ``√2·σ`` of an
    isotropic 2D Gaussian fit to a uniform disc of that radius, so
    ``σ = r/√2`` and the covariance is ``(r²/2)·I``.
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    center: tuple[float, float]
    effective_radius_m: float = Field(default=0.05, gt=0.0)
    active_edge: Literal["rise", "fall"] = "rise"
    label: str | None = None

    def to_event_source(self) -> EventLocationSource:
        sigma = self.effective_radius_m / np.sqrt(2.0)
        return _isotropic_event_source(
            source_id=self.id,
            center=self.center,
            sigma=sigma,
            label=self.label,
            source_type="rfid",
        )


class TTLEventsConfig(BaseModel):
    """Configuration block for TTL event sensors (beam / zone / RFID).

    Sources of all three types share an events parquet whose rows are
    ``(time, source_id, edge)``. ``source_id`` must be unique across the
    configured beam, zone-trigger, and RFID-reader lists.

    ``events_file`` is optional and may be omitted when the events
    come from the NWB DIO bridge — see the SessionConfig validator
    (``_validate_dio_consistency``) for the conditional-required rule.
    """

    model_config = ConfigDict(extra="forbid")

    events_file: Path | None = None
    beams: list[BeamSpec] = Field(default_factory=list)
    zone_triggers: list[ZoneTriggerSpec] = Field(default_factory=list)
    rfid_readers: list[RFIDReaderSpec] = Field(default_factory=list)
    max_events_per_frame: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> TTLEventsConfig:
        seen: dict[int, str] = {}
        for kind, specs in (
            ("beams", self.beams),
            ("zone_triggers", self.zone_triggers),
            ("rfid_readers", self.rfid_readers),
        ):
            for spec in specs:
                if spec.id in seen:
                    raise ValueError(
                        "ttl_events source ids must be unique across beams, "
                        f"zone_triggers, and rfid_readers; id={spec.id} appears "
                        f"in both {seen[spec.id]} and {kind}."
                    )
                seen[spec.id] = kind
        return self


class SessionConfig(BaseModel):
    """Top-level YAML config for one trodestrack run."""

    model_config = ConfigDict(extra="forbid")

    inputs: InputsConfig
    imu: IMUConfig = Field(default_factory=IMUConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    led_identity: LedIdentityConfig = Field(default_factory=LedIdentityConfig)
    ttl_events: TTLEventsConfig | None = None

    @model_validator(mode="after")
    def _validate_dio_consistency(self) -> SessionConfig:
        """Enforce the NWB DIO → TTL events bridge schema contract.

        Phase 4c moves these from "deferred" to "required" — without
        them, a user could configure an NWB+DIO session with no
        ``events_file`` (loader-runnable) or a DIO bridge whose
        ``name_to_source_id`` referenced unknown geometry ids
        (loader-time KeyError far from the YAML).
        """

        nwb = self.inputs.nwb
        has_dio_bridge = (
            self.inputs.format == "nwb"
            and nwb is not None
            and nwb.dio_to_ttl is not None
        )

        if has_dio_bridge:
            # The geometry block is required because the EKF/UKF
            # event channel needs source positions, covariances, and
            # ids — DIO without ttl_events would be loader-runnable
            # but produce a fused trajectory with no event updates.
            if self.ttl_events is None:
                raise ValueError(
                    "inputs.nwb.dio_to_ttl is configured but ttl_events "
                    "is missing — the EKF/UKF event channel needs the "
                    "geometry block (beams/zone_triggers/rfid_readers) "
                    "to recover source positions and covariances. Add a "
                    "ttl_events block matching the DIO TimeSeries names."
                )
            assert nwb is not None and nwb.dio_to_ttl is not None
            known_ids = {
                spec.id
                for spec_list in (
                    self.ttl_events.beams,
                    self.ttl_events.zone_triggers,
                    self.ttl_events.rfid_readers,
                )
                for spec in spec_list
            }
            unknown = [
                (name, sid)
                for name, sid in nwb.dio_to_ttl.name_to_source_id.items()
                if sid not in known_ids
            ]
            if unknown:
                raise ValueError(
                    "inputs.nwb.dio_to_ttl.name_to_source_id values "
                    f"{unknown} reference source ids not configured in "
                    f"ttl_events (known ids: {sorted(known_ids)}). "
                    "Update the geometry block or the DIO mapping so "
                    "every TimeSeries name lands on a known source."
                )

        # ``events_file`` is required unless the DIO bridge is providing
        # the events from the NWB file itself.
        if self.ttl_events is not None and self.ttl_events.events_file is None:
            if not has_dio_bridge:
                raise ValueError(
                    "ttl_events.events_file is required unless "
                    "inputs.format='nwb' and inputs.nwb.dio_to_ttl is "
                    "set (the NWB DIO bridge is the only configured "
                    "alternative event source)."
                )

        return self


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

    flat_path_fields = (
        "imu_timestamps",
        "imu_measurements",
        "camera_timestamps",
        "led1_positions",
        "led2_positions",
        "camera_mask",
        "imu_file",
        "position_file",
    )
    inputs_update: dict[str, object] = {
        name: resolve(getattr(config.inputs, name)) for name in flat_path_fields
    }
    if config.inputs.trodes_native is not None:
        block = config.inputs.trodes_native
        resolved_pos = resolve(block.position_tracking_file)
        resolved_ts = resolve(block.camera_timestamps_file)
        assert resolved_pos is not None and resolved_ts is not None
        inputs_update["trodes_native"] = block.model_copy(
            update={
                "position_tracking_file": resolved_pos,
                "camera_timestamps_file": resolved_ts,
            }
        )
    if config.inputs.dlc_keypoints is not None:
        block = config.inputs.dlc_keypoints
        resolved_h5 = resolve(block.h5_file)
        assert resolved_h5 is not None
        inputs_update["dlc_keypoints"] = block.model_copy(
            update={
                "h5_file": resolved_h5,
                "camera_timestamps_file": resolve(block.camera_timestamps_file),
                "timestamp_file": resolve(block.timestamp_file),
            }
        )
    if config.inputs.nwb is not None:
        block = config.inputs.nwb
        resolved_nwb = resolve(block.nwb_file)
        assert resolved_nwb is not None
        inputs_update["nwb"] = block.model_copy(update={"nwb_file": resolved_nwb})
    inputs = config.inputs.model_copy(update=inputs_update)
    outputs = config.outputs.model_copy(
        update={"output_dir": resolve(config.outputs.output_dir)}
    )
    update_fields: dict[str, object] = {"inputs": inputs, "outputs": outputs}
    if config.ttl_events is not None:
        update_fields["ttl_events"] = config.ttl_events.model_copy(
            update={"events_file": resolve(config.ttl_events.events_file)}
        )
    return config.model_copy(update=update_fields)

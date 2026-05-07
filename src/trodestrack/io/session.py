"""Config-driven session loading, diagnostics, and safety checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from trodestrack.config.schemas import SessionConfig
from trodestrack.io.led_identity import resolve_led_identity
from trodestrack.models.ekf import EKFConfig, EKFResult, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
from trodestrack.qa.imu_calibration import (
    ImuCalibrationReport,
    format_imu_calibration_report,
    run_imu_calibration_diagnostics,
)

DEG_TO_RAD = np.pi / 180.0


@dataclass(frozen=True)
class PreparedSession:
    """Filter-ready session arrays and diagnostics."""

    t_imu: np.ndarray
    U_imu: np.ndarray
    t_cam: np.ndarray
    Z_cam_led1: np.ndarray
    Z_cam_led2: np.ndarray
    mask_cam: np.ndarray
    conf_cam: np.ndarray | None
    led_distance: float | None
    diagnostics: dict[str, object]
    config: SessionConfig
    gyro_z_for_led_identity: np.ndarray | None = None

    @property
    def source_format(self) -> str:
        """Configured input format for this prepared session."""

        return self.config.inputs.format


@dataclass(frozen=True)
class SafetyReport:
    """Real-data plausibility summary for fused vs vision-only outputs."""

    passed: bool
    message: str
    camera_range_m: tuple[float, float]
    fused_range_m: tuple[float, float]
    max_fused_speed_mps: float
    vision_loglik: float
    fused_loglik: float


def load_session(config: SessionConfig) -> PreparedSession:
    """Load a YAML-configured session into filter-ready arrays."""

    if config.inputs.format == "prepared_arrays":
        session = _load_prepared_arrays(config)
    else:
        session = _load_spikegadgets_trodes(config)

    if config.led_identity.mode == "auto":
        gyro_z = session.gyro_z_for_led_identity
        if gyro_z is None and session.U_imu.ndim == 2 and session.U_imu.shape[1] >= 3:
            gyro_z = session.U_imu[:, 0]
        corrected = resolve_led_identity(
            session.t_cam,
            session.Z_cam_led1,
            session.Z_cam_led2,
            session.mask_cam,
            led_distance=session.led_distance,
            config=config.led_identity,
            t_imu=session.t_imu,
            gyro_z=gyro_z,
        )
        diagnostics = dict(session.diagnostics)
        diagnostics["led_identity"] = corrected.diagnostics
        diagnostics["led_identity_swapped"] = corrected.swapped
        return PreparedSession(
            t_imu=session.t_imu,
            U_imu=session.U_imu,
            t_cam=session.t_cam,
            Z_cam_led1=corrected.led1,
            Z_cam_led2=corrected.led2,
            mask_cam=session.mask_cam,
            conf_cam=session.conf_cam,
            led_distance=session.led_distance,
            diagnostics=diagnostics,
            config=config,
            gyro_z_for_led_identity=session.gyro_z_for_led_identity,
        )
    return session


def write_session_diagnostics(
    session: PreparedSession,
    output_dir: Path,
    safety_report: SafetyReport | None = None,
) -> None:
    """Write loader/calibration/LED-identity diagnostics to the run directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_raw = dict(session.diagnostics)
    swapped = diagnostics_raw.pop("led_identity_swapped", None)
    if safety_report is not None:
        diagnostics_raw["safety_check"] = safety_report
    diagnostics = _json_ready(diagnostics_raw)
    with (output_dir / "session_diagnostics.json").open("w") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)

    calibration = session.diagnostics.get("imu_calibration")
    if isinstance(calibration, ImuCalibrationReport):
        (output_dir / "imu_calibration_report.txt").write_text(
            format_imu_calibration_report(calibration) + "\n"
        )
        with (output_dir / "imu_calibration_report.json").open("w") as f:
            json.dump(_json_ready(calibration), f, indent=2, sort_keys=True)

    led_diag = session.diagnostics.get("led_identity")
    if isinstance(led_diag, dict):
        with (output_dir / "led_identity_diagnostics.json").open("w") as f:
            json.dump(_json_ready(led_diag), f, indent=2, sort_keys=True)
    if swapped is not None:
        np.save(
            output_dir / "led_identity_swapped.npy", np.asarray(swapped, dtype=bool)
        )


def run_real_data_safety_check(
    session: PreparedSession,
    ekf_config: EKFConfig,
    filter_result: EKFResult,
) -> SafetyReport:
    """Compare fused real-data output against the camera/vision-only envelope."""

    config = session.config.outputs
    if (
        session.source_format != "spikegadgets_trodes"
        or not config.run_safety_checks
        or ekf_config.state_mode == "vision_only"
    ):
        marginal_loglik = float(np.asarray(filter_result.marginal_loglik))
        return SafetyReport(
            passed=True,
            message="disabled",
            camera_range_m=(0.0, 0.0),
            fused_range_m=(0.0, 0.0),
            max_fused_speed_mps=0.0,
            vision_loglik=float("nan"),
            fused_loglik=marginal_loglik,
        )

    layout = get_layout(ekf_config.state_mode)
    means = np.asarray(filter_result.filtered_means)
    pos = means[:, list(layout.pos_idx)]
    cam_mid = 0.5 * (session.Z_cam_led1 + session.Z_cam_led2)
    valid = session.mask_cam & np.isfinite(cam_mid).all(axis=1)
    if not np.any(valid):
        raise ValueError("Safety check requires at least one finite dual-LED frame.")
    camera_range = tuple(float(x) for x in np.ptp(cam_mid[valid], axis=0))
    fused_range = tuple(float(x) for x in np.ptp(pos, axis=0))

    vision_config = replace(
        ekf_config, state_mode="vision_only", use_mahalanobis_gating=False
    )
    dummy_imu = np.zeros((session.t_imu.shape[0], 3))
    vision = extended_kalman_filter(
        vision_config,
        session.t_imu,
        dummy_imu,
        session.t_cam,
        session.Z_cam_led1,
        session.Z_cam_led2,
        session.mask_cam,
        conf_cam=session.conf_cam,
    )

    dt = np.diff(session.t_cam)
    vel = np.diff(pos, axis=0) / np.maximum(dt[:, None], 1e-6)
    max_speed = float(np.nanmax(np.linalg.norm(vel, axis=1))) if vel.size else 0.0

    allowed = np.asarray(camera_range) * config.safety_envelope_multiplier
    allowed = allowed + config.safety_extra_range_m
    passed = bool(
        np.all(np.asarray(fused_range) <= allowed)
        and max_speed <= config.safety_max_speed_mps
    )
    message = "passed"
    if not passed:
        message = (
            "IMU-fused trajectory is physically implausible relative to the "
            "camera midpoint envelope. Use state_mode: vision_only or fix IMU "
            "scale/axis/time-offset calibration before trusting fused output."
        )
    return SafetyReport(
        passed=passed,
        message=message,
        camera_range_m=camera_range,
        fused_range_m=fused_range,
        max_fused_speed_mps=max_speed,
        vision_loglik=float(np.asarray(vision.marginal_loglik)),
        fused_loglik=float(np.asarray(filter_result.marginal_loglik)),
    )


def _load_prepared_arrays(config: SessionConfig) -> PreparedSession:
    inputs = config.inputs
    assert inputs.imu_timestamps is not None
    assert inputs.imu_measurements is not None
    assert inputs.camera_timestamps is not None
    assert inputs.led1_positions is not None
    t_imu = np.loadtxt(inputs.imu_timestamps)
    U_imu = np.loadtxt(inputs.imu_measurements)
    t_cam = np.loadtxt(inputs.camera_timestamps)
    led1 = np.loadtxt(inputs.led1_positions)
    led2 = (
        np.loadtxt(inputs.led2_positions)
        if inputs.led2_positions is not None
        else np.full((len(t_cam), 2), np.nan)
    )
    mask = (
        np.loadtxt(inputs.camera_mask).astype(bool)
        if inputs.camera_mask is not None
        else np.isfinite(led1).all(axis=1)
    )
    led_distance = config.filter.led_distance or _median_led_distance(led1, led2, mask)
    _validate_time_vector(t_imu, "IMU timestamps")
    _validate_time_vector(t_cam, "camera timestamps")
    U_arr = np.asarray(U_imu, dtype=float)
    return PreparedSession(
        t_imu=np.asarray(t_imu, dtype=float),
        U_imu=U_arr,
        t_cam=np.asarray(t_cam, dtype=float),
        Z_cam_led1=np.asarray(led1, dtype=float),
        Z_cam_led2=np.asarray(led2, dtype=float),
        mask_cam=np.asarray(mask, dtype=bool),
        conf_cam=None,
        led_distance=led_distance,
        diagnostics={"loader": {"format": "prepared_arrays"}},
        config=config,
        gyro_z_for_led_identity=_gyro_z_for_led_identity(U_arr),
    )


def _load_spikegadgets_trodes(config: SessionConfig) -> PreparedSession:
    inputs = config.inputs
    assert inputs.imu_file is not None
    assert inputs.position_file is not None
    pos_df = pd.read_parquet(inputs.position_file)
    imu_df = pd.read_parquet(inputs.imu_file)
    _require_columns(imu_df, config.imu.axis_map.values(), source=str(inputs.imu_file))
    _require_columns(
        pos_df,
        (
            config.camera.led1_x_column,
            config.camera.led1_y_column,
            config.camera.led2_x_column,
            config.camera.led2_y_column,
        ),
        source=str(inputs.position_file),
    )

    imu_unique = _remove_sample_hold(imu_df, config)
    t_imu_unix = _index_or_time_column(imu_unique)
    t_cam_unix = _index_or_time_column(pos_df)
    t_start = min(float(t_imu_unix[0]), float(t_cam_unix[0]))
    t_imu = t_imu_unix - t_start + config.imu.time_offset_s
    t_cam = t_cam_unix - t_start + config.camera.time_offset_s
    _validate_time_vector(t_imu, "IMU timestamps after preprocessing")
    _validate_time_vector(t_cam, "camera timestamps")

    U_full = _convert_imu_to_si(imu_unique, config)
    U_filter = _project_imu_for_filter(U_full, config.filter.state_mode)
    led1, led2, conf_cam = _load_leds(pos_df, config)
    mask = np.isfinite(led1).all(axis=1) & np.isfinite(led2).all(axis=1)
    led_distance = config.filter.led_distance or _median_led_distance(led1, led2, mask)

    diagnostics: dict[str, object] = {
        "loader": {
            "format": "spikegadgets_trodes",
            "imu_raw_samples": len(imu_df),
            "imu_unique_samples": len(imu_unique),
            "camera_frames": len(pos_df),
            "sample_hold_strategy": config.imu.sample_hold_strategy,
            "effective_imu_rate_hz": float(1.0 / np.median(np.diff(t_imu))),
            "camera_rate_hz": float(1.0 / np.median(np.diff(t_cam))),
            "led_distance_m": float(led_distance),
        }
    }

    if config.imu.run_calibration and U_full.shape[1] == 6:
        try:
            report = run_imu_calibration_diagnostics(
                t_imu=t_imu,
                gyro_z=U_full[:, 2],
                accel_xyz=U_full[:, 3:6],
                t_cam=t_cam,
                led1=led1,
                led2=led2,
            )
            diagnostics["imu_calibration"] = report
            _validate_calibration_for_fusion(report, config)
        except ValueError as e:
            diagnostics["imu_calibration_error"] = str(e)

    return PreparedSession(
        t_imu=t_imu,
        U_imu=U_filter,
        t_cam=t_cam,
        Z_cam_led1=led1,
        Z_cam_led2=led2,
        mask_cam=mask,
        conf_cam=conf_cam,
        led_distance=led_distance,
        diagnostics=diagnostics,
        config=config,
        gyro_z_for_led_identity=U_full[:, 2],
    )


def _remove_sample_hold(imu_df: pd.DataFrame, config: SessionConfig) -> pd.DataFrame:
    strategy = config.imu.sample_hold_strategy
    if strategy == "none":
        return imu_df.copy()
    if strategy == "gyro_z_change":
        col = config.imu.axis_map["gyro_z"]
        values = imu_df[col].to_numpy()
        keep = np.concatenate([[True], np.diff(values) != 0])
    else:
        cols = [config.imu.axis_map[name] for name in _axis_names()]
        values = imu_df[cols].to_numpy()
        keep = np.concatenate([[True], np.any(np.diff(values, axis=0) != 0, axis=1)])
    return imu_df.loc[keep].copy()


def _require_columns(df: pd.DataFrame, columns: object, *, source: str) -> None:
    required = tuple(dict.fromkeys(str(col) for col in columns))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {', '.join(missing)}."
        )


def _validate_time_vector(t: np.ndarray, name: str) -> None:
    arr = np.asarray(t, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}.")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing.")


def _convert_imu_to_si(imu_df: pd.DataFrame, config: SessionConfig) -> np.ndarray:
    cols = config.imu.axis_map
    signs = config.imu.axis_signs
    gyro_scale = config.imu.gyro_scale_dps_per_lsb * DEG_TO_RAD
    accel_scale = config.imu.accel_scale_g_per_lsb * config.imu.gravity_mps2
    values = {
        "gyro_x": signs.get("gyro_x", 1.0)
        * imu_df[cols["gyro_x"]].to_numpy()
        * gyro_scale,
        "gyro_y": signs.get("gyro_y", 1.0)
        * imu_df[cols["gyro_y"]].to_numpy()
        * gyro_scale,
        "gyro_z": signs.get("gyro_z", 1.0)
        * imu_df[cols["gyro_z"]].to_numpy()
        * gyro_scale,
        "accel_x": signs.get("accel_x", 1.0)
        * imu_df[cols["accel_x"]].to_numpy()
        * accel_scale,
        "accel_y": signs.get("accel_y", 1.0)
        * imu_df[cols["accel_y"]].to_numpy()
        * accel_scale,
        "accel_z": signs.get("accel_z", 1.0)
        * imu_df[cols["accel_z"]].to_numpy()
        * accel_scale,
    }
    if config.imu.mode == "2d":
        return np.column_stack([values["gyro_z"], values["accel_x"], values["accel_y"]])
    return np.column_stack(
        [
            values["gyro_x"],
            values["gyro_y"],
            values["gyro_z"],
            values["accel_x"],
            values["accel_y"],
            values["accel_z"],
        ]
    )


def _project_imu_for_filter(U_full: np.ndarray, state_mode: str) -> np.ndarray:
    if state_mode == "vision_only":
        return np.zeros((U_full.shape[0], 3))
    if U_full.shape[1] == 3:
        if state_mode == "2d_cam_6dof_imu_orientation":
            raise ValueError(
                "state_mode='2d_cam_6dof_imu_orientation' requires 6-channel "
                "IMU input [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]; "
                "set imu.mode: 3d or provide prepared 6-channel arrays."
            )
        if state_mode == "2d_cam_3d_imu":
            return U_full
        return U_full
    if U_full.shape[1] != 6:
        raise ValueError(f"Unsupported IMU shape {U_full.shape}.")
    if state_mode == "2d_cam_6dof_imu_orientation":
        return U_full
    if state_mode == "2d_cam_3d_imu":
        return U_full[:, [2, 3, 4, 5]]
    return U_full[:, [2, 3, 4]]


def _gyro_z_for_led_identity(U_imu: np.ndarray) -> np.ndarray | None:
    if U_imu.ndim != 2:
        return None
    if U_imu.shape[1] == 6:
        return U_imu[:, 2]
    if U_imu.shape[1] >= 3:
        return U_imu[:, 0]
    return None


def _load_leds(
    pos_df: pd.DataFrame, config: SessionConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    cam = config.camera
    led1 = (
        pos_df[[cam.led1_x_column, cam.led1_y_column]].to_numpy(dtype=float)
        * cam.meters_per_pixel
    )
    led2 = (
        pos_df[[cam.led2_x_column, cam.led2_y_column]].to_numpy(dtype=float)
        * cam.meters_per_pixel
    )
    conf_cam = None
    if cam.confidence_led1_column and cam.confidence_led2_column:
        c1 = pos_df[cam.confidence_led1_column].to_numpy(dtype=float)
        c2 = pos_df[cam.confidence_led2_column].to_numpy(dtype=float)
        conf_cam = np.column_stack([c1, c1, c2, c2])
    return led1, led2, conf_cam


def _index_or_time_column(df: pd.DataFrame) -> np.ndarray:
    if "time" in df.columns:
        return df["time"].to_numpy(dtype=float)
    return df.index.to_numpy(dtype=float)


def _median_led_distance(led1: np.ndarray, led2: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(led1).all(axis=1) & np.isfinite(led2).all(axis=1)
    if not np.any(valid):
        return 0.04
    return float(np.nanmedian(np.linalg.norm(led2[valid] - led1[valid], axis=1)))


def _validate_calibration_for_fusion(
    report: ImuCalibrationReport, config: SessionConfig
) -> None:
    if not _uses_imu(config.filter.state_mode):
        return
    if report.stationary_samples < 10:
        raise ValueError(
            "not enough stationary samples for IMU calibration diagnostics "
            f"({report.stationary_samples}); use state_mode: vision_only or "
            "provide a longer session with low-motion periods."
        )
    if abs(report.yaw_rate_fit.correlation) < 0.1:
        raise ValueError(
            "gyro_z does not correlate with LED-derived yaw rate "
            f"(correlation={report.yaw_rate_fit.correlation:.3f}); check axis "
            "mapping, sign, time offset, or use state_mode: vision_only."
        )


def _uses_imu(state_mode: str) -> bool:
    return state_mode in {
        "2d_full",
        "2d_cam_3d_imu",
        "2d_cam_6dof_imu_orientation",
    }


def _axis_names() -> tuple[str, ...]:
    return ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z")


def _json_ready(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        return _json_ready(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(x) for x in obj]
    return obj

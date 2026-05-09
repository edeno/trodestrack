"""Parquet IMU loader shared across format readers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trodestrack.config.schemas import IMUConfig, SessionConfig

DEG_TO_RAD = np.pi / 180.0


def project_imu_for_filter(U_full: np.ndarray, state_mode: str) -> np.ndarray:
    """Public alias for the state-mode IMU projection.

    Native loaders that get U_full from a non-parquet source (NWB
    analog group, future raw-rec readers) still need the same
    projection the parquet path applies after ``convert_imu_columns_to_si``.
    """

    return _project_imu_for_filter(U_full, state_mode)


def require_columns(df: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    """Public alias for the missing-column raise used across loaders."""

    _require_columns(df, columns, source=source)


def index_or_time_column(df: pd.DataFrame) -> np.ndarray:
    """Public alias for the parquet ``time`` / index extractor."""

    return _index_or_time_column(df)


def convert_imu_columns_to_si(
    raw_columns: dict[str, np.ndarray], imu_cfg: IMUConfig
) -> np.ndarray:
    """Apply axis-sign + gyro/accel scale to per-axis raw IMU columns.

    Source-agnostic: callers supply ``raw_columns`` keyed by canonical
    axis names (``gyro_x``, ``gyro_y``, ``gyro_z``, ``accel_x``,
    ``accel_y``, ``accel_z``) and get back the same SI-converted
    ``U_full`` the parquet path produces. Used by both
    ``_convert_imu_to_si`` (parquet column lookup) and the NWB
    ``from_analog_container`` (channel-id column lookup).

    Output shape is ``(n, 3)`` for ``imu_cfg.mode == "2d"``
    (``[gyro_z, accel_x, accel_y]``) and ``(n, 6)`` for ``"3d"``
    (``[gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]``).
    """

    signs = imu_cfg.axis_signs
    gyro_scale = imu_cfg.gyro_scale_dps_per_lsb * DEG_TO_RAD
    accel_scale = imu_cfg.accel_scale_g_per_lsb * imu_cfg.gravity_mps2

    def converted(axis: str) -> np.ndarray:
        scale = gyro_scale if axis.startswith("gyro") else accel_scale
        return signs.get(axis, 1.0) * raw_columns[axis].astype(float) * scale

    values = {axis: converted(axis) for axis in _axis_names()}
    if imu_cfg.mode == "2d":
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


@dataclass(frozen=True)
class ParquetIMU:
    """Result of reading an IMU parquet file and applying preprocessing.

    ``t_imu_unix`` is the post-deduplication timestamp in the source
    file's clock; the caller is responsible for clock alignment
    (subtracting a ``t_start`` and adding ``imu.time_offset_s``) and
    for ``_validate_time_vector``-style checks.
    """

    t_imu_unix: np.ndarray
    U_filter: np.ndarray
    U_full: np.ndarray
    raw_samples: int
    unique_samples: int


def load_imu_parquet(imu_file: Path, config: SessionConfig) -> ParquetIMU:
    """Read an IMU parquet, deduplicate sample-and-hold, project for filter.

    Encapsulates the
    ``read_parquet`` → ``_require_columns`` → ``_remove_sample_hold`` →
    ``_convert_imu_to_si`` → ``_project_imu_for_filter`` chain that
    ``_load_spikegadgets_trodes`` used to inline at
    ``session.py:288-321`` and that the new native loaders also need
    when ``inputs.imu_file`` is configured.
    """

    imu_df = pd.read_parquet(imu_file)
    _require_columns(imu_df, config.imu.axis_map.values(), source=str(imu_file))
    imu_unique = _remove_sample_hold(imu_df, config)
    t_imu_unix = _index_or_time_column(imu_unique)
    U_full = _convert_imu_to_si(imu_unique, config)
    U_filter = _project_imu_for_filter(U_full, config.filter.state_mode)
    return ParquetIMU(
        t_imu_unix=t_imu_unix,
        U_filter=U_filter,
        U_full=U_full,
        raw_samples=len(imu_df),
        unique_samples=len(imu_unique),
    )


def _require_columns(df: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    required = tuple(dict.fromkeys(str(col) for col in columns))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {', '.join(missing)}."
        )


def _index_or_time_column(df: pd.DataFrame) -> np.ndarray:
    if "time" in df.columns:
        return df["time"].to_numpy(dtype=float)
    return df.index.to_numpy(dtype=float)


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


def _convert_imu_to_si(imu_df: pd.DataFrame, config: SessionConfig) -> np.ndarray:
    cols = config.imu.axis_map
    raw = {axis: imu_df[cols[axis]].to_numpy() for axis in _axis_names()}
    return convert_imu_columns_to_si(raw, config.imu)


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


def _axis_names() -> tuple[str, ...]:
    return ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z")

"""Unit tests for the parquet IMU loader extracted helper."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from trodestrack.config.schemas import (
    FilterConfig,
    IMUConfig,
    InputsConfig,
    SessionConfig,
)
from trodestrack.io.imu_parquet import load_imu_parquet

DEG_TO_RAD = np.pi / 180.0


def _make_imu_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [100.0, 100.005, 100.010, 100.015, 100.020],
            "Headstage_GyroX": [1, 1, 1, 1, 1],
            "Headstage_GyroY": [2, 2, 2, 2, 2],
            "Headstage_GyroZ": [0, 0, 10, 10, 20],
            "Headstage_AccelX": [100, 100, 110, 110, 120],
            "Headstage_AccelY": [200, 200, 210, 210, 220],
            "Headstage_AccelZ": [300, 300, 310, 310, 320],
        }
    )


StateMode = Literal[
    "vision_only",
    "2d_full",
    "2d_cam_3d_imu",
    "2d_cam_6dof_imu_orientation",
]
SampleHoldStrategy = Literal["gyro_z_change", "any_axis_change", "none"]
IMUMode = Literal["2d", "3d"]


def _make_session_config(
    tmp_path: Path,
    imu_df: pd.DataFrame,
    *,
    state_mode: StateMode = "2d_cam_3d_imu",
    sample_hold_strategy: SampleHoldStrategy = "gyro_z_change",
    imu_mode: IMUMode = "3d",
) -> SessionConfig:
    imu_path = tmp_path / "imu.parquet"
    pos_path = tmp_path / "position.parquet"
    imu_df.to_parquet(imu_path)
    # ``inputs.position_file`` is required by the schema for the
    # ``spikegadgets_trodes`` format; the loader under test never reads
    # it, but we still need to satisfy the validator.
    pd.DataFrame(
        {
            "time": [100.0, 100.033],
            "xloc": [10.0, 11.0],
            "yloc": [20.0, 20.0],
            "xloc2": [14.0, 15.0],
            "yloc2": [20.0, 20.0],
        }
    ).to_parquet(pos_path)
    return SessionConfig(
        inputs=InputsConfig(
            format="spikegadgets_trodes",
            imu_file=imu_path,
            position_file=pos_path,
        ),
        imu=IMUConfig(mode=imu_mode, sample_hold_strategy=sample_hold_strategy),
        filter=FilterConfig(state_mode=state_mode),
    )


def _imu_path(config: SessionConfig) -> Path:
    """Narrow ``inputs.imu_file`` to ``Path`` (the test config always
    sets it; the schema's ``Path | None`` is for the format-dispatch
    branches)."""

    assert config.inputs.imu_file is not None
    return config.inputs.imu_file


def test_dedup_via_gyro_z_change(tmp_path) -> None:
    """``gyro_z_change`` strategy keeps only rows where gyro_z changes
    (plus the first sample)."""

    imu_df = _make_imu_dataframe()
    config = _make_session_config(tmp_path, imu_df)

    imu = load_imu_parquet(_imu_path(config), config)

    # gyro_z = [0, 0, 10, 10, 20] → keep indices 0, 2, 4 (first plus
    # changes). Raw=5, unique=3.
    assert imu.raw_samples == 5
    assert imu.unique_samples == 3
    np.testing.assert_array_equal(imu.t_imu_unix, [100.0, 100.010, 100.020])


def test_si_conversion_matches_inline_formula(tmp_path) -> None:
    """The 6-channel U_full is built from gyro * deg_to_rad * scale and
    accel * scale * gravity."""

    imu_df = _make_imu_dataframe()
    config = _make_session_config(tmp_path, imu_df)
    gyro_scale = config.imu.gyro_scale_dps_per_lsb * DEG_TO_RAD
    accel_scale = config.imu.accel_scale_g_per_lsb * config.imu.gravity_mps2

    imu = load_imu_parquet(_imu_path(config), config)

    # Post-dedup rows (gyro_z_change): indices 0, 2, 4 → values
    # [1,2,0,100,200,300], [1,2,10,110,210,310], [1,2,20,120,220,320].
    expected_full = np.array(
        [
            [1, 2, 0, 100, 200, 300],
            [1, 2, 10, 110, 210, 310],
            [1, 2, 20, 120, 220, 320],
        ],
        dtype=float,
    )
    expected_full[:, :3] *= gyro_scale
    expected_full[:, 3:] *= accel_scale
    np.testing.assert_allclose(imu.U_full, expected_full)


def test_state_mode_2d_cam_3d_imu_projection(tmp_path) -> None:
    """For 6-channel U_full and ``state_mode='2d_cam_3d_imu'`` the
    projection picks columns ``[2, 3, 4, 5]`` (gyro_z + accel_xyz)."""

    imu_df = _make_imu_dataframe()
    config = _make_session_config(tmp_path, imu_df, state_mode="2d_cam_3d_imu")

    imu = load_imu_parquet(_imu_path(config), config)

    np.testing.assert_array_equal(imu.U_filter, imu.U_full[:, [2, 3, 4, 5]])


def test_state_mode_2d_cam_6dof_projection(tmp_path) -> None:
    """``state_mode='2d_cam_6dof_imu_orientation'`` keeps the full 6-column
    matrix."""

    imu_df = _make_imu_dataframe()
    config = _make_session_config(
        tmp_path, imu_df, state_mode="2d_cam_6dof_imu_orientation"
    )

    imu = load_imu_parquet(_imu_path(config), config)

    np.testing.assert_array_equal(imu.U_filter, imu.U_full)


def test_state_mode_vision_only_zeros_imu(tmp_path) -> None:
    """``state_mode='vision_only'`` returns a zero IMU stream sized to
    match the deduplicated samples."""

    imu_df = _make_imu_dataframe()
    config = _make_session_config(tmp_path, imu_df, state_mode="vision_only")

    imu = load_imu_parquet(_imu_path(config), config)

    np.testing.assert_array_equal(imu.U_filter, np.zeros((imu.unique_samples, 3)))


def test_missing_imu_column_raises(tmp_path) -> None:
    """Missing an axis-mapped column raises a clear error naming the
    file and the missing column(s)."""

    imu_df = _make_imu_dataframe().drop(columns=["Headstage_GyroZ"])
    config = _make_session_config(tmp_path, imu_df)

    with pytest.raises(ValueError, match="Headstage_GyroZ"):
        load_imu_parquet(_imu_path(config), config)

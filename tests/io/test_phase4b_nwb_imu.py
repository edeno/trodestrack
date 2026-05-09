"""Phase 4b — NWB analog IMU validation slice.

Fixtures extend the Phase 4a Trodes-style NWB by attaching the
``processing/analog/analog/analog`` TimeSeries the
``trodes_to_nwb.convert_analog`` writer produces (channel ids in the
description string, triple-space separator).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

# Skip the module when ``[nwb]`` (pynwb) isn't installed.
pytest.importorskip("pynwb")

import pynwb
from pynwb.behavior import BehavioralEvents, Position

from trodestrack.config import (
    CameraConfig,
    FilterConfig,
    IMUConfig,
    InputsConfig,
    NWBConfig,
    SessionConfig,
)
from trodestrack.io import load_session
from trodestrack.io.nwb import from_analog_container, load_nwb_session

NS_PER_S = 1_000_000_000


# ---------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------


def _make_nwb_with_position_and_analog(
    tmp_path: Path,
    *,
    n_frames: int = 30,
    include_analog: bool = True,
    n_imu_samples: int = 60,
    base_systime_s: float = 0.0,
    extra_ecu_channels: int = 4,
) -> Path:
    """Build a Phase-4a-style NWB plus an optional analog TimeSeries.

    Channel order in the analog TimeSeries: a few ECU channels
    followed by the six headstage IMU channels in the order
    ``Headstage_GyroX/Y/Z, Headstage_AccelX/Y/Z``.
    """

    nwbfile = pynwb.NWBFile(
        session_description="Phase 4b test session",
        identifier=str(uuid4()),
        session_start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        session_id="session_0",
    )

    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    timestamps = np.arange(n_frames, dtype=float) / 30.0
    led1_data = np.column_stack([100.0 + np.arange(n_frames), np.full(n_frames, 200.0)])
    led2_data = np.column_stack([110.0 + np.arange(n_frames), np.full(n_frames, 210.0)])
    position = Position()
    position.create_spatial_series(
        name="led_0_series_0",
        description="LED1",
        data=led1_data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
    )
    position.create_spatial_series(
        name="led_1_series_0",
        description="LED2",
        data=led2_data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
    )
    behavior.add(position)

    if include_analog:
        analog_module = nwbfile.create_processing_module(
            name="analog", description="Contains all analog data"
        )
        ecu_channels = [f"ECU_Ain{i + 1}" for i in range(extra_ecu_channels)]
        imu_channels = [
            "Headstage_GyroX",
            "Headstage_GyroY",
            "Headstage_GyroZ",
            "Headstage_AccelX",
            "Headstage_AccelY",
            "Headstage_AccelZ",
        ]
        all_channels = ecu_channels + imu_channels
        # Description: triple-space separator with trailing "   "
        # (matches trodes_to_nwb.convert_analog.__merge_row_description).
        description = "   ".join(all_channels) + "   "

        # Build distinguishable per-channel signals so axis-map
        # validation has bite. ECU channels = 0; IMU channels = small
        # ramps with channel-specific offsets.
        analog_t = base_systime_s + np.arange(n_imu_samples) / 1500.0
        data = np.zeros((n_imu_samples, len(all_channels)), dtype=np.int16)
        for i in range(len(imu_channels)):
            offset = (i + 1) * 100
            data[:, len(ecu_channels) + i] = (np.arange(n_imu_samples) + offset).astype(
                np.int16
            )

        analog_events = BehavioralEvents(name="analog")
        analog_events.add_timeseries(
            pynwb.TimeSeries(
                name="analog",
                description=description,
                data=data,
                timestamps=analog_t,
                unit="-1",
            )
        )
        analog_module.add(analog_events)

    path = tmp_path / "session.nwb"
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    return path


# ---------------------------------------------------------------------
# Container API: from_analog_container.
# ---------------------------------------------------------------------


def test_from_analog_container_matches_load_nwb_session(tmp_path: Path) -> None:
    """``from_analog_container`` returns the same ``(t_imu, U_full)``
    as ``load_nwb_session`` populates in ``extras.imu`` for the same
    file."""

    nwb_path = _make_nwb_with_position_and_analog(tmp_path)
    imu_cfg = IMUConfig(run_calibration=False)

    _, extras = load_nwb_session(NWBConfig(nwb_file=nwb_path), imu_cfg=imu_cfg)
    assert extras.imu is not None
    via_path_t, via_path_u = extras.imu

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        analog_ts = (
            nwbfile.processing["analog"].data_interfaces["analog"].time_series["analog"]
        )
        via_container_t, via_container_u = from_analog_container(analog_ts, imu_cfg)

    np.testing.assert_array_equal(via_path_t, via_container_t)
    np.testing.assert_array_equal(via_path_u, via_container_u)


def test_from_analog_container_eager_after_io_close(tmp_path: Path) -> None:
    """After ``from_analog_container`` returns, the underlying
    NWBHDF5IO can be closed and the arrays remain readable."""

    nwb_path = _make_nwb_with_position_and_analog(tmp_path)
    imu_cfg = IMUConfig(run_calibration=False)

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        analog_ts = (
            nwbfile.processing["analog"].data_interfaces["analog"].time_series["analog"]
        )
        t_imu, U_full = from_analog_container(analog_ts, imu_cfg)
    # IO is closed — the arrays must still respond.
    assert t_imu.size > 0
    assert U_full.shape[0] == t_imu.size


# ---------------------------------------------------------------------
# Channel-id matching + SI conversion parity with parquet path.
# ---------------------------------------------------------------------


def test_si_conversion_matches_parquet_path(tmp_path: Path) -> None:
    """An NWB analog TimeSeries with the same raw counts as a parquet
    IMU produces an identical ``U_full`` after SI conversion."""

    from trodestrack.io.imu_parquet import load_imu_parquet

    nwb_path = _make_nwb_with_position_and_analog(tmp_path)
    imu_cfg = IMUConfig(run_calibration=False)

    # NWB-analog path.
    _, extras = load_nwb_session(NWBConfig(nwb_file=nwb_path), imu_cfg=imu_cfg)
    assert extras.imu is not None
    nwb_t, nwb_u = extras.imu

    # Build a parquet with the exact same raw IMU samples and read
    # via ``load_imu_parquet`` — the SI output must be bit-equal.
    n = nwb_t.size
    df = pd.DataFrame(
        {
            "time": nwb_t,
            "Headstage_GyroX": (np.arange(n) + 100).astype(np.int16),
            "Headstage_GyroY": (np.arange(n) + 200).astype(np.int16),
            "Headstage_GyroZ": (np.arange(n) + 300).astype(np.int16),
            "Headstage_AccelX": (np.arange(n) + 400).astype(np.int16),
            "Headstage_AccelY": (np.arange(n) + 500).astype(np.int16),
            "Headstage_AccelZ": (np.arange(n) + 600).astype(np.int16),
        }
    )
    parquet_path = tmp_path / "imu.parquet"
    df.to_parquet(parquet_path)
    parquet_config = SessionConfig(
        inputs=InputsConfig(
            format="spikegadgets_trodes",
            imu_file=parquet_path,
            position_file=parquet_path,  # not read; satisfies validator
        ),
        imu=IMUConfig(run_calibration=False, sample_hold_strategy="none"),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="2d_cam_3d_imu"),
    )
    imu_data = load_imu_parquet(parquet_path, parquet_config)

    np.testing.assert_array_equal(nwb_u, imu_data.U_full)


def test_axis_map_missing_channel_raises(tmp_path: Path) -> None:
    """If ``imu_cfg.axis_map`` references a name not present in the
    description-encoded channel ids, the loader raises with the
    available list."""

    nwb_path = _make_nwb_with_position_and_analog(tmp_path)
    bad_imu = IMUConfig(
        run_calibration=False,
        axis_map={
            "gyro_x": "Headstage_GyroX",
            "gyro_y": "Headstage_GyroY",
            "gyro_z": "Headstage_GyroZ",
            "accel_x": "Headstage_AccelX",
            "accel_y": "Headstage_AccelY",
            "accel_z": "DOES_NOT_EXIST",
        },
    )

    with pytest.raises(ValueError, match=r"axis_map"):
        load_nwb_session(NWBConfig(nwb_file=nwb_path), imu_cfg=bad_imu)


# ---------------------------------------------------------------------
# IMU source resolution: precedence and absence-handling.
# ---------------------------------------------------------------------


def test_load_session_uses_nwb_analog_when_present(tmp_path: Path) -> None:
    """Without ``inputs.imu_file``, the NWB analog group is the IMU
    source; ``state_mode='2d_cam_3d_imu'`` runs cleanly and the
    diagnostic surfaces ``imu_source='nwb_analog'``."""

    n_frames = 30
    nwb_path = _make_nwb_with_position_and_analog(
        tmp_path, n_frames=n_frames, n_imu_samples=80
    )
    config = SessionConfig(
        inputs=InputsConfig(format="nwb", nwb=NWBConfig(nwb_file=nwb_path)),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="2d_cam_3d_imu"),
    )

    session = load_session(config)

    assert session.diagnostics["loader"]["imu_source"] == "nwb_analog"
    # With state_mode=2d_cam_3d_imu, U_filter is (n, 4) projection of
    # the 6-channel U_full — gyro_z + accel_xyz.
    assert session.U_imu.ndim == 2
    assert session.U_imu.shape[1] == 4


def test_parquet_overrides_nwb_analog(tmp_path: Path) -> None:
    """``inputs.imu_file`` (parquet) wins over the NWB analog group,
    and the diagnostic reports ``imu_source='parquet'``."""

    nwb_path = _make_nwb_with_position_and_analog(tmp_path)
    n = 80
    df = pd.DataFrame(
        {
            "time": np.linspace(-0.05, 1.05, n),
            "Headstage_GyroX": np.zeros(n, dtype=int),
            "Headstage_GyroY": np.zeros(n, dtype=int),
            "Headstage_GyroZ": np.arange(n, dtype=int),
            "Headstage_AccelX": np.zeros(n, dtype=int),
            "Headstage_AccelY": np.zeros(n, dtype=int),
            "Headstage_AccelZ": np.zeros(n, dtype=int),
        }
    )
    parquet_path = tmp_path / "imu_override.parquet"
    df.to_parquet(parquet_path)
    config = SessionConfig(
        inputs=InputsConfig(
            format="nwb",
            nwb=NWBConfig(nwb_file=nwb_path),
            imu_file=parquet_path,
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="2d_cam_3d_imu"),
    )

    session = load_session(config)

    assert session.diagnostics["loader"]["imu_source"] == "parquet"


def test_nwb_no_analog_imu_consuming_state_mode_raises(tmp_path: Path) -> None:
    """An NWB without ``processing["analog"]`` and an IMU-consuming
    ``state_mode`` raises with the three-option remediation message
    (same surface as Trodes-native / DLC)."""

    nwb_path = _make_nwb_with_position_and_analog(tmp_path, include_analog=False)
    config = SessionConfig(
        inputs=InputsConfig(format="nwb", nwb=NWBConfig(nwb_file=nwb_path)),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="2d_cam_3d_imu"),
    )

    with pytest.raises(ValueError) as excinfo:
        load_session(config)
    msg = str(excinfo.value)
    assert "inputs.imu_file" in msg
    assert "vision_only" in msg
    assert "nwb" in msg.lower()


def test_nwb_no_analog_vision_only_succeeds(tmp_path: Path) -> None:
    """An NWB without ``processing["analog"]`` and ``vision_only``
    falls through to the synthetic zero-IMU stream."""

    nwb_path = _make_nwb_with_position_and_analog(tmp_path, include_analog=False)
    config = SessionConfig(
        inputs=InputsConfig(format="nwb", nwb=NWBConfig(nwb_file=nwb_path)),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="vision_only"),
    )

    session = load_session(config)

    assert session.diagnostics["loader"]["imu_source"] == "synthetic"
    np.testing.assert_array_equal(session.U_imu, 0.0)

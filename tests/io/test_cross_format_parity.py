"""cross-format integration test.

A single ground-truth pixel-space LED trajectory is written into all
three native formats (Trodes binaries, DLC HDF5, NWB Position).
Loading each via ``load_session`` and running the EKF must produce
filtered means within 1e-3 m across all pairs — confirming the
loaders genuinely converge on the same underlying camera samples.

State mode is ``vision_only`` so the IMU contribution is the
synthetic zero stream regardless of format and any divergence
reflects a loader-side mismatch, not IMU-fusion drift.
"""

from __future__ import annotations

import datetime
import pickle
import struct
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

# All three formats need their respective extras for fixture writing.
pytest.importorskip("tables")  # for DLC
pytest.importorskip("pynwb")  # for NWB

import pynwb
from pynwb.behavior import Position

from trodestrack.config import (
    CameraConfig,
    DLCKeypointsConfig,
    FilterConfig,
    IMUConfig,
    InputsConfig,
    NWBConfig,
    OutputsConfig,
    SessionConfig,
    TrodesNativeConfig,
    load_session_config,
)
from trodestrack.io import load_session
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

NS_PER_S = 1_000_000_000
HW_NS_INTERVAL = int(NS_PER_S / 30)


# ---------------------------------------------------------------------
# Shared ground-truth.
# ---------------------------------------------------------------------


def _ground_truth_pixels(n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic camera trajectory in pixel space.

    Returns ``(led1_pixels, led2_pixels, hw_timestamps_ns)`` where
    timestamps include a 1-second PTP pause near the start (to
    exercise the trodes_native pause-trim path).
    """

    # Integer-valued positions: the Trodes binary stores LED coords
    # as uint16 (xloc / yloc / xloc2 / yloc2), so any float steps in
    # the ground-truth would truncate on the trodes_native side and
    # diverge from the DLC HDF5 / NWB SpatialSeries float arrays.
    led1 = np.column_stack(
        [100 + np.arange(n_frames), 200 + np.arange(n_frames)]
    ).astype(float)
    led2 = np.column_stack(
        [110 + np.arange(n_frames), 210 + np.arange(n_frames)]
    ).astype(float)
    base_ns = 1_700_000_000_000_000_000
    hw_timestamps = np.empty(n_frames, dtype=np.int64)
    n_pre_pause = 2
    for i in range(n_frames):
        if i < n_pre_pause:
            hw_timestamps[i] = base_ns + i * HW_NS_INTERVAL
        elif i == n_pre_pause:
            hw_timestamps[i] = hw_timestamps[i - 1] + NS_PER_S  # 1s pause
        else:
            hw_timestamps[i] = hw_timestamps[i - 1] + HW_NS_INTERVAL
    return led1, led2, hw_timestamps


def _post_pause_slice(
    led1: np.ndarray, led2: np.ndarray, hw_ns: np.ndarray, n_pre_pause: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop the pre-pause frames the trodes_native loader will trim."""

    return (
        led1[n_pre_pause:],
        led2[n_pre_pause:],
        hw_ns[n_pre_pause:],
    )


# ---------------------------------------------------------------------
# Format adapters.
# ---------------------------------------------------------------------


def _write_trodes_binary(path: Path, fields_spec: str, records: bytes) -> None:
    blob = (f"<Start settings>\nFields: {fields_spec}\n<End settings>\n").encode(
        "ascii"
    )
    path.write_bytes(blob + records)


def _write_trodes_native(
    tmp_path: Path,
    led1: np.ndarray,
    led2: np.ndarray,
    hw_ns: np.ndarray,
    n_pre_pause: int = 2,
) -> tuple[Path, Path]:
    """Write the Trodes ``.videoPositionTracking`` +
    ``.videoTimeStamps.cameraHWSync`` pair. Returns ``(pos_path,
    ts_path)`` with the trodes_native fixture layout."""

    pos_path = tmp_path / "session.videoPositionTracking"
    ts_path = tmp_path / "session.videoTimeStamps.cameraHWSync"
    sample_step = 1000
    pos_timestamps = [1000 + i * sample_step for i in range(len(led1))]

    pos_records = b"".join(
        struct.pack(
            "<IHHHH",
            pos_timestamps[i],
            int(led1[i, 0]),
            int(led1[i, 1]),
            int(led2[i, 0]),
            int(led2[i, 1]),
        )
        for i in range(len(led1))
    )
    _write_trodes_binary(
        pos_path,
        "<time uint32><xloc uint16><yloc uint16><xloc2 uint16><yloc2 uint16>",
        pos_records,
    )

    ts_records = b"".join(
        struct.pack("<IHQ", pos_timestamps[i], i, int(hw_ns[i]))
        for i in range(len(led1))
    )
    _write_trodes_binary(
        ts_path,
        "<PosTimestamp uint32><frameCount uint16><HWTimestamp uint64>",
        ts_records,
    )
    return pos_path, ts_path


def _write_dlc_keypoints(
    tmp_path: Path,
    led1_post_pause: np.ndarray,
    led2_post_pause: np.ndarray,
    hw_ns_post_pause: np.ndarray,
) -> Path:
    """Write a DLC HDF5 + sibling _meta.pickle for the post-pause
    frames. ``timestamps_source='timestamp_file'`` is paired with an
    absolute-seconds txt file so the camera clock matches the other
    formats' Unix-like base.
    """

    scorer = "DLC_resnet50_test"
    n = led1_post_pause.shape[0]
    coords = ["x", "y", "likelihood"]
    bodyparts = ["led_green", "led_red"]
    cols = pd.MultiIndex.from_product(
        [[scorer], bodyparts, coords], names=["scorer", "bodyparts", "coords"]
    )
    df = pd.DataFrame(np.zeros((n, 6)), columns=cols)
    df.loc[:, (scorer, "led_green", "x")] = led1_post_pause[:, 0].astype(float)
    df.loc[:, (scorer, "led_green", "y")] = led1_post_pause[:, 1].astype(float)
    df.loc[:, (scorer, "led_green", "likelihood")] = 0.99
    df.loc[:, (scorer, "led_red", "x")] = led2_post_pause[:, 0].astype(float)
    df.loc[:, (scorer, "led_red", "y")] = led2_post_pause[:, 1].astype(float)
    df.loc[:, (scorer, "led_red", "likelihood")] = 0.99

    h5_path = tmp_path / f"video{scorer}.h5"
    df.to_hdf(h5_path, key="df_with_missing", format="table", mode="w")

    meta_path = h5_path.with_name(h5_path.stem + "_meta.pickle")
    with meta_path.open("wb") as f:
        pickle.dump(
            {
                "data": {
                    "Scorer": scorer,
                    "fps": 30.0,
                    "nframes": n,
                    "frame_dimensions": (640, 480),
                    "cropping_parameters": [0, 640, 0, 480],
                    "pytorch-config": {"metadata": {"individuals": ["single"]}},
                }
            },
            f,
        )

    timestamp_path = tmp_path / "dlc_timestamps.txt"
    np.savetxt(
        timestamp_path,
        np.asarray(hw_ns_post_pause, dtype=np.int64) / NS_PER_S,
    )
    return h5_path


def _write_dlc_timestamps(tmp_path: Path, hw_ns_post_pause: np.ndarray) -> Path:
    timestamp_path = tmp_path / "dlc_timestamps.txt"
    np.savetxt(
        timestamp_path,
        np.asarray(hw_ns_post_pause, dtype=np.int64) / NS_PER_S,
    )
    return timestamp_path


def _write_nwb_position(
    tmp_path: Path,
    led1_post_pause: np.ndarray,
    led2_post_pause: np.ndarray,
    hw_ns_post_pause: np.ndarray,
) -> Path:
    """Write an NWB Position container for the post-pause frames."""

    nwbfile = pynwb.NWBFile(
        session_description="cross-format parity",
        identifier=str(uuid4()),
        session_start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        session_id="parity",
    )
    timestamps = np.asarray(hw_ns_post_pause, dtype=np.int64) / NS_PER_S
    position = Position()
    position.create_spatial_series(
        name="led_0_series_0",
        description="LED1",
        data=led1_post_pause.astype(float),
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
    )
    position.create_spatial_series(
        name="led_1_series_0",
        description="LED2",
        data=led2_post_pause.astype(float),
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
    )
    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    behavior.add(position)

    path = tmp_path / "session.nwb"
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    return path


# ---------------------------------------------------------------------
# Cross-format parity test.
# ---------------------------------------------------------------------


def _build_session(config: SessionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run ``load_session`` + EKF and return ``(t_cam, filtered_means)``."""

    session = load_session(config)
    ekf_config = EKFConfig(state_mode="vision_only")
    result = extended_kalman_filter(
        ekf_config,
        session.t_imu,
        session.U_imu,
        session.t_cam,
        session.Z_cam_led1,
        session.Z_cam_led2,
        session.mask_cam,
        conf_cam=session.conf_cam,
    )
    return session.t_cam, np.asarray(result.filtered_means)


def _write_spikegadgets_parquet(
    tmp_path: Path,
    led1_post_pause: np.ndarray,
    led2_post_pause: np.ndarray,
    hw_ns_post_pause: np.ndarray,
) -> tuple[Path, Path]:
    """Write the parquet pair the existing ``spikegadgets_trodes``
    loader consumes (one IMU parquet, one position parquet) for the
    post-pause frames. The IMU parquet is a no-op stream — gyro_z is
    monotonic so ``sample_hold_strategy='gyro_z_change'`` keeps every
    row but the EKF runs ``vision_only`` so values don't matter."""

    timestamps_s = np.asarray(hw_ns_post_pause, dtype=np.int64) / NS_PER_S

    pos_parquet = tmp_path / "position.parquet"
    pd.DataFrame(
        {
            "time": timestamps_s,
            "xloc": led1_post_pause[:, 0].astype(np.int32),
            "yloc": led1_post_pause[:, 1].astype(np.int32),
            "xloc2": led2_post_pause[:, 0].astype(np.int32),
            "yloc2": led2_post_pause[:, 1].astype(np.int32),
        }
    ).to_parquet(pos_parquet)

    # IMU parquet aligned 1:1 with the camera samples — the native
    # loaders' synthetic-IMU path (vision_only) builds ``t_imu`` by
    # copying ``t_cam``, so matching that here makes the EKF
    # prediction-step structure identical across all four formats.
    # gyro_z is monotone so ``gyro_z_change`` dedup keeps every row.
    imu_t = timestamps_s.copy()
    imu_parquet = tmp_path / "imu.parquet"
    pd.DataFrame(
        {
            "time": imu_t,
            "Headstage_GyroX": np.zeros(imu_t.size, dtype=int),
            "Headstage_GyroY": np.zeros(imu_t.size, dtype=int),
            "Headstage_GyroZ": np.arange(imu_t.size, dtype=int),
            "Headstage_AccelX": np.zeros(imu_t.size, dtype=int),
            "Headstage_AccelY": np.zeros(imu_t.size, dtype=int),
            "Headstage_AccelZ": np.zeros(imu_t.size, dtype=int),
        }
    ).to_parquet(imu_parquet)
    return pos_parquet, imu_parquet


def test_filtered_means_within_1e_3_across_formats(tmp_path: Path) -> None:
    """Same camera trajectory loaded via the existing
    ``spikegadgets_trodes`` parquet path AND the three native loaders
    (Trodes binaries, DLC HDF5, NWB Position) yields filtered means
    within 1e-3 m across every pair (the cross-format parity gate). The
    parquet baseline confirms the new loaders remain
    parity-compatible with the established workflow."""

    n_frames = 30
    led1_full, led2_full, hw_ns_full = _ground_truth_pixels(n_frames)
    led1_post, led2_post, hw_ns_post = _post_pause_slice(
        led1_full, led2_full, hw_ns_full
    )

    pos_path, ts_path = _write_trodes_native(tmp_path, led1_full, led2_full, hw_ns_full)
    h5_path = _write_dlc_keypoints(tmp_path, led1_post, led2_post, hw_ns_post)
    dlc_ts_path = _write_dlc_timestamps(tmp_path, hw_ns_post)
    nwb_path = _write_nwb_position(tmp_path, led1_post, led2_post, hw_ns_post)
    pos_parquet, imu_parquet = _write_spikegadgets_parquet(
        tmp_path, led1_post, led2_post, hw_ns_post
    )

    common_kwargs = dict(
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="vision_only"),
        outputs=OutputsConfig(run_safety_checks=False),
    )

    config_parquet = SessionConfig(
        inputs=InputsConfig(
            format="spikegadgets_trodes",
            imu_file=imu_parquet,
            position_file=pos_parquet,
        ),
        **common_kwargs,
    )
    config_native = SessionConfig(
        inputs=InputsConfig(
            format="trodes_native",
            trodes_native=TrodesNativeConfig(
                position_tracking_file=pos_path,
                camera_timestamps_file=ts_path,
            ),
        ),
        **common_kwargs,
    )
    config_dlc = SessionConfig(
        inputs=InputsConfig(
            format="dlc_keypoints",
            dlc_keypoints=DLCKeypointsConfig(
                h5_file=h5_path,
                led1_bodypart="led_green",
                led2_bodypart="led_red",
                timestamps_source="timestamp_file",
                timestamp_file=dlc_ts_path,
                apply_crop_offset=False,
            ),
        ),
        **common_kwargs,
    )
    config_nwb = SessionConfig(
        inputs=InputsConfig(format="nwb", nwb=NWBConfig(nwb_file=nwb_path)),
        **common_kwargs,
    )

    t_cam_parquet, means_parquet = _build_session(config_parquet)
    t_cam_native, means_native = _build_session(config_native)
    t_cam_dlc, means_dlc = _build_session(config_dlc)
    t_cam_nwb, means_nwb = _build_session(config_nwb)

    # The four formats should land on the same camera clock after
    # alignment (each path subtracts its own ``t_start``, all
    # equal here since the data shares a Unix-like systime).
    np.testing.assert_allclose(t_cam_parquet, t_cam_native, atol=1e-12)
    np.testing.assert_allclose(t_cam_parquet, t_cam_dlc, atol=1e-12)
    np.testing.assert_allclose(t_cam_parquet, t_cam_nwb, atol=1e-12)

    # The plan's gate: filtered means within 1e-3 m across every
    # pair of (parquet, trodes_native, dlc_keypoints, nwb).
    np.testing.assert_allclose(means_parquet, means_native, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(means_parquet, means_dlc, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(means_parquet, means_nwb, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(means_native, means_dlc, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(means_native, means_nwb, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(means_dlc, means_nwb, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------
# Examples valid: each example YAML parses cleanly via load_session_config.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_name",
    [
        "session_trodes_native.yaml",
        "session_dlc_keypoints.yaml",
        "session_nwb.yaml",
    ],
)
def test_example_yaml_parses_to_session_config(yaml_name: str) -> None:
    """Each example YAML in ``examples/`` parses cleanly into a
    ``SessionConfig`` via ``load_session_config`` — schema valid,
    paths resolved relative to the YAML.

    The example YAMLs are template configs (matching the convention
    of the existing ``examples/session_spikegadgets_trodes.yaml``);
    paths point at user-supplied placeholders so the examples don't
    ship with bundled data fixtures. End-to-end load + EKF for
    every format is already exercised by
    ``test_filtered_means_within_1e_3_across_formats`` against an
    in-process ground truth; this test is a schema-validity gate to
    catch drift / YAML breakage in the example templates themselves.
    """

    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    yaml_path = examples_dir / yaml_name
    assert yaml_path.exists(), f"missing example fixture: {yaml_path}"

    config = load_session_config(yaml_path)
    assert config.inputs.format in {
        "trodes_native",
        "dlc_keypoints",
        "nwb",
    }

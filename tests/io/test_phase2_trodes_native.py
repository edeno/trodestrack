"""Phase 2 — ``trodes_native`` loader validation slice.

Fixtures are synthesized in-process via the same byte layout the
vendored Trodes parser reads, so no checked-in binaries are needed.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trodestrack.config import (
    CameraConfig,
    FilterConfig,
    IMUConfig,
    InputsConfig,
    SessionConfig,
    TrodesNativeConfig,
)
from trodestrack.io import load_session
from trodestrack.io.loaders._trodes_native import load_trodes_native_position

NS_PER_S = 1_000_000_000
DEFAULT_HW_NS_INTERVAL = int(NS_PER_S / 30)  # ~30 fps


# ----------------------------------------------------------------------
# Fixture builders. Trodes binaries are
# ``<Start settings>\n<key>: <value>\n...<End settings>\n`` followed by
# a packed record array matching the ``Fields:`` declaration.
# ----------------------------------------------------------------------


def _write_trodes_binary(
    path: Path, header: dict[str, str], fields_spec: str, records: bytes
) -> None:
    """Write a Trodes binary with the given header + records.

    ``fields_spec`` is the verbatim ``Fields:`` value (e.g.
    ``"<PosTimestamp uint32><frameCount uint16><HWTimestamp uint64>"``);
    ``records`` is the already-packed record-array bytes.
    """

    lines = ["<Start settings>"]
    for key, value in header.items():
        lines.append(f"{key}: {value}")
    lines.append(f"Fields: {fields_spec}")
    lines.append("<End settings>")
    blob = ("\n".join(lines) + "\n").encode("ascii")
    path.write_bytes(blob + records)


def _pack_ptp_timestamps(
    pos_timestamps: list[int], frame_counts: list[int], hw_timestamps_ns: list[int]
) -> bytes:
    """Pack ``<PosTimestamp uint32><frameCount uint16><HWTimestamp uint64>`` rows."""

    parts: list[bytes] = []
    for ps, fc, hw in zip(pos_timestamps, frame_counts, hw_timestamps_ns, strict=True):
        parts.append(struct.pack("<IHQ", ps, fc, hw))
    return b"".join(parts)


def _pack_position_tracking(
    sample_times: list[int],
    led1: list[tuple[int, int]],
    led2: list[tuple[int, int]],
) -> bytes:
    """Pack ``<time uint32><xloc uint16><yloc uint16><xloc2 uint16><yloc2 uint16>`` rows."""

    parts: list[bytes] = []
    for t, (x1, y1), (x2, y2) in zip(sample_times, led1, led2, strict=True):
        parts.append(struct.pack("<IHHHH", t, x1, y1, x2, y2))
    return b"".join(parts)


def _make_ptp_timestamps_file(
    tmp_path: Path,
    *,
    n_pre_pause: int = 5,
    n_post_pause: int = 8,
    sample_step: int = 1000,
    suffix: str = ".videoTimeStamps.cameraHWSync",
    fields_override: str | None = None,
    records_override: bytes | None = None,
    base_hw_ns: int = 1_700_000_000_000_000_000,  # 2023-11-14 (post-2000 sanity)
) -> tuple[Path, dict[str, Any]]:
    """Synthesize a PTP timestamps binary with a 1-second pause near the start.

    Returns ``(path, info)`` where ``info`` includes the Trodes
    sample-count clock and the post-pause indices, so the position
    tracking fixture can be aligned against it.
    """

    path = tmp_path / f"session{suffix}"
    n = n_pre_pause + n_post_pause
    pos_timestamps = [1000 + i * sample_step for i in range(n)]
    frame_counts = list(range(n))
    hw_step_ns = DEFAULT_HW_NS_INTERVAL
    hw_timestamps = []
    for i in range(n):
        if i < n_pre_pause:
            hw_timestamps.append(base_hw_ns + i * hw_step_ns)
        elif i == n_pre_pause:
            # 1-second pause separates pre- from post-pause samples.
            hw_timestamps.append(hw_timestamps[-1] + NS_PER_S)
        else:
            hw_timestamps.append(hw_timestamps[-1] + hw_step_ns)
    fields_spec = (
        fields_override
        if fields_override is not None
        else "<PosTimestamp uint32><frameCount uint16><HWTimestamp uint64>"
    )
    records = (
        records_override
        if records_override is not None
        else _pack_ptp_timestamps(pos_timestamps, frame_counts, hw_timestamps)
    )
    _write_trodes_binary(path, header={}, fields_spec=fields_spec, records=records)
    return path, {
        "pos_timestamps": pos_timestamps,
        "frame_counts": frame_counts,
        "hw_timestamps_ns": hw_timestamps,
        "n_pre_pause": n_pre_pause,
    }


def _make_position_tracking_file(
    tmp_path: Path,
    sample_times: list[int],
    *,
    led1: list[tuple[int, int]] | None = None,
    led2: list[tuple[int, int]] | None = None,
) -> Path:
    """Synthesize a ``.videoPositionTracking`` aligned to ``sample_times``."""

    path = tmp_path / "session.videoPositionTracking"
    n = len(sample_times)
    if led1 is None:
        led1 = [(100 + i, 200) for i in range(n)]
    if led2 is None:
        led2 = [(110 + i, 210) for i in range(n)]
    records = _pack_position_tracking(sample_times, led1, led2)
    fields_spec = "<time uint32><xloc uint16><yloc uint16><xloc2 uint16><yloc2 uint16>"
    _write_trodes_binary(path, header={}, fields_spec=fields_spec, records=records)
    return path


def _build_session_config(
    tmp_path: Path,
    pos_path: Path,
    ts_path: Path,
    *,
    state_mode: str = "vision_only",
    imu_path: Path | None = None,
    meters_per_pixel: float = 0.01,
) -> SessionConfig:
    inputs_kwargs: dict[str, Any] = {
        "format": "trodes_native",
        "trodes_native": TrodesNativeConfig(
            position_tracking_file=pos_path,
            camera_timestamps_file=ts_path,
        ),
    }
    if imu_path is not None:
        inputs_kwargs["imu_file"] = imu_path
    return SessionConfig(
        inputs=InputsConfig(**inputs_kwargs),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=meters_per_pixel),
        filter=FilterConfig(state_mode=state_mode),  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------
# Filename / column rejection.
# ----------------------------------------------------------------------


def test_non_ptp_filename_rejected_cameraHWFrameCount(tmp_path: Path) -> None:
    """``cameraHWFrameCount`` suffix raises with a v1-scope message."""

    ts_path, info = _make_ptp_timestamps_file(
        tmp_path, suffix=".videoTimeStamps.cameraHWFrameCount"
    )
    pos_path = _make_position_tracking_file(tmp_path, info["pos_timestamps"])
    with pytest.raises(ValueError, match="non-PTP"):
        load_trodes_native_position(pos_path, ts_path)


def test_non_ptp_filename_rejected_plain_videoTimeStamps(tmp_path: Path) -> None:
    """Plain ``videoTimeStamps`` suffix raises with a v1-scope message."""

    ts_path, info = _make_ptp_timestamps_file(tmp_path, suffix=".videoTimeStamps")
    pos_path = _make_position_tracking_file(tmp_path, info["pos_timestamps"])
    with pytest.raises(ValueError, match="non-PTP"):
        load_trodes_native_position(pos_path, ts_path)


def test_unknown_suffix_rejected(tmp_path: Path) -> None:
    """A timestamps file with an unrecognized suffix is also rejected."""

    ts_path, info = _make_ptp_timestamps_file(tmp_path, suffix=".garbage")
    pos_path = _make_position_tracking_file(tmp_path, info["pos_timestamps"])
    with pytest.raises(ValueError, match="recognized"):
        load_trodes_native_position(pos_path, ts_path)


def test_ptp_gate_via_column_missing_HWTimestamp(tmp_path: Path) -> None:
    """``cameraHWSync`` filename with no ``HWTimestamp`` column raises clearly.

    The filename suffix is a heuristic; the authoritative gate is
    column presence. We synthesize a HWSync-suffixed file with a
    PosTimestamp+frameCount-only Fields header.
    """

    n = 5
    pos_timestamps = [1000 + i * 1000 for i in range(n)]
    frame_counts = list(range(n))
    records = b"".join(
        struct.pack("<IH", ps, fc)
        for ps, fc in zip(pos_timestamps, frame_counts, strict=True)
    )
    ts_path, _ = _make_ptp_timestamps_file(
        tmp_path,
        n_pre_pause=2,
        n_post_pause=3,
        fields_override="<PosTimestamp uint32><frameCount uint16>",
        records_override=records,
    )
    pos_path = _make_position_tracking_file(tmp_path, pos_timestamps)

    with pytest.raises(ValueError, match="HWTimestamp"):
        load_trodes_native_position(pos_path, ts_path)


# ----------------------------------------------------------------------
# Happy-path: PTP join + LED extraction.
# ----------------------------------------------------------------------


def test_loader_extracts_leds_and_ptp_seconds(tmp_path: Path) -> None:
    """Standard PTP fixture loads to a ``PositionPixels`` with the
    pre-pause frames trimmed and HWTimestamp converted to seconds.
    """

    n_pre = 3
    n_post = 6
    ts_path, info = _make_ptp_timestamps_file(
        tmp_path, n_pre_pause=n_pre, n_post_pause=n_post
    )
    pos_path = _make_position_tracking_file(tmp_path, info["pos_timestamps"])

    pixels = load_trodes_native_position(pos_path, ts_path)

    # PTP pause removal drops the first ``n_pre`` frames.
    assert pixels.t_cam.shape == (n_post,)
    assert pixels.led1_pixels.shape == (n_post, 2)
    assert pixels.led2_pixels is not None
    assert pixels.led2_pixels.shape == (n_post, 2)

    # PTP seconds = HWTimestamp / 1e9. Compare against the post-pause
    # samples in the fixture.
    hw_ns_post = np.asarray(info["hw_timestamps_ns"][n_pre:])
    np.testing.assert_allclose(pixels.t_cam, hw_ns_post / NS_PER_S)

    # LED1 pixels match the post-pause records.
    expected_led1_x = np.asarray([100 + i for i in range(n_pre, n_pre + n_post)])
    np.testing.assert_array_equal(pixels.led1_pixels[:, 0], expected_led1_x)
    assert (pixels.led1_pixels[:, 1] == 200).all()

    # Diagnostics record pause removal and frame count.
    assert pixels.diagnostics["ptp_pause_frames_removed"] == n_pre
    assert pixels.diagnostics["frame_count"] == n_post
    assert pixels.diagnostics["format"] == "trodes_native"


def test_zero_led_rows_become_nan(tmp_path: Path) -> None:
    """Trodes online tracker emits zero rows when LEDs are lost; the
    loader masks them as NaN."""

    n_pre = 2
    n_post = 4
    ts_path, info = _make_ptp_timestamps_file(
        tmp_path, n_pre_pause=n_pre, n_post_pause=n_post
    )
    n_total = n_pre + n_post
    led1 = [(100 + i, 200) for i in range(n_total)]
    led2 = [(110 + i, 210) for i in range(n_total)]
    # Knock out LED1 on the first post-pause frame and LED2 on the
    # second.
    led1[n_pre] = (0, 0)
    led2[n_pre + 1] = (0, 0)
    pos_path = _make_position_tracking_file(
        tmp_path, info["pos_timestamps"], led1=led1, led2=led2
    )

    pixels = load_trodes_native_position(pos_path, ts_path)

    assert np.isnan(pixels.led1_pixels[0]).all()
    assert pixels.led2_pixels is not None
    assert np.isnan(pixels.led2_pixels[1]).all()
    # Non-zeroed rows remain finite.
    assert np.isfinite(pixels.led1_pixels[1:]).all()
    assert np.isfinite(pixels.led2_pixels[2:]).all()


# ----------------------------------------------------------------------
# IMU resolution: vision_only synthesizes; IMU-consuming raises.
# ----------------------------------------------------------------------


def test_imu_absent_with_vision_only_succeeds(tmp_path: Path) -> None:
    """No ``imu_file`` + ``state_mode='vision_only'`` produces a synthetic
    zero IMU stream."""

    ts_path, info = _make_ptp_timestamps_file(tmp_path)
    pos_path = _make_position_tracking_file(tmp_path, info["pos_timestamps"])
    config = _build_session_config(
        tmp_path, pos_path, ts_path, state_mode="vision_only"
    )

    session = load_session(config)

    # Synthetic IMU: 3 channels, length matches camera.
    assert session.U_imu.shape == (session.t_cam.size, 3)
    np.testing.assert_array_equal(session.U_imu, 0.0)
    assert session.diagnostics["loader"]["imu_source"] == "synthetic"


@pytest.mark.parametrize(
    "state_mode",
    ["2d_full", "2d_cam_3d_imu", "2d_cam_6dof_imu_orientation"],
)
def test_imu_absent_with_imu_consuming_state_mode_raises(
    tmp_path: Path, state_mode: str
) -> None:
    """Every IMU-consuming ``state_mode`` raises the three-option
    remediation message when no IMU source is provided."""

    ts_path, info = _make_ptp_timestamps_file(tmp_path)
    pos_path = _make_position_tracking_file(tmp_path, info["pos_timestamps"])
    config = _build_session_config(tmp_path, pos_path, ts_path, state_mode=state_mode)

    with pytest.raises(ValueError) as excinfo:
        load_session(config)
    msg = str(excinfo.value)
    assert "inputs.imu_file" in msg
    assert "vision_only" in msg
    assert "trodes_native" in msg


def test_imu_parquet_overrides_when_provided(tmp_path: Path) -> None:
    """Configuring ``inputs.imu_file`` routes the loader through
    ``load_imu_parquet`` and the run completes."""

    ts_path, info = _make_ptp_timestamps_file(tmp_path)
    pos_path = _make_position_tracking_file(tmp_path, info["pos_timestamps"])

    # Build a small IMU parquet aligned roughly to the camera HW
    # timestamps (in seconds since epoch).
    hw_post = np.asarray(info["hw_timestamps_ns"][info["n_pre_pause"] :]) / NS_PER_S
    imu_t = np.linspace(hw_post[0] - 0.1, hw_post[-1] + 0.1, 50)
    imu_path = tmp_path / "imu.parquet"
    pd.DataFrame(
        {
            "time": imu_t,
            "Headstage_GyroX": np.arange(50, dtype=int),
            "Headstage_GyroY": np.arange(50, dtype=int),
            "Headstage_GyroZ": np.arange(50, dtype=int),
            "Headstage_AccelX": np.arange(50, 100, dtype=int),
            "Headstage_AccelY": np.arange(100, 150, dtype=int),
            "Headstage_AccelZ": np.arange(150, 200, dtype=int),
        }
    ).to_parquet(imu_path)

    config = _build_session_config(
        tmp_path,
        pos_path,
        ts_path,
        state_mode="2d_cam_3d_imu",
        imu_path=imu_path,
    )

    session = load_session(config)

    # 3-channel IMU input under ``2d_cam_3d_imu`` projects to 4 channels
    # (gyro_z + accel_xyz). Length matches the deduplicated parquet.
    assert session.U_imu.ndim == 2
    assert session.U_imu.shape[1] == 4
    assert session.diagnostics["loader"]["imu_source"] == "parquet"
    assert session.diagnostics["loader"]["format"] == "trodes_native"


# ----------------------------------------------------------------------
# Pixel→meter scaling parity with the parquet path.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Cross-format parity: trodes_native vs spikegadgets_trodes parquet on
# the same underlying camera data → same filtered means.
# ----------------------------------------------------------------------


def test_filtered_means_parity_with_spikegadgets_trodes(tmp_path: Path) -> None:
    """Loading the same camera trajectory via ``trodes_native`` and via
    ``spikegadgets_trodes`` parquet yields filtered means within 1e-6.

    Both runs use ``state_mode='vision_only'`` so the EKF consumes only
    camera data; the synthetic zero-IMU stream is identical between
    paths and any divergence reflects a loader-side mismatch.
    """

    from trodestrack.config.schemas import OutputsConfig
    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

    n_pre = 2
    n_post = 30
    base_hw_ns = 1_700_000_000_000_000_000
    ts_path, info = _make_ptp_timestamps_file(
        tmp_path, n_pre_pause=n_pre, n_post_pause=n_post, base_hw_ns=base_hw_ns
    )
    pos_path_native = _make_position_tracking_file(tmp_path, info["pos_timestamps"])

    # Build the equivalent parquet from the same post-pause records.
    hw_post_seconds = (
        np.asarray(info["hw_timestamps_ns"][n_pre:], dtype=np.int64) / NS_PER_S
    )
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    pos_parquet = parquet_dir / "position.parquet"
    pd.DataFrame(
        {
            "time": hw_post_seconds,
            "xloc": [100 + i for i in range(n_pre, n_pre + n_post)],
            "yloc": [200] * n_post,
            "xloc2": [110 + i for i in range(n_pre, n_pre + n_post)],
            "yloc2": [210] * n_post,
        }
    ).to_parquet(pos_parquet)

    # spikegadgets_trodes also requires an IMU parquet, but we feed
    # zero-rate vision_only so the IMU values don't reach the filter.
    imu_parquet = parquet_dir / "imu.parquet"
    imu_t = np.linspace(hw_post_seconds[0] - 0.05, hw_post_seconds[-1] + 0.05, 80)
    # Non-trivial gyro_z so the gyro_z_change sample-hold dedup keeps
    # every row; the values themselves are irrelevant under
    # ``state_mode='vision_only'`` (the EKF doesn't consume IMU).
    pd.DataFrame(
        {
            "time": imu_t,
            "Headstage_GyroX": np.zeros(80, dtype=int),
            "Headstage_GyroY": np.zeros(80, dtype=int),
            "Headstage_GyroZ": np.arange(80, dtype=int),
            "Headstage_AccelX": np.zeros(80, dtype=int),
            "Headstage_AccelY": np.zeros(80, dtype=int),
            "Headstage_AccelZ": np.zeros(80, dtype=int),
        }
    ).to_parquet(imu_parquet)

    common_kwargs: dict[str, Any] = dict(
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="vision_only"),
        outputs=OutputsConfig(run_safety_checks=False),
    )

    config_native = SessionConfig(
        inputs=InputsConfig(
            format="trodes_native",
            imu_file=imu_parquet,
            trodes_native=TrodesNativeConfig(
                position_tracking_file=pos_path_native,
                camera_timestamps_file=ts_path,
            ),
        ),
        **common_kwargs,
    )
    config_parquet = SessionConfig(
        inputs=InputsConfig(
            format="spikegadgets_trodes",
            imu_file=imu_parquet,
            position_file=pos_parquet,
        ),
        **common_kwargs,
    )

    session_native = load_session(config_native)
    session_parquet = load_session(config_parquet)

    # Camera arrays must be identical at the loader boundary.
    np.testing.assert_allclose(session_native.t_cam, session_parquet.t_cam, atol=1e-12)
    np.testing.assert_allclose(
        session_native.Z_cam_led1, session_parquet.Z_cam_led1, atol=1e-12
    )
    np.testing.assert_allclose(
        session_native.Z_cam_led2, session_parquet.Z_cam_led2, atol=1e-12
    )

    ekf_config = EKFConfig(state_mode="vision_only")

    def _run(session) -> np.ndarray:
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
        return np.asarray(result.filtered_means)

    means_native = _run(session_native)
    means_parquet = _run(session_parquet)

    np.testing.assert_allclose(means_native, means_parquet, atol=1e-6, rtol=1e-6)


def test_pixel_to_meter_scaling_matches_camera_config(tmp_path: Path) -> None:
    """``camera.meters_per_pixel`` is applied to the loaded pixels
    before they reach ``PreparedSession``."""

    n_pre = 2
    n_post = 3
    ts_path, info = _make_ptp_timestamps_file(
        tmp_path, n_pre_pause=n_pre, n_post_pause=n_post
    )
    pos_path = _make_position_tracking_file(tmp_path, info["pos_timestamps"])
    config = _build_session_config(
        tmp_path,
        pos_path,
        ts_path,
        state_mode="vision_only",
        meters_per_pixel=0.01,
    )

    session = load_session(config)

    # Post-pause LED1 pixels are (100+i, 200) for i=2..4; meters_per_pixel=0.01.
    expected_x = np.asarray([100 + i for i in range(n_pre, n_pre + n_post)]) * 0.01
    np.testing.assert_allclose(session.Z_cam_led1[:, 0], expected_x)
    np.testing.assert_allclose(session.Z_cam_led1[:, 1], 2.0)

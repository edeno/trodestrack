"""``dlc_keypoints`` loader validation slice.

Fixtures are synthesized in-process: a single-animal DLC HDF5 via
``DataFrame.to_hdf(..., key="df_with_missing")`` plus a sibling
``_meta.pickle``. A multi-animal variant is built separately for the
rejection test. Trodes-PTP timestamps are reused from the trodes_native
fixture builders.
"""

from __future__ import annotations

import pickle
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Skip the whole module when ``[dlc]`` (PyTables) isn't installed —
# every test depends on ``DataFrame.to_hdf(..., format='table')``.
pytest.importorskip("tables")

from trodestrack.config import (
    CameraConfig,
    DLCKeypointsConfig,
    FilterConfig,
    IMUConfig,
    InputsConfig,
    SessionConfig,
)
from trodestrack.io import load_session
from trodestrack.io.loaders._dlc_keypoints import (
    load_dlc_keypoints_position,
)

# ----------------------------------------------------------------------
# Fixture builders.
# ----------------------------------------------------------------------


def _write_dlc_h5(
    path: Path,
    *,
    n_frames: int = 30,
    bodyparts: list[str] | None = None,
    scorer: str = "DLC_resnet50_test",
    led_xy_step: float = 1.0,
    led_likelihood: float = 0.9,
    multi_animal: bool = False,
) -> pd.DataFrame:
    """Write a DLC ``df_with_missing`` HDF5 with controllable likelihoods.

    Returns the DataFrame that was written so tests can match indices.
    """

    if bodyparts is None:
        bodyparts = ["led_green", "led_red"]
    coords = ["x", "y", "likelihood"]
    cols: list[Any] = [[scorer], bodyparts, coords]
    cols_names = ["scorer", "bodyparts", "coords"]
    if multi_animal:
        cols.insert(1, ["animal_0", "animal_1"])
        cols_names.insert(1, "individuals")
    index = pd.MultiIndex.from_product(cols, names=cols_names)

    n_cols = 1
    for level in cols:
        n_cols *= len(level)
    data = np.zeros((n_frames, n_cols), dtype=float)

    df = pd.DataFrame(data, columns=index)
    if multi_animal:
        for animal in cols[1]:
            for i, bp in enumerate(bodyparts):
                df.loc[:, (scorer, animal, bp, "x")] = (
                    100.0 + i * 10.0 + np.arange(n_frames) * led_xy_step
                )
                df.loc[:, (scorer, animal, bp, "y")] = 200.0 + i * 10.0
                df.loc[:, (scorer, animal, bp, "likelihood")] = led_likelihood
    else:
        for i, bp in enumerate(bodyparts):
            df.loc[:, (scorer, bp, "x")] = (
                100.0 + i * 10.0 + np.arange(n_frames) * led_xy_step
            )
            df.loc[:, (scorer, bp, "y")] = 200.0 + i * 10.0
            df.loc[:, (scorer, bp, "likelihood")] = led_likelihood

    df.to_hdf(path, key="df_with_missing", format="table", mode="w")
    return df


def _write_meta_pickle(
    h5_path: Path,
    *,
    fps: float = 30.0,
    nframes: int = 30,
    saver: str = "pytorch",
    cropping_parameters: list[int] | None = None,
    frame_dimensions_override: tuple[int, int] | None = None,
    scorer: str = "DLC_resnet50_test",
) -> None:
    """Write a sibling ``_meta.pickle`` matching either saver layout."""

    if frame_dimensions_override is not None:
        frame_dimensions = frame_dimensions_override
    elif saver == "tensorflow":
        # TF stores (ny, nx) = (height, width).
        frame_dimensions = (480, 640)
    else:
        # PyTorch stores (w, h).
        frame_dimensions = (640, 480)

    inner: dict[str, Any] = {
        "Scorer": scorer,
        "fps": fps,
        "nframes": nframes,
        "frame_dimensions": frame_dimensions,
        "cropping_parameters": cropping_parameters or [0, 640, 0, 480],
    }
    if saver == "pytorch":
        inner["pytorch-config"] = {"metadata": {"individuals": ["single"]}}
    elif saver == "tensorflow":
        inner["DLC-model-config file"] = "/path/to/model"

    pickle_path = h5_path.with_name(h5_path.stem + "_meta.pickle")
    with pickle_path.open("wb") as f:
        pickle.dump({"data": inner}, f)


def _write_h5_and_meta(
    tmp_path: Path,
    *,
    n_frames: int = 30,
    bodyparts: list[str] | None = None,
    scorer: str = "DLC_resnet50_test",
    saver: str = "pytorch",
    cropping_parameters: list[int] | None = None,
    led_likelihood: float = 0.9,
    frame_dimensions_override: tuple[int, int] | None = None,
) -> Path:
    h5_path = tmp_path / f"video{scorer}.h5"
    _write_dlc_h5(
        h5_path,
        n_frames=n_frames,
        bodyparts=bodyparts,
        scorer=scorer,
        led_likelihood=led_likelihood,
    )
    _write_meta_pickle(
        h5_path,
        nframes=n_frames,
        saver=saver,
        cropping_parameters=cropping_parameters,
        frame_dimensions_override=frame_dimensions_override,
        scorer=scorer,
    )
    return h5_path


# ----------------------------------------------------------------------
# Multi-animal rejection.
# ----------------------------------------------------------------------


def test_multi_animal_rejected(tmp_path: Path) -> None:
    """Multi-animal MultiIndex (``individuals`` level present) is
    rejected up front rather than silently slicing individual 0."""

    h5_path = tmp_path / "video.h5"
    _write_dlc_h5(h5_path, multi_animal=True)
    _write_meta_pickle(h5_path)

    with pytest.raises(ValueError, match="multi-animal"):
        load_dlc_keypoints_position(h5_path, "led_green", "led_red")


# ----------------------------------------------------------------------
# Likelihood gate.
# ----------------------------------------------------------------------


def test_likelihood_gate_nans_low_confidence_rows(tmp_path: Path) -> None:
    """Rows with ``likelihood < threshold`` are NaN'd in
    ``led{1,2}_pixels``; the per-frame raw likelihoods stay in
    ``confidence``."""

    n = 6
    h5_path = tmp_path / "video.h5"
    _write_dlc_h5(h5_path, n_frames=n)
    _write_meta_pickle(h5_path, nframes=n)

    # Patch likelihoods directly: drop LED1 on row 1, LED2 on row 3.
    df = pd.read_hdf(h5_path, key="df_with_missing")
    assert isinstance(df, pd.DataFrame)
    df.loc[1, ("DLC_resnet50_test", "led_green", "likelihood")] = 0.1
    df.loc[3, ("DLC_resnet50_test", "led_red", "likelihood")] = 0.2
    df.to_hdf(h5_path, key="df_with_missing", format="table", mode="w")

    pixels = load_dlc_keypoints_position(
        h5_path, "led_green", "led_red", likelihood_threshold=0.6
    )

    assert np.isnan(pixels.led1_pixels[1]).all()
    assert pixels.led2_pixels is not None
    assert np.isnan(pixels.led2_pixels[3]).all()
    # Other rows remain finite.
    assert np.isfinite(pixels.led1_pixels[[0, 2, 3, 4, 5]]).all()
    assert np.isfinite(pixels.led2_pixels[[0, 1, 2, 4, 5]]).all()
    # Confidence still carries the raw likelihoods (no NaN'ing there).
    assert pixels.confidence is not None
    assert pixels.confidence[1, 0] == pytest.approx(0.1)
    assert pixels.confidence[3, 2] == pytest.approx(0.2)


# ----------------------------------------------------------------------
# Crop offset.
# ----------------------------------------------------------------------


def test_crop_offset_applied_when_non_trivial(tmp_path: Path) -> None:
    """Non-zero ``cropping_parameters`` ``(x_min, y_min)`` is added to
    LED coordinates so output is in the original video frame."""

    n = 4
    h5_path = _write_h5_and_meta(
        tmp_path,
        n_frames=n,
        cropping_parameters=[50, 690, 30, 510],  # x_min=50, y_min=30
    )

    pixels = load_dlc_keypoints_position(
        h5_path, "led_green", "led_red", apply_crop_offset=True
    )

    # LED1 base x is 100; with x_min=50 offset, frame 0 sees 150.
    assert pixels.led1_pixels[0, 0] == pytest.approx(150.0)
    assert pixels.led1_pixels[0, 1] == pytest.approx(230.0)
    assert pixels.diagnostics["crop_offset_applied"] == (50.0, 30.0)


def test_crop_offset_skipped_when_zero(tmp_path: Path) -> None:
    """``cropping_parameters=[0, w, 0, h]`` (no crop) leaves coords
    unchanged."""

    h5_path = _write_h5_and_meta(tmp_path, cropping_parameters=[0, 640, 0, 480])

    pixels = load_dlc_keypoints_position(
        h5_path, "led_green", "led_red", apply_crop_offset=True
    )

    assert pixels.led1_pixels[0, 0] == pytest.approx(100.0)
    assert pixels.diagnostics["crop_offset_applied"] == (0.0, 0.0)


def test_apply_crop_offset_false_disables(tmp_path: Path) -> None:
    """Even with non-zero crop, ``apply_crop_offset=False`` skips it."""

    h5_path = _write_h5_and_meta(tmp_path, cropping_parameters=[50, 690, 30, 510])
    pixels = load_dlc_keypoints_position(
        h5_path, "led_green", "led_red", apply_crop_offset=False
    )
    assert pixels.led1_pixels[0, 0] == pytest.approx(100.0)
    assert pixels.diagnostics["crop_offset_applied"] == (0.0, 0.0)


# ----------------------------------------------------------------------
# frame_dimensions saver-detection.
# ----------------------------------------------------------------------


def test_pytorch_saver_normalizes_to_width_height(tmp_path: Path) -> None:
    """PyTorch saver writes ``(w, h)``; loader passes through verbatim."""

    h5_path = _write_h5_and_meta(
        tmp_path, saver="pytorch", frame_dimensions_override=(800, 600)
    )
    pixels = load_dlc_keypoints_position(h5_path, "led_green", "led_red")

    assert pixels.frame_dimensions == (800, 600)
    assert pixels.diagnostics["saver"] == "pytorch"


def test_tensorflow_saver_normalizes_to_width_height(tmp_path: Path) -> None:
    """TF saver writes ``(ny, nx)`` = ``(h, w)``; loader swaps to ``(w, h)``."""

    h5_path = _write_h5_and_meta(
        tmp_path, saver="tensorflow", frame_dimensions_override=(600, 800)
    )
    pixels = load_dlc_keypoints_position(h5_path, "led_green", "led_red")

    # On-disk (600, 800) is (height, width) for TF; normalized swaps it
    # back to (800, 600).
    assert pixels.frame_dimensions == (800, 600)
    assert pixels.diagnostics["saver"] == "tensorflow"


# ----------------------------------------------------------------------
# Timestamps source switching.
# ----------------------------------------------------------------------


def test_timestamps_meta_pickle_default(tmp_path: Path) -> None:
    """``meta_pickle`` synthesizes ``np.arange(nframes) / fps``."""

    n = 12
    h5_path = _write_h5_and_meta(tmp_path, n_frames=n)

    pixels = load_dlc_keypoints_position(h5_path, "led_green", "led_red")

    np.testing.assert_allclose(pixels.t_cam, np.arange(n) / 30.0)


def test_timestamps_timestamp_file(tmp_path: Path) -> None:
    """``timestamp_file`` reads a 1-D float-seconds array."""

    n = 8
    h5_path = _write_h5_and_meta(tmp_path, n_frames=n)
    ts_file = tmp_path / "frame_times.txt"
    times = np.linspace(10.0, 11.0, n)
    np.savetxt(ts_file, times)

    pixels = load_dlc_keypoints_position(
        h5_path,
        "led_green",
        "led_red",
        timestamps_source="timestamp_file",
        timestamp_file=ts_file,
    )

    np.testing.assert_allclose(pixels.t_cam, times)


def test_timestamps_timestamp_file_length_mismatch_raises(tmp_path: Path) -> None:
    """Mismatched timestamp count raises a clear error."""

    h5_path = _write_h5_and_meta(tmp_path, n_frames=8)
    ts_file = tmp_path / "frame_times.txt"
    np.savetxt(ts_file, np.linspace(0.0, 1.0, 12))

    with pytest.raises(ValueError, match="timestamp_file"):
        load_dlc_keypoints_position(
            h5_path,
            "led_green",
            "led_red",
            timestamps_source="timestamp_file",
            timestamp_file=ts_file,
        )


def test_timestamps_trodes_hw_sync(tmp_path: Path) -> None:
    """``trodes_hw_sync`` reads HWTimestamp from a Trodes
    ``*.videoTimeStamps.cameraHWSync`` binary."""

    n = 6
    h5_path = _write_h5_and_meta(tmp_path, n_frames=n)

    # Synthesize a Trodes cameraHWSync binary aligned 1:1 with the DLC
    # frames. Reuse the same byte layout as the trodes_native fixtures.
    ts_path = tmp_path / "session.videoTimeStamps.cameraHWSync"
    ns_per_s = 1_000_000_000
    base_ns = 1_700_000_000_000_000_000
    step_ns = ns_per_s // 30
    pos_timestamps = [1000 + i * 1000 for i in range(n)]
    frame_counts = list(range(n))
    hw_timestamps = [base_ns + i * step_ns for i in range(n)]
    fields_spec = "<PosTimestamp uint32><frameCount uint16><HWTimestamp uint64>"
    header = (f"<Start settings>\nFields: {fields_spec}\n<End settings>\n").encode(
        "ascii"
    )
    records = b"".join(
        struct.pack("<IHQ", ps, fc, hw)
        for ps, fc, hw in zip(pos_timestamps, frame_counts, hw_timestamps, strict=True)
    )
    ts_path.write_bytes(header + records)

    pixels = load_dlc_keypoints_position(
        h5_path,
        "led_green",
        "led_red",
        timestamps_source="trodes_hw_sync",
        camera_timestamps_file=ts_path,
    )

    expected = np.asarray(hw_timestamps, dtype=np.float64) / ns_per_s
    np.testing.assert_allclose(pixels.t_cam, expected)


def test_timestamps_trodes_hw_sync_length_mismatch_raises(tmp_path: Path) -> None:
    """Trodes timestamps length must match DLC frame count."""

    h5_path = _write_h5_and_meta(tmp_path, n_frames=10)

    ts_path = tmp_path / "session.videoTimeStamps.cameraHWSync"
    fields_spec = "<PosTimestamp uint32><frameCount uint16><HWTimestamp uint64>"
    header = (f"<Start settings>\nFields: {fields_spec}\n<End settings>\n").encode(
        "ascii"
    )
    records = b"".join(
        struct.pack("<IHQ", 1000 + i * 1000, i, 1_700_000_000_000_000_000 + i * 1000)
        for i in range(5)  # only 5 timestamps for 10 DLC frames
    )
    ts_path.write_bytes(header + records)

    with pytest.raises(ValueError, match="must agree 1:1"):
        load_dlc_keypoints_position(
            h5_path,
            "led_green",
            "led_red",
            timestamps_source="trodes_hw_sync",
            camera_timestamps_file=ts_path,
        )


# ----------------------------------------------------------------------
# Bodypart selection.
# ----------------------------------------------------------------------


def test_unknown_bodypart_raises(tmp_path: Path) -> None:
    """Configured bodypart names are validated against the HDF5
    columns; missing ones raise a clear error listing what's available."""

    h5_path = _write_h5_and_meta(tmp_path, bodyparts=["led_green", "led_red"])

    with pytest.raises(ValueError, match="led_blue"):
        load_dlc_keypoints_position(h5_path, "led_blue", "led_red")


# ----------------------------------------------------------------------
# Confidence layout matches existing _load_leds shape.
# ----------------------------------------------------------------------


def test_confidence_shape_matches_load_leds(tmp_path: Path) -> None:
    """``confidence`` is ``(n, 4)`` laid out as
    ``[c1, c1, c2, c2]`` so the EKF observation gating sees the same
    shape as the existing parquet path produces in ``_load_leds``."""

    n = 5
    h5_path = _write_h5_and_meta(tmp_path, n_frames=n)
    pixels = load_dlc_keypoints_position(h5_path, "led_green", "led_red")

    assert pixels.confidence is not None
    assert pixels.confidence.shape == (n, 4)
    np.testing.assert_array_equal(pixels.confidence[:, 0], pixels.confidence[:, 1])
    np.testing.assert_array_equal(pixels.confidence[:, 2], pixels.confidence[:, 3])


# ----------------------------------------------------------------------
# session.load_session() integration: extra missing + IMU resolution.
# ----------------------------------------------------------------------


def _build_dlc_session_config(
    h5_path: Path,
    *,
    state_mode: str = "vision_only",
) -> SessionConfig:
    return SessionConfig(
        inputs=InputsConfig(
            format="dlc_keypoints",
            dlc_keypoints=DLCKeypointsConfig(
                h5_file=h5_path,
                led1_bodypart="led_green",
                led2_bodypart="led_red",
            ),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode=state_mode),  # type: ignore[arg-type]
    )


def test_load_session_dlc_keypoints_vision_only(tmp_path: Path) -> None:
    """End-to-end: load_session() with state_mode=vision_only produces
    a synthetic zero IMU stream sized to the camera vector."""

    n = 12
    h5_path = _write_h5_and_meta(tmp_path, n_frames=n)
    config = _build_dlc_session_config(h5_path, state_mode="vision_only")

    session = load_session(config)

    assert session.t_cam.shape == (n,)
    assert session.U_imu.shape == (n, 3)
    np.testing.assert_array_equal(session.U_imu, 0.0)
    assert session.diagnostics["loader"]["format"] == "dlc_keypoints"
    assert session.diagnostics["loader"]["imu_source"] == "synthetic"


def test_load_session_dlc_keypoints_imu_consuming_raises(tmp_path: Path) -> None:
    """No IMU + IMU-consuming state_mode raises with the three-option
    remediation message (same surface as trodes_native)."""

    h5_path = _write_h5_and_meta(tmp_path)
    config = _build_dlc_session_config(h5_path, state_mode="2d_cam_3d_imu")

    with pytest.raises(ValueError) as excinfo:
        load_session(config)
    msg = str(excinfo.value)
    assert "dlc_keypoints" in msg
    assert "inputs.imu_file" in msg
    assert "vision_only" in msg


def test_meta_pickle_with_unix_imu_raises_overlap_error(tmp_path: Path) -> None:
    """``timestamps_source='meta_pickle'`` synthesizes relative seconds
    starting at 0; pairing it with a Unix-like ``inputs.imu_file``
    (typically ~1.7e9 s) used to silently produce an IMU-configured
    fused trajectory with zero IMU samples per camera interval. The
    loader-level overlap check now raises with a remediation message."""

    h5_path = _write_h5_and_meta(tmp_path, n_frames=5)

    # IMU parquet on a Unix-like clock (~2023-11-14).
    imu_path = tmp_path / "imu.parquet"
    n_imu = 30
    imu_t = np.linspace(1_700_000_000.0, 1_700_000_001.0, n_imu)
    pd.DataFrame(
        {
            "time": imu_t,
            "Headstage_GyroX": np.zeros(n_imu, dtype=int),
            "Headstage_GyroY": np.zeros(n_imu, dtype=int),
            "Headstage_GyroZ": np.arange(n_imu, dtype=int),
            "Headstage_AccelX": np.zeros(n_imu, dtype=int),
            "Headstage_AccelY": np.zeros(n_imu, dtype=int),
            "Headstage_AccelZ": np.zeros(n_imu, dtype=int),
        }
    ).to_parquet(imu_path)

    config = SessionConfig(
        inputs=InputsConfig(
            format="dlc_keypoints",
            imu_file=imu_path,
            dlc_keypoints=DLCKeypointsConfig(
                h5_file=h5_path,
                led1_bodypart="led_green",
                led2_bodypart="led_red",
            ),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="2d_cam_3d_imu"),
    )

    with pytest.raises(ValueError) as excinfo:
        load_session(config)
    msg = str(excinfo.value)
    assert "do not overlap" in msg
    assert "trodes_hw_sync" in msg
    assert "timestamp_file" in msg
    assert "time_offset_s" in msg


def test_dlc_extra_missing_raises_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``[dlc]`` (PyTables) is missing, ``load_session()`` raises
    ImportError naming the install command — not a generic pandas
    traceback."""

    h5_path = _write_h5_and_meta(tmp_path)
    monkeypatch.setitem(sys.modules, "tables", None)
    config = _build_dlc_session_config(h5_path)

    with pytest.raises(ImportError, match=r"trodestrack\[dlc\]"):
        load_session(config)


# ----------------------------------------------------------------------
# Single-LED tracking_geometry.
# ----------------------------------------------------------------------


def test_single_led1_dlc_loads_with_one_bodypart(tmp_path: Path) -> None:
    """``tracking_geometry='single_led1'`` reads only led1_bodypart;
    LED2 array is filled with NaN."""

    h5_path = _write_h5_and_meta(tmp_path, n_frames=20)
    pixels = load_dlc_keypoints_position(
        h5_path,
        led1_bodypart="led_green",
        led2_bodypart=None,
        tracking_geometry="single_led1",
    )

    assert pixels.led1_pixels.shape == (20, 2)
    assert pixels.led2_pixels is not None
    assert pixels.led2_pixels.shape == (20, 2)
    assert np.isnan(pixels.led2_pixels).all()
    assert np.isfinite(pixels.led1_pixels).all()
    assert pixels.diagnostics["tracking_geometry"] == "single_led1"
    # Confidence layout: observed columns get real values, missing
    # columns get neutral 1.0 (uniform).
    assert pixels.confidence is not None
    np.testing.assert_array_equal(pixels.confidence[:, 2], np.full(20, 1.0))
    np.testing.assert_array_equal(pixels.confidence[:, 3], np.full(20, 1.0))


def test_single_led2_dlc_swaps_observed_and_missing_arrays(tmp_path: Path) -> None:
    """``tracking_geometry='single_led2'`` puts the observation in
    LED2 and NaNs LED1."""

    h5_path = _write_h5_and_meta(tmp_path, n_frames=20)
    pixels = load_dlc_keypoints_position(
        h5_path,
        led1_bodypart=None,
        led2_bodypart="led_red",
        tracking_geometry="single_led2",
    )

    assert pixels.led2_pixels is not None
    assert np.isfinite(pixels.led2_pixels).all()
    assert np.isnan(pixels.led1_pixels).all()


def test_single_led_dlc_session_runs_vision_only(tmp_path: Path) -> None:
    """End-to-end ``load_session`` with single-LED DLC + vision_only
    produces an LED-pair-shaped session with mask matching the
    observed LED's finiteness."""

    h5_path = _write_h5_and_meta(tmp_path, n_frames=24)
    config = SessionConfig(
        inputs=InputsConfig(
            format="dlc_keypoints",
            dlc_keypoints=DLCKeypointsConfig(
                h5_file=h5_path,
                tracking_geometry="single_led1",
                led1_bodypart="led_green",
            ),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="vision_only", led_distance=0.05),
    )

    session = load_session(config)

    assert session.Z_cam_led1.shape == (24, 2)
    assert session.Z_cam_led2.shape == (24, 2)
    assert np.isnan(session.Z_cam_led2).all()
    np.testing.assert_array_equal(
        session.mask_cam, np.isfinite(session.Z_cam_led1).all(axis=1)
    )
    assert session.led_distance == pytest.approx(0.05)


def test_dlc_dual_led_missing_bodypart_raises_at_schema_time() -> None:
    """``tracking_geometry='dual_led'`` (default) requires both bodypart
    fields. A copy-paste leftover with only one set must fail at
    config-load time, not silently NaN one LED."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="led2_bodypart"):
        DLCKeypointsConfig.model_validate(
            {
                "h5_file": "x.h5",
                "led1_bodypart": "led_green",
                # led2_bodypart missing — old behavior would have
                # required both via field-level validation; new
                # behavior (after lifting required-ness into the
                # validator) catches this at the cross-field check.
            },
        )


@pytest.mark.parametrize(
    "geometry,forbidden,value",
    [
        ("single_led1", "led2_bodypart", "led_red"),
        ("single_led2", "led1_bodypart", "led_green"),
    ],
)
def test_dlc_single_led_rejects_other_bodypart(
    geometry: str, forbidden: str, value: str
) -> None:
    """Setting the *other* LED's bodypart under single-LED is rejected
    so a copy-paste leftover can't silently steer the loader path."""

    from pydantic import ValidationError

    observed = "led1_bodypart" if geometry == "single_led1" else "led2_bodypart"
    with pytest.raises(ValidationError, match=forbidden):
        DLCKeypointsConfig.model_validate(
            {
                "h5_file": "x.h5",
                "tracking_geometry": geometry,
                observed: "x",
                forbidden: value,
            },
        )

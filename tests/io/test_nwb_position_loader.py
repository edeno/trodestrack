"""NWB position loader validation slice.

Fixtures are synthesized in-process via ``pynwb`` itself (one
Trodes-style ``Position``, one ndx-pose v0.2.x ``PoseEstimation`` with
a ``Skeleton`` chain). The v0.1.x ``PoseEstimation`` fixture is built
from a v0.2.x file by deleting the ``Skeletons`` link with h5py
post-write so the test does not require pinning an old ``ndx-pose``
package version.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

# Skip the module when ``[nwb]`` (pynwb) isn't installed.
pytest.importorskip("pynwb")

import pynwb
from hdmf.common import VectorData
from pynwb.behavior import Position

from trodestrack.config import (
    CameraConfig,
    FilterConfig,
    IMUConfig,
    InputsConfig,
    NWBConfig,
    NWBLEDSourceConfig,
    SessionConfig,
)
from trodestrack.io import load_session
from trodestrack.io.nwb import (
    from_pose_estimation_container,
    from_position_container,
    load_nwb_session,
)

# ---------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------


def _make_nwb_skeleton(
    session_id: str = "session_0",
) -> pynwb.NWBFile:
    """Construct a minimal NWBFile with a behavior processing module."""

    return pynwb.NWBFile(
        session_description="test session",
        identifier=str(uuid4()),
        session_start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        session_id=session_id,
    )


def _add_trodes_position(
    nwbfile: pynwb.NWBFile,
    *,
    n_frames: int = 30,
    unit: str = "pixels",
    conversion: float = 1.0,
    led1_name: str = "led_0_series_0",
    led2_name: str = "led_1_series_0",
) -> Position:
    """Attach a Trodes-style Position container with two SpatialSeries.

    LED1 traces ``(100+i, 200)``, LED2 traces ``(110+i, 210)``.
    Timestamps span ``[0, n_frames-1] / 30`` seconds.
    """

    timestamps = np.arange(n_frames, dtype=float) / 30.0
    led1_data = np.column_stack([100.0 + np.arange(n_frames), np.full(n_frames, 200.0)])
    led2_data = np.column_stack([110.0 + np.arange(n_frames), np.full(n_frames, 210.0)])

    position = Position()
    position.create_spatial_series(
        name=led1_name,
        description="LED1",
        data=led1_data,
        unit=unit,
        conversion=conversion,
        reference_frame="Upper left corner of video frame",
        timestamps=timestamps,
    )
    position.create_spatial_series(
        name=led2_name,
        description="LED2",
        data=led2_data,
        unit=unit,
        conversion=conversion,
        reference_frame="Upper left corner of video frame",
        timestamps=timestamps,
    )

    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    behavior.add(position)
    return position


def _add_pose_estimation(
    nwbfile: pynwb.NWBFile,
    *,
    n_frames: int = 30,
    bodyparts: tuple[str, str] = ("led_green", "led_red"),
    likelihood: float = 0.9,
    unit: str = "pixels",
    conversion: float = 1.0,
) -> Any:
    """Attach an ndx-pose v0.2.x PoseEstimation + Skeleton."""

    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons

    timestamps = np.arange(n_frames, dtype=float) / 30.0
    series_list = []
    for i, bp in enumerate(bodyparts):
        data = np.column_stack(
            [100.0 + i * 10.0 + np.arange(n_frames), np.full(n_frames, 200.0 + i * 10)]
        ).astype(np.float32)
        confidence = np.full(n_frames, likelihood, dtype=np.float32)
        series = PoseEstimationSeries(
            name=bp,
            description=f"Pose estimate for {bp}",
            data=data,
            unit=unit,
            conversion=conversion,
            reference_frame="Upper left corner of video frame",
            timestamps=timestamps,
            confidence=confidence,
            confidence_definition="Softmax output of the deep neural network",
        )
        series_list.append(series)

    skeleton = Skeleton(
        name="rat_skeleton",
        nodes=list(bodyparts),
        edges=np.array([[0, 1]], dtype=np.uint8),
    )
    skeletons = Skeletons(skeletons=[skeleton])

    pose = PoseEstimation(
        pose_estimation_series=series_list,
        description="DLC pose estimation (test)",
        original_videos=["video.mp4"],
        labeled_videos=["video_labeled.mp4"],
        source_software="DeepLabCut",
        source_software_version="2.3.0",
        scorer="DLC_resnet50",
        skeleton=skeleton,
    )

    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    behavior.add(skeletons)
    behavior.add(pose)
    return pose


def _write_nwb(nwbfile: pynwb.NWBFile, path: Path) -> None:
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def _trodes_position_fixture(
    tmp_path: Path,
    *,
    unit: str = "pixels",
    conversion: float = 1.0,
    led1_name: str = "led_0_series_0",
    led2_name: str = "led_1_series_0",
) -> Path:
    nwbfile = _make_nwb_skeleton()
    _add_trodes_position(
        nwbfile,
        unit=unit,
        conversion=conversion,
        led1_name=led1_name,
        led2_name=led2_name,
    )
    path = tmp_path / "trodes_position.nwb"
    _write_nwb(nwbfile, path)
    return path


def _pose_estimation_fixture(
    tmp_path: Path,
    *,
    likelihood: float = 0.9,
    bodyparts: tuple[str, str] = ("led_green", "led_red"),
) -> Path:
    nwbfile = _make_nwb_skeleton()
    _add_pose_estimation(nwbfile, likelihood=likelihood, bodyparts=bodyparts)
    path = tmp_path / "pose_estimation.nwb"
    _write_nwb(nwbfile, path)
    return path


def _v0_1_x_pose_fixture(tmp_path: Path) -> Path:
    """Strip the v0.2.x Skeleton link and replant nodes/edges inline.

    The result mirrors what ndx-pose <0.2.0 wrote: a PoseEstimation
    group with inline ``nodes`` / ``edges`` datasets and no
    ``Skeletons`` container alongside it.
    """

    import h5py

    src = _pose_estimation_fixture(tmp_path)
    path = tmp_path / "pose_estimation_v0_1_x.nwb"
    # Copy then mutate to avoid clobbering the v0.2.x fixture.
    path.write_bytes(src.read_bytes())
    with h5py.File(path, "a") as f:
        pose_group = f["processing/behavior/PoseEstimation"]
        # Delete the skeleton soft link so the loader can't take the
        # v0.2.x branch.
        if "skeleton" in pose_group:
            del pose_group["skeleton"]
        # Inline ``nodes`` / ``edges`` mirroring the pre-0.2.0 layout.
        if "nodes" not in pose_group:
            pose_group.create_dataset(
                "nodes", data=np.array(["led_green", "led_red"], dtype="S")
            )
        if "edges" not in pose_group:
            pose_group.create_dataset("edges", data=np.array([[0, 1]], dtype=np.uint8))
        # Drop the now-orphan Skeletons container so the file stays valid.
        if "Skeletons" in f["processing/behavior"]:
            del f["processing/behavior/Skeletons"]
    return path


# ---------------------------------------------------------------------
# [nwb] extra missing.
# ---------------------------------------------------------------------


def test_load_session_nwb_extra_missing_raises_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_session()`` for ``inputs.format=nwb`` raises a clear
    ImportError when the [nwb] extra (pynwb) is missing."""

    nwb_path = _trodes_position_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, "pynwb", None)
    config = SessionConfig(
        inputs=InputsConfig(
            format="nwb",
            nwb=NWBConfig(nwb_file=nwb_path),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="vision_only"),
    )
    with pytest.raises(ImportError, match=r"trodestrack\[nwb\]"):
        load_session(config)


# ---------------------------------------------------------------------
# Module import does not require pynwb.
# ---------------------------------------------------------------------


def test_import_trodestrack_io_nwb_does_not_import_pynwb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Container-layer entries don't import pynwb at module load. With
    pynwb removed from ``sys.modules`` and import blocked, ``import
    trodestrack.io.nwb`` succeeds. Only ``load_nwb_session()``
    triggers the lazy import."""

    # Drop any cached references so the next ``import`` re-runs the
    # module body.
    for mod_name in list(sys.modules):
        if mod_name.startswith("trodestrack.io.nwb") or mod_name == "pynwb":
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    # Block pynwb at import time.
    monkeypatch.setitem(sys.modules, "pynwb", None)

    import importlib

    nwb_module = importlib.import_module("trodestrack.io.nwb")

    assert hasattr(nwb_module, "from_position_container")
    assert hasattr(nwb_module, "from_pose_estimation_container")
    assert hasattr(nwb_module, "load_nwb_session")
    # pynwb stayed blocked (module-load did not force a real import).
    assert sys.modules.get("pynwb") is None


# ---------------------------------------------------------------------
# Container API parity with path loader (Trodes Position).
# ---------------------------------------------------------------------


def test_from_position_container_matches_load_nwb_session(tmp_path: Path) -> None:
    """``from_position_container`` produces the same ``PositionPixels``
    as ``load_nwb_session`` does for the same file."""

    nwb_path = _trodes_position_fixture(tmp_path)
    cfg = NWBConfig(nwb_file=nwb_path)

    via_path, _ = load_nwb_session(cfg)

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        position = nwbfile.processing["behavior"]["Position"]
        via_container = from_position_container(position, cfg.led_source)

    np.testing.assert_array_equal(via_path.led1_pixels, via_container.led1_pixels)
    assert via_path.led2_pixels is not None
    assert via_container.led2_pixels is not None
    np.testing.assert_array_equal(via_path.led2_pixels, via_container.led2_pixels)
    np.testing.assert_array_equal(via_path.t_cam, via_container.t_cam)
    assert via_path.coords_meters_per_pixel == via_container.coords_meters_per_pixel


def test_from_pose_estimation_container_matches_load_nwb_session(
    tmp_path: Path,
) -> None:
    """``from_pose_estimation_container`` produces the same
    ``PositionPixels`` as ``load_nwb_session`` for the same file."""

    nwb_path = _pose_estimation_fixture(tmp_path)
    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(
            led1_bodypart="led_green", led2_bodypart="led_red"
        ),
    )

    via_path, _ = load_nwb_session(cfg)

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        pose = nwbfile.processing["behavior"]["PoseEstimation"]
        via_container = from_pose_estimation_container(pose, cfg.led_source)

    np.testing.assert_array_equal(via_path.led1_pixels, via_container.led1_pixels)
    assert via_path.led2_pixels is not None
    assert via_container.led2_pixels is not None
    np.testing.assert_array_equal(via_path.led2_pixels, via_container.led2_pixels)
    np.testing.assert_array_equal(via_path.t_cam, via_container.t_cam)


# ---------------------------------------------------------------------
# Eager numpy materialization.
# ---------------------------------------------------------------------


def test_container_eager_materialization_after_io_close(tmp_path: Path) -> None:
    """After calling ``from_position_container`` and closing the
    backing IO, the returned arrays remain readable. No
    ``h5py.Dataset`` references should leak through."""

    nwb_path = _trodes_position_fixture(tmp_path)
    cfg = NWBConfig(nwb_file=nwb_path)

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        position = nwbfile.processing["behavior"]["Position"]
        pixels = from_position_container(position, cfg.led_source)
    # IO closed; touching the arrays must still work.
    assert pixels.led1_pixels.sum() > 0
    assert pixels.led2_pixels is not None
    assert pixels.led2_pixels.sum() > 0
    assert pixels.t_cam[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Auto-detect by neurodata type with non-default container names.
# ---------------------------------------------------------------------


def test_auto_detect_handles_non_default_series_names(tmp_path: Path) -> None:
    """``container='auto'`` matches by neurodata_type, not by the
    writer-default series names."""

    nwb_path = _trodes_position_fixture(
        tmp_path, led1_name="custom_led_alpha", led2_name="custom_led_beta"
    )
    # Auto-detection should still find the Position container; the
    # SpatialSeries names fall through to the "any two series" branch.
    cfg = NWBConfig(nwb_file=nwb_path)

    pixels, _ = load_nwb_session(cfg)

    assert pixels.led1_pixels.shape == (30, 2)


def test_auto_detect_handles_renamed_position_container(tmp_path: Path) -> None:
    """``container='auto'`` matches a ``Position`` container by
    ``neurodata_type`` even when the container itself has a
    non-default name (writers may rename it ``custom_position`` or
    ``RawPosition`` per their schema)."""

    nwbfile = _make_nwb_skeleton()
    n_frames = 30
    timestamps = np.arange(n_frames, dtype=float) / 30.0
    led1_data = np.column_stack([100.0 + np.arange(n_frames), np.full(n_frames, 200.0)])
    led2_data = np.column_stack([110.0 + np.arange(n_frames), np.full(n_frames, 210.0)])

    # Construct a Position container with a non-default name. The
    # neurodata_type is still "Position" — the auto-detect pathway
    # must match on type, not name.
    position = Position(name="custom_position_container")
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
    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    behavior.add(position)

    nwb_path = tmp_path / "renamed_position.nwb"
    _write_nwb(nwbfile, nwb_path)
    pixels, _ = load_nwb_session(NWBConfig(nwb_file=nwb_path))

    assert pixels.led1_pixels.shape == (30, 2)


# ---------------------------------------------------------------------
# Pair-validation guards.
# ---------------------------------------------------------------------


def test_paired_led_timestamps_must_match(tmp_path: Path) -> None:
    """LED1 and LED2 SpatialSeries must share a clock — silently
    pairing LED2 samples with LED1's time base would let a mis-paired
    write produce a fused-but-misaligned trajectory."""

    nwbfile = _make_nwb_skeleton()
    n = 5
    led1_data = np.column_stack([100.0 + np.arange(n), np.full(n, 200.0)])
    led2_data = np.column_stack([110.0 + np.arange(n), np.full(n, 210.0)])
    t1 = np.arange(n, dtype=float)
    t2 = t1 + 10.0  # offset by 10s — same shape, different clock

    position = Position()
    position.create_spatial_series(
        name="led_0_series_0",
        description="LED1",
        data=led1_data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=t1,
    )
    position.create_spatial_series(
        name="led_1_series_0",
        description="LED2",
        data=led2_data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=t2,
    )
    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    behavior.add(position)

    nwb_path = tmp_path / "mismatched_timestamps.nwb"
    _write_nwb(nwbfile, nwb_path)

    with pytest.raises(ValueError, match="timestamps differ"):
        load_nwb_session(NWBConfig(nwb_file=nwb_path))


def test_paired_led_conversion_mismatch_rejected(tmp_path: Path) -> None:
    """Per-series ``conversion`` mismatches are rejected up front.

    Silently picking LED1's calibration would produce a fused-but-
    miscalibrated trajectory at the same fault surface as the
    timestamp-mismatch guard.
    """

    nwbfile = _make_nwb_skeleton()
    n = 5
    led1_data = np.column_stack([100.0 + np.arange(n), np.full(n, 200.0)])
    led2_data = np.column_stack([110.0 + np.arange(n), np.full(n, 210.0)])
    timestamps = np.arange(n, dtype=float)

    position = Position()
    # LED1 is the sentinel "no calibration baked in".
    position.create_spatial_series(
        name="led_0_series_0",
        description="LED1",
        data=led1_data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
    )
    # LED2 is calibrated — different scalar from LED1.
    position.create_spatial_series(
        name="led_1_series_0",
        description="LED2",
        data=led2_data,
        unit="meters",
        conversion=0.005,
        reference_frame="upper-left",
        timestamps=timestamps,
    )
    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    behavior.add(position)

    nwb_path = tmp_path / "mismatched_conversion.nwb"
    _write_nwb(nwbfile, nwb_path)

    with pytest.raises(ValueError, match="calibration mismatch"):
        load_nwb_session(NWBConfig(nwb_file=nwb_path))


# ---------------------------------------------------------------------
# ndx-pose v0.1.x compatibility.
# ---------------------------------------------------------------------


def test_v0_1_x_inline_nodes_resolves(tmp_path: Path) -> None:
    """A v0.1.x ndx-pose file (inline ``nodes`` on PoseEstimation, no
    ``Skeletons``) loads to the same downstream output as the v0.2.x
    Skeleton-chain version."""

    v02_path = _pose_estimation_fixture(tmp_path)
    v01_path = _v0_1_x_pose_fixture(tmp_path)
    led_source = NWBLEDSourceConfig(led1_bodypart="led_green", led2_bodypart="led_red")

    pixels_v02, _ = load_nwb_session(
        NWBConfig(nwb_file=v02_path, led_source=led_source)
    )
    pixels_v01, _ = load_nwb_session(
        NWBConfig(nwb_file=v01_path, led_source=led_source)
    )

    # The validation row's central assertion: "both produce same
    # downstream output." Position data is read from
    # ``pose.pose_estimation_series[bodypart]`` regardless of how
    # ``nodes``/``edges`` are stored, so v0.1.x and v0.2.x layouts
    # must yield byte-identical PositionPixels arrays.
    np.testing.assert_array_equal(pixels_v02.led1_pixels, pixels_v01.led1_pixels)
    assert pixels_v02.led2_pixels is not None
    assert pixels_v01.led2_pixels is not None
    np.testing.assert_array_equal(pixels_v02.led2_pixels, pixels_v01.led2_pixels)
    np.testing.assert_array_equal(pixels_v02.t_cam, pixels_v01.t_cam)
    # Schema-version detection is best-effort diagnostic; v0.2.x is
    # always reliably detected when a real Skeleton chain is present.
    assert pixels_v02.diagnostics["ndx_pose_schema_version"] == "v0.2.x"


# ---------------------------------------------------------------------
# Conversion=1.0 sentinel.
# ---------------------------------------------------------------------


def test_conversion_1_0_pixels_is_sentinel(tmp_path: Path) -> None:
    """``unit='pixels'`` + ``conversion=1.0`` is a sentinel meaning
    "no calibration baked in"; ``coords_meters_per_pixel`` is None,
    so the YAML default takes over via ``pixels_to_meters``."""

    nwb_path = _trodes_position_fixture(tmp_path, unit="pixels", conversion=1.0)
    pixels, _ = load_nwb_session(NWBConfig(nwb_file=nwb_path))

    assert pixels.coords_meters_per_pixel is None


def test_conversion_meters_unit_is_calibrated(tmp_path: Path) -> None:
    """``unit='meters'`` means already-calibrated; ``conversion``
    becomes the file-side scalar for ``pixels_to_meters``."""

    nwb_path = _trodes_position_fixture(tmp_path, unit="meters", conversion=0.005)
    pixels, _ = load_nwb_session(NWBConfig(nwb_file=nwb_path))

    assert pixels.coords_meters_per_pixel == pytest.approx(0.005)


def test_conversion_non_unit_pixels_passes_through(tmp_path: Path) -> None:
    """``unit='pixels'`` + ``conversion != 1.0`` is treated as
    already-converted (writer baked the calibration in)."""

    nwb_path = _trodes_position_fixture(tmp_path, unit="pixels", conversion=0.0035)
    pixels, _ = load_nwb_session(NWBConfig(nwb_file=nwb_path))

    assert pixels.coords_meters_per_pixel == pytest.approx(0.0035)


# ---------------------------------------------------------------------
# Attribute-name policy: read both definition / confidence_definition.
# ---------------------------------------------------------------------


def test_confidence_definition_resolves_via_either_name(tmp_path: Path) -> None:
    """The loader reads the confidence-definition string regardless of
    which Python attribute name pynwb / ndx-pose surfaces."""

    nwb_path = _pose_estimation_fixture(tmp_path)
    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(
            led1_bodypart="led_green", led2_bodypart="led_red"
        ),
    )
    pixels, _ = load_nwb_session(cfg)

    assert pixels.diagnostics["confidence_definition"] is not None


# ---------------------------------------------------------------------
# meters_per_pixel_override.
# ---------------------------------------------------------------------


def test_meters_per_pixel_override_wins_over_file_conversion(tmp_path: Path) -> None:
    """``inputs.nwb.meters_per_pixel_override`` takes precedence over
    a file-stored conversion in ``pixels_to_meters``'s output."""

    nwb_path = _trodes_position_fixture(tmp_path, unit="meters", conversion=0.005)
    config = SessionConfig(
        inputs=InputsConfig(
            format="nwb",
            nwb=NWBConfig(nwb_file=nwb_path, meters_per_pixel_override=0.001),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.0022),
        filter=FilterConfig(state_mode="vision_only"),
    )

    session = load_session(config)

    # Override scales pixels by 0.001; LED1 frame 0 = (100, 200) px →
    # (0.1, 0.2) m. File conversion 0.005 and YAML 0.0022 are ignored.
    assert session.Z_cam_led1[0, 0] == pytest.approx(0.1)
    assert session.Z_cam_led1[0, 1] == pytest.approx(0.2)


def test_pose_estimation_likelihood_gate(tmp_path: Path) -> None:
    """Per-frame likelihoods below ``led_source.likelihood_threshold``
    are NaN'd in ``led{1,2}_pixels`` while raw values are preserved
    in ``confidence``."""

    # Build a fixture where some frames have low confidence.
    nwbfile = _make_nwb_skeleton()
    n = 10
    timestamps = np.arange(n, dtype=float) / 30.0

    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons

    series_list = []
    for i, bp in enumerate(("led_green", "led_red")):
        data = np.column_stack(
            [100.0 + i * 10 + np.arange(n), np.full(n, 200.0 + i * 10)]
        ).astype(np.float32)
        confidence = np.full(n, 0.9, dtype=np.float32)
        # Dropouts: LED1 row 2, LED2 row 5.
        if i == 0:
            confidence[2] = 0.1
        else:
            confidence[5] = 0.05
        series_list.append(
            PoseEstimationSeries(
                name=bp,
                description=bp,
                data=data,
                unit="pixels",
                conversion=1.0,
                reference_frame="upper-left",
                timestamps=timestamps,
                confidence=confidence,
                confidence_definition="likelihood",
            )
        )
    skel = Skeleton(name="s", nodes=["led_green", "led_red"])
    skels = Skeletons(skeletons=[skel])
    pose = PoseEstimation(
        pose_estimation_series=series_list,
        description="test",
        source_software="DeepLabCut",
        scorer="DLC_resnet50",
        skeleton=skel,
    )
    behavior = nwbfile.create_processing_module("behavior", "behavior")
    behavior.add(skels)
    behavior.add(pose)
    nwb_path = tmp_path / "pose_dropouts.nwb"
    _write_nwb(nwbfile, nwb_path)

    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(
            led1_bodypart="led_green",
            led2_bodypart="led_red",
            likelihood_threshold=0.6,
        ),
    )
    pixels, _ = load_nwb_session(cfg)

    assert np.isnan(pixels.led1_pixels[2]).all()
    assert pixels.led2_pixels is not None
    assert np.isnan(pixels.led2_pixels[5]).all()
    # Other rows finite.
    assert np.isfinite(pixels.led1_pixels[[0, 1, 3, 4, 5, 6, 7, 8, 9]]).all()
    # Raw confidence preserved.
    assert pixels.confidence is not None
    assert pixels.confidence[2, 0] == pytest.approx(0.1)
    assert pixels.confidence[5, 2] == pytest.approx(0.05)


# ---------------------------------------------------------------------
# Single-LED tracking_geometry: Position container.
# ---------------------------------------------------------------------


def _add_single_trodes_position(
    nwbfile: pynwb.NWBFile,
    *,
    n_frames: int = 30,
    series_name: str = "led_0_series_0",
    extra_series_names: tuple[str, ...] = (),
) -> Position:
    """Position container with one or more SpatialSeries (single-LED).

    The first series traces ``(100+i, 200)``. ``extra_series_names``
    add additional series so the auto-detect-rejects-multiple path
    can be exercised.
    """

    timestamps = np.arange(n_frames, dtype=float) / 30.0
    main_data = np.column_stack([100.0 + np.arange(n_frames), np.full(n_frames, 200.0)])

    position = Position()
    position.create_spatial_series(
        name=series_name,
        description="Single LED",
        data=main_data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
    )
    for i, name in enumerate(extra_series_names, start=1):
        position.create_spatial_series(
            name=name,
            description=f"Extra series {i}",
            data=main_data + i,
            unit="pixels",
            conversion=1.0,
            reference_frame="upper-left",
            timestamps=timestamps,
        )

    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    behavior.add(position)
    return position


def _single_trodes_position_fixture(
    tmp_path: Path,
    *,
    series_name: str = "led_0_series_0",
    extra_series_names: tuple[str, ...] = (),
) -> Path:
    nwbfile = _make_nwb_skeleton()
    _add_single_trodes_position(
        nwbfile, series_name=series_name, extra_series_names=extra_series_names
    )
    path = tmp_path / "single_trodes_position.nwb"
    _write_nwb(nwbfile, path)
    return path


def test_single_led1_position_auto_detects_one_spatial_series(tmp_path: Path) -> None:
    """With ``tracking_geometry='single_led1'`` and one SpatialSeries,
    the loader pulls it as LED1 and fills LED2 with NaN."""

    nwb_path = _single_trodes_position_fixture(tmp_path)
    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(tracking_geometry="single_led1"),
    )

    pixels, _ = load_nwb_session(cfg)

    assert pixels.led1_pixels.shape == (30, 2)
    assert pixels.led2_pixels is not None
    assert pixels.led2_pixels.shape == (30, 2)
    # Observed LED has the (100+i, 200) trace; missing LED is all NaN.
    np.testing.assert_array_equal(pixels.led1_pixels[:, 1], np.full(30, 200.0))
    assert np.isnan(pixels.led2_pixels).all()
    assert pixels.diagnostics["tracking_geometry"] == "single_led1"
    assert pixels.diagnostics["observed_series_name"] == "led_0_series_0"


def test_single_led2_position_swaps_observed_and_missing_arrays(tmp_path: Path) -> None:
    """``tracking_geometry='single_led2'`` puts observation in LED2."""

    nwb_path = _single_trodes_position_fixture(tmp_path, series_name="anything")
    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(tracking_geometry="single_led2"),
    )

    pixels, _ = load_nwb_session(cfg)

    assert pixels.led2_pixels is not None
    np.testing.assert_array_equal(pixels.led2_pixels[:, 1], np.full(30, 200.0))
    assert np.isnan(pixels.led1_pixels).all()


def test_single_led_position_named_series_disambiguates_multiple(
    tmp_path: Path,
) -> None:
    """When the container holds >1 SpatialSeries, ``led1_series_name``
    is required to pick which one is the physical LED."""

    nwb_path = _single_trodes_position_fixture(
        tmp_path, series_name="primary", extra_series_names=("secondary",)
    )
    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(
            tracking_geometry="single_led1", led1_series_name="primary"
        ),
    )

    pixels, _ = load_nwb_session(cfg)
    assert pixels.diagnostics["observed_series_name"] == "primary"


def test_single_led_position_with_multiple_series_and_no_name_raises(
    tmp_path: Path,
) -> None:
    """Multiple series + no name in config → loader cannot pick one."""

    nwb_path = _single_trodes_position_fixture(
        tmp_path, series_name="a", extra_series_names=("b",)
    )
    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(tracking_geometry="single_led1"),
    )
    with pytest.raises(ValueError, match="exactly one SpatialSeries"):
        load_nwb_session(cfg)


def test_dual_led_with_only_one_series_still_raises(tmp_path: Path) -> None:
    """Backward compat: default ``tracking_geometry='dual_led'`` still
    rejects a Position container that holds only one SpatialSeries."""

    nwb_path = _single_trodes_position_fixture(tmp_path)
    cfg = NWBConfig(nwb_file=nwb_path)
    with pytest.raises(ValueError, match="auto-detect"):
        load_nwb_session(cfg)


# ---------------------------------------------------------------------
# Single-LED tracking_geometry: PoseEstimation container.
# ---------------------------------------------------------------------


def _single_pose_estimation_fixture(
    tmp_path: Path,
    *,
    bodypart: str = "led_green",
    likelihood: float = 0.9,
) -> Path:
    """ndx-pose v0.2.x PoseEstimation with one bodypart."""

    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons

    nwbfile = _make_nwb_skeleton()
    n_frames = 30
    timestamps = np.arange(n_frames, dtype=float) / 30.0
    data = np.column_stack(
        [100.0 + np.arange(n_frames), np.full(n_frames, 200.0)]
    ).astype(np.float32)
    confidence = np.full(n_frames, likelihood, dtype=np.float32)
    series = PoseEstimationSeries(
        name=bodypart,
        description=f"Single bodypart {bodypart}",
        data=data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
        confidence=confidence,
        confidence_definition="Softmax output of the deep neural network",
    )
    skeleton = Skeleton(
        name="rat_skeleton",
        nodes=[bodypart],
        edges=np.zeros((0, 2), dtype=np.uint8),
    )
    skeletons = Skeletons(skeletons=[skeleton])
    pose = PoseEstimation(
        pose_estimation_series=[series],
        description="Single-LED pose estimation (test)",
        original_videos=["video.mp4"],
        labeled_videos=["video_labeled.mp4"],
        source_software="DeepLabCut",
        source_software_version="2.3.0",
        scorer="DLC_resnet50",
        skeleton=skeleton,
    )
    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    behavior.add(skeletons)
    behavior.add(pose)
    path = tmp_path / "single_pose_estimation.nwb"
    _write_nwb(nwbfile, path)
    return path


def test_single_led1_pose_estimation_loads_with_one_bodypart(tmp_path: Path) -> None:
    nwb_path = _single_pose_estimation_fixture(tmp_path, bodypart="led_green")
    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(
            tracking_geometry="single_led1", led1_bodypart="led_green"
        ),
    )

    pixels, _ = load_nwb_session(cfg)

    assert pixels.led1_pixels.shape == (30, 2)
    assert pixels.led2_pixels is not None
    assert np.isnan(pixels.led2_pixels).all()
    assert pixels.diagnostics["observed_bodypart"] == "led_green"
    # Confidence layout: observed columns get real values, missing
    # columns get neutral 1.0.
    assert pixels.confidence is not None
    assert pixels.confidence.shape == (30, 4)
    np.testing.assert_allclose(pixels.confidence[:, 0], np.full(30, 0.9), rtol=1e-6)
    np.testing.assert_array_equal(pixels.confidence[:, 2], np.full(30, 1.0))


def test_single_led_pose_estimation_missing_bodypart_raises(tmp_path: Path) -> None:
    """Required bodypart name not present → clear error with the
    available names."""

    nwb_path = _single_pose_estimation_fixture(tmp_path, bodypart="led_green")
    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(
            tracking_geometry="single_led1", led1_bodypart="not_present"
        ),
    )
    with pytest.raises(ValueError, match="not_present"):
        load_nwb_session(cfg)


def test_single_led_pose_estimation_likelihood_gate_applies_to_observed(
    tmp_path: Path,
) -> None:
    """Below-threshold frames in the observed bodypart get NaN'd while
    the missing-LED column stays NaN throughout."""

    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons

    nwbfile = _make_nwb_skeleton()
    n = 10
    timestamps = np.arange(n, dtype=float) / 30.0
    data = np.column_stack([100.0 + np.arange(n), np.full(n, 200.0)]).astype(np.float32)
    confidence = np.full(n, 0.9, dtype=np.float32)
    confidence[3] = 0.1
    series = PoseEstimationSeries(
        name="led_green",
        description="single",
        data=data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
        confidence=confidence,
        confidence_definition="softmax",
    )
    skeleton = Skeleton(
        name="rat",
        nodes=["led_green"],
        edges=np.zeros((0, 2), dtype=np.uint8),
    )
    pose = PoseEstimation(
        pose_estimation_series=[series],
        description="",
        original_videos=["v.mp4"],
        labeled_videos=["vl.mp4"],
        source_software="DLC",
        source_software_version="2",
        scorer="x",
        skeleton=skeleton,
    )
    behavior = nwbfile.create_processing_module("behavior", "behavior")
    behavior.add(Skeletons(skeletons=[skeleton]))
    behavior.add(pose)
    nwb_path = tmp_path / "single_drop.nwb"
    _write_nwb(nwbfile, nwb_path)

    cfg = NWBConfig(
        nwb_file=nwb_path,
        led_source=NWBLEDSourceConfig(
            tracking_geometry="single_led1",
            led1_bodypart="led_green",
            likelihood_threshold=0.6,
        ),
    )
    pixels, _ = load_nwb_session(cfg)

    assert np.isnan(pixels.led1_pixels[3]).all()
    assert pixels.led2_pixels is not None
    assert np.isnan(pixels.led2_pixels).all()


# ---------------------------------------------------------------------
# End-to-end load_session: mask reflects observed-LED finiteness only.
# ---------------------------------------------------------------------


def test_single_led_session_mask_cam_only_tracks_observed_led(tmp_path: Path) -> None:
    """``mask_cam`` for a single-LED session is true exactly where the
    observed LED is finite (the missing LED is all-NaN, so the OR'd
    mask collapses to the observed LED's finiteness)."""

    nwb_path = _single_trodes_position_fixture(tmp_path)
    config = SessionConfig(
        inputs=InputsConfig(
            format="nwb",
            nwb=NWBConfig(
                nwb_file=nwb_path,
                led_source=NWBLEDSourceConfig(tracking_geometry="single_led1"),
            ),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="vision_only", led_distance=0.05),
    )

    session = load_session(config)

    assert session.Z_cam_led1.shape == (30, 2)
    assert session.Z_cam_led2.shape == (30, 2)
    assert np.isnan(session.Z_cam_led2).all()
    np.testing.assert_array_equal(
        session.mask_cam, np.isfinite(session.Z_cam_led1).all(axis=1)
    )
    assert session.led_distance == pytest.approx(0.05)


def _single_led_with_analog_fixture(
    tmp_path: Path,
    *,
    n_frames: int = 60,
    n_imu_samples: int = 120,
) -> Path:
    """Single-LED Position container plus a Trodes-shaped analog IMU
    TimeSeries (6 headstage channels)."""

    from pynwb.behavior import BehavioralEvents

    nwbfile = _make_nwb_skeleton()
    behavior = nwbfile.create_processing_module(name="behavior", description="behavior")
    timestamps = np.arange(n_frames, dtype=float) / 30.0
    led_data = np.column_stack([100.0 + np.arange(n_frames), np.full(n_frames, 200.0)])
    position = Position()
    position.create_spatial_series(
        name="led_0_series_0",
        description="single LED",
        data=led_data,
        unit="pixels",
        conversion=1.0,
        reference_frame="upper-left",
        timestamps=timestamps,
    )
    behavior.add(position)

    analog_module = nwbfile.create_processing_module(
        name="analog", description="analog data"
    )
    imu_channels = [
        "Headstage_GyroX",
        "Headstage_GyroY",
        "Headstage_GyroZ",
        "Headstage_AccelX",
        "Headstage_AccelY",
        "Headstage_AccelZ",
    ]
    description = "   ".join(imu_channels) + "   "
    # All-zero IMU keeps the EKF stable for a smoke test (no synthetic
    # gyro/accel that the filter would have to integrate through).
    analog_t = np.linspace(timestamps[0], timestamps[-1], n_imu_samples)
    data = np.zeros((n_imu_samples, len(imu_channels)), dtype=np.int16)
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

    path = tmp_path / "single_led_with_analog.nwb"
    _write_nwb(nwbfile, path)
    return path


def test_single_led_fused_mode_runs_end_to_end(tmp_path: Path) -> None:
    """``state_mode='2d_cam_3d_imu'`` with single-LED NWB + analog
    IMU runs the full loader → EKF → safety-check pipeline. Catches
    wiring regressions in the loader, the LED-pair-shaped observation
    handoff, and the safety-check envelope path (which must use the
    observed LED's trajectory, not the all-NaN midpoint)."""

    from trodestrack.io.session import run_real_data_safety_check
    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

    nwb_path = _single_led_with_analog_fixture(tmp_path)
    config = SessionConfig(
        inputs=InputsConfig(
            format="nwb",
            nwb=NWBConfig(
                nwb_file=nwb_path,
                led_source=NWBLEDSourceConfig(tracking_geometry="single_led1"),
            ),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="2d_cam_3d_imu", led_distance=0.05),
    )

    session = load_session(config)
    assert session.led_distance == pytest.approx(0.05)
    assert session.diagnostics["loader"]["imu_source"] == "nwb_analog"
    assert np.isnan(session.Z_cam_led2).all()

    ekf_config = EKFConfig(
        state_mode="2d_cam_3d_imu", led_distance=session.led_distance
    )
    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=session.t_imu,
        U_imu=session.U_imu,
        t_cam=session.t_cam,
        Z_cam_led1=session.Z_cam_led1,
        Z_cam_led2=session.Z_cam_led2,
        mask_cam=session.mask_cam,
    )
    means = np.asarray(result.filtered_means)
    assert means.shape[0] == session.t_cam.shape[0]
    assert np.isfinite(means).all()

    # Safety check: the envelope path must read from LED1 (LED2 is
    # all-NaN); without the single-LED branch in ``_camera_envelope``
    # the dual-LED midpoint would be NaN everywhere and this would
    # raise "requires at least N finite ... frame(s)".
    safety = run_real_data_safety_check(session, ekf_config, result)
    assert safety.dual_led_frame_count == session.t_cam.shape[0]


# ---------------------------------------------------------------------
# Camera envelope helper (safety-check unit test).
# ---------------------------------------------------------------------


def test_camera_envelope_dual_led_returns_midpoint() -> None:
    from trodestrack.io.session import _camera_envelope

    n = 10
    led1 = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    led2 = np.column_stack([np.arange(n, dtype=float) + 4.0, np.zeros(n)])
    mask = np.ones(n, dtype=bool)

    envelope, valid, source = _camera_envelope(led1, led2, mask)
    assert source == "dual_led_midpoint"
    np.testing.assert_array_equal(envelope[:, 0], np.arange(n, dtype=float) + 2.0)
    assert valid.all()


def test_camera_envelope_single_led1_uses_led1_directly() -> None:
    from trodestrack.io.session import _camera_envelope

    n = 10
    led1 = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    led2 = np.full((n, 2), np.nan)
    mask = np.ones(n, dtype=bool)

    envelope, valid, source = _camera_envelope(led1, led2, mask)
    assert source == "single_led1"
    np.testing.assert_array_equal(envelope, led1)
    assert valid.all()


def test_camera_envelope_single_led2_uses_led2_directly() -> None:
    from trodestrack.io.session import _camera_envelope

    n = 10
    led1 = np.full((n, 2), np.nan)
    led2 = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    mask = np.ones(n, dtype=bool)

    envelope, valid, source = _camera_envelope(led1, led2, mask)
    assert source == "single_led2"
    np.testing.assert_array_equal(envelope, led2)
    assert valid.all()


# Touch ``VectorData`` to silence unused-import lints when only some
# of the optional helpers above need it.
_ = VectorData

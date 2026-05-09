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


# Touch ``VectorData`` to silence unused-import lints when only some
# of the optional helpers above need it.
_ = VectorData

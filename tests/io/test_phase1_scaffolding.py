"""Phase 1 — schema and loader-stub validation slice.

These tests cover:

- Schema enforcement: each new ``inputs.format`` value requires the
  matching nested config block.
- Path resolution: relative paths inside the new blocks resolve from
  the YAML's parent directory (matching the existing flat-paths
  contract).
- TTL events behavior unchanged: ``TTLEventsConfig.events_file`` is
  still required (Phase 4c relaxes it).
- Stub dispatch: missing extras raise ``ImportError`` naming the
  install command; with extras present, stubs raise
  ``NotImplementedError`` naming the implementing phase.

Loader stubs only verify the import surface; full ingest lands in
Phases 2 / 3 / 4a / 4c.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from trodestrack.config import load_session_config
from trodestrack.config.schemas import (
    DLCKeypointsConfig,
    NWBConfig,
    SessionConfig,
    TrodesNativeConfig,
)
from trodestrack.io import load_session
from trodestrack.io.session import (
    _load_dlc_keypoints,
    _load_nwb,
)

# ----------------------------------------------------------------------
# Schema: missing per-format block raises a clear ValidationError.
# ----------------------------------------------------------------------


def test_format_trodes_native_requires_block() -> None:
    """``format='trodes_native'`` without a ``trodes_native:`` block
    fails schema validation."""

    with pytest.raises(ValidationError, match=r"inputs\.trodes_native"):
        SessionConfig.model_validate(
            {"inputs": {"format": "trodes_native"}},
        )


def test_format_dlc_keypoints_requires_block() -> None:
    with pytest.raises(ValidationError, match=r"inputs\.dlc_keypoints"):
        SessionConfig.model_validate(
            {"inputs": {"format": "dlc_keypoints"}},
        )


def test_format_nwb_requires_block() -> None:
    with pytest.raises(ValidationError, match=r"inputs\.nwb"):
        SessionConfig.model_validate(
            {"inputs": {"format": "nwb"}},
        )


# ----------------------------------------------------------------------
# Schema: existing TTL behavior unchanged (Phase 4c will relax this).
# ----------------------------------------------------------------------


def test_ttl_events_file_still_required() -> None:
    """``TTLEventsConfig.events_file`` remains required in Phase 1."""

    with pytest.raises(ValidationError, match="events_file"):
        SessionConfig.model_validate(
            {
                "inputs": {
                    "format": "nwb",
                    "nwb": {"nwb_file": "session.nwb"},
                },
                "ttl_events": {
                    # Missing events_file — must still raise.
                    "beams": [],
                    "zone_triggers": [],
                    "rfid_readers": [],
                },
            },
        )


# ----------------------------------------------------------------------
# DLCKeypointsConfig: timestamps_source consistency validator.
# ----------------------------------------------------------------------


def test_dlc_trodes_hw_sync_requires_camera_timestamps_file() -> None:
    with pytest.raises(ValidationError, match="camera_timestamps_file"):
        DLCKeypointsConfig.model_validate(
            {
                "h5_file": "dlc.h5",
                "led1_bodypart": "led_green",
                "led2_bodypart": "led_red",
                "timestamps_source": "trodes_hw_sync",
            },
        )


def test_dlc_timestamp_file_source_requires_timestamp_file() -> None:
    with pytest.raises(ValidationError, match="timestamp_file"):
        DLCKeypointsConfig.model_validate(
            {
                "h5_file": "dlc.h5",
                "led1_bodypart": "led_green",
                "led2_bodypart": "led_red",
                "timestamps_source": "timestamp_file",
            },
        )


# ----------------------------------------------------------------------
# NWBLEDSourceConfig: paired-name validator.
# ----------------------------------------------------------------------


def test_nwb_led_source_partial_series_pair_raises() -> None:
    """Half-set ``led1_series_name`` without ``led2_series_name`` used
    to silently fall through to auto-detect for the missing side; now
    the schema rejects it at config-load time."""

    from trodestrack.config import NWBLEDSourceConfig

    with pytest.raises(ValidationError, match=r"led1_series_name"):
        NWBLEDSourceConfig.model_validate({"led1_series_name": "custom_led1"})


def test_nwb_led_source_partial_bodypart_pair_raises() -> None:
    """Same pair guard applies to the ndx-pose bodypart fields."""

    from trodestrack.config import NWBLEDSourceConfig

    with pytest.raises(ValidationError, match=r"led1_bodypart"):
        NWBLEDSourceConfig.model_validate({"led1_bodypart": "led_green"})


def test_nwb_led_source_both_series_names_or_neither() -> None:
    """Setting both halves succeeds; setting neither succeeds."""

    from trodestrack.config import NWBLEDSourceConfig

    NWBLEDSourceConfig.model_validate(
        {"led1_series_name": "a", "led2_series_name": "b"}
    )
    NWBLEDSourceConfig.model_validate({})


# ----------------------------------------------------------------------
# Path resolution: nested-block Paths resolve from the YAML directory.
# ----------------------------------------------------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    config_path = tmp_path / "session.yaml"
    config_path.write_text(body.lstrip())
    return config_path


def test_path_resolution_trodes_native(tmp_path: Path) -> None:
    """Relative paths under ``trodes_native:`` resolve from the YAML
    directory."""

    config_path = _write_yaml(
        tmp_path,
        """
inputs:
  format: trodes_native
  trodes_native:
    position_tracking_file: data/session.videoPositionTracking
    camera_timestamps_file: data/session.videoTimeStamps.cameraHWSync
""",
    )

    config = load_session_config(config_path)

    assert config.inputs.trodes_native is not None
    block = config.inputs.trodes_native
    expected_pos = tmp_path / "data" / "session.videoPositionTracking"
    expected_ts = tmp_path / "data" / "session.videoTimeStamps.cameraHWSync"
    assert block.position_tracking_file == expected_pos
    assert block.camera_timestamps_file == expected_ts


def test_path_resolution_dlc_keypoints(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path,
        """
inputs:
  format: dlc_keypoints
  dlc_keypoints:
    h5_file: data/dlc.h5
    led1_bodypart: led_green
    led2_bodypart: led_red
    timestamps_source: trodes_hw_sync
    camera_timestamps_file: data/sync.videoTimeStamps.cameraHWSync
""",
    )

    config = load_session_config(config_path)

    assert config.inputs.dlc_keypoints is not None
    block = config.inputs.dlc_keypoints
    assert block.h5_file == tmp_path / "data" / "dlc.h5"
    assert (
        block.camera_timestamps_file
        == tmp_path / "data" / "sync.videoTimeStamps.cameraHWSync"
    )


def test_path_resolution_nwb(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path,
        """
inputs:
  format: nwb
  nwb:
    nwb_file: data/session.nwb
""",
    )

    config = load_session_config(config_path)

    assert config.inputs.nwb is not None
    assert config.inputs.nwb.nwb_file == tmp_path / "data" / "session.nwb"


def test_absolute_path_passes_through_unchanged(tmp_path: Path) -> None:
    """Absolute paths in nested blocks are not re-rooted."""

    abs_pos = tmp_path / "abs" / "session.videoPositionTracking"
    abs_ts = tmp_path / "abs" / "session.videoTimeStamps.cameraHWSync"
    config_path = _write_yaml(
        tmp_path,
        f"""
inputs:
  format: trodes_native
  trodes_native:
    position_tracking_file: {abs_pos}
    camera_timestamps_file: {abs_ts}
""",
    )

    config = load_session_config(config_path)

    assert config.inputs.trodes_native is not None
    block = config.inputs.trodes_native
    assert block.position_tracking_file == abs_pos
    assert block.camera_timestamps_file == abs_ts


# ----------------------------------------------------------------------
# Stub loader dispatch.
# ----------------------------------------------------------------------


def _dlc_keypoints_config(tmp_path: Path) -> SessionConfig:
    return SessionConfig.model_validate(
        {
            "inputs": {
                "format": "dlc_keypoints",
                "dlc_keypoints": {
                    "h5_file": str(tmp_path / "dlc.h5"),
                    "led1_bodypart": "led_green",
                    "led2_bodypart": "led_red",
                },
            },
        },
    )


def _nwb_config(tmp_path: Path) -> SessionConfig:
    return SessionConfig.model_validate(
        {
            "inputs": {
                "format": "nwb",
                "nwb": {"nwb_file": str(tmp_path / "session.nwb")},
            },
        },
    )


def test_dlc_keypoints_extra_missing_raises_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``tables`` removed from ``sys.modules`` and import blocked,
    the stub raises ``ImportError`` naming the install command."""

    monkeypatch.setitem(sys.modules, "tables", None)
    config = _dlc_keypoints_config(tmp_path)
    with pytest.raises(ImportError, match=r"trodestrack\[dlc\]"):
        _load_dlc_keypoints(config)


def test_nwb_extra_missing_raises_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pynwb", None)
    config = _nwb_config(tmp_path)
    with pytest.raises(ImportError, match=r"trodestrack\[nwb\]"):
        _load_nwb(config)


# ----------------------------------------------------------------------
# load_session() top-level dispatch goes through the stubs.
# ----------------------------------------------------------------------


def test_load_session_dispatches_to_dlc_keypoints_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_session`` routes ``format='dlc_keypoints'`` to the stub.

    Uses the missing-extra path for deterministic coverage regardless
    of whether ``[dlc]`` is installed in the test environment.
    """

    monkeypatch.setitem(sys.modules, "tables", None)
    config = _dlc_keypoints_config(tmp_path)
    with pytest.raises(ImportError, match=r"trodestrack\[dlc\]"):
        load_session(config)


def test_load_session_dispatches_to_nwb_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_session`` routes ``format='nwb'`` to the stub.

    Uses the missing-extra path for deterministic coverage regardless
    of whether ``[nwb]`` is installed in the test environment.
    """

    monkeypatch.setitem(sys.modules, "pynwb", None)
    config = _nwb_config(tmp_path)
    with pytest.raises(ImportError, match=r"trodestrack\[nwb\]"):
        load_session(config)


# ----------------------------------------------------------------------
# Public API: new schema classes are importable from trodestrack.config.
# ----------------------------------------------------------------------


def test_new_schemas_are_in_public_config_api() -> None:
    """``from trodestrack.config import <NewConfig>`` works for every
    Phase 1 schema, matching the existing ``SessionConfig`` /
    ``LedIdentityConfig`` re-export convention."""

    import trodestrack.config as cfg

    for name in (
        "TrodesNativeConfig",
        "DLCKeypointsConfig",
        "NWBConfig",
        "NWBLEDSourceConfig",
        "NWBDIOToTTLConfig",
    ):
        assert hasattr(cfg, name), f"trodestrack.config does not export {name}"
        assert name in cfg.__all__, f"trodestrack.config.__all__ does not list {name}"


# ----------------------------------------------------------------------
# Sanity: nested block dataclasses construct standalone.
# ----------------------------------------------------------------------


def test_nested_blocks_construct_standalone() -> None:
    """The new schemas are usable standalone (matters for Spyglass-style
    callers that construct configs programmatically)."""

    TrodesNativeConfig(
        position_tracking_file=Path("a.videoPositionTracking"),
        camera_timestamps_file=Path("a.videoTimeStamps.cameraHWSync"),
    )
    DLCKeypointsConfig(
        h5_file=Path("a.h5"),
        led1_bodypart="led_green",
        led2_bodypart="led_red",
    )
    NWBConfig(nwb_file=Path("a.nwb"))

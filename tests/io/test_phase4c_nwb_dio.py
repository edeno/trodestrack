"""Phase 4c — NWB DIO → TTL bridge validation slice.

Schema-side tests cover the conditional ``events_file`` requirement,
the geometry-block requirement when DIO is configured, and the
``name_to_source_id`` cross-check. Loader-side tests cover the
container API (``from_behavioral_events`` accepting both
``BehavioralEvents`` and ``dict[str, TimeSeries]`` shapes), eager
materialization, end-to-end event population through ``load_session``,
and the parquet-overrides-NWB-DIO precedence.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

# Skip the module when ``[nwb]`` (pynwb) isn't installed.
pytest.importorskip("pynwb")

import pynwb
from pynwb.behavior import BehavioralEvents, Position

from trodestrack.config import (
    BeamSpec,
    CameraConfig,
    FilterConfig,
    IMUConfig,
    InputsConfig,
    NWBConfig,
    NWBDIOToTTLConfig,
    SessionConfig,
    TTLEventsConfig,
    ZoneTriggerSpec,
)
from trodestrack.io import load_session
from trodestrack.io.nwb import from_behavioral_events, load_nwb_session

# ---------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------


def _make_nwb_with_position_and_dio(
    tmp_path: Path,
    *,
    n_frames: int = 30,
    include_dio: bool = True,
    dio_event_names: tuple[str, ...] = ("beam_1", "zone_a"),
    base_systime_s: float = 0.5,
) -> Path:
    """Build a Phase-4a-style NWB with optional DIO TimeSeries.

    DIO encoding follows ``trodes_to_nwb/spike_gadgets_raw_io.py:953``:
    int8 0/1 where each value is a transition (1 = rising, 0 = falling)
    and the very first sample is the initial level.
    """

    nwbfile = pynwb.NWBFile(
        session_description="Phase 4c test session",
        identifier=str(uuid4()),
        session_start_time=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        session_id="session_0",
    )

    # Position: same Trodes-style two-LED layout as Phase 4a/4b.
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

    if include_dio:
        # Each DIO TimeSeries: 6 samples, first is initial level
        # (dropped by the loader), then 5 alternating transitions.
        # Distinguishable per-channel timing so a parity test can
        # tell them apart in the merged output.
        events_container = BehavioralEvents(name="behavioral_events")
        for i, name in enumerate(dio_event_names):
            data = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
            ts = np.asarray(
                [base_systime_s + i * 0.01 + j * 0.05 for j in range(6)],
                dtype=float,
            )
            events_container.add_timeseries(
                pynwb.TimeSeries(
                    name=name,
                    description=f"DIO channel {name}",
                    data=data,
                    timestamps=ts,
                    unit="N/A",
                )
            )
        behavior.add(events_container)

    path = tmp_path / "session.nwb"
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    return path


def _make_dio_session_config(
    nwb_path: Path,
    *,
    parquet_events_file: Path | None = None,
) -> SessionConfig:
    return SessionConfig(
        inputs=InputsConfig(
            format="nwb",
            nwb=NWBConfig(
                nwb_file=nwb_path,
                dio_to_ttl=NWBDIOToTTLConfig(
                    name_to_source_id={"beam_1": 1, "zone_a": 2},
                ),
            ),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="vision_only"),
        ttl_events=TTLEventsConfig(
            events_file=parquet_events_file,
            beams=[
                BeamSpec(id=1, emitter=(0.0, 0.0), receiver=(0.1, 0.0)),
            ],
            zone_triggers=[
                ZoneTriggerSpec(id=2, center=(0.5, 0.5)),
            ],
        ),
    )


def _write_parquet_events(path: Path, n: int = 4) -> None:
    pd.DataFrame(
        {
            "time": np.linspace(0.6, 0.95, n),
            "source_id": np.full(n, 1, dtype=np.int64),
            "edge": ["rise"] * n,
        }
    ).to_parquet(path)


# ---------------------------------------------------------------------
# Schema-side: events_file conditional + dio_to_ttl cross-checks.
# ---------------------------------------------------------------------


def test_events_file_required_when_no_dio_bridge() -> None:
    """``ttl_events.events_file=None`` is rejected unless the DIO
    bridge is configured."""

    with pytest.raises(ValidationError, match=r"events_file is required"):
        SessionConfig(
            inputs=InputsConfig(
                format="prepared_arrays",
                imu_timestamps=Path("a"),
                imu_measurements=Path("b"),
                camera_timestamps=Path("c"),
                led1_positions=Path("d"),
            ),
            ttl_events=TTLEventsConfig(
                events_file=None,
                beams=[BeamSpec(id=1, emitter=(0.0, 0.0), receiver=(0.1, 0.0))],
            ),
        )


def test_events_file_optional_with_dio_bridge() -> None:
    """``ttl_events.events_file=None`` is accepted when the NWB DIO
    bridge is configured."""

    SessionConfig(
        inputs=InputsConfig(
            format="nwb",
            nwb=NWBConfig(
                nwb_file=Path("session.nwb"),
                dio_to_ttl=NWBDIOToTTLConfig(name_to_source_id={"beam_1": 1}),
            ),
        ),
        ttl_events=TTLEventsConfig(
            events_file=None,
            beams=[BeamSpec(id=1, emitter=(0.0, 0.0), receiver=(0.1, 0.0))],
        ),
    )


def test_ttl_events_required_when_dio_bridge_set() -> None:
    """``ttl_events`` itself must be present when the DIO bridge is
    configured — otherwise the EKF/UKF event channel has no source
    geometry."""

    with pytest.raises(ValidationError, match=r"ttl_events is missing"):
        SessionConfig(
            inputs=InputsConfig(
                format="nwb",
                nwb=NWBConfig(
                    nwb_file=Path("session.nwb"),
                    dio_to_ttl=NWBDIOToTTLConfig(
                        name_to_source_id={"beam_1": 1},
                    ),
                ),
            ),
            ttl_events=None,
        )


def test_name_to_source_id_unknown_id_rejected() -> None:
    """Every value in ``name_to_source_id`` must be a known TTL
    source id — else the loader hits a KeyError far from the YAML."""

    with pytest.raises(ValidationError, match=r"name_to_source_id"):
        SessionConfig(
            inputs=InputsConfig(
                format="nwb",
                nwb=NWBConfig(
                    nwb_file=Path("session.nwb"),
                    dio_to_ttl=NWBDIOToTTLConfig(
                        name_to_source_id={"beam_1": 99},
                    ),
                ),
            ),
            ttl_events=TTLEventsConfig(
                events_file=None,
                beams=[BeamSpec(id=1, emitter=(0.0, 0.0), receiver=(0.1, 0.0))],
            ),
        )


# ---------------------------------------------------------------------
# Container API: from_behavioral_events.
# ---------------------------------------------------------------------


def test_from_behavioral_events_accepts_container_and_dict_shape(
    tmp_path: Path,
) -> None:
    """``BehavioralEvents`` (path-loader source) and ``dict[str,
    TimeSeries]`` (Spyglass-style ``make()`` shape) produce
    bit-identical event tables."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path)
    dio_cfg = NWBDIOToTTLConfig(name_to_source_id={"beam_1": 1, "zone_a": 2})

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        events_container = nwbfile.processing["behavior"].data_interfaces[
            "behavioral_events"
        ]
        # Container-shape input.
        t1, sid1, edge1 = from_behavioral_events(events_container, dio_cfg)
        # Dict-shape input — same data, different surface.
        ts_dict = dict(events_container.time_series)
        t2, sid2, edge2 = from_behavioral_events(ts_dict, dio_cfg)

    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(sid1, sid2)
    np.testing.assert_array_equal(edge1, edge2)


def test_from_behavioral_events_eager_after_io_close(tmp_path: Path) -> None:
    """After ``from_behavioral_events`` returns, the underlying
    NWBHDF5IO can be closed and the arrays remain readable."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path)
    dio_cfg = NWBDIOToTTLConfig(name_to_source_id={"beam_1": 1, "zone_a": 2})

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        events = nwbfile.processing["behavior"].data_interfaces["behavioral_events"]
        t_evt, source_id, edge = from_behavioral_events(events, dio_cfg)
    # IO closed — touching the arrays must still work.
    assert t_evt.size > 0
    assert source_id.size == t_evt.size
    assert edge.size == t_evt.size


def test_from_behavioral_events_drops_initial_level_and_decodes_edges(
    tmp_path: Path,
) -> None:
    """The first sample (initial level) is dropped; remaining ``int8``
    0/1 values decode to ``edge`` per the EDGE_NAME_TO_INT mapping
    (0=fall, 1=rise)."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path, dio_event_names=("beam_1",))
    dio_cfg = NWBDIOToTTLConfig(name_to_source_id={"beam_1": 1})

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        events = nwbfile.processing["behavior"].data_interfaces["behavioral_events"]
        t_evt, source_id, edge = from_behavioral_events(events, dio_cfg)

    # Fixture wrote [0, 1, 0, 1, 0, 1] for beam_1 — drop first sample
    # → 5 transitions [1, 0, 1, 0, 1].
    assert t_evt.shape == (5,)
    np.testing.assert_array_equal(edge, np.array([1, 0, 1, 0, 1], dtype=int))
    np.testing.assert_array_equal(source_id, np.full(5, 1, dtype=np.int64))


# ---------------------------------------------------------------------
# load_session() integration.
# ---------------------------------------------------------------------


def test_load_session_populates_event_arrays(tmp_path: Path) -> None:
    """End-to-end: ``inputs.nwb.dio_to_ttl`` configured + matching
    ``ttl_events`` geometry → ``session.event_indices_per_frame``
    populated, diagnostic surfaces ``event_source='nwb_behavioral_events'``."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path)
    config = _make_dio_session_config(nwb_path)

    session = load_session(config)

    assert session.event_sources != ()
    assert session.event_indices_per_frame is not None
    diag = session.diagnostics["ttl_events"]
    assert diag["event_source"] == "nwb_behavioral_events"
    assert diag["nwb_dio_present"] is True
    # 5 transitions × 2 channels = 10 events total.
    assert diag["n_events_total"] == 10


def test_load_session_no_dio_config_skips_event_arrays(tmp_path: Path) -> None:
    """NWB without ``dio_to_ttl`` does not populate event arrays even
    when ``behavioral_events`` is present in the file."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path)
    config = SessionConfig(
        inputs=InputsConfig(
            format="nwb",
            nwb=NWBConfig(nwb_file=nwb_path),
        ),
        imu=IMUConfig(run_calibration=False),
        camera=CameraConfig(meters_per_pixel=0.01),
        filter=FilterConfig(state_mode="vision_only"),
    )

    session = load_session(config)

    assert session.event_sources == ()
    assert session.event_indices_per_frame is None


def test_load_session_dio_configured_but_no_behavioral_events_raises(
    tmp_path: Path,
) -> None:
    """``dio_to_ttl`` set but the NWB has no ``behavioral_events``
    container raises with a clear message."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path, include_dio=False)
    config = _make_dio_session_config(nwb_path)

    with pytest.raises(ValueError, match=r"behavioral_events"):
        load_session(config)


def test_from_behavioral_events_unknown_name_raises(tmp_path: Path) -> None:
    """A ``name_to_source_id`` entry pointing at a non-existent
    TimeSeries raises with the available names listed."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path)
    bad_dio = NWBDIOToTTLConfig(name_to_source_id={"nonexistent_dio": 1, "beam_1": 1})

    with (
        pytest.raises(ValueError, match=r"nonexistent_dio"),
        pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io,
    ):
        nwbfile = io.read()
        events = nwbfile.processing["behavior"].data_interfaces["behavioral_events"]
        from_behavioral_events(events, bad_dio)


def test_parquet_wins_over_nwb_dio(tmp_path: Path) -> None:
    """Both ``events_file`` (parquet) and ``dio_to_ttl`` configured →
    parquet takes precedence; NWB DIO is read but ignored, both
    paths recorded in diagnostics."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path)
    parquet_path = tmp_path / "events.parquet"
    _write_parquet_events(parquet_path)
    config = _make_dio_session_config(nwb_path, parquet_events_file=parquet_path)

    session = load_session(config)

    diag = session.diagnostics["ttl_events"]
    assert diag["event_source"] == "parquet"
    assert diag["nwb_dio_present"] is True  # NWB DIO was loaded but ignored
    # Parquet has 4 rise events; NWB had 10 (5×2). The kept count
    # confirms we read from parquet, not NWB.
    assert diag["n_events_total"] == 4


# ---------------------------------------------------------------------
# Diagnostics: dio_event_count surfaces.
# ---------------------------------------------------------------------


def test_load_nwb_session_extras_dio_events_populated(tmp_path: Path) -> None:
    """When ``cfg.dio_to_ttl`` is set, ``load_nwb_session`` populates
    ``extras.dio_events`` with the (t_evt, source_id, edge) triple."""

    nwb_path = _make_nwb_with_position_and_dio(tmp_path)
    cfg = NWBConfig(
        nwb_file=nwb_path,
        dio_to_ttl=NWBDIOToTTLConfig(name_to_source_id={"beam_1": 1, "zone_a": 2}),
    )

    _, extras = load_nwb_session(cfg)

    assert extras.dio_events is not None
    t_evt, _source_id, _edge = extras.dio_events
    assert t_evt.size == 10  # 5 transitions × 2 channels
    assert extras.diagnostics["dio_source"] == "nwb_behavioral_events"
    assert extras.diagnostics["dio_event_count"] == 10

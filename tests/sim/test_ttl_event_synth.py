"""Unit tests for the synthetic TTL event helper."""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.config.schemas import (
    BeamSpec,
    RFIDReaderSpec,
    ZoneTriggerSpec,
)
from trodestrack.sim.ttl_events import events_from_trajectory


def test_beam_crossing_detected_once_per_pass():
    """Crossing a beam once produces one active edge plus one reset edge."""
    t = np.linspace(0.0, 1.0, 100)
    # Trajectory along x-axis from -0.5 to +0.5, y=0.
    xy = np.column_stack([np.linspace(-0.5, 0.5, 100), np.zeros(100)])
    beam = BeamSpec(id=1, emitter=(0.0, -0.1), receiver=(0.0, 0.1), sigma_perp_m=0.005)
    events = events_from_trajectory(t, xy, beams=[beam])
    assert len(events) == 2
    assert events[0].source_id == 1
    assert events[0].edge == "fall"
    assert events[1].source_id == 1
    assert events[1].edge == "rise"
    # Crossing should be at t ≈ 0.5 (midpoint of trajectory).
    assert 0.45 <= events[0].time <= 0.55
    assert events[1].time > events[0].time


def test_zone_events_emitted_on_entry_and_exit():
    """Single trajectory pass through a zone produces active and reset edges."""
    t = np.linspace(0.0, 1.0, 100)
    xy = np.column_stack([np.linspace(-0.5, 0.5, 100), np.zeros(100)])
    zone = ZoneTriggerSpec(id=2, center=(0.0, 0.0), sigma_m=0.02)
    events = events_from_trajectory(
        t, xy, zone_triggers=[zone], zone_trigger_radius_m=0.05
    )
    assert len(events) == 2
    assert events[0].source_id == 2
    assert events[0].edge == "rise"
    assert events[1].source_id == 2
    assert events[1].edge == "fall"
    assert events[0].time < 0.5 < events[1].time


def test_rfid_events_emitted_on_entry_and_exit():
    t = np.linspace(0.0, 1.0, 100)
    xy = np.column_stack([np.linspace(-0.5, 0.5, 100), np.zeros(100)])
    reader = RFIDReaderSpec(id=3, center=(0.0, 0.0), effective_radius_m=0.05)
    events = events_from_trajectory(t, xy, rfid_readers=[reader])
    assert len(events) == 2
    assert events[0].source_id == 3
    assert events[0].edge == "rise"
    assert events[1].source_id == 3
    assert events[1].edge == "fall"


def test_events_sorted_by_time():
    t = np.linspace(0.0, 1.0, 100)
    xy = np.column_stack([np.linspace(-0.5, 0.5, 100), np.zeros(100)])
    beam_left = BeamSpec(
        id=1, emitter=(-0.2, -0.1), receiver=(-0.2, 0.1), sigma_perp_m=0.005
    )
    beam_right = BeamSpec(
        id=2, emitter=(0.2, -0.1), receiver=(0.2, 0.1), sigma_perp_m=0.005
    )
    events = events_from_trajectory(t, xy, beams=[beam_right, beam_left])
    active_events = [event for event in events if event.edge == "fall"]
    assert len(events) == 4
    assert len(active_events) == 2
    # Times monotonically increasing.
    assert active_events[0].time < active_events[1].time
    # Left beam fires first.
    assert active_events[0].source_id == 1


def test_zone_entry_exit_times_match_analytic_radius_crossing():
    """Linear interpolation of the radius crossing should land within one
    sample interval of the analytic entry/exit times for a known trajectory.
    """
    # Trajectory: x(t) = -1 + 2t, y = 0. Speed 2 m/s along +x.
    n = 1001
    t = np.linspace(0.0, 1.0, n)
    xy = np.column_stack([-1.0 + 2.0 * t, np.zeros(n)])
    sample_dt = float(np.median(np.diff(t)))

    radius = 0.3
    # Analytic entry/exit: x = ±radius → t = (1 ± radius) / 2.
    expected_entry = (1.0 - radius) / 2.0  # 0.35 s
    expected_exit = (1.0 + radius) / 2.0  # 0.65 s

    zone = ZoneTriggerSpec(id=10, center=(0.0, 0.0), sigma_m=0.02)
    events = events_from_trajectory(
        t, xy, zone_triggers=[zone], zone_trigger_radius_m=radius
    )
    assert [e.edge for e in events] == ["rise", "fall"]
    # Linear interp should land within ``sample_dt`` of the true crossing.
    assert abs(events[0].time - expected_entry) <= sample_dt
    assert abs(events[1].time - expected_exit) <= sample_dt


def test_beam_reset_delay_honored():
    """The ``beam_reset_delay_s`` argument controls reset-edge offset."""
    n = 200
    t = np.linspace(0.0, 1.0, n)
    xy = np.column_stack([np.linspace(-0.5, 0.5, n), np.zeros(n)])
    beam = BeamSpec(id=1, emitter=(0.0, -0.1), receiver=(0.0, 0.1), sigma_perp_m=0.005)
    custom_delay = 0.1
    events = events_from_trajectory(
        t, xy, beams=[beam], beam_reset_delay_s=custom_delay
    )
    assert len(events) == 2
    active, reset = events
    assert active.edge == "fall"
    assert reset.edge == "rise"
    assert abs((reset.time - active.time) - custom_delay) < 1e-12


def test_beam_reset_after_session_end_is_dropped():
    """A reset edge whose time exceeds ``t[-1]`` must not be emitted."""
    n = 100
    t = np.linspace(0.0, 1.0, n)
    # Trajectory crosses the beam at the very last sample.
    xy = np.column_stack([np.linspace(-0.5, 0.0, n), np.zeros(n)])
    beam = BeamSpec(id=1, emitter=(0.0, -0.1), receiver=(0.0, 0.1), sigma_perp_m=0.005)
    # Reset delay larger than remaining session time.
    events = events_from_trajectory(t, xy, beams=[beam], beam_reset_delay_s=10.0)
    assert len(events) == 1
    assert events[0].edge == "fall"


def test_beam_reset_delay_must_be_positive():
    t = np.linspace(0.0, 1.0, 10)
    xy = np.column_stack([np.linspace(-0.5, 0.5, 10), np.zeros(10)])
    with pytest.raises(ValueError, match="beam_reset_delay_s"):
        events_from_trajectory(t, xy, beam_reset_delay_s=0.0)

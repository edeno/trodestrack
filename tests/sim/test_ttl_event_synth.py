"""Unit tests for the synthetic TTL event helper (Milestone 3)."""

from __future__ import annotations

import numpy as np

from trodestrack.config.schemas import (
    BeamSpec,
    RFIDReaderSpec,
    ZoneTriggerSpec,
)
from trodestrack.sim.ttl_events import events_from_trajectory


def test_beam_crossing_detected_once_per_pass():
    """Crossing a beam once produces exactly one event."""
    t = np.linspace(0.0, 1.0, 100)
    # Trajectory along x-axis from -0.5 to +0.5, y=0.
    xy = np.column_stack([np.linspace(-0.5, 0.5, 100), np.zeros(100)])
    beam = BeamSpec(id=1, emitter=(0.0, -0.1), receiver=(0.0, 0.1), sigma_perp_m=0.005)
    events = events_from_trajectory(t, xy, beams=[beam])
    assert len(events) == 1
    assert events[0].source_id == 1
    assert events[0].edge == "fall"
    # Crossing should be at t ≈ 0.5 (midpoint of trajectory).
    assert 0.45 <= events[0].time <= 0.55


def test_zone_event_emitted_on_entry():
    """Single trajectory entry into a zone produces one event."""
    t = np.linspace(0.0, 1.0, 100)
    xy = np.column_stack([np.linspace(-0.5, 0.5, 100), np.zeros(100)])
    zone = ZoneTriggerSpec(id=2, center=(0.0, 0.0), sigma_m=0.02)
    events = events_from_trajectory(
        t, xy, zone_triggers=[zone], zone_trigger_radius_m=0.05
    )
    assert len(events) == 1
    assert events[0].source_id == 2
    assert events[0].edge == "rise"


def test_rfid_event_emitted_on_entry():
    t = np.linspace(0.0, 1.0, 100)
    xy = np.column_stack([np.linspace(-0.5, 0.5, 100), np.zeros(100)])
    reader = RFIDReaderSpec(id=3, center=(0.0, 0.0), effective_radius_m=0.05)
    events = events_from_trajectory(t, xy, rfid_readers=[reader])
    assert len(events) == 1
    assert events[0].source_id == 3


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
    assert len(events) == 2
    # Times monotonically increasing.
    assert events[0].time < events[1].time
    # Left beam fires first.
    assert events[0].source_id == 1

"""Synthetic TTL event generation from a sampled trajectory.

Used by integration tests to drive the EKF event channel without a real
parquet file. The helpers walk a (t, x, y) trajectory and emit:

- **Beam break events** when the trajectory segment between two samples
  crosses a beam (detected via the segment-segment intersection test).
- **Zone trigger events** when the trajectory enters a zone disc of
  radius ``trigger_radius_m`` from outside.
- **RFID detection events** when the trajectory enters the reader's
  ``effective_radius_m``.

For simplicity, every event is emitted as a single ``"fall"`` (beam) or
``"rise"`` (zone / reader) edge — matching the default ``active_edge`` of
the corresponding spec class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trodestrack.config.schemas import BeamSpec, RFIDReaderSpec, ZoneTriggerSpec


@dataclass(frozen=True)
class SyntheticEvent:
    time: float
    source_id: int
    edge: str


def _segments_cross(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> tuple[bool, float]:
    """Test whether segments (p1->p2) and (q1->q2) cross.

    Returns ``(crossed, t)`` where ``t in [0, 1]`` is the parameter along
    p1->p2 at the crossing, or ``(False, 0.0)`` if they do not cross.
    """
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return False, 0.0
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, float(t)
    return False, 0.0


def _emit_beam_events(
    t: np.ndarray, xy: np.ndarray, beam: BeamSpec
) -> list[SyntheticEvent]:
    """Emit one beam-break event per trajectory segment that crosses the beam."""
    events: list[SyntheticEvent] = []
    emitter = np.asarray(beam.emitter, dtype=float)
    receiver = np.asarray(beam.receiver, dtype=float)
    for i in range(len(t) - 1):
        crossed, t_local = _segments_cross(xy[i], xy[i + 1], emitter, receiver)
        if crossed:
            events.append(
                SyntheticEvent(
                    time=float(t[i] + t_local * (t[i + 1] - t[i])),
                    source_id=beam.id,
                    edge=beam.active_edge,
                )
            )
    return events


def _emit_radius_events(
    t: np.ndarray,
    xy: np.ndarray,
    *,
    source_id: int,
    edge: str,
    center: tuple[float, float],
    radius: float,
) -> list[SyntheticEvent]:
    """Emit one event per trajectory entry into a disc around ``center``."""
    events: list[SyntheticEvent] = []
    centre_arr = np.asarray(center, dtype=float)
    dist = np.linalg.norm(xy - centre_arr, axis=1)
    inside = dist <= radius
    # Trigger on rising edge (False -> True) of the inside indicator.
    enters = np.where(inside[1:] & ~inside[:-1])[0]
    for i in enters:
        events.append(
            SyntheticEvent(
                time=float(t[i + 1]),
                source_id=source_id,
                edge=edge,
            )
        )
    return events


def events_from_trajectory(
    t: np.ndarray,
    xy: np.ndarray,
    *,
    beams: list[BeamSpec] | None = None,
    zone_triggers: list[ZoneTriggerSpec] | None = None,
    rfid_readers: list[RFIDReaderSpec] | None = None,
    zone_trigger_radius_m: float | None = None,
) -> list[SyntheticEvent]:
    """Synthesize TTL events for a sampled (t, xy) trajectory.

    Parameters
    ----------
    t : np.ndarray, shape (n,)
        Sample times in seconds.
    xy : np.ndarray, shape (n, 2)
        World-frame positions in meters.
    beams, zone_triggers, rfid_readers : list of specs, optional
        Configured event sources.
    zone_trigger_radius_m : float | None
        Activation radius for zone triggers (defaults to ``zone.sigma_m``
        if ``None``). Lets sim activation differ from measurement noise.
    """
    events: list[SyntheticEvent] = []
    for beam in beams or []:
        events.extend(_emit_beam_events(t, xy, beam))
    for zone in zone_triggers or []:
        radius = (
            zone_trigger_radius_m if zone_trigger_radius_m is not None else zone.sigma_m
        )
        events.extend(
            _emit_radius_events(
                t,
                xy,
                source_id=zone.id,
                edge=zone.active_edge,
                center=zone.center,
                radius=radius,
            )
        )
    for reader in rfid_readers or []:
        events.extend(
            _emit_radius_events(
                t,
                xy,
                source_id=reader.id,
                edge=reader.active_edge,
                center=reader.center,
                radius=reader.effective_radius_m,
            )
        )
    events.sort(key=lambda e: e.time)
    return events

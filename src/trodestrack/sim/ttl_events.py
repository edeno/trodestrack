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


def _emit_beam_events(
    t: np.ndarray, xy: np.ndarray, beam: BeamSpec
) -> list[SyntheticEvent]:
    """Emit one beam-break event per trajectory crossing of the beam.

    Crossings are detected as sign changes of the signed cross product
    ``(receiver - emitter) × (sample - emitter)``. Half-open sign change
    (positive → non-positive OR negative → non-negative) counts samples
    that land exactly on the line once, on the approaching segment.
    """
    events: list[SyntheticEvent] = []
    emitter = np.asarray(beam.emitter, dtype=float)
    receiver = np.asarray(beam.receiver, dtype=float)
    s = receiver - emitter
    seg_len_sq = float(s @ s)
    if seg_len_sq <= 0.0:
        return events
    delta = xy - emitter
    side = s[0] * delta[:, 1] - s[1] * delta[:, 0]

    for i in range(len(t) - 1):
        s0, s1 = side[i], side[i + 1]
        crossed = (s0 > 0 and s1 <= 0) or (s0 < 0 and s1 >= 0)
        if not crossed:
            continue
        denom = s0 - s1
        t_local = float(s0 / denom) if denom != 0.0 else 0.0
        crossing_xy = xy[i] + t_local * (xy[i + 1] - xy[i])
        u = float((crossing_xy - emitter) @ s) / seg_len_sq
        if 0.0 <= u <= 1.0:
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

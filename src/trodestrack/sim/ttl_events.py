"""Synthetic TTL event generation from a sampled trajectory.

Used by integration tests to drive the EKF event channel without a real
parquet file. The helpers walk a (t, x, y) trajectory and emit:

- **Beam break events** when the trajectory segment between two samples
  crosses a beam (detected via the segment-segment intersection test). A
  short reset edge is emitted after the break to mimic a digital pulse.
- **Zone trigger events** when the trajectory enters and exits a zone disc
  of radius ``trigger_radius_m``.
- **RFID detection events** when the trajectory enters and exits the
  reader's ``effective_radius_m``.

The emitted stream is Trodes-like DIO state changes: the configured
``active_edge`` marks entry / trigger onset, and the opposite edge marks
exit / reset. Downstream ingest filters to each source's configured active
edge before constructing Kalman updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from trodestrack.config.schemas import BeamSpec, RFIDReaderSpec, ZoneTriggerSpec

EdgeName = Literal["rise", "fall"]


@dataclass(frozen=True)
class SyntheticEvent:
    time: float
    source_id: int
    edge: EdgeName


def _opposite_edge(edge: EdgeName) -> EdgeName:
    return "rise" if edge == "fall" else "fall"


def _emit_beam_events(
    t: np.ndarray,
    xy: np.ndarray,
    beam: BeamSpec,
    *,
    reset_delay_s: float,
) -> list[SyntheticEvent]:
    """Emit active/reset edge pairs for each beam crossing.

    Crossings are detected as sign changes of the signed cross product
    ``(receiver - emitter) × (sample - emitter)``. Half-open sign change
    (positive → non-positive OR negative → non-negative) counts samples
    that land exactly on the line once, on the approaching segment. Since
    beam geometry is represented as a line segment, reset timing is modeled
    as a short fixed pulse after the active edge.
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
            event_time = float(t[i] + t_local * (t[i + 1] - t[i]))
            events.append(
                SyntheticEvent(
                    time=event_time,
                    source_id=beam.id,
                    edge=beam.active_edge,
                )
            )
            reset_time = event_time + reset_delay_s
            if reset_time <= t[-1]:
                events.append(
                    SyntheticEvent(
                        time=reset_time,
                        source_id=beam.id,
                        edge=_opposite_edge(beam.active_edge),
                    )
                )
    return events


def _emit_radius_events(
    t: np.ndarray,
    xy: np.ndarray,
    *,
    source_id: int,
    active_edge: EdgeName,
    center: tuple[float, float],
    radius: float,
) -> list[SyntheticEvent]:
    """Emit active/reset edge pairs for entries/exits of a disc."""
    events: list[SyntheticEvent] = []
    centre_arr = np.asarray(center, dtype=float)
    dist = np.linalg.norm(xy - centre_arr, axis=1)
    inside = dist <= radius

    for i in np.where(inside[1:] != inside[:-1])[0]:
        d0 = float(dist[i])
        d1 = float(dist[i + 1])
        denom = d1 - d0
        alpha = float((radius - d0) / denom) if denom != 0.0 else 1.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        edge = active_edge if inside[i + 1] else _opposite_edge(active_edge)
        events.append(
            SyntheticEvent(
                time=float(t[i] + alpha * (t[i + 1] - t[i])),
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
    beam_reset_delay_s: float | None = None,
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
    beam_reset_delay_s : float | None
        Delay between a beam's active edge and reset edge. Defaults to one
        median trajectory sample interval, giving a short pulse without
        requiring a rat-body-width model.
    """
    if t.shape[0] < 2:
        return []
    reset_delay_s = (
        float(beam_reset_delay_s)
        if beam_reset_delay_s is not None
        else float(np.median(np.diff(t)))
    )
    if reset_delay_s <= 0.0:
        raise ValueError("beam_reset_delay_s must be positive.")

    events: list[SyntheticEvent] = []
    for beam in beams or []:
        events.extend(_emit_beam_events(t, xy, beam, reset_delay_s=reset_delay_s))
    for zone in zone_triggers or []:
        radius = (
            zone_trigger_radius_m if zone_trigger_radius_m is not None else zone.sigma_m
        )
        events.extend(
            _emit_radius_events(
                t,
                xy,
                source_id=zone.id,
                active_edge=zone.active_edge,
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
                active_edge=reader.active_edge,
                center=reader.center,
                radius=reader.effective_radius_m,
            )
        )
    events.sort(key=lambda e: e.time)
    return events

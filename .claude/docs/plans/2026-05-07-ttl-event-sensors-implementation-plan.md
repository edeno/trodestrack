# TTL Event Sensors (Beam Break, Zone Trigger, RFID) Implementation Plan

## Status

Not started. The original incremental refactor (`incremental_refactor_plan.md`)
flagged "TTL/RFID Event Sensors" as a deferred milestone with separate
`ttl_zone.py` / `rfid_zone.py` stubs. This plan supersedes that note —
beam break, TTL zone triggers, and RFID readers all share the same
underlying signal (a TTL pulse from a Trodes DIO channel). Zone triggers
and RFID readers are point-like 2D position fixes; beam breaks are line
constraints perpendicular to the beam. A single `EventLocationModel`
can still cover all three by supporting both point and line event
sources under one ingest and EKF update path.

## Goals

- **One measurement model, three user-facing source types.**
  `EventLocationModel` consumes resolved point or line geometry; the
  user picks between `BeamSpec` (computes line normal / offset from
  emitter+receiver geometry), `ZoneTriggerSpec` (anchor + isotropic σ),
  and `RFIDReaderSpec` (anchor + effective radius). All share the same
  ingest pipeline and the same EKF wiring.
- **Shared TTL parquet ingest** from Trodes DIO. Hardware sync is
  free because Trodes DIO timestamps live on the IMU/camera clock.
- **Bound position drift during long camera dropouts.** Beam grids
  reset perpendicular position error to mm-scale on every crossing;
  zone triggers and RFID readers provide absolute 2D fixes when
  the rat is near them.
- **Use the same Kalman-update math as the existing sensor wrappers**
  through a dedicated `update_event_location` helper. The current
  `MeasurementModel` protocol is frame-indexed with fixed `meas_dim`;
  TTL events need per-frame variable source IDs and mixed 1D/2D rows,
  so this plan intentionally uses a custom wrapper rather than claiming
  protocol conformance.

## Non-Goals

- Multi-animal disambiguation (which animal triggered which event).
- Soft / analog signals (occlusion-area, signal-strength). Only
  binary TTL edges are modeled; which edge is active is configured per
  source so beam-break onset and pulse-onset wiring both work.
- Non-Trodes ingest (MED-PC, Arduino, plain CSV). Users on those
  controllers ship a tiny adapter producing the parquet schema
  this plan defines.
- Inter-event IMU pre-integration (event-time precision better than
  the 1/`fs_cam` quantum). Events are bucketed into camera frames.
- 3D event sources (e.g., RFID antennas at known z). Default to
  2D (`layout.pos_idx[:2]`); flagged for a future extension if
  hardware ever exposes it.
- Magnetometer events / non-spatial pulses. Out of scope.

## Background

### Why TTL is the unifying signal

Trodes records TTL rising/falling edges on its DIO channels with the
same hardware clock as the IMU and camera. In a typical setup:

- Each **beam break** unit (IR emitter/receiver pair) is wired to one
  DIO channel; the line goes high when the beam is unbroken, low when
  the rat crosses.
- Each **zone trigger** (nose poke, lever press, opto trigger, gate)
  is wired to one DIO channel; the pulse fires on engagement.
- Each **RFID reader** outputs a TTL pulse on a DIO channel when a
  tag (worn by the rat) enters detection range.

The events parquet schema is therefore agnostic to source type:

```
columns: [time (s), source_id (int), edge ("rise" | "fall")]
```

`source_id` maps to one of the configured sources in the YAML; the
sensor type is determined by *which* source list the id appears in
(beams, zones, or readers).

### The unifying math

Every TTL event source produces a spatial constraint at the event time.
Zone triggers and RFID readers produce a **2D point measurement**:

```
H = position selector (2, n_state)             # via layout.pos_idx[:2]
z = anchor (2,)                                # world meters
R = 2×2 covariance                             # world-frame covariance
```

Beam breaks produce a **1D line measurement**:

```
h(x) = nᵀ · p - c                              # p = state position[:2]
z = 0
R = σ_perp²
```

where `n` is the unit normal to the beam line and `c = nᵀ · emitter`
(equivalently `nᵀ · receiver`).

The geometry of each source type determines the measurement:

| Source | Measurement | Noise |
| --- | --- | --- |
| **Zone trigger** | 2D point at zone center | `σ_zone² · I` |
| **RFID reader** | 2D point at reader location | `(r_eff / √2)² · I` |
| **Beam break** | 1D perpendicular line constraint | `σ_perp²` |

### Why beam break is a line update, not a midpoint fix

Intuitively a beam break feels like a **1D linear constraint**
(perpendicular distance to the line), and the implementation should
model it that way. A 2D anisotropic measurement anchored at the beam
midpoint is only equivalent in the limiting case where the along-beam
innovation is ignored. With any finite along-beam variance it still
pulls the estimate weakly toward the beam midpoint, which is the wrong
geometry for a long beam.

The beam update should therefore use the scalar measurement:

```
z = 0
h(x) = nᵀ · p - c
H = nᵀ · position_selector
R = σ_perp²
```

If a future implementation wants to express beams as a 2D pseudo-
measurement, it must anchor the along-beam coordinate at the predicted
state projection before the update, not at the fixed midpoint.

## Design Principles

- **Single event update model.** `EventLocationModel.predict /
  jacobian / meas_cov / innovation` operates on resolved point or line
  event sources regardless of source type. It is consumed by
  `update_event_location`, not by the existing fixed-`meas_dim`
  `MeasurementModel` protocol. JIT trace is one shape; debugging is
  one place.
- **Per-source-type spec dataclasses for ergonomics.** Users
  write `BeamSpec(id, emitter, receiver, sigma_perp_m)`,
  `ZoneTriggerSpec(id, center, sigma_m)`,
  `RFIDReaderSpec(id, center, effective_radius_m)`. Each spec has a
  `to_event_source()` method that yields the resolved measurement
  fields the model consumes.
- **Stacked update per camera frame.** All events in
  `[t_cam[k-1], t_cam[k]]` fold into one block update with
  `H ∈ R^{M × n_state}`, where each event contributes either one
  row for a beam line or two rows for a point source, and
  `R = block_diag(R_1, …, R_K)`.
  Mathematically equivalent to sequential application; cleaner JAX
  trace.
- **Padded for `lax.scan`.** Per-frame event index is padded to
  `MAX_EVENTS_PER_FRAME` (default 8) with sentinel `-1`. Because line
  events contribute one H row and point events contribute two, the
  stacked update is built at the worst-case row budget
  `MAX_EVENT_MEAS_ROWS = 2 * MAX_EVENTS_PER_FRAME`. Within the K used
  event slots, line events fill 1 real row + 1 sentinel-padded row,
  point events fill 2 real rows; the remaining `(MAX_EVENTS_PER_FRAME
  − K) * 2` rows are sentinel-padded. All sentinel rows get a large
  `R` so the Kalman gain treats them as no-ops and they contribute
  zero to the innovation log-likelihood.
- **Trodes DIO is the canonical source.** Hardware sync is assumed;
  no separate clock-alignment step.
- **Layout-aware indexing.** `layout.pos_idx[:2]` for all current
  layouts; 3D extension flagged but not in scope.
- **Optional channel.** Absent config = no behavior change. Empty
  events file = short-circuit.
- **Source-id polymorphism with resolved geometry.** Runtime lookup
  resolves `source_id` to a fixed point or line measurement. The
  source-type metadata lives in diagnostics for the visualization
  layer.

## Architecture

### New module — `src/trodestrack/models/sensors/event_location.py`

```python
@dataclass(frozen=True)
class EventLocationSource:
    """Resolved geometry the model consumes per event."""
    source_id: int
    kind: Literal["point", "line"]
    anchor: np.ndarray | None = None      # (2,) point source, world meters
    R: np.ndarray | None = None           # (2, 2) point-source covariance
    normal: np.ndarray | None = None      # (2,) beam-line unit normal
    offset: float | None = None           # c in n.T @ p - c = 0
    sigma_perp_m: float | None = None     # beam-line perpendicular sigma
    label: str | None = None
    source_type: str = "unknown"   # for diagnostics only


class EventLocationModel:
    """2D-position measurement model for TTL event sources.

    All discrete-event spatial sensors (beam break, zone trigger,
    RFID reader) collapse to this model: point sources contribute 2D
    position fixes and beam breaks contribute 1D line constraints. The
    source-type distinction lives in the resolved geometry and in
    diagnostics; the EKF wiring is shared.
    """

    def __init__(
        self,
        sources: tuple[EventLocationSource, ...],
        layout: StateLayout,
        dtype=jnp.float32,
    ): ...

    @property
    def max_meas_dim_per_event(self) -> int:
        # Line sources use one row; point sources use two rows.
        # The EKF wrapper pads to MAX_EVENTS_PER_FRAME * 2 rows.
        return 2

    def predict(self, state_mean, *, source_indices) -> jnp.ndarray: ...
    def jacobian(self, state_mean, *, source_indices) -> jnp.ndarray: ...
    def meas_cov(self, *, source_indices) -> jnp.ndarray: ...
    def innovation(self, *, source_indices, meas_pred) -> jnp.ndarray: ...
```

`source_indices` is a 1D array of resolved source IDs (or `-1`
sentinel) active in the current camera frame. Padded sentinel rows
get a large `R` so the Kalman update treats them as no-ops.

### Schema additions — `src/trodestrack/config/schemas.py`

```python
class BeamSpec(BaseModel):
    """A beam-break source. Computes a line normal and offset from
    emitter/receiver geometry."""
    id: int
    emitter: tuple[float, float]
    receiver: tuple[float, float]
    sigma_perp_m: float = Field(default=0.005, gt=0.0)
    active_edge: Literal["rise", "fall"] = "fall"
    """Edge that means the beam is crossed. Many beam-break circuits
    are high when unbroken and go low on crossing, so the default is
    falling edge. Override to ``"rise"`` for inverted wiring."""
    label: str | None = None

    def to_event_source(self) -> EventLocationSource: ...


class ZoneTriggerSpec(BaseModel):
    """A point-trigger source (nose poke, lever press, gate)."""
    id: int
    center: tuple[float, float]
    sigma_m: float = Field(default=0.02, gt=0.0)
    active_edge: Literal["rise", "fall"] = "rise"
    label: str | None = None

    def to_event_source(self) -> EventLocationSource:
        anchor = np.array(self.center)
        R = (self.sigma_m ** 2) * np.eye(2)
        return EventLocationSource(
            source_id=self.id,
            kind="point",
            anchor=anchor,
            R=R,
            label=self.label,
            source_type="zone",
        )


class RFIDReaderSpec(BaseModel):
    """An RFID reader source. ``effective_radius_m`` is the
    detection range; treated as the √2·σ of a 2D isotropic
    Gaussian fit to a uniform disc."""
    id: int
    center: tuple[float, float]
    effective_radius_m: float = Field(default=0.05, gt=0.0)
    active_edge: Literal["rise", "fall"] = "rise"
    label: str | None = None

    def to_event_source(self) -> EventLocationSource:
        anchor = np.array(self.center)
        sigma = self.effective_radius_m / np.sqrt(2.0)
        R = (sigma ** 2) * np.eye(2)
        return EventLocationSource(
            source_id=self.id,
            kind="point",
            anchor=anchor,
            R=R,
            label=self.label,
            source_type="rfid",
        )


class TTLEventsConfig(BaseModel):
    events_file: Path
    beams: list[BeamSpec] = Field(default_factory=list)
    zone_triggers: list[ZoneTriggerSpec] = Field(default_factory=list)
    rfid_readers: list[RFIDReaderSpec] = Field(default_factory=list)
    max_events_per_frame: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _validate_unique_ids(self): ...    # ids unique across all source-type lists
```

`SessionConfig.ttl_events: TTLEventsConfig | None = None`.

### New ingest helper — `src/trodestrack/io/ttl_events.py`

```python
def load_ttl_events(events_file: Path) -> tuple[
    np.ndarray,    # t_evt (n_evt,)
    np.ndarray,    # source_id (n_evt,)
    np.ndarray,    # edge (n_evt,) — 1=rise, 0=fall
]: ...

def per_frame_event_indices(
    t_evt: np.ndarray,
    source_id: np.ndarray,
    edge: np.ndarray,
    t_cam: np.ndarray,
    *,
    source_active_edges: Mapping[int, int],     # 1=rise, 0=fall per source
    max_events_per_frame: int,
) -> np.ndarray:
    """Returns ``(n_cam, max_events_per_frame)`` array of source
    ids, padded with -1. Validates that no camera frame exceeded
    the pad limit (raises with a clear message if it did)."""
```

The same helper handles beam, zone, and RFID events — they're
distinguished only by which source-type list their `source_id`
appears in. The schema's `_validate_unique_ids` guarantees no
`source_id` appears in more than one list.

### Loader wiring — `src/trodestrack/io/session.py`

`load_session` resolves `ttl_events` (when configured) into:

- A flat list of `EventLocationSource` objects (one per
  beam / zone / reader), built by calling each Spec's
  `to_event_source()`.
- A per-frame source-id index array.

Stored on `PreparedSession` as `event_sources: tuple[EventLocationSource, ...]`
and `event_indices_per_frame: np.ndarray | None`.

### EKF wiring — `src/trodestrack/models/ekf.py`

In the per-camera-frame `step` body, after the LED / heading /
ZUPT updates:

```python
event_source_ids = event_indices_per_frame[t_idx]   # padded (MAX_EVENTS,)
state_filt, log_lik_event = update_event_location(
    state_after_zupt,
    event_model,
    event_source_ids,
    valid_mask=event_source_ids >= 0,
)
log_lik_k = log_lik_pos + log_lik_heading + log_lik_zupt + log_lik_event
```

`update_event_location` short-circuits via
`lax.cond(jnp.any(valid_mask), do_update, no_update)` so frames with
no events pay zero update cost.

### Simulator extension — `src/trodestrack/sim/rat_imu.py`

`RatIMUSimConfig` gains an optional `ttl_event_geometry` block
mirroring the YAML schema. The simulator emits synthetic events
when the trajectory crosses any configured beam, enters any zone
(within `σ_zone`), or enters any reader's effective radius.

### Diagnostic video — `src/trodestrack/viz/`

- New `EventArrayArtist` draws all configured sources on the arena
  view: beams as line segments, zones as small circles, readers as
  larger circles. Lights up the source's marker for ~3 frames after
  a trigger.
- Timeline panel: stacked event tracks (one row per source type)
  showing trigger times.

## Milestones

### Milestone 1 — Shared ingest infrastructure

- TTL events parquet loader (`io/ttl_events.py`).
- Per-frame source-id index builder with padding.
- `TTLEventsConfig` + the three Spec dataclasses with
  `to_event_source()` math.
- Schema validation: unique IDs across all source types; valid
  edges; pad-limit enforced at load time.
- Tests for parquet → per-frame indices; schema unique-id
  enforcement; per-spec `to_event_source()` math (point-source anchor
  and R; beam-source normal, offset, `sigma_perp_m`, and active edge).

**Exit criteria:** all three Spec classes can produce
`EventLocationSource` objects from YAML; `tests/io/test_ttl_events.py`
green.

### Milestone 2 — `EventLocationModel` core

- Implement `EventLocationModel` with `predict / jacobian / meas_cov
  / innovation`.
- Stacked-event update: K simultaneous events fold into one padded
  `(MAX_EVENT_MEAS_ROWS, n_state)` block update.
- Padded-sentinel handling: `source_id == -1` rows get large `R` and
  are masked out of the innovation log-likelihood.
- Unit tests:
  - Predict point sources on / off the anchor and beam sources on /
    off the line.
  - Stacked H shape with one row for beam lines and two rows for point
    sources.
  - Padded sentinels produce no posterior change.
  - **Beam-break geometry**: the scalar line update leaves the
    along-beam coordinate unchanged except through prior covariance
    coupling and matches a manual 1D-perpendicular Kalman update to
    numerical tolerance.
  - JIT-compatibility smoke.

**Exit criteria:** model is callable in isolation; beam-break geometry
test green to numerical tolerance.

### Milestone 3 — Sim extension and synthetic scenario tests

- `RatIMUSimConfig.ttl_event_geometry` emits synthetic beam /
  zone / RFID events.
- Property: events monotonic; source IDs valid; pad-limit honored.
- Scenario per source type (separate tests):
  - Beam grid (4×4) reduces position RMSE during 5s camera
    dropout vs no-event baseline.
  - Zone trigger snaps position to feeder location within σ_zone.
  - RFID detection collapses position uncertainty to the reader's
    effective radius.

**Exit criteria:**
`tests/integration/test_ttl_event_sensors_session.py` green; sim
emits per-source-type events that cross the model's update path.

### Milestone 4 — EKF wiring

- `update_event_location` wrapper in `filter_common.py`.
- Wire into `extended_kalman_filter` (and `_3d` if 3D EKF gets
  event support — same model, different `pos_idx` selector).
- Numerical parity test: with `ttl_events` config absent, filter
  output is bitwise identical to current behavior.
- Numerical parity test: empty events file but config present
  produces identical output to "config absent".

**Exit criteria:** parity tests green; full sweep
`pytest -m "not slow and not benchmark"` no regressions.

### Milestone 5 — Diagnostic video and CLI

- `EventArrayArtist` renders all source types with type-distinct
  markers.
- Timeline panel with per-source-type tracks.
- `trodestrack online --config session.yaml` honors the new
  `ttl_events` block; `filter_outputs.npz` bundle gains
  `event_triggers` array.
- Real-data smoke against a hand-constructed geometry + TTL events
  parquet covering at least one beam, one zone, and one reader.

**Exit criteria:** end-to-end run produces the augmented npz
bundle and a video with the events panel; documented in
`docs/getting-started/python-api.md` and `docs/TUNING.md`.

## Validation Matrix

| Test | Layer | Asserts |
| --- | --- | --- |
| `BeamSpec.to_event_source()` | spec | line normal is perpendicular to the beam; offset is consistent with emitter/receiver; active edge defaults to fall |
| `ZoneTriggerSpec.to_event_source()` | spec | anchor = center; R = σ²·I |
| `RFIDReaderSpec.to_event_source()` | spec | anchor = center; R = (r/√2)²·I |
| Schema unique IDs | config | duplicate id across source types → `ValidationError` |
| Predict on point anchor / beam line | model | point-source innovation is zero on anchor; beam-source innovation is zero on line |
| Stacked H shape | model | one row for beam lines, two rows for point sources, padded to static max rows |
| Padded sentinels are no-op | model | posterior unchanged when only `-1` rows present |
| **Beam-break geometry** | model | scalar line update matches manual 1D perpendicular update to numerical tolerance |
| Sim event timing | `sim/rat_imu.py` | events monotonic; source IDs valid |
| Beam-grid scenario | EKF + sim | position RMSE during 5s dropout drops by ≥30% |
| Zone-trigger scenario | EKF + sim | position fix at zone center within σ_zone |
| RFID scenario | EKF + sim | position uncertainty collapses to ≤ effective_radius after detection |
| Numerical parity (no events) | EKF | bitwise identical to current filter |
| Empty-events parity | EKF | bitwise identical to "no ttl_events config" |
| Schema: missing geometry | config | clean `ValidationError` |
| Real-data smoke | end-to-end | NPZ bundle contains `event_triggers`; viz renders |

## Metrics

- **Position RMSE** during 5-second camera dropout with a 4×4 beam
  grid (25 cm spacing): target ≤ 5 cm vs the ~2.3 m mean drift
  baseline (PRD §4.2).
- **Zone-trigger position fix accuracy**: when a known-location
  trigger fires, posterior position mean within `σ_zone` of the
  configured center.
- **Filter throughput penalty**: ≤ 5% wall-clock increase when
  events configured but sparse (<1/sec), measured on the 30-min
  benchmark.
- **Abstraction win**: total LOC for three source types is
  smaller than three independent sensor implementations would be
  (target: ~30% LOC reduction on source, ~40% on tests).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Beam-break line constraint is accidentally implemented as a midpoint point-fix | Unit-test the scalar line Jacobian and assert along-beam coordinate is not pulled toward the midpoint except through prior covariance coupling. |
| Pad-sentinel logic in `lax.scan` triggers shape recompiles | `MAX_EVENTS_PER_FRAME` is a static config arg; load-time assertion that no camera frame exceeds it. |
| Source-id overlap between beams / zones / readers silently maps to wrong source | Schema unique-id validator; loader cross-checks event-file source IDs against configured sources, rejects unknown IDs. |
| Trodes DIO event stream on a clock that doesn't sync to camera/IMU | Documented assumption that DIO is on the Trodes clock; non-Trodes users responsible for pre-aligning timestamps. |
| Hand-survey error in source coordinates (beam posts, zone centers, reader locations) | Diagnostic plot in `qa/`: per-source residual histogram at trigger time. Large mean residual flags miscalibrated geometry. |
| RFID false-positives (stray detections at the reader's edge) | `effective_radius_m` is the user's tuning knob: tighter R inflates posterior less per false-positive. Optional debounce in the loader for known-noisy readers. |
| Multi-beam ghost triggers from one rat pass | Optional per-source debounce window in the loader (drop triggers within `≤debounce_ms` of the previous trigger on the same source). |

## Rollout Strategy

- Ship behind `ttl_events: …` config; absent config = zero
  behavior change.
- Numerical parity test gates the EKF wiring milestone.
- Real-data validation against one user-supplied dataset for each
  source type (one beam-grid session, one zone-trigger session,
  one RFID session) before public announcement.
- Update CLI help text and `docs/getting-started/python-api.md`
  with a worked example covering all three source types in one
  YAML.

## Documentation Updates

- New section in `docs/TUNING.md`: "TTL event sensors for
  dropout-drift mitigation and absolute fixes" with per-source-type
  tuning guidance (`sigma_perp_m`, `sigma_m`, `effective_radius_m`).
- Worked example in `docs/getting-started/python-api.md` showing a
  YAML config with all three source types populated.
- Sample config in `examples/session_with_ttl_events.yaml`.
- Optional appendix in this plan's Background (or a new "Sensor
  Math" doc) explaining why the unified abstraction works
  mathematically — useful onboarding for engineers extending to
  new event types.

## Open Questions

1. Does the Trodes DIO infrastructure already write DIO events to a
   parquet, or do users hand-roll the conversion from a raw DIO
   binary file? Drives whether we ship a one-liner ingester in
   `io/loaders/`.
2. For 14D / 16D experimental layouts, do events update only
   `pos_idx[:2]` (x, y) or all of `pos_idx` (including z)? Default
   to `pos_idx[:2]` since the geometry is physically 2D for all
   three source types.
3. Should `EventLocationModel` integrate with the Mahalanobis
   gating path (gate triggers whose innovation > k·σ)? Probably
   yes for robustness against false positives; flag a follow-up.
4. Future extensibility: are there event sources that *don't* fit
   the point-or-line spatial constraint abstraction? Acoustic /
   ultrasonic distance sensors would be 1D range constraints
   (different geometry);
   accelerometer-based fall detection is event-only with no
   spatial implication. The current abstraction covers the common
   spatial-event cases; range and event-only sensors are separate
   plans if hardware ever exposes them.
5. Should beam sources default to `active_edge="fall"` globally?
   Default yes for common high-when-unbroken beam-break circuits, but
   require per-source override and include a diagnostic count by source
   and edge so inverted wiring is easy to spot.

## Estimated Effort

- ~250 LOC for `EventLocationModel` + ~150 LOC for the three
  Spec dataclasses + ~150 LOC for shared ingest + ~100 LOC for sim
  extension + ~150 LOC for diagnostic viz = **~800 LOC source**.
- ~350 LOC tests.
- ~120 lines docs.
- **1.5–2 weeks** focused work for one engineer familiar with the
  EKF wrapper. The first source type (beam break or zone) takes
  most of the time; the second and third are mostly Pydantic +
  test work because they reuse the model.
- Compare to the original "three independent sensors" approach
  (~430 + 250 + 250 = ~930 LOC source + ~600 LOC tests): unified
  abstraction saves ~30% on source LOC and ~40% on test LOC because
  the model logic is tested once.
- No external dependencies. No calibration tooling beyond manual
  YAML survey of source coordinates.

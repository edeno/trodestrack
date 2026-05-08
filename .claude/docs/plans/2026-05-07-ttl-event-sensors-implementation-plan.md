# TTL Event Sensors (Beam Break, Zone Trigger, RFID) Implementation Plan

## Status

Implemented on `ttl-event-sensors` (Milestones 1–4 + sim/scenario tests).
Both `extended_kalman_filter` and `unscented_kalman_filter` accept the
event channel and share one input-validation contract via
`sensors.event_location.resolve_event_inputs`. The original incremental refactor
(`incremental_refactor_plan.md`) flagged "TTL/RFID Event Sensors" as a
deferred milestone with separate `ttl_zone.py` / `rfid_zone.py` stubs.
This plan supersedes that note — beam break, TTL zone triggers, and RFID
readers all share the same underlying signal (a TTL pulse from a Trodes
DIO channel). All three collapse to a single measurement model: a **2D
point fix at a known anchor with anisotropic 2×2 covariance**. Zone
triggers and RFID readers use isotropic R (the rat is somewhere within
the zone / detection radius). Beam breaks use a Gaussian approximation
to the finite beam segment: a tight perpendicular σ from the IR-beam
width and an along-beam σ from the beam length. This works for very
short beams (≈ isotropic point fix) and makes long beams line-like by
using a much larger along-beam variance, while still remaining a finite
Gaussian measurement rather than an exact line constraint.

## Goals

- **One measurement model, three user-facing source types.**
  `EventLocationModel` consumes resolved 2D anchors and covariances; the
  user picks between `BeamSpec` (computes midpoint anchor + anisotropic
  R from emitter+receiver geometry), `ZoneTriggerSpec` (anchor + isotropic σ),
  and `RFIDReaderSpec` (anchor + effective radius). All share the same
  ingest pipeline and the same EKF / UKF wiring.
- **Many sensors per session.** A session can configure any number of
  beams, zones, and readers. The loader maps user-facing `source_id`
  values from the events parquet into compact source indices for JAX,
  and each camera frame can contain multiple simultaneous events up to
  `max_events_per_frame`.
- **Shared TTL parquet ingest** from Trodes DIO. Hardware sync is
  free because Trodes DIO timestamps live on the IMU/camera clock.
- **Bound position drift during long camera dropouts.** Beam grids
  reset perpendicular position error to mm-scale on every crossing;
  zone triggers and RFID readers provide absolute 2D fixes when
  the rat is near them.
- **Use the same Kalman-update math as the existing sensor wrappers**
  through a dedicated `update_event_location` helper. The current
  `MeasurementModel` protocol is frame-indexed with fixed `meas_dim`;
  TTL events need per-frame variable source IDs and a variable number
  of events, so this plan intentionally uses a custom wrapper rather
  than claiming protocol conformance.

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

Every TTL event source produces a **2D point measurement at a known
anchor with an anisotropic 2×2 measurement covariance**:

```
H = position selector (2, n_state)             # via layout.pos_idx[:2]
z = anchor (2,)                                # world meters
R = R_rot · diag(σ_x², σ_y²) · R_rotᵀ          # event-local σ rotated to world
```

The geometry of each source type determines the anchor and
event-local σ pair:

| Source | `anchor` | `σ_x` (event-local) | `σ_y` (event-local) | `R` orientation |
| --- | --- | --- | --- | --- |
| **Zone trigger** | zone center | `σ_zone` | `σ_zone` | identity (R is isotropic) |
| **RFID reader** | reader location | `r_eff / √2` | `r_eff / √2` | identity (R is isotropic) |
| **Beam break** | beam midpoint | `σ_perp` (across beam) | `max(σ_perp, L/√12)` (along beam) | rotation that maps event-local x to the beam normal |

`L = ‖receiver − emitter‖` is the beam length. `L/√12` is the standard
deviation of a uniform distribution over `[−L/2, +L/2]`, used here as
a moment-matched Gaussian approximation for the unknown along-beam
position at trigger time. The `max(σ_perp, …)` floor handles the
degenerate "very short beam" case (`L → 0`) where we still want
σ_along ≥ σ_perp so R doesn't become singular.

### Why this works for both short and long beams

| Beam length | `σ_perp` | `σ_along = L/√12` | Effective measurement |
| --- | --- | --- | --- |
| 1 cm (a small IR sensor) | 5 mm | 3 mm | Approximately isotropic point fix at midpoint. Reasonable because the rat is constrained to a very small segment when triggered. |
| 10 cm | 5 mm | 29 mm | Mildly anisotropic. Perpendicular constraint dominates but along-beam information is still used. |
| 1 m (corridor IR beam) | 5 mm | 290 mm | Strongly anisotropic. Approaches a perpendicular-line update when the along-beam variance is large relative to the filter's prior uncertainty. |

The key tradeoff: beam breaks are represented by a Gaussian whose first
two moments match a uniform prior over the finite beam segment. This is
not the exact finite-segment likelihood: it still has Gaussian support
outside the segment and can weakly contract or pull along the beam
toward the midpoint. Setting `σ_along = L/√12` makes that along-beam
effect scale with the surveyed beam length instead of with an arbitrary
multiple of `σ_perp`. Short beams behave like point fixes; long beams
become weak along-beam constraints, but implementations and tests
should treat this as an approximation rather than claiming exact
equivalence to a line-only update.

## Design Principles

- **Single event update model.** `EventLocationModel.predict /
  jacobian / meas_cov / innovation` operates on resolved
  `(anchor, R_2x2)` event sources regardless of source type. Every
  event is a 2D point measurement; per-source-type differences live
  entirely in how the anchor and R are computed at config time. It
  is consumed by `update_event_location`, not by the existing
  fixed-`meas_dim` `MeasurementModel` protocol (which is frame-indexed
  and assumes a constant per-frame measurement). JIT trace is one
  shape; debugging is one place.
- **Per-source-type spec dataclasses for ergonomics.** Users
  write `BeamSpec(id, emitter, receiver, sigma_perp_m)`,
  `ZoneTriggerSpec(id, center, sigma_m)`,
  `RFIDReaderSpec(id, center, effective_radius_m)`. Each spec has a
  `to_event_source()` method that resolves the geometry into the
  unified `(anchor, R_2x2)` pair the model consumes.
- **Stacked update per camera frame.** All events in
  `[t_cam[k-1], t_cam[k]]` fold into one block update with
  `H ∈ R^{2K × n_state}` (two rows per event since every event is a
  2D measurement) and `R = block_diag(R_1, …, R_K)`.
  Mathematically equivalent to sequential application; cleaner JAX
  trace.
- **Padded for `lax.scan`.** Per-frame event index is padded to
  `MAX_EVENTS_PER_FRAME` (default 8) with sentinel `-1`. Because every
  event contributes exactly two rows, the stacked update is built at
  fixed shape `(2 · MAX_EVENTS_PER_FRAME, n_state)`. Padded sentinel
  rows are gated three ways: their `H` rows are zero, their innovation
  rows are zero, and their `R` block is identity. The zero `H` is what
  drives the Kalman gain on those rows to exactly zero; the well-
  conditioned identity `R` keeps `psd_solve`'s relative diagonal boost
  stable when the valid-event `R` blocks are tiny (a single large-`R`
  block in the same matrix would inflate the boost and corrupt the
  valid update). The event log-likelihood is also explicitly masked
  over valid rows, so the "events config present but no events" case
  is bitwise identical to the no-events-config case.
- **Trodes DIO is the canonical source.** Hardware sync is assumed;
  no separate clock-alignment step.
- **Layout-aware indexing.** `layout.pos_idx[:2]` for all current
  layouts; 3D extension flagged but not in scope.
- **Optional channel.** Absent config = no behavior change. Empty
  events file = short-circuit.
- **Source-id polymorphism with dense JAX arrays.** User-facing
  `source_id` values can be arbitrary integers in the events parquet,
  so the loader builds a host-side `source_id -> compact_index` map.
  The JAX path only sees dense arrays:
  `event_source_anchors[n_sources, 2]`,
  `event_source_covariances[n_sources, 2, 2]`, and
  `event_indices_per_frame[n_cam, max_events_per_frame]` containing
  compact indices or `-1`. Source-type metadata stays in diagnostics
  for the visualization layer.

## Architecture

### New module — `src/trodestrack/models/sensors/event_location.py`

```python
@dataclass(frozen=True)
class EventLocationSource:
    """Resolved geometry the model consumes per event.

    Every TTL event source is a 2D point measurement at ``anchor``
    with anisotropic 2x2 covariance ``R``. Per-source-type
    distinctions (beam vs zone vs reader) live entirely in how the
    spec class computed these two fields; the model is unaware of
    the original source type.
    """
    source_id: int
    anchor: np.ndarray         # (2,) world meters
    R: np.ndarray              # (2, 2) world-frame covariance, PSD
    label: str | None = None
    source_type: str = "unknown"   # for diagnostics only


class EventLocationModel:
    """2D-position measurement model for TTL event sources.

    All discrete-event spatial sensors (beam break, zone trigger,
    RFID reader) collapse to this model: a 2D position fix at the
    source's anchor with an anisotropic 2x2 covariance. The
    source-type distinction lives entirely in how the user's spec
    classes compute (anchor, R) at config time; the EKF and UKF wiring
    is shared via a single ``update_event_location`` call site.
    """

    def __init__(
        self,
        source_anchors: jnp.ndarray,      # (n_sources, 2)
        source_covariances: jnp.ndarray, # (n_sources, 2, 2)
        layout: StateLayout,
        dtype=jnp.float32,
    ): ...

    @property
    def meas_dim_per_event(self) -> int:
        # Every event contributes a 2D measurement (2 rows in the
        # stacked H). The EKF wrapper pads to MAX_EVENTS_PER_FRAME * 2
        # rows total.
        return 2

    def predict(self, state_mean, *, source_indices) -> jnp.ndarray: ...
    def jacobian(self, state_mean, *, source_indices) -> jnp.ndarray: ...
    def meas_cov(self, *, source_indices) -> jnp.ndarray: ...
    def innovation(self, *, source_indices, meas_pred) -> jnp.ndarray: ...
```

`source_indices` is a 1D array of compact source indices (or `-1`
sentinel) active in the current camera frame. Padded sentinel rows
are gated by zero `H` rows, zero innovation rows, and an identity
`R` block — the zero `H` is what drives the Kalman gain on those
rows to exactly zero. Their log-likelihood contribution is also
masked to zero. (See the Design Principles entry on padding for why
identity-`R` is preferred over a large-`R` "soft gate".)

### Schema additions — `src/trodestrack/config/schemas.py`

```python
class BeamSpec(BaseModel):
    """A beam-break source. Computes anchor (midpoint) and
    anisotropic R from emitter/receiver geometry. ``σ_perp`` is the
    perpendicular noise (typically the IR-beam width); ``σ_along``
    is computed as ``max(σ_perp, ‖receiver − emitter‖ / √12)`` so
    short beams behave like isotropic point fixes and long beams
    behave like weak along-beam, strong perpendicular constraints."""
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
    source_id_to_index: Mapping[int, int],       # external id -> compact index
    max_events_per_frame: int,
) -> np.ndarray:
    """Returns ``(n_cam, max_events_per_frame)`` array of source
    compact indices, padded with -1. Validates unknown source ids and
    that no camera frame exceeded the pad limit (raises with a clear
    message if it did)."""
```

The same helper handles beam, zone, and RFID events — they're
distinguished only by which source-type list their `source_id`
appears in. The schema's `_validate_unique_ids` guarantees no
`source_id` appears in more than one list.

`load_session_config()` must also resolve `ttl_events.events_file`
relative to the YAML file, matching the existing behavior for
`inputs.*` paths and `outputs.output_dir`.

### Loader wiring — `src/trodestrack/io/session.py`

`load_session` resolves `ttl_events` (when configured) into:

- A flat list of `EventLocationSource` objects (one per
  beam / zone / reader), built by calling each Spec's
  `to_event_source()`.
- Dense arrays consumed by JAX:
  `event_source_anchors: np.ndarray` with shape `(n_sources, 2)` and
  `event_source_covariances: np.ndarray` with shape `(n_sources, 2, 2)`.
- A per-frame compact-index array with shape
  `(n_cam, max_events_per_frame)`, where `-1` marks padded slots.
- Diagnostics preserving the user-facing `source_id`, `source_type`,
  and labels for plotting and troubleshooting.

Stored on `PreparedSession` as
`event_sources: tuple[EventLocationSource, ...]`,
`event_source_anchors: np.ndarray | None`,
`event_source_covariances: np.ndarray | None`, and
`event_indices_per_frame: np.ndarray | None`.

### Filter wiring — `src/trodestrack/models/ekf.py` and `src/trodestrack/models/ukf.py`

Add the same three optional event arguments to both public APIs and
pass dense arrays into the jitted core. The validation logic lives in
``sensors.event_location.resolve_event_inputs`` and is shared between
the two wrappers via a ``func_name`` argument that names the calling
filter in error messages.

```python
def extended_kalman_filter(
    ...,
    conf_cam: np.ndarray | None = None,
    event_source_anchors: np.ndarray | None = None,
    event_source_covariances: np.ndarray | None = None,
    event_indices_per_frame: np.ndarray | None = None,
) -> EKFResult: ...

def unscented_kalman_filter(
    ...,
    conf_cam: np.ndarray | None = None,
    event_source_anchors: np.ndarray | None = None,
    event_source_covariances: np.ndarray | None = None,
    event_indices_per_frame: np.ndarray | None = None,
) -> UKFResult: ...
```

If all three event arguments are `None`, each wrapper constructs an
empty no-op event channel (`event_indices_per_frame` filled with `-1`)
before calling its JIT'd core. If any one is provided, all three are
required and validated:

- `event_source_anchors.shape == (n_sources, 2)`.
- `event_source_covariances.shape == (n_sources, 2, 2)` and finite PSD.
- `event_indices_per_frame.shape[0] == len(t_cam)`.
- valid entries are `0 <= index < n_sources`; padded entries are `-1`.

In the per-camera-frame `step` body of each filter, after the LED /
heading / ZUPT updates (the EKF and UKF scan bodies use the same
helper call):

```python
event_source_indices = event_indices_per_frame[t_idx]   # padded (MAX_EVENTS,)
state_filt, log_lik_event = update_event_location(
    state_after_zupt,
    event_model,
    event_source_indices,
    valid_mask=event_source_indices >= 0,
)
log_lik_k = log_lik_pos + log_lik_heading + log_lik_zupt + log_lik_event
```

`update_event_location` short-circuits via
`lax.cond(jnp.any(valid_mask), do_update, no_update)` so frames with
no events pay zero update cost. In the update branch, it forms the
fixed-size stacked `H`, innovation, and block-diagonal `R`, then masks
the log-likelihood to valid event rows only. Sentinel rows must not
change the posterior or log-likelihood, which is what the empty-events
parity test pins.

### Simulator extension — `src/trodestrack/sim/rat_imu.py`

`RatIMUSimConfig` gains an optional `ttl_event_geometry` block
mirroring the YAML schema. The simulator emits synthetic events
when the trajectory crosses any configured beam, enters any zone, or
enters any reader's effective radius. Zone simulation needs an explicit
activation radius/shape separate from measurement noise if those
concepts differ; v1 should either add `trigger_radius_m` to
`ZoneTriggerSpec` or document that synthetic zones use `sigma_m` as
both trigger radius and measurement uncertainty.

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
- Per-frame compact-index builder with padding.
- `TTLEventsConfig` + the three Spec dataclasses with
  `to_event_source()` math.
- Schema validation: unique IDs across all source types; valid
  edges; config-relative `ttl_events.events_file` path resolution;
  dense source arrays; pad-limit enforced at load time.
- Tests for parquet → per-frame indices; schema unique-id
  enforcement; per-spec `to_event_source()` math (point-source anchor
  and R; beam-source midpoint anchor, anisotropic R eigenvectors /
  eigenvalues, `sigma_perp_m`, and active edge).

**Exit criteria:** all three Spec classes can produce
`EventLocationSource` objects from YAML; `tests/io/test_ttl_events.py`
green.

### Milestone 2 — `EventLocationModel` core

- Implement `EventLocationModel` with `predict / jacobian / meas_cov
  / innovation`.
- Stacked-event update: K simultaneous events fold into one padded
  `(2 · MAX_EVENTS_PER_FRAME, n_state)` block update.
- Padded-sentinel handling: compact index `-1` rows get zero `H`,
  zero innovation, identity `R`, and are masked out of the innovation
  log-likelihood. (Originally specified as "large `R`"; switched to
  zero `H` plus identity `R` because the large-`R` block inflates
  ``psd_solve``'s relative diagonal boost when valid `R` is small,
  corrupting the valid update.)
- Unit tests:
  - Predict on / off the anchor for every source type.
  - Stacked H shape: two rows per event for K events;
    `H ∈ R^{2K × n_state}` (no kind-based branching).
  - Padded sentinels produce no posterior change.
  - **Beam short / long limits**: a short beam (`L < σ_perp`)
    produces an approximately isotropic R; a long beam
    (`L >> σ_perp`) produces highly anisotropic R with much weaker
    along-beam information than perpendicular information. Parametric
    tests should pin the covariance geometry and bound along-beam
    posterior contraction rather than requiring exact line-update
    equivalence.
  - JIT-compatibility smoke.

**Exit criteria:** model is callable in isolation; beam short-vs-long
limit tests green to numerical tolerance.

### Milestone 3 — Sim extension and synthetic scenario tests

- Synthetic event generator. Implemented as a standalone helper
  ``trodestrack.sim.ttl_events.events_from_trajectory(t, xy, beams=,
  zone_triggers=, rfid_readers=)`` that walks a sampled trajectory and
  emits beam crossings (sign-flip detector with half-open boundary),
  zone-trigger entries (disc activation), and RFID detections.
  Originally scoped as a ``RatIMUSimConfig.ttl_event_geometry`` field;
  the standalone helper kept the simulator API surface unchanged and
  is sufficient for scenario tests. Threading the geometry through
  ``RatIMUSimConfig`` is a deferred follow-up if the YAML simulator
  needs first-class events.
- Property: events monotonic; source IDs valid; pad-limit honored.
- Scenario per source type (separate tests):
  - Two beam-grid scenarios:
    - 1D 5-beam set spanning a 2 s dropout (cheap regression test).
    - 4×4 cell grid (5 vertical + 5 horizontal beams) over an 8 s
      session with a 5 s camera dropout, parametrized across EKF and
      UKF; asserts ≥30% position-RMSE reduction during the dropout
      with a failure message reporting both RMSEs.
  - Zone trigger snaps position to feeder location within σ_zone.
  - RFID detection collapses position uncertainty to the reader's
    effective radius.

**Exit criteria:**
`tests/integration/test_ttl_event_sensors_session.py` green; sim
emits per-source-type events that cross the model's update path.

### Milestone 4 — Filter wiring (EKF and UKF)

- `update_event_location` wrapper in
  `models/sensors/event_location.py`; shared input-validation helper
  `resolve_event_inputs` in the same module so both filter wrappers
  enforce one dtype/shape/range/PSD contract.
- Wire optional dense event arrays through `extended_kalman_filter` /
  `_extended_kalman_filter_impl` and `unscented_kalman_filter` /
  `_unscented_kalman_filter_impl`. The event update runs after the
  camera + heading + ZUPT updates in each scan body. Using the linear
  event update inside the UKF is consistent because the event
  measurement is a linear 2D position selector with Gaussian noise.
- 3D-EKF support is a follow-up unless a concrete 3D event fixture is
  added; the same model can later use a different `pos_idx` selector.
- Numerical parity tests for both filters: with no event arguments
  the output is bitwise identical to the prior code path; with event
  arguments configured but no actual events firing, the output equals
  the no-event-arguments case (within float tolerance from the
  different `max_events_per_frame` JIT trace).
- Config-run parity test: `load_session(...ttl_events...)` passes the
  dense event arrays into `extended_kalman_filter`; legacy non-config
  calls remain unchanged.

**Exit criteria:** parity tests green for both filters; full sweep
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
| `BeamSpec.to_event_source()` | spec | anchor = midpoint; R eigenvectors aligned with beam tangent / normal; eigenvalues are σ_perp² and max(σ_perp, L/√12)²; active edge defaults to fall |
| `ZoneTriggerSpec.to_event_source()` | spec | anchor = center; R = σ²·I |
| `RFIDReaderSpec.to_event_source()` | spec | anchor = center; R = (r/√2)²·I |
| Schema unique IDs | config | duplicate id across source types → `ValidationError` |
| Config path resolution | config | relative `ttl_events.events_file` resolves relative to the YAML file |
| Source-id compaction | loader | arbitrary external source ids map to dense compact indices; unknown event source id fails clearly |
| Predict on source anchor | model | point-source innovation is zero on its anchor; beam-source innovation is zero at the beam midpoint anchor |
| Stacked H shape | model | two rows per event; padded to `(2 · MAX_EVENTS_PER_FRAME, n_state)` |
| Padded sentinels are no-op | model | posterior unchanged and event log-likelihood exactly zero when only `-1` rows present |
| **Beam short/long limits** | model | short beam (`L = 0`) → isotropic update; long beam (`L >> σ_perp`) → highly anisotropic R and bounded along-beam posterior contraction |
| Sim event timing | `sim/rat_imu.py` | events monotonic; source IDs valid |
| Beam-grid scenario | EKF + sim | position RMSE during 5s dropout drops by ≥30% |
| Zone-trigger scenario | EKF + sim | position fix at zone center within σ_zone |
| RFID scenario | EKF + sim | position uncertainty collapses to ≤ effective_radius after detection |
| Numerical parity (no events) | EKF and UKF | bitwise identical to current filter |
| Empty-events parity | EKF and UKF | identical to "no ttl_events config" within float tolerance |
| Single-event influence | EKF and UKF | one zone trigger pulls posterior position toward anchor |
| Validation rejects bad inputs | EKF and UKF | partial-args / non-PD R / out-of-range index errors share one message contract |
| Multiple sensors in one frame | EKF | two or more simultaneous compact indices produce the same posterior as sequential event updates |
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
| `σ_along` formula regression silently over-constrains the beam tangent at long range | Parametric short/long beam tests pin both limits; the `max(σ_perp, L/√12)` floor is asserted via a unit test that constructs `BeamSpec(L=0)` and confirms the resulting R is well-conditioned isotropic. |
| Arbitrary parquet `source_id` values break JAX indexing or collide across source types | Schema validates unique configured IDs; loader builds and tests a dense `source_id -> compact_index` map before constructing JAX arrays. |
| Padded sentinel events perturb marginal log-likelihood | Unit tests assert sentinel-only frames leave both posterior and event log-likelihood unchanged; implementation masks log-likelihood over valid rows rather than relying on large `R`. |
| Pad-sentinel logic in `lax.scan` triggers shape recompiles | `MAX_EVENTS_PER_FRAME` is a static config arg; load-time assertion that no camera frame exceeds it. |
| Source-id overlap between beams / zones / readers silently maps to wrong source | Schema unique-id validator; loader cross-checks event-file source IDs against configured sources, rejects unknown IDs. |
| Trodes DIO event stream on a clock that doesn't sync to camera/IMU | Documented assumption that DIO is on the Trodes clock; non-Trodes users responsible for pre-aligning timestamps. |
| Hand-survey error in source coordinates (beam posts, zone centers, reader locations) | Diagnostic plot in `qa/`: per-source residual histogram at trigger time. Large mean residual flags miscalibrated geometry. |
| RFID false-positives (stray detections at the reader's edge) | `effective_radius_m` is the user's tuning knob: tighter R inflates posterior less per false-positive. Optional debounce in the loader for known-noisy readers. |
| Multi-beam ghost triggers from one rat pass | Optional per-source debounce window in the loader (drop triggers within `≤debounce_ms` of the previous trigger on the same source). |
| Zone simulation confuses trigger size with measurement noise | Add `trigger_radius_m` for simulated zones if needed, or explicitly document/test that v1 synthetic zones use `sigma_m` for both activation and update uncertainty. |

## Rollout Strategy

- Ship behind `ttl_events: …` config; absent config = zero
  behavior change.
- Numerical parity tests (no-args + empty-events) gate the EKF and UKF
  wiring milestone.
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
  as a Gaussian approximation — useful onboarding for engineers
  extending to new event types.

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
   the single Gaussian point-fix abstraction? Acoustic / ultrasonic
   distance sensors would be 1D range constraints (different geometry);
   accelerometer-based fall detection is event-only with no
   spatial implication. The current abstraction covers the common
   spatial-event cases; true line, range, and event-only sensors are
   separate plans if hardware ever exposes them.
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

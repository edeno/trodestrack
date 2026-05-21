# Phase 5 — Type design hardening

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Push runtime invariants into type construction. Five surgical changes: `state_mode` Literal, `EventLocationSource` validation, `StateLayout` validation, `FilterState.create()`, `PreparedSession` field regrouping. Ships as `0.4.0` (minor — breaking schema/dataclass shape changes).

**Inputs to read first:**

- [src/trodestrack/models/state_layout.py](../../../../src/trodestrack/models/state_layout.py) — the `StateLayout` dataclass and `LAYOUT_REGISTRY`. Phase 5 adds the `StateMode` Literal alias here.
- [src/trodestrack/models/filter_common.py](../../../../src/trodestrack/models/filter_common.py) lines 174-229 — `FilterCoreConfig.state_mode` field and `__post_init__`; the silent `object.__setattr__` mutation is at line 228-229.
- [src/trodestrack/models/filter_common.py](../../../../src/trodestrack/models/filter_common.py) lines 498-511 — `FilterState` NamedTuple.
- [src/trodestrack/models/filter_common.py](../../../../src/trodestrack/models/filter_common.py) — find `validate_initial_state` by grep; it's the source-of-truth for the FilterState invariants.
- [src/trodestrack/config/schemas.py](../../../../src/trodestrack/config/schemas.py) lines 211 (FilterConfig.state_mode — already Literal), 293-308 (EventLocationSource).
- [src/trodestrack/io/session.py](../../../../src/trodestrack/io/session.py) lines 30-49 — `PreparedSession` dataclass.
- Tests for the touched types: `tests/filters/test_config_immutability.py`, `tests/models/test_state_layout.py` (verify exists; create if not), `tests/io/test_session_loading.py`.

**Contracts referenced:**

- [`StateMode` Literal alias](shared-contracts.md#statemode-literal-alias) — defined here, exported from `models/state_layout.py`.
- [No backwards-compatibility shims](shared-contracts.md#no-backwards-compatibility-shims) — old field names are removed, not aliased.

## Tasks

### Task 1 — Define `StateMode` Literal and `STATE_MODES` tuple

In [src/trodestrack/models/state_layout.py](../../../../src/trodestrack/models/state_layout.py), add at the bottom of the file (after the `LAYOUT_REGISTRY` definition):

```python
from typing import Literal, get_args

StateMode = Literal[
    "2d_full",
    "vision_only",
    "imu_only",
    "2d_cam_3d_imu",
    "2d_cam_6dof_imu_orientation",
    "3d_euler",
    "3d_quat",
    "3d_cam_6dof_imu",
]

STATE_MODES: tuple[str, ...] = get_args(StateMode)
```

Export both via `__all__` if the module has one (it likely doesn't yet — leave that alone).

### Task 2 — Add `StateLayout` construction-time validation

In [src/trodestrack/models/state_layout.py:22-93](../../../../src/trodestrack/models/state_layout.py#L22-L93), add `__post_init__` to the `StateLayout` dataclass:

```python
def __post_init__(self) -> None:
    """Validate that the layout describes a consistent state vector.

    Without this guard, a layout with out-of-range or overlapping
    indices passes construction silently and only fails downstream
    inside a JAX trace with an opaque clamp-or-broadcast error.
    """
    if self.n < 1:
        raise ValueError(f"StateLayout.n must be >= 1; got {self.n}.")

    if isinstance(self.heading_idx, int):
        heading_indices: tuple[int, ...] = (self.heading_idx,)
    elif isinstance(self.heading_idx, tuple):
        heading_indices = self.heading_idx
    else:
        raise TypeError(
            "StateLayout.heading_idx must be int or tuple of ints; got "
            f"{type(self.heading_idx).__name__}."
        )

    all_indices = (
        tuple(self.pos_idx)
        + tuple(self.vel_idx)
        + heading_indices
        + tuple(self.bias_gyro_idx)
        + tuple(self.bias_accel_idx)
    )

    # Range check
    out_of_range = [i for i in all_indices if not 0 <= i < self.n]
    if out_of_range:
        raise ValueError(
            f"StateLayout indices out of range [0, {self.n}): {out_of_range}."
        )

    # Disjoint check
    if len(set(all_indices)) != len(all_indices):
        seen: set[int] = set()
        dupes = [i for i in all_indices if i in seen or seen.add(i)]  # type: ignore[func-returns-value]
        raise ValueError(
            f"StateLayout indices must be disjoint; duplicates: {sorted(set(dupes))}."
        )

    # Exhaustion check (every index in [0, n) is claimed by exactly one component)
    if set(all_indices) != set(range(self.n)):
        missing = sorted(set(range(self.n)) - set(all_indices))
        raise ValueError(
            f"StateLayout indices do not exhaust [0, {self.n}); missing: {missing}."
        )

    # heading_idx structural check
    if isinstance(self.heading_idx, tuple) and len(self.heading_idx) not in (3, 4):
        raise ValueError(
            "StateLayout.heading_idx tuple must have length 3 (Euler) or "
            f"4 (quaternion); got {len(self.heading_idx)}."
        )
```

Run the existing test suite. Any failures from existing `LAYOUT_REGISTRY` entries indicate a real bug in the registry that should be fixed alongside.

Add a test in `tests/models/test_state_layout.py` (create if it doesn't exist):

- `test_state_layout_rejects_out_of_range_indices`
- `test_state_layout_rejects_overlapping_indices`
- `test_state_layout_rejects_non_exhaustive_indices`
- `test_state_layout_rejects_wrong_heading_tuple_length`
- `test_state_layout_accepts_all_registered_layouts` — `for mode, layout in LAYOUT_REGISTRY.items(): assert layout` (constructs successfully).
- `test_state_modes_match_layout_registry_keys` — `assert set(STATE_MODES) == set(LAYOUT_REGISTRY.keys())`.

### Task 3 — Replace silent `vision_only ⇒ enable_zupt=False` mutation

In [src/trodestrack/models/filter_common.py:228-229](../../../../src/trodestrack/models/filter_common.py#L228-L229), replace the `object.__setattr__` mutation:

```python
# OLD:
if self.state_mode == "vision_only" and self.enable_zupt:
    object.__setattr__(self, "enable_zupt", False)

# NEW:
if self.state_mode == "vision_only" and self.enable_zupt:
    raise ValueError(
        "enable_zupt=True is incompatible with state_mode='vision_only': "
        "ZUPT requires IMU stationarity detection. Set enable_zupt=False "
        "explicitly when using vision_only, or use a state mode that "
        "consumes IMU data."
    )
```

Find callers that construct `FilterCoreConfig(state_mode="vision_only", enable_zupt=True)` (likely tests; grep `tests/`). Fix them to pass `enable_zupt=False` explicitly.

### Task 4 — Promote `state_mode` to Literal in `FilterCoreConfig`

In [src/trodestrack/models/filter_common.py:174-177](../../../../src/trodestrack/models/filter_common.py#L174-L177):

```python
from trodestrack.models.state_layout import StateMode, STATE_MODES

# ...
@dataclass(frozen=True)
class FilterCoreConfig:
    # ...
    state_mode: StateMode = "2d_cam_3d_imu"
```

Add a runtime guard in `__post_init__` (mypy doesn't enforce Literal at runtime):

```python
if self.state_mode not in STATE_MODES:
    raise ValueError(
        f"state_mode must be one of {STATE_MODES}; got {self.state_mode!r}. "
        "Add to LAYOUT_REGISTRY and StateMode (in state_layout.py) if "
        "introducing a new mode."
    )
```

### Task 5 — Add `FilterState.create()` classmethod

In [src/trodestrack/models/filter_common.py:498-511](../../../../src/trodestrack/models/filter_common.py#L498-L511), extend the `FilterState` NamedTuple with a validating constructor. NamedTuples don't support classmethods that override `__new__` cleanly, but they DO support regular classmethods:

```python
class FilterState(NamedTuple):
    """..."""

    mean: jnp.ndarray
    cov: jnp.ndarray

    @classmethod
    def create(
        cls,
        mean: jnp.ndarray,
        cov: jnp.ndarray,
        layout: "StateLayout | None" = None,
    ) -> "FilterState":
        """Construct a FilterState with shape/PSD validation.

        Calls validate_initial_state() before constructing; raises on
        wrong shape, non-symmetric covariance, or negative eigenvalues.
        Prefer this over raw FilterState(mean, cov) for new code.

        Parameters
        ----------
        mean : jnp.ndarray
            State mean (n,).
        cov : jnp.ndarray
            State covariance (n, n). Must be symmetric and positive
            (semi-)definite.
        layout : StateLayout, optional
            If provided, additionally validates that mean.shape[0] ==
            layout.n.
        """
        validate_initial_state(mean, cov)  # existing function
        if layout is not None and mean.shape[0] != layout.n:
            raise ValueError(
                f"FilterState.mean has shape {mean.shape}; layout requires "
                f"({layout.n},). Use the StateLayout matching the state_mode."
            )
        return cls(mean=mean, cov=cov)
```

Existing call sites that use raw `FilterState(mean, cov)` are left alone — `create()` is purely additive. Document in the CHANGELOG that new code should prefer `FilterState.create()`.

### Task 6 — Add `EventLocationSource.__post_init__` validation

In [src/trodestrack/config/schemas.py:293-308](../../../../src/trodestrack/config/schemas.py#L293-L308), tighten the dataclass:

```python
SourceType = Literal["beam", "zone", "rfid"]

@dataclass(frozen=True)
class EventLocationSource:
    """..."""  # existing docstring

    source_id: int
    anchor: np.ndarray  # (2,) world meters
    covariance: np.ndarray  # (2, 2) world-frame measurement covariance, PSD
    label: str | None = None
    source_type: SourceType = "beam"

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor, dtype=float)
        if anchor.shape != (2,):
            raise ValueError(
                f"EventLocationSource.anchor must have shape (2,); got "
                f"{anchor.shape}."
            )
        if not np.all(np.isfinite(anchor)):
            raise ValueError(
                "EventLocationSource.anchor must be finite; got "
                f"{anchor.tolist()}."
            )
        object.__setattr__(self, "anchor", anchor)

        cov = np.asarray(self.covariance, dtype=float)
        if cov.shape != (2, 2):
            raise ValueError(
                f"EventLocationSource.covariance must have shape (2, 2); "
                f"got {cov.shape}."
            )
        if not np.all(np.isfinite(cov)):
            raise ValueError("EventLocationSource.covariance must be finite.")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError(
                "EventLocationSource.covariance must be symmetric; "
                f"max asymmetry={float(np.max(np.abs(cov - cov.T))):.2e}."
            )
        eigs = np.linalg.eigvalsh(cov)
        if eigs.min() < -1e-10:
            raise ValueError(
                f"EventLocationSource.covariance must be PSD; min eigenvalue="
                f"{float(eigs.min()):.2e}."
            )
        object.__setattr__(self, "covariance", cov)
```

`source_type: SourceType = "beam"` drops the `"unknown"` default value — the comment-analyzer noted it's never legitimately constructed. Grep `grep -n 'source_type="unknown"' src/ tests/` for any actual usage; if present, fix to a valid type.

### Task 7 — Group `PreparedSession`'s TTL fields into `EventChannel`

In [src/trodestrack/io/session.py:30-49](../../../../src/trodestrack/io/session.py#L30-L49), introduce a nested dataclass and collapse the four parallel optional fields:

```python
@dataclass(frozen=True)
class EventChannel:
    """Resolved TTL-event channel data for a session.

    Either all four fields are present (TTL events configured and loaded)
    or this object is None entirely. Replaces four parallel Optional
    fields on PreparedSession that had to move together.
    """

    sources: tuple[EventLocationSource, ...]
    anchors: np.ndarray              # (n_sources, 2)
    covariances: np.ndarray          # (n_sources, 2, 2)
    indices_per_frame: np.ndarray    # (n_cam, max_events_per_frame), int32, -1 padded

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError(
                "EventChannel with empty sources should be represented as "
                "None instead."
            )
        n_sources = len(self.sources)
        if self.anchors.shape != (n_sources, 2):
            raise ValueError(
                f"anchors shape {self.anchors.shape} mismatched with "
                f"len(sources)={n_sources}; expected ({n_sources}, 2)."
            )
        if self.covariances.shape != (n_sources, 2, 2):
            raise ValueError(
                f"covariances shape {self.covariances.shape}; expected "
                f"({n_sources}, 2, 2)."
            )


@dataclass(frozen=True)
class PreparedSession:
    """Filter-ready session arrays and diagnostics."""

    t_imu: np.ndarray
    U_imu: np.ndarray
    t_cam: np.ndarray
    Z_cam_led1: np.ndarray
    Z_cam_led2: np.ndarray
    mask_cam: np.ndarray
    conf_cam: np.ndarray | None
    led_distance: float | None
    diagnostics: dict[str, object]
    config: SessionConfig
    gyro_z_for_led_identity: np.ndarray | None = None
    U_imu_for_calibration: np.ndarray | None = None
    events: EventChannel | None = None    # replaces 4 parallel fields

    @property
    def source_format(self) -> str:
        return self.config.inputs.format
```

Update every consumer. Grep `grep -rnE 'event_sources|event_source_anchors|event_source_covariances|event_indices_per_frame' src/ tests/`. Expected hits:

- `src/trodestrack/io/session.py` itself — `_attach_ttl_events` constructs `EventChannel(sources=..., anchors=..., covariances=..., indices_per_frame=...)` and passes `events=channel` (or `events=None` if no TTL events configured).
- `src/trodestrack/runtime/offline.py` — anywhere that destructures `session.event_sources`, `session.event_source_anchors`, etc., now reads from `session.events.sources`, etc., guarded by `if session.events is not None`.
- Filter call sites — same pattern.
- Tests in `tests/io/test_session_loading.py`, `tests/integration/test_ttl_event_sensors_session.py`.

No backwards-compat shims — the old field names are gone.

### Task 8 — CHANGELOG entry

Add under `## [0.4.0] — unreleased`:

```
### Changed (breaking)
- `PreparedSession` now exposes TTL events via a single optional `events: EventChannel | None` field, replacing the four parallel optional fields (`event_sources`, `event_source_anchors`, `event_source_covariances`, `event_indices_per_frame`). Update consumers to read from `session.events.sources` (etc.) when `session.events is not None`.
- `FilterCoreConfig.state_mode` is now `Literal["2d_full", "vision_only", "imu_only", "2d_cam_3d_imu", "2d_cam_6dof_imu_orientation", "3d_euler", "3d_quat", "3d_cam_6dof_imu"]` with a runtime membership check. Typos now fail at construction.
- `FilterCoreConfig(state_mode="vision_only", enable_zupt=True)` now raises `ValueError` instead of silently disabling ZUPT. Set `enable_zupt=False` explicitly.
- `EventLocationSource.source_type` is now `Literal["beam", "zone", "rfid"]`. The default value changed from `"unknown"` to `"beam"`.
- `StateLayout(...)` now validates indices at construction: all indices must be in `[0, n)`, disjoint, and exhaust the state vector. Custom layouts with out-of-range or overlapping indices now fail loudly.
- `EventLocationSource` now validates anchor/covariance shape, finiteness, symmetry, and PSD at construction.

### Added
- `FilterState.create(mean, cov, layout=None)` classmethod: validates shape and PSD at construction. Prefer over raw `FilterState(mean, cov)` for new code.
- `StateMode` Literal and `STATE_MODES` tuple exported from `models.state_layout` for downstream consumers (CLI, schemas, tests).
- `EventChannel` dataclass in `io.session` grouping the TTL-event fields.
```

## Deliberately not in this phase

- **`PreparedSession.diagnostics: dict[str, object]` → `SessionDiagnostics` dataclass** — flagged by the type-design review as worthwhile but out of scope for this phase. Open a follow-up issue.
- **`MeasurementModel.subspace` return-tuple sum type** — type-design suggestion; skipped for scope reasons.
- **`FilterConfig`'s 22 `Optional[…]=None` sentinel fields refactor** — would require breaking config-file compatibility and is high-risk for a low-value cleanup.
- **PyTree registration on new types** — `EventChannel` doesn't need to be a JAX pytree (it's host-only). `FilterState` already is one via `NamedTuple`. No new registrations.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_state_layout_rejects_out_of_range_indices` | `StateLayout(n=5, pos_idx=(0, 1), vel_idx=(2, 99), heading_idx=4, ...)` raises ValueError matching "out of range". |
| `test_state_layout_rejects_overlapping_indices` | `StateLayout(n=5, pos_idx=(0, 1), vel_idx=(1, 2), heading_idx=4, ...)` raises ValueError matching "disjoint". |
| `test_state_layout_rejects_non_exhaustive_indices` | `StateLayout(n=5, pos_idx=(0, 1), vel_idx=(2, 3), heading_idx=-99, ...)` — well, that fails on out-of-range first; construct one with `n=10, pos_idx=(0,1), vel_idx=(2,3), heading_idx=4, bias_gyro=(), bias_accel=()` — only 5 indices for n=10; assert raises. |
| `test_state_layout_rejects_wrong_heading_tuple_length` | `StateLayout(n=5, ..., heading_idx=(0, 1))` raises (length must be 3 or 4). |
| `test_state_layout_accepts_all_registered_layouts` | Loop over `LAYOUT_REGISTRY.values()` — no exceptions. |
| `test_state_modes_match_layout_registry_keys` | `set(STATE_MODES) == set(LAYOUT_REGISTRY.keys())`. |
| `test_filter_core_config_rejects_unknown_state_mode` | `FilterCoreConfig(state_mode="typo")` raises ValueError naming the allowed values. |
| `test_filter_core_config_rejects_vision_only_with_zupt` | `FilterCoreConfig(state_mode="vision_only", enable_zupt=True)` raises ValueError. |
| `test_filter_state_create_rejects_wrong_shape` | `FilterState.create(jnp.zeros(5), jnp.eye(4))` raises ValueError. |
| `test_filter_state_create_rejects_non_psd_cov` | `FilterState.create(jnp.zeros(2), jnp.array([[1.0, 0.0], [0.0, -1.0]]))` raises. |
| `test_filter_state_create_accepts_valid_input` | `FilterState.create(jnp.zeros(2), jnp.eye(2))` returns a `FilterState`. |
| `test_event_location_source_rejects_wrong_anchor_shape` | `EventLocationSource(source_id=1, anchor=np.zeros(3), covariance=np.eye(2))` raises. |
| `test_event_location_source_rejects_non_symmetric_cov` | `EventLocationSource(..., covariance=np.array([[1, 0], [0.5, 1]]))` raises. |
| `test_event_location_source_rejects_non_psd_cov` | `EventLocationSource(..., covariance=np.array([[1, 0], [0, -1]]))` raises. |
| `test_event_location_source_source_type_literal_rejects_unknown` | `EventLocationSource(..., source_type="bogus")` raises Pydantic / runtime error. |
| `test_prepared_session_uses_event_channel_when_ttl_events_configured` | Load a session with TTL events; assert `session.events is not None`, `session.events.sources` matches the configured sources, `session.events.anchors.shape == (n, 2)`. |
| `test_prepared_session_events_is_none_when_no_ttl_events` | Load a session without TTL events; assert `session.events is None`. |
| `test_event_channel_rejects_inconsistent_shapes` | `EventChannel(sources=(s1, s2), anchors=np.zeros((1, 2)), ...)` raises. |
| `test_full_integration_suite_still_passes` | `uv run pytest tests/integration/` exits 0. |

## Fixtures

Reuse existing fixtures in `tests/filters/conftest.py` and `tests/io/test_session_loading.py`. Add minimal in-file dataclass constructors for the negative tests.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented.
- All `LAYOUT_REGISTRY` entries pass the new `StateLayout.__post_init__` check (Task 2). If any failed, the registry entry was buggy — fix it alongside.
- `grep -rnE 'event_sources|event_source_anchors|event_source_covariances|event_indices_per_frame' src/ tests/` returns zero hits (modulo the CHANGELOG entry).
- `grep -rnE 'object.__setattr__.*enable_zupt' src/` returns zero hits (the silent mutation is gone).
- No backwards-compat shims for the regrouped fields (no `@property` returning the old field names).
- Integration test suite passes (Task 9 test list includes `test_full_integration_suite_still_passes`).
- CHANGELOG entry is in `## [0.4.0]` (minor version bump for the breaking schema changes).
- Docstrings, test names, and module names don't reference this plan or its milestones.

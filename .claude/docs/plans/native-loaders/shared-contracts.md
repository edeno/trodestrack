# Shared Contracts

[← back to README](README.md)

These contracts span phases. Phase files link in here rather than
inlining; treat the contracts as authoritative when implementing.

- [`PositionPixels` — loader-private intermediate](#positionpixels--loader-private-intermediate)
- [`pixels_to_meters` — calibration helper](#pixels_to_meters--calibration-helper-at-top-level)
- [IMU source resolution](#imu-source-resolution)
- [NWB DIO → TTL events bridge (schema contract)](#nwb-dio--ttl-events-bridge-schema-contract)
- [NWB container-layer API (public; Spyglass integration seam)](#nwb-container-layer-api-public-spyglass-integration-seam)

## `PositionPixels` — loader-private intermediate

Lives in `src/trodestrack/io/loaders/_shared.py`. Every format reader
returns one of these before meter conversion:

```python
@dataclass(frozen=True)
class PositionPixels:
    led1_pixels: np.ndarray         # (n_cam, 2), NaN for invalid
    led2_pixels: np.ndarray | None  # optional second LED
    t_cam: np.ndarray               # (n_cam,) seconds
    confidence: np.ndarray | None   # (n_cam, 4) [c1x, c1y, c2x, c2y] in [0, 1]
    frame_dimensions: tuple[int, int] | None  # (width, height) when known
    diagnostics: dict[str, object]  # source-specific notes

    # Calibration the reader recovered from the source file. None means
    # "no calibration baked into the source"; the conversion helper
    # falls back to ``CameraConfig.meters_per_pixel``. NWB readers
    # populate this from the per-series ``conversion`` only when
    # ``unit == "meters"`` or ``conversion != 1.0``; the
    # ``pixels`` + ``conversion=1.0`` sentinel case leaves it ``None``.
    # The reader does *not* pre-multiply pixel data by this value;
    # ``pixels_to_meters`` is the single point of conversion.
    coords_meters_per_pixel: float | None = None
```

## `pixels_to_meters` — calibration helper at top level

Lives in `src/trodestrack/io/pixel_to_meters.py` (the same module path
the geom-homography plan reserves). Existing `_load_leds`
(`session.py:609`) migrates to call it in Phase 0; the new loaders
call it in Phase 1+.

```python
def pixels_to_meters(
    pixels: PositionPixels,
    camera_config: CameraConfig,
    *,
    meters_per_pixel_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply camera calibration. Returns (led1_meters, led2_meters, conf).

    Resolution order for the scale factor (today scalar; geom-homography
    plan adds the homography branch later, same module path):

    1. ``meters_per_pixel_override`` (NWBConfig knob — wins; for
       post-hoc re-calibration).
    2. ``pixels.coords_meters_per_pixel`` (file-side, e.g. NWB
       per-series ``conversion`` when the file is calibrated).
    3. ``camera_config.meters_per_pixel`` (the YAML default).

    Without this precedence, NWB-calibrated data would silently re-scale
    by the YAML's ``meters_per_pixel`` default
    (``CameraConfig.meters_per_pixel`` has a non-null default of
    0.0022), so we track the file-side value explicitly.
    """
```

## IMU source resolution

The new loaders **do not require** an `inputs.imu_file` parquet. v1
precedence per format:

| Format | Default IMU source | If unavailable |
| --- | --- | --- |
| `nwb` | `processing["analog"]["analog"]["analog"]` (the path `trodes_to_nwb/convert_analog.py:89-104` writes; `_NWB_ANALOG_DATA_PATH = "processing/analog/analog/analog/data"`). Channel ids matched against `IMUConfig.axis_map`; same SI conversion as the parquet path. | Fall through to the synthetic-vision-only stream below. |
| `trodes_native` / `dlc_keypoints` | (No default; raw `.rec` IMU is deferred.) | Fall through to the synthetic-vision-only stream below. |
| `spikegadgets_trodes` / `prepared_arrays` | Existing behavior. Unchanged. | n/a |

If `inputs.imu_file` is configured, it **overrides** the default IMU
source for any format; the parquet wins.

If no IMU source is available (raw Trodes / DLC with no parquet, or NWB
without `processing["analog"]`), the loader's behavior depends on the
configured `filter.state_mode`:

- **`vision_only`** (no IMU consumed): the loader synthesizes a
  zero-valued IMU stream aligned to `t_cam` with the channel count
  `vision_only` expects (the fused-filter modes ignore values anyway,
  but the array shape must match). This is the legal no-IMU path.
- **`2d_full` / `2d_cam_3d_imu` / `2d_cam_6dof_imu_orientation`** (any
  mode where `_uses_imu` is True; see `session.py:698`): **raise** with
  a message naming the three remediation options:
  `"inputs.format=<format> requires an IMU source. Provide
  inputs.imu_file, use NWB analog IMU (NWB only), or set
  filter.state_mode: vision_only."` Silently zero-filling here would
  produce a fused trajectory that looks IMU-configured but had no
  inertial data — a "wrong but plausible" failure mode.

The default `filter.state_mode` is `2d_cam_3d_imu`
(`schemas.py:211`), so the raise path is what users hit by default
when they forget to configure an IMU source. The synthetic-zero IMU
path is reserved for explicit `vision_only` runs.

A `load_imu_parquet(imu_file, config)` helper is extracted in Phase 0
from the inline `_remove_sample_hold` / `_convert_imu_to_si` /
`_project_imu_for_filter` chain (`session.py:502, 538, 577`); it's
reused by every loader. `_load_spikegadgets_trodes` migrates to call it
(parity-gated).

## NWB DIO → TTL events bridge (schema contract)

The geometry block (`beams`/`zone_triggers`/`rfid_readers`) is **still
required** for NWB DIO sessions — the EKF/UKF event channel can't run
without source positions, covariances, and `id`s. Only the
`events_file` field changes:

- `TTLEventsConfig.events_file: Path | None = None` (was required).
- A **session-level** validator on `SessionConfig` enforces:
  - If `inputs.format != "nwb"` or `inputs.nwb.dio_to_ttl is None`,
    `events_file` is required.
  - If `inputs.format == "nwb"` and `inputs.nwb.dio_to_ttl` is set,
    `events_file` may be omitted; if both are provided, the parquet
    wins (NWB DIO is ignored, both sources recorded in diagnostics).
  - If `inputs.nwb.dio_to_ttl` is set, `ttl_events` itself must be
    present (not `None`) — `SessionConfig.ttl_events` is currently
    optional ([schemas.py:470](../../../../src/trodestrack/config/schemas.py#L470));
    this validator closes the loophole.
  - Every value in `inputs.nwb.dio_to_ttl.name_to_source_id` must
    appear as an `id` in one of `ttl_events.beams`,
    `ttl_events.zone_triggers`, or `ttl_events.rfid_readers`.
  - This cross-check spans `inputs` and `ttl_events`; it lives on
    `SessionConfig`, not on `NWBDIOToTTLConfig` (which has no view of
    the geometry block).

Loader-time validation (separate from schema-time):

- The NWB loader confirms each key in `name_to_source_id` resolves to
  a `TimeSeries` under `processing["behavior"]["behavioral_events"]`.
  Unmatched names raise with a list of the names found in the file.
- Schema-time validation never imports `pynwb` (it's an optional
  extra) and never opens the NWB file. The `_validate_required_paths`
  validator only checks field spellings, required paths, and
  intra-config consistency.

## NWB container-layer API (public; Spyglass integration seam)

Primary downstream consumer: a Spyglass table's `make()` that calls
`fetch_nwb` on `RawPosition` / `DIOEvents` / `SensorData` and assembles
the resulting pynwb containers into trodestrack inputs. We therefore
expose a **public container-level API** as first-class surface, not a
hidden seam under the path-based loader. The path-based loader becomes
a thin wrapper that opens the file, picks containers, and delegates.

**Public entry points** live in `src/trodestrack/io/nwb/__init__.py`.
**Only `load_nwb_session` imports `pynwb`** (lazily, inside the
function body). The four container entries do **not** import `pynwb`
at module load — they accept already-loaded objects and read attributes
duck-typed. Type annotations on container parameters are
`TYPE_CHECKING`-guarded string forms (or untyped) so `import
trodestrack.io.nwb` never triggers `pynwb` import. This lets a Spyglass
`make()` (which has already imported `pynwb` to call `fetch_nwb`) call
the entries directly without trodestrack carrying a hard `pynwb`
dependency.

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pynwb  # type-checking only; never imported at runtime here

def from_position_container(
    position: "pynwb.behavior.Position",
    cfg: NWBLEDSourceConfig,
) -> PositionPixels:
    """Trodes-style route. Reads the named SpatialSeries pair, applies
    the conversion=1.0 sentinel rule, populates ``coords_meters_per_pixel``."""

def from_pose_estimation_container(
    pose,  # ndx_pose.PoseEstimation; untyped at runtime, optional dep
    cfg: NWBLEDSourceConfig,
) -> PositionPixels:
    """ndx-pose route. Same Skeleton fallback chain (v0.1.x ↔ v0.2.x)
    documented in the position section below."""

def from_analog_container(
    analog_ts,  # pynwb.base.TimeSeries; untyped at runtime
    imu_cfg: IMUConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t_imu, U_imu_si). Channel ids matched against
    ``imu_cfg.axis_map``; same conversion as ``load_imu_parquet``."""

def from_behavioral_events(
    events,  # BehavioralEvents OR dict[str, TimeSeries]; see below
    dio_cfg: NWBDIOToTTLConfig,
) -> DIOEventTable:
    """Edge detection on int8 0/1 streams; drops first sample.

    Accepts the trodestrack-canonical shapes: a ``BehavioralEvents``
    container (path-loader source) or ``dict[str, TimeSeries]``
    keyed by event name. Spyglass's ``(DIOEvents & key).fetch_nwb()``
    returns ``list[dict]`` with ``dio_event_name`` / ``dio`` columns;
    Spyglass's ``make()`` is responsible for assembling
    ``{row["dio_event_name"]: row["dio"] for row in ...}`` before
    calling this function. trodestrack's API is Spyglass-agnostic."""

def load_nwb_session(
    cfg: NWBConfig,
) -> tuple[PositionPixels, NWBSessionExtras]:
    """Path-based wrapper. Lazy-imports ``pynwb``; opens with
    ``NWBHDF5IO``, walks containers, delegates to the four entry
    points above. Used by ``inputs.format=nwb`` direct-NWB users."""
```

**Eager-materialization contract.** All four container entry points
**must convert lazy HDF5 datasets to numpy arrays before returning**.
Spyglass `fetch_nwb` keeps the underlying NWB file open behind the
returned containers; the caller is free to close its IO handle after
our function returns. Concretely: every `series.data[...]`,
`series.timestamps[...]`, `confidence[...]` access materializes — no
returning of `h5py.Dataset` references, no holding container references
in `PositionPixels` / `NWBSessionExtras`. Tested by closing the IO
handle after the call and re-using the returned arrays.

**Container-name addressing.** The container-layer takes the container
*object* the caller already located; it doesn't walk
`processing["behavior"]` itself. Auto-detect-by-neurodata-type only
runs in `load_nwb_session` (the path-based wrapper). Spyglass is the
authority on which container goes where in its own NWB conventions.

**Namespace registration.** The container layer assumes neurodata-type
extensions (`ndx-pose`, and Spyglass's `ndx-franklab-novela` when
relevant) are already registered in pynwb's global namespace cache by
the caller. Spyglass's `fetch_nwb` registers them on import; direct-NWB
users are covered by `NWBHDF5IO(..., load_namespaces=True)` in
`load_nwb_session` reading the embedded namespace.

**No `[spyglass]` extra.** trodestrack does not import `spyglass` or
`datajoint`. Spyglass writes a table whose `make()` calls our public
container API; the dependency direction is one-way (Spyglass → us).

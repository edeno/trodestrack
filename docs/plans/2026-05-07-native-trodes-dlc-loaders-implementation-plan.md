# Native Trodes / DLC Loaders Implementation Plan

## Status

Not started. The `io/` package currently exposes two input formats:

- `prepared_arrays` — pre-built text files (`t_imu.txt`,
  `U_imu.txt`, `t_cam.txt`, LED arrays, optional mask).
- `spikegadgets_trodes` — parquet workflow with sample-and-hold
  removal, IMU SI conversion, camera column mapping. Position
  parquet is assumed *already-tracked* (LED1/LED2 pixel coords).

Two ingest gaps that PRD §5 / §8 list:

- **Native Trodes LED-detection format** — Trodes exports raw LED
  detections as `*.videoPositionTracking` binary files (or
  associated `.h5` / `.csv` exports). Users on this branch today
  have to convert to parquet first.
- **Native DLC keypoint format** — DLC exports per-frame keypoint
  detections as `.h5` (default) or `.csv` with a multi-index
  columns layout (`(scorer, bodypart, coords)`). Users have to
  pick the LED keypoints, convert pixels to meters, and emit a
  parquet themselves.

Today's workaround is "convert upstream"; this plan ships the
converters as supported `inputs.format` choices.

## Goals

- Add two new `inputs.format` values:
  - `trodes_native`: load Trodes LED detections directly from the
    binary export.
  - `dlc_keypoints`: load DLC `.h5` keypoint export, select two
    keypoints as LED1/LED2, convert pixels to meters via
    homography or scalar.
- Reuse the existing `_load_spikegadgets_trodes` skeleton and the
  shared `PreparedSession` output contract — both new loaders
  produce the same `(t_imu, U_imu, t_cam, Z_cam_led1, Z_cam_led2,
  mask_cam, conf_cam, led_distance, diagnostics, config)`.
- Keep IMU loading orthogonal to camera loading. Both new formats
  consume the existing `imu_file` parquet path; only the *position*
  side changes.

## Non-Goals

- Hardware acquisition / live ingest from Trodes. The new loaders
  read offline files, not live streams.
- Format conversion CLI (e.g., `trodestrack convert-trodes`). The
  loaders are part of the session pipeline; users who want a
  one-off conversion can run the loader and dump the
  `PreparedSession` arrays.
- DLC re-tracking, frame extraction, or model training. Scope is
  *consume an existing DLC export*.
- Multi-animal DLC keypoints (PRD scopes single-animal v1).

## Background

### Trodes native format

Trodes exports LED detections via the
[SpikeGadgets/trodes-python-tools](https://github.com/SpikeGadgets/trodes-python-tools)
ecosystem. The on-disk artifacts users typically have:

- `*.videoPositionTracking` — binary file with timestamped LED
  positions in pixels. Columns: `timestamp_uint32, x1, y1, x2, y2`
  (sometimes `intensity1, intensity2`).
- `*.cameraHWFrameCount` — wall-clock-to-camera-frame map.
- `*.cameraHWSync` (or DIO) — Trodes-clock-to-camera-clock sync
  data.

The Trodes-python-tools `readTrodesExtractedDataFile` helper
already parses the binary; we depend on it (it's a small,
well-maintained package).

### DLC keypoint format

DLC exports keypoints from `analyze_videos` as either:

- `*.h5` (default, fast): pandas multi-index DataFrame with
  columns `(scorer, bodypart, coord)` where `coord ∈ {x, y,
  likelihood}`. Index is integer frame number.
- `*.csv`: same shape, slower I/O.

Users typically pick two named bodyparts as LED1 / LED2 (e.g.,
`led_front`, `led_back`). The likelihood column maps directly to
our `conf_cam` channel.

### Pixel → meters

Both Trodes and DLC outputs are in pixel coordinates. Conversion
options (the existing `meters_per_pixel` scalar already covers
the simplest case):

- **Scalar `meters_per_pixel`** — works for orthogonal cameras
  with no perspective distortion (the bundled Arthur slice).
- **2D homography matrix** — maps pixel `(u, v)` to world
  `(x, y)` for arbitrary planar setups. Requires the
  `geom/` module / `trodestrack calib-homography` CLI from
  the companion plan.

This plan assumes both options are available; `geom/` ships
first or in parallel.

## Design Principles

- **One contract, many ingesters.** Every loader produces the
  same `PreparedSession`; the EKF / smoother / safety-check
  paths see no format-specific code.
- **Orthogonal IMU and camera loading.** Don't refactor the
  existing IMU parquet path; the new loaders only swap out the
  position side.
- **Pixel→meters is camera calibration, not input identity.** Keep the
  scalar and homography choices under `camera` so every position loader
  uses the same calibration fields.
- **Strict per-format validation.** The current `InputsConfig` is a
  single Pydantic model keyed by `inputs.format`. **Decision: extend
  the existing format-specific `_validate_required_paths` validator
  surgically** rather than refactor to a discriminated union.
  Rationale: the existing validator already enumerates required
  fields per format; adding two more `format` literals and matching
  required-field lists is a localized change. A discriminated-union
  refactor would touch every test in `test_session_config.py`
  (~20 tests assert against the current single-model schema) for
  marginal type-safety gain. Flag the discriminated-union refactor
  as a separate milestone if a fifth or sixth input format ever
  lands.
- **`extra="forbid"`** stays on every config; misspelled fields
  fail at schema time, not deep in the loader.

## Architecture

### Schema additions — `src/trodestrack/config/schemas.py`

Extend `InputsConfig.format`:

```python
format: Literal[
    "prepared_arrays",
    "spikegadgets_trodes",
    "trodes_native",
    "dlc_keypoints",
] = "prepared_arrays"
```

Add new field groups (validated only when their format is selected):

- `trodes_native`:
  - `imu_file: Path`             # existing parquet workflow reused
  - `position_file: Path`         # `.videoPositionTracking` binary
  - `camera_sync_file: Path | None`

- `dlc_keypoints`:
  - `imu_file: Path`             # parquet
  - `position_file: Path`         # DLC `.h5` (or `.csv`)
  - `camera_timestamps_file: Path | None`
  - `fps_cam: float | None`
  - `led1_bodypart: str`
  - `led2_bodypart: str`
  - `min_likelihood: float = Field(default=0.5, ge=0.0, le=1.0)`

`InputsConfig._validate_required_paths` is extended to require the
right fields per format. For `dlc_keypoints`, require exactly one of
`camera_timestamps_file` or `fps_cam` unless a Trodes camera-sync file
is provided. Camera calibration remains on `CameraConfig`:

```python
class CameraConfig(BaseModel):
    meters_per_pixel: float | None = Field(default=0.0022, gt=0.0)
    homography_file: Path | None = None
```

`meters_per_pixel` and `homography_file` are mutually exclusive after
accounting for the scalar default; see the homography plan.

### New ingester modules — `src/trodestrack/io/loaders/`

```
src/trodestrack/io/
  loaders/
    __init__.py
    trodes_native.py     # _load_trodes_native(config) -> PreparedSession
    dlc_keypoints.py     # _load_dlc_keypoints(config) -> PreparedSession
    pixel_to_meters.py   # shared scalar / homography conversion helper
```

`session.py:load_session` dispatches on `config.inputs.format`:

```python
if config.inputs.format == "prepared_arrays":
    session = _load_prepared_arrays(config)
elif config.inputs.format == "spikegadgets_trodes":
    session = _load_spikegadgets_trodes(config)
elif config.inputs.format == "trodes_native":
    session = _load_trodes_native(config)
elif config.inputs.format == "dlc_keypoints":
    session = _load_dlc_keypoints(config)
```

### Shared pixel→meters helper

```python
def pixel_to_meters_xy(
    pixels: np.ndarray,
    *,
    camera: CameraConfig,
) -> np.ndarray:
    """Apply scalar OR homography. Exactly one must be set."""
```

Validates mutual exclusion at call time; raises a clean
`ValueError` if both or neither are provided.

### Trodes loader internals

```python
def _load_trodes_native(config: SessionConfig) -> PreparedSession:
    inputs = config.inputs
    imu_df = pd.read_parquet(inputs.imu_file)         # reuse existing path
    position = readTrodesExtractedDataFile(inputs.position_file)
    led1_px, led2_px, t_cam = _parse_trodes_position(position)
    # Optional: align cam clock via inputs.camera_sync_file
    if inputs.camera_sync_file is not None:
        t_cam = _apply_trodes_camera_sync(t_cam, inputs.camera_sync_file)
    led1 = pixel_to_meters_xy(led1_px, camera=config.camera)
    led2 = pixel_to_meters_xy(led2_px, camera=config.camera)
    # ... reuse the rest of _load_spikegadgets_trodes flow:
    #     IMU SI conversion, sample-and-hold removal, time alignment,
    #     mask construction, calibration diagnostics, safety check.
    return PreparedSession(...)
```

The trick is that Trodes binary format already gives meaningful
NaN markers for dropouts; preserve them through the conversion.

### DLC loader internals

```python
def _load_dlc_keypoints(config: SessionConfig) -> PreparedSession:
    inputs = config.inputs
    df = _read_dlc_table(inputs.position_file)        # h5 or csv
    # Select LED1/LED2 columns by bodypart name.
    led1_px, conf1 = _extract_dlc_bodypart(df, inputs.led1_bodypart)
    led2_px, conf2 = _extract_dlc_bodypart(df, inputs.led2_bodypart)
    # Mask frames below min_likelihood.
    mask_led1 = conf1 >= inputs.min_likelihood
    mask_led2 = conf2 >= inputs.min_likelihood
    led1_px[~mask_led1] = np.nan
    led2_px[~mask_led2] = np.nan
    # Build conf_cam = [c1, c1, c2, c2] (matches the existing
    # _load_leds layout).
    conf_cam = np.column_stack([conf1, conf1, conf2, conf2])
    # Frame times: explicit timestamps file wins; otherwise use fps_cam.
    t_cam = _load_dlc_frame_times(inputs)
    # Pixel→meters.
    led1 = pixel_to_meters_xy(led1_px, camera=config.camera)
    led2 = pixel_to_meters_xy(led2_px, camera=config.camera)
    # ... time alignment + IMU + safety check.
    return PreparedSession(...)
```

DLC's frame index is an integer; require a separate
`camera_timestamps_file` if Trodes-style hardware sync is available,
or fall back to `t_cam = frame_idx / fps_cam` only when `fps_cam` is
configured explicitly.

## Milestones

### Milestone 1 — Shared helpers

- `pixel_to_meters_xy` helper with both scalar and homography
  paths, validated mutually exclusive.
- Unit tests for shape / dtype / homography roundtrip.
- No format dispatch yet; helper is callable in isolation.

**Exit criteria:** `tests/io/test_pixel_to_meters.py` green.

### Milestone 2 — Trodes native loader

- `_load_trodes_native` + minimal `_parse_trodes_position` +
  optional `_apply_trodes_camera_sync` (skip if no sync file).
- Schema additions for `inputs.format == "trodes_native"`.
- Sample fixture: a tiny committed `*.videoPositionTracking` file
  under `tests/fixtures/` (small, ≤10 KB). Do not gitignore it unless
  the repo uses an explicit `!tests/fixtures/...` allow-list rule.
- Unit + scenario tests: round-trip a 30-second Trodes session
  through the loader and run an EKF; verify no NaN/Inf and that
  the safety check passes.

**Exit criteria:** `tests/io/test_trodes_native_loader.py` green;
`SessionConfig.model_validate({"inputs": {"format":
"trodes_native", ...}})` passes.

### Milestone 3 — DLC keypoint loader

- `_load_dlc_keypoints` + `_extract_dlc_bodypart` (handles the
  multi-index column lookup).
- Schema additions for `inputs.format == "dlc_keypoints"`.
- Sample fixture: a hand-built DLC `.h5` from a synthetic
  `simulate_rat_imu` session (sim → fake DLC h5).
- Unit + scenario tests.

**Exit criteria:** `tests/io/test_dlc_keypoints_loader.py` green;
likelihood-gating round-trip.

### Milestone 4 — CLI integration

- `trodestrack online --config session.yaml` and `smooth --config`
  honor the new formats. No new flags.
- Real-data smoke against a user-supplied Trodes export and a
  user-supplied DLC export.
- Doc snippets in `docs/getting-started/python-api.md` showing
  one Trodes config and one DLC config.

**Exit criteria:** end-to-end run reproduces the same outputs as
the user's hand-converted parquet pipeline (within numerical
tolerance from the homography).

### Milestone 5 — Documentation

- README "Real-data ingest" section now lists four input formats.
- `examples/session_dlc_keypoints.yaml` and
  `examples/session_trodes_native.yaml` templates.
- Note in `docs/TROUBLESHOOTING.md` for common per-format errors
  (missing bodypart name, unrecognized Trodes binary header).

**Exit criteria:** mkdocs strict build clean; both example YAMLs
runnable with sample fixtures.

## Validation Matrix

| Test | Layer | Asserts |
|---|---|---|
| `pixel_to_meters_xy` scalar roundtrip | `loaders/pixel_to_meters.py` | meters → pixels → meters identity |
| `pixel_to_meters_xy` homography roundtrip | helper | corners map correctly |
| Trodes loader shape | `_load_trodes_native` | `t_cam.shape == led1.shape[:1]` |
| Trodes loader NaN preservation | helper | dropout markers preserved |
| DLC loader bodypart selection | `_extract_dlc_bodypart` | unknown bodypart → clean `KeyError` upgraded to `ValueError` |
| DLC loader likelihood gating | helper | frames below `min_likelihood` produce NaN positions |
| End-to-end Trodes scenario | EKF | no NaN/Inf in filter output |
| End-to-end DLC scenario | EKF | no NaN/Inf in filter output |
| Schema strictness | config | unknown fields per format → `ValidationError` |
| Mutually exclusive scalar/homography | helper | both set or neither set → `ValueError` |

## Metrics

- **Coverage**: every PRD-listed input format
  (`prepared_arrays / spikegadgets_trodes / trodes_native /
  dlc_keypoints`) has a working loader and a passing scenario
  test.
- **Compatibility**: existing config workflow tests stay green.
- **Real-data parity**: a Trodes session converted via
  `trodes_native` produces the same filter output (to ≤1 mm
  position RMSE) as the same session converted by hand to
  `spikegadgets_trodes` parquet.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Trodes-python-tools API drift across versions | Pin a minimum version; smoke-test against the latest compatible release; the helper is small enough to vendor if needed. |
| DLC `.h5` schema varies (multi-animal, different scorer name) | Extract by bodypart name explicitly; fail clearly if the structure doesn't match the documented single-animal layout. |
| Pixel→meters accuracy depends on user homography | Validate at load time that LED-pair distance after conversion is within ±20% of `led_distance` (config field); flag clearly otherwise. |
| Camera-clock sync ambiguity in Trodes exports | Make `camera_sync_file` optional with a documented warning when absent ("camera timestamps assumed already in IMU frame"). |
| New runtime deps for `.h5` reading | Pandas `read_hdf` requires PyTables (`tables`). Prefer CSV fixtures for mandatory CI, lazy-import `tables` for `.h5`, and raise a friendly install error when a user selects an HDF5 DLC file without the optional dependency. For Trodes binary, lazy-import and version-pin `trodes_python_tools` only when `trodes_native` is selected. |

## Rollout Strategy

- Trodes loader (Milestone 2) ships first — existing Trodes users
  benefit immediately.
- DLC loader (Milestone 3) ships in a second PR; smaller blast
  radius.
- Both go behind format-discriminated config; users on the
  existing `prepared_arrays` / `spikegadgets_trodes` paths see
  zero behavior change.
- Real-data smoke required against one user dataset per format
  before announcing.

## Documentation Updates

- README "Real-data ingest" section: enumerate the four formats
  with one-line summaries and a YAML snippet for each.
- `docs/getting-started/python-api.md`: worked example for each
  new format with the full config.
- `examples/session_trodes_native.yaml`,
  `examples/session_dlc_keypoints.yaml`: runnable templates.
- `docs/TROUBLESHOOTING.md`: per-format error symptoms.

## Open Questions

1. Should the Trodes loader auto-detect the
   `.videoPositionTracking` schema across Trodes versions, or
   require the user to specify a schema variant? Default:
   auto-detect from the file header; fall back to the documented
   format with a warning.
2. For DLC, what's the expected behavior when the file contains
   only one bodypart instead of two LEDs? Reject at schema time
   (a single-LED workflow is a separate change to the camera
   measurement model, not a loader concern).
3. Should we ship `trodestrack convert --from trodes_native --to
   prepared_arrays`? Probably not — it duplicates the loader's
   work and adds CLI surface for marginal benefit.

## Estimated Effort

- ~600 LOC source + ~400 LOC tests + ~150 lines docs.
- 1–2 weeks per format (Trodes and DLC are roughly equal-sized).
- One new optional runtime dep (Trodes-python-tools, lazy
  import).
- Depends on the `geom/` plan for first-class homography support;
  scalar `meters_per_pixel` is enough to ship without it.

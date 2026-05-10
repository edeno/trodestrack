# `geom/` Module + Homography Tool + `trodestrack calib-homography` CLI

## Status

Not started. PRD §8 lists `geom/` as a top-level package; PRD §9
specifies `trodestrack calib-homography --video frame.png --out
homography.yaml`; in that PRD text, `--video` is being used for a still
frame path. This plan makes the still-image path explicit as
`trodestrack calib-homography --image frame.png --out
homography.yaml`, with `--video session.mp4 --frame-index N` as an
optional convenience when the `video` extra is installed. Neither the
CLI nor the `geom/` package exists today. The existing
`meters_per_pixel` scalar in `CameraConfig` covers
orthogonal-camera setups (the bundled Arthur slice) but breaks
under perspective: for a tilted camera the scale factor differs
across the arena, and a single scalar gives wrong meters
everywhere except near the calibration anchor.

## Goals

- Add a 2D homography helper (`pixel ↔ meters` mapping) usable
  from both the Python API and the YAML real-data workflow.
- Ship an interactive CLI for the common case: user clicks four
  arena corners on a sample frame, gets back a `homography.yaml`
  file ready to drop into a session config.
- Wire homography into the current `spikegadgets_trodes` loader and
  the shared pixel-conversion helper that the companion native-loader
  plan will reuse for future `trodes_native` / `dlc_keypoints`
  formats. Users with non-orthogonal cameras should be able to ingest
  pixel-coordinate position data without hand-converting.

## Non-Goals

- Full camera intrinsics calibration (focal length, distortion
  coefficients, full pinhole model). PRD §15 lists this under
  "extensibility (3D roadmap)"; out of scope for v1.
- Multi-camera setups (one homography per session).
- Online re-calibration during a session.
- Video-streaming corner detection (auto-detect the arena from a
  frame). The CLI is interactive; users click corners.
- Replacing `meters_per_pixel`. The scalar path stays for
  orthogonal-camera users (it's simpler, faster, and sufficient).

## Background

A 2D homography is a 3×3 matrix `H` mapping homogeneous pixel
coordinates `[u, v, 1]ᵀ` to homogeneous world coordinates
`[x, y, 1]ᵀ` (up to a scale factor). It's the right tool for any
**planar** scene viewed from any angle — i.e., a flat arena floor
viewed by a camera mounted at any tilt.

Given four point correspondences (pixel ↔ world), `H` is the
unique 8-DOF projective transform that maps them. With the
arena's four corners at known world positions (e.g., a 1 m × 1 m
square at corners `(0, 0), (1, 0), (1, 1), (0, 1)`) and the user
clicking those four corners in pixel space, we have a
well-defined projective fit. With exactly four correspondences, the
fit will usually interpolate those clicked points exactly; corner
residuals are useful for detecting numerical failure, not for proving
that the user clicked the correct physical corners. Calibration quality
comes from the overlay/grid sanity check and, when available, extra
landmarks beyond the four fit corners.

Math convention used by this plan:

```
λ_i · [x_i, y_i, 1]ᵀ = H_px_to_world @ [u_i, v_i, 1]ᵀ
```

stacked across N≥4 correspondences. Solve via direct linear
transform (DLT) + optional Levenberg-Marquardt refinement on
geometric error when N > 4.

Coordinate convention:

- Pixel inputs are image coordinates `(u, v) = (column, row)`, with
  origin at the top-left of the video frame unless the upstream
  position file documents a different convention.
- World outputs are arena-floor coordinates `(x, y)` in meters, using
  trodestrack's existing y-up filter convention.
- The homography itself absorbs any y-axis flip. Do not apply a
  separate `frame_height - y` conversion outside the homography path;
  the clicked pixel corners and their paired world corners define the
  complete mapping.

The helper is well-trodden. Existing solid-quality
implementations:

- OpenCV's `cv2.findHomography` (~200 LOC of wrapping; OpenCV
  brings ~100 MB of dependencies — too heavy for a runtime dep).
- scikit-image's `transform.ProjectiveTransform` (~50 LOC of
  wrapping; ~10 MB; reasonable).
- Hand-rolled DLT in NumPy (~100 LOC; zero new deps).

Recommend: hand-rolled DLT for both the runtime path and headless CLI
tests (no dependency weight). The interactive CLI can use matplotlib
for click capture and image loading; do not add scikit-image unless
implementation evidence shows the hand-rolled path is insufficient.

Existing neurospatial code provides useful implementation patterns,
but not the projective solver itself. In particular,
`/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/ops/transforms.py`
contains `VideoCalibration`, which is a good model for forward/inverse
transform access and serialization. Its tests under
`/Users/edeno/Documents/GitHub/neurospatial/tests/animation/` and
`/Users/edeno/Documents/GitHub/neurospatial/tests/ops/` are useful
models for coordinate-convention, roundtrip, and degenerate-landmark
coverage. Do not adapt neurospatial's `calibrate_from_landmarks` /
`estimate_transform` as the homography core: those functions estimate
rigid / similarity / affine transforms, not true projective
homographies.

Neurospatial's annotation workflow is useful for UX patterns, not as a
dependency. It treats video-frame annotation as first-class, loads a
chosen `frame_index`, keeps annotation state visible, supports
delete/rename/save interactions, and has explicit `(row, col)` napari
to `(x, y)` video-coordinate conversion. Relevant files are:

- `/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/annotation/core.py`
  (`annotate_video`, frame-index loading, result processing).
- `/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/annotation/_napari_widget.py`
  (visible state, labels, delete/save interactions).
- `/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/annotation/converters.py`
  (`(row, col)` shape vertices to `(x, y)` video pixels).
- `/Users/edeno/Documents/GitHub/neurospatial/src/neurospatial/annotation/io.py`
  (LabelMe/CVAT import patterns; out of scope for v1).

For trodestrack v1, keep the lighter matplotlib point-picker, but
borrow those interaction ideas: explicit frame selection, labeled
expected clicks, undo/reset/cancel, save confirmation, and tests for
coordinate order and cancellation. Do not require napari / magicgui for
this calibration CLI.

## Design Principles

- **Two artifacts, one math.** The `geom/` library does the math
  (DLT fit, apply homography, validate). The CLI is a thin
  matplotlib-based interactive wrapper.
- **YAML output.** The CLI emits a `homography.yaml` with the
  3×3 matrix, the four world-corner positions used for the fit,
  the user-provided pixel corners, a residual-RMS diagnostic, and
  enough coordinate-convention metadata to avoid hidden y-flip
  assumptions. YAML stays human-readable so users can edit /
  version-control it.
- **Loader-side reuse.** The current `spikegadgets_trodes` loader
  multiplies LED pixel columns by `camera.meters_per_pixel` inline in
  `io/session.py`. This plan extracts that conversion into a small
  `pixel_to_meters_xy` helper that accepts either the scalar path or a
  3×3 homography matrix; the companion native-loaders plan reuses the
  helper instead of creating a second conversion path.
- **Validation at load time.** Pydantic schema accepts
  `homography_file: Path | None`; `load_session_config` resolves it
  relative to the YAML file, and `load_session` reads and validates it
  (3×3 finite, non-singular) because validation depends on file
  contents.
- **No required OpenCV dependency.** Hand-roll DLT in NumPy / SciPy.
  The still-image path has no new dependency; only optional
  video-frame extraction uses the existing `video` extra.

## Current Integration Points

These facts are true in the current trodestrack package and should be
treated as implementation constraints:

- CLI registration happens in `src/trodestrack/__init__.py::main`;
  subcommands expose `add_*_parser` functions in `src/trodestrack/cli/`.
- `CameraConfig` lives in `src/trodestrack/config/schemas.py` and
  currently has `meters_per_pixel: float = Field(default=0.0022, gt=0.0)`.
- `load_session_config` parses YAML, calls `SessionConfig.model_validate(raw)`,
  then `_resolve_paths`; it currently resolves input/output/TTL paths,
  but not camera calibration paths.
- `_load_leds` in `src/trodestrack/io/session.py` currently multiplies
  LED pixel columns by `camera.meters_per_pixel` inline.
- The only current configured input formats are `prepared_arrays` and
  `spikegadgets_trodes`; `trodes_native` and `dlc_keypoints` belong to
  the companion native-loaders plan.
- `pyproject.toml` already defines an optional `video` extra with
  `opencv-python`, so `--video --frame-index` should use lazy `cv2`
  imports and actionable errors instead of adding another video reader.

## Architecture

### New package — `src/trodestrack/geom/`

```
src/trodestrack/geom/
  __init__.py
  homography.py         # core math
  arena.py              # arena-shape helpers (bounds, scaling)
```

`homography.py` contract:

```python
@dataclass(frozen=True)
class Homography:
    matrix: np.ndarray            # (3, 3), pixel homogeneous column -> world
    pixel_corners: np.ndarray     # (n_points, 2), user-clicked
    world_corners: np.ndarray     # (n_points, 2), known anchors in meters
    residual_rms_pixels: float   # inverse reprojection: world anchors -> pixels
    residual_rms_world_m: float   # forward reprojection: pixel anchors -> world
    pixel_coordinate_origin: str = "top_left"
    pixel_coordinate_order: str = "col_row"

    @classmethod
    def from_correspondences(
        cls,
        pixel_corners: np.ndarray,    # (≥4, 2)
        world_corners: np.ndarray,    # (≥4, 2)
    ) -> Homography:
        """DLT + optional LM refinement."""

    def pixel_to_world(
        self, pixels: np.ndarray
    ) -> np.ndarray:
        """Map (n, 2) pixels to (n, 2) world meters."""

    def world_to_pixel(
        self, world_xy: np.ndarray
    ) -> np.ndarray:
        """Map (n, 2) world meters back to (n, 2) pixels."""

    def to_yaml(self) -> dict[str, object]: ...

    @classmethod
    def from_yaml(cls, data: dict[str, object]) -> Homography: ...

    @classmethod
    def from_yaml_file(cls, path: Path) -> Homography: ...
```

`arena.py`: helpers like `arena_bounds_from_world_corners` to
support the existing arena-bounds-check path in QA.

### CLI subcommand — `src/trodestrack/cli/calib_homography.py`

Add `add_calib_homography_parser` in this module and register it from
`src/trodestrack/__init__.py` alongside `online`, `smooth`, and
`report`. This matches the current CLI pattern; `src/trodestrack/cli/__init__.py`
is not the command-registration site today.

```
trodestrack calib-homography
    --image frame.png
    # or: --video session.mp4 --frame-index 100
    --arena-corners 0,0 1.0,0 1.0,1.0 0,1.0    # world meters
    --out homography.yaml
    [--label-corners "BL,BR,TR,TL"]            # cosmetic labels
```

Workflow:

1. Loads a still image from `--image` using existing matplotlib image
   support, or extracts one frame from `--video --frame-index` using
   `cv2.VideoCapture` when the existing optional `video` extra is
   installed.
2. Opens an interactive figure; the user clicks 4 corners in the
   order specified by `--label-corners`.
3. Fits the homography via `Homography.from_correspondences`.
4. Overlays the back-projected world grid on the frame so the
   user can sanity-check.
5. Prompts for confirmation, then writes `homography.yaml`.

The interactive prompt should be a small matplotlib point picker rather
than a bare `fig.ginput` call. It keeps clicked points and expected
labels visible in the figure, supports undo/reset/cancel, and requires
an explicit confirmation after the grid overlay is rendered. Headless
mode (`--headless --pixel-corners u1,v1 u2,v2 u3,v3 u4,v4`) skips the
click capture and runs the fit directly — required for CI and
reproducibility.

### Schema additions — `src/trodestrack/config/schemas.py`

Camera calibration lives on `CameraConfig`, not on individual input
formats:

```python
homography_file: Path | None = Field(default=None, description="...")
```

**Approach to mutual exclusion (avoids breaking existing configs):**
keep `meters_per_pixel: float = 0.0022` as the existing scalar default.
Use a raw-YAML precheck in `load_session_config` plus
`resolve_camera_calibration` in the loader path to pick the active
calibration based on which fields the user *explicitly* set:

- `homography_file is None` → scalar path; `meters_per_pixel` is
  active (default or user-set).
- `homography_file is not None` → homography path; `meters_per_pixel`
  is ignored. If the user *also* explicitly sets `meters_per_pixel`
  (i.e., the YAML contains both keys), raise `ValueError` from
  `load_session_config` with a clear mutual-exclusion message — this
  is the only mutex-fail case. The existing default is silently
  ignored when a homography is configured.

Pydantic validation still enforces types, finite values, and
`extra="forbid"` after the raw-YAML precheck. It should not be the
mechanism that detects "explicitly set" for mutual exclusion:
`load_session_config` checks the parsed YAML for the
`camera.meters_per_pixel` key before validation; if both
`meters_per_pixel` and `homography_file` are present in the raw dict,
raise. Otherwise pass through to Pydantic.

`_resolve_paths` must also resolve `camera.homography_file` relative to
the session YAML path, matching how it already resolves `inputs.*`,
`outputs.output_dir`, and `ttl_events.events_file`.

This avoids deprecating the scalar default and keeps existing
prepared-array / spikegadgets configs unchanged.

### Loader changes — `src/trodestrack/io/pixel_to_meters.py`

Create this helper module and update `_load_leds` in
`src/trodestrack/io/session.py` to use it. Resolve the active camera
calibration once at load time, then pass the resolved values to
conversion helpers. This keeps the default scalar from conflicting with
a configured homography:

```python
@dataclass(frozen=True)
class ResolvedCameraCalibration:
    meters_per_pixel: float | None
    homography: Homography | None


def resolve_camera_calibration(
    camera: CameraConfig,
) -> ResolvedCameraCalibration:
    """Return exactly one active pixel-to-world calibration."""
```

`load_session_config` implements the raw-YAML explicitness rule above
before `SessionConfig.model_validate(raw)` discards raw-key presence.
After that, `resolve_camera_calibration` only needs the resolved
`CameraConfig`: if `homography_file` is configured, load
`Homography.from_yaml_file(camera.homography_file)` and return
`(None, homography)` even though the Pydantic model still carries the
scalar default; otherwise return `(camera.meters_per_pixel, None)`.

```python
def pixel_to_meters_xy(
    pixels: np.ndarray,
    *,
    calibration: ResolvedCameraCalibration,
) -> np.ndarray:
    if (calibration.meters_per_pixel is None) == (calibration.homography is None):
        raise ValueError(
            "Provide exactly one of meters_per_pixel or homography."
        )
    if calibration.meters_per_pixel is not None:
        return pixels * calibration.meters_per_pixel
    return calibration.homography.pixel_to_world(pixels)
```

### Diagnostic — `src/trodestrack/qa/`

Add `qa.homography_diagnostics`: render a debug image showing the
back-projected world grid on the calibration frame, plus the
fit residual per landmark. For exactly four landmarks, the overlay is
the primary quality check because the fitted homography should pass
through the clicked points. Dropped into the YAML real-data run's
`session_diagnostics.json` and as `homography_overlay.png` next to the
run outputs.

## Milestones

### Milestone 1 — `geom/homography.py` core math

- DLT fit (NumPy SVD).
- Optional LM refinement on geometric error
  (`scipy.optimize.least_squares`) only when more than four
  correspondence pairs are provided.
- `pixel_to_world`, `world_to_pixel`, and roundtrip helpers.
- `to_yaml` / `from_yaml`.
- Unit tests:
  - DLT recovers a known homography exactly from 4
    perfect correspondences.
  - With N > 4 noisy correspondences, refinement reduces residual
    without worsening held-out landmark error.
  - Roundtrip pixel→world→pixel within tolerance.
  - Pixel coordinate convention is explicit: image-space `(col, row)`
    top-left corners map to y-up world coordinates with no extra
    downstream y-flip.
  - YAML serialization roundtrip.
  - Singular / colinear corners raise a clean `ValueError`.

**Exit criteria:** `tests/geom/test_homography.py` green; helper
callable in isolation.

### Milestone 2 — Loader wiring

- `Homography.from_yaml` integrated into the loader path.
- Raw-YAML mutual-exclusion check plus `resolve_camera_calibration`.
- Update `pixel_to_meters_xy` to accept a resolved camera calibration.
- Scenario test: a `spikegadgets_trodes` session with a
  homography file produces sane filter output (no NaN/Inf,
  position RMSE within tolerance against a hand-converted
  baseline).

**Exit criteria:** `tests/io/test_homography_loader_integration.py`
green; both `meters_per_pixel` and `homography_file` paths
exercised in CI.

### Milestone 3 — `trodestrack calib-homography` CLI

- New parser + subcommand in `cli/calib_homography.py`.
- Still-image input (`--image`) plus optional video-frame input
  (`--video --frame-index`) using the existing optional `video` extra.
- Interactive matplotlib point picker with labeled clicks,
  undo/reset/cancel, and confirmation after overlay rendering.
- Headless mode (`--pixel-corners`) for CI and scripts.
- Diagnostic overlay rendering.
- YAML output with metadata (timestamp, user-provided corners,
  forward and inverse residual RMS).
- Unit tests for headless mode, video-frame extraction, cancellation,
  wrong point counts, and coordinate order; smoke test for interactive
  mode via patched picker events.

**Exit criteria:** `tests/cli/test_calib_homography.py` green;
`trodestrack calib-homography --help` documents the contract.

### Milestone 4 — Diagnostic image and metadata

- `homography_overlay.png` written by the calibration CLI and
  optionally re-rendered by the YAML real-data run when a
  homography file is configured.
- `session_diagnostics.json` gains a `homography` block with
  `residual_rms_pixels`, `residual_rms_world_m`, `world_corners`, etc.

**Exit criteria:** real-data run on the Arthur slice with a
homography config produces the overlay image and JSON metadata.

### Milestone 5 — Documentation

- New section in `docs/getting-started/python-api.md`: "Camera
  calibration with `trodestrack calib-homography`".
- `examples/calibrate_homography_demo.py`: scripted example
  using a synthetic frame.
- README "Real-data ingest" section now mentions homography as an
  alternative to `meters_per_pixel`.
- `docs/TROUBLESHOOTING.md`: common calibration errors
  (large residual, singular fit, mismatched corner order).

**Exit criteria:** mkdocs strict build clean; example script runs
end-to-end.

## Validation Matrix

| Test | Layer | Asserts |
|---|---|---|
| DLT exact recovery | `geom/homography.py` | known H recovered to machine precision |
| N > 4 noise robustness | helper | refinement reduces fit residual without worsening held-out landmark error |
| Pixel↔world roundtrip | helper | `world_to_pixel(pixel_to_world(p)) == p` to tolerance |
| Coordinate convention | helper | image `(col,row)` top-left pixels map to y-up world points with no external y-flip |
| Singular corners reject | helper | colinear pixel corners → `ValueError` |
| YAML roundtrip | helper | `from_yaml(to_yaml(h)) == h` |
| Schema mutual exclusion | config | both explicit `meters_per_pixel` and `homography_file` in YAML → clear `ValueError` |
| Homography path resolution | config | relative `camera.homography_file` resolves relative to the session YAML |
| Loader scenario | EKF | session with homography produces no NaN/Inf, RMSE within tolerance |
| CLI headless mode | CLI | `--pixel-corners` produces correct YAML |
| CLI video frame selection | CLI | `--video session.mp4 --frame-index N` calibrates against the selected frame and rejects out-of-range indices clearly |
| CLI interactive smoke | CLI (mocked) | patched picker events produce correct YAML and respect labeled click order |
| CLI cancellation/reset | CLI (mocked) | cancel exits without YAML; reset clears points before save |

## Metrics

- **Calibration accuracy**: for N > 4 calibration landmarks,
  reprojection residual RMS ≤ 2 pixels on a clean synthetic case and
  ≤ 5 pixels on a realistic frame with click jitter. For the default
  four-corner workflow, residual RMS is not treated as proof of
  calibration quality; the required diagnostic is a back-projected
  world-grid overlay that makes wrong corner order and gross geometry
  errors visually obvious.
- **Filter accuracy parity**: a session calibrated via homography
  produces position RMSE within 1 cm of a hand-calibrated
  ground-truth baseline.
- **CLI usability**: full calibration round (load frame, click
  corners, write YAML) takes ≤2 minutes for a first-time user.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| User clicks corners in wrong order | `--label-corners` argument names the expected order; CLI shows the labels as overlaid prompts during click capture. |
| Non-planar arena (e.g., walls slope inward) breaks the planar-homography assumption | Document explicitly that homography assumes a flat floor; flag large residual RMS as a warning. |
| matplotlib backend on headless CI | Headless mode (`--pixel-corners`) bypasses interactive click; CI never invokes the GUI path. |
| Optional video dependency expands runtime footprint | Keep still-image `--image` path dependency-free; gate `--video --frame-index` behind the existing `video` extra and raise an actionable import error if unavailable. |
| Click jitter dominates accuracy | Encourage extra landmarks when accuracy matters; N > 4 refinement can reduce jitter effects, while the overlay is the primary quality check for four-corner fits. |
| Inverse homography numerically unstable for near-singular fits | DLT validates condition number; reject fits below threshold with a clear error. |

## Rollout Strategy

- `geom/` ships as a runtime module; users on the existing
  `meters_per_pixel` path see no behavior change.
- CLI ships behind `trodestrack calib-homography`; users who
  don't run it never see it.
- Real-data validation against one user-supplied tilted-camera
  session before announcing.
- After one round of user feedback, consider a webcam-driven
  calibration helper (out of scope for v1).

## Documentation Updates

- New section in `docs/getting-started/python-api.md`: "Camera
  calibration with `trodestrack calib-homography`".
- README "Real-data ingest" section now lists homography as a
  pixel→meters option.
- `examples/calibrate_homography_demo.py` runnable script.
- `docs/TROUBLESHOOTING.md` calibration-error symptoms.

## Open Questions

1. Should we ship a "wizard" mode that walks the user through
   the calibration flow with prompts? Probably no for v1 —
   the click-and-confirm flow is already minimal.
2. Multi-arena support (one homography per arena region)? Out of
   scope; users with multiple arenas run separate sessions.
3. Should external LabelMe/CVAT imports be accepted for calibration
   landmarks? Probably no for v1; neurospatial has useful import
   patterns, but the four-corner CLI should remain the first path.

## Estimated Effort

- ~400 LOC source + ~250 LOC tests + ~100 lines docs + CLI
  parser.
- 1 week focused work for one engineer comfortable with
  matplotlib's interactive capture.
- Zero new required runtime deps. Still-image calibration uses existing
  matplotlib support; video-frame extraction uses the existing optional
  `video` extra rather than adding napari / magicgui / scikit-image.
- Sequencing: ship before or in parallel with the native loaders
  plan, since both `trodes_native` and `dlc_keypoints` benefit
  from homography support but can ship without it (using the
  scalar fallback).

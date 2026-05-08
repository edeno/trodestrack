# `geom/` Module + Homography Tool + `trodestrack calib-homography` CLI

## Status

Not started. PRD §8 lists `geom/` as a top-level package; PRD §9
specifies `trodestrack calib-homography --video frame.png --out
homography.yaml` as a CLI subcommand. Neither exists today. The
existing `meters_per_pixel` scalar in `CameraConfig` covers
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
- Wire homography into the existing loaders
  (`spikegadgets_trodes`, plus `trodes_native` / `dlc_keypoints`
  from the companion plan) so users with non-orthogonal cameras
  can ingest pixel-coordinate position data without hand-converting.

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
well-defined least-squares fit.

Math convention used by this plan:

```
λ_i · [x_i, y_i, 1]ᵀ = H_px_to_world @ [u_i, v_i, 1]ᵀ
```

stacked across N≥4 correspondences. Solve via direct linear
transform (DLT) + optional Levenberg-Marquardt refinement on
geometric error.

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

## Design Principles

- **Two artifacts, one math.** The `geom/` library does the math
  (DLT fit, apply homography, validate). The CLI is a thin
  matplotlib-based interactive wrapper.
- **YAML output.** The CLI emits a `homography.yaml` with the
  3×3 matrix, the four world-corner positions used for the fit,
  the user-provided pixel corners, and a residual-RMS diagnostic.
  YAML stays human-readable so users can edit / version-control
  it.
- **Loader-side reuse.** The `pixel_to_meters_xy` helper from
  the native-loaders plan accepts either a scalar
  `meters_per_pixel` or a 3×3 homography matrix. This plan
  populates the matrix path.
- **Validation at load time.** Pydantic schema accepts
  `homography_file: Path | None`; `load_session` reads and validates
  it (3×3 finite, non-singular) because validation depends on file
  contents.
- **No OpenCV dependency.** Hand-roll DLT in NumPy / SciPy.

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
    pixel_corners: np.ndarray     # (4, 2), user-clicked
    world_corners: np.ndarray     # (4, 2), known anchors in meters
    residual_rms_pixels: float

    @classmethod
    def from_correspondences(
        cls,
        pixel_corners: np.ndarray,    # (≥4, 2)
        world_corners: np.ndarray,    # (≥4, 2)
    ) -> Homography:
        """DLT + optional LM refinement."""

    def apply_to_pixels(
        self, pixels: np.ndarray
    ) -> np.ndarray:
        """Map (n, 2) pixels to (n, 2) world meters."""

    def apply_to_world(
        self, world_xy: np.ndarray
    ) -> np.ndarray:
        """Map (n, 2) world meters back to (n, 2) pixels."""

    def to_yaml(self) -> dict[str, object]: ...

    @classmethod
    def from_yaml(cls, data: dict[str, object]) -> Homography: ...
```

`arena.py`: helpers like `arena_bounds_from_world_corners` to
support the existing arena-bounds-check path in QA.

### CLI subcommand — `src/trodestrack/cli/calib_homography.py`

Argparse parser registered alongside `online`, `smooth`, `report`
in `__init__.py`:

```
trodestrack calib-homography
    --video frame.png
    --arena-corners 0,0 1.0,0 1.0,1.0 0,1.0    # world meters
    --out homography.yaml
    [--label-corners "BL,BR,TR,TL"]            # cosmetic labels
```

Workflow:

1. Loads `frame.png` with `matplotlib`.
2. Opens an interactive figure; the user clicks 4 corners in the
   order specified by `--label-corners`.
3. Fits the homography via `Homography.from_correspondences`.
4. Overlays the back-projected world grid on the frame so the
   user can sanity-check.
5. Prompts for confirmation, then writes `homography.yaml`.

The interactive prompt keeps state in the matplotlib figure;
clicks are captured via `fig.ginput(4, timeout=0)`. Headless
mode (`--headless --pixel-corners u1,v1 u2,v2 u3,v3 u4,v4`)
skips the click capture and runs the fit directly — required for
CI and reproducibility.

### Schema additions — `src/trodestrack/config/schemas.py`

Camera calibration lives on `CameraConfig`, not on individual input
formats:

```python
homography_file: Path | None = Field(default=None, description="...")
```

**Approach to mutual exclusion (avoids breaking existing configs):**
keep `meters_per_pixel: float = 0.0022` as the existing scalar default.
Add a `model_validator` that picks the active calibration based on
which fields the user *explicitly* set:

- `homography_file is None` → scalar path; `meters_per_pixel` is
  active (default or user-set).
- `homography_file is not None` → homography path; `meters_per_pixel`
  is ignored. If the user *also* explicitly sets `meters_per_pixel`
  (i.e., the YAML contains both keys), raise `ValidationError` —
  this is the only mutex-fail case. The existing default is
  silently ignored when a homography is configured.

Detecting "explicitly set" requires reading the raw YAML dict in
`load_session_config` (Pydantic's `model_fields_set` only catches
fields set via the constructor, not deserialization). The simplest
implementation: `load_session_config` checks the parsed YAML for the
`camera.meters_per_pixel` key before validation; if both
`meters_per_pixel` and `homography_file` are present in the raw dict,
raise. Otherwise pass through to Pydantic.

This avoids deprecating the scalar default and keeps existing
prepared-array / spikegadgets configs unchanged.

### Loader changes — `src/trodestrack/io/loaders/pixel_to_meters.py`

```python
def pixel_to_meters_xy(
    pixels: np.ndarray,
    *,
    meters_per_pixel: float | None,
    homography: Homography | None,
) -> np.ndarray:
    if (meters_per_pixel is None) == (homography is None):
        raise ValueError(
            "Provide exactly one of meters_per_pixel or homography."
        )
    if meters_per_pixel is not None:
        return pixels * meters_per_pixel
    return homography.apply_to_pixels(pixels)
```

### Diagnostic — `src/trodestrack/qa/`

Add `qa.homography_diagnostics`: render a debug image showing the
back-projected world grid on the calibration frame, plus the
residual error per corner. Dropped into the YAML real-data run's
`session_diagnostics.json` and as `homography_overlay.png` next
to the run outputs.

## Milestones

### Milestone 1 — `geom/homography.py` core math

- DLT fit (NumPy SVD).
- Optional LM refinement on geometric error
  (`scipy.optimize.least_squares`).
- `apply_to_pixels`, `apply_to_world`, and roundtrip helpers.
- `to_yaml` / `from_yaml`.
- Unit tests:
  - DLT recovers a known homography exactly from 4
    perfect correspondences.
  - LM refinement reduces residual on noisy inputs.
  - Roundtrip pixel→world→pixel within tolerance.
  - YAML serialization roundtrip.
  - Singular / colinear corners raise a clean `ValueError`.

**Exit criteria:** `tests/geom/test_homography.py` green; helper
callable in isolation.

### Milestone 2 — Loader wiring

- `Homography.from_yaml` integrated into the loader path.
- Schema mutual-exclusion validator.
- Update `pixel_to_meters_xy` to accept `Homography` objects.
- Scenario test: a `spikegadgets_trodes` session with a
  homography file produces sane filter output (no NaN/Inf,
  position RMSE within tolerance against a hand-converted
  baseline).

**Exit criteria:** `tests/io/test_homography_loader_integration.py`
green; both `meters_per_pixel` and `homography_file` paths
exercised in CI.

### Milestone 3 — `trodestrack calib-homography` CLI

- New parser + subcommand in `cli/calib_homography.py`.
- Interactive matplotlib click capture.
- Headless mode (`--pixel-corners`) for CI and scripts.
- Diagnostic overlay rendering.
- YAML output with metadata (timestamp, user-provided corners,
  residual RMS).
- Unit tests for headless mode; smoke test for interactive mode
  via patched `fig.ginput`.

**Exit criteria:** `tests/cli/test_calib_homography.py` green;
`trodestrack calib-homography --help` documents the contract.

### Milestone 4 — Diagnostic image and metadata

- `homography_overlay.png` written by the calibration CLI and
  optionally re-rendered by the YAML real-data run when a
  homography file is configured.
- `session_diagnostics.json` gains a `homography` block with
  `residual_rms_pixels`, `world_corners`, etc.

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
| LM noise robustness | helper | residual RMS reduces vs DLT-only on noisy inputs |
| Pixel↔world roundtrip | helper | `apply(apply_inverse(p)) == p` to tolerance |
| Singular corners reject | helper | colinear pixel corners → `ValueError` |
| YAML roundtrip | helper | `from_yaml(to_yaml(h)) == h` |
| Schema mutual exclusion | config | both active `meters_per_pixel` and `homography_file` set after default resolution → `ValidationError` |
| Loader scenario | EKF | session with homography produces no NaN/Inf, RMSE within tolerance |
| CLI headless mode | CLI | `--pixel-corners` produces correct YAML |
| CLI interactive smoke | CLI (mocked) | patched `ginput` produces correct YAML |

## Metrics

- **Calibration accuracy**: residual RMS ≤ 2 pixels on a clean
  calibration (known synthetic case); ≤ 5 pixels on a realistic
  frame with click jitter.
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
| Click jitter dominates accuracy | LM refinement reduces it; the residual RMS in the YAML metadata surfaces the issue. |
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

1. Should the CLI optionally accept the calibration frame from a
   video file instead of a still image? Probably yes — `--video
   session.mp4 --frame 100` is a one-liner if `imageio` is
   available; otherwise users `ffmpeg`-extract a frame
   themselves.
2. Should we ship a "wizard" mode that walks the user through
   the calibration flow with prompts? Probably no for v1 —
   the click-and-confirm flow is already minimal.
3. Multi-arena support (one homography per arena region)? Out of
   scope; users with multiple arenas run separate sessions.
4. How to handle corner-click cancellation? Default: matplotlib's
   right-click removes the last point; ESC cancels.
5. Should `Homography` support inverse transformation
   (world→pixel) as a public API? Probably yes — the diagnostic
   overlay needs it.

## Estimated Effort

- ~400 LOC source + ~250 LOC tests + ~100 lines docs + CLI
  parser.
- 1 week focused work for one engineer comfortable with
  matplotlib's interactive capture.
- One optional new dep (`scikit-image` or `imageio`) for the CLI
  if we want video-frame extraction; otherwise zero new runtime
  deps.
- Sequencing: ship before or in parallel with the native loaders
  plan, since both `trodes_native` and `dlc_keypoints` benefit
  from homography support but can ship without it (using the
  scalar fallback).

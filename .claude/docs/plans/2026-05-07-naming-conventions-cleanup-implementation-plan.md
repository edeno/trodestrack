# Public-API Naming Cleanup for Neuroscientist Users

## Status

Not started. The 2025-10-13 [Variable Naming Review](../VARIABLE_NAMING_REVIEW.md)
assessed the project as "acceptable with caveats" *under the
assumption that the audience is engineers maintaining filter
internals*. trodestrack's actual audience is **neuroscientists
analyzing their recorded data** — researchers who write Python
against the public API but rarely open `models/ekf.py`. Under
that audience, the review's "domain notation is fine, just
document it" verdict is too soft.

The package's public Python surface still uses control-theory /
Kalman-filter shorthand (`t_imu`, `U_imu`, `Z_cam_led1`,
`mask_cam`, `X_truth`) that requires a textbook glossary to
read. The YAML config and CLI flags are already
neuroscientist-friendly (`inputs.imu_timestamps`,
`--led1-positions`); the gap is the **Python function
signatures and the simulator output dictionary**, which every
example and QA snippet exposes.

## Goals

- **Make the public Python API readable without a textbook.**
  Neuroscientist-readable names for every argument, return-dict
  key, and example users encounter. Internal filter math
  (`P`, `Q`, `R`, `F`, `H`, `K`, `m`) stays — implementers
  maintaining the filter benefit from textbook parity.
- **Eliminate `mask_cam` polarity ambiguity** at the API
  boundary by introducing an unambiguously-named alias and
  documenting the contract once.
- **Provide migration paths**, not breaking changes. Existing
  scripts written against `t_imu`, `U_imu`, etc. keep working
  for at least one minor version.
- **Document the public/internal boundary** so contributors know
  which names belong where.

## Non-Goals

- **No rename of internal `P`/`Q`/`R`/`F`/`H`/`K`/`m`.** These
  are textbook notation and the audience for filter internals
  *is* engineers. Renaming them would make the code harder to
  cross-check against Särkkä, Bar-Shalom, etc.
- **No rename of `_jax`-suffixed JIT-internal variables.** They
  help distinguish host-side numpy from device-side JAX during
  tracing and never leak to users.
- **No "verbose mode" runtime toggle.** Two parallel name sets
  break grep and add maintenance surface.
- **No silent breakage.** Every rename ships with a backward-compat
  alias and a deprecation warning before removal.
- **No YAML / CLI changes.** Both already use descriptive names
  (`inputs.imu_timestamps`, `--imu-measurements`); only the
  Python surface needs work.

## Background

trodestrack's public Python surface today:

```python
result = extended_kalman_filter(
    ekf_config,
    sim["t_imu"],            # cryptic
    sim["U_imu"],            # cryptic
    sim["t_cam_exp"],        # what is _exp?
    sim["Z_cam_led1"],       # cryptic
    sim["Z_cam_led2"],       # cryptic
    sim["mask_cam"],         # polarity ambiguous
)
positions = result.filtered_means[:, layout.pos_idx]   # readable
```

The simulator output dict shipped to users:

```python
sim = simulate_rat_imu(config, seed=0)
# sim.keys():
#   t_imu, U_imu, bias_gyro, bias_accel_x, bias_accel_y
#   t_cam_exp, t_cam_obs, Z_cam_led1, Z_cam_led2, mask_cam,
#   swap_applied, X_truth
```

A neuroscientist reading this needs to know:

- `t_*` is timestamps in seconds (control-theory shorthand).
- `U_*` is "control input" — actually means IMU measurements.
- `Z_*` is "observation" — actually means LED pixel positions.
- `X_truth` is "state vector ground truth" — actually means
  `[x, y, vx, vy, theta]`.
- `t_cam_exp` vs `t_cam_obs` is exposure-time vs observation-time
  (camera-pipeline arcana).
- `mask_cam[i] == True` *might* mean "frame i is good" or
  "frame i is masked out"; the docstring doesn't say.

Every one of these maps to something a neuroscientist already
understands (`imu_timestamps`, `imu_measurements`, `led1_positions`,
`true_states`, `camera_exposure_times`, `camera_valid_mask`).
The current names exist because the original implementer
followed control-theory papers — appropriate for the *implementer*,
hostile to the *user*.

## Design Principles

- **Public API speaks the user's language.** Neuroscientists
  using the package see names that map to physical quantities:
  `imu_timestamps`, `imu_measurements`, `led1_positions`,
  `camera_valid_mask`, `true_states`.
- **Internal filter math stays.** `P`, `Q`, `R`, `H`, `K`, `m`
  inside `models/ekf.py` and `runtime/offline.py` keep their
  textbook letters — the audience for *that* code is engineers
  maintaining the filter, who do read Särkkä.
- **The boundary is the public API.** Function signatures,
  return-dict keys, example scripts — anything a user touches
  by reading the README or running an example. Anything inside
  a function body that doesn't escape is internal.
- **Aliases, not rewrites.** Existing scripts keep working.
  `extended_kalman_filter` accepts both `t_imu=...` and
  `imu_timestamps=...`. The simulator returns a dict that
  responds to both `sim["t_imu"]` and `sim["imu_timestamps"]`.
- **One canonical name per concept** going forward; the canonical name
  is the new friendly spelling and the alias is the old deprecated
  spelling.
- **Mask polarity contract documented and asserted at the API
  boundary**, not just in tests.

## Architecture

### 1. The boundary inventory

What's "public" and needs friendly names:

| Surface | Today | New (canonical) | Old (deprecated alias) |
| --- | --- | --- | --- |
| `extended_kalman_filter` arg | `t_imu` | `imu_timestamps` | `t_imu` (kw-only) |
| same | `U_imu` | `imu_measurements` | `U_imu` |
| same | `t_cam` | `camera_timestamps` | `t_cam` |
| same | `Z_cam_led1` | `led1_positions` | `Z_cam_led1` |
| same | `Z_cam_led2` | `led2_positions` | `Z_cam_led2` |
| same | `mask_cam` | `camera_valid_mask` | `mask_cam` |
| same | `conf_cam` | `led_confidence` | `conf_cam` |
| `rts_smoother` arg | same names as EKF | same renames | same |
| `extended_kalman_filter_3d` arg | same | same | same |
| `unscented_kalman_filter` arg | same | same | same |
| `simulate_rat_imu` return key | `t_imu` | `imu_timestamps` | `t_imu` |
| same | `U_imu` | `imu_measurements` | `U_imu` |
| same | `t_cam_exp` | `camera_exposure_times` | `t_cam_exp` |
| same | `t_cam_obs` | `camera_observation_times` | `t_cam_obs` |
| same | `Z_cam_led1` | `led1_positions` | `Z_cam_led1` |
| same | `Z_cam_led2` | `led2_positions` | `Z_cam_led2` |
| same | `mask_cam` | `camera_valid_mask` | `mask_cam` |
| same | `swap_applied` | `led_swap_events` | `swap_applied` |
| same | `bias_gyro` | `gyro_bias_truth` | `bias_gyro` |
| same | `bias_accel_x` | `accel_x_bias_truth` | `bias_accel_x` |
| same | `bias_accel_y` | `accel_y_bias_truth` | `bias_accel_y` |
| same | `X_truth` | `true_states` | `X_truth` |
| `EKFResult` field | `filtered_means` | (already friendly) | n/a |
| result log-likelihood fields | `marginal_loglik` | leave unchanged in this plan | n/a |

What stays internal and **does not change**:

- `models/ekf.py`, `runtime/offline.py`, `models/filter_common.py`:
  `P`, `Q`, `R`, `F`, `H`, `K`, `m`, `S`, `predict_step`,
  `update_step` internals. These are filter-implementation code.
- `_jax`-suffixed variables inside JIT'd scan bodies.
- All `models/sensors/*.py` measurement-model internals.

### 2. Function-signature compatibility shim

Each public filter function gets a wrapper that accepts both
old and new names. Pattern:

```python
def extended_kalman_filter(
    ekf_config: EKFConfig,
    imu_timestamps: np.ndarray | None = None,
    imu_measurements: np.ndarray | None = None,
    camera_timestamps: np.ndarray | None = None,
    led1_positions: np.ndarray | None = None,
    led2_positions: np.ndarray | None = None,
    camera_valid_mask: np.ndarray | None = None,
    *,
    led_confidence: np.ndarray | None = None,
    initial_state: EKFState | None = None,
    led_distance: float | None = None,
    # Deprecated aliases (kw-only, emit DeprecationWarning if used)
    t_imu: np.ndarray | None = None,
    U_imu: np.ndarray | None = None,
    t_cam: np.ndarray | None = None,
    Z_cam_led1: np.ndarray | None = None,
    Z_cam_led2: np.ndarray | None = None,
    mask_cam: np.ndarray | None = None,
    conf_cam: np.ndarray | None = None,
) -> EKFResult:
    imu_timestamps = _resolve_alias(
        "imu_timestamps", imu_timestamps, "t_imu", t_imu
    )
    # ... same for the rest
    return _extended_kalman_filter_impl(
        ekf_config, imu_timestamps, imu_measurements, ...
    )
```

`_resolve_alias` lives in `trodestrack/_compat.py` — a freestanding
module that imports only from the standard library (`warnings`).
**Import discipline**: `_compat.py` must not import from `cli/`,
`models/`, `runtime/`, `qa/`, `sim/`, `io/`, or `viz/` so that any of
those layers can import it without a circular dependency. The helper
is small (~30 LOC: one `_resolve_alias(canonical_name, canonical_value,
legacy_name, legacy_value) -> value` function plus a deprecation-warning
emitter). It emits `DeprecationWarning` when only the old name is used.

### 3. `SimOut` dict — bilingual subclass

The simulator is annotated as returning `SimOut`, currently a
`TypedDict` in `sim/utils.py`, and the implementation returns a plain
dict literal. Replace or rename that `TypedDict` before introducing a
runtime `SimOut(dict)` subclass that responds to both names without
copying data:

```python
class SimOut(dict):
    """Simulator output dict with bilingual keys.

    New code should use the friendly keys (``imu_timestamps``,
    ``led1_positions``, etc.). Legacy keys (``t_imu``, ``Z_cam_led1``,
    etc.) are accepted with a ``DeprecationWarning``.
    """

    _ALIASES = {
        "t_imu": "imu_timestamps",
        "U_imu": "imu_measurements",
        "t_cam_exp": "camera_exposure_times",
        ...
    }

    def __getitem__(self, key):
        if key in self._ALIASES:
            warnings.warn(
                f"sim['{key}'] is deprecated; use "
                f"sim['{self._ALIASES[key]}'] instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            key = self._ALIASES[key]
        return super().__getitem__(key)
```

`simulate_rat_imu` populates only the new keys; legacy access works
through `__getitem__`, `get`, and `__contains__` alias handling with a
warning on value access. `keys()` and serialization expose only the
canonical keys to avoid duplicating arrays. After two minor versions,
drop the alias map.

### 4. Mask polarity contract — explicit

Beyond renaming `mask_cam → camera_valid_mask`, document the
polarity in the new helper alias and surface it via type alias:

```python
# trodestrack/typing.py
CameraValidMask: TypeAlias = np.ndarray
"""``(N,)`` bool array. ``mask[i] == True`` means camera frame
``i`` is usable; ``False`` means dropped or invalid."""
```

The helper `cli/utils.validate_camera_mask` already enforces
boolean dtype + values in `{0, 1}`; this plan adds a one-line
polarity assertion to its docstring and to every public docstring
that still uses the old `mask_cam` name during the deprecation
period.

### 5. New canonical names doc — `docs/getting-started/glossary.md`

User-facing page:

- "What does each name in the API mean?" — a table mapping
  every public name to its physical interpretation, units,
  and shape.
- "Migrating from the old (cryptic) names" — diff snippets
  showing before/after for the most common patterns
  (loading a session, running EKF, plotting truth vs estimate).
- Examples updated to use new names; CHANGELOG entry pinning
  the migration window.

This replaces the engineer-flavored `docs/CONVENTIONS.md` from
the prior version of this plan; engineer-facing internal
notation is documented separately in a `CONTRIBUTING.md` section.

### 6. Update every example to canonical names

`examples/02_..._09_*.py`, every doc snippet under
`docs/getting-started/*.md`, the `simulate_rat_imu` examples in
the QA tests — all switch to new names. The aliases keep
existing user scripts working; we just update *our own*
examples to model the canonical pattern.

### 7. CONTRIBUTING.md — engineer-facing internal notation

Short section explaining *internal* notation:

- `P`, `Q`, `R`, `F`, `H`, `K`, `m` are textbook letters and
  map directly to Särkkä Ch. 5 / Bar-Shalom Ch. 5.
- `_jax` suffixes inside JIT'd scan bodies distinguish JAX
  arrays from host-side numpy.
- Capital `U`, `Z` are control-theory matrix notation (legacy
  only inside private helpers; do not use in new public APIs).

This is a contributor doc, not user-facing.

## Milestones

### Milestone 1 — Glossary + canonical-name docs

- New `docs/getting-started/glossary.md` documenting every
  public name with units and shape.
- README "Quickstart" section updated to use canonical names.
- mkdocs nav entry under "Getting Started".
- No code changes yet.

**Exit criteria:** glossary covers every public name; mkdocs
strict build clean.

### Milestone 2 — Type aliases + polarity contract

- `src/trodestrack/typing.py` with the alias set
  (`Timestamps`, `IMUMeasurements`, `LEDPositions`,
  `CameraValidMask`, `StateSeries`, `Covariance`, …) — semantic
  documentation for arrays.
- `CameraValidMask` docstring asserts polarity (`True == usable`).
- Re-export from `trodestrack.__init__`.
- Update `cli/utils.validate_camera_mask` docstring with the
  polarity contract.
- Test: `tests/typing/test_aliases_importable.py` smoke check.

**Exit criteria:** new module imports cleanly; aliases visible
in mkdocs autodoc.

### Milestone 3 — Filter-function alias shims

- Wrap `extended_kalman_filter`,
  `extended_kalman_filter_3d`, `unscented_kalman_filter`,
  `rts_smoother` with the new-name canonical signature plus
  kw-only deprecated aliases.
- `_resolve_alias` helper in freestanding `trodestrack/_compat.py`
  emits `DeprecationWarning` on legacy use; import contract documented
  (stdlib-only deps).
- Backward-compat tests: every existing test calling these
  functions with old names continues to pass (with a warning).
- Forward-compat tests: same call with new names produces
  identical output.

**Exit criteria:** no test failures; deprecation warnings fire
exactly once per legacy call site.

### Milestone 4 — `SimOut` bilingual dict

- Rename or replace the existing `SimOut` `TypedDict`, then return a
  runtime `SimOut(dict)` subclass that maps legacy keys to new keys
  with a `DeprecationWarning`.
- `simulate_rat_imu` populates the new keys directly (no copy
  on the new path).
- Tests: legacy access works with warning; new access is silent;
  `key in sim` works for old and new names; `sim.keys()` exposes only
  canonical names.

**Exit criteria:** examples that hand-wrote `sim["t_imu"]`
continue to work with a warning; updated examples use the new
keys cleanly.

### Milestone 5 — Update examples and docs

- Every example script under `examples/` switches to canonical
  names.
- Every doc snippet under `docs/` switches to canonical names.
- README quickstart reflects new names.
- A "Migrating from the old names" page links the deprecation
  cycle; users running pre-rename scripts see warnings and can
  follow the migration table.

**Exit criteria:** example scripts pass `pytest --doctest-modules`
(if doctests exist) or run without warnings; mkdocs strict
build clean.

### Milestone 6 — `CONTRIBUTING.md` for internal notation

- New `CONTRIBUTING.md` (or section in existing CLAUDE.md)
  describing the *internal* notation (`P`, `Q`, `R`, etc.) and
  the public/internal boundary policy: new code in
  `models/sensors/*.py` and similar internal paths uses
  textbook letters; new code in user-facing entry points uses
  the canonical descriptive names.

**Exit criteria:** doc references the boundary explicitly so
contributors know which names belong where.

### Milestone 7 — Deprecation removal (next release)

- Bump minor version.
- Remove the deprecated `t_imu` / `U_imu` / etc. kwargs and
  the `SimOut` alias map.
- Bump CHANGELOG with the migration table.

**Exit criteria:** post-release sweep stays green; users who
followed the migration warnings have nothing to fix.

## Validation Matrix

| Test | Layer | Asserts |
| --- | --- | --- |
| `test_aliases_importable` | typing | `from trodestrack import CameraValidMask` works |
| Legacy-keyword backward compat | EKF wrapper | calling with old names produces same output + emits 1 `DeprecationWarning` per kwarg |
| Forward-compat parity | EKF wrapper | calling with new names produces bitwise-identical output to old names |
| `SimOut` legacy access | sim | `sim["t_imu"]` works + emits warning |
| `SimOut` canonical access | sim | `sim["imu_timestamps"]` works silently |
| Glossary coverage | docs | every name in `extended_kalman_filter` signature appears in glossary |
| Mask polarity contract | tests | every public function with `mask_cam`/`camera_valid_mask` arg has the polarity sentence in its docstring |
| Examples don't warn | examples | running an example produces no `DeprecationWarning` |

## Metrics

- **Public API readability**: a neuroscientist reading the
  README quickstart understands every variable name without
  cross-reference.
- **Migration friction**: zero. Every legacy call site keeps
  working with a single-line `DeprecationWarning`.
- **Test count delta**: +8–15 tests (alias smoke + backward-compat
  parity per filter wrapper + SimOut bilingual access/membership).
- **API breakage during deprecation window**: zero.
- **API breakage post-removal**: limited to scripts that
  ignored two minor versions of `DeprecationWarning`.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Users miss the deprecation warnings and break post-removal | Deprecation cycle spans ≥ 2 minor versions; CHANGELOG entry; one-page migration guide; warnings are explicit about the canonical replacement. |
| Renames drift inside the codebase faster than the deprecation alias coverage | Single `_resolve_alias` helper used everywhere; one place to update; tests cover both paths. |
| Bilingual `SimOut` masks legitimate KeyErrors | Subclass only intercepts known legacy keys; unknown keys raise `KeyError` as before. |
| Internal `P`/`Q`/`R` show up in error tracebacks users see | Internal functions raise `ValueError` with descriptive messages already; the variable name in the traceback is implementation detail. Acceptable. |
| Docstring polarity-contract test creates a false sense of completeness | Test is a regression guard, not a substitute for review; flag in test docstring. |
| Two deprecation paths (function args + sim dict) increase test surface | Both go through `_resolve_alias` / one `_ALIASES` map; tests parametrize over the alias table. |

## Rollout Strategy

- **Two minor-version deprecation window**, not one. Users on
  longitudinal projects (multi-month analyses) need to see
  warnings on a second run before removal.
- Each milestone lands as its own PR — easy to review, easy to
  roll back.
- After Milestone 5, run a real-data smoke against the Arthur
  slice using both legacy and canonical names; confirm
  identical outputs.
- Communicate in the v1.0 release notes:
  "Renamed all public Python arguments and SimOut keys for
  clarity. Legacy names continue to work with deprecation
  warnings until v1.2."

## Documentation Updates

- New `docs/getting-started/glossary.md`: full mapping of public
  names to physical meaning, units, shape.
- New `docs/migrating-from-cryptic-names.md` (or section in
  glossary): diff snippets for the common patterns
  (load → filter → plot).
- README quickstart switches to canonical names.
- `CONTRIBUTING.md` (or CLAUDE.md section): public/internal
  naming boundary policy.
- CHANGELOG entry for each milestone.

## Open Questions

1. Is `imu_measurements` clearer than `imu_samples` for `U_imu`?
   `_measurements` is more precise but slightly longer. Default
   `imu_measurements` for parity with the YAML config field
   `inputs.imu_measurements`.
2. `gyro_bias_truth` vs `true_gyro_bias` — which reads better
   in the simulator output? Default `_truth` suffix to match
   the existing `X_truth → true_states` rename.
3. Should we provide a `simulate_rat_imu(..., legacy_keys=False)`
   flag to suppress legacy keys entirely? Probably not for
   v1; users wanting strict mode set
   `warnings.filterwarnings("error", category=DeprecationWarning)`.
4. Should result objects eventually add a `log_likelihood` property
   alias for `marginal_loglik`? Leave out of this cleanup. Result
   objects are NamedTuple-like today, and field renames would cascade
   through EKF, UKF, smoother, saved outputs, tests, and docs. Revisit
   only in a separate result-object API plan.
5. What about `compute_nees(...)` parameter `states_true`,
   `states_est`? Already descriptive; leave alone. Audit the
   QA module's signatures and rename only the cryptic ones.

## Estimated Effort

- ~500 LOC source (alias shims + `SimOut` subclass + type
  aliases) + ~250 LOC tests + ~250 lines docs (glossary +
  migration guide).
- 1–2 weeks focused work for one engineer comfortable with the
  Python warnings API and the existing test surface.
- Zero new dependencies. Zero numerical change. Zero API
  breakage during the deprecation window.
- Sequencing: ship after the beam-break / loaders / streaming
  milestones (those PRs already use canonical names internally;
  that earlier momentum minimizes the alias surface this plan
  has to cover).

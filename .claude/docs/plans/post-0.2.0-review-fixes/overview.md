# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

The fixes touch four areas of the live tree at `master` @ `d853bbc`. File:line refs below are anchors the executor uses to scope each edit precisely.

**IO loader (`io/session.py`, ~960 LOC):**
- `src/trodestrack/io/session.py:30-49` — `PreparedSession` dataclass; Phase 5 regroups the 4 TTL-related fields into a nested `EventChannel | None`.
- `src/trodestrack/io/session.py:471-499` — `_add_imu_calibration_diagnostics`; Phase 1 narrows the `try` at line 483 to only `_validate_calibration_for_fusion`.
- `src/trodestrack/io/session.py:526-535` — `_validate_time_vector`; Phase 4 enriches the error messages with recovery suggestions, modeled on `cli/utils.py:159-166`.
- `src/trodestrack/io/session.py:609-626` — `_load_leds`; Phase 1 tightens the truthy column-name check at line 622.
- `src/trodestrack/io/session.py:629-632` — `_index_or_time_column`; Phase 1 makes the missing-`time`-column case raise rather than silently using `df.index`.
- `src/trodestrack/io/session.py:635-639` — `_median_led_distance`; Phase 1 replaces the silent `0.04 m` fallback with a `ValueError` plus a documented opt-in.

**TTL events (`io/ttl_events.py`):**
- `src/trodestrack/io/ttl_events.py:206-222` — silent drop of edge-mismatched and out-of-range events; Phase 1 surfaces per-source kept/dropped counts.

**CLI surface (`cli/`):**
- `src/trodestrack/__init__.py:8` — `__version__ = "0.1.0"`; Phase 1 bumps to `0.2.0` to match `pyproject.toml`.
- `src/trodestrack/__init__.py:74-81` — subcommand registration; Phase 4 replaces `add_online_parser` with `add_filter_parser`.
- `src/trodestrack/__init__.py:35-38, 49-54` — top-level epilog refers to `online`; Phase 4 rewrites.
- `src/trodestrack/cli/online.py` (~480 LOC) — rename to `cli/filter.py` in Phase 4.
- `src/trodestrack/cli/smooth.py` — Phase 4 adds tqdm progress + moves `--use-heading-measurement` into the filter argparse group.
- `src/trodestrack/cli/report.py` — Phase 4 adds a helper that consumes `online`/`smooth` outputs directly, removing the dead-end documented in the current epilog.
- `src/trodestrack/cli/utils.py:16-54` — `friendly_cli_errors`; Phase 1 wires up the `TRODESTRACK_DEBUG=1` env var the docstring promises.

**Filter core (`models/`):**
- `src/trodestrack/models/filter_common.py:174-229` — `FilterCoreConfig.state_mode`; Phase 5 promotes to `Literal[...]`. Phase 5 also removes the silent `object.__setattr__` mutation at line 228-229.
- `src/trodestrack/models/filter_common.py:498-511` — `FilterState` NamedTuple; Phase 5 adds `FilterState.create(mean, cov, layout)` classmethod that runs `validate_initial_state`.
- `src/trodestrack/models/filter_common.py:2020-2043` — `update_zupt`'s dimension-only layout lookup; Phase 6 threads the active `layout` through the call signature.
- `src/trodestrack/models/filter_common.py:2315-2342` — `gaussian_log_likelihood` dead jitter fallback; Phase 6 removes.
- `src/trodestrack/models/ekf.py:1464-1490` — 3D IEKF Python `for` loop; Phase 6 converts to `lax.scan` to match the 2D pattern at `models/ekf.py:737-739`.
- `src/trodestrack/models/state_layout.py:22-93` — `StateLayout` dataclass; Phase 5 adds construction-time index validation.
- `src/trodestrack/models/ukf.py:455, 566, 706` — stale `(17, 8)` / `(17, 4)` shape annotations + stale line-number ref; Phase 3 fixes.
- `src/trodestrack/models/sensors/camera_position_3d.py:5-7` — "Arthur-specific" docstring; Phase 3 rewrites.
- `src/trodestrack/models/sensors/camera_position_3d.py:241-243` — `geometric_jacobian` uses `jacfwd`; Phase 6 replaces with analytic form mirroring the 2D model.

**QA & viz:**
- `src/trodestrack/qa/report.py:38-41` — `PRD_*` constants; Phase 3 renames to `TARGET_*` and updates all call sites.
- `src/trodestrack/qa/report.py:339-357, 466-483` — Phase 4 adds PASS/FAIL banner and fixes the green/red start/end markers.
- `src/trodestrack/qa/plots.py:189, 205, 240, 247, 291, 360, 378, 421, 428, 518-611` — Phase 3 renames PRD references; Phase 4 adds NEES/NIS histogram titles.
- `src/trodestrack/qa/metrics.py:4, 902, 937-961, 985` — Phase 3 rewrites PRD references in docstrings.
- `src/trodestrack/viz/components.py:1257-1322` — Phase 3 renames PRD legend labels; Phase 3 also sweeps the ~50 trivial "what" comments documented in the comment-analyzer report.

**Config schemas:**
- `src/trodestrack/config/schemas.py:206-254` — `FilterConfig.state_mode` is already `Literal`. Phase 5 mirrors that on `FilterCoreConfig`.
- `src/trodestrack/config/schemas.py:293-308` — `EventLocationSource`; Phase 5 adds `__post_init__` validation and tightens `source_type` to `Literal["beam","zone","rfid"]`.

**Tests:** New files in `tests/cli/`, `tests/filters/`, `tests/models/`, `tests/io/` per Phase 2. The existing test layout (mirrors `src/` structure) is preserved.

## Scope and dependency policy

### Goals

- Eliminate the three documented silent-failure paths in the IO loader (C1, C2) and the diagnostics try-too-wide (I1).
- Close the test-coverage gap on the 3D quaternion EKF/UKF path and the `report` CLI (C4, C5).
- Remove all remaining scaffolding-doc references from the live source tree (`PRD_*`, "Arthur-specific", stale line numbers, stale shape annotations).
- Replace the misleading `online` CLI command name with `filter`; bridge the `online`/`smooth` → `report` gap.
- Push runtime invariants into type construction (`Literal`s, `__post_init__` validation, classmethod constructors).
- Bring the 3D filter path to parity with the 2D path on JAX hygiene (`lax.scan` over Python loops, analytic Jacobians, layout threading).

### Non-Goals

- **No deprecation shims for renamed APIs.** Confirmed at planning time: no external dependents. Old names are deleted in the same PR that introduces new ones.
- **No new filter modes** — Phases 5/6 tighten the existing 2D/3D paths but don't add new state layouts or sensor types.
- **No CI configuration changes** unless a phase introduces tests that require a new marker.
- **No `docs/` site overhaul.** Phase 4 updates the README and top-level `--help`; the MkDocs site under `docs/user-guide/` is touched only where it directly references renamed CLI commands.
- **No reformatting outside touched files.** `ruff format` will normalize the touched files; unrelated files are left alone.
- **No mypy strict-mode promotion.** The project currently runs mypy as a manual check (per `CLAUDE.md`); we add the `Literal` types but don't tighten the global config.

### Dependency policy

`tqdm` is already a project dependency (`pyproject.toml:20`). No new runtime dependencies are introduced. Phase 2 may rely on `hypothesis` (already in `[dev]`) for property-based tests on the 3D path.

## Metrics

How the work is verified:

- **Phase 1**: every named silent-failure path raises in a test (no fallback values reach production). `__version__` matches `pyproject.toml`.
- **Phase 2**: line coverage for `cli/report.py` ≥ 80% (was 0%); `models/ekf.py::extended_kalman_filter_3d` reaches ≥ 80% line coverage and has at least one analytic-correctness test per scenario (stationary tilt, yaw-only motion, dropout drift, NEES).
- **Phase 3**: `grep -rE '\bPRD[_ ]'` against `src/` returns zero hits; `grep -E '\(17, [0-9]+\)' src/trodestrack/models/ukf.py` returns zero; no "Arthur" in `src/`.
- **Phase 4**: `trodestrack --help` shows `filter` (not `online`); `trodestrack filter --output-dir <X>` followed by `trodestrack report --run <X>` succeeds without manually constructing a `qa_inputs/` layout; tqdm progress visible during a >5s run.
- **Phase 5**: `FilterCoreConfig(state_mode="typo")` raises at construction (currently fails deep inside `get_layout`); `EventLocationSource(anchor=np.array([1.0]), ...)` raises at construction; `PreparedSession` has at most 1 optional TTL field (the `EventChannel`).
- **Phase 6**: `extended_kalman_filter_3d` JIT compile time on `tests/benchmark/test_ekf_3d_core_jit.py` decreases or stays equal (verifies the Python-loop unrolling cost is gone); `update_zupt(..., layout=X)` is called with an explicit layout from every call site; full test suite passes.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Phase 1's "raise instead of silent fallback" may break the bundled Arthur dataset integration test ([tests/integration/test_arthur_config.py](../../../../tests/integration/test_arthur_config.py)) if that data has zero dual-LED frames or relies on missing-`time`-column behavior. | Phase 1 task list includes running the integration suite locally before opening the PR. If Arthur data triggers the new error, fix is to set `filter.led_distance: 0.04` explicitly in the YAML config — the bundled config should already be doing this; verify and patch if not. |
| Phase 3's PRD-constant rename touches ~30 sites across 6 files. Easy to miss user-visible plot legend strings. | Phase 3 includes a `grep -rE '\bPRD[_ ]'` sweep across `src/` AND `tests/` as the final task — anything left after the rename is a missed site. Plot snapshot tests not currently in place; rely on the grep. |
| Phase 4's CLI rename (`online` → `filter`) will break any external scripts or lab-wiki documentation. | Confirmed at planning time: change in place, no alias. CHANGELOG entry in Phase 4 explicitly documents the rename so users see it in `0.3.0` release notes. |
| Phase 5's `FilterCoreConfig.state_mode: Literal[...]` may break programmatic construction in callers that pass `str` from a dynamic source. | Mypy doesn't enforce Literal at runtime — the existing string passes through. The new `__post_init__` validation (via membership check against `LAYOUT_REGISTRY` keys) catches typos at runtime regardless. |
| Phase 6's 3D IEKF `lax.scan` rewrite is the highest-risk numerical change. | Phase 6 captures `EKFResult` outputs from a fixed seed on the existing 3D test fixtures *before* the refactor (`models/ekf.py::extended_kalman_filter_3d`), then asserts bit-exact equality (or `rtol=1e-7`) after. |
| Phase 6's analytic 3D-camera Jacobian may diverge from `jacfwd` for quaternion non-unit-norm states. | Test compares analytic Jacobian against `jacfwd` across a grid of pose+orientation samples with `rtol=1e-6`; ships only if all samples agree. |

## Rollout Strategy

Phases ship as independent PRs in the order listed, each merged to `master` before the next begins. The version sequence is:

- Phase 1 → bump to `0.2.1` (patch — bug fixes, no API rename).
- Phase 2 → no version bump (tests only).
- Phase 3 → bump to `0.2.2` (patch — internal renames; user-visible plot legend strings change `"PRD: ±10 cm/s"` → `"target: ±10 cm/s"`).
- Phase 4 → bump to `0.3.0` (minor — CLI rename is a breaking change).
- Phase 5 → bump to `0.4.0` (minor — schema/dataclass shape changes are breaking).
- Phase 6 → bump to `0.4.1` (patch — internal refactors).

Phases 1, 3, 6 are low-risk and can be parallelized in separate worktrees if the executor prefers. Phases 4 and 5 must land sequentially because Phase 5's `state_mode: Literal` change affects the CLI argparse choices.

## Open Questions

None. The three branching decisions (CLI rename strategy, external-dependent policy, Phase 6 inclusion) were resolved at planning time.

## Estimated Effort

| Phase | LOC delta (rough) | Files touched |
| --- | --- | --- |
| Phase 1 | +120 / -40 | 5 src files, 4 test files |
| Phase 2 | +800 / -50 | 5 new test files, 1 modified |
| Phase 3 | +0 / -150 (net negative — comment removal) | 6 src files |
| Phase 4 | +250 / -180 | 7 src files, README, CHANGELOG |
| Phase 5 | +180 / -90 | 6 src files, 8 test files (call-site updates) |
| Phase 6 | +120 / -180 | 4 src files |

Total: roughly +1500 / -700 LOC, ~25 files touched across `src/`, ~12 new test files.

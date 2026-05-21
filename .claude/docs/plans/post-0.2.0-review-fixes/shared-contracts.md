# Shared Contracts

[← back to PLAN.md](PLAN.md)

Conventions referenced by two or more phases. Each contract lives here exactly once; phases link in by anchor.

## Index

- [`StateMode` Literal alias](#statemode-literal-alias) — Phase 4, Phase 5
- [No backwards-compatibility shims](#no-backwards-compatibility-shims) — Phase 1, Phase 3, Phase 4, Phase 5, Phase 6
- [Test-file layout convention](#test-file-layout-convention) — Phase 2, Phase 5
- [Baseline-then-refactor pattern](#baseline-then-refactor-pattern) — Phase 3, Phase 6

---

## `StateMode` Literal alias

Phase 5 introduces a single `Literal` alias used both by `FilterCoreConfig.state_mode` and by the CLI's `--state-mode` argparse `choices`. Defined once in `src/trodestrack/models/state_layout.py`:

```python
from typing import Literal

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
```

The runtime tuple `STATE_MODES: tuple[str, ...] = get_args(StateMode)` is derived via `typing.get_args(StateMode)` and exported alongside the alias. The CLI passes `STATE_MODES` to argparse's `choices=`. Phase 4 uses `STATE_MODES`; Phase 5 owns the definition.

**Invariant: do not weaken.** Any new layout added to `LAYOUT_REGISTRY` must also be added to `StateMode`. A test in `tests/models/test_state_layout.py` asserts `set(STATE_MODES) == set(LAYOUT_REGISTRY.keys())`.

---

## No backwards-compatibility shims

Confirmed at planning time: no external dependents on the changed APIs. The repo follows the global `CLAUDE.md` guidance "When introducing a replacement, remove the old path in the same change."

This means:
- Phase 1: replace silent fallbacks with `raise` — no flag to opt back into the old behavior.
- Phase 3: rename `PRD_POSITION_RMSE_M` → `TARGET_POSITION_RMSE_M`; no `PRD_POSITION_RMSE_M = TARGET_POSITION_RMSE_M` re-export.
- Phase 4: rename `online` subcommand → `filter`; no alias, no deprecation warning. Old `cli/online.py` is deleted, not kept as a thin wrapper.
- Phase 5: regroup `PreparedSession` fields into `EventChannel`; no `@property` shims on the old field names.
- Phase 6: delete dead jitter fallback; no `# noqa` suppression for the unused branch.

CHANGELOG entries in each phase document the breaking change explicitly so users see it in release notes.

---

## Test-file layout convention

Tests live under `tests/<module>/test_<thing>.py`, mirroring `src/trodestrack/<module>/<thing>.py`. New tests follow this pattern:

| New test file | Mirrors |
| --- | --- |
| `tests/cli/test_filter_command.py` (Phase 4 rename) | `src/trodestrack/cli/filter.py` |
| `tests/cli/test_report_command.py` (Phase 2 fills the empty shell) | `src/trodestrack/cli/report.py` |
| `tests/filters/test_ekf_3d_analytic.py` (Phase 2) | `src/trodestrack/models/ekf.py::extended_kalman_filter_3d` |
| `tests/models/sensors/test_heading_pseudo.py` (Phase 2) | `src/trodestrack/models/sensors/heading_pseudo.py` |
| `tests/models/sensors/test_camera_position.py` (Phase 2) | `src/trodestrack/models/sensors/camera_position.py` |
| `tests/io/test_session_loading.py` (Phase 1 adds tests; Phase 2 modifies for real-vision safety check) | `src/trodestrack/io/session.py` |

Each test file imports the production code under test directly — no helper modules that re-export it. Slow tests (>2s) get `@pytest.mark.slow` per `pyproject.toml:114`. Use existing fixtures in `tests/filters/conftest.py` where possible rather than building new ones.

---

## Baseline-then-refactor pattern

Phases 3 (comment cleanup) and 6 (filter polish) are refactors that must not change numerical output. Both phases capture a baseline before any code change:

1. **Capture baseline** (first task of the phase): run the relevant test or example on `master` HEAD and pickle the output arrays to a tmp file. Record `git rev-parse HEAD` in a comment.
2. **Refactor** (subsequent tasks).
3. **Compare** (last task before opening the PR): re-run the same test, unpickle the baseline, assert array equality.

Phase 6 uses `rtol=1e-7` (numerical refactors can introduce floating-point reorderings). Phase 3 uses bit-exact equality (`np.array_equal`) since comment changes don't touch executable code.

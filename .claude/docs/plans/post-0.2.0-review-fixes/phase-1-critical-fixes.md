# Phase 1 — Critical silent-failure & version fixes

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/trodestrack/io/session.py](../../../../src/trodestrack/io/session.py) lines 1-90 (PreparedSession + load_session), 470-535 (calibration diagnostics + time validation), 600-700 (LED loading, median LED distance, fusion calibration validator). The silent fallbacks live in this file.
- [src/trodestrack/io/ttl_events.py](../../../../src/trodestrack/io/ttl_events.py) lines 180-244 — `per_frame_event_indices`. Lines 209 and 218 are the two silent-drop points.
- [src/trodestrack/cli/utils.py](../../../../src/trodestrack/cli/utils.py) lines 16-54 — `friendly_cli_errors`. The docstring at line 33-34 already promises a `--debug` escape hatch that doesn't exist.
- [src/trodestrack/__init__.py](../../../../src/trodestrack/__init__.py) line 8 — `__version__ = "0.1.0"` (wrong; pyproject.toml ships `0.2.0`).
- [pyproject.toml](../../../../pyproject.toml) line 3 — source of truth for the version.
- [tests/io/test_session_loading.py](../../../../tests/io/test_session_loading.py) — existing patterns for fixtures and mocked-EKF testing of safety check.

**Contracts referenced:**

- [No backwards-compatibility shims](shared-contracts.md#no-backwards-compatibility-shims) — replace silent paths with raises; do not add an opt-back-in flag.

## Tasks

### Task 1 — Bump `__version__` to match pyproject

Edit [src/trodestrack/__init__.py:8](../../../../src/trodestrack/__init__.py#L8) from `__version__ = "0.1.0"` to `__version__ = "0.2.0"`. The package version on PyPI / GitHub release is `0.2.0` (per `pyproject.toml:3` and the `d853bbc Release 0.2.0` commit); this string is what `trodestrack --version` prints and what bug-report templates capture.

Add a CHANGELOG entry under a new `## [0.2.1] — unreleased` section: `"Fixed: trodestrack --version now reports the correct package version."`

### Task 2 — Raise instead of silent `0.04 m` fallback in `_median_led_distance`

Replace the body at [src/trodestrack/io/session.py:635-639](../../../../src/trodestrack/io/session.py#L635-L639) with:

```python
def _median_led_distance(led1: np.ndarray, led2: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(led1).all(axis=1) & np.isfinite(led2).all(axis=1)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        raise ValueError(
            "Cannot auto-detect LED spacing: zero dual-LED frames are valid "
            "(both LEDs present and finite). Set camera.led_distance "
            "explicitly in the SessionConfig YAML, or pass --led-distance on "
            "the CLI. The previous 0.04 m fallback was removed because it "
            "silently produced wrong-sized heading pseudo-measurements when "
            "no dual-LED frames were available."
        )
    return float(np.nanmedian(np.linalg.norm(led2[valid] - led1[valid], axis=1)))
```

Verify that all callers handle the new `ValueError` correctly. Grep: `grep -nE '_median_led_distance' src/`. The single caller is in `_load_spikegadgets_trodes` (around `src/trodestrack/io/session.py:312-320` — re-grep to confirm exact line at edit time). That caller is reached from `load_session()` which is already wrapped by `friendly_cli_errors` at the CLI boundary.

If `camera.led_distance` is set in the YAML config (`src/trodestrack/config/schemas.py:166-180`), `_median_led_distance` should not be called — verify the existing branch logic in `_load_spikegadgets_trodes` respects an explicit `led_distance` before falling through to auto-detection. If it doesn't, add the precedence check.

The `cli/online.py` `--help` text for `--led-distance` currently advertises a "0.04 m fallback" — Phase 4 rewrites the CLI help, but for this phase update the help string to read "auto-detected from dual-LED frames; must be set explicitly via --led-distance or YAML config.camera.led_distance if no dual-LED frames exist."

### Task 3 — Reject missing-`time`-column in `_index_or_time_column`

Replace [src/trodestrack/io/session.py:629-632](../../../../src/trodestrack/io/session.py#L629-L632) with:

```python
def _index_or_time_column(df: pd.DataFrame, *, source: str) -> np.ndarray:
    if "time" in df.columns:
        return df["time"].to_numpy(dtype=float)
    raise ValueError(
        f"{source} is missing required 'time' column. The previous fallback "
        "of using df.index silently substituted sample numbers (0, 1, 2, ...) "
        "for seconds, producing dt values off by the sampling rate and "
        "wildly miscalibrated filter outputs."
    )
```

Update the two call sites in `_load_spikegadgets_trodes` (around `src/trodestrack/io/session.py:312-313` — re-grep to confirm) to pass `source="IMU dataframe"` and `source="camera dataframe"` respectively.

### Task 4 — Narrow IMU-calibration try block

At [src/trodestrack/io/session.py:483-499](../../../../src/trodestrack/io/session.py#L483-L499), separate the two operations so only the *verdict* function is guarded:

```python
def _add_imu_calibration_diagnostics(session: PreparedSession) -> PreparedSession:
    config = session.config
    U_full = session.U_imu_for_calibration
    if (
        session.source_format != "spikegadgets_trodes"
        or not config.imu.run_calibration
        or U_full is None
        or U_full.shape[1] != 6
    ):
        return session

    diagnostics = dict(session.diagnostics)
    # run_imu_calibration_diagnostics computes metrics — bugs here are real
    # bugs and should propagate, not be swallowed into a diagnostics string.
    report = run_imu_calibration_diagnostics(
        t_imu=session.t_imu,
        gyro_z=U_full[:, 2],
        accel_xyz=U_full[:, 3:6],
        t_cam=session.t_cam,
        led1=session.Z_cam_led1,
        led2=session.Z_cam_led2,
    )
    diagnostics["imu_calibration"] = report
    diagnostics["imu_calibration_led_identity_applied"] = (
        config.led_identity.mode == "auto"
    )
    # Only the verdict function (_validate_calibration_for_fusion) raises
    # ValueError as a deliberate signal — capture that into diagnostics so
    # callers can decide whether to gate on it.
    try:
        _validate_calibration_for_fusion(report, config)
    except ValueError as e:
        diagnostics["imu_calibration_error"] = str(e)
    return replace(session, diagnostics=diagnostics)
```

### Task 5 — Tighten confidence column truthy check in `_load_leds`

At [src/trodestrack/io/session.py:621-625](../../../../src/trodestrack/io/session.py#L621-L625), replace the truthy check with explicit `None` discrimination. Pair with a Pydantic-side fix: add `min_length=1` to the confidence column fields in `src/trodestrack/config/schemas.py:166-205` (re-grep to find `confidence_led1_column`).

Loader change:

```python
conf_cam = None
if cam.confidence_led1_column is not None and cam.confidence_led2_column is not None:
    c1 = pos_df[cam.confidence_led1_column].to_numpy(dtype=float)
    c2 = pos_df[cam.confidence_led2_column].to_numpy(dtype=float)
    conf_cam = np.column_stack([c1, c1, c2, c2])
```

Schema change: locate the two `confidence_led*_column: str | None = None` fields and change to `confidence_led*_column: str | None = Field(default=None, min_length=1)`. Empty strings now fail Pydantic validation at config-load time with an informative error.

### Task 6 — Surface TTL-event drop counts

At [src/trodestrack/io/ttl_events.py:206-244](../../../../src/trodestrack/io/ttl_events.py#L206-L244), refactor `per_frame_event_indices` to also return a diagnostics dict, and update the caller in [src/trodestrack/io/session.py](../../../../src/trodestrack/io/session.py) (grep for `per_frame_event_indices`) to plumb the diagnostics into `session.diagnostics["ttl_events"]`.

Change the return signature:

```python
def per_frame_event_indices(
    t_evt: np.ndarray,
    edge: np.ndarray,
    source_id: np.ndarray,
    t_cam: np.ndarray,
    source_active_edges: Mapping[int, int],
    source_id_to_index: Mapping[int, int],
    max_events_per_frame: int,
) -> tuple[np.ndarray, dict[str, object]]:
    ...
    diagnostics = {
        "n_events_total": int(t_evt_orig.size),
        "n_dropped_edge_mismatch": int(n_orig - keep.sum()),
        "n_dropped_before_t_cam": int((frame_idx == 0).sum()),
        "n_dropped_after_t_cam": int((frame_idx == n_cam).sum()),
        "n_events_kept": int(out_kept_count),
        "n_events_kept_per_source": {
            int(sid): int(kept) for sid, kept in per_source_kept.items()
        },
    }
    return out, diagnostics
```

(Implementation detail: capture `t_evt_orig`, `n_orig` and `per_source_kept` while filtering.) The exact bookkeeping is a straightforward extension of the existing logic at lines 206-222.

If `n_events_kept == 0 and n_events_total > 0`, raise `ValueError` from `per_frame_event_indices` with a message naming the most-likely cause (clock-offset misconfig, wrong active-edge config). The user clearly intended events to apply.

If any *configured* source contributed zero kept events but the file did have events from other sources, emit a warning via `warnings.warn` (don't raise) — the user may intentionally configure inactive sources.

### Task 7 — Wire `TRODESTRACK_DEBUG=1` escape hatch in `friendly_cli_errors`

At [src/trodestrack/cli/utils.py:16-54](../../../../src/trodestrack/cli/utils.py#L16-L54), implement the `--debug` / env-var escape hatch the docstring at line 33-34 promises. Modify the wrapper:

```python
import os

def friendly_cli_errors(func: F) -> F:
    """..."""  # (existing docstring, with the speculative `--debug` paragraph rewritten)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        debug = os.environ.get("TRODESTRACK_DEBUG", "").lower() in ("1", "true", "yes")
        try:
            return func(*args, **kwargs)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            if debug:
                raise
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            if debug:
                raise
            print(
                f"Unexpected error ({type(e).__name__}): {e}\n"
                "Re-run with TRODESTRACK_DEBUG=1 for a full traceback.",
                file=sys.stderr,
            )
            sys.exit(1)

    return wrapper  # type: ignore[return-value]
```

Update the docstring to document the env var as a real feature, and drop the "if the CLI grows one" hedge.

### Task 8 — Tests

Add the following tests. See [Validation slice](#validation-slice) for the assertion table.

- `tests/io/test_session_loading.py` (extend the existing file):
  - `test_median_led_distance_raises_when_no_dual_led_frames`
  - `test_index_or_time_column_raises_when_time_missing`
  - `test_add_imu_calibration_diagnostics_propagates_unexpected_errors` — patch `run_imu_calibration_diagnostics` to raise a `ValueError` that is NOT a calibration verdict (e.g. a shape mismatch from inside the calibration helper), assert it bubbles out (currently it's swallowed into the diagnostics dict).
- `tests/io/test_session_loading.py`:
  - `test_load_leds_treats_empty_confidence_column_as_missing` — confirm Pydantic rejects empty string at config-load time (this tests the schema, not the loader; loader-side test isn't needed because the schema gate is sufficient).
- `tests/io/test_ttl_events.py` (extend the existing file):
  - `test_per_frame_event_indices_returns_drop_counts`
  - `test_per_frame_event_indices_raises_when_all_events_dropped`
  - `test_per_frame_event_indices_warns_when_configured_source_has_no_events`
- `tests/cli/test_cli_value_error_friendliness.py` (extend):
  - `test_friendly_cli_errors_reraises_when_debug_env_set` — set `TRODESTRACK_DEBUG=1`, invoke a wrapped function that raises `ValueError`, assert the exception propagates rather than calling `sys.exit`.
  - `test_friendly_cli_errors_includes_exception_type_in_unexpected` — invoke wrapped function raising `KeyError("foo")`, capture stderr, assert it includes `"KeyError"`.

### Task 9 — CHANGELOG and docstrings

Add to CHANGELOG under `## [0.2.1] — unreleased`:

```
### Fixed
- `trodestrack --version` now reports the correct package version (was 0.1.0; pyproject.toml ships 0.2.0).
- `io.session._median_led_distance` now raises `ValueError` instead of silently returning a hardcoded 0.04 m fallback when no dual-LED frames are available. Set `camera.led_distance` explicitly in the YAML config or pass `--led-distance` on the CLI.
- `io.session._index_or_time_column` now raises `ValueError` when the input parquet is missing the `time` column. The previous fallback silently substituted sample numbers for seconds.
- `io.session._add_imu_calibration_diagnostics` no longer swallows non-verdict `ValueError`s into a diagnostics string; only `_validate_calibration_for_fusion` exceptions are captured.
- `io.ttl_events.per_frame_event_indices` now surfaces per-source kept/dropped counts via a new diagnostics return value, raises when all events are dropped, and warns when a configured source contributes no events.
- `config.schemas.CameraConfig.confidence_led{1,2}_column` now rejects empty strings at validation time.
- `cli.utils.friendly_cli_errors` now re-raises with a full traceback when `TRODESTRACK_DEBUG=1` is set in the environment, and includes the exception class name in the "Unexpected error" message.
```

## Deliberately not in this phase

- **CLI rename `online` → `filter`** — Phase 4.
- **Progress reporting (tqdm)** — Phase 4.
- **PRD constant renames** — Phase 3.
- **State-mode Literal types** — Phase 5.
- **`PreparedSession` field regrouping** — Phase 5.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_median_led_distance_raises_when_no_dual_led_frames` | Calling `_median_led_distance` with `mask=np.zeros(N, bool)` raises `ValueError` matching "auto-detect LED spacing". |
| `test_index_or_time_column_raises_when_time_missing` | Passing a `pd.DataFrame(columns=['x','y'])` raises `ValueError` matching "missing required 'time' column"; passing a frame with `time` returns the column unchanged. |
| `test_add_imu_calibration_diagnostics_propagates_unexpected_errors` | Patch `run_imu_calibration_diagnostics` to raise `ValueError("shape mismatch")`; the call propagates rather than appearing in `session.diagnostics["imu_calibration_error"]`. |
| `test_add_imu_calibration_diagnostics_still_captures_verdict_errors` | When `_validate_calibration_for_fusion` raises, the message ends up in `session.diagnostics["imu_calibration_error"]` and no exception propagates. |
| `test_load_leds_treats_empty_confidence_column_as_missing` | `CameraConfig(..., confidence_led1_column="")` raises Pydantic `ValidationError`. |
| `test_per_frame_event_indices_returns_drop_counts` | Feed 5 events with 2 edge-mismatched and 1 before `t_cam[0]`; diagnostics dict shows `n_dropped_edge_mismatch=2, n_dropped_before_t_cam=1, n_events_kept=2`. |
| `test_per_frame_event_indices_raises_when_all_events_dropped` | All 5 events have wrong edge; `per_frame_event_indices` raises `ValueError`. |
| `test_per_frame_event_indices_warns_when_configured_source_has_no_events` | Configure 2 sources but the input file only contains events for source 1; `warnings.warn` fires with a message naming source 2. |
| `test_friendly_cli_errors_reraises_when_debug_env_set` | `monkeypatch.setenv("TRODESTRACK_DEBUG", "1")`; wrapped function raising `ValueError("x")` propagates instead of exiting. |
| `test_friendly_cli_errors_includes_exception_type_in_unexpected` | Wrapped function raising `KeyError("missing")` — captured stderr contains `"KeyError"` and the `TRODESTRACK_DEBUG` hint. |
| `test_version_string_matches_pyproject` | `trodestrack.__version__` equals the `version` field parsed from `pyproject.toml`. |

Mark `test_per_frame_event_indices_*` as standard unit tests (fast). No `@pytest.mark.slow` needed in this phase.

## Fixtures

All fixtures synthesized inline or extend existing patterns in `tests/io/test_session_loading.py` and `tests/io/test_ttl_events.py`. No new file fixtures.

The integration check (Arthur dataset, [tests/integration/test_arthur_config.py](../../../../tests/integration/test_arthur_config.py)) must still pass. If it fails after Task 2, the bundled YAML config under `data/arthur_benchmark_results/` likely doesn't set `camera.led_distance` explicitly. Fix is to update the YAML to set it; document in the PR description.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies. Shared setup is in fixtures, not copy-pasted across tests.
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the `return 0.04` fallback line and the `return df.index.to_numpy` fallback are gone — verify by grep).
- CHANGELOG updated.
- The bundled Arthur integration test still passes (or the bundled YAML config is patched alongside).

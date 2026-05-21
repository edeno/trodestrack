# Phase 4 — UX improvements (CLI rename, progress, report bridge, QA polish)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase ships as a `0.3.0` minor release because of the breaking CLI rename `online` → `filter`.

**Inputs to read first:**

- [src/trodestrack/__init__.py](../../../../src/trodestrack/__init__.py) — top-level CLI dispatcher; epilog at lines 28-57 references `online`.
- [src/trodestrack/cli/online.py](../../../../src/trodestrack/cli/online.py) — full file; will be renamed to `cli/filter.py`. Module docstring at lines 1-31 also needs rewriting (the "online" naming permeates).
- [src/trodestrack/cli/smooth.py](../../../../src/trodestrack/cli/smooth.py) — for progress reporting, argparse group reorganization.
- [src/trodestrack/cli/report.py](../../../../src/trodestrack/cli/report.py) — for the bridge from `filter`/`smooth` outputs.
- [src/trodestrack/cli/utils.py](../../../../src/trodestrack/cli/utils.py) lines 96-100 — `load_data_file` exception catch swallows type info.
- [src/trodestrack/cli/utils.py](../../../../src/trodestrack/cli/utils.py) lines 159-166 — gold-standard error message pattern (referenced for Task 6).
- [src/trodestrack/io/session.py](../../../../src/trodestrack/io/session.py) lines 526-535 — `_validate_time_vector` error messages; rewrite to match the gold standard.
- [src/trodestrack/qa/report.py](../../../../src/trodestrack/qa/report.py) lines 339-357 — summary page; add PASS/FAIL banner.
- [src/trodestrack/qa/report.py](../../../../src/trodestrack/qa/report.py) lines 466-483 — trajectory plot start/end markers (currently green/red — not colorblind-safe).
- [src/trodestrack/qa/plots.py](../../../../src/trodestrack/qa/plots.py) lines 518-611 — NEES/NIS histogram functions; add titles.
- [src/trodestrack/viz/styles.py](../../../../src/trodestrack/viz/styles.py) — palette definition.
- [README.md](../../../../README.md) lines 318-370 — "Project Status" section; bloated and benchmark numbers repeat.

**Contracts referenced:**

- [`StateMode` Literal alias](shared-contracts.md#statemode-literal-alias) — defined in Phase 5 but consumed here for the CLI `--state-mode` `choices`. Phase 4 MUST land after Phase 5 if Phase 5 ships first; if Phase 4 ships first, hardcode the choices list and have Phase 5 swap in `STATE_MODES`.
- [No backwards-compatibility shims](shared-contracts.md#no-backwards-compatibility-shims) — `online` is deleted, not aliased.

## Tasks

### Task 1 — Rename `cli/online.py` → `cli/filter.py`

Use `git mv` to preserve history:

```bash
git mv src/trodestrack/cli/online.py src/trodestrack/cli/filter.py
git mv tests/cli/test_online_command.py tests/cli/test_filter_command.py
```

Inside the renamed file:
- Rewrite the module docstring (currently `cli/online.py:1-31`). New text: `"""CLI command for forward-only EKF filtering (no smoothing) on sensor-fused tracking data. Loads complete IMU/camera/LED arrays from disk and runs extended_kalman_filter once. Use trodestrack smooth for best accuracy when all data is available."""`. Remove every disclaimer about "despite the name, this is a BATCH command" — the new name doesn't need defending.
- Rename `add_online_parser` → `add_filter_parser`. Update the `subparsers.add_parser("online", ...)` call: change `"online"` → `"filter"`. Rewrite the `help=` text: `"Run forward-only EKF filtering (no smoothing) on input files"`. Rewrite the `description=`: remove the entire "Despite the name" paragraph (it no longer applies). Add a new closing paragraph: `"For best accuracy when all data is available, use trodestrack smooth."`
- Inside the function body, every `"online"` string literal that names the command (e.g. in `require_cli_inputs(args, ..., command="online")`) becomes `"filter"`.

### Task 2 — Update top-level dispatcher in `__init__.py`

In [src/trodestrack/__init__.py](../../../../src/trodestrack/__init__.py):
- Line 75: change `from trodestrack.cli.online import add_online_parser` to `from trodestrack.cli.filter import add_filter_parser`.
- Line 81: change `add_online_parser(subparsers)` to `add_filter_parser(subparsers)`.
- Lines 28-57 (epilog): rewrite all examples to use `trodestrack filter` instead of `trodestrack online`. Drop the "Run online filtering only (faster, lower latency)" framing — change to "Run forward-only filtering (no smoother lookahead)".

### Task 3 — Update tests for the rename

In `tests/cli/test_filter_command.py` (renamed from `test_online_command.py`):
- Class name `TestOnlineCommand` → `TestFilterCommand`.
- Every test name `test_online_*` → `test_filter_*`.
- Every `sys.argv = ["trodestrack", "online", ...]` → `["trodestrack", "filter", ...]`.

Grep `grep -rn 'online' tests/` for any remaining test-file references. The integration test at [tests/integration/test_prd_session.py](../../../../tests/integration/test_prd_session.py) likely also calls it; update.

### Task 4 — Add tqdm progress reporting

Both `cli/filter.py` and `cli/smooth.py` print `"Running Extended Kalman Filter..."` then go silent during the JAX run. Wrap the filter calls with tqdm.

The filter / smoother itself is a `jax.lax.scan` and doesn't natively support per-step callbacks. The simplest user-visible improvement is to show progress at the *scan boundaries* the host code already knows about:

- During JIT warmup: print `"Compiling JAX kernels..."` before the first call, `"Compilation done in {dt:.1f}s"` after.
- During the actual run: report total wall-clock time after completion.

Implementation: wrap the filter call (around `cli/filter.py:397-408` and `cli/smooth.py:382-404`) with `time.perf_counter()` deltas; on JIT first-call, print a one-line progress message; after completion, print `f"Filter completed in {dt:.1f}s ({n_cam} camera frames @ {n_cam/dt:.0f} fps)"`.

Per-frame progress requires either (a) chunking the scan and calling it in batches with a tqdm-wrapped loop, or (b) using `jax.debug.callback` with a counter. Option (a) is simpler and doesn't pessimize JAX performance — wrap the camera frames in chunks of 1000 and report tqdm progress between chunks. This is a meaningful change to the filter call shape; if the chunking adds complexity, prefer (a) only when the session is longer than 10000 frames:

```python
from tqdm import tqdm

if n_cam > 10_000:
    chunk_size = 1000
    chunks = [(i, min(i + chunk_size, n_cam)) for i in range(0, n_cam, chunk_size)]
    result_chunks = []
    state = initial_state
    for start, end in tqdm(chunks, desc="Filtering", unit="chunk"):
        chunk_result = extended_kalman_filter(..., t_cam[start:end], ..., initial_state=state)
        result_chunks.append(chunk_result)
        state = chunk_result.final_state
    result = _concatenate_results(result_chunks)
else:
    result = extended_kalman_filter(...)
```

If chunked execution would require non-trivial refactoring of `extended_kalman_filter` to accept `initial_state` and return `final_state`, **defer the per-frame progress to a follow-up** and ship only the JIT-warmup + total-runtime messages this phase. Document the deferral.

### Task 5 — Bridge `filter`/`smooth` output to `report` input

The `report` command currently consumes a `qa_inputs/` directory that has ground-truth-aligned arrays (`positions_true.npy`, `nees.npy`, etc.), but `filter`/`smooth` write a different layout (`filtered_means.txt`, `marginal_loglik.txt`). The CLI epilog at [src/trodestrack/__init__.py:49-54](../../../../src/trodestrack/__init__.py#L49-L54) documents this dead-end but offers no recovery path.

Implement a `--from-run` mode on `report` that accepts the `filter`/`smooth` output directory and synthesizes `qa_inputs/` on the fly when ground truth is also provided:

```python
# in cli/report.py
parser.add_argument(
    "--from-run",
    type=Path,
    help="Path to a trodestrack filter/smooth output directory. "
         "Requires --ground-truth-positions and --ground-truth-headings.",
)
parser.add_argument(
    "--ground-truth-positions",
    type=Path,
    help="Path to ground-truth positions .npy or .txt file (N, 2). "
         "Required when --from-run is used.",
)
parser.add_argument(
    "--ground-truth-headings",
    type=Path,
    help="Path to ground-truth headings .npy or .txt file (N,). "
         "Required when --from-run is used.",
)
```

The new `_build_qa_inputs_from_run(run_dir, gt_pos, gt_head, state_layout)` helper:
1. Loads `filtered_means.txt` / `smoothed_means.txt` (whichever exists).
2. Uses `state_layout` to extract position/velocity/heading from the state vector.
3. Computes NEES against ground truth.
4. Returns the dict that `generate_qa_report` consumes — no temp directory needed; `load_run_data` and the helper produce the same shape.

Update the `__init__.py` epilog to show the new workflow:

```
# Full pipeline: smooth → report
trodestrack smooth --config session.yaml --output-dir run1/
trodestrack report --from-run run1/ \
                   --ground-truth-positions truth_pos.txt \
                   --ground-truth-headings truth_head.txt \
                   --pdf run1/report.pdf
```

If ground truth is not available (real sessions), the user runs `report --run qa_inputs/` against an externally-prepared QA layout — that path still works.

### Task 6 — Enrich `_validate_time_vector` error messages

In [src/trodestrack/io/session.py:526-535](../../../../src/trodestrack/io/session.py#L526-L535), rewrite each `raise ValueError(...)` to match the actionable pattern at [src/trodestrack/cli/utils.py:159-166](../../../../src/trodestrack/cli/utils.py#L159-L166). For "must be strictly increasing", identify the first non-increasing index and report both values + dt:

```python
def _validate_time_vector(t: np.ndarray, name: str) -> None:
    arr = np.asarray(t, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D array; got shape {arr.shape}. "
            "Check that the file was read as a single column or that the "
            "DataFrame column is not nested."
        )
    if arr.size < 2:
        raise ValueError(
            f"{name} must contain at least two samples (got {arr.size}). "
            "The filter derives the sample period from np.diff(t)."
        )
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        first_bad = int(np.argmax(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s); first at index "
            f"{first_bad}. Drop or interpolate dropped frames before passing "
            "to the loader."
        )
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        first_bad = int(np.argmax(diffs <= 0))
        raise ValueError(
            f"{name} must be strictly increasing; first non-increasing step "
            f"at index {first_bad + 1} (t[{first_bad}]={arr[first_bad]!r}, "
            f"t[{first_bad + 1}]={arr[first_bad + 1]!r}, "
            f"dt={diffs[first_bad]!r}). Common causes: duplicate timestamps "
            "from sample-and-hold expansion, or out-of-order rows from a "
            "concatenation."
        )
```

### Task 7 — Improve `load_data_file` error message

In [src/trodestrack/cli/utils.py:96-100](../../../../src/trodestrack/cli/utils.py#L96-L100), include the exception class name so numpy parse errors are recognizable:

```python
try:
    data = np.loadtxt(path)
except Exception as e:
    print(
        f"Error loading {name} from {path}: {type(e).__name__}: {e}",
        file=sys.stderr,
    )
    sys.exit(1)
```

### Task 8 — Add PASS/FAIL banner to QA report summary

In [src/trodestrack/qa/report.py](../../../../src/trodestrack/qa/report.py) (around lines 339-357 — re-grep `_create_summary_page` at edit time), prepend a verdict line above the metrics table:

```python
position_pass = position_rmse <= TARGET_POSITION_RMSE_M
velocity_pass = velocity_rmse <= TARGET_VELOCITY_RMSE_MS
heading_pass = heading_mae_deg <= TARGET_HEADING_MAE_DEG
all_pass = position_pass and velocity_pass and heading_pass

if all_pass:
    verdict = "RESULT: PASS"
else:
    failed = [
        name for name, ok in [
            ("position", position_pass),
            ("velocity", velocity_pass),
            ("heading", heading_pass),
        ] if not ok
    ]
    verdict = f"RESULT: FAIL ({', '.join(failed)})"
```

Render `verdict` in a colored title block (green for pass, red for fail) above the metrics table. Then keep the existing metric rows.

### Task 9 — Fix colorblind-unsafe start/end markers

In [src/trodestrack/qa/report.py:466-483](../../../../src/trodestrack/qa/report.py#L466-L483) trajectory plot, the start (green circle) and end (red square) markers are unreadable for deuteranopia. Switch to a colorblind-safe pair: keep the *shape difference* (circle vs square) but use orange/blue or both-black-with-different-fills:

```python
ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=80, c="#0072B2",
           edgecolor="black", linewidth=1.5, label="start", zorder=3)
ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", s=80, c="#E69F00",
           edgecolor="black", linewidth=1.5, label="end", zorder=3)
```

(`#0072B2` and `#E69F00` are from the Okabe-Ito colorblind-safe palette.) Update the [src/trodestrack/viz/styles.py](../../../../src/trodestrack/viz/styles.py) `COLORS` dict if these aren't already present.

### Task 10 — Add titles to NEES/NIS histograms

In [src/trodestrack/qa/plots.py](../../../../src/trodestrack/qa/plots.py) `plot_nees_histogram` (around line 518) and `plot_nis_histogram` (around line 608):

```python
ax.set_title("NEES Histogram — Filter Consistency Check")
# ... and ...
ax.set_title("NIS Histogram — Innovation Consistency Check")
```

### Task 11 — Move `--use-heading-measurement` to filter argparse group in `smooth`

In [src/trodestrack/cli/smooth.py](../../../../src/trodestrack/cli/smooth.py) (find `--use-heading-measurement` by grep), the flag is currently in the smoother argparse group. The heading measurement is applied during the forward EKF pass, not the RTS backward pass. Move the `add_argument("--use-heading-measurement", ...)` call from the smoother group into the filter group. Pure argparse-group reshuffling; no behavior change.

### Task 12 — README cleanup

In [README.md](../../../../README.md):
- Cut the "Project Status" section (around lines 318-370). Move the milestone/status content to a new `STATUS.md` at the repo root, or to the `CHANGELOG.md`.
- The benchmark numbers ("~38× realtime / ~0.41 ms per frame") appear three times in the README (lines 13, 349, 358 per the UX review). Keep one instance — the most relevant one is the headline at line 13. Delete the other two.
- The README on-ramp should now be: install → quickstart → one CLI example → link to docs site for the rest.

### Task 13 — Update `--help` for state-mode interactions

In `cli/filter.py` and `cli/smooth.py`, the `--state-mode` and `--imu-measurements` help strings cram multi-mode shape rules into single sentences. Rewrite to lead with the default case and append the alternatives:

```python
parser.add_argument(
    "--imu-measurements",
    type=Path,
    required=False,
    help=(
        "Path to IMU measurements file. Default state mode (2d_cam_3d_imu) "
        "expects shape (N_imu, 3) for [gyro_z, accel_x, accel_y]. "
        "Other state modes: vision_only=no IMU; imu_only/2d_full=(N_imu, 3); "
        "2d_cam_6dof_imu_orientation=(N_imu, 6) for full 6-DOF."
    ),
    metavar="FILE",
)
```

Also: drop the "3d_cam_6dof_imu (experimental)" mention from `--state-mode` if Phase 5 ships first and the choices list is generated from `STATE_MODES` (the Literal alias). Otherwise leave the parenthetical.

### Task 14 — CHANGELOG entry

Add under `## [0.3.0] — unreleased`:

```
### Changed (breaking)
- The `online` CLI subcommand has been renamed to `filter`. Users with scripts calling `trodestrack online ...` must update to `trodestrack filter ...`. The semantics are unchanged — the new name accurately reflects that the command runs a forward-only batch filter (not a streaming/real-time ingest loop).

### Added
- `trodestrack report --from-run <dir>` now accepts a `filter`/`smooth` output directory directly when paired with `--ground-truth-positions` and `--ground-truth-headings`. Previously, users had to manually construct a `qa_inputs/` directory before running `report`.
- Progress reporting during long filter runs: JIT compilation status, total wall-clock time, and (for sessions > 10000 camera frames) per-chunk tqdm progress.
- QA report summary page now leads with a PASS/FAIL verdict line based on project acceptance targets.
- NEES/NIS histograms now have descriptive titles.

### Fixed
- QA report trajectory plot start/end markers are now colorblind-safe (Okabe-Ito orange/blue instead of green/red).
- `cli/utils.load_data_file` error messages now include the exception class name so numpy parse errors are recognizable.
- `io.session._validate_time_vector` error messages now include the offending index/value and suggest common root causes.
- `cli.smooth`'s `--use-heading-measurement` flag is now in the filter argparse group where it belongs (the flag applies to the forward EKF pass, not the smoother).

### Removed
- README "Project Status" section moved to CHANGELOG / STATUS.md to keep the on-ramp focused.
```

## Deliberately not in this phase

- **`__version__` bump** — Phase 1 already handled the 0.1.0 → 0.2.0 typo fix. This phase bumps to `0.3.0` as part of the CHANGELOG release entry.
- **State-mode `Literal` alias** — Phase 5. If Phase 5 has not shipped at the time this PR opens, hardcode the argparse `choices` list and have Phase 5 sweep it to `STATE_MODES` later.
- **`PreparedSession` regroup / `EventChannel`** — Phase 5.
- **Filter-core 3D IEKF parity** — Phase 6.
- **Per-frame JAX-progress callbacks via `jax.debug.callback`** — out of scope. The chunked-tqdm approach in Task 4 is sufficient.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_filter_command_runs` (renamed) | `trodestrack filter --config ...` succeeds; output files match the legacy `online` shape. |
| `test_online_command_no_longer_exists` | `trodestrack online ...` exits non-zero with argparse's "unknown subcommand" message. |
| `test_report_from_run_synthesizes_qa_inputs_from_smooth_output` | Run `smooth` to `tmp_path/run/`, then `report --from-run tmp_path/run --ground-truth-positions ... --ground-truth-headings ... --pdf tmp_path/r.pdf`; PDF exists and contains expected metric values. |
| `test_report_from_run_requires_ground_truth_args` | `report --from-run X` without ground-truth args exits 1 with a friendly error. |
| `test_filter_emits_jit_compile_message_then_total_runtime` | Capture stdout from a `filter` run; assert it contains "Compiling JAX kernels" and "Filter completed in" lines. |
| `test_filter_chunked_progress_visible_for_long_session` | Mock a session with 20000 camera frames; assert tqdm progress bar text appears in captured stderr (tqdm writes to stderr by default). |
| `test_qa_report_summary_page_shows_pass_for_passing_metrics` | Build a fixture where all metrics meet targets; render report; assert PDF bytes contain "RESULT: PASS". |
| `test_qa_report_summary_page_shows_fail_with_failing_metric_names` | Build fixture with `position_rmse=0.05` (fails 0.02 target); assert PDF contains "RESULT: FAIL" and "position". |
| `test_validate_time_vector_error_messages_include_offending_index` | Pass a `t` with a backwards step at index 5; assert error message contains `"index 6"` (1-indexed step) and `dt`. |
| `test_validate_time_vector_error_includes_recovery_hint` | Each error message contains at least one recovery suggestion (regex `"check|drop|interpolate|configure"`). |
| `test_load_data_file_error_includes_exception_class_name` | Pass a file with mixed string/numeric content; assert error contains `"ValueError"` (or whichever numpy raises). |
| `test_smooth_help_groups_use_heading_measurement_with_filter_args` | Capture `trodestrack smooth --help`; assert `--use-heading-measurement` appears in the "filter parameters" group, not "smoother parameters". |
| `test_qa_report_nees_histogram_has_title` | Render report; introspect the matplotlib figure (use `pdf2image` if needed or unit-test the plot function directly); assert title contains "NEES". |
| `test_qa_report_start_end_markers_are_okabe_ito` | Unit-test the trajectory plot helper; assert the colors used for start/end markers are `#0072B2` / `#E69F00` (not `green` / `red`). |

## Fixtures

- All filter/smooth/report tests synthesize sessions via `simulate_rat_imu`.
- README cleanup needs no test fixtures.
- Help-text tests use argparse's `format_help()` directly — no subprocess.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- `grep -rn 'online' src/ tests/ docs/` returns zero results (modulo the CHANGELOG entry that documents the rename).
- `git mv` was used so the file history is preserved.
- Every task in this phase is implemented.
- The deferred "per-frame JAX-progress callbacks via jax.debug.callback" decision is honored — Task 4 should ship with the chunked-tqdm approach OR with only the JIT-warmup + total-runtime messages if chunking would require refactoring `extended_kalman_filter`'s signature.
- The `report --from-run` bridge does NOT require a separate config file (the helper extracts what it needs from the `filter`/`smooth` output's `metadata.txt` if present).
- README on-ramp does not include the moved "Project Status" content.
- The CHANGELOG entry is in `## [0.3.0]` (minor version bump for the breaking CLI rename).
- Validation slice tests pass.
- Docstrings, test names, and module names don't reference this plan.

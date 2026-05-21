# Phase 3 — Scaffolding cleanup (PRD references, stale comments)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase is a coordinated rename + comment sweep. No behavior changes — Phase 3 must produce byte-identical filter outputs on the same inputs (see [Baseline-then-refactor pattern](shared-contracts.md#baseline-then-refactor-pattern)).

**Inputs to read first:**

- [src/trodestrack/qa/report.py](../../../../src/trodestrack/qa/report.py) lines 38-41 — `PRD_*` constants. These are the canonical definitions; every other PRD reference traces back to these.
- [src/trodestrack/qa/plots.py](../../../../src/trodestrack/qa/plots.py) — the broadest spread of PRD references (parameter docs, comments, *and* user-facing plot legend labels).
- [src/trodestrack/qa/metrics.py](../../../../src/trodestrack/qa/metrics.py) lines 4, 902, 937-961, 985 — PRD references in docstrings.
- [src/trodestrack/viz/components.py](../../../../src/trodestrack/viz/components.py) lines 1257-1322 — user-facing legend labels like `"PRD: ±10 cm/s"`. Also the file with ~50 trivial "what" comments documented in the comment-analyzer report.
- [src/trodestrack/models/ukf.py](../../../../src/trodestrack/models/ukf.py) lines 455, 566, 706 — stale shape annotations `(17, 8)` / `(17, 4)`, stale line-number reference "see around line 362".
- [src/trodestrack/models/sensors/camera_position_3d.py](../../../../src/trodestrack/models/sensors/camera_position_3d.py) lines 5-7 — "Arthur-specific" docstring.
- [src/trodestrack/models/sensors/camera_position.py](../../../../src/trodestrack/models/sensors/camera_position.py) around line 390 — "Critical for PR2/PR3 JAX compatibility" docstring reference. (Re-grep `PR2/PR3` at edit time to find current line number.)
- [src/trodestrack/models/state_layout.py](../../../../src/trodestrack/models/state_layout.py) lines 289-290 — `LAYOUT_3D_CAM_6DOF_IMU` docstring: "the explicit state-mode name used by the tilt/orientation implementation plan." (Verify the line range — Phase 1 may have shifted it.)
- [src/trodestrack/models/filter_common.py](../../../../src/trodestrack/models/filter_common.py) line 2591 — `"# IMU Index Computation (previously in filter_utils.py)"`.

**Contracts referenced:**

- [No backwards-compatibility shims](shared-contracts.md#no-backwards-compatibility-shims) — `PRD_*` constants are deleted, not aliased to `TARGET_*`.
- [Baseline-then-refactor pattern](shared-contracts.md#baseline-then-refactor-pattern) — capture output baseline before any change; assert bit-exact equality after.

## Tasks

### Task 1 — Capture pre-refactor baseline

Before any edits, run one example end-to-end and pickle the outputs to a temporary location:

```bash
uv run python -c "
import pickle, numpy as np
from pathlib import Path
import subprocess, json
# Run example 08 which exercises qa.report + qa.plots + qa.metrics
subprocess.check_call(['uv', 'run', 'python', 'examples/08_qa_report_generation.py'])
# Capture metric outputs by importing the QA module directly with a fixed seed
from trodestrack.qa.metrics import compute_nees_stats, compute_dropout_drift
# ... (use a deterministic 30 s simulated session, save metric arrays)
"
```

Save the pickle path in the PR description so the comparison task can find it. Suggested location: `/tmp/phase3-baseline-$(git rev-parse --short HEAD).pkl`.

### Task 2 — Rename `PRD_*` constants in `qa/report.py`

In [src/trodestrack/qa/report.py:38-41](../../../../src/trodestrack/qa/report.py#L38-L41), replace the constant block:

```python
# Project acceptance targets for filter quality (rat tracking, ground-truth-aligned)
TARGET_POSITION_RMSE_M = 0.02       # 2 cm
TARGET_VELOCITY_RMSE_MS = 0.10      # 10 cm/s
TARGET_HEADING_MAE_DEG = 7.0        # 7 degrees
```

Update every reference in this file. Grep: `grep -nE 'PRD' src/trodestrack/qa/report.py`. The summary-page string templates around line 347-355 also use the literal text `"(PRD req: ≤2.0 cm)"` — rewrite as `"(target: ≤2.0 cm)"` (note: Phase 4 will further enrich this with PASS/FAIL; Phase 3 keeps it informational).

### Task 3 — Rename PRD references in `qa/plots.py`

Grep for `PRD` (case-insensitive) in [src/trodestrack/qa/plots.py](../../../../src/trodestrack/qa/plots.py): `grep -niE 'prd' src/trodestrack/qa/plots.py`. Documented hits (verify line numbers at edit time): 174, 188-189, 201, 205, 240-247, 291, 345, 360, 378, 421, 428.

For parameter docs (`"""...PRD threshold..."""`), rewrite as `"...project acceptance target..."`. For inline comments (`# PRD requirement`), rewrite as `# acceptance target`. For user-facing matplotlib `label=` strings like `label="PRD threshold"`, rewrite as `label="target"`. These legend labels ship to users in the QA report PDF; the change is user-visible.

**Public function-parameter renames** (these are not just comment changes — they break callers): `plot_position_error(..., prd_threshold_m=...)` → `plot_position_error(..., target_threshold_m=...)`. Same for `plot_velocity_error`'s `prd_threshold_m` and `plot_heading_error`'s `prd_threshold_deg`. Update every caller (grep `prd_threshold_m`, `prd_threshold_deg` across `src/` and `tests/`).

### Task 4 — Rename PRD references in `qa/metrics.py`

In [src/trodestrack/qa/metrics.py](../../../../src/trodestrack/qa/metrics.py), rewrite the docstrings at line 4 (module-level), 902 (`compute_dropout_drift`), 937-961, 985:

- Module docstring (line 4): change `"PRD requirements (all in SI units)..."` to `"Project acceptance targets (all in SI units)..."`.
- `compute_dropout_drift` (line 902): change `"The PRD requirement is that *IMU-only tracking error*..."` to `"The project acceptance target is that *IMU-only tracking error*..."`.
- "PRD Acceptance Criteria (§4.2)" references: change to `"Project acceptance criteria"` (drop the section reference; no external doc to cite).

### Task 5 — Rename PRD references in `viz/components.py`

In [src/trodestrack/viz/components.py](../../../../src/trodestrack/viz/components.py) lines 1257-1322 (verify with `grep -nE 'PRD' src/trodestrack/viz/components.py`):

- Variable/comment references like `# PRD target line: ±10 cm/s` → `# acceptance target line: ±10 cm/s`.
- User-facing `label="PRD: ±10 cm/s"` (line 1292, 1322 per the comment-analyzer report) → `label="target: ±10 cm/s"`.

### Task 6 — Fix stale shape annotations in `models/ukf.py`

In [src/trodestrack/models/ukf.py](../../../../src/trodestrack/models/ukf.py):

- Line 455: `sigmas_prop = vmap(f)(sigmas)  # (17, 8)` → `sigmas_prop = vmap(f)(sigmas)  # (2n+1, n)`.
- Line 566: `sigmas_meas = vmap(camera_model.predict)(sigmas)  # (17, 4)` → `sigmas_meas = vmap(camera_model.predict)(sigmas)  # (2n+1, meas_dim)`.

### Task 7 — Fix stale line-number cross-reference

In [src/trodestrack/models/ukf.py:706](../../../../src/trodestrack/models/ukf.py#L706), change `"This mirrors the wrap applied in predict_step (see around line 362)."` to `"This mirrors the wrap_angle call in predict_step's covariance reconstruction."`. The verified actual line is 475 (`heading_deviations = wrap_angle(deviations[:, h_idx])`), but the rewrite uses a symbol reference rather than a line number so it doesn't rot again.

### Task 8 — Drop "Arthur-specific" docstring

In [src/trodestrack/models/sensors/camera_position_3d.py:5-7](../../../../src/trodestrack/models/sensors/camera_position_3d.py#L5-L7), change `"It is intentionally independent of any loader or Arthur-specific data shape."` to `"It is intentionally independent of any specific loader or upstream dataset shape."`

### Task 9 — Drop scaffolding-doc references in remaining files

- `src/trodestrack/models/sensors/camera_position.py` around line 390: remove `"Critical for PR2/PR3 JAX compatibility:"` from the docstring. The "JAX compatibility" rationale itself is real — keep it, drop the `PR2/PR3` reference. Rewrite as `"Critical for JAX-compatibility: ..."`. Re-grep `PR2/PR3` at edit time to confirm there aren't additional hits.
- `src/trodestrack/models/state_layout.py` around lines 289-290 (`LAYOUT_3D_CAM_6DOF_IMU` docstring): drop "the explicit state-mode name used by the tilt/orientation implementation plan." Rewrite as `"the explicit state-mode name for 3D camera + 6-DOF IMU."`.
- [src/trodestrack/models/filter_common.py:2591](../../../../src/trodestrack/models/filter_common.py#L2591): drop `(previously in filter_utils.py)`. The section header becomes `# IMU Index Computation`.

### Task 10 — Sweep trivial "what" comments in `viz/components.py`

This is the largest single touch. Read [src/trodestrack/viz/components.py](../../../../src/trodestrack/viz/components.py) end-to-end. Delete trivial section-header comments that just announce the next 1-3 lines. Examples from the comment-analyzer report (verify at edit time — line numbers may have shifted):

- Line 129: `# Update body position` (the next line is `self.body.center = (x, y)`)
- Line 132: `# Update heading arrow using transform (no recreation needed)`
- Line 200: `# LED marker: colored circle (reduced size for better hierarchy)`
- Lines 296, 302, 318, 326, 339: `# Show LED marker and confidence halo`, `# Show residuals if enabled...`, etc.
- Many more in the 400-980 range (see comment-analyzer report).

Keep:
- The dropout-marker `None`-guard comments at lines 254-258 and 330-333 (explain a bug).
- The `_set_scrolling_xlim` rationale (matplotlib warning suppression).
- Trail color resolution at lines 381-384.

Rule of thumb: if removing the comment would NOT confuse a future reader of the next 3-5 lines, delete it. If the next lines need the comment to be understood, keep it.

After the sweep, run `wc -l src/trodestrack/viz/components.py` and report the LOC reduction in the PR description. Expected reduction: ~40-80 lines.

### Task 11 — Drop other small redundancies flagged by the comment-analyzer

- [src/trodestrack/models/ukf.py:97-99](../../../../src/trodestrack/models/ukf.py#L97-L99): trailing comments on `UKFConfig.alpha` field — delete (docstring already covers).
- [src/trodestrack/models/ukf.py:197-201](../../../../src/trodestrack/models/ukf.py#L197-L201): historical "Prior versions of this preset used alpha=1e-3..." Note block — delete.
- [src/trodestrack/models/ukf.py:301](../../../../src/trodestrack/models/ukf.py#L301): `# Compute Cholesky decomposition: P = L @ L.T` — delete (line below is `jnp.linalg.cholesky(cov_reg)`).
- [src/trodestrack/models/ukf.py:429-431](../../../../src/trodestrack/models/ukf.py#L429-L431): trailing comment on `n = len(m)  # State dimension, resolved from layout (...)` — delete.
- [src/trodestrack/models/ukf.py:458-462](../../../../src/trodestrack/models/ukf.py#L458-L462): delete `# Reconstruct predicted mean (weighted sum)` and the `# Note: For heading dimension, we need circular mean, but it's handled below` line; keep the two-line "circular mean" rationale beneath.
- [src/trodestrack/models/ukf.py:1087,1111](../../../../src/trodestrack/models/ukf.py#L1087) and [src/trodestrack/models/ekf.py:975](../../../../src/trodestrack/models/ekf.py#L975): `# Convert to JAX arrays`, `# Use original config as-is` — delete.
- [src/trodestrack/sim/rat_imu.py:230-305](../../../../src/trodestrack/sim/rat_imu.py#L230-L305): trailing field comments that duplicate the docstring — delete the duplicative ones, KEEP the ones adding units the docstring lacks (e.g. `# rad/s / √s` on `gyro_bias_rw_density`).

### Task 12 — Grep sweep to confirm cleanup

After all edits, run:

```bash
grep -rniE '\bprd' src/ tests/
grep -nE '\(17, [0-9]+\)' src/trodestrack/models/ukf.py
grep -rn 'Arthur' src/
grep -rnE '(see around line|previously in filter_utils|PR2/PR3)' src/
grep -rnE 'prd_threshold' src/ tests/
```

All five greps should return zero hits. If anything remains, fix it. The grep on `prd` is case-insensitive (`-i`) because identifiers like `prd_threshold_m` are lowercase.

### Task 13 — Post-refactor baseline comparison

Re-run the same example from Task 1; assert array equality against the pickle:

```python
import pickle, numpy as np
baseline = pickle.load(open("/tmp/phase3-baseline-<hash>.pkl", "rb"))
# Re-run, collect new outputs, compare:
for key in baseline:
    np.testing.assert_array_equal(baseline[key], new[key]), f"divergence at {key}"
```

Phase 3 is comment-only; bit-exact equality is required.

### Task 14 — CHANGELOG entry

Add under `## [0.2.2] — unreleased`:

```
### Changed
- QA report and diagnostic plot legends now use "target" instead of "PRD" terminology. Affected user-visible strings include plot legend labels like "PRD: ±10 cm/s" → "target: ±10 cm/s" in the QA PDF and live diagnostic videos.

### Internal
- Renamed PRD_* constants to TARGET_* throughout qa/. No public API; the constants were not exposed in __all__.
- Removed scaffolding-doc references ("Arthur-specific", "PR2/PR3", "previously in filter_utils.py") from source docstrings.
- Fixed stale shape annotations in models/ukf.py.
- Sweep of redundant section-header comments in viz/components.py.
```

## Deliberately not in this phase

- **PASS/FAIL banner on QA report summary page** — Phase 4 (UX improvement that changes the summary structure, not just constant names).
- **NEES/NIS histogram titles** — Phase 4.
- **Colorblind-safe start/end markers** — Phase 4.
- **Pickling baseline-comparison utility** — one-shot for this phase; do not commit the pickle or baseline-running script.

## Validation slice

| Test | Asserts |
| --- | --- |
| `grep -rnE '\bPRD' src/ tests/` | Returns zero hits after Task 12. |
| `grep -nE '\(17, [0-9]+\)' src/trodestrack/models/ukf.py` | Returns zero hits. |
| `grep -rn 'Arthur' src/` | Returns zero hits. |
| Existing test suite passes unchanged | `uv run pytest` exits 0 with the same test count as `master` HEAD. |
| Pre/post pickle comparison (Task 13) | All baseline arrays bit-exact equal to post-refactor outputs. |
| `wc -l src/trodestrack/viz/components.py` after vs before | LOC reduced by 40-80 (recorded in PR description). |

No new tests are required (Phase 3 is comment-only). The existing test suite is the validation.

## Fixtures

Reuse the simulated session from `examples/08_qa_report_generation.py` for baseline capture. No checked-in fixtures.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Grep sweeps (Task 12) all return zero.
- Existing test suite passes with the same test count.
- The pickle comparison in Task 13 succeeded bit-exact.
- No production code (anything beyond comments, string literals, and constant names) was modified. Diff should show only comment/string/identifier changes — no logic edits.
- The "Deliberately not in this phase" list is honored (no PASS/FAIL banner, no colorblind fix — those are Phase 4).
- Docstrings, test names, and module names don't reference this plan.
- CHANGELOG updated with both user-visible ("target" vs "PRD" legends) and internal items.

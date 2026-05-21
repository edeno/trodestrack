# Phase 2 — Test coverage gaps

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [tests/filters/test_ekf_analytic.py](../../../../tests/filters/test_ekf_analytic.py) — the gold-standard analytic-correctness test pattern for 2D. The 3D tests should mirror its structure: deterministic seed, simulate ground truth, run filter, assert RMSE/MAE/NEES bounds with explicit cm/deg units.
- [tests/runtime/test_smoother_layout_smoke.py](../../../../tests/runtime/test_smoother_layout_smoke.py) lines around 323-453 — existing 3D smoother smoke tests. The new 3D-EKF tests use the same fixture style.
- [src/trodestrack/models/ekf.py](../../../../src/trodestrack/models/ekf.py) line 1451 onward — `extended_kalman_filter_3d` entry point. Read its docstring and required inputs.
- [src/trodestrack/cli/report.py](../../../../src/trodestrack/cli/report.py) — the production code being tested. Note `load_run_data` and the `--run` directory layout it expects.
- [tests/cli/test_report_command.py](../../../../tests/cli/test_report_command.py) — currently an empty shell class.
- [tests/cli/test_smooth_command.py](../../../../tests/cli/test_smooth_command.py) — existing CLI test pattern using `patch("sys.argv", [...])`. Use the same style.
- [examples/08_qa_report_generation.py](../../../../examples/08_qa_report_generation.py) — example showing how a `qa_inputs/` directory is constructed. The new `test_report_command_*` tests reuse this code path to build a fixture run directory.
- [src/trodestrack/sim/rat_imu.py](../../../../src/trodestrack/sim/rat_imu.py) — `simulate_rat_imu` for the 3D-EKF and safety-check tests.
- [tests/models/test_orientation.py](../../../../tests/models/test_orientation.py) — existing orientation tests; the gimbal-lock test extends this file.
- [tests/io/test_session_loading.py](../../../../tests/io/test_session_loading.py) lines 664-1008 — current safety-check tests mock the vision EKF; the new real-vision test should not.

**Contracts referenced:**

- [Test-file layout convention](shared-contracts.md#test-file-layout-convention) — file naming and `@pytest.mark.slow` policy.

## Tasks

### Task 1 — Fill `tests/cli/test_report_command.py`

The current file is an empty `TestReportCommand` class shell. Write end-to-end tests that exercise `cli/report.py::load_run_data` and the full `report` subcommand on a real fixture directory built from a `smooth` run.

Required tests (in `tests/cli/test_report_command.py`):

- `test_report_command_renders_pdf_from_qa_inputs_directory` — Build a `qa_inputs/` directory in `tmp_path` using the same code path as [examples/08_qa_report_generation.py](../../../../examples/08_qa_report_generation.py). Invoke `trodestrack report --run tmp_path/qa_inputs --pdf tmp_path/report.pdf` via `patch("sys.argv", [...])`. Assert: the PDF exists and is non-empty; `load_run_data` returns a dict with the expected keys.
- `test_report_command_raises_friendly_error_when_required_file_missing` — Same fixture but delete `positions_true.npy`; the command should exit 1 with stderr matching `"missing"` and naming the absent file.
- `test_report_command_validates_array_shape_consistency` — Build the fixture but make `positions_est` shape `(N+1, 2)` while `positions_true` is `(N, 2)`; expect a clean shape-mismatch error rather than a cryptic broadcast failure.
- `test_load_run_data_returns_expected_keys_and_shapes` — Direct unit test of `load_run_data`: build a fixture, call the function, assert the returned dict has `t, positions_true, positions_est, velocities_true, velocities_est, headings_true, headings_est, nees, state_dim` and the array shapes are consistent (`t.shape == (N,)`, `positions_*.shape == (N, 2)`, etc.).
- `test_report_command_with_custom_title_appears_in_pdf` — Use `--title "Session 2024-10-11"`; open the PDF (use `pypdf` if available, else assert the title bytes appear in the PDF file via `pdf_bytes.find(b"Session 2024-10-11") != -1`).

### Task 2 — Analytic correctness tests for `extended_kalman_filter_3d`

Create `tests/filters/test_ekf_3d_analytic.py`. Mirror the structure of [tests/filters/test_ekf_analytic.py](../../../../tests/filters/test_ekf_analytic.py).

Required tests:

- `test_ekf_3d_stationary_pitch_roll_recovers_gravity_orientation` — Simulate a stationary headstage with true pitch=10°, roll=5°, yaw=0°. Run `extended_kalman_filter_3d` for 30 s at 100 Hz IMU; assert recovered pitch and roll are within 2° of truth, quaternion norm stays at 1 ± 1e-6 throughout.
- `test_ekf_3d_yaw_only_motion_converges_gyro_bias_z` — Simulate constant yaw rate of 30°/s with a true gyro_z bias of 0.05 rad/s; assert the estimated bias converges to within 5e-3 rad/s of truth after 10 s.
- `test_ekf_3d_5s_dropout_drift_under_prd_target` — Simulate 30 s with vision blacked out from t=10 s to t=15 s; assert position drift during the dropout is below 0.15 m (matches the 2D bound at [tests/filters/test_ekf_analytic.py:564](../../../../tests/filters/test_ekf_analytic.py#L564)).
- `test_ekf_3d_nees_consistency_on_4d_state` — Compute NEES on the 4D `(x, y, z, yaw)` subset of state. Assert mean NEES is between 1.0 and 8.0 (looser than the 2D bound because gravity-coupled orientation has wider innovation tails). Add a TODO comment to tighten once 3D filter tuning stabilizes, mirroring the existing TODO at [tests/filters/test_ekf_analytic.py:553-556](../../../../tests/filters/test_ekf_analytic.py#L553-L556).
- `test_ekf_3d_perfect_input_no_drift` — Feed the filter perfect-truth measurements (zero noise); assert that posterior means equal truth within `rtol=1e-5` and covariances stay bounded.

Mark all five with `@pytest.mark.slow` (each simulates a full 30 s session — ~3-10 s wall-clock per test).

Add a parallel `tests/filters/test_ukf_3d_analytic.py` with the same five tests, calling `unscented_kalman_filter(state_mode="3d_quat", ...)`. Mark `slow` likewise.

### Task 3 — Gimbal-lock test for `estimate_orientation`

Extend [tests/models/test_orientation.py](../../../../tests/models/test_orientation.py) with:

- `test_orientation_near_pitch_singularity_keeps_quaternion_unit_norm` — Simulate stationary headstage at true pitch=88°, then 92° (both within 2° of the gimbal-lock singularity). Run `estimate_orientation` for 5 s at 100 Hz. Assert: `result.quaternion` has norm 1 ± 2e-7 throughout; `result.yaw` is finite (no NaN); `result.pitch` recovers within 3° of truth (looser tolerance than non-singular cases — the gimbal-lock region is mathematically harder).
- `test_orientation_through_pitch_singularity_does_not_nan` — Simulate a pitch sweep from -89° to +89° over 10 s; assert no NaN in quaternion, roll, pitch, or yaw outputs at any timestep.

### Task 4 — Real-vision-EKF safety check test

Add to [tests/io/test_session_loading.py](../../../../tests/io/test_session_loading.py):

- `test_safety_check_passes_on_clean_simulated_session_with_real_vision_ekf` — Use `simulate_rat_imu` to build a `PreparedSession` with known-good geometry (rat in a 1m × 1m arena, dual-LED visible 90% of frames). Call `run_real_data_safety_check` WITHOUT patching `extended_kalman_filter` — the real vision-only EKF should run. Assert `report.passed is True`; assert deviation metrics are below configured envelope.
- `test_safety_check_flags_implausible_session_with_real_vision_ekf` — Same setup but inject 200% LED-position noise; assert `report.passed is False` and the message names the failing metric.

These tests are not redundant with the existing mocked tests at lines 664-1008 — they validate that the *real* vision EKF used inside the safety check converges on a known-good session, which the mocks can't establish.

### Task 5 — CLI smooth/online finiteness tests

Extend [tests/cli/test_smooth_command.py](../../../../tests/cli/test_smooth_command.py) and `tests/cli/test_online_command.py`:

- `test_smooth_command_outputs_are_finite_and_psd` — Run the smooth command on a 10 s simulated session, load `smoothed_means.txt` and `smoothed_covariances.txt`, assert all values are finite, all covariance matrices are symmetric within `rtol=1e-8`, and the smallest eigenvalue is > 0.
- Equivalent test for the forward-only filter command. If Phase 4 has already shipped (CLI rename to `filter`), name it `test_filter_command_outputs_are_finite_and_psd`; otherwise name it after the current command (`test_online_command_outputs_are_finite_and_psd`) and Phase 4's rename will sweep it. Either way the name describes the command, not the plan milestone.

### Task 6 — Focused unit tests for un-covered sensor models

Create `tests/models/sensors/test_heading_pseudo.py` (currently no dedicated file):

- `test_heading_pseudo_predict_matches_hand_derived` — One frame with known LED1, LED2 positions and known heading; assert `predict(state)` matches the closed-form `atan2(led2_y - led1_y, led2_x - led1_x)`.
- `test_heading_pseudo_jacobian_matches_jacfwd` — Use `jax.jacfwd(model.predict)` to compute reference; assert `model.jacobian(state)` agrees to `rtol=1e-6` for 20 random states.
- `test_heading_pseudo_innovation_wraps_to_minus_pi_pi` — Innovation when measurement is +179° and prediction is -179° should be ≈ -2° (or +2° — verify against the implementation's convention), not ±358°.
- `test_heading_pseudo_gate_rejects_when_led_spacing_implausible` — Construct a state where the implied LED spacing differs from `config.led_distance` by 50%; assert the gate returns no-update.

Create `tests/models/sensors/test_camera_position.py` (currently no dedicated file):

- `test_camera_position_predict_matches_hand_derived` — Known state and LED offsets; assert `predict()` matches the rotated-offset formula.
- `test_camera_position_jacobian_matches_jacfwd` — Same parity pattern as above.
- `test_camera_position_partial_observation_single_led` — Mask led2 invalid; assert the prediction returns NaN in the led2 slots and the Jacobian zeroes those rows.

### Task 7 — UKF dropout-drift parity test

Add to `tests/filters/test_ukf_accuracy.py` (or create `tests/filters/test_ukf_dropout.py` if the file is already long):

- `test_ukf_5s_dropout_drift_matches_ekf_within_factor_2` — Run both EKF and UKF on the same simulated session with the same 5 s dropout; assert UKF drift ≤ 2× EKF drift. (Tighter than absolute since UKF spreads sigma points more conservatively under no observations.)

### Task 8 — Property-based gap fill (optional within this phase)

If the executor finishes the above with time to spare, add hypothesis-driven invariant tests:

- `tests/models/test_quaternion.py` already covers unit-norm preservation; add a `tests/models/test_orientation.py` property test asserting `estimate_orientation(t, gyro=0, accel=g)` produces a quaternion whose conjugate-applied gravity matches `g` for any starting orientation, modulo yaw.

This task is `Deliberately not required for the phase` — flag it explicitly in the PR description as "if convenient, otherwise defer."

## Deliberately not in this phase

- **`extended_kalman_filter_3d` JIT-compile-time reduction** — Phase 6 (changing the Python loop to `lax.scan`). Phase 2's tests provide the parity oracle that Phase 6 verifies against.
- **`cli/report.py` UX bridge** (allowing `report` to consume `online`/`smooth` outputs directly without a separate `qa_inputs/` step) — Phase 4. Phase 2's tests use the current expected directory layout.
- **3D EKF input validation** — out of scope; Phase 2 only adds tests, doesn't modify production code.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_report_command_renders_pdf_from_qa_inputs_directory` | PDF exists at `tmp_path/report.pdf`, size > 1 KB; `load_run_data` returns dict with all expected keys. |
| `test_report_command_raises_friendly_error_when_required_file_missing` | Exit code 1; stderr contains "missing" and the absent filename. |
| `test_report_command_validates_array_shape_consistency` | Clean error matching "shape mismatch" or similar; no Python traceback. |
| `test_load_run_data_returns_expected_keys_and_shapes` | All expected keys present; array shapes consistent (`positions_*` are `(N, 2)`, `nees` is `(N,)`). |
| `test_ekf_3d_stationary_pitch_roll_recovers_gravity_orientation` | Recovered pitch within 2° of truth; quaternion norm 1 ± 1e-6. |
| `test_ekf_3d_yaw_only_motion_converges_gyro_bias_z` | Estimated `b_gz` within 5e-3 rad/s of true value after 10 s. |
| `test_ekf_3d_5s_dropout_drift_under_prd_target` | Position drift during 5 s vision blackout ≤ 0.15 m. |
| `test_ekf_3d_nees_consistency_on_4d_state` | Mean NEES in [1.0, 8.0]; documented TODO for tightening. |
| `test_ekf_3d_perfect_input_no_drift` | Posterior means equal truth within `rtol=1e-5`. |
| `test_ukf_3d_*` (5 mirror tests) | Same bounds as 3D EKF, possibly looser by 20% to account for sigma-point spread. |
| `test_orientation_near_pitch_singularity_keeps_quaternion_unit_norm` | Quaternion norm 1 ± 2e-7 through pitch = ±88° and ±92°; yaw finite throughout. |
| `test_orientation_through_pitch_singularity_does_not_nan` | All outputs finite over a -89° to +89° pitch sweep. |
| `test_safety_check_passes_on_clean_simulated_session_with_real_vision_ekf` | `report.passed is True` without any EKF patching. |
| `test_safety_check_flags_implausible_session_with_real_vision_ekf` | `report.passed is False`; message names the failing metric. |
| `test_smooth_command_outputs_are_finite_and_psd` | All output values finite; covariances symmetric `rtol=1e-8`; min eigenvalue > 0. |
| `test_{filter,online}_command_outputs_are_finite_and_psd` | Same as above for the forward-only filter command; name matches the live command at the time Phase 2 lands. |
| `test_heading_pseudo_*` (4 tests) | `predict`, `jacobian` match hand-derived/jacfwd; innovation wraps; gate rejects implausible spacing. |
| `test_camera_position_*` (3 tests) | `predict`, `jacobian` parity; partial-observation masking. |
| `test_ukf_5s_dropout_drift_matches_ekf_within_factor_2` | `ukf_drift <= 2.0 * ekf_drift`. |

All 3D-EKF/UKF analytic tests are `@pytest.mark.slow`. The CLI report tests are fast (PDF generation on small fixtures completes in < 2 s).

## Fixtures

- 3D-EKF/UKF tests: use `simulate_rat_imu` from [src/trodestrack/sim/rat_imu.py](../../../../src/trodestrack/sim/rat_imu.py). Add a `simulate_3d_session` helper (no leading underscore — Phase 6 also imports it) in `tests/filters/conftest.py` that returns the `(t_imu, U_imu, t_cam, Z_cam_led1, Z_cam_led2, mask_cam, truth)` tuple for a deterministic seed. Reuse the helper across `test_ekf_3d_analytic.py` and `test_ukf_3d_analytic.py`.
- Report-command tests: write a `_build_qa_inputs_dir(tmp_path, n)` helper in `tests/cli/conftest.py` that creates `tmp_path/qa_inputs/{timestamps,positions_true,positions_est,...}.npy` with deterministic content. Mirror the structure from [examples/08_qa_report_generation.py](../../../../examples/08_qa_report_generation.py).
- Safety-check tests: extend the existing fixture pattern in [tests/io/test_session_loading.py](../../../../tests/io/test_session_loading.py). Use a fixed seed (`np.random.default_rng(42)`).

No on-disk test data files added — all fixtures are synthesized.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no production code changes.
- Validation slice tests pass; all 3D analytic tests are marked `@pytest.mark.slow`.
- Tests aren't trivial — assertions are quantitative (RMSE bounds, NEES ranges, parity tolerances), not "function runs without error."
- Test names don't reference this plan or its milestones (e.g. no `test_phase_2_*`).
- Test fixtures are shared via `conftest.py`, not copy-pasted across test files.
- Coverage report shows `cli/report.py` ≥ 80% (was 0%) and `models/ekf.py::extended_kalman_filter_3d` covered by at least 4 distinct test scenarios.

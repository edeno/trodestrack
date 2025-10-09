# trodestrack — Integrated Review & Prioritized Fix Plan

## Repository snapshot: key symbol locations

Below are symbol line maps to help you jump to the right places:

### `ekf.py`

- **extended_kalman_filter** — lines: 1122
- **update_heading** — lines: 1002
- **gaussian_log_likelihood** — lines: 316
- **apply_lifted_inverse** — lines: 216
- **chi2_threshold** — lines: 348

### `ukf.py`

- **unscented_kalman_filter** — lines: 508
- **update_step** — lines: 386
- **gaussian_log_likelihood_ukf** — lines: 273

### `offline.py`

- **rts_smoother** — lines: 60
- **sigma_point_smoother** — lines: 300
- **compute_imu_index_arrays** — lines: 114, 367

### `rat_imu.py`

- **simulate_rat_imu** — lines: 349
- **compute_gravity_in_tilted_frame** — lines: 45

### `simple.py`

- **simulate_stationary** — lines: 119
- **simulate_constant_velocity** — lines: 251

---

## P0 — Blockers (merge gate)

1. **Unify Units to SI (m, m/s, rad) in Metrics & Tests**
   - **Why**: Prevent silent threshold breakage and align with PRD reproducibility.
   - **Touchpoints**:
     - `src/trodestrack/qa/metrics.py` → normalize returns for `compute_position_rmse`, `compute_velocity_rmse`, `compute_heading_error` (see lines above).
     - Tests expecting degrees/centimeters: update assertions to SI

2. **Generalize χ² Envelopes and Apply Everywhere**
   - **Why**: Gating and consistency checks must track measurement DoF (2/4 now; higher later) and arbitrary confidences (90/95/99%).
   - **Implement**: `chi2_bounds(df:int, confidence:float)` + `within_envelope(values, df, confidence)` in `qa/metrics.py`; add `confidence` parameter to NEES/NIS stats.
   - **Touchpoints**:
     - `src/trodestrack/qa/metrics.py: compute_nees`, `compute_nis` (lines referenced above).
     - `tests/filters/test_ekf_gating.py` — remove `scipy.stats.chi2` import and call the helper instead.

3. **UKF Feature Parity: Add Heading Pseudo‑Measurement**
   - **Why**: PRD parity and robustness under dual‑LED visibility.
   - **Touchpoints**:
     - `src/trodestrack/models/ukf.py: update_step` — add a 1‑D heading update mirroring EKF’s `update_heading` (see EKF line refs).
     - Reuse a common `joseph_update(P,K,H,R)` helper.

4. **State‑Dimension Generalization in Smoothers**
   - **Why**: Future 3D and state changes must not break smoothing.
   - **Touchpoints**:
     - `src/trodestrack/runtime/offline.py: rts_smoother`, `sigma_point_smoother` — derive `n = filtered_means.shape[-1]`; remove hardcoded 8s.
     - Extract `build_Q_rate(config, n)`; use in both EKF/UKF smoothers.

5. **Stability: Linalg Hardening & Joseph Form**
   - **Why**: Prevent divergence with near‑singular covariances and keep PSD.
   - **Touchpoints**:
     - `src/trodestrack/models/ekf.py: gaussian_log_likelihood`, `apply_lifted_inverse` — add jitter and sign checks around `slogdet`; prefer Cholesky when feasible.
     - `src/trodestrack/models/ukf.py: gaussian_log_likelihood` — ditto.
     - New helper `joseph_update(P,K,H,R)` used in EKF position/heading and UKF.

6. **Reproducibility: Avoid Runtime Config Mutation**
   - **Why**: Immutable configs are a PRD requirement for reproducibility.
   - **Touchpoints**:
     - `src/trodestrack/models/ekf.py: extended_kalman_filter` — when auto‑inferring LED spacing, do not mutate config; instead return `estimated_led_distance` in the result and log it.

7. **Fix Test Defects & Flakes**
   - **Why**: CI stability and correctness.
   - **Touchpoints**:
     - `tests/filters/test_dropout_diagnostic.py` — script‑style side effects at import; move under `diagnostics/` or guard with `if __name__ == '__main__':`.
     - `tests/filters/test_ukf_accuracy.py` — replace any `compute_position_rmse` used on velocities with `compute_velocity_rmse`.
     - `tests/filters/test_vision_robustness.py` — make swaps verifiable (compare against truth arrays or a `swapped` mask).
     - `tests/filters/test_prd_acceptance.py` — mark the 5‑s dropout drift assertion as `xfail(strict=False)`.

---

## P1 — Should land before an alpha tag

8. **Typed Results & Validators**
   - Dataclasses/TypedDicts for `DropoutDriftResult`, `NEESStats`, `NISStats`, `FilterResult`, `SmootherResult`.
   - Central `_check_shapes/_normalize_mask` helpers in `qa/metrics.py`.

9. **Vectorize Hot Paths**
   - Stack Cholesky factors and use triangular solves in batch for NEES/NIS.
   - `vmap` column solves in EKF (`inv_S_HP`) to remove Python loops.

10. **Docstrings to CLAUDE Spec**

- NumPy style for every public function: **Units / Range / Default / Example / PRD citation**.
- Emphasize accelerometer specific‑force vs inertial acceleration in sim docs.

11. **Determinism Plumbing**

- One RNG service; capture `seed` and derived values (e.g., inferred LED spacing) in results/logs.
- Build IMU index arrays entirely in NumPy, then feed JAX to avoid device variance.

12. **Per‑Timestep Covariance Dominance in Smoothers**

- Assert `P_smooth[k] ⪯ P_filt[k]` for all k (PSD of difference ≥ −tol).

---

## P2 — Quality / polish

13. **LED Swap Mixture Update (optional)**

- Two‑hypothesis (swap/no‑swap) measurement update; softens decisions under ambiguity.

14. **Dropout Mitigations**

- ZUPT hook; optionally freeze bias during long gaps; adaptive Q during dropouts.

15. **FFT ACF & Benchmarks**

- `method="direct|fft"` in ACF; add `pytest-benchmark` coverage for core ops.

---

## File-by-file actionable nits (surgical)

- **models/ekf.py**
  - Add jitter/sign check in `gaussian_log_likelihood` (see lines above).
  - Replace any ad‑hoc covariance update with a shared `joseph_update`.
  - In `extended_kalman_filter`, return `estimated_led_distance` instead of mutating config.
  - Consider `vmap` over columns in `apply_lifted_inverse`.

- **models/ukf.py**
  - Implement heading pseudo‑measurement in `update_step`.
  - Use shared `joseph_update` and jittered log‑likelihood.

- **runtime/offline.py**
  - Derive `n` from data in `rts_smoother` and `sigma_point_smoother`.
  - Factor `build_Q_rate(config, n)`.
  - Ensure `compute_imu_index_arrays` is backend‑consistent (NumPy→JAX once).

- **sim/rat_imu.py & sim/simple.py**
  - Strengthen docstrings with units and emphasize **specific force** for `U_imu`.
  - (Optional) Return `X_truth_cam` sampled at camera times; include `seed` in output.

- **qa/metrics.py**
  - Normalize units to SI; add `chi2_bounds`, `within_envelope`, typed results, and singular‑policy argument (`on_singular="inf|nan|raise"`).
  - Vectorize NEES/NIS loops.

- **tests/**
  - Remove top‑level side effects, fix velocity RMSE calls, standardize `(truth, estimate)` argument order, and add per‑timestep PSD dominance checks in smoother tests.

---

## Acceptance checklist (to flip to APPROVE)

- [ ] Metrics return SI units; heading helpers consistent.
- [ ] χ² utilities in place; tests use them (or SciPy kept but consistent).
- [ ] UKF heading pseudo‑measurement passes parity tests.
- [ ] Smoothers dimension‑agnostic; tests assert PSD dominance per step.
- [ ] No config mutation at runtime; inferred params surfaced in results.
- [ ] Tests are deterministic; no script side effects in `tests/`.

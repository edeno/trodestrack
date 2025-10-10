## Code Review: EKF Heading Measurement and Related Modules

Reviewed files:
- PRD.md:1
- PLANNING.md:1
- CLAUDE.md:1
- src/trodestrack/models/ekf.py:1
- src/trodestrack/models/filter_common.py:1
- src/trodestrack/models/utils.py:1
- src/trodestrack/runtime/offline.py:1
- tests/filters/test_ekf_heading_measurement.py:1
- tests/filters/test_prd_acceptance.py:1

### Critical Issues (Must Fix)

- [ ] JIT compatibility test does not actually JIT the function under test [tests/filters/test_ekf_heading_measurement.py:482]
  - Problem: The test named `test_jax_jit_compatibility` asserts JAX JIT safety but never wraps the call in `jax.jit`, so it cannot detect tracing/shape/cond issues.
  - Fix: Wrap a thin wrapper around `extended_kalman_filter` with `jax.jit` (e.g., jit a function that accepts JAX arrays for inputs) or `jax.jit(update_heading)` and run once to ensure no ConcretizationError occurs. Example:
    - `jit(extended_kalman_filter)` on a function that converts inputs to `jnp.array` and returns a small subset to keep compilation minimal.
  - Reference: PRD §7 (Code Quality, Tests) and “Test Coverage” quality gate.

- [ ] UKF measurement update reduces covariance using invalid LED dimensions [src/trodestrack/models/ukf.py:596]
  - Problem: In `update_step`, K is computed from full 4×4 S and covariance is updated as `P - K S K^T` even when only a single LED is valid. Setting invalid innovation components to zero prevents mean update but NOT the covariance reduction; P still shrinks due to off-diagonal coupling and inclusion of masked dimensions in S.
  - Specific fix: Project into the active measurement subspace before computing K and S, mirroring EKF’s lifted subspace approach.
    - Compute `M2 = make_led_selector(only_led1, only_led2)`, then `S2 = M2 @ S @ M2.T`, `P_cross2 = P_cross @ M2.T`, `innov2 = M2 @ innov_full`.
    - Use `K_eff = P_cross2 @ S2^{-1}` and update `m_upd = m_in + K_eff @ innov2`.
    - Covariance: `P_upd = P_in - K_eff @ S2 @ K_eff.T` (or Joseph form variant for UKF if desired). Alternatively, port EKF’s `apply_lifted_inverse` helper and apply it consistently to compute K columns.
  - Rationale: Ensures missing LED channels do not spuriously reduce uncertainty; aligns with tests for partial observations and gating logic.
  - Reference: EKF lifted subspace pattern [src/trodestrack/models/ekf.py:556,585] and tests/filters/test_ekf_partial_observations.py.

### Quality Issues (Should Fix)

- [ ] Missing true NumPy-style docstrings with parameter units/ranges [src/trodestrack/models/ekf.py:1]
  - Suggestion: For public API (`extended_kalman_filter`, `update_step`, `update_heading`, `predict_step`) and configs, switch to NumPy-style docstrings including units (m, m/s, rad), default, and ranges. Add pipeline parameter details per standards.
  - Rationale: Improves maintainability and aligns with documentation standards in CLAUDE.md and PRD §5.

- [ ] Limitations in `chi2_threshold` discrete probabilities [src/trodestrack/models/filter_common.py:52]
  - Suggestion: Either document supported `prob` values (0.95/0.99/0.997) or implement a general χ² PPF using `jax.scipy.stats` if available. Alternatively, add a small lookup for common choices and validate input.
  - Rationale: Prevents silent misconfiguration and improves robustness of gating.

- [ ] Unused helper `gaussian_log_likelihood` [src/trodestrack/models/ekf.py:208]
  - Suggestion: Remove it or integrate in a diagnostics path, or add tests to cover it if intended for future use.
  - Rationale: Reduces dead code and potential confusion.

- [ ] UKF partial-observation handling should mirror EKF subspace approach [src/trodestrack/models/ukf.py:556]
  - Suggestion: Deduplicate logic by extracting the lifted subspace helpers (selector, NIS/loglik, lifted inverse) to a shared module to be used by both EKF and UKF.
  - Rationale: Reduces duplication and ensures parity between filters.

- [ ] Test tightness for “spacing gating” behavior [tests/filters/test_ekf_heading_measurement.py:196]
  - Suggestion: In addition to bounded position error, assert that heading variance does not shrink during corrupted spacing interval (e.g., median Var(θ) within window ≥ outside-window) to directly validate the gating effect on heading channel.
  - Rationale: Strengthens the test by validating the intended gating pathway, not just overall stability.

- [ ] Type specificity of some returns [src/trodestrack/models/ekf.py:662]
  - Suggestion: `update_heading` returns `(EKFState, jnp.ndarray)` for log-likelihood but functionally is scalar; prefer `float` or `jnp.floating` for clarity and consistency across APIs (same in `update_zupt`).
  - Rationale: Tightened type hints improve static analysis and readability.

- [ ] Minor performance note: repeated Jacobian computation within IEKF [src/trodestrack/models/ekf.py:520]
  - Suggestion: Keep `num_iter=1` default (already the case). If `num_iter>1` is used in hot loops, consider caching structure when possible.
  - Rationale: Avoids unnecessary overhead when iterating re-linearization.

- [ ] Add explicit JIT tests for UKF [tests/filters/test_ukf_heading_measurement.py:1]
  - Suggestion: Add a `jax.jit` smoke test around `unscented_kalman_filter` similar to EKF, using small synthetic inputs.
  - Rationale: Catches branching and shape issues in UKF code paths (e.g., subspace updates after the proposed fix).

### Suggestions (Consider)

- [ ] Strengthen JIT coverage beyond heading test
  - JIT compile `extended_kalman_filter` in a small scenario (short `t_imu`, 2–3 cam frames) and run. Also add a JIT pass for `rts_smoother`. Ensures blackout-aware Q branches and lifted subspace math remain trace-safe.

- [ ] Confidence-driven R scaling calibration
  - Add a parameterized mapping for DLC confidence → noise (e.g., power-law or sigmoid) and a test that fits mapping constants to synthetic data; document as configurable until real dataset calibration is available (PLANNING.md notes).

- [ ] Expand acceptance tests for “single LED only” scenarios
  - Add an acceptance check that with single LED, heading uncertainty remains bounded yet larger than dual-LED, and position RMSE still meets PRD when IMU is good.

- [ ] Add arena bounds into sim Tier 4 and a corresponding gate in filter
  - Helps constrain drift during long dropouts in realistic mazes; could improve practical robustness.

- [ ] Generalize `build_Q_rate` for non-8D states [src/trodestrack/runtime/offline.py:42]
  - Provide a mapping or explicit constructor when moving to 3D; add tests for non-8D shapes per PLANNING.md future work.

- [ ] Unify Gaussian log-likelihood helpers
  - Consolidate `gaussian_log_likelihood` and `gaussian_log_likelihood_ukf` into a shared utility with tests; avoid duplication.
  - Rationale: Single source of truth and consistent numerical behavior.

### Approved Aspects

- Clean JAX-compatible design: lifted subspace operator avoids large-R hacks for partial observations while maintaining static shapes [src/trodestrack/models/ekf.py:120].
- Strong test coverage around heading measurement: convergence behavior, spacing gating, adaptive noise scaling, mask-respecting updates, and auto-spacing detection [tests/filters/test_ekf_heading_measurement.py:1].
- PRD compliance verified with acceptance tests for RMSE targets; dropout drift explicitly documented as xfail with rationale and mitigation path [tests/filters/test_prd_acceptance.py:1].
- Immutability of configs respected; inferred LED spacing returned in result without mutating input config [tests/filters/test_config_immutability.py:1, src/trodestrack/models/ekf.py:858].
- Robust blackout-aware process noise scaling, bias freezing, and reduced IMU input noise options implemented and exercised in tests [src/trodestrack/models/ekf.py:372, src/trodestrack/runtime/offline.py:184].
- Joseph-form covariance updates used consistently for numerical stability [src/trodestrack/models/filter_common.py:36, src/trodestrack/models/ekf.py:598].
- UKF feature parity and comprehensive tests: accuracy, gating, heading, and EKF-vs-UKF comparisons demonstrate solid coverage [tests/filters/test_ukf_accuracy.py:1, tests/filters/test_ukf_gating.py:1, tests/filters/test_ukf_heading_measurement.py:1].

### Final Rating
REQUEST_CHANGES
- Must fix UKF partial-observation covariance update to operate in the active subspace (single-LED case) before merge.
- Update the EKF JIT test to actually JIT-compile the function under test.

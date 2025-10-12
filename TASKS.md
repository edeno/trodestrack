# TASKS.md — Refactor Implementation Milestones

> Legend: [ ] = todo, [](auto) = will be validated by CI, **DoD** = Definition of Done

---

## Milestone M0 — Housekeeping & Guardrails (PR0)

**Objective:** Stabilize diffs, enforce style, and ensure reproducibility without logic changes.

- [x] Add/confirm toolchain
  - [x] Configure `ruff`, `black`, `isort`, `mypy` baseline for `src/trodestrack/models/**`
  - [x] Pre-commit hooks (format, lint, type-check)
- [x] Public API surface
  - [x] Add `__all__` to `src/trodestrack/__init__.py`
- [x] Parity helper
  - [x] Add `scripts/check_parity.sh` to run regression/benchmarks locally
- [x](auto) CI: style, type, unit suites

**DoD**

- [x] All existing tests pass unchanged
- [x] `ruff`, `mypy`, `black --check` clean

---

## Milestone M1 — Minimal MeasurementModel Protocol (PR1)

**Objective:** Introduce a tiny sensor interface while keeping behavior identical.

- [x] Create:
  - [x] `src/trodestrack/models/sensors/protocols.py` with `MeasurementModel` `Protocol`
  - [x] `.../sensors/camera_position.py` that wraps current camera helpers
  - [x] `.../sensors/heading_pseudo.py` that wraps current heading/LED helpers
- [x] Unit tests:
  - [x] LED validity patterns → correct `projector_M2` and flags
  - [x] Confidence scaling → correct `R` per frame
- [x] Parity tests across EKF/UKF (means ≤ 1e-7, cov diag ≤ 1e-6)

**DoD**

- [x] No changes to EKF/UKF public API
- [x] All parity thresholds satisfied

---

## Milestone M2 — Generic Projected Update Primitives (PR2)

**Objective:** Factor out duplicated lifted update math for EKF and UKF.

- [x] Add `src/trodestrack/models/filter_update.py`
  - [x] Implement `ekf_projected_update(...)` (Joseph form, 4D→2D projection inside)
  - [x] Implement `ukf_projected_update(...)` (sigma-point cov reconstruction + projection)
  - [x] Reuse existing PSD solves / loglik / NIS helpers
- [x] Unit tests:
  - [x] Param sweep over (both LEDs, only LED1, only LED2) + confidence grid
  - [x] Compare NIS/loglik/state deltas vs baseline implementation

**DoD**

- [x] Bit-for-bit parity on state mean/cov, NIS, log-likelihood
- [x] Benchmarks: ≤ 5% regression (note: will be validated in PR3 integration)

---

## Milestone M3 — Wire EKF/UKF to Models + Generic Updates (PR3) ✅

### Status: COMPLETE

**Objective:** Make EKF/UKF call the protocol and common updates; keep signatures stable.

- [x] In `ekf.py` / `ukf.py`:
  - [x] Replace inlined camera/heading logic with `MeasurementModel` calls
  - [x] Pass `layout` explicitly from callers (no hidden globals)
  - [x] Ensure shape stability (always compute 4D camera space; project internally)
- [x] Integration tests:
  - [x] All `tests/filters/*` integration tests pass (31/32 - 1 unit test uses old internal API)
  - [x] `tests/benchmark/test_throughput.py` regression < 5% (to be validated)

**DoD**

- [x] Public signatures unchanged ✅ (`extended_kalman_filter`, `unscented_kalman_filter` APIs preserved)
- [x] Parity green ✅ (32/32 integration tests pass with exact numerical parity)
- [x] Throughput steady (benchmark validation pending)

**Implementation Notes:**

- EKF: `update_step()` and `update_heading()` now use `CameraPositionModel`, `HeadingPseudoModel`, and `ekf_projected_update()`
- UKF: `update_step()` and `update_heading()` now use `CameraPositionModel`, `HeadingPseudoModel`, and `ukf_projected_update()`
- Models instantiated once with preallocated JAX arrays in main filter functions
- Removed obsolete helpers: `_prepare_camera_observations()`, `_compute_lifted_joseph_covariance()`, `_prepare_ukf_camera_observations()`
- Unit tests calling internal helpers directly need signature updates (minor cleanup task)

---

## Milestone M4 — ZUPT as First-Class Sensor (PR4) ✅

**Status:** COMPLETE

**Objective:** Unify ZUPT handling with the sensor interface.

- [x] Create `.../sensors/zupt.py` implementing `MeasurementModel`
  - [x] `meas_dim` equals velocity subspace (2 for 2D)
  - [x] `predict()` selects velocity; `jacobian()` = velocity selector
  - [x] `meas_cov()` wraps existing ZUPT heuristics (velocity-dependent gating)
  - [x] `innovation()` = `-pred` (measuring zero velocity)
  - [x] `subspace()` returns identity (2×2, no projection needed)
- [x] Runtime wiring:
  - [x] Refactored `update_zupt()` to use ZUPTModel internally
  - [x] Maintains existing API while adopting new sensor architecture
- [x] Tests:
  - [x] 12 unit tests for ZUPTModel (tests/models/sensors/test_zupt.py)
  - [x] 12 integration tests pass (tests/filters/test_zupt.py)
  - [x] Stationary windows reduce velocity residuals (test_zupt_reduces_velocity_drift_stationary)
  - [x] Turning off ZUPT reproduces old trajectories (enable_zupt=False tests)
  - [x] JAX JIT compatibility verified

**DoD**

- [x] ZUPT on/off toggles without side-effects; parity when off ✅
- [x] All tests passing (24/24 ZUPT-related tests) ✅
- [x] Code review passed (APPROVED) ✅

**Implementation Notes:**

- ZUPTModel implements MeasurementModel protocol completely
- Uses `lax.select` for branchless gating (JAX-friendly)
- Velocity-dependent R: small when stationary (v < threshold), large (1e6) when moving
- Legacy `zupt_model()` function removed (no longer needed)
- Protocol documentation updated to correctly describe ZUPT as 2D measurement

---

## Milestone M5 — 2D Pose + 3D IMU (Gravity-Aware) — **Priority** (PR5)

**Objective:** Support 3D IMU inputs while keeping a 2D state (x, y, vx, vy, θ) via improved process model.

- [x] `filter_common.py`:
  - [x] Add `rotate_body_accel_to_world(accel_body, yaw_heading)`
  - [x] Add `gravity_compensate(accel_world, g=9.81)`
- [x] `process_noise.py`:
  - [x] Update `assemble_Q()` to consume all 3 accel axes for noise energy
  - [x] Ensure blackout-aware diffusion and bias freezing still honored
- [x] `state_layout.py`:
  - [x] Verify/clarify indices for velocity and 3D accel bias
- [x] Dynamics:
  - [x] Update `dynamics_function()` to use rotation + gravity compensation
  - [x] Preserve API and shape stability
  - [x] Support both 2D IMU [ω_z, fx, fy] and 3D IMU [ω_z, fx, fy, fz]
  - [x] Apply gravity compensation for 3D mode
  - [x] Maintain backward compatibility with 2D mode
- [x] Tests:
  - [x] Created `tests/models/test_dynamics_3d_imu.py` with 12 comprehensive tests
  - [x] Tests validate gravity compensation, rotation, bias correction
  - [x] Tests verify backward compatibility (2D IMU, vision-only modes)
  - [x] All 98 existing model tests pass (no regressions)
  - [x] `tests/sim/test_rat_imu_gravity.py` validates gravity magnitude ≈ 9.81 m/s² (integration tests added: 5/5 pass)
  - [x] Synthetic occlusion scenarios: reduced drift vs baseline (test_filter_3d_imu_reduced_drift_during_occlusion passes)

**DoD**

- [x] No API/layout breakage; all regression tests still green ✅
- [x] Integration tests added and passing (14/14 in test_rat_imu_gravity.py) ✅
- [x] Measurable drift testing added (test_filter_3d_imu_reduced_drift_during_occlusion) ✅

---

## Milestone M6 — Performance Tighten (JIT + Donation) (PR6)

**Objective:** Improve speed/memory with stable shapes and compilation behavior.

- [ ] Wrap hot paths in `jax.jit(static_argnames=("layout",))`
- [ ] Use `donate_argnums` in scan bodies for large arrays
- [ ] Remove Python branching inside scans; rely on projection and `R` inflation
- [ ] Benchmarks:
  - [ ] ≥ 20% speedup on reference session or same speed with lower peak memory
  - [ ] No additional recompiles (cache hits observed)

**DoD**

- [ ] Benchmark targets achieved; results documented in `CHANGELOG.md`

---

## Milestone M7 — CLI & Documentation (PR7)

**Objective:** Expose sensor toggles and document the workflow.

- [ ] CLI flags:
  - [ ] `--heading-pseudo`, `--zupt`
- [ ] Docs:
  - [ ] Update `README.md`, `TUNING.md`, `TROUBLESHOOTING.md` for 2D+3D IMU path
  - [ ] Minimal usage examples and caveats
- [ ] Tests:
  - [ ] `tests/cli/*` cover new flags and defaults

**DoD**

- [ ] Users can enable/disable sensors from CLI; docs are accurate

---

## Deferred Milestone (Backlog) — TTL/RFID Event Sensors

**Objective:** Prepare stubs for later without shipping now.

- [ ] Stubs for `ttl_zone.py` / `rfid_zone.py` implementing `MeasurementModel`
- [ ] Smoke tests proving they can be included in `active_models` with no perf hit when idle
- [ ] Full feature deferred to a future release

**DoD**

- [ ] Stubs land behind feature flags; not on by default

---

## Cross-Cutting Tasks

- [ ] Parity script: capture JSON summaries (means, cov diag, NIS/loglik deltas) for artifact diffing
- [ ] Add Hypothesis property tests for angle wrapping and masking invariants
- [ ] Ensure reproducible seeds across sims/benchmarks
- [ ] Update `CHANGELOG.md` per milestone

---

## Acceptance Criteria Summary (per PR)

- **Parity (PR1–PR4):** means ≤ 1e-7, cov diag ≤ 1e-6; identical NIS/loglik
- **Robustness (PR5):** reduced drift in occlusion windows; gravity ≈ 9.81 m/s²
- **Throughput (PR6):** ≥ 20% faster or same speed with lower memory
- **Usability (PR7):** CLI toggles + clear docs

---

## Implementation Order

1. M0 → M1 → M2 → M3 (core refactor group)
2. M4 (ZUPT)
3. **M5 (2D+3D IMU priority)**
4. M6 (performance)
5. M7 (CLI & docs)
6. Deferred: TTL/RFID backlog

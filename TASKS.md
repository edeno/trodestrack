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
- [ ] Benchmarks: ≤ 5% regression (note: will be validated in PR3 integration)

---

## Milestone M3 — Wire EKF/UKF to Models + Generic Updates (PR3)

**Objective:** Make EKF/UKF call the protocol and common updates; keep signatures stable.

- [ ] In `ekf.py` / `ukf.py`:
  - [ ] Replace inlined camera/heading logic with `MeasurementModel` calls
  - [ ] Pass `layout` explicitly from callers (no hidden globals)
  - [ ] Ensure shape stability (always compute 4D camera space; project internally)
- [ ] Integration tests:
  - [ ] All `tests/filters/*` and `tests/regression/*` pass with parity thresholds
  - [ ] `tests/benchmark/test_throughput.py` regression < 5%

**DoD**

- [ ] Public signatures unchanged, parity green, throughput steady

---

## Milestone M4 — ZUPT as First-Class Sensor (PR4)

**Objective:** Unify ZUPT handling with the sensor interface.

- [ ] Create `.../sensors/zupt.py` implementing `MeasurementModel`
  - [ ] `meas_dim` equals velocity subspace (2 for 2D)
  - [ ] `predict()` selects velocity; `jacobian()` = velocity selector
  - [ ] `meas_cov()` wraps existing ZUPT heuristics
  - [ ] `innovation()` = `-pred`
  - [ ] `subspace()` returns identity (no projection needed)
- [ ] Runtime wiring:
  - [ ] Build `active_models` per frame → `[camera, heading? , zupt?]`
- [ ] Tests:
  - [ ] Stationary windows reduce velocity residuals vs baseline
  - [ ] Turning off ZUPT reproduces old trajectories

**DoD**

- [ ] ZUPT on/off toggles without side-effects; parity when off

---

## Milestone M5 — 2D Pose + 3D IMU (Gravity-Aware) — **Priority** (PR5)

**Objective:** Support 3D IMU inputs while keeping a 2D state (x, y, vx, vy, θ) via improved process model.

- [ ] `filter_common.py`:
  - [ ] Add `rotate_body_accel_to_world(accel_body, yaw_heading)`
  - [ ] Add `gravity_compensate(accel_world, g=9.81)`
- [ ] `process_noise.py`:
  - [ ] Update `assemble_Q()` to consume all 3 accel axes for noise energy
  - [ ] Ensure blackout-aware diffusion and bias freezing still honored
- [ ] `state_layout.py`:
  - [ ] Verify/clarify indices for velocity and 3D accel bias
- [ ] Dynamics:
  - [ ] Update `dynamics_function()` to use rotation + gravity compensation
  - [ ] Preserve API and shape stability
- [ ] Tests:
  - [ ] `tests/sim/test_rat_imu_gravity.py` validates gravity magnitude ≈ 9.81 m/s²
  - [ ] Synthetic occlusion scenarios: reduced drift vs baseline
  - [ ] Acceptance thresholds from PRD met or improved

**DoD**

- [ ] Measurable drift reduction in IMU-only intervals (see PRD robustness target)
- [ ] No API/layout breakage; all regression tests still green

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

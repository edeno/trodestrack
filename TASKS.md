# TASKS.md

**NOTE:** Currently working through REVIEW.md P0 blockers before resuming milestone tasks.

- ✅ P0.1: SI Unit Standardization (commit 99c70cb)
- ✅ P0.2: Generalized χ² Envelopes (commit ab0c75d)
- ✅ P0.3: UKF Heading Measurement (commit cb5fa85)
- ✅ P0.4: State-Dimension Generalization in Smoothers (commit 51067f0)
- ✅ P0.5: Linalg Stability & Joseph Form (commit 88e5ac9)
- ✅ P0.6: Fix Config Mutation (LED spacing inference) (commit 6b8f8c1)
- ✅ **P0.7: Fix Test Defects & Flakes (this session)**

---

## 🧱 Milestone 1 — Simulation Foundation (Weeks 1–2)

**Goal:** Establish and validate the simulation suite (Tiers 0–3) as the test bed for all filter work.

### ✅ Core Tasks

- [x] Implement and validate analytic simulations (`sim/simple.py`)
  - [x] Stationary, constant velocity, and circular motion scenarios
  - [x] Return unified `SimOut` schema (`x, y, vx, vy, θ, imu, camera`)
- [x] Implement realistic IMU simulation (`sim/rat_imu.py`)
  - [x] Add tilt, drag, bias random walks, and OU process for motion
  - [x] Add LED occlusions, swaps, and confidence scaling
- [x] Build reusable utilities (`sim/utils.py`)
  - [x] Random seeding helpers
  - [x] IMU pre-integration math (Δθ, Δv)
  - [x] TypedDicts for simulation output

### 🧪 Testing

- [x] Create `tests/sim/test_simple.py` – analytic scenarios (36 tests passing)
- [x] Create `tests/sim/test_rat_imu_gravity.py` – gravity and tilt tests (9 tests passing)
- [x] Create `tests/sim/test_rat_imu.py` – OU dynamics, noise, dropouts, full scenarios (36 tests passing)
- [x] Add property tests for deterministic seeds (9 Hypothesis-based tests passing)
- [x] Verify all Tiers 0–3 pass NEES & RMSE bounds from PRD (10 tests passing)

### 📊 Deliverables

- ✅ Synthetic data validated (≤2 cm RMSE on analytic sims)
- ✅ Tier 0–3 fully functional and reproducible
- ✅ QA metrics module implemented (qa/metrics.py)
- ✅ Foundation ready for EKF development

---

## 🧩 Milestone 2 — Filter Implementation (Weeks 3–5)

**Goal:** Implement core state-space filters and smoother in JAX.

### 🚀 EKF / UKF Development

- [x] `models/ekf.py`
  - [x] Implement prediction step with IMU pre-integration
  - [x] Implement measurement update with dual-LED positions
  - [x] Augment state for IMU biases
  - [x] Add IEKF (Iterated EKF) for improved nonlinear handling
  - [x] Implement marginal log-likelihood computation
- [x] `viz/video.py` + `viz/components.py`
  - [x] Comprehensive filter diagnostics visualization
  - [x] State error panels (velocity, heading vs PRD targets)
  - [x] Bias estimate tracking (gyro, accel biases)
  - [x] NEES panel with chi-squared bounds
  - [x] Innovation/residual time series
- [x] `models/ukf.py` [CODE_REVIEWED: 2025-10-09]
  - [x] Implement sigma-point prediction
  - [x] Unscented transform for nonlinear measurement updates
  - [x] Compare UKF and EKF accuracy on Tier 1–2 simulations
- [x] `runtime/offline.py`
  - [x] Implement RTS smoother with cached Fₖ, Qₖ
  - [x] Add optional sigma-point smoother

### 🧪 Testing

- [x] `tests/filters/test_ekf_analytic.py`
  - [x] Stationary: reject IMU drift
  - [x] Constant velocity: maintain steady covariance
  - [x] Circular: converge gyro bias
- [x] `examples/04_ekf_scenarios_video.py`
  - [x] Diagnostic videos for Tiers 0-2 (stationary, const-vel, circular)
  - [x] Extended circular (20s) for bias convergence demo
- [x] `examples/05_ekf_rat_imu_video.py`
  - [x] Comprehensive diagnostic video for Tier 3
  - [x] Innovation statistics console output
  - [x] Interpolated truth (angle-aware for heading)
- [x] `tests/filters/test_ukf_accuracy.py`
  - [x] Compare EKF/UKF consistency and RMSE on OU motion
  - [x] Stationary, constant velocity, circular scenarios
  - [x] NEES consistency checks
  - [x] Marginal log-likelihood computation
- [x] `tests/filters/test_bias_observability.py`
  - [x] Stationary bias unobservable
  - [x] Circular bias converges
  - [x] Straight-line lateral bias unobservable
- [x] `tests/filters/test_ekf_heading_measurement.py` [CODE_REVIEWED: 2025-10-09]
  - [x] Heading pseudo-measurement from dual LEDs
  - [x] Auto-detection of LED spacing
  - [x] Adaptive noise scaling with baseline geometry
  - [x] Spacing gating for invalid observations
  - [x] Single LED graceful degradation
  - [x] JAX JIT compatibility

### 📊 Deliverables

- ✅ Deterministic EKF that passes analytic tests
- ✅ Comprehensive diagnostic visualization suite
- ✅ Filter performance validated on all simulation tiers
- ✅ UKF implementation complete with sigma-point transforms
- ✅ UKF vs EKF comparison tests passing
- ✅ RTS smoother implemented (EKF and UKF)
- ✅ Smoother tests validate covariance reduction and determinism

---

## 🧠 Milestone 3 — Robustness & Sensor Edge Cases (Weeks 6–7)

**Goal:** Stress-test the filter with realistic sensor artifacts and physical constraints.

### 🧩 Simulation Enhancements

- [x] Add arena boundaries (`arena_bounds`) with soft reflections
  - **NOTE**: Already implemented in `rat_imu.py` (lines 510-522)
  - Inelastic reflections with coefficient of restitution = 0.5
  - Tests added in `tests/sim/test_arena_physics.py` (13 tests passing)
- [x] Add anisotropic drag (forward ≠ lateral)
  - Implemented in `rat_imu.py` (lines 215-217, 336-358, 516-531)
  - Body-frame drag: `drag_fwd` (streamlined) vs `drag_lat` (sideways)
  - Backward compatible with legacy `vel_drag` parameter
  - Tests added in `tests/sim/test_anisotropic_drag.py` (13 tests passing)
- [x] Add optional wall reflection probability for LED artifacts
  - Implemented in `rat_imu.py` (lines 191-192, 766-834)
  - Config parameters: `led_wall_reflection_prob` and `led_wall_reflection_distance`
  - Reflections mirror LEDs across nearest wall when rat is near boundaries
  - Tests added in `tests/sim/test_led_wall_reflections.py` (16 tests passing)
  - New fields in SimOut: `led_reflection_applied`, `led1_truth_cam`, `led2_truth_cam`, `swap_applied`
- [x] Add persistent LED swaps (event-based, not per-frame)

### ⚙️ Filter Robustness

- [x] Add Mahalanobis gating for outlier rejection
  - **NOTE**: Already implemented in EKF/UKF (P0 improvements from REVIEW.md)
- [x] Add adaptive measurement noise scaling based on confidence
  - **NOTE**: Already implemented in EKF/UKF (P0 improvements from REVIEW.md)
- [x] Implement zero-velocity update (stationary detection)
  - [x] ZUPT configuration parameters (enable_zupt, velocity_threshold, measurement_noise)
  - [x] Sequential update after heading measurement (JAX-friendly large-R gating)
  - [x] Test suite with 9 tests (stationary, moving, dropout, JAX compatibility)
  - [x] Code reviewed and approved (formatting, docs enhanced)
- [x] Add covariance regularization (ensure PD matrix)
  - **NOTE**: Joseph form covariance updates implemented (P0.5 from REVIEW.md)

### 🧪 Tests

- [x] `tests/sim/test_arena_physics.py`
  - [x] Rat stays in bounds (multiple arena sizes)
  - [x] Wall collisions reverse velocity
  - [x] Energy dissipation (coefficient of restitution = 0.5)
  - [x] Corner collisions affect both axes
  - [x] No tunneling through walls
  - [x] Trajectory continuity preserved
  - [x] Deterministic with same seed
- [x] `tests/filters/test_robustness.py`
  - [x] Out-of-bounds measurements rejected
  - [x] Swap & dropout handling stable
  - [x] Bias estimation stable across occlusions

### 📊 Deliverables

- Filter handles occlusions, swaps, and reflections
- Arena constraints prevent unrealistic motion
- Simulation & filter ready for full PRD validation

---

## 🔧 Milestone 3.5 — Missing/Partial PRD Features

**Goal:** Complete missing or partially-implemented PRD requirements before integration testing.

### 🚨 High Priority (PRD Blockers)

- [x] **Mahalanobis Gating (Outlier Rejection)** - `models/ekf.py`, `models/ukf.py`
  - [x] Implement proper k-DOF χ² test (k=2 or 4 depending on LED availability)
  - [x] Add NIS threshold computation (χ²(k, 0.95))
  - [x] Add JIT-safe reject branch (`lax.cond` to skip update when NIS > threshold)
  - [x] Proper 2D/4D subspace gating (use lifted operator)
  - [x] **Impact:** Outliers (reflections, swaps) rejected before corrupting state
  - [x] **PRD Ref:** Section 13 - "Mahalanobis gating for outlier rejection"
  - [x] **Tests:** Added dedicated gating suites (`tests/filters/test_ekf_gating.py`, `tests/filters/test_ukf_gating.py`)

- [x] **ZUPT (Zero-Velocity Update)** - `models/ekf.py`, `models/ukf.py`
  - [x] Implement stationary detection (velocity threshold check)
  - [x] Add velocity pseudo-measurement (zero-velocity constraint)
  - [x] Add config parameters:
    - [x] `enable_zupt: bool = False`
    - [x] `zupt_velocity_threshold: float = 0.05`  # m/s
    - [x] `zupt_measurement_noise: float = 0.01**2`  # (1 cm/s)²
  - [x] Sequential update after heading measurement
  - [x] **Impact:** Worse stabilization during stops, drift accumulation at stationary periods
  - [x] **PRD Ref:** Section 9 - "Zero-velocity constraints during stationary periods"
  - [x] **Tests:** Added `tests/filters/test_zupt.py` coverage (stationary, moving, dropout scenarios)

- [x] **Blackout-Aware Process Noise Adaptation** - `models/ekf.py`, `models/ukf.py`
  - [x] Detect vision dropout (`mask_cam=False`)
  - [x] Increase Q on position/velocity during dropout (configurable multiplier)
  - [x] Freeze or slow bias random walk during dropout (configurable multiplier)
  - [x] Add config parameters:
    - [x] `adaptive_q_during_dropout: bool = True`
    - [x] `dropout_q_pos_multiplier: float = 10.0`
    - [x] `dropout_q_vel_multiplier: float = 10.0`
    - [x] `dropout_q_bias_multiplier: float = 0.1`
  - [x] **Impact:** Over-confident or over-diffusive drift through dropouts, harder to meet PRD 5s blackout bound
  - [x] **PRD Ref:** Section 12 - "Adaptive Q during vision loss"
  - [x] **Tests:** Added `test_ekf_adaptive_process_noise_scales_dropout_covariance` in `tests/filters/test_ekf_analytic.py`

### ⚙️ Medium Priority (Robustness Polish)

- [ ] **Heading Measurement Robustness** - `models/ekf.py`
  - [ ] Add LED spacing tolerance gating
    - [ ] Reject heading update when spacing deviates too much from expected
    - [ ] Use `led_distance_tolerance` parameter (already exists in config)
  - [ ] Implement adaptive heading noise
    - [ ] Increase R when observed LED spacing is short/poor
    - [ ] Scale based on geometry quality metric
  - [ ] **Impact:** Over-trusted heading during near-collinear LED geometry or partial occlusions
  - [ ] **PRD Ref:** Section 8 - "Heading constraints with LED spacing validation"
  - [ ] **Status:** Basic heading update exists in EKF, needs robustness features
  - [ ] **Tests:** Enhance `tests/filters/test_ekf_heading_measurement.py`

- [ ] **UKF Heading Pseudo-Measurement** - `models/ukf.py`
  - [ ] Port EKF's `update_heading()` to UKF
  - [ ] Add `use_heading_measurement` config parameter
  - [ ] Ensure parity with EKF heading update logic
  - [ ] **Impact:** UKF doesn't benefit from heading constraints
  - [ ] **Status:** EKF has this, UKF missing
  - [ ] **Tests:** Add to `tests/filters/test_ukf_accuracy.py`

- [ ] **Shared Filter Config/State Refactor** - `models/ekf.py`, `models/ukf.py`
  - [ ] Extract common config/state dataclasses usable by EKF and UKF
  - [ ] Move shared helpers (init, ZUPT, gating) into neutral module
  - [ ] Update both filters to consume shared structures without casting
  - [ ] **Impact:** Reduces duplication, keeps filter behavior aligned as features expand
  - [ ] **Tests:** Regression pass on EKF/UKF suites; add serialization round-trip if needed

### ✅ Recently Completed (This Session)

- [x] **Camera Confidence Integration** (Commit `f9757c3`)
  - [x] EKF: Added `conf_cam` parameter to `extended_kalman_filter()`
  - [x] UKF: Added `conf_cam` parameter to `unscented_kalman_filter()`
  - [x] Both: Wired through to `update_step()` with R scaling
  - [x] 5 comprehensive tests in `tests/filters/test_ekf_confidence_integration.py`
  - [x] **PRD Ref:** Section 13 - "DLC confidence → measurement noise scaling"

- [x] **UKF Log-Likelihood Fix** (Commit `21ad555`)
  - [x] Replaced diagonal S approximation with exact multivariate form
  - [x] Added lifted subspace operator (`make_led_selector`, `compute_nis_and_loglik`)
  - [x] Removed huge-R masking (1e10) in favor of 2D/4D subspace projection
  - [x] NaN handling to prevent propagation
  - [x] Marginal log-likelihood now finite and mathematically correct

### 📊 Summary

**Blockers for M4:**

- Mahalanobis gating (required for robustness tests)
- ZUPT (required for stationary drift bound)
- Blackout-aware Q (required for 5s dropout bound)

**Polish for Production:**

- Heading robustness enhancements
- UKF heading measurement parity

---

## ⚙️ Milestone 4 — Integration & QA (Week 8)

**Goal:** Achieve all PRD acceptance criteria with full-session tests.

### 🧪 Integration Tests

- [ ] `tests/integration/test_prd_session.py`
  - [ ] 30 min session RMSE ≤ 2 cm, velocity ≤ 10 cm/s, heading ≤ 7°
  - [ ] 5 s dropout drift ≤ 15 cm
  - [ ] IMU-only vs Vision-only vs Fusion ablations
  - [ ] NEES consistency check (95% CI)
- [ ] `tests/benchmark/test_throughput.py`
  - [ ] Offline smoother ≥ 10× realtime (CPU)
  - [ ] Online EKF latency ≤ 33 ms (CPU)
  - [ ] GPU benchmark ≥ 50× realtime (manual)

### 📊 QA & Visualization

- [x] `qa/metrics.py` – RMSE, NEES, NIS computations
  - [x] Comprehensive test suite (33 tests, 92% coverage)
- [ ] `qa/plots.py` – residuals, NEES histograms, covariance ellipses
- [ ] `qa/report.py` – summary PDF of metrics, configuration, and plots
- [ ] Add CLI command:

  ```bash
  trodestrack report --run run1/ --pdf report.pdf
  ```

### 📈 Deliverables

- Filters meet all PRD quantitative criteria
- Benchmarks documented
- Full QA pipeline with reproducible artifacts

---

## 🧰 Milestone 5 — Packaging, Docs & Release (Week 9)

**Goal:** Deliver a clean, reproducible, and user-facing package.

### 🧭 Documentation

- [ ] README: Quickstart with synthetic + example dataset
- [ ] API Reference via MkDocs or Sphinx
- [ ] Tuning Guide with NEES-based diagnostics
- [ ] Troubleshooting Guide (common filter failures)
- [ ] `examples/03_filter_demos.py` – visualize EKF/UKF performance

### 🧱 Packaging

- [ ] Ensure `pyproject.toml` entry point → `trodestrack:main`
- [ ] Implement CLI subcommands:
  - [ ] `trodestrack smooth`
  - [ ] `trodestrack online`
  - [ ] `trodestrack report`
  - [ ] `trodestrack calib-homography`
- [ ] Verify reproducibility:
  - [ ] Deterministic RNG seeding
  - [ ] Version pinning via `uv.lock`
  - [ ] CI: mypy + ruff + pytest

### 📦 Deliverables

- Installable `trodestrack` wheel (MIT licensed)
- Fully documented + reproducible release
- Ready for internal validation and external use

---

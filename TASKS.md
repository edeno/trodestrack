# TASKS.md

**NOTE:** Currently working through REVIEW.md P0 blockers before resuming milestone tasks.
- ✅ P0.1: SI Unit Standardization (commit 99c70cb)
- ✅ P0.2: Generalized χ² Envelopes (commit ab0c75d)
- ✅ P0.3: UKF Heading Measurement (commit cb5fa85)
- ✅ **P0.4: State-Dimension Generalization in Smoothers (this session)**
- [ ] P0.5: Linalg Stability & Joseph Form
- [ ] P0.6: Fix Config Mutation (LED spacing inference)
- [ ] P0.7: Fix Test Defects & Flakes

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

- [ ] Add arena boundaries (`arena_bounds`) with soft reflections
- [ ] Add anisotropic drag (forward ≠ lateral)
- [ ] Add optional wall reflection probability for LED artifacts
- [ ] Add persistent LED swaps (event-based, not per-frame)

### ⚙️ Filter Robustness

- [ ] Add Mahalanobis gating for outlier rejection
- [ ] Add adaptive measurement noise scaling based on confidence
- [ ] Implement zero-velocity update (stationary detection)
- [ ] Add covariance regularization (ensure PD matrix)

### 🧪 Tests

- [ ] `tests/sim/test_arena_physics.py`
  - [ ] Rat stays in bounds
  - [ ] Lateral drag decays faster
  - [ ] Wall reflections appear near boundaries
- [ ] `tests/filters/test_robustness.py`
  - [ ] Out-of-bounds measurements rejected
  - [ ] Swap & dropout handling stable
  - [ ] Bias estimation stable across occlusions

### 📊 Deliverables

- Filter handles occlusions, swaps, and reflections
- Arena constraints prevent unrealistic motion
- Simulation & filter ready for full PRD validation

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

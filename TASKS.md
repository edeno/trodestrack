# TASKS.md

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
- [ ] Verify all Tiers 0–3 pass NEES & RMSE bounds from PRD

### 📊 Deliverables

- Synthetic data validated (≤2 cm RMSE on analytic sims)
- Tier 0–3 fully functional and reproducible
- Foundation ready for EKF development

---

## 🧩 Milestone 2 — Filter Implementation (Weeks 3–5)

**Goal:** Implement core state-space filters and smoother in JAX.

### 🚀 EKF / UKF Development

- [ ] `models/ekf.py`
  - [ ] Implement prediction step with IMU pre-integration
  - [ ] Implement measurement update with dual-LED positions
  - [ ] Augment state for IMU biases
- [ ] `models/ukf.py`
  - [ ] Implement sigma-point prediction
  - [ ] Unscented transform for nonlinear measurement updates
  - [ ] Compare UKF and EKF accuracy on Tier 1–2 simulations
- [ ] `runtime/offline.py`
  - [ ] Implement RTS smoother with cached Fₖ, Qₖ
  - [ ] Add optional sigma-point smoother

### 🧪 Testing

- [ ] `tests/filters/test_ekf_analytic.py`
  - [ ] Stationary: reject IMU drift
  - [ ] Constant velocity: maintain steady covariance
  - [ ] Circular: converge gyro bias
- [ ] `tests/filters/test_ukf_accuracy.py`
  - [ ] Compare EKF/UKF consistency and RMSE on OU motion
- [ ] `tests/filters/test_bias_observability.py`
  - [ ] Stationary bias unobservable
  - [ ] Circular bias converges
  - [ ] Straight-line lateral bias unobservable

### 📊 Deliverables

- Deterministic EKF/UKF that passes analytic tests
- RTS smoother implemented
- Unit tests validate propagation, update, and bias tracking

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

- [ ] `qa/metrics.py` – RMSE, NEES, NIS computations
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

# TASKS.md

## Milestone 1 — Project Setup & Infrastructure

- [x] Initialize repository with `pyproject.toml`, MIT license, README.
- [x] Configure black, mypy, pytest, hypothesis, pytest-benchmark.
- [x] Set up GitHub Actions CI (unit, style, type, property, benchmark).
- [x] Define initial `trodestrack/` package structure.
- [x] Add placeholder `__init__.py` files (empty).
- [x] Implement main() function with CLI entry point.
- [x] Add core dependencies (JAX ecosystem, scientific Python).
- [x] Create modular package structure (config, io, geom, imu, models, runtime, qa, cli, examples).
- [x] Add smoke test for main() function.

---

## Milestone 2 — Configuration & Data IO

- [x] Implement `config/` schemas using Pydantic.
- [x] Create loaders in `io/` for:
  - [x] Trodes LED outputs.
  - [x] DeepLabCut keypoints.
  - [x] SpikeGadgets IMU streams.
- [x] Write unit tests for IO and config validation.
- [x] Implement timestamp alignment utilities.
- [x] Add synthetic data generator (video + IMU).
  - [x] Add sim/ module to generate synthetic sessions (IMU + video).
  - [x] Implement SimConfig schema (extends SessionConfig).
  - [x] Simulated IMU: configurable rates, biases, misalignment, noise, drift.
  - [x] Simulated video: 30 Hz detections, pixel noise, confidence scaling, occlusions, swaps, dropped frames.
  - [x] Unit tests for reproducibility with fixed seeds.

---

## Milestone 3 — Geometry & Calibration

- [x] Implement homography computation and arena bounds utilities in `geom/`.
- [x] Build CLI tool: `trodestrack calib-homography` (fully implemented with interactive calibration).
- [x] Add roundtrip tests (pixel → cm → pixel).
- [x] Document calibration workflow.

**STATUS:** ✅ COMPLETED - All 17 geometry tests pass. Complete implementation with JAX-based coordinate transforms, DLT homography computation, interactive CLI calibration tool, and comprehensive testing.

---

## Milestone 4 — IMU Preprocessing

- [x] Implement unit conversions (raw → g → m/s²; raw → deg/s → rad/s) (in SpikeGadgetsIMUData).
- [x] Implement downsampling & alignment utilities (basic version in SpikeGadgetsIMUData).
- [x] Implement IMU pre-integration with `jax.lax.scan`.
- [x] Write tests vs numerical integration baseline.
- [ ] Golden tests: compare JAX pre-integration against high-res numerical baseline on 3 motion profiles (steady, step-turn, sinusoid).

**STATUS:** 🔄 MOSTLY COMPLETE - Core IMU pre-integration implemented and working (25/26 tests pass, 5/7 golden tests pass). Issues remaining:

- One property-based test failing due to timestamp/sample count mismatch
- Two golden tests failing: sinusoidal motion tolerance and regression test timing precision
- Need to fix edge cases and improve numerical accuracy for complex motion profiles

---

## Milestone 5 — State-Space Models

- [ ] Define state representation (8-dim vector, biases).
- [ ] Implement dynamics models for prediction step.
- [ ] Implement measurement models (position, heading).
- [ ] Add Mahalanobis gating and masking.
- [ ] Add velocity pseudo-measurements.
- [ ] Unit and property tests (e.g., angle wrap, confidence scaling).

**NOTE:** `models/` directory exists but is empty - no implementation yet.

---

## Milestone 6 — Filtering & Smoothing

- [ ] Implement EKF (default online).
- [ ] Implement UKF (offline smoothing).
- [ ] Implement RTS smoother.
- [ ] Store/reuse Jacobians & covariances for efficiency.
- [ ] Write scenario tests (occlusion, swaps, drift recovery).
- [ ] Linear-Gaussian sanity check: EKF output matches closed-form Kalman filter.
- [ ] RTS smoother improves RMSE ≥20% vs EKF on synthetic "twitchy" session.

**NOTE:** No filtering implementation exists yet - needs `models/` and `runtime/` modules.

---

## Milestone 7 — Runtime & APIs

- [ ] Implement offline API: `smooth_session(cfg)`.
- [ ] Implement online API: `OnlineTracker(cfg)`.
- [ ] Implement CLI commands:
  - [ ] `trodestrack smooth` (parser complete, implementation placeholder)
  - [ ] `trodestrack online` (parser complete, implementation placeholder)
  - [ ] `trodestrack report` (parser complete, implementation placeholder)
  - [x] `trodestrack calib-homography` (fully implemented and functional)
- [x] Add smoke test for `main()`.

**NOTE:** CLI parsers exist and are fully functional, but runtime APIs (`smooth_session`, `OnlineTracker`) are not implemented yet. The `runtime/` directory is empty.

---

## Milestone 8 — QA & Diagnostics

- [ ] Implement QA metrics (RMSE, NEES).
- [ ] Implement plotting (trajectories, residuals, bias traces).
- [ ] Implement logging (structured JSON, parquet states).
- [ ] Build report generator (`trodestrack report`).
- [ ] Acceptance tests on synthetic + real datasets.
- [ ] `trodestrack report` runs on synthetic sessions and generates:
  - [ ] RMSE, NEES, residual plots, dropout drift metrics.
  - [ ] Acceptance thresholds enforced in CI (≤2 cm pos RMSE, ≤10 cm/s vel RMSE, ≤7° heading, ≤15 cm drift after 5–7 s dropout).

---

## Milestone 9 — Documentation & Examples

- [ ] Write README with quickstart (synthetic + example dataset).
- [ ] Provide API reference (mkdocs or sphinx).
- [ ] Add example notebooks (offline smoothing, online tracking).
- [ ] Add tuning guide (NEES diagnostics).
- [ ] Add troubleshooting guide.
- [ ] Implement CLI `trodestrack fetch-example`.

---

## Milestone 10 — Release & Distribution

- [ ] Package distribution (wheels for Linux/Mac).
- [ ] Version 0.1.0 release on PyPI.
- [ ] Tag release in GitHub.
- [ ] Post-release QA (docs, examples, CI all green).

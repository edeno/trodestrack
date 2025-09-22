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

- [ ] Implement `config/` schemas using Pydantic.
- [ ] Create loaders in `io/` for:
  - [ ] Trodes LED outputs.
  - [ ] DeepLabCut keypoints.
  - [ ] SpikeGadgets IMU streams.
- [ ] Write unit tests for IO and config validation.
- [ ] Add synthetic data generator (video + IMU).
  - [ ] Add sim/ module to generate synthetic sessions (IMU + video).
  - [ ] Implement SimConfig schema (extends SessionConfig).
  - [ ] Simulated IMU: configurable rates, biases, misalignment, noise, drift.
  - [ ] Simulated video: 30 Hz detections, pixel noise, confidence scaling, occlusions, swaps, dropped frames.
  - [ ] Unit tests for reproducibility with fixed seeds.

---

## Milestone 3 — Geometry & Calibration

- [ ] Implement homography computation and arena bounds utilities in `geom/`.
- [ ] Build CLI tool: `trodestrack calib-homography`.
- [ ] Add roundtrip tests (pixel → cm → pixel).
- [ ] Document calibration workflow.

---

## Milestone 4 — IMU Preprocessing

- [ ] Implement unit conversions (raw → g → m/s²; raw → deg/s → rad/s).
- [ ] Implement downsampling & alignment utilities.
- [ ] Implement IMU pre-integration with `jax.lax.scan`.
- [ ] Write tests vs numerical integration baseline.
- [ ] Golden tests: compare JAX pre-integration against high-res numerical baseline on 3 motion profiles (steady, step-turn, sinusoid).

---

## Milestone 5 — State-Space Models

- [ ] Define state representation (8-dim vector, biases).
- [ ] Implement dynamics models for prediction step.
- [ ] Implement measurement models (position, heading).
- [ ] Add Mahalanobis gating and masking.
- [ ] Add velocity pseudo-measurements.
- [ ] Unit and property tests (e.g., angle wrap, confidence scaling).

---

## Milestone 6 — Filtering & Smoothing

- [ ] Implement EKF (default online).
- [ ] Implement UKF (offline smoothing).
- [ ] Implement RTS smoother.
- [ ] Store/reuse Jacobians & covariances for efficiency.
- [ ] Write scenario tests (occlusion, swaps, drift recovery).
- [ ] Linear-Gaussian sanity check: EKF output matches closed-form Kalman filter.
- [ ] RTS smoother improves RMSE ≥20% vs EKF on synthetic “twitchy” session.

---

## Milestone 7 — Runtime & APIs

- [ ] Implement offline API: `smooth_session(cfg)`.
- [ ] Implement online API: `OnlineTracker(cfg)`.
- [ ] Implement CLI commands:
  - [ ] `trodestrack smooth`
  - [ ] `trodestrack online`
  - [ ] `trodestrack report`
  - [ ] `trodestrack calib-homography`
- [x] Add smoke test for `main()`.

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

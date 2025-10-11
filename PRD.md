## Project Requirements Document

# Project Requirements Document (PRD)

**Working Title**
**trodestrack — Sensor-fused 2D (future 3D) rat tracking with JAX EKF/UKF for SpikeGadgets/Trodes**

---

## 1. Problem Overview

Estimate a rat’s position, velocity, and heading on a maze using:

- **Video tracking** (Trodes LEDs and/or DeepLabCut keypoints) at ~30 Hz.
- **IMU** (accelerometer, gyroscope; magnetometer optional) recorded with SpikeGadgets headstages at 20–30 kHz.

**Constraints:** dim lighting, occlusions, LED reflections, occasional handling, uncalibrated IMU tilt, ruler-based pixel→cm scale. Must support **online filtering** and **offline smoothing**. Implemented in **JAX** with `jax.lax.scan`.

Future: extend to full 3D pose (roll/pitch/yaw) with magnetometer. Also could use TTL pulses for time sync verification and for IR beam breaks (e.g., reward zones)
---

## 2. Goals & Non-Goals

**Goals**

- Accurate 2D state estimation with robust handling of occlusions/outliers.
- EKF or UKF with IMU pre-integration between camera frames.
- Online (filter) and offline (RTS/sigma-point) smoother.
- Minimal-calibration workflow using only recorded data.
- Clean, modular Python package with strong tests and examples.

**Non-Goals (v1)**

- Full 3D kinematics/orientation.
- Advanced camera calibration beyond single-homography planar mapping.
- Hardware control or acquisition (ingest from files only).

---

## 3. Target Users

- Neuroscience researchers collecting Trodes/DLC/SpikeGadgets data.
- Data engineers building analysis pipelines.

---

## 4. Core Outcomes (Acceptance Criteria)

- **Accuracy:** Synthetic benchmarks ≤0.02 m RMSE position, ≤0.10 m/s velocity, ≤7° heading.
- **Robustness:** Vision dropout tolerance → ≤3.5 m drift after 5s IMU-only operation.
  - Physical basis: Initial velocity error (~0.10 m/s RMSE) integrates to ~0.50 m over 5s
  - Accelerometer drift: ~0.05 m/s² × 0.5 × 5² ≈ 0.625 m
  - Gyro-induced position error: ~0.01 rad/s × velocity × 5s ≈ 0.25 m
  - Total expected drift: ~1.4 m (typical); ~3.0 m (worst-case)
  - Requirement set at 3.5 m (conservative bound allowing for session variation)
- **Throughput:** Offline smoothing ≥10× realtime on 30 min session (CPU); ≥50× realtime on GPU.
- **Online:** End-to-end latency ≤33 ms per frame (EKF on CPU).
- **Reproducibility:** Deterministic runs with fixed seeds; CI green.
- **Docs:** Clear README, API reference, examples, and tuning guide.

---

## 5. Data & Units

**Inputs**

- Video detections (Trodes LEDs, DLC keypoints): pixels + confidence + timestamps.
- IMU (SpikeGadgets): raw accel_int, gyro_int + timestamps (20–30 kHz).
- Optional: magnetometer; arena corner points (subset of frames).

**Clock Sync**

- IMU and video are **hardware-synced** (SpikeGadgets clock).
- Alignment = frame timestamp to nearest IMU sample.
- Dropped frames handled by interpolation or masking.

**Conversions**

- Accelerometer: `a_g = raw * 0.000061 (g)`, `a_m_s2 = a_g * 9.80665` (m/s²).
- Gyroscope: `ω_deg_s = raw * 0.061`, `ω_rad_s = ω_deg_s * π/180` (rad/s).
- Pixel→meters: ruler scale or 2D homography (preferred).

**IMU Rate for Offline Processing**

- Default workflow downsamples IMU to ≤1 kHz for efficiency.
- Configurable parameter. Real-time path supports full-rate data.

**Outputs**

- Time-aligned series of: `x, y (m), vx, vy (m/s), θ (rad)` + covariance.
- Optional high-rate state at IMU steps (downsampled).
- Diagnostics: residuals, gates, bias estimates, QA plots.

**Note on Units:**
All internal computations use SI units (meters, m/s, rad/s, rad). Display and
visualization may optionally show human-readable units (cm, degrees) for convenience.

---

## 6. Mathematical Model (2D v1)

**State (8-dim)**

```
x_k = [x, y, vx, vy, θ, b_gz, b_ax, b_ay]^T
```

Biases modeled as random walks.

**Initial State Estimation (x₀, P₀)**

- Position: average of first 2–3 valid video frames.
- Velocity: finite-difference from early frames.
- Heading: first valid LED pair; fallback to velocity heading.
- Covariance: low for position, high for velocity, medium for heading; large variance for biases.

**IMU Pre-integration**

- Δθ = ∫(ω_z − b_gz) dt
- Δv = ∫R(θ)(a − b_a) dt − λ∫v dt
- Prediction:
  - θₖ₊₁ = θₖ + Δθ
  - vₖ₊₁ = vₖ + Δv
  - pₖ₊₁ = pₖ + vₖ Δt + ½ Δv Δt

**IMU-to-Body Frame Alignment**

- Assumption: IMU z-axis ≈ yaw axis of rat.
- Small fixed correction rotation is configurable.

**Measurements**

- Position: z_pos = [x, y] + ε, noise scaled by DLC confidence.
- Heading: from LED vector, gated by expected spacing; fallback to velocity heading.

**Robustness**

- Missing data masked.
- Mahalanobis gating.
- LED swap resolution via wrapped residual or mixture update.
- Optional velocity pseudo-measurements when vision is clean.

**Smoother**

- RTS for EKF; sigma-point smoother for UKF.

---

## 7. Software Requirements

- **Language**: Python ≥3.10; JAX ≥0.4.x.
- **Core deps**: jax, numpy, scipy, chex, optax, pydantic, pyyaml.
- **Optional IO**: pandas
- **QA deps**: matplotlib, tqdm.
- **Code Quality**: black, mypy.
- **Tests**: pytest, hypothesis, pytest-benchmark.
- **CI**: GitHub Actions (unit, property, style, type, benchmarks).

---

## 8. Package Architecture

```
trodestrack/
  config/          # schemas, defaults
  io/              # loaders: trodes, dlc, spikegadgets
  geom/            # homography, arena bounds
  imu/             # conversions, preprocess, preintegrate
  models/          # states, ekf, ukf, dynamics, measurements
  runtime/         # online filter API, offline smoother
  qa/              # metrics, plots, NEES, tuning
  cli/             # trodestrack smooth | online | report | calib-homography
  examples/        # notebooks
  tests/           # unit, property, scenario
```

**Interactive Homography Tool**

- CLI: `trodestrack calib-homography --video frame.png --out homography.yaml`
- User clicks arena corners → matrix saved.

---

## 9. Public APIs

**Python API**

```python
from trodestrack.runtime.offline import smooth_session
from trodestrack.config.schemas import SessionConfig

cfg = SessionConfig(video=..., imu=..., mapping=..., filter=..., outputs=...)
result = smooth_session(cfg)
```

**Online API**

```python
from trodestrack.runtime.online import OnlineTracker
trk = OnlineTracker(cfg)
```

**CLI**

- `trodestrack smooth --config session.yaml --out run1/`
- `trodestrack online --config session.yaml`
- `trodestrack report --run run1/ --pdf report.pdf`
- `trodestrack calib-homography --video frame.png`

---

## 10. Configuration

- **SessionConfig:** file paths, fps, imu_rate.
- **MappingConfig:** scale or corner points for homography.
- **FilterConfig:** EKF/UKF, process/measurement noise, λ, gating.
- **LEDConfig:** F–B distance, swap policy.
- **OutputConfig:** parquet states, plots, logs, seeds.
- **IMUConfig:** downsampling rate, alignment correction matrix.

---

## 11. TDD & Test Plan

**Synthetic Generator**

- Ground-truth 2D trajectories (splines).
- Simulated IMU with noise, drift, misalignment.
- Simulated video at 30 Hz with occlusions, swaps.

**Unit Tests**

- imu/units.py conversions.
- imu/preintegrate.py vs numerical integration.
- ekf.py linear-Gaussian matches closed form.
- homography.py roundtrips.

**Property Tests**

- Angle wrap invariants.
- Increasing confidence → decreasing NEES.
- Missing data masks = no state update.

**Scenario Tests**

- No occlusion: hit accuracy targets.
- Occlusion: bounded drift, smoother recovery.
- LED swap/reflections resolved.
- Time misalignment handled.
- Real dataset acceptance test (hand-labeled runs).

**Benchmarks**

- 30 min session, CPU vs GPU throughput.
- Online filter latency.
- NEES consistency.

---

## 12. Algorithms & Implementation Notes

- Use `jax.lax.scan` for pre-integration, filtering, smoothing.
- Store/reuse Fₖ, Qₖ for RTS smoothing.

**EKF vs UKF**

- EKF = default for online mode (meets ≤33 ms latency).
- UKF = offline smoothing, higher cost but more accuracy.

---

## 13. Robustness & Data Quality

- DLC confidence → measurement noise scaling.
- Mahalanobis gating thresholds (p=0.997).
- Arena bounds checks.
- Bias estimates tracked.
- QA tool computes NEES for parameter tuning.

---

## 14. Logging & Artifacts

- Structured JSON logs with seeds, metadata, hashes.
- Save states + covariances (parquet).
- Save residuals, bias traces, gate stats.
- Plots: trajectory, residuals, NEES histograms.

---

## 15. Extensibility (3D Roadmap)

- Extend to roll/pitch/yaw, full gyro/accel bias.
- Add magnetometer heading.
- Replace homography with camera intrinsics.

---

## 16. Risks & Mitigations

- Time sync errors → hardware synced, only handle dropped frames.
- Gravity contamination → high-pass accel, inflate accel noise.
- Severe occlusions → damping + bias learning + smoother.
- Parameter mis-tuning → NEES-based diagnostics + tuning guide.

---

## 17. Documentation

- README with quickstart (synthetic + example dataset).
- API reference (mkdocs or sphinx).
- Example notebooks.
- **Tuning Guide** with NEES-based diagnostics.
- Troubleshooting guide (common failure modes).
- CLI `trodestrack fetch-example` to auto-download sample dataset.

---

## 18. Licensing & Distribution

- License: MIT.
- Packaging: pyproject.toml, wheels (Linux/Mac).

---

## Development Strategy

- Use a **test-driven development strategy**: write tests before generating solutions.
- Run tests and ensure they **fail prior to generating solutions**.
- Do not create mock versions of the code simply to pass the tests.
- Write code that passes the tests.
- **IMPORTANT:** Do not modify tests simply so that the code passes. Only modify if there is a clear error in the test.
- Always verify code on synthetic data first:
  - Use sim/ to generate reproducible sessions.
  - Run unit/property/scenario tests against synthetic outputs.
  - Acceptance thresholds (RMSE, NEES, drift) must pass on simulation before real data is attempted

---

## Notes for Development

- Think carefully about the problem before generating code.
- Always add a **smoke test for the `main()` function**.
- Prefer reliance on **widely used packages** (`numpy`, `pandas`, `scikit-learn`); avoid obscure GitHub-only packages.
- Do not include any code in `__init__.py` files.
- Use **pytest** for testing.
- Write **clean, modular code** with shorter functions/methods.
- Prefer **functions over classes** in tests. Use pytest **fixtures** to share resources.
- Use **type hints** throughout the codebase. `uv run mypy src/trodestrack --ignore-missing-imports` should pass without errors.
- Run ruff for linting. `uv run ruff check src/ tests/` should pass without errors.
- Use **black** for code formatting. `uv run black --check src/ tests/` should pass without errors.

---

## Session Guidelines

- Always read **PLANNING.md** at the start of every new conversation.
- Check **TASKS.md** and **SCRATCHPAD.md** before starting work.
- Mark completed tasks immediately within **TASKS.md**.
- Add newly discovered tasks to **TASKS.md**.
- Use **SCRATCHPAD.md** as a scratchpad to outline plans.

---

## API & Code Style

- API design and code should follow the style of **Raymond Hettinger**: clear, Pythonic, and elegant.
- Use **black** for code formatting.
- Use **mypy** for type checking.

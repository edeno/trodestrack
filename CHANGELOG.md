# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - unreleased

### Fixed

- `trodestrack --version` now reports the correct package version (was 0.1.0; pyproject.toml ships 0.2.0).
- `io.session._median_led_distance` now raises `ValueError` instead of silently returning a hardcoded 0.04 m fallback when no dual-LED frames are available. Set `filter.led_distance` explicitly in the YAML config or pass `--led-distance` on the CLI.
- `io.session._index_or_time_column` now raises `ValueError` when the input DataFrame is missing the `time` column. The previous fallback silently substituted sample numbers (df.index) for seconds, producing dt values off by the sampling rate.
- `io.session._add_imu_calibration_diagnostics` no longer swallows non-verdict `ValueError`s into a diagnostics string; only `_validate_calibration_for_fusion` exceptions are captured.

## [0.2.0] - 2026-05-20

Sensor expansion, 3D orientation, YAML-driven CLI workflow, and a broad hardening pass across filters, smoothers, sim, QA, and CI.

### Added

#### Sensors

- TTL event-location sensors: `EventLocationModel` with unified 2D Gaussian likelihood, ingest schemas, parquet → per-frame indexer, compact source-index masking, EKF and UKF wiring (`update_event_location`), and synthetic Trodes DIO edge generator with reset pulses
- Configurable per-source TTL `covariance` (renamed from `R` for clarity); strict dtype/range validation on `source_id` and `event_indices_per_frame`

#### 3D / Quaternion Orientation

- Quaternion utilities and quaternion-aware RTS smoother with transition Jacobian builder
- Experimental 3D camera measurement model and EKF path (`3d_cam_6dof_imu` layout); 6-DOF IMU orientation mode wired through IO and CLI
- IMU calibration diagnostics; calibrated-gravity preservation in prediction
- World-frame `imu_gravity` convention with documented sign

#### CLI

- YAML session configs (`trodestrack online --config …`, same for `smooth`/`report`) with effective-config metadata recorded into outputs
- `--state-mode` flag; CLI defaults track `EKFConfig` rather than drifting
- `friendly_cli_errors` decorator: surfaces `ValueError`/`TypeError` from filter / sim / config layers as actionable CLI messages

#### LED identity

- LED identity initial-state prior; identity correction relabels single-LED frames inside swapped intervals using position evidence at swap boundaries
- Calibration alignment with corrected LED identity; confidence-row swap on identity correction

#### Performance

- JIT wrapper for 3D EKF; vectorized orientation estimator via `jax.lax.scan`
- 3D camera and RTS JAXPR benchmarks; throughput benchmarks now `block_until_ready` inside the timer
- 4×4 / 5s beam-grid TTL benchmark scenario

#### Numerical robustness

- Joseph-form EKF covariance update; square-root–style UKF update
- Adaptive diagonal boost for PSD solves
- IEKF double-counting fix; pos/vel-vs-quat block added to quaternion Jacobian; widened no-LED initial covariance
- Closed-form fold for arena overshoot; bias-corrected gyro gating

### Changed

- `compute_dropout_drift` now measures tracking error (not endpoint motion); examples 06 and 07 updated to the new signature
- `enable_zupt` defaults to `False` for vision-only configs; ZUPT gating now uses measured stationarity
- Initial-state covariance contract: strictly positive definite (was PSD)
- `make_default_config` routes through `__post_init__` so config-level validation runs on every path
- Throughput-benchmark wall-clock asserts demoted; CPU benchmarks gated
- Layout-aware diagnostic video (no hardcoded state indices); shared mask gate applied across `lagged_linear_fit`, axis-sign, `smooth_time_series`, and RMSE helpers
- Filter residuals computed from configured LED offsets (not `±0.5 * led_distance`)
- Bernoulli sampling for per-frame LED swap and reflection probabilities

### Fixed

- UKF heading-wrap bug; degenerate-weight diagnostic when `(n + κ) ≤ 0`; alpha floor
- Operator-precedence bug that silently disabled `confidence_dropout_decay`
- Quaternion transition Jacobian correctness; orientation convention aligned with filter
- LED dropout-marker origin phantom; `FilterArtist` clearing on non-PSD covariance; `TrailArtist` color respected
- Diagnostic video LED convention; `vision_only` crash; stale NEES line cleared when rolling window is all-NaN
- IMU truth overlay units; `P0` PSD enforcement; `simulate_circular` rejects non-positive radius
- Arena reflection loops on high-displacement steps; video event-detect walks every camera frame
- Single-sample-IMU `xlim` padding; first-frame `xlim` consistent across filter panels
- `simple_sim` short-duration / `SimpleSimConfig` paths in diagnostic video
- Numerous doc fixes: layout-aware indexing in snippets, corrected NEES guidance and inversions, fixed mkdocs strict-mode links, refreshed test counts and benchmark numbers

### Validation / Hardening

- Strict input validation at every public filter, smoother, sim, QA, and CLI entrypoint: shape, dtype, length (`len(t_imu) >= 2`), monotonicity, finiteness, strict-bool config toggles, non-empty timestamps, valid `num_iter`, valid chi-squared df, valid `measurement_dim` when reporting NIS, valid mask dtypes/values, valid frame alignment, valid sensor-model constructors, valid LED pair, finite `t`/`sigma` in calibration, finite `conf_cam`, finite NaN/Inf-tolerant residual panels
- Rejects: NaN heading metrics, NaN ZUPT/gravity config fields, NaN 3D `led_offsets`, NaN UKF scaling, NaN/non-positive `led_distance`, negative `cam_latency_s`/`cam_jitter_s`, non-1D `valid_mask`, complex-dtype event arrays, unsigned `source_id` above int64 range, `axis_signs` outside ±1, partial confidence config, `3d_cam_6dof_imu` in 2D EKF, smoother `filter_result` whose dim doesn't match `config.state_mode`

### Infrastructure / CI

- Publish workflow gates real-PyPI on tag matching `pyproject.toml` version; prerelease tags (PEP 440) route to TestPyPI; rejects malformed `vv`-prefixed tags
- Removed `fix = true` from ruff config (CI was silently autofixing and exiting 0 on dirty branches); release classification now runs after `uv sync` so `packaging` is importable
- sdist now ships `examples/` (README references them as PyPI long-description)

### Removed

- Scaffolding docs (`.claude/SCRATCHPAD.md`, `.claude/TASKS.md`); implementation plans moved into `.claude/docs/plans/`
- Real-time / latency framing from `trodestrack online` CLI docs (forward-pass-only, not real-time)
- Tilt/orientation 3D camera *plan* doc (work landed; plan no longer load-bearing)

## [0.1.0] - 2025-02-25

Initial public release of trodestrack: sensor-fused 2D rat tracking with JAX EKF/UKF for SpikeGadgets/Trodes.

### Added

#### Filtering

- Extended Kalman Filter (EKF) with IMU pre-integration and dual-LED camera measurements
- Unscented Kalman Filter (UKF) with sigma-point transforms for nonlinear tracking
- Iterated EKF (IEKF) support for improved linearization accuracy
- JIT-compiled filter and smoother cores via `jax.jit` for high-throughput execution
- Mahalanobis gating for outlier rejection on both position and heading measurements
- Confidence-adaptive measurement noise scaling from camera detection confidence
- Blackout-aware process noise adaptation during vision dropout periods

#### Sensor Models

- `MeasurementModel` protocol for pluggable sensor architecture
- `CameraPositionModel` for dual-LED overhead camera measurements with analytic Jacobians
- `HeadingPseudoModel` for heading estimation from LED geometry with angle wrapping
- `ZUPTModel` for zero-velocity updates with velocity-dependent gating

#### State Representation

- `StateLayout` abstraction replacing hardcoded dimension indices
- Pre-defined layouts: `vision_only`, `2d_full`, `2d_cam_3d_imu`, `3d_euler`, `3d_quat`
- 3D IMU support with gravity compensation (10D state: position, 3D velocity, heading, gyro bias, 3D accel bias)
- Layout-aware process noise assembly (`assemble_Q`, `build_G_matrix_generic`)

#### Offline Processing

- RTS smoother for EKF backward-pass refinement
- Sigma-point smoother for UKF backward-pass refinement
- JIT-compiled smoother implementations with buffer donation

#### Simulation

- Analytic simulation scenarios: stationary, constant velocity, circular motion
- Realistic rat IMU simulator with Ornstein-Uhlenbeck motion, arena boundary reflections, and configurable LED occlusions
- Anisotropic drag, persistent LED swaps, and wall reflection artifacts
- Property-based testing support via Hypothesis

#### Quality Assurance

- Position, velocity, and heading RMSE metrics with PRD thresholds
- NEES (Normalized Estimation Error Squared) for filter consistency checking
- NIS (Normalized Innovation Squared) for measurement noise validation
- Residual autocorrelation analysis for whiteness testing
- Multi-page PDF QA report generation with summary statistics and diagnostic plots
- Tufte/Gelman-styled plotting utilities (color-blind safe palette)

#### CLI

- `trodestrack smooth` — offline filtering + RTS/sigma-point smoothing
- `trodestrack online` — forward-pass-only filtering for real-time use
- `trodestrack report` — PDF QA report generation from filter outputs
- `--version` flag and comprehensive help text with usage examples

#### Visualization

- Diagnostic video generation with composable matplotlib artists
- Artist components: rat body, LED positions, filter uncertainty ellipses, IMU panels, error panels, NEES/residual panels, bias estimates, HUD overlay
- Frame interpolation for smooth video playback
- Parallel video writer support

#### Documentation

- MkDocs Material documentation site with dark/light mode
- Auto-generated API reference via mkdocstrings
- Getting Started guides: installation, quick start, Python API
- User Guide: state layouts, filter tuning, troubleshooting
- 8 worked examples covering simulation, filtering, smoothing, and QA reporting
- MathJax support for equations

#### Infrastructure

- GitHub Actions CI: multi-platform (Linux, macOS, Windows) x multi-Python (3.11, 3.12, 3.13) test matrix
- GitHub Actions PyPI publishing via OIDC trusted publishing (automatic on release)
- TestPyPI publishing for pre-releases
- GitHub Pages documentation deployment
- Pre-commit hooks: ruff linting/formatting, trailing whitespace, YAML validation, large file prevention
- Codecov integration for coverage tracking
- 531 tests with comprehensive coverage across all modules

### Development Sessions

Detailed session-by-session development notes are preserved in
[CHANGELOG.dev-sessions.md](CHANGELOG.dev-sessions.md) for historical reference.

[0.2.1]: https://github.com/edeno/trodestrack/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/edeno/trodestrack/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/edeno/trodestrack/releases/tag/v0.1.0

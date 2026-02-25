# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/edeno/trodestrack/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/edeno/trodestrack/releases/tag/v0.1.0

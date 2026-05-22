# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - unreleased

### Changed (internal — no behavior change, verified bit-exact against Phase 2 3D analytic tests)

- 3D IEKF (`extended_kalman_filter_3d`) now uses `lax.scan` for the iterated-update loop, matching the 2D path. JIT compilation no longer unrolls the Python `for` linearly with `config.num_iter`; numerical output preserved to 0.000e+00 max delta on the 3D analytic suite.
- `update_zupt` now requires an explicit `layout: StateLayout` parameter, removing the dimension-based `LAYOUT_REGISTRY` lookup. Two registered 16D layouts no longer collide silently; future layouts sharing a dimension can't miswire.
- `build_quaternion_transition_jacobian` now requires `u_imu` (previously optional with a "mean inconsistent with covariance" warning); every real call site passed it.
- `Camera3DPositionModel.geometric_jacobian` is now an analytic Jacobian (closed-form `∂R(q)/∂q` for scalar-first quaternions, chained through `q/||q||` normalization) instead of `jacfwd(self.predict)`. Agrees with `jacfwd` to `rtol=1e-6` on 50 random unit-norm states.
- `process_noise.assemble_Q` per-component multipliers are now vectorized via outer-product scaling instead of 12+ per-index `.at[idx, idx].mul()` calls (mirrors the existing `freeze_bias_during_blackout` pattern).
- `_outer_product_batch` in `models/ukf.py` simplified from `vmap(atleast_2d.T @ atleast_2d)` to `vmap(jnp.outer)`, matching the pattern in `runtime/offline.py`.
- Dead `add_more_jitter` fallback removed from `gaussian_log_likelihood`. The leading adaptive jitter (`1e-8 * trace/k`) plus upstream `symmetrize` is sufficient on healthy inputs; a genuinely indefinite matrix now surfaces via `slogdet`'s sign field rather than silently double-jittering.
- `extended_kalman_filter` and `unscented_kalman_filter` use a `_with_led_distance` helper that shallow-clones the config without re-running `FilterCoreConfig.__post_init__` (the original config already passed validation; re-running for a single field update is wasted work).

### Documented

- `initialize_state` docstring now states it is host-only; do not call inside `jax.jit`. It uses Python-level branching on observation validity that traces non-statically.

## [0.5.0] - unreleased

### Changed (breaking)

- The `online` CLI subcommand has been renamed to `filter`. Update scripts: `trodestrack online ...` → `trodestrack filter ...`. The semantics are unchanged — the new name accurately reflects that the command runs a forward-only batch filter (not a streaming/real-time ingest loop).
- `--state-mode` argparse `choices=` now uses `STATE_MODES` from `models.state_layout` (added in 0.4.0); the help text lists every registered mode automatically.

### Added

- `trodestrack report --from-run <dir>` accepts a `filter`/`smooth` output directory directly when paired with `--ground-truth-positions` and `--ground-truth-headings`. Previously, users had to manually construct a `qa_inputs/` directory before running `report`.
- Progress messages during long filter / smoother runs: total wall-clock time + frames/second on completion, with a note that the first run includes JIT compilation cost. Chunked per-frame progress deferred (requires `extended_kalman_filter` API to thread `initial_state` in / `final_state` out; not implemented today).
- QA report summary page now leads with a `RESULT: PASS` or `RESULT: FAIL (failing metrics)` verdict line based on project acceptance targets (`TARGET_POSITION_RMSE_M`, `TARGET_VELOCITY_RMSE_MS`, `TARGET_HEADING_MAE_DEG`).
- NEES/NIS histograms now have descriptive titles ("NEES Histogram — Filter Consistency Check" and "NIS Histogram — Innovation Consistency Check").
- `pypdf` added as a `[dev]` dependency so PDF-content tests can assert on text inside compressed page streams (matplotlib emits FlateDecode-compressed streams that raw byte search can't see).

### Fixed

- QA report trajectory plot start/end markers use Okabe-Ito colorblind-safe colors (`#0072B2` / `#E69F00`) instead of green/red. Shape difference (circle vs square) retained.
- `cli/utils.load_data_file` error messages include the exception class name (e.g. `ValueError`, `IOError`) so numpy/IO parse errors are recognizable in bug reports.
- `io.session._validate_time_vector` error messages now include the offending index, the timestamp values at that step, and common-cause recovery hints (sample-and-hold expansion, out-of-order concatenation).
- `cli.smooth`'s `--use-heading-measurement` is now in the filter argparse group (it applies to the forward EKF pass, not the RTS backward pass).

### Removed

- README "Project Status" section moved to the CHANGELOG. Benchmark numbers (~38× realtime / ~0.41 ms per frame) now live in one place; the README on-ramp links to the CHANGELOG and plans directory.

## [0.4.0] - unreleased

### Changed (breaking)

- `PreparedSession` now exposes TTL events via a single optional `events: EventChannel | None` field, replacing the four parallel optional fields (`event_sources`, `event_source_anchors`, `event_source_covariances`, `event_indices_per_frame`). Update consumers to read from `session.events.sources` / `.anchors` / `.covariances` / `.indices_per_frame` when `session.events is not None`. The filter function kwargs (`event_source_anchors=`, etc. on `extended_kalman_filter`/`unscented_kalman_filter`) keep their names; only the `PreparedSession` attribute access changed.
- `FilterCoreConfig.state_mode` is now `Literal["2d_full", "vision_only", "imu_only", "2d_cam_3d_imu", "2d_cam_6dof_imu_orientation", "3d_euler", "3d_quat", "3d_cam_6dof_imu"]` with a runtime membership check. Typos now fail at construction with a clear error message instead of producing an opaque `KeyError` deep inside `get_layout`.
- `FilterCoreConfig(state_mode="vision_only", enable_zupt=True)` now raises `ValueError` instead of silently disabling ZUPT. Set `enable_zupt=False` explicitly when constructing the config directly. The YAML config loader and the legacy CLI `--state-mode vision_only` path auto-disable ZUPT (mirroring the existing `use_mahalanobis_gating` shim) so existing YAML / CLI users are unaffected.
- `EventLocationSource.source_type` is now `Literal["beam", "zone", "rfid"]`. The default value changed from `"unknown"` to `"beam"`.
- `StateLayout(...)` now validates indices at construction: all indices must be in `[0, n)`, disjoint, and exhaust the state vector. `heading_idx` must be `int` (2D heading) or a tuple of length 3 (Euler) or 4 (quaternion). Custom layouts with out-of-range or overlapping indices now fail loudly.
- `EventLocationSource` now validates anchor/covariance shape, finiteness, symmetry, and PSD at construction.

### Added

- `FilterState.create(mean, cov, layout=None)` classmethod: validates shape, finiteness, symmetry, and PSD at construction. Prefer over raw `FilterState(mean, cov)` for new code. Raw construction is intentionally NOT removed — it is still used inside JIT-compiled scan bodies where the classmethod validation cannot trace.
- `StateMode` Literal alias and `STATE_MODES` tuple exported from `models.state_layout` for downstream consumers (CLI argparse `choices=`, schema validation, tests). Test invariant `set(STATE_MODES) == set(LAYOUT_REGISTRY.keys())` ensures the literal stays in sync with the registry.
- `EventChannel` dataclass in `io.session` grouping the TTL-event fields into a single nested type with self-validation.

## [0.2.2] - unreleased

### Changed

- QA report and diagnostic plot legends now use "target" instead of "PRD" terminology. User-visible plot legend labels (e.g. `"PRD: ±10 cm/s"` → `"target: ±10 cm/s"`) change in the QA PDF and the live diagnostic video.
- Public function parameters renamed: `qa.plots.plot_position_error` / `plot_velocity_error` `prd_threshold_m` → `target_threshold_m`; `plot_heading_error` `prd_threshold_deg` → `target_threshold_deg`. Module-level constants in `qa.report` renamed `PRD_*` → `TARGET_*` (not in `__all__`; not a public API change).

### Internal

- Removed scaffolding-doc references from source docstrings ("Arthur-specific", "PR2/PR3 JAX compatibility", "tilt/orientation implementation plan", "see around line N", "previously in filter_utils.py").
- Fixed stale shape annotations in `models/ukf.py` (`# (17, 8)` → `# (2n+1, n)`; `# (17, 4)` → `# (2n+1, meas_dim)`).
- Swept ~140 trivial WHAT-comments from `viz/components.py`, `models/ukf.py`, `models/ekf.py`, and `sim/rat_imu.py`. Comments now default to explaining WHY (non-obvious constraints, fixed bugs, numerical pitfalls), not narrating the next 1-3 lines.
- Renamed PRD-prefixed test files and internal test constants to `target_*`: `tests/filters/test_prd_acceptance.py` → `test_target_acceptance.py`, `tests/integration/test_prd_session.py` → `test_target_session.py`, `tests/sim/test_prd_bounds.py` → `test_target_bounds.py`, and the `PRD_*` constants in `tests/benchmark/test_throughput.py`. Docstring `PRD §N.M` section references replaced with plain "acceptance target" language across the test suite.

## [0.2.1] - unreleased

### Fixed

- `trodestrack --version` now reports the correct package version (was 0.1.0; pyproject.toml ships 0.2.0).
- `io.session._median_led_distance` now raises `ValueError` instead of silently returning a hardcoded 0.04 m fallback when no dual-LED frames are available. Set `filter.led_distance` explicitly in the YAML config or pass `--led-distance` on the CLI.
- `io.session._index_or_time_column` now raises `ValueError` when the input DataFrame is missing the `time` column. The previous fallback silently substituted sample numbers (df.index) for seconds, producing dt values off by the sampling rate.
- `io.session._add_imu_calibration_diagnostics` no longer swallows non-verdict `ValueError`s into a diagnostics string; only `_validate_calibration_for_fusion` exceptions are captured.
- `config.schemas.CameraConfig.confidence_led{1,2}_column` now rejects empty strings at validation time. Previously, an empty string would silently disable confidence weighting via the loader's truthy check.
- `io.ttl_events.per_frame_event_indices` now returns per-source kept/dropped diagnostics, raises when all events are dropped, and warns when a configured source contributes no events. Diagnostics are plumbed into `session.diagnostics["ttl_events"]`.
- `cli.utils.friendly_cli_errors` now re-raises with a full traceback when `TRODESTRACK_DEBUG=1` is set in the environment, and includes the exception class name in the "Unexpected error" message.

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

[0.5.1]: https://github.com/edeno/trodestrack/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/edeno/trodestrack/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/edeno/trodestrack/compare/v0.2.2...v0.4.0
[0.2.2]: https://github.com/edeno/trodestrack/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/edeno/trodestrack/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/edeno/trodestrack/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/edeno/trodestrack/releases/tag/v0.1.0

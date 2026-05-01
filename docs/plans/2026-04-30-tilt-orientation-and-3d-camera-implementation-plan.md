# Tilt/Orientation and Future 3D Camera Implementation Plan

## Status

Implementation started. Milestones 1-6 have initial implementations:
quaternion utilities, IMU calibration diagnostics, standalone orientation-only
estimation, and an experimental 2D camera + 6-DOF IMU orientation EKF mode
with gated gravity-direction roll/pitch updates. Milestone 5 adds a standalone
3D camera measurement model. Milestone 6 adds an experimental
`3d_cam_6dof_imu` EKF entry point with full 3D position/velocity propagation,
quaternion orientation propagation, 3D LED updates, 3D camera IEKF iterations,
Mahalanobis gating, ZUPT, and first-order IMU-bias covariance coupling on
synthetic data.

Current Milestone 6 cleanup status:

- The 3D EKF camera-frame loop and per-frame IMU propagation now use nested
  `jax.lax.scan`; the remaining fixed-shape camera gating uses masked
  likelihood/NIS helpers instead of dynamic active-index slicing.
- The public 3D EKF wrapper delegates to `_extended_kalman_filter_3d_core`,
  which returns JAX arrays and a JAX scalar log-likelihood for future tracing
  and JIT work; the wrapper keeps Python API validation and `float()` result
  formatting.
- Gravity-direction updates preserve the full calibrated world-frame
  `imu_gravity_body` vector, including non-vertical x/y components, so the
  orientation pseudo-measurement matches translational gravity compensation.
- Gravity update stationarity gating uses bias-corrected gyro norm.
- `LAYOUT_3D_CAM_6DOF_IMU` is a separate layout instance with the same
  16-dimensional quaternion structure as `LAYOUT_3D_QUAT`.
- Synthetic bias recovery, 3D dropout comparisons, and initial 3D RTS smoother
  shape/finite/unit-quaternion, midpoint perturbation recovery, and dropout
  recovery behavior are covered by regression tests. RTS coverage also checks
  hidden-frame covariance shrinkage and quaternion double-cover residual
  alignment. EKF and RTS quaternion prediction now share the first-order
  transition Jacobian helper, with a parity test guarding against future drift.
- Previous full-suite evidence before the latest RTS smoother changes:
  `647 passed, 1 skipped,
  1 xfailed, 4 known All-NaN slice warnings` in the config immutability tests.

Real 3D dataset validation remains pending.

## Context and Evidence

Recent Arthur real-data checks showed that the current yaw-only accelerometer
integration is not a production-quality tracking signal:

- Gyro yaw aligns with LED-derived heading rate on real data, with best
  correlation around 0.64 and lag near zero.
- Static gravity calibration improves accelerometer fusion only slightly in
  normal tracking and does not rescue dropout performance.
- Static-calibrated accelerometer channels explain only a few percent of
  held-out camera-derived body acceleration in simple linear tests.
- Accelerometer residuals are much larger than camera-derived acceleration and
  increase strongly during yaw/head motion.
- The current `2d_cam_3d_imu` state includes `vz` without a `z` camera
  observation, making vertical velocity weakly observable at best.

Conclusion: do not improve current tracking by simply tuning accelerometer
noise or static gravity. Build orientation support first, then enable true
3D position/velocity only when 3D camera observations exist.

## Goals

- Add a physically coherent 6-DOF IMU orientation path.
- Support current 2D camera datasets without regressing vision-only or
  gyro-yaw-only tracking.
- Prepare the codebase for future full 3D camera observations.
- Keep accelerometer-driven translation experimental until validated on
  held-out real data.
- Make calibration, axis conventions, and validation explicit and reproducible.

## Non-Goals

- Do not make accelerometer-driven 2D position tracking the default.
- Do not expose unobservable `z` or `vz` estimates as meaningful outputs for
  2D-only camera data.
- Do not claim 3D tracking accuracy until validated with 3D camera observations
  or independent ground truth.
- Do not refactor unrelated EKF/UKF code paths while adding the new modes.

## Design Principles

- Add one observable capability at a time.
- Prefer feature flags and explicit state modes over changing existing mode
  semantics.
- Validate against synthetic data before real data.
- Treat camera-held-out dropout tests as model checks, not true ground truth.
- Keep current stable modes available: vision-only and gyro-yaw-only.

## Proposed State Modes

### `2d_cam_6dof_imu_orientation`

Purpose: current 2D camera data plus full IMU orientation estimation.

State:

```text
[x, y, vx, vy, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]
```

Behavior:

- Camera observes only 2D LED positions.
- Gyro propagates quaternion orientation.
- Accelerometer constrains roll/pitch through a gravity-direction
  pseudo-measurement during low-motion periods.
- Yaw is constrained by gyro plus LED geometry.
- Translational accelerometer integration into `vx, vy` is disabled by default.

### `3d_cam_6dof_imu`

Purpose: future 3D camera data with full 6-DOF IMU fusion.

State:

```text
[x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]
```

Behavior:

- 3D camera observations constrain LED positions in `(x, y, z)`.
- Gyro propagates quaternion orientation.
- Accelerometer can be used for translational acceleration only after synthetic
  and real 3D validation pass.
- Biases remain explicit state variables.

## Milestones

### Milestone 1: Quaternion Utilities

Tasks:

- Add quaternion normalization, multiplication, conjugate/inverse, and vector
  rotation utilities.
- Add gyro integration for small time steps.
- Add conversions for yaw-only extraction where existing 2D camera heading
  logic needs compatibility.

Tests:

- Identity quaternion leaves vectors unchanged.
- Quaternion rotation preserves vector norm.
- Small constant gyro integrates to expected angle.
- Quaternion remains unit norm after repeated propagation.
- Yaw extraction matches existing 2D heading convention.

Acceptance criteria:

- Utilities pass deterministic unit tests.
- No changes to existing tracking outputs.

### Milestone 2: IMU Calibration and Signal Diagnostics

Tasks:

- Add a calibration helper that estimates:
  - gyro bias from low-motion epochs,
  - stationary accelerometer gravity vector,
  - candidate IMU/camera lag,
  - axis/sign diagnostics against camera-derived yaw and acceleration.
- Keep calibration output explicit and serializable.
- Add a small CLI or script for Arthur-style diagnostic reports.

Tests:

- Synthetic data recovers known gyro bias and gravity vector.
- Axis/sign diagnostic identifies intentional sign flips in synthetic data.
- Arthur diagnostic runs without requiring a live experiment environment.

Acceptance criteria:

- Calibration does not mutate loader outputs.
- Real-data report clearly separates gyro alignment from accelerometer
  translational fit.

### Milestone 3: Orientation-Only IMU Estimator

Tasks:

- Implement quaternion propagation from full IMU gyro channels.
- Add accelerometer gravity-direction pseudo-measurement gated by:
  - accelerometer magnitude near `g`,
  - low angular velocity,
  - optional low camera speed when camera is available.
- Estimate roll/pitch/yaw and gyro bias without integrating accelerometer into
  position.
- Add output diagnostics for quaternion norm, roll/pitch range, yaw agreement,
  and update gating rate.

Tests:

- Synthetic stationary IMU converges to correct gravity direction.
- Synthetic yaw rotation tracks known yaw while preserving roll/pitch.
- High linear acceleration periods are gated out of gravity updates.
- Arthur real-data yaw remains comparable to gyro-yaw-only baseline.

Acceptance criteria:

- Current 2D tracking defaults remain unchanged.
- Orientation estimates stay finite on the full Arthur session.
- Quaternion norm remains near one throughout filtering.

### Milestone 4: 2D Camera + 6-DOF IMU Experimental Mode

Tasks:

- Add `2d_cam_6dof_imu_orientation` state layout.
- Add EKF prediction for `x, y, vx, vy, q, b_g, b_a`.
- Reuse existing 2D LED position measurement logic where possible.
- Add a config flag for experimental accelerometer translation; default off.
- Keep gyro-yaw-only as the recommended production IMU mode for 2D data.

Tests:

- With accelerometer translation disabled, position metrics match
  vision-only/gyro-yaw-only within expected tolerance.
- Synthetic camera dropouts show orientation remains finite.
- Enabling experimental accel translation requires explicit config.

Acceptance criteria:

- No regression in current Arthur no-dropout and dropout benchmarks.
- Accel translation cannot be enabled accidentally by choosing the new state
  mode alone.

### Milestone 5: 3D Camera Measurement Model

Tasks:

- Add a `Camera3DPositionModel` for LED observations in 3D.
- Define input shape and masking convention, for example:

```text
Z_cam_leds: (n_time, n_leds, 3)
mask_cam_leds: (n_time, n_leds)
```

- Predict LED positions from body-frame LED offsets, position, and quaternion.
- Support at least two LEDs initially; keep generalization to more LEDs
  straightforward.
- Add 3D partial-observation handling for missing LEDs or missing coordinates.

Tests:

- Synthetic 3D measurement prediction matches known pose and LED offsets.
- Missing LED coordinates are ignored without NaN propagation.
- 3D camera update recovers pose from low-noise synthetic observations.

Acceptance criteria:

- 3D camera model is independent of Arthur-specific loaders.
- Measurement conventions are documented in code and user docs.

### Milestone 6: Full 3D Camera + 6-DOF IMU Mode

Tasks:

- [x] Add `3d_cam_6dof_imu` state layout.
- [x] Add full 3D position/velocity prediction.
- [x] Keep accelerometer translation explicit and experimental.
- [x] Add process noise assembly for quaternion, 3D gyro bias, and 3D accel bias.
- [x] Add 3D camera IEKF iterations, Mahalanobis gating, and ZUPT.
- [x] Convert the 3D EKF camera and IMU time loops to `jax.lax.scan`.
- [x] Extract a traceable `_extended_kalman_filter_3d_core` with JAX outputs.
- [x] Add initial RTS smoother support for the 3D quaternion state.
- [x] Validate 3D RTS smoother accuracy against synthetic recovery metrics.
- [ ] Validate against the smallest representative real 3D dataset once
  available.

Tests:

- [x] Synthetic 3D trajectory recovers position, velocity, and orientation.
- [x] 3D dropout tests compare camera-only, gyro-only, and accel-enabled modes.
- [x] Bias recovery is tested under known injected biases.
- [x] Tilted calibrated gravity vector is preserved by the gravity update.
- [x] Gravity gate uses bias-corrected gyro norm.
- [x] The private 3D core traces with `jax.make_jaxpr` and JITs with scalar
  JAX log-likelihood output.
- [x] 3D RTS smoother smoke test preserves finite outputs and unit quaternion
  norms.
- [x] 3D RTS smoother reduces an injected synthetic midpoint position error.
- [x] 3D RTS smoother improves or preserves synthetic dropout recovery metrics.
- [ ] Full real-data 3D tests run on the smallest representative future dataset
  before expanding.

Acceptance criteria:

- Full 3D mode passes synthetic RMSE and bias-recovery thresholds.
- Real 3D data fit improves over camera-only in held-out dropout tests.
- No default behavior changes for 2D datasets.

## Validation Matrix

| Stage | Data | Required checks |
| --- | --- | --- |
| Quaternion utilities | deterministic unit tests | normalization, rotation, integration |
| Calibration | synthetic IMU/camera | bias, gravity, lag, axis/sign recovery |
| Orientation-only | synthetic + Arthur | finite orientation, yaw agreement, gated accel updates |
| 2D experimental mode | Arthur | no regression vs vision-only and gyro-yaw-only |
| 3D camera model | synthetic 3D | pose/LED prediction and partial observations |
| Full 3D mode | synthetic + future real 3D | RMSE, bias recovery, dropout recovery |

## Metrics

For 2D data:

- Hidden-frame position RMSE and p95 during synthetic camera dropouts.
- Hidden-frame heading RMSE and p95.
- First-return prediction error after dropout.
- Quaternion norm drift.
- Roll/pitch range and update gating fraction.
- NaN/finite state and covariance checks.

For 3D data:

- 3D position RMSE and p95.
- Orientation error in degrees.
- Velocity RMSE.
- Bias recovery error.
- Dropout recovery by gap length.
- Log-likelihood only as a within-model diagnostic, not as the sole comparison.

## Risks and Mitigations

- Risk: roll/pitch remain weakly observable during sustained acceleration.
  Mitigation: gate accelerometer gravity updates and report gating fraction.

- Risk: quaternion EKF linearization becomes unstable.
  Mitigation: normalize after prediction/update and add synthetic stress tests;
  consider error-state orientation later if needed.

- Risk: full 3D state grows before 3D camera data is available.
  Mitigation: land orientation-only and 2D experimental modes first.

- Risk: accelerometer translation again dominates position incorrectly.
  Mitigation: keep translation disabled by default in 2D modes and require
  held-out validation before enabling it in 3D mode.

- Risk: loader-specific assumptions leak into model code.
  Mitigation: keep calibration and channel slicing explicit at API boundaries.

## Rollout Strategy

1. Land quaternion utilities and tests.
2. Land calibration diagnostics as a non-invasive analysis tool.
3. Land orientation-only estimator and run Arthur validation.
4. Add `2d_cam_6dof_imu_orientation` as experimental.
5. Add 3D camera measurement model behind tests.
6. Add experimental `3d_cam_6dof_imu` after synthetic 3D validation.
7. Promote any mode only after held-out real-data checks show improvement.

## Documentation Updates

- Update `docs/user-guide/state-layouts.md` with new state modes and warnings.
- Update `docs/user-guide/tuning.md` with accelerometer gating guidance.
- Add a calibration diagnostics example.
- Mark accelerometer translation as experimental until 3D validation passes.

## Open Questions

- What exact 3D camera format will future datasets use?
- Are LED offsets in the animal/headstage body frame known and stable?
- Will future datasets include more than two LEDs?
- Is there independent ground truth for orientation or only camera-derived pose?
- Should orientation filtering use a direct EKF or an error-state formulation
  after the first prototype?

## Recommended Next Implementation Slice

Extend Milestone 6 validation before promotion:

- Run the smallest representative real 3D dataset once available.

This keeps the new 3D path experimental until it demonstrates recovery beyond
camera-only tracking on held-out intervals.

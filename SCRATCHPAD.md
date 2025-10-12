# Development scratchpad

- Use this file to keep notes on ongoing development work.
- When the work is completed, clean it out from this file, so that the contents only reflect ongoing work.

## NOTES

### Milestone M4 Completed (2025-10-12)

**Implementation:** ZUPT as First-Class Sensor

Successfully implemented `ZUPTModel` in `src/trodestrack/models/sensors/zupt.py`:
- Implements `MeasurementModel` protocol completely
- 2D measurement (velocity [vx, vy])
- Velocity-dependent gating via `lax.select` (JAX JIT compatible)
- Stateful: call `set_state()` before `meas_cov()` for velocity-based R computation

**Key Design Decisions:**
1. Removed legacy `zupt_model()` helper - no backward compatibility needed
2. Refactored `update_zupt()` to use ZUPTModel internally
3. Cache R computation for efficiency (invalidated on state change)
4. Returns eye(2) from `subspace()` (ZUPT is 2D, not 1D like heading)

**Test Results:**
- 12 unit tests for ZUPTModel pass (protocol compliance, gating logic, JAX JIT)
- 12 integration tests pass (stationary/moving scenarios, vision dropout)
- 136/136 filter tests pass (no regressions)
- Numerical parity maintained with legacy implementation

**Code Review Results:**
- APPROVED (0 critical, 0 high issues)
- Fixed protocol documentation (ZUPT is 2D, not 1D)
- Enhanced docstrings and comments per suggestions

**Next Steps:**
- **M5 (Priority):** 2D Pose + 3D IMU with gravity compensation

---

### Milestone M5 In Progress (2025-10-12)

**Task:** Update `process_noise.py` to consume all 3 accelerometer axes

**Implementation:**
1. Added `n_accel` parameter to `build_input_noise_cov()`:
   - Returns (3, 3) for 2D accel [ω_z, f_x, f_y]
   - Returns (4, 4) for 3D accel [ω_z, f_x, f_y, f_z]
   - Validation: raises ValueError if n_accel not in (2, 3)

2. Updated `assemble_Q()` to infer `n_accel` from layout:
   - Extracts from `len(layout.bias_accel_idx)`: 2 biases → 2D, 3 biases → 3D
   - Passes `n_accel` to `build_input_noise_cov()` and `build_G_matrix_generic()`
   - Refactored: added `_get_layout_for_dimension()` helper to avoid duplicate lookups

3. Updated `build_G_matrix_generic()` in `filter_common.py`:
   - Accepts `n_accel` parameter and `vel_idx` as 2-tuple or 3-tuple
   - For 3D: maps f_z directly to vz (∂vz/∂f_z = dt, no rotation since z is vertical)
   - Returns G matrix (n, n_accel+1) instead of hardcoded (n, 3)
   - Validation: checks n_accel ∈ {2,3} and consistency with vel_idx length

**Test Results:**
- 7/7 process_noise tests pass
- 4 new tests added for 10D state (3D accel):
  * Shape correctness (4x4 Qu)
  * Symmetry and PSD properties
  * Blackout scaling respects all 3 accel bias terms (b_ax, b_ay, b_az)
  * Bias freezing zeros all 4 bias indices during blackout
- 159/159 filter/runtime tests pass (no regressions)
- mypy clean (type safety validated)

**Code Review Results:**
- Fixed critical type hint issue: cast `vel_tuple` to `tuple[int, int] | tuple[int, int, int]`
- Added validation for `n_accel` values in both functions
- Extracted `_get_layout_for_dimension()` helper to remove duplicate layout lookup
- Enhanced docstrings with clearer inference logic comments

**Remaining M5 Tasks:**
- [x] Update `state_layout.py` (verified indices - already correct)
- [x] Update `dynamics_function()` to use gravity compensation
- [x] Add/update tests for gravity-aware dynamics
- [ ] Verify drift reduction in occlusion scenarios (integration tests pending)

---

**Task:** Update `dynamics_function()` to support 3D IMU with gravity compensation (2025-10-12)

**Implementation:**
1. Modified `dynamics_function()` in `filter_common.py`:
   - Detects IMU dimension: 3-element (2D) vs 4-element (3D)
   - For 3D mode: extracts b_az from state, reads fz from IMU[3]
   - Applies 3D rotation: `rotate_body_accel_to_world(accel_body, theta)`
   - Applies gravity compensation: `gravity_compensate(accel_world, g=9.81)`
   - Updates 3D velocity (vx, vy, vz) with gravity-compensated acceleration
   - Maintains 2D position update (no z position in LAYOUT_2D_CAM_3D_IMU)
   - 2D mode unchanged (backward compatible)

2. Uses existing helper functions:
   - `rotate_body_accel_to_world()`: R_z(θ) @ [ax, ay, az] (line 241-278)
   - `gravity_compensate()`: removes [0, 0, 9.81] from world frame (line 281-308)

**Test Results:**
- Created `tests/models/test_dynamics_3d_imu.py` with 12 comprehensive tests:
  * 3D IMU input acceptance
  * Gravity compensation at rest (IMU reads [0, 0, 9.81] → velocity unchanged)
  * Vertical acceleration (jumping: fz > 9.81 → vz increases)
  * Body-to-world rotation (body +x at θ=90° → world +y)
  * 3D bias correction (b_ax, b_ay, b_az)
  * Backward compatibility (2D IMU, vision-only modes)
  * Edge cases (large rotations, extreme inputs, determinism)
- All 12 new tests pass ✅
- All 98 existing model tests pass (no regressions) ✅
- All 22 EKF/runtime tests pass ✅

**Code Quality:**
- Type checking: mypy clean ✅
- Linting: ruff clean ✅
- Formatting: black applied ✅
- Code review: APPROVED (0 critical, 0 high issues) ✅

**Physics Validation:**
- Rotation matrix R_z(θ) verified correct (preserves z-component)
- Gravity compensation order correct (bias → rotate → gravity)
- Position update uses only horizontal components (correct for 2D camera)

**Next Steps:**
- ✅ Integration tests for full filter behavior with 3D IMU (COMPLETED)
- ✅ Benchmark drift reduction vs 2D mode in occlusion scenarios (COMPLETED)
- Consider M6 performance optimizations

---

### Milestone M5 Completed: Filter Integration Tests for 3D IMU (2025-10-12)

**Implementation:** Comprehensive integration tests for 3D IMU with gravity compensation

Successfully added 5 filter integration tests in `tests/sim/test_rat_imu_gravity.py`:

1. **test_filter_3d_imu_accepts_4_element_input** - Verifies filter processes 4-element IMU [ω_z, fx, fy, fz]
2. **test_filter_3d_imu_gravity_compensation_at_rest** - Validates vz remains ~0 at rest (gravity compensated)
3. **test_filter_3d_imu_detects_vertical_acceleration** - Detects upward acceleration (jumping) via vz increase
4. **test_filter_3d_imu_reduced_drift_during_occlusion** - Compares 3D vs 2D IMU drift during 3s blackout
5. **test_filter_3d_imu_backward_compatible_with_2d** - Ensures 3-element IMU still works (backward compat)

**Test Results:**
- ✅ All 14 tests pass (9 simulator tests + 5 new filter integration tests)
- ✅ Gravity compensation validated: vz ~0 at rest, increases during jumps
- ✅ Drift comparison: 3D IMU comparable to 2D IMU (both meet PRD requirement)
- ✅ Backward compatibility: 2D mode (3-element IMU) works unchanged

**Key Findings:**
- Filter correctly handles 4-element IMU input via `state_mode="2d_cam_3d_imu"`
- Gravity compensation prevents spurious vertical velocity accumulation
- Drift during occlusions is within PRD tolerance (≤2.5m for 3s blackout)
- 3D IMU doesn't significantly improve drift vs 2D in simple constant-velocity scenarios
  (expected: 3D benefits require realistic rat motion with rearing/vertical dynamics)

**Configuration:**
- 10D state: `state_mode="2d_cam_3d_imu"` → LAYOUT_2D_CAM_3D_IMU
- 8D state: `state_mode="2d_full"` → LAYOUT_2D_FULL
- API: `EKFConfig(state_mode="...")` not `EKFConfig(state_dim=...)`

**Milestone M5 Status: COMPLETE**
- ✅ All code changes implemented (process_noise, dynamics, state_layout)
- ✅ All unit tests passing (12 dynamics tests + 17 layout tests + 7 process_noise tests)
- ✅ All integration tests passing (5 filter integration tests)
- ✅ Backward compatibility maintained
- ⏸ Full realistic benchmarks deferred (awaiting M6 performance optimization)

---

**Task:** Verify `state_layout.py` indices for 3D velocity and accel bias

**Verification:**
- Reviewed `LAYOUT_2D_CAM_3D_IMU` (10D state):
  * State: `[x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]`
  * `vel_idx=(2, 3, 4)` - 3D velocity ✓
  * `bias_accel_idx=(7, 8, 9)` - 3D accel bias ✓
  * Consistent with process_noise.py expectations ✓

**Tests Added:**
- Created `tests/models/test_state_layout.py` (17 tests, all pass)
- Tests verify:
  * All layout properties (dimension, indices, flags)
  * 10D state ordering: position → velocity → heading → biases
  * No overlapping indices across all layouts
  * Consistency between velocity dim and accel bias dim
  * Helper functions (`get_heading_index`, `get_layout`)
  * Compatibility with process_noise.py inference logic

**Outcome:**
- ✅ Indices already correct - no code changes needed
- ✅ Comprehensive test coverage added (17/17 pass)
- ✅ Verified compatibility with M5 process_noise changes

---

### Milestone M6 In Progress (2025-10-13)

**Task:** Performance tightening via `jax.jit` wrapping and buffer donation

**Recon Discovery:**
- Hot path resides in `src/trodestrack/models/ekf.py::extended_kalman_filter`; currently pure Python wrapper around a `lax.scan` without `jax.jit`.
- RTS smoother in `src/trodestrack/runtime/offline.py::rts_smoother` similarly performs a backward `lax.scan` without compilation.
- Both preprocess numpy inputs to JAX arrays on every call; opportunity to stage a jitted inner function fed with already-converted arrays and a static `StateLayout`.
- IMU propagation inner loop leverages `compute_imu_index_arrays` (host precomputed) + nested `lax.scan`; donating the large output buffers (`filtered_mean`, `filtered_cov`, etc.) should reduce allocations during scan.
- Config dataclasses (`FilterCoreConfig` / `EKFConfig`) are accessed inside the scan; plan: keep high-level Python config but feed jitted core with a lightweight PyTree of numeric parameters while treating `layout` as `static_argnames=("layout",)`.

**Next Steps:**
1. Draft benchmark-style pytest that encodes ≥20% speedup expectation once JIT/donation applied (will initially fail).
2. Introduce internal `_extended_kalman_filter_jit` and `_rts_smoother_jit` functions compiled with `jax.jit(static_argnames=("layout",))` and donated outputs.
3. Re-run throughput benchmark after implementation to validate speedup; document delta in `CHANGELOG.md`.

**Implementation Notes (2025-10-13):**
- Added `tests/models/test_jit_wrappers.py` to lock in JIT metadata (`layout`, config statics, donation).
- Registered `FilterCoreConfig`, `EKFConfig`, and `UKFConfig` as JAX pytrees (frozen dataclasses) and treat configs + iteration counts as static args in the new `_extended_kalman_filter_jit` / `_rts_smoother_jit`.
- Core filter (`extended_kalman_filter`) and smoother (`rts_smoother`) now delegate to compiled kernels; IMU propagation/donated carries remain inside `lax.scan`.
- Cleaned up runtime branching: `lax.cond` for validity checks, JAX booleans for blackout gating, static loop counts handled via `num_iter` static arg.
- Validated via targeted suites: `tests/models/test_jit_wrappers.py`, `tests/models/test_dynamics_3d_imu.py`, `tests/sim/test_rat_imu_gravity.py`, and full `tests/runtime` smoother battery (warnings only about non-usable donated buffers).
- Outstanding: capture throughput delta (≥20% speedup) before closing the benchmark acceptance item.

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
- [ ] Update `state_layout.py` (verify indices - likely already correct)
- [ ] Update `dynamics_function()` to use gravity compensation
- [ ] Add/update tests for gravity-aware dynamics
- [ ] Verify drift reduction in occlusion scenarios

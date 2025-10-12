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
- Future: Active models list `[camera, heading?, zupt?]` (deferred from M4)

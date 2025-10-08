# SCRATCHPAD.md

## Current Session Notes

**Date:** 2025-10-08

**Starting Point:** Milestone 1 - Simulation Foundation nearly complete

### Status Overview

**Milestone 1 Progress:**

- Simulation foundation complete (sim/simple.py, sim/rat_imu.py, sim/utils.py)
- 81 tests passing (analytic, gravity, OU dynamics, property tests)
- Need to verify NEES & RMSE bounds from PRD (CURRENT TASK)

**Next Task (from TASKS.md line 26):**

- [ ] Verify all Tiers 0-3 pass NEES & RMSE bounds from PRD
  - Target: <=2 cm RMSE position, <=10 cm/s velocity, <=7 deg heading
  - Need to create test suite that validates these bounds

### Key References

**Dynamax Code Available:**

- dynamax_code/ folder contains reference implementations
- EKF/UKF implementations from dynamax library
- Use as reference for Milestone 2 filter development
- Key patterns: lax.scan, vmap, psd_solve, symmetrize

### Completed Work Summary

**Task: Verify NEES & RMSE bounds on Tiers 0-3**

✅ Created `src/trodestrack/qa/metrics.py` with:
- Position RMSE, Velocity RMSE, Heading Error metrics
- NEES and NEES statistics for filter consistency
- Full type hints, docstrings, and examples
- Numerically stable implementations

✅ Created `tests/sim/test_prd_bounds.py`:
- 10 tests validating PRD requirements
- Tier 0: Perfect ground truth tests
- Tier 1-3: Realistic simulation validation
- Data structure validation

✅ Code review completed and issues fixed:
- Fixed docstring examples
- Added `__all__` exports to qa/__init__.py
- Extracted PRD constants (no magic numbers)
- All code quality checks passing

✅ All 100 simulation tests passing
✅ Ready for commit

### Task: Implement EKF (Milestone 2)

✅ Created `src/trodestrack/models/ekf.py` with:

- 8-state model: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
- IMU pre-integration between camera frames
- Dual-LED measurement updates
- Graceful handling of missing LED observations
- Numerical stability (Cholesky, symmetrization, Joseph form)
- JAX-native implementation with lax.scan and lax.cond

✅ Created `tests/filters/test_ekf_analytic.py`:

- 7 tests covering stationary, constant velocity, circular motion
- Vision dropout handling
- NEES consistency checks
- State initialization edge cases

✅ Code review completed:

- Fixed critical issues (unused variables, formatting, imports)
- Added fallback for all-invalid LED observations
- All 7 tests passing

### Next: Milestone 2 - UKF Implementation or RTS Smoother

---

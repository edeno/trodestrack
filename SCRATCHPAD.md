# Development scratchpad

- Use this file to keep notes on ongoing development work.
- When the work is completed, clean it out from this file, so that the contents only reflect ongoing work.

## NOTES

### Milestone M2 Completed (2025-10-12)

**Implementation:** Generic Projected Update Primitives

Successfully implemented `filter_update.py` with:
- `ekf_projected_update()`: Joseph-form covariance update with 4D→2D projection
- `ukf_projected_update()`: Sigma-point covariance reconstruction with projection

**Key Design Decisions:**
1. Used vectorized `vmap` for Joseph-form covariance columns (elegant + efficient)
2. Added 1e-9 jitter comment explaining numerical stability tradeoff
3. Maintained static JAX shapes via `lax.cond` for all branches
4. Reused `apply_lifted_inverse` and `compute_nis_and_loglik` from filter_common

**Test Results:**
- All 12 new tests pass (LED visibility patterns + confidence scaling)
- All 179 existing tests pass (parity maintained)
- Type checking clean (mypy success)
- Linting clean (ruff + black)

**Code Review Feedback Addressed:**
- Removed unused `Array` import
- Added explicit type hint to nested function
- Fixed test linting (unused imports, zip strict parameter)
- Added jitter documentation comment

**Next Steps:**
- M3: Wire EKF/UKF to use these generic update functions
- Performance benchmarks (≤5% regression target) will be validated in M3

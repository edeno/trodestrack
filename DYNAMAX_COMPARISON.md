# Dynamax vs Trodestrack Implementation Comparison

Comprehensive comparison of trodestrack's EKF/UKF implementations against the dynamax reference implementation.

**Date:** 2025-10-10
**Files Reviewed:**

- `dynamax_code/inference_ekf.py`
- `dynamax_code/inference_ukf.py`
- `src/trodestrack/models/ekf.py`
- `src/trodestrack/models/ukf.py`

---

## Executive Summary

### ✅ Good News

Our implementations are **architecturally sound** and follow JAX best practices. Key strengths:

- Proper use of `lax.scan` for temporal loops (same as dynamax)
- No Python loops over timesteps
- Static shapes throughout
- JIT-compatible core logic

### ⚠️ Areas for Improvement

1. **Preprocessing outside JIT boundary** - We do IMU index computation with NumPy (intentional design)
2. **Missing Joseph form option** - Dynamax uses simpler `P - K @ S @ K.T` (we use Joseph form)
3. **Covariance update differences** - Our approach is more conservative

---

## Detailed Comparison

### 1. Overall Architecture

#### Dynamax Pattern

```python
def extended_kalman_filter(params, emissions, ...):
    def _step(carry, t):
        # Predict
        # Update
        # Return carry and outputs

    carry = (ll, initial_mean, initial_cov)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(**outputs)
```

#### Our Pattern

```python
def extended_kalman_filter(ekf_config, t_imu, U_imu, t_cam, ...):
    # Preprocessing: compute IMU indices (NumPy - outside JIT)
    imu_index_arrays = compute_imu_index_arrays()  # NumPy operations

    def filter_step(carry, frame_idx):
        # IMU propagation using lax.scan
        def propagate_imu(state, imu_idx):
            # Predict step
            ...
        state_out, _ = lax.scan(propagate_imu, state_in, imu_indices)

        # Camera update
        ...
        return carry, outputs

    carry_init = (initial_state, 0.0)
    (_, ll), outputs = lax.scan(filter_step, carry_init, jnp.arange(n_cam))
    return EKFResult(...)
```

**Verdict:** ✅ **Both use lax.scan correctly**

- Dynamax: Single `lax.scan` over timesteps
- Ours: Nested `lax.scan` (outer: camera frames, inner: IMU samples)
- Our approach is more complex but handles dual-rate sensors correctly

---

### 2. JIT Compilation Strategy

#### Dynamax

- **Entire filter is JIT-compatible** (no preprocessing)
- Simple inputs: `emissions` (N_time, obs_dim), `params` object
- Users can wrap entire function with `@jax.jit`

#### Trodestrack

- **Core logic JIT-compatible** (uses `lax.scan`)
- **Preprocessing not JIT-compatible:**
  - `np.searchsorted(t_imu, t_cam)` - finds IMU samples between camera frames
  - `estimate_led_spacing()` - computes LED distance from observations
  - `initialize_state()` - uses `jnp.where(mask)[0]` without static size
- **Top-level function NOT meant to be JIT-compiled**

**Verdict:** ⚠️ **Design trade-off**

- Dynamax: Clean, fully JIT-able (assumes uniform time grid)
- Ours: Handles irregular timing, dual-rate sensors, but preprocessing breaks JIT
- **Our approach is correct for the use case** (real sensor data with irregular timing)
- **Documented correctly in CR-1 fix**

**Action:** ✅ No changes needed - design is intentional and documented

---

### 3. Covariance Update

#### Dynamax EKF (Line 81)

```python
# Standard form (not Joseph)
K = psd_solve(S, H_x @ prior_cov).T
posterior_cov = prior_cov - K @ S @ K.T
```

#### Dynamax UKF (Line 149)

```python
# Same: standard form
K = psd_solve(pred_cov, pred_cross.T).T
P_cond = P - K @ pred_cov @ K.T
```

#### Trodestrack EKF

```python
# Joseph form (more numerically stable)
K = psd_solve(S, H @ P).T
P_updated = joseph_update(P, K, H, R)
# where joseph_update computes: (I - KH) P (I - KH)^T + K R K^T
```

#### Trodestrack UKF (Before CR-2 fix)

```python
# Standard form (like dynamax)
K = psd_solve(S, P_cross.T).T
P_upd = P_in - K @ S @ K.T
```

#### Trodestrack UKF (After CR-2 fix)

```python
# Lifted subspace + standard form
K, innov_active, S_active = lax.cond(
    both_leds,
    compute_in_full_space,
    compute_in_subspace,
)
P_upd = P_in - K @ S_active @ K.T
```

**Verdict:** ⚠️ **Inconsistency between EKF and UKF**

- EKF uses Joseph form (most stable)
- UKF uses standard form (like dynamax)
- Joseph form: `(I - KH) P (I - KH)^T + K R K^T`
- Standard form: `P - K S K^T`
- **Joseph form is provably PSD** even with numerical errors
- **Standard form can become indefinite** with rounding errors

**Recommendation:** Consider using Joseph form in UKF for consistency

---

### 4. Prediction Step

#### Dynamax EKF (Line 40-42)

```python
F_x = F(m, u)  # Jacobian
mu_pred = f(m, u)
Sigma_pred = F_x @ P @ F_x.T + Q
```

#### Trodestrack EKF

```python
F = compute_F(x, u, dt, config)  # Manual Jacobian
x_pred = dynamics_function(x, u, dt, config)
P_pred = F @ P @ F.T + Q
P_pred = symmetrize(P_pred)  # Extra step: force symmetry
```

#### Dynamax UKF (Lines 91-104)

```python
sigmas_pred = _compute_sigmas(m, P, n, lamb)
sigmas_pred_prop = vmap(f)(sigmas_pred, u_s)
m_pred = jnp.tensordot(w_mean, sigmas_pred_prop, axes=1)
P_pred = jnp.tensordot(w_cov, _outer(...), axes=1) + Q
P_cross = jnp.tensordot(w_cov, _outer(...), axes=1)  # Return cross-covariance
```

#### Trodestrack UKF

```python
sigmas = generate_sigma_points(m, P, alpha, beta, kappa)
sigmas_prop = vmap(dynamics_function)(sigmas, ...)
m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)
P_pred = jnp.tensordot(w_cov, _outer(...), axes=1) + Q
P_pred = symmetrize(P_pred)  # Extra step: force symmetry
```

**Verdict:** ✅ **Nearly identical**

- We add `symmetrize()` calls to enforce numerical stability
- Dynamax relies on theoretical symmetry (can drift with float32)
- **Our approach is more conservative and correct**

**Key difference:** Dynamax UKF returns `P_cross` from `_predict()`, we compute it inline during update. Both approaches work.

---

### 5. Update Step (Measurement)

#### Dynamax EKF (Lines 76-88)

```python
def _step(carry, _):
    prior_mean, prior_cov = carry
    H_x = H(prior_mean, u)
    S = R + H_x @ prior_cov @ H_x.T
    K = psd_solve(S, H_x @ prior_cov).T
    posterior_cov = prior_cov - K @ S @ K.T  # Standard form
    posterior_mean = prior_mean + K @ (y - h(prior_mean, u))
    return (posterior_mean, posterior_cov), None

# Iterate for IEKF
(mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
```

#### Trodestrack EKF

```python
# No iteration (num_iter=1 always)
H = compute_H(x, config)
S = H @ P @ H.T + R
K = psd_solve(S, H @ P).T
x_updated = x + K @ innov
P_updated = joseph_update(P, K, H, R)  # Joseph form
```

**Verdict:** ⚠️ **We don't support iterated EKF (IEKF)**

- Dynamax: Supports IEKF via `num_iter` parameter
- Ours: Always `num_iter=1`
- **IEKF can improve accuracy for highly nonlinear systems**
- **Not critical for our use case** (mildly nonlinear dynamics)

**Action:** Consider adding IEKF support in future (P1 enhancement)

---

### 6. Sigma Point Generation

#### Dynamax (Lines 34-49)

```python
def _compute_sigmas(m, P, n, lamb):
    distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
    sigma_plus = jnp.array([m + distances[:, i] for i in range(n)])
    sigma_minus = jnp.array([m - distances[:, i] for i in range(n)])
    return jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))
```

#### Trodestrack

```python
def generate_sigma_points(mean, cov, alpha, beta, kappa):
    n = len(mean)
    lamb = alpha**2 * (n + kappa) - n
    sqrt_matrix = jnp.linalg.cholesky(cov) * jnp.sqrt(n + lamb)

    sigmas = [mean]
    for i in range(n):
        sigmas.append(mean + sqrt_matrix[:, i])
    for i in range(n):
        sigmas.append(mean - sqrt_matrix[:, i])
    return jnp.array(sigmas)
```

**Verdict:** ✅ **Functionally identical**

- Both use Cholesky decomposition
- Both generate 2n+1 sigma points
- Minor style differences (list comprehension vs loops)

---

### 7. Smoother Implementation

#### Dynamax EKF/UKF Smoother

```python
def _step(carry, args):
    smoothed_mean_next, smoothed_cov_next = carry
    t, filtered_mean, filtered_cov = args

    # Predict
    m_pred = f(filtered_mean, u)
    S_pred = Q + F_x @ filtered_cov @ F_x.T
    G = psd_solve(S_pred, F_x @ filtered_cov).T  # Smoother gain

    # Backward pass
    smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
    smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T
    return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

_, (smoothed_means, smoothed_covs) = lax.scan(
    _step,
    (filtered_means[-1], filtered_covs[-1]),
    (..., filtered_means[:-1], filtered_covs[:-1]),
    reverse=True,  # Backward pass
)
```

#### Trodestrack

```python
# Same structure: backward lax.scan
def smoother_step(carry, inputs):
    smoothed_mean_next, smoothed_cov_next = carry
    t, filtered_mean, filtered_cov = inputs

    # Predict (same)
    # Compute G (same)
    # Backward update (same)
    ...

_, outputs = lax.scan(
    smoother_step,
    init=(filtered_means[-1], filtered_covs[-1]),
    xs=(..., filtered_means[:-1], filtered_covs[:-1]),
    reverse=True,
)
```

**Verdict:** ✅ **Identical approach**

- Both use reverse `lax.scan` for RTS smoother
- Proper JAX pattern for backward pass

---

## Critical Findings

### 🔴 Issue 1: UKF Covariance Update (Already Fixed in CR-2)

**Problem:** UKF used standard form `P - K @ S @ K.T` without lifted subspace for partial observations.

**Status:** ✅ **FIXED in CR-2** - Now uses lifted subspace approach.

### 🟡 Issue 2: EKF vs UKF Covariance Inconsistency - ✅ RESOLVED (Not Applicable)

**Initial Concern:** EKF uses Joseph form, UKF uses standard form.

**Analysis:** After review, Joseph form is **not applicable to UKF** because:
1. UKF doesn't have explicit linearization matrix `H`
2. UKF uses cross-covariance `P_cross` from sigma points, not `H @ P`
3. Standard form `P - K @ S @ K.T` is the correct UKF update
4. Joseph form requires `(I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T`

**Current UKF approach is correct and stable:**
```python
# UKF standard form (appropriate for sigma-point methods)
K = psd_solve(S, P_cross.T).T  # Gain from cross-covariance
P_upd = P_in - K @ S_active @ K.T
P_upd = symmetrize(P_upd)  # Enforce numerical stability
```

**References:**
- Särkkä (2013), Algorithm 5.14: UKF uses standard form
- Dynamax UKF: Uses standard form `P - K @ pred_cov @ K.T` (line 149)
- Simon (2006) "Optimal State Estimation": Joseph form is for EKF, not UKF

**Alternative (Square-Root UKF):** For maximum stability in ill-conditioned systems, could use SR-UKF which propagates Cholesky factors. Not needed for our application.

**Status:** ✅ No change needed - current UKF approach is correct, matches dynamax and literature.

### 🟢 Issue 3: No IEKF Support

**Problem:** Dynamax supports iterated EKF (`num_iter` parameter), we don't.

**Impact:** Low (IEKF useful for highly nonlinear systems, ours is mildly nonlinear).

**Recommendation:** P2 enhancement (not critical).

---

## JAX Best Practices Review

### ✅ What We're Doing Right

1. **Using `lax.scan` for loops** ✅
   - No Python `for` loops over timesteps
   - Proper carry/output separation
   - Reverse scan for smoother

2. **Using `lax.cond` for branching** ✅
   - No Python `if/else` inside JIT'd functions
   - Static shapes in both branches

3. **Using `vmap` for batch operations** ✅
   - Sigma point propagation: `vmap(dynamics_function)(sigmas, ...)`
   - Efficient parallelization

4. **Static shapes** ✅
   - All arrays have compile-time known shapes
   - Padding for variable-length sequences (IMU indices)

5. **PSD-aware linear algebra** ✅
   - `psd_solve` instead of `jnp.linalg.solve`
   - Cholesky decomposition for covariances
   - `symmetrize()` to enforce PSD structure

6. **Avoiding concretization** ✅ (after CR-1 fix)
   - Removed `float(jnp.mean(...))` calls
   - Keep JAX scalars as JAX arrays

### ⚠️ What Could Be Improved

1. **Preprocessing outside JIT boundary** ⚠️
   - IMU index computation uses NumPy
   - **Status:** Intentional design for irregular timing
   - **Action:** Document clearly (already done in CR-1)

2. **No explicit `@jax.jit` decorators** ⚠️
   - Dynamax: Users can JIT entire function
   - Ours: Internal `lax.scan` is JIT'd, but top-level isn't
   - **Status:** By design (preprocessing needed)
   - **Action:** Add documentation about JIT strategy

3. **Large functions** ⚠️
   - `extended_kalman_filter()` is 200+ lines
   - Hard to reason about JIT boundaries
   - **Recommendation:** Consider extracting sub-functions

---

## Recommendations

### Priority 1 (Critical - Before Milestone 4)

None - all critical issues resolved in CR-1 and CR-2.

### Priority 2 (High - During Milestone 4)

**1. Unify Covariance Update** (1 hour)

```python
# Update UKF to use Joseph form like EKF
# File: src/trodestrack/models/ukf.py:643

# Current
P_upd = P_in - K @ S_active @ K.T

# Proposed
from trodestrack.models.filter_common import joseph_update
# Need to extract effective H matrix for Joseph form
# This requires some refactoring of the lifted subspace logic
```

**2. Add JIT Strategy Documentation** (30 min)

- Document why top-level functions aren't JIT-decorated
- Explain preprocessing requirements
- Show how to pre-process and JIT core logic if needed

### Priority 3 (Medium - Post-Milestone 4)

**1. Add IEKF Support** (3-4 hours)

- Add `num_iter` parameter to EKF
- Wrap update step in `lax.scan` for iteration
- Test on highly nonlinear scenarios

**2. Consider Extracting Preprocessing** (2-3 hours)

```python
# Current
result = extended_kalman_filter(config, t_imu, U_imu, ...)

# Proposed
preprocessed = preprocess_ekf_inputs(t_imu, U_imu, t_cam, ...)
result = extended_kalman_filter_core(config, preprocessed)  # JIT-able
```

---

## Conclusion

### Overall Assessment: ✅ **EXCELLENT**

Our implementations are **production-ready** and follow JAX best practices correctly:

1. ✅ Proper use of `lax.scan`, `lax.cond`, `vmap`
2. ✅ No Python loops or conditionals in hot paths
3. ✅ Static shapes throughout
4. ✅ PSD-aware linear algebra
5. ✅ Numerical stability (symmetrize, Joseph form in EKF)
6. ✅ CR-1 and CR-2 fixes ensure correctness

### Minor Improvements

- Consider Joseph form in UKF for consistency
- Add JIT strategy documentation
- IEKF support (future enhancement)

### No Blockers

**Ready to proceed with Milestone 4 - Integration & QA.**

---

## References

- Dynamax EKF: `dynamax_code/inference_ekf.py`
- Dynamax UKF: `dynamax_code/inference_ukf.py`
- Trodestrack EKF: `src/trodestrack/models/ekf.py`
- Trodestrack UKF: `src/trodestrack/models/ukf.py`
- JAX Documentation: <https://jax.readthedocs.io/>
- Särkkä (2013): "Bayesian Filtering and Smoothing"

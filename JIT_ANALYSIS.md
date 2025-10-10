# Full JIT Compilation Analysis

Analysis of what prevents full JIT compilation of trodestrack's EKF/UKF filters and what would be required to make them fully JIT-able.

**Date:** 2025-10-10
**Context:** Following dynamax comparison review

---

## Current JIT Status

### ✅ What IS JIT-Compatible

The **core filtering logic** uses JAX primitives correctly:
```python
def filter_step(carry, t_idx):
    # ... uses lax.scan, lax.cond, vmap ...
    state_out, _ = lax.scan(propagate_imu, state_in, imu_indices)
    # ... more JAX operations ...
    return carry, outputs

# This entire section JITs automatically via lax.scan
carry_init = (initial_state, 0.0)
(_, ll), outputs = lax.scan(filter_step, carry_init, jnp.arange(n_cam))
```

### ⚠️ What Is NOT JIT-Compatible

The **preprocessing section** uses operations that break JIT:

```python
def extended_kalman_filter(ekf_config, t_imu, U_imu, t_cam, ...):
    # 1. NumPy searchsorted
    cuts = np.searchsorted(t_imu, t_cam)  # ❌ NumPy operation
    counts = np.diff(np.r_[0, cuts])       # ❌ NumPy operation
    max_imu_per_frame = int(counts.max()) # ❌ Python int()

    # 2. Python loop with conditionals
    def compute_imu_index_arrays():
        all_indices = []
        for i in range(n_cam):  # ❌ Python for loop (NOT in lax.scan)
            if i == 0:          # ❌ Python if (outside lax.cond)
                indices = jnp.full(...)
            else:
                # JAX ops here are fine
                indices = ...
            all_indices.append(indices)  # ❌ Python list append
        return jnp.array(all_indices)

    # 3. Optional initialization
    if initial_state is None:  # ❌ Python conditional
        initial_state = initialize_state(...)  # ❌ Non-JIT function

    # 4. LED spacing estimation
    if ekf_config.led_distance is None:  # ❌ Python conditional
        estimated_led_distance = estimate_led_spacing(...)  # ❌ Uses jnp.where without size
```

---

## Barriers to Full JIT

### Barrier 1: NumPy `searchsorted` (Lines 815-817)

**Problem:**
```python
cuts = np.searchsorted(t_imu, t_cam)  # NumPy, not JAX
```

**Why it's there:**
- Maps irregular camera timestamps to IMU indices
- Finds which IMU samples fall between each camera frame

**JAX Alternative:**
```python
# Option 1: Use jnp.searchsorted (exists in JAX)
cuts = jnp.searchsorted(t_imu_jax, t_cam_jax)  # ✅ JIT-able

# Option 2: Compute on-the-fly in filter_step
# More expensive but avoids preprocessing
```

**Effort:** Low - Simple replacement

---

### Barrier 2: Python Loop in `compute_imu_index_arrays` (Lines 822-839)

**Problem:**
```python
def compute_imu_index_arrays():
    all_indices = []
    for i in range(n_cam):  # Python loop
        if i == 0:
            ...
    return jnp.array(all_indices)
```

**Why it's there:**
- Pre-computes which IMU samples belong to each camera interval
- Creates padded array for efficient `lax.scan`

**JAX Alternative:**
```python
# Option 1: Use lax.scan to build array
def build_indices(carry, i):
    t_prev = lax.cond(i == 0, lambda: t_cam_jax[0], lambda: t_cam_jax[i-1])
    t_current = t_cam_jax[i]
    mask = (t_imu_jax > t_prev) & (t_imu_jax <= t_current)
    indices = jnp.where(mask, size=max_imu_per_frame, fill_value=-1)[0]
    return carry, indices

_, imu_index_arrays = lax.scan(build_indices, None, jnp.arange(n_cam))

# Option 2: Vectorized approach
# Use vmap over camera frames
```

**Effort:** Medium - Requires restructuring loop

---

### Barrier 3: `initialize_state` (Lines 801-808)

**Problem:**
```python
if initial_state is None:
    initial_state = initialize_state(...)  # Uses jnp.where without size parameter
```

**Why it's there:**
- Bootstraps filter state from first valid observations
- Uses `jnp.where(mask)[0]` which requires static size for JIT

**Inside `initialize_state` (filter_common.py:177):**
```python
valid_indices = jnp.where(mask)[0]  # ❌ Dynamic size
first_valid = valid_indices[0] if len(valid_indices) > 0 else 0  # ❌ Python conditional
```

**JAX Alternative:**
```python
# Option 1: Always require initial_state parameter (no auto-init)
def extended_kalman_filter(ekf_config, ..., initial_state):  # Required, not Optional
    # No initialization logic
    ...

# Option 2: JIT-safe initialization (complex)
def initialize_state_jit(led1_obs, led2_obs, mask, dt_cam, led_distance):
    # Use argmax to find first valid index
    valid_mask = mask.astype(jnp.int32)
    first_valid_idx = jnp.argmax(valid_mask)  # ✅ JIT-safe

    # Use lax.select instead of Python conditionals
    pos_init = lax.select(
        mask[first_valid_idx],
        (led1_obs[first_valid_idx] + led2_obs[first_valid_idx]) / 2,
        jnp.zeros(2)
    )
    ...
```

**Effort:** High - Requires rewriting entire initialization logic

---

### Barrier 4: `estimate_led_spacing` (Lines 790-795)

**Problem:**
```python
if ekf_config.led_distance is None:
    estimated_led_distance = estimate_led_spacing(...)
```

**Inside `estimate_led_spacing` (ekf.py:99):**
```python
def estimate_led_spacing(...):
    valid_frames = mask_cam & jnp.isfinite(Z_cam_led1[:, 0]) & ...
    distances = jnp.linalg.norm(Z_cam_led1 - Z_cam_led2, axis=1)
    valid_distances = jnp.where(valid_frames, distances, jnp.nan)
    median_distance = jnp.nanmedian(valid_distances)  # ✅ Actually JIT-safe!
    return float(median_distance)  # ❌ Concretization
```

**JAX Alternative:**
```python
# Just remove float() call
median_distance = jnp.nanmedian(valid_distances)  # Keep as JAX scalar
return median_distance
```

**Effort:** Low - Remove one `float()` call

---

### Barrier 5: Python Conditional for Optional Parameters

**Problem:**
```python
if ekf_config.led_distance is None:  # Python if
    ...
else:
    ...
```

**JAX Alternative:**
```python
# Option 1: Always require parameter (no None)
# Option 2: Use sentinel value and lax.cond
led_distance = lax.cond(
    ekf_config.led_distance < 0,  # Use -1 as sentinel
    lambda: estimate_led_spacing(...),
    lambda: ekf_config.led_distance
)
```

**Effort:** Low-Medium

---

## Proposed Solution: Dual-API Design

### Option A: Separate Preprocessing (Recommended)

```python
@dataclass
class EKFPreprocessed:
    """Pre-computed values for JIT-compatible filtering."""
    t_imu: jnp.ndarray
    U_imu: jnp.ndarray
    t_cam: jnp.ndarray
    Z_cam_led1: jnp.ndarray
    Z_cam_led2: jnp.ndarray
    mask_cam: jnp.ndarray
    initial_state: EKFState
    imu_index_arrays: jnp.ndarray  # Pre-computed
    dt_imu_mean: float
    config: EKFConfig  # With led_distance filled in


def preprocess_ekf_inputs(
    ekf_config: EKFConfig,
    t_imu: np.ndarray,
    U_imu: np.ndarray,
    t_cam: np.ndarray,
    Z_cam_led1: np.ndarray,
    Z_cam_led2: np.ndarray,
    mask_cam: np.ndarray,
    initial_state: EKFState | None = None,
) -> EKFPreprocessed:
    """Preprocess inputs for JIT-compatible filtering (NumPy operations allowed)."""
    # Convert to JAX
    t_imu_jax = jnp.array(t_imu)
    # ... all other conversions

    # Estimate LED spacing if needed
    if ekf_config.led_distance is None:
        led_distance = estimate_led_spacing(...)
        config_for_filter = EKFConfig(**{**ekf_config.__dict__, 'led_distance': led_distance})
    else:
        config_for_filter = ekf_config

    # Initialize state if needed
    if initial_state is None:
        initial_state = initialize_state(...)  # Can use non-JIT version

    # Compute IMU indices (NumPy allowed)
    cuts = np.searchsorted(t_imu, t_cam)
    # ... build imu_index_arrays with Python loops

    return EKFPreprocessed(
        t_imu=t_imu_jax,
        U_imu=U_imu_jax,
        ...
        imu_index_arrays=jnp.array(imu_index_arrays),
        config=config_for_filter,
    )


@jax.jit  # ✅ Can JIT this!
def extended_kalman_filter_core(preprocessed: EKFPreprocessed) -> EKFResult:
    """JIT-compatible core filter (pure JAX operations)."""

    def filter_step(carry, t_idx):
        # All inputs are already JAX arrays
        imu_indices = preprocessed.imu_index_arrays[t_idx]
        # ... rest of filter logic (unchanged)

    carry_init = (preprocessed.initial_state, 0.0)
    (_, ll), outputs = lax.scan(filter_step, carry_init, jnp.arange(len(preprocessed.t_cam)))

    return EKFResult(...)


# User-facing API (backward compatible)
def extended_kalman_filter(
    ekf_config: EKFConfig,
    t_imu: np.ndarray,
    ...
) -> EKFResult:
    """Convenient API (same as before)."""
    preprocessed = preprocess_ekf_inputs(ekf_config, t_imu, ...)
    return extended_kalman_filter_core(preprocessed)
```

**Advantages:**
- ✅ Backward compatible (existing code works)
- ✅ Users can call `extended_kalman_filter()` as before
- ✅ Advanced users can preprocess once, JIT core, run multiple times
- ✅ Clear separation of concerns
- ✅ No complex rewrites of initialization logic

**Usage:**
```python
# Simple usage (no JIT)
result = extended_kalman_filter(config, t_imu, U_imu, ...)

# Advanced usage (with JIT)
preprocessed = preprocess_ekf_inputs(config, t_imu, U_imu, ...)
jitted_filter = jax.jit(extended_kalman_filter_core)

# Run filter multiple times with different configs (JIT benefit)
result1 = jitted_filter(preprocessed)
result2 = jitted_filter(replace(preprocessed, config=config2))
```

---

### Option B: Make Everything JIT-able (Not Recommended)

**Changes Required:**
1. Replace `np.searchsorted` → `jnp.searchsorted`
2. Replace Python loop → `lax.scan` in `compute_imu_index_arrays`
3. Rewrite `initialize_state` to avoid `jnp.where` without size
4. Remove `float()` concretizations
5. Replace `if/else` → `lax.cond` for optionals

**Effort:** ~8-12 hours
**Complexity:** High
**Risk:** Medium (may introduce bugs)
**Benefit:** Marginal (users rarely need to JIT entire function)

---

## Recommendation

### ✅ **Implement Option A: Dual-API Design**

**Priority:** P2 (Nice-to-have, not critical)

**Effort:** ~4-6 hours
- 2 hours: Extract preprocessing to separate function
- 1 hour: Create JIT-safe core function
- 1 hour: Write tests
- 1 hour: Documentation

**Benefits:**
- Users who want JIT can preprocess once, run many times
- Backward compatible
- Clean separation of NumPy (preprocessing) and JAX (core)
- No complex rewrites

**When to do it:**
- Post-Milestone 4 (not blocking)
- If users request JIT support
- If we need to run same data through multiple filter configs

---

## Comparison with Dynamax

**Dynamax approach:**
- Assumes uniform time grid (no irregular timestamps)
- No preprocessing needed
- Entire function is JIT-able
- **Trade-off:** Less flexible for real sensor data

**Trodestrack approach:**
- Handles irregular timestamps
- Dual-rate sensors (IMU @ 200 Hz, camera @ 30 Hz)
- Preprocessing required for indexing
- **Trade-off:** Not directly JIT-able, but more realistic

**Verdict:** Our approach is correct for the use case. Dual-API design gives best of both worlds.

---

## Conclusion

### Current Status
- ✅ Core filtering logic IS JIT-compatible (uses `lax.scan`)
- ⚠️ Preprocessing prevents full JIT of top-level function
- ✅ This is **intentional design** for dual-rate sensor fusion

### To Enable Full JIT
**Recommended:** Implement dual-API design (Option A)
- 4-6 hours effort
- Backward compatible
- Clear benefits for advanced users

**Not Recommended:** Rewrite everything to be JIT-able (Option B)
- 8-12 hours effort
- Complex, risky
- Marginal benefit

### Action
**Priority:** P2 (Post-Milestone 4)
**Status:** Document design, implement if user demand justifies effort

---

## References
- JAX Documentation: https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html
- Dynamax: `dynamax_code/inference_ekf.py`
- Trodestrack EKF: `src/trodestrack/models/ekf.py`
- CR-1 Fix: Commit 0a61172

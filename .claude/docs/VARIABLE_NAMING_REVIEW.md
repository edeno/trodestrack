# Variable Naming Convention Review - TrodesTrack

**Date:** 2025-10-13
**Reviewer:** Code Quality Specialist

---

## Executive Summary

**Overall Assessment:** ⚠️ **MIXED - Some Excellent Patterns, Some Concerns**

TrodesTrack demonstrates **strong domain-specific naming** conventions appropriate for signal processing and Kalman filtering, but deviates from general Python best practices in several areas. The naming reflects the mathematical nature of the code (Kalman filtering, state estimation) where single-letter variables like `P`, `Q`, `F`, `H` are standard notation.

**Key Finding:** The project uses **two distinct naming philosophies**:
1. **Mathematical/Signal Processing Convention**: Single letters (`P`, `Q`, `F`, `H`) for filter matrices - standard in academic literature
2. **Descriptive Python Convention**: Full words (`filtered_means`, `measurement_noise_pos`) for high-level APIs

Both are valid, but **mixing them creates inconsistency** that may confuse newcomers.

---

## Detailed Analysis

### ✅ Excellent Patterns

#### 1. NamedTuple Result Types (Outstanding!)

```python
class EKFResult(NamedTuple):
    filtered_means: jnp.ndarray      # ✅ Clear and descriptive
    filtered_covariances: jnp.ndarray  # ✅ Unambiguous
    predicted_means: jnp.ndarray      # ✅ Self-documenting
    predicted_covariances: jnp.ndarray
    marginal_loglik: float
    estimated_led_distance: float | None
```

**Why this is excellent:**
- Full descriptive names (not `x_filt`, `P`)
- Explicit about filtered vs predicted
- Unambiguous units implied (`_distance` → meters, `_loglik` → log-likelihood)
- Accessible to users unfamiliar with Kalman filter notation

---

#### 2. Configuration Classes (Very Good)

```python
class EKFConfig(FilterCoreConfig):
    process_noise_pos: float         # ✅ Clear physical meaning
    process_noise_vel: float         # ✅ Self-documenting
    measurement_noise_pos: float     # ✅ Obvious what it controls
    damping_coeff: float             # ✅ Standard physics term
    led_distance: float | None       # ✅ Explicit units implied
```

**Why this works:**
- Configuration uses full descriptive names
- Physical interpretation clear without comments
- Avoids cryptic abbreviations

---

#### 3. Function Names (Excellent)

```python
def extended_kalman_filter(...)      # ✅ Full words, clear intent
def rts_smoother(...)                # ✅ RTS is standard acronym
def extract_position(...)            # ✅ Verb + noun pattern
def compute_nees(...)                # ✅ Verb + acronym (NEES is standard)
def simulate_rat_imu(...)            # ✅ Domain-specific, descriptive
```

**Best practices followed:**
- Verb-noun pattern for functions
- Full words over abbreviations (except standard acronyms)
- Domain terminology used correctly (NEES, RTS, IMU)

---

#### 4. Constants (Good)

```python
PRD_POSITION_RMSE_M = 0.02         # ✅ UPPER_CASE for constants
PRD_VELOCITY_RMSE_MS = 0.10        # ✅ Units in name (_M, _MS)
PRD_HEADING_MAE_DEG = 7.0          # ✅ Metric type in name (MAE)

LAYOUT_2D_FULL = StateLayout(...)  # ✅ Descriptive constant
HEADING_GATE_THRESHOLD = ...       # ✅ Clear purpose
```

---

### ⚠️ Questionable Patterns

#### 1. Single-Letter Variable Names (Mathematical Convention)

**Found in filter implementations:**

```python
# In predict_step():
m, P = state.mean, state.cov        # ⚠️ Mathematical notation
F = jacfwd(f)                       # ⚠️ F for Jacobian
Q = assemble_Q(...)                 # ⚠️ Q for process noise

# In update_step():
H = camera_model.jacobian(m)       # ⚠️ H for observation Jacobian
R = camera_model.meas_cov(...)     # ⚠️ R for measurement noise
K = psd_solve(S, H @ P).T          # ⚠️ K for Kalman gain
S = H @ P @ H.T + R                # ⚠️ S for innovation covariance
```

**Issue:**
- Violates PEP 8 guideline: "Never use the characters 'l' (lowercase letter el), 'O' (uppercase letter oh), or 'I' (uppercase letter eye) as single character variable names"
- Not self-documenting for readers unfamiliar with Kalman filtering
- Hard to search/grep (try searching for "F =" in codebase)

**Counterargument (Valid):**
- **This IS standard notation** in Kalman filtering literature
- Academic papers use P, Q, F, H, R, K universally
- Engineers trained in signal processing expect this notation
- Changing would make code **harder** to match with textbooks

**Verdict:** This is a **domain-specific exception** to Python norms. In robotics/control/signal processing, single-letter matrix names are the standard. However, the codebase should be explicit about this choice.

---

#### 2. Inconsistent Suffixes for Array Types

```python
# JAX arrays have "_jax" suffix:
t_imu_jax: jnp.ndarray             # ⚠️ Suffix indicates JAX array
U_imu_jax: jnp.ndarray
Z_cam_led1_jax: jnp.ndarray

# But numpy arrays don't have suffix:
t_imu: np.ndarray                  # No "_np" suffix
U_imu: np.ndarray
```

**Issue:**
- Inconsistent: Why suffix JAX but not numpy?
- Hungarian notation (encoding type in name) is discouraged in modern Python
- Type hints already specify the type: `jnp.ndarray` vs `np.ndarray`

**Better approach:**
```python
# Let type hints do the work:
def function(
    t_imu: jnp.ndarray,      # Type hint is sufficient
    U_imu: jnp.ndarray,
) -> jnp.ndarray:
```

**Current justification:**
- Helps distinguish host-side (numpy) vs device-side (JAX) arrays
- Prevents accidental mixing during JIT compilation
- Internal implementation detail, not exposed to users

**Verdict:** Acceptable for **internal** JIT implementations, but should not leak into public API.

---

#### 3. Capitalized Data Arrays (Inconsistent with PEP 8)

```python
U_imu: jnp.ndarray       # ⚠️ Capital U (like constant)
Z_cam_led1: jnp.ndarray  # ⚠️ Capital Z (measurement convention)
P: jnp.ndarray           # ⚠️ Capital P (covariance matrix)
Q: jnp.ndarray           # ⚠️ Capital Q (process noise)
```

**Issue:**
- PEP 8 reserves CAPITAL_NAMES for constants
- These are **variables** (data), not constants
- Violates Python conventions

**Counterargument:**
- **Signal processing convention**: Capital letters for matrices/arrays
  - `U` = input matrix
  - `Z` = measurement vector
  - `P` = covariance matrix
- Matches MATLAB/Octave/Julia code from academic literature
- Distinguishes matrices (capital) from scalars (lowercase): `P` vs `p`

**Verdict:** Another **domain-specific exception**. In control theory:
- `u` = scalar input, `U` = input matrix
- `z` = scalar measurement, `Z` = measurement vector
- This is intentional and follows field conventions

---

#### 4. Abbreviated Suffixes (Cryptic for Newcomers)

```python
t_imu        # Time stamps for IMU
t_cam        # Time stamps for camera
U_imu        # IMU inputs/control
Z_cam_led1   # Camera measurements LED 1
mask_cam     # Camera validity mask
conf_cam     # Camera confidence scores
```

**Issue:**
- `t_`, `U_`, `Z_` are cryptic without domain knowledge
- Not obvious that `t` = timestamps, `U` = inputs, `Z` = measurements

**Better (for general Python):**
```python
imu_timestamps       # ✅ Explicit
imu_measurements     # ✅ Clear
camera_measurements  # ✅ Self-documenting
camera_mask         # ✅ Readable
```

**Current justification:**
- Follows control theory notation
- `t` = time is universal
- `U` = control input (standard in optimal control)
- `Z` = observation (standard in filtering/estimation)

**Verdict:** Acceptable for **signal processing domain**, but should be **documented explicitly** in developer guide.

---

### ❌ Problematic Patterns

#### 1. Inconsistent Plural Forms

```python
filtered_means: jnp.ndarray      # ✅ Plural (N samples)
filtered_covariances: jnp.ndarray  # ✅ Plural

BUT:

t_imu: np.ndarray    # ⚠️ Singular but contains N timestamps
U_imu: np.ndarray    # ⚠️ Singular but is (N, d) matrix
```

**Issue:**
- `filtered_means` is plural (implies multiple states)
- `t_imu` is singular but contains multiple timestamps
- Inconsistent about whether variable name reflects plurality

**Fix:**
```python
# Be consistent:
imu_timestamps  # Plural
imu_inputs      # Plural
camera_measurements  # Plural
```

---

#### 2. Ambiguous Boolean Naming

```python
has_vision_t: bool              # ✅ Good (clear boolean)

BUT:

mask_cam: jnp.ndarray           # ⚠️ Ambiguous: True=valid or invalid?
```

**Issue:**
- `mask` doesn't indicate polarity
- Does `mask_cam[i] == True` mean "camera valid" or "camera masked out"?

**Fix:**
```python
# Be explicit about polarity:
camera_valid: jnp.ndarray     # True = data is valid
camera_dropout: jnp.ndarray   # True = data is missing
is_vision_available: bool     # Clear boolean
```

---

#### 3. Vague Function Parameter Names

```python
def extract_position(
    states: jnp.ndarray,    # ⚠️ Vague: filtered, predicted, smoothed?
    layout: StateLayout,
) -> np.ndarray:
```

**Issue:**
- `states` is ambiguous (filtered? predicted? truth?)
- Could be `filtered_states`, `predicted_states`, `smoothed_states`

**Fix:**
```python
def extract_position(
    state_vectors: jnp.ndarray,  # ✅ More descriptive
    layout: StateLayout,
) -> np.ndarray:
    """Extract positions from state vectors.

    Parameters
    ----------
    state_vectors : array (N, n)
        State vectors (filtered, predicted, or smoothed).
    """
```

---

## Comparison to Best Practices

### PEP 8 Compliance

| PEP 8 Guideline | TrodesTrack | Verdict |
|-----------------|-------------|---------|
| Function names: lowercase_with_underscores | ✅ Yes | PASS |
| Class names: CapWords | ✅ Yes | PASS |
| Constants: UPPER_CASE | ✅ Yes | PASS |
| Variables: lowercase_with_underscores | ⚠️ Partial (except matrices) | CONDITIONAL |
| Avoid single-letter names | ❌ No (P, Q, F, H, K used) | FAIL* |
| Descriptive names preferred | ✅ Yes (public API) | PASS |

*FAIL with valid justification (domain conventions)

---

### Domain-Specific Best Practices (Kalman Filtering)

| Convention | TrodesTrack | Verdict |
|------------|-------------|---------|
| P for covariance | ✅ Yes | PASS |
| Q for process noise | ✅ Yes | PASS |
| R for measurement noise | ✅ Yes | PASS |
| F for state transition Jacobian | ✅ Yes | PASS |
| H for observation Jacobian | ✅ Yes | PASS |
| K for Kalman gain | ✅ Yes | PASS |
| x for state vector | ✅ Yes (as `m` for mean) | PASS |
| z for measurement | ✅ Yes (as `Z_cam`) | PASS |

**Conclusion:** TrodesTrack follows **standard Kalman filtering notation** faithfully.

---

## Recommendations

### 🔴 Critical (Must Fix)

**None.** The current naming is **intentional and domain-appropriate**.

---

### 🟡 High Priority (Should Fix)

#### 1. Document Naming Conventions (1 hour)

**Add to CLAUDE.md or new CONVENTIONS.md:**

```markdown
## Naming Conventions

TrodesTrack follows two complementary naming philosophies:

### Public API: Descriptive Python Naming

User-facing code uses full descriptive names:
- `filtered_means`, `predicted_covariances` (not `x`, `P`)
- `measurement_noise_pos` (not `R`)
- `process_noise_vel` (not `Q_vel`)

### Internal Implementation: Mathematical Notation

Filter internals use standard Kalman filtering notation:
- `P` = state covariance matrix
- `Q` = process noise covariance
- `R` = measurement noise covariance
- `F` = state transition Jacobian (∂f/∂x)
- `H` = observation Jacobian (∂h/∂x)
- `K` = Kalman gain
- `m` = state mean vector

This matches academic literature and simplifies comparison with textbooks (Särkkä, Bar-Shalom, etc.).

### Signal Processing Conventions

Data arrays follow control theory notation:
- `t_imu` = IMU timestamps (not `imu_times`)
- `U_imu` = IMU input matrix (control signal)
- `Z_cam` = camera measurement vector (observations)

Capital letters denote matrices/arrays:
- `U` = input matrix (N, d)
- `Z` = measurement matrix (N, m)
- `P`, `Q`, `R` = covariance matrices

### Why This Matters

When debugging filters, you need to compare code with papers. Using `filtered_covariance_at_timestep_k` instead of `P[k]` makes cross-referencing impossible.
```

**Acceptance criteria:**
- [ ] Documentation explains the two naming philosophies
- [ ] Justifies deviations from PEP 8
- [ ] Links to relevant textbooks/papers

---

#### 2. Fix Ambiguous Boolean Naming (30 minutes)

**Current:**
```python
mask_cam: jnp.ndarray  # ⚠️ True = valid or masked?
```

**Fix:** Rename for clarity:
```python
# Option 1: Positive polarity
is_camera_valid: jnp.ndarray  # True = data available

# Option 2: Explicit mask
camera_dropout_mask: jnp.ndarray  # True = data missing
```

**Files to update:**
- Search for `mask_cam` across codebase
- Ensure consistent polarity (True = valid)
- Update docstrings to clarify

---

#### 3. Make State Parameter Names More Specific (1 hour)

**Current:**
```python
def extract_position(
    states: jnp.ndarray,  # Ambiguous
    layout: StateLayout,
) -> np.ndarray:
```

**Better:**
```python
def extract_position(
    state_vectors: jnp.ndarray,  # More specific
    layout: StateLayout,
) -> np.ndarray:
    """Extract positions from state vectors.

    Parameters
    ----------
    state_vectors : array (N, n)
        State vectors from filter or smoother output.
        Can be filtered_means, predicted_means, or smoothed_means.
    """
```

**Files to update:**
- `state_layout.py` helper functions
- All docstrings mentioning `states` parameter

---

### 🟢 Low Priority (Nice to Have)

#### 4. Add Type Aliases for Clarity (30 minutes)

```python
# In types.py or utils.py:
from typing import TypeAlias
import numpy as np

# Time-series data aliases
Timestamps: TypeAlias = np.ndarray  # (N,) timestamps in seconds
IMUInputs: TypeAlias = np.ndarray   # (N, d) IMU measurements
CameraMeasurements: TypeAlias = np.ndarray  # (N, 2) position observations

# Filter state aliases
StateVector: TypeAlias = np.ndarray    # (n,) single state
StateSeries: TypeAlias = np.ndarray    # (N, n) state time series
Covariance: TypeAlias = np.ndarray     # (n, n) covariance matrix
CovarianceSeries: TypeAlias = np.ndarray  # (N, n, n) covariances

# Usage:
def extract_position(
    state_series: StateSeries,  # ✅ Self-documenting!
    layout: StateLayout,
) -> np.ndarray:
```

**Benefits:**
- Self-documenting code
- Type checkers understand semantics
- Easier to refactor (change alias definition once)

---

#### 5. Consider Verbose Mode for Single-Letter Variables (Optional)

For debugging/teaching, provide verbose alternatives:

```python
# In filter_common.py or similar:
VERBOSE_NAMES = False  # Set True for teaching/debugging

if VERBOSE_NAMES:
    state_covariance = P
    process_noise_cov = Q
    measurement_noise_cov = R
    kalman_gain = K
else:
    # Use standard notation
    pass
```

**Use case:** When teaching students or debugging, toggle verbose names.

---

## Summary of Issues by Severity

### Critical (Must Fix): 0
- None identified

### High (Should Fix): 3
1. ⚠️ Undocumented naming conventions (causes confusion)
2. ⚠️ Ambiguous boolean polarity (`mask_cam`)
3. ⚠️ Vague parameter names (`states`)

### Medium (Nice to Fix): 2
4. 💡 Inconsistent plural forms
5. 💡 JAX array suffixes leak into public API

### Low (Optional): 2
6. 💡 Type aliases for documentation
7. 💡 Verbose mode for teaching

---

## Conclusion

**Final Verdict:** ⚠️ **ACCEPTABLE WITH CAVEATS**

TrodesTrack's naming conventions are **appropriate for its domain** (Kalman filtering, signal processing, robotics). The use of single-letter variables (P, Q, F, H) and capital-letter matrices (U, Z) follows **standard academic notation** and is preferred in this field.

**However, the project MUST:**
1. **Document these conventions explicitly** so users understand the rationale
2. **Fix ambiguous names** (like `mask_cam`) for clarity
3. **Keep mathematical notation internal** (don't leak into public API)

**Key Insight:** The perceived "violations" of Python naming conventions are actually **adherence to domain best practices**. A Kalman filter implementation that doesn't use P, Q, R would be **harder** to maintain and verify against literature.

**Recommendation:** Add a "Naming Conventions" section to documentation explaining:
- Why single-letter variables are used (matches textbooks)
- When to use descriptive names (public API)
- When to use mathematical notation (internal implementations)

---

**Review completed:** 2025-10-13
**Key finding:** Naming is domain-appropriate but undocumented
**Action required:** Document conventions (1.5 hours of work)

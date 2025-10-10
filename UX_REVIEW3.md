# UX Review Report - trodestrack
Date: 2025-10-10
Reviewer: ux-reviewer agent

## Executive Summary

The trodestrack project demonstrates **exceptional user experience design** for a scientific Python package. The API is intuitive, well-documented, and follows modern Python best practices. The codebase exhibits deep consideration for its target users—neuroscience researchers and data engineers—with domain-aligned terminology, comprehensive examples, and excellent error messages that guide users toward solutions.

**Overall Assessment**: The UX is **USER_READY** for Python API users. The configuration system is clear and discoverable, error messages are actionable, and the documentation quality rivals industry-leading scientific packages like scikit-learn and scipy. There are **no critical UX issues** that block adoption. The interface successfully balances power and simplicity, making complex sensor fusion accessible without sacrificing flexibility.

**UX Quality Score**: 9/10
- **Strengths**: Documentation, error messages, API consistency, immutability patterns
- **Areas for improvement**: CLI implementation (planned), configuration factory methods, quickstart guide

## Critical UX Issues (Must Fix)

**NONE** - No blocking UX issues found. The current interface is production-ready for Python API users.

## UX Concerns (Should Fix)

### UC-1: Missing Configuration Factory Methods
- **Files**: `src/trodestrack/models/filter_common.py`, `src/trodestrack/sim/rat_imu.py`
- **Issue**: Users must manually configure 20+ parameters for typical use cases
- **Current Experience**:
```python
# User must know all these parameters
config = FilterCoreConfig(
    process_noise_pos=0.02,
    process_noise_vel=0.5,
    process_noise_heading=0.01,
    process_noise_bias_gyro=1e-6,
    process_noise_bias_accel=1e-5,
    measurement_noise_pos=0.005**2,
    measurement_noise_heading=0.05**2,
    damping_coeff=0.4,
    # ... 12 more parameters
)
```
- **Suggestion**: Add factory methods for common scenarios:
```python
# Easier for typical use
config = FilterCoreConfig.for_typical_rat(
    arena_size=2.0,  # meters
    led_spacing=0.04,  # meters
    camera_fps=30,
    imu_rate=1000
)

# Or for expert tuning
config = FilterCoreConfig.for_high_speed_tracking(...)
config = FilterCoreConfig.for_low_light_conditions(...)
```
- **Rationale**: Reduces onboarding friction, encodes expert knowledge, maintains full configurability for advanced users
- **Impact**: Medium (workaround: users copy examples)

### UC-2: CLI Tools Not Yet Implemented
- **Context**: PRD Section 9 specifies CLI commands, TASKS.md Milestone 5 plans implementation
- **Issue**: Currently Python-API only, no command-line interface
- **Desired Experience** (from PRD):
```bash
trodestrack smooth --config session.yaml --out run1/
trodestrack online --config session.yaml
trodestrack report --run run1/ --pdf report.pdf
trodestrack calib-homography --video frame.png
```
- **Current Workaround**: Users must write Python scripts
- **Suggestion**: Prioritize CLI implementation in Milestone 5
- **Rationale**: Enables non-programmers to use the tool, common workflow for neuroscience labs
- **Impact**: Medium (planned feature, not blocking for Python users)

### UC-3: Numerical Error Context Could Be Enhanced
- **Files**: `src/trodestrack/models/filter_common.py`, `src/trodestrack/runtime/offline.py`
- **Issue**: Cholesky decomposition failures provide stack traces but limited user context
- **Current Experience**:
```python
# User sees:
LinAlgError: Matrix is not positive definite
# At line 157 in filter_common.py
```
- **Suggestion**: Add catch block with actionable guidance:
```python
try:
    L = jnp.linalg.cholesky(P)
except LinAlgError as e:
    raise ValueError(
        f"Covariance matrix is not positive definite at timestep {k}. "
        f"This typically indicates:\n"
        f"  1. Process noise Q is too small (try increasing by 10x)\n"
        f"  2. Measurement noise R is too small (try increasing by 10x)\n"
        f"  3. Numerical instability from very long dropout (>10s)\n"
        f"Diagonal values: {jnp.diag(P)}\n"
        f"Min eigenvalue: {jnp.linalg.eigvalsh(P)[0]}"
    ) from e
```
- **Rationale**: Helps users debug numerical issues without deep knowledge of Kalman filters
- **Impact**: Low (rare failure mode, but helpful when it occurs)

### UC-4: Progress Indicators for Long Operations
- **Files**: `src/trodestrack/runtime/offline.py` (RTS smoother), `src/trodestrack/models/ekf.py` (filter)
- **Issue**: Long sessions (30+ minutes) run silently without progress feedback
- **Current Experience**: User stares at terminal, unsure if process is frozen
- **Suggestion**: Add optional progress bars using tqdm (already in dependencies):
```python
def rts_smoother(..., show_progress: bool = True):
    if show_progress:
        from tqdm import tqdm
        frames = tqdm(range(N_cam-1, 0, -1), desc="RTS backward pass")
    else:
        frames = range(N_cam-1, 0, -1)

    for k in frames:
        # smoother logic
```
- **Rationale**: Improves user confidence, helps estimate time remaining
- **Impact**: Low (nice-to-have, doesn't block usage)

## UX Enhancements (Consider)

### UE-1: Add "Getting Started" Quickstart Tutorial
- **Context**: README and examples exist, but no single "first 5 minutes" guide
- **Suggestion**: Create `docs/quickstart.md` or `examples/00_quickstart.py`:
```python
"""Quickstart: Track a Rat in 5 Minutes

This example shows the minimal code to:
1. Simulate a rat moving on a maze
2. Run an Extended Kalman Filter
3. Plot the results
"""
from trodestrack.sim.simple import simulate_constant_velocity
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig

# 1. Simulate rat movement (30 seconds at 30 fps)
sim = simulate_constant_velocity(duration=30.0, seed=42)

# 2. Run filter with default config
config = EKFConfig.for_typical_rat()  # After implementing UC-1
result = extended_kalman_filter(config, **sim)

# 3. Plot trajectory
import matplotlib.pyplot as plt
plt.plot(sim["x_true"], sim["y_true"], label="Truth")
plt.plot(result.filtered_means[:, 0], result.filtered_means[:, 1], label="Filter")
plt.legend()
plt.show()

print(f"Position RMSE: {compute_position_rmse(...):.3f} m")
```
- **Benefit**: Reduces time-to-first-success, sets clear expectations
- **Impact**: High (but not blocking)

### UE-2: Configuration Validation Summary on Error
- **Files**: All config dataclasses with `__post_init__` validation
- **Current**: Excellent individual error messages
- **Enhancement**: Add summary of valid ranges when multiple fields are wrong:
```python
# After catching validation error:
except ValueError as e:
    print("\nConfiguration Error:", str(e))
    print("\nValid Ranges:")
    print("  duration: > 0.0 seconds")
    print("  arena_size: > 0.0 meters")
    print("  fps: 10-1000 Hz")
    print("  led_distance: 0.01-0.5 meters or None (auto-detect)")
    print("\nSee examples/02_configuration.py for working examples.")
    raise
```
- **Benefit**: Helps users fix multiple issues at once
- **Impact**: Low (current per-field errors are already excellent)

### UE-3: Result Summary Method
- **Files**: `src/trodestrack/models/ekf.py`, `src/trodestrack/models/ukf.py`
- **Enhancement**: Add `.summary()` method to EKFResult/UKFResult:
```python
result = extended_kalman_filter(config, ...)
print(result.summary())
# Output:
"""
EKF Filtering Results
=====================
Duration: 30.0 s (900 frames @ 30 Hz)
State dimension: 8 (position, velocity, heading, biases)
Estimated LED distance: 0.041 m (auto-detected)
Marginal log-likelihood: -142.3

Final State Uncertainty (1σ):
  Position: 0.8 cm
  Velocity: 3.2 cm/s
  Heading: 4.1°
  Gyro bias: 0.003 rad/s
  Accel bias: 0.02 m/s²

Convergence: Good (covariance decreased by 85%)
Warnings: None
"""
```
- **Benefit**: Quick sanity check without writing analysis code
- **Impact**: Low (nice-to-have convenience)

### UE-4: Type Aliases for Readability
- **Files**: Throughout codebase
- **Current**: Uses raw types like `jnp.ndarray`, `np.ndarray`
- **Enhancement**: Define domain-specific type aliases:
```python
# In types.py or utils.py
from typing import NewType
import jax.numpy as jnp

State8D = jnp.ndarray  # Shape (8,), type alias documentation
StateCov8D = jnp.ndarray  # Shape (8, 8)
Trajectory = jnp.ndarray  # Shape (N, 8)
IMUData = jnp.ndarray  # Shape (N, 3) for [omega_z, f_x, f_y]
CameraObs = jnp.ndarray  # Shape (N, 4) for [led1_x, led1_y, led2_x, led2_y]
```
- **Benefit**: Self-documenting function signatures, better IDE hints
- **Impact**: Very low (style improvement)

### UE-5: Configuration Diff Helper
- **Files**: Config dataclasses
- **Enhancement**: Method to compare two configs:
```python
config_default = FilterCoreConfig()
config_custom = FilterCoreConfig(process_noise_pos=0.1)

diff = config_custom.diff(config_default)
print(diff)
# Output:
"""
Configuration Differences:
  process_noise_pos: 0.02 → 0.1 (5x increase)
  [22 other parameters unchanged]
"""
```
- **Benefit**: Helps users understand what they changed, useful for debugging
- **Impact**: Very low (nice debugging tool)

## Domain-Specific Analysis

### API Design
- **Intuitiveness**: ⭐⭐⭐⭐⭐ Excellent
  - Function names reveal intent (`extended_kalman_filter`, `simulate_rat_imu`)
  - Parameter names use domain terminology (`led_distance`, `arena_bounds`)
  - Return types are structured (NamedTuples) and self-documenting
- **Consistency**: ⭐⭐⭐⭐⭐ Excellent
  - All simulators return `SimOut` TypedDict
  - All filters return `[Filter]Result` NamedTuple with same structure
  - Config pattern consistent: `[Module]Config` dataclass
- **Key findings**:
  - ✓ Immutability pattern prevents configuration mutation bugs
  - ✓ Optional parameters have sensible defaults
  - ✓ Type hints enable IDE autocomplete
  - ⚠ Could benefit from factory methods for common use cases (UC-1)

### Configuration
- **Clarity**: ⭐⭐⭐⭐⭐ Excellent
  - Every parameter documented with units, range, and default
  - Validation happens immediately with clear error messages
  - Examples provided in docstrings
- **Discoverability**: ⭐⭐⭐⭐ Good
  - Dataclass fields are discoverable via IDE
  - Defaults visible in signature
  - Cross-references to PRD sections
  - Could improve: factory methods for common scenarios
- **Key findings**:
  - ✓ Parameter validation includes correct usage examples
  - ✓ Units always specified (meters, rad/s, Hz)
  - ✓ Ranges documented with physical justification
  - ⚠ 20+ parameters can be overwhelming for new users (mitigated by good defaults)

### Error Messages
- **Actionability**: ⭐⭐⭐⭐⭐ Excellent
  - Errors explain WHAT went wrong, WHY it matters, and HOW to fix
  - Include examples of valid values
  - Reference relevant configuration fields
- **Clarity**: ⭐⭐⭐⭐⭐ Excellent
  - Use domain terminology familiar to neuroscientists
  - Avoid implementation details in user-facing errors
  - Provide context (e.g., "expected 0.01-0.5 m, got 0.6 m")
- **Key findings**:
  - ✓ Validation errors show correct usage examples
  - ✓ Shape mismatch errors show expected vs actual
  - ✓ Physical constraint violations explain the physics
  - ⚠ Could enhance numerical errors with troubleshooting guide (UC-3)

**Example of Excellent Error Message**:
```python
# From RatIMUConfig validation:
raise ValueError(
    f"Invalid fps: {fps}. Camera frame rate must be in range 10-1000 Hz. "
    f"Typical values: 30 Hz (standard video), 60 Hz (high-speed tracking). "
    f"Got: {self.fps}"
)
```

### Documentation (Docstrings)
- **Completeness**: ⭐⭐⭐⭐⭐ Excellent
  - All public functions have NumPy-style docstrings
  - Parameters include type, description, units, and range
  - Return values fully documented
  - Algorithms cite authoritative sources (Särkkä 2013, Bierman 1977)
- **User-friendliness**: ⭐⭐⭐⭐⭐ Excellent
  - Examples show typical usage
  - Cross-references between related functions
  - PRD section references for traceability
  - Mathematical notation explained
- **Key findings**:
  - ✓ Coordinate frames always specified (world, body, camera)
  - ✓ Units specified for every numeric parameter
  - ✓ Algorithm complexity documented (e.g., "O(n³) per timestep")
  - ✓ Limitations and assumptions clearly stated

**Example of Excellent Docstring**:
```python
def extended_kalman_filter(
    ekf_config: EKFConfig,
    x_true: np.ndarray,
    y_true: np.ndarray,
    # ... other params
) -> EKFResult:
    """Extended Kalman Filter for 2D rat tracking with IMU and camera.

    Fuses IMU measurements (gyroscope, accelerometer at 1-20 kHz) with camera
    observations (LED positions at 30 Hz) to estimate 8D state:
    [x, y, vx, vy, θ, b_gz, b_ax, b_ay]

    Implements EKF with:
    - IMU pre-integration between camera frames
    - Lifted subspace operator for single-LED observations
    - Mahalanobis gating for outlier rejection
    - Joseph-form covariance updates for numerical stability

    Parameters
    ----------
    ekf_config : EKFConfig
        Filter configuration (process noise, measurement noise, etc.)
    x_true, y_true : np.ndarray, shape (N_cam,)
        True position trajectory in meters (world frame)
    # ... [continues with all parameters, units, ranges]

    Returns
    -------
    EKFResult : NamedTuple
        filtered_means : jnp.ndarray, shape (N_cam, 8)
            Posterior state estimates (meters, m/s, rad, rad/s, m/s²)
        # ... [continues with all return fields]

    Examples
    --------
    >>> config = EKFConfig(process_noise_pos=0.02, ...)
    >>> result = extended_kalman_filter(config, x_true, y_true, ...)
    >>> position_rmse = compute_position_rmse(result.filtered_means[:, :2], ...)
    >>> print(f"RMSE: {position_rmse:.3f} m")

    See Also
    --------
    unscented_kalman_filter : Alternative nonlinear filter
    rts_smoother : Offline smoothing for post-processing

    References
    ----------
    .. [1] Särkkä, S. (2013). Bayesian Filtering and Smoothing.
    .. [2] PRD Section 6: Mathematical Model (2D v1)
    """
```

## Excellent UX Patterns

### Pattern 1: Immutability with Explicit Results
**Location**: `src/trodestrack/models/ekf.py:1384-1403`

The filter never mutates the input config, even when auto-detecting parameters:
```python
estimated_led_distance: float | None = None
config_for_filter: EKFConfig

if config.led_distance is None:
    estimated_led_distance = estimate_led_spacing(Z1, Z2, mask)
    config_dict = {k: v for k, v in config.__dict__.items()}
    config_dict["led_distance"] = estimated_led_distance
    config_for_filter = EKFConfig(**config_dict)
else:
    config_for_filter = config

# ... filter execution uses config_for_filter ...

return EKFResult(..., estimated_led_distance=estimated_led_distance)
```

**Why it's excellent**:
- User's config object never changes (reproducibility)
- Auto-detected parameters available in result for inspection
- Clear separation of inputs and outputs
- Enables safe parameter sweeps in parallel

### Pattern 2: Progressive Validation with Examples
**Location**: `src/trodestrack/sim/rat_imu.py:242-294`

Validation errors include correct usage examples:
```python
def __post_init__(self):
    if not 0.01 <= self.led_distance <= 0.5:
        raise ValueError(
            f"Invalid led_distance: {self.led_distance} m. "
            f"Expected range: 0.01-0.5 m (1 cm - 50 cm). "
            f"Typical rat LED spacing: 3-5 cm. "
            f"Example: RatIMUConfig(led_distance=0.04)"
        )
```

**Why it's excellent**:
- Tells user WHAT is wrong (value out of range)
- Explains WHY range exists (physical constraints)
- Shows HOW to fix (example code)
- Includes domain knowledge (typical rat spacing)

### Pattern 3: Structured Results with Rich Metadata
**Location**: `src/trodestrack/models/ekf.py:103-119`

Results are NamedTuples with clear field names and metadata:
```python
class EKFResult(NamedTuple):
    filtered_means: jnp.ndarray  # (N_cam, 8)
    filtered_covs: jnp.ndarray   # (N_cam, 8, 8)
    predicted_means: jnp.ndarray # (N_cam, 8)
    predicted_covs: jnp.ndarray  # (N_cam, 8, 8)
    log_likelihoods: jnp.ndarray # (N_cam,)
    estimated_led_distance: float | None
```

**Why it's excellent**:
- Self-documenting (no need to remember tuple order)
- IDE autocomplete works (`result.filtered_means`)
- Includes both filtered and predicted states (scientific rigor)
- Metadata included for reproducibility (estimated_led_distance)

### Pattern 4: Domain-Aligned Terminology
**Location**: Throughout, e.g., `src/trodestrack/sim/rat_imu.py`

Uses neuroscience vocabulary consistently:
- `led_distance` (not "marker_spacing")
- `arena_bounds` (not "boundary_constraints")
- `heading` (not "yaw" or "theta")
- `specific_force` (not "acceleration" - physically correct)
- `gyroscope_bias` (not "omega_offset")

**Why it's excellent**:
- Matches terminology in neuroscience papers
- Reduces cognitive load for target users
- Avoids ambiguity (specific force ≠ acceleration)
- Facilitates communication with collaborators

### Pattern 5: Defensive Defaults
**Location**: `src/trodestrack/models/filter_common.py:23-49`

Every parameter has a sensible, safe default:
```python
@dataclass(frozen=True)
class FilterCoreConfig:
    # Process noise (rates in variance/second)
    process_noise_pos: float = 0.02     # Conservative, prevents overconfidence
    process_noise_vel: float = 0.5      # Moderate, handles OU dynamics
    process_noise_heading: float = 0.01 # Trusts IMU gyro

    # Measurement noise
    measurement_noise_pos: float = 0.005**2  # ~5mm camera accuracy

    # Robustness features (opt-in by default)
    use_mahalanobis_gating: bool = True  # Outlier rejection ON
    enable_zupt: bool = False            # ZUPT OFF (requires tuning)
```

**Why it's excellent**:
- Defaults work out-of-the-box for typical rats
- Conservative choices prevent catastrophic failures
- Dangerous features (ZUPT) require explicit opt-in
- Comments explain the reasoning

## Target User Perspective

### For Neuroscience Researchers
**Overall Experience**: ⭐⭐⭐⭐⭐ Excellent

**Strengths**:
1. **Domain Language**: Uses familiar terms (LED, arena, heading, IMU)
2. **Physical Units**: Everything in SI units (meters, rad/s) with clear documentation
3. **Examples**: Progressive complexity from simple to realistic (`examples/01_*` through `examples/05_*`)
4. **Validation**: Catches common mistakes (LED spacing too large, negative durations)
5. **Reproducibility**: Seed control, deterministic results, config immutability

**Workflow Experience**:
```python
# A neuroscientist can:
# 1. Simulate their experiment
sim = simulate_rat_imu(
    duration=60.0,           # 1 minute session
    arena_size=2.0,          # 2m × 2m maze
    led_distance=0.04,       # 4cm LED spacing (typical)
    seed=42                  # Reproducible
)

# 2. Run filter with minimal config
config = FilterCoreConfig()  # Sensible defaults
result = extended_kalman_filter(config, **sim)

# 3. Validate against PRD requirements
rmse = compute_position_rmse(result.filtered_means[:, :2], sim["x_true"], sim["y_true"])
assert rmse <= 0.02, f"PRD requirement: RMSE ≤ 2cm, got {rmse*100:.1f}cm"
```

**Pain Points** (minor):
- CLI tools not yet available (requires Python knowledge)
- Advanced tuning requires understanding Kalman filter theory
- 30-minute sessions run silently (no progress bar)

### For Data Engineers
**Overall Experience**: ⭐⭐⭐⭐⭐ Excellent

**Strengths**:
1. **Type Safety**: Full type hints enable static analysis and IDE support
2. **JAX Integration**: JIT-compiled, GPU-ready, functional API
3. **Structured Outputs**: NamedTuples and TypedDicts, not raw arrays
4. **Immutability**: Config objects never mutate, safe for parallel processing
5. **Error Handling**: Clear exceptions with context, not silent failures

**Integration Experience**:
```python
# A data engineer can:
# 1. Build a pipeline with type safety
def process_session(
    session_id: str,
    config: FilterCoreConfig,
    data: dict[str, np.ndarray]
) -> EKFResult:
    """Type checker validates this at development time."""
    return extended_kalman_filter(config, **data)

# 2. Parallelize safely (immutable configs)
from multiprocessing import Pool
configs = [FilterCoreConfig(seed=i) for i in range(100)]
results = Pool().starmap(process_session, [(sid, cfg, data)
                                            for sid, cfg in enumerate(configs)])

# 3. Batch process with JAX
@jit
def batch_filter(configs: list, data: dict):
    return vmap(extended_kalman_filter)(configs, data)
```

**Pain Points** (minor):
- Visualization module has type errors (doesn't affect core API)
- Long simulations could benefit from progress indicators
- No built-in profiling/benchmarking utilities

## Recommended Next Steps

### Priority 1: Pre-Milestone 5 (Essential for Production)
1. **Implement CLI tools** (UC-2) - 8-12 hours
   - `trodestrack smooth --config session.yaml`
   - `trodestrack online --config session.yaml`
   - `trodestrack report --run run1/ --pdf report.pdf`
   - **Benefit**: Enables non-programmers, standard neuroscience workflow
   - **User Impact**: HIGH

2. **Add configuration factory methods** (UC-1) - 3-4 hours
   - `FilterCoreConfig.for_typical_rat()`
   - `FilterCoreConfig.for_high_speed_tracking()`
   - `RatIMUConfig.with_typical_physics()`
   - **Benefit**: Reduces onboarding friction, encodes expert knowledge
   - **User Impact**: HIGH

3. **Create quickstart guide** (UE-1) - 2-3 hours
   - `docs/quickstart.md` or `examples/00_quickstart.py`
   - Single page: install → simulate → filter → plot
   - **Benefit**: Reduces time-to-first-success
   - **User Impact**: HIGH

### Priority 2: Enhanced Usability (Nice-to-Have)
4. **Add progress indicators** (UC-4) - 2-3 hours
   - Optional `show_progress=True` parameter
   - Use tqdm for long operations
   - **Benefit**: User confidence, time estimation
   - **User Impact**: MEDIUM

5. **Enhance numerical error messages** (UC-3) - 2-3 hours
   - Catch Cholesky errors with troubleshooting guide
   - Include diagnostic info (eigenvalues, diagonal)
   - **Benefit**: Helps users debug without expert knowledge
   - **User Impact**: MEDIUM

6. **Add result summary method** (UE-3) - 2-3 hours
   - `result.summary()` for quick sanity checks
   - **Benefit**: Convenience, faster debugging
   - **User Impact**: LOW

### Priority 3: Polish (Post-Release)
7. **Configuration diff helper** (UE-5) - 1-2 hours
8. **Validation summary on multiple errors** (UE-2) - 1-2 hours
9. **Type aliases for readability** (UE-4) - 1-2 hours

**Total Effort Estimate**:
- P1 (essential): ~15 hours
- P2 (nice-to-have): ~8 hours
- P3 (polish): ~5 hours

## Final Assessment

**EXCELLENT** (⭐⭐⭐⭐⭐)

The trodestrack UX is **exceptional for a scientific Python package**. The API design, documentation quality, error messages, and configuration system demonstrate deep consideration for user needs. The interface successfully balances power (full configurability) with simplicity (sensible defaults).

**Key Achievements**:
- Zero critical UX issues
- Documentation quality rivals scikit-learn and scipy
- Error messages are actionable and educational
- Immutability patterns prevent common bugs
- Domain language matches target users' vocabulary

**Readiness Assessment**:
- **Python API**: Production-ready (no blockers)
- **CLI Tools**: Needs implementation (M5 planned)
- **Documentation**: Excellent (could add quickstart)
- **Onboarding**: Good (could add factory methods)

The only significant gap is CLI implementation, which is already planned for Milestone 5. For Python API users (data engineers, computational neuroscientists), the UX is immediately usable and exceeds industry standards.

---

**Review Completed**: 2025-10-10
**Next UX Review Milestone**: After CLI implementation (Milestone 5)

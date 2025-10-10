# UX Review Report - trodestrack

**Date:** 2025-10-10
**Reviewer:** UX Reviewer Agent
**Version:** clean-slate branch (post-refactor)

---

## Executive Summary

The trodestrack codebase demonstrates **strong UX fundamentals** with excellent API design, comprehensive documentation, and thoughtful error handling. The project shows mature software engineering practices with a clear focus on scientific reproducibility and usability for neuroscience researchers.

**Key Strengths:**

- Exceptionally clear docstrings with mathematical context, examples, and units
- Consistent naming aligned with neuroscience domain (e.g., "LED", "IMU", "heading", "specific force")
- Excellent configuration validation with actionable error messages
- Rich simulation capabilities with sensible defaults that work out-of-the-box
- Strong type hints and immutability patterns reduce user errors

**Key Opportunities:**

- CLI commands are planned but not yet implemented (per M5 milestone)
- Some configuration complexity could benefit from helper constructors
- Error messages are generally good but occasionally lack recovery steps
- Documentation could add "Getting Started" narrative flow

**Overall:** The codebase is **USER_READY** for Python API users. Once CLI tools are implemented, they should follow the strong UX patterns established in the simulation and filter APIs.

---

## Critical UX Issues (Must Fix)

None identified. The codebase follows strong UX principles throughout.

---

## UX Concerns (Should Fix)

### Configuration Complexity

- [ ] **FilterCoreConfig has 19+ parameters** - overwhelming for first-time users [`src/trodestrack/models/filter_common.py:14-50`]
  - **Impact:** Users may not know where to start with tuning
  - **Suggestion:** Add factory methods like `FilterCoreConfig.for_typical_rat()`, `FilterCoreConfig.for_high_precision()`, or `FilterCoreConfig.minimal()` with documented use cases
  - **Example:**

    ```python
    # Instead of:
    config = FilterCoreConfig(
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        # ... 17 more parameters
    )

    # Offer:
    config = FilterCoreConfig.for_typical_rat()  # Sensible defaults
    config.led_distance = 0.05  # Override specific values
    ```

- [ ] **RatIMUSimConfig validation happens in **post_init**** - errors only appear after construction [`src/trodestrack/sim/rat_imu.py:262-451`]
  - **Impact:** Users get errors after creating object, not during IDE autocomplete
  - **Good:** Validation messages are excellent and actionable
  - **Suggestion:** Consider adding a `validate()` class method that can be called explicitly, or document that validation happens post-init

### Error Message Completeness

- [ ] **Cholesky decomposition failures in metrics** return `np.inf` without explanation [`src/trodestrack/qa/metrics.py:261-263`]
  - **Current:** `except np.linalg.LinAlgError: nees[i] = np.inf`
  - **Better:** Add warning or raise informative error:

    ```python
    except np.linalg.LinAlgError:
        warnings.warn(
            f"Covariance at timestep {i} is singular (not positive definite). "
            f"Filter may be over-confident or numerically unstable. "
            f"NEES set to infinity. Check P matrix conditioning."
        )
        nees[i] = np.inf
    ```

- [ ] **NaN handling in observations** - silent propagation could confuse users
  - **Location:** Throughout filter code (e.g., `ekf.py:493-515`)
  - **Current behavior:** NaN observations are gracefully skipped
  - **Opportunity:** Add optional debug logging or validation mode that warns about excessive NaN rates
  - **Example:** "Warning: 45% of LED1 observations are NaN - check camera tracking quality"

### Discoverability

- [ ] **No CLI entry point yet** - users must write Python scripts [`pyproject.toml:45`]
  - **Status:** Entry point defined (`trodestrack:main`) but not implemented
  - **Impact:** Neuroscientists who prefer command-line tools have no interface yet
  - **Recommendation:** When implementing CLI (post-M5), follow patterns from README examples with clear subcommands like:

    ```bash
    trodestrack simulate --duration 60 --output sim.pkl
    trodestrack filter ekf --imu imu.npy --camera cam.npy --output filtered.pkl
    trodestrack qc --data filtered.pkl --report qc_report.html
    ```

---

## UX Enhancements (Consider)

### API Ergonomics

- [ ] **Add convenience constructors for common scenarios**
  - `RatIMUSimConfig.quick_test(duration_s=10)` - minimal setup for testing
  - `RatIMUSimConfig.high_dropout(dropout_prob=0.4)` - stress testing
  - `EKFConfig.conservative()` - low-confidence, robust settings
  - `EKFConfig.aggressive()` - high-confidence, fast convergence

- [ ] **Provide example gallery** with copy-paste snippets
  - Already excellent examples in `examples/01_*.py` and `examples/02_*.py`
  - Enhancement: Add `examples/gallery.md` with annotated code blocks for common tasks:
    - "How to run EKF with custom noise parameters"
    - "How to generate a QC report"
    - "How to handle long camera dropouts"

### Documentation Flow

- [ ] **Add "Quickstart" tutorial** that guides users through full workflow
  - Current docs assume familiarity with Kalman filters
  - Opportunity: 5-minute guide from simulation → filtering → QC report
  - Target audience: Neuroscientist with basic Python knowledge

- [ ] **Glossary of domain terms** in README or separate GLOSSARY.md
  - Examples: "specific force vs inertial acceleration", "NIS/NEES", "RTS smoother"
  - Helps bridge gap between neuroscience and estimation theory audiences

### Progress Feedback

- [ ] **Add progress bars for long operations** (filter, smoother)
  - Dependency: `tqdm` (already in dependencies!)
  - Example: Filtering 60-second trajectory with 200 Hz IMU takes ~10 seconds
  - User benefit: Confidence that code is running, not hung
  - Implementation pattern:

    ```python
    from tqdm import tqdm

    for i in tqdm(range(n_cam), desc="Filtering"):
        # ... filter step
    ```

### Configuration Validation Enhancement

- [ ] **Add configuration "linting" tool** that checks for common pitfalls
  - Example checks:
    - `led_distance` set but `use_second_led=False` → warning
    - `mahalanobis_threshold_prob > 0.999` → warning (too permissive)
    - `process_noise_pos` orders of magnitude different from `measurement_noise_pos` → flag mismatch
  - Could be a method: `config.validate_with_warnings()`

---

## Domain-Specific Analysis

### API Design

- **Intuitiveness:** ★★★★★ Excellent
  - Function names clearly describe purpose: `extended_kalman_filter()`, `simulate_rat_imu()`, `compute_position_rmse()`
  - Parameter names use domain language: `led_distance`, `gyro_noise_density`, `heading`
  - Return types are well-structured NamedTuples with descriptive field names

- **Consistency:** ★★★★★ Excellent
  - All timestamps use seconds (not milliseconds or frames)
  - All distances use meters (not cm or pixels)
  - All angles use radians (not degrees), with helpful conversion in examples
  - Config classes all use dataclass pattern with validation

- **Key findings:**
  - ✅ Filter functions return NamedTuple results with clear semantics (`filtered_means`, `predicted_means`, `marginal_loglik`)
  - ✅ Simulation output is dictionary with self-documenting keys (`X_truth`, `U_imu`, `Z_cam_led1`)
  - ✅ Immutability enforced: filters create new configs rather than mutating input (see `ekf.py:858-869`)
  - ✅ Consistent `mask` pattern: boolean arrays where True = valid data

### Configuration

- **Clarity:** ★★★★☆ Very Good
  - Parameter names are descriptive: `cam_dropout_prob` not `p_dropout`
  - Units included in docstrings: "duration_s: float (seconds)", "cam_sigma_m: float (meters)"
  - Default values are scientifically motivated (from PRD requirements)

- **Discoverability:** ★★★☆☆ Good, with room for improvement
  - Defaults work well for typical use cases
  - Validation messages guide users to correct ranges
  - **Opportunity:** No "presets" or example configs in code (only in examples/)

- **Key findings:**
  - ✅ Validation is comprehensive with helpful error messages
  - ✅ Backward compatibility: `vel_drag` → `drag_fwd`/`drag_lat` transition handled gracefully
  - ⚠️ Some parameters require domain expertise: `process_noise_gyro_bias`, `dropout_q_bias_multiplier`
  - 💡 Consider: Configuration builder pattern for complex scenarios

### Error Messages

- **Actionability:** ★★★★☆ Very Good
  - Validation errors include examples of correct usage
  - Range violations state the expected range clearly
  - Shape mismatches show both actual and expected shapes

- **Clarity:** ★★★★★ Excellent
  - Messages use plain language, not jargon
  - Multi-line formatting improves readability (see `rat_imu.py:267-271`)
  - Context provided: WHY the constraint exists, not just WHAT failed

- **Key findings:**
  - ✅ **Outstanding example** (`rat_imu.py:315-343`): LED swap validation explains both the constraint AND shows usage

    ```python
    raise ValueError(
        f"led_swap_mode must be 'per_frame' or 'persistent', got '{self.led_swap_mode}'.\n"
        f"Example: led_swap_mode='persistent' for event-based swaps"
    )
    ```

  - ✅ Shape mismatch errors show dimensions: `f"Shape mismatch: true {positions_true.shape} vs est {positions_est.shape}"`
  - ⚠️ Some low-level errors (Cholesky failures) could provide more guidance on root cause
  - 💡 **Missing:** Error codes/categories for programmatic handling (e.g., `E_INVALID_CONFIG`, `E_SINGULAR_MATRIX`)

### Documentation (Docstrings)

- **Completeness:** ★★★★★ Excellent
  - Every public function has comprehensive docstrings
  - Parameters include types, units, and valid ranges
  - Return values clearly documented with shapes
  - Mathematical formulas included where relevant (e.g., `ekf.py:208-233`)

- **User-friendliness:** ★★★★★ Excellent
  - Examples provided for non-trivial functions
  - Cross-references to PRD and literature
  - Algorithm descriptions at high level before implementation details
  - Target audience clearly considered (neuroscientists AND engineers)

- **Key findings:**
  - ✅ **Best practice example** (`qa/metrics.py:22-75`): Includes docstring, type hints, example with expected output, AND physical interpretation
  - ✅ Module-level docstrings set context (e.g., `rat_imu.py:1-21`)
  - ✅ Inline comments explain "why", not "what" (e.g., bias freezing logic in `ekf.py:428-438`)
  - ✅ References to literature (Särkkä 2013) help users dive deeper
  - 💡 **Opportunity:** Add "See Also" sections linking related functions

---

## Excellent UX Patterns

### 1. **Configuration Validation with Examples**

The `RatIMUSimConfig.__post_init__()` validation is a masterclass in user-friendly error handling:

```python
if self.duration_s <= 0:
    raise ValueError(
        f"Simulation duration must be positive, got {self.duration_s}s.\n"
        f"Example: duration_s=60.0 (60 seconds)"
    )
```

**Why it works:**

- Shows the bad value (helps with debugging)
- States the constraint clearly
- Provides a concrete example
- Uses natural language, not technical jargon

### 2. **Immutable Configuration Pattern**

Filters never mutate input configs, instead creating new instances:

```python
# From ekf.py:860-869
if ekf_config.led_distance is None:
    estimated_led_distance = estimate_led_spacing(...)
    config_dict = {k: v for k, v in ekf_config.__dict__.items()}
    config_dict["led_distance"] = estimated_led_distance
    config_for_filter = EKFConfig(**config_dict)
else:
    config_for_filter = ekf_config
```

**Benefits:**

- Prevents spooky action at a distance
- Makes testing easier (no side effects)
- Aligns with JAX's functional paradigm

### 3. **Self-Documenting Return Types**

Use of NamedTuples instead of bare tuples:

```python
class EKFResult(NamedTuple):
    """EKF filtering result."""
    filtered_means: jnp.ndarray  # (N_cam, 8)
    filtered_covariances: jnp.ndarray  # (N_cam, 8, 8)
    predicted_means: jnp.ndarray
    predicted_covariances: jnp.ndarray
    marginal_loglik: float
    estimated_led_distance: float | None
```

**Why it's excellent:**

- IDE autocomplete works perfectly
- No "which index is which?" confusion
- Shape annotations prevent dimension errors
- Optional fields clearly marked with `| None`

### 4. **Units Everywhere**

Consistent unit documentation prevents errors:

```python
def compute_position_rmse(
    positions_true: NDArray[np.float64],  # (N, 2) in meters
    positions_est: NDArray[np.float64],   # (N, 2) in meters
    mask: NDArray[np.bool_] | None = None,
) -> float:  # RMSE in meters
```

**Value:**

- No confusion about cm vs meters
- Makes scientific papers easier to write (units match)
- Prevents common integration errors

### 5. **Examples in Docstrings**

Every complex function includes runnable examples:

```python
Example:
    >>> true_pos = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> est_pos = np.array([[0.1, 0.1], [1.1, 1.1]])
    >>> rmse = compute_position_rmse(true_pos, est_pos)
    >>> print(f"{rmse:.4f} m")
    0.1414 m
```

**Benefits:**

- Can be copy-pasted to try feature
- Serves as unit test (with doctest)
- Shows expected output format

### 6. **Domain-Aligned Terminology**

Code uses neuroscience vocabulary, not generic terms:

- `led_distance` not `marker_spacing`
- `heading` not `yaw` or `orientation`
- `specific_force` vs `inertial_acceleration` distinction
- `arena_w`, `arena_h` not `width`, `height`

**Why this matters:** Neuroscientists can map code directly to their experimental setup without translation.

### 7. **Comprehensive Simulation Output**

The `simulate_rat_imu()` return dictionary includes everything needed for debugging:

- Ground truth: `X_truth`, `yaw_rate_truth`, `accel_body_truth`
- Measurements: `U_imu`, `Z_cam_led1`, `Z_cam_led2`
- Metadata: `swap_applied`, `confidence_led1`, `mask_cam`
- Configuration: `config` (full provenance!)

**Value:** Users can validate every step of the pipeline against known truth.

### 8. **Progressive Disclosure in Examples**

Examples build in complexity:

1. `01_simple_simulations.py` - Analytic trajectories (stationary, constant velocity, circular)
2. `02_rat_imu_simulation.py` - Realistic rat motion with 5 scenarios of increasing complexity

Each example is self-contained and runnable with `uv run python examples/XX_*.py`.

---

## Target User Perspective

### For Neuroscience Researchers

**Strengths:**

- ✅ Terminology matches experimental domain (LED, arena, rat, heading)
- ✅ Default parameters work for typical SpikeGadgets hardware
- ✅ Examples produce publication-quality plots (Tufte/Gelman principles)
- ✅ Validation catches common experimental mistakes (e.g., swapping Hz and timesteps)

**Friction Points:**

- ⚠️ Must write Python code (no GUI or CLI yet)
- ⚠️ Filter tuning requires understanding Q/R matrices (common in Kalman filtering)
- ⚠️ No "import my Trodes data" function yet (planned I/O module)

**Recommendations:**

1. Add "Cookbook" with recipes for common tasks
2. Provide template scripts for standard analyses
3. When CLI is added, prioritize simplicity over flexibility for default use case

### For Data Engineers

**Strengths:**

- ✅ Clean API boundaries with clear contracts
- ✅ Type hints enable static analysis
- ✅ JAX-compatible (can be JIT-compiled, GPU-accelerated)
- ✅ Immutable patterns prevent state bugs

**Friction Points:**

- ⚠️ No streaming/online mode (processes full batches)
- ⚠️ Limited configuration persistence (no save/load config to YAML)
- ⚠️ No plugin system for custom dynamics or measurement models

**Recommendations:**

1. Add config serialization: `config.to_yaml()`, `Config.from_yaml(path)`
2. Document JAX JIT compilation patterns for performance
3. Provide benchmarking utilities for profiling

---

## Recommended Next Steps

### Priority 1: Foundation (Before First Release)

1. **Implement CLI entry point** (`trodestrack:main`)
   - Start with `trodestrack simulate` and `trodestrack filter`
   - Follow Click or Typer framework for consistency
   - Include `--help` text with examples (like validation messages)

2. **Add configuration presets/factories**

   ```python
   FilterCoreConfig.for_typical_rat()
   FilterCoreConfig.for_high_precision()
   RatIMUSimConfig.quick_test()
   ```

3. **Improve error context for numerical failures**
   - Cholesky decomposition failures → suggest checking condition number
   - Singular covariance → point to process noise tuning guide

### Priority 2: Usability (First Minor Release)

4. **Create "Getting Started" tutorial**
   - 5-minute walkthrough: sim → filter → QC
   - Target audience: neuroscientist with basic Python
   - Include expected output at each step

5. **Add progress indicators for long operations**
   - Filter/smoother loops (use `tqdm`)
   - Simulation generation (optional, for long durations)

6. **Configuration validation enhancements**
   - `config.lint()` → warnings for common mistakes
   - `config.to_dict()` / `Config.from_dict()` for serialization

### Priority 3: Discoverability (Second Minor Release)

7. **Example gallery in documentation**
   - "How do I..." sections with copy-paste code
   - Cover 10-15 common scenarios
   - Link from README

8. **Glossary of domain terms**
   - Bridge neuroscience ↔ estimation theory vocabulary
   - Define NEES, NIS, specific force, RTS, etc.

9. **Video tutorial or animated walkthrough**
   - Show full workflow visually
   - Narrate key concepts (optional, low priority)

### Longer-term Enhancements

10. **Configuration builder DSL**

    ```python
    config = (ConfigBuilder()
        .for_rat_tracking()
        .with_high_dropout_robustness()
        .enable_heading_measurements()
        .build())
    ```

11. **Interactive tuning notebook**
    - Jupyter notebook that visualizes Q/R effects
    - Helps users build intuition for Kalman filtering

12. **Plugin system for custom models**
    - Allow researchers to inject custom dynamics
    - Provide abstract base classes for extensions

---

## Final Assessment

**Rating: EXCELLENT** ⭐⭐⭐⭐⭐

**Justification:**

The trodestrack codebase demonstrates **exceptional UX quality** across all evaluated dimensions:

1. **API Design (5/5):** Clean, intuitive, domain-aligned naming with excellent type hints
2. **Configuration (4/5):** Comprehensive with helpful defaults; could add presets
3. **Error Messages (4.5/5):** Clear, actionable, with examples; minor gaps in low-level errors
4. **Documentation (5/5):** Outstanding docstrings with examples, units, and cross-references
5. **Domain Fit (5/5):** Terminology perfectly matches neuroscience experimental context

**The project is USER_READY for Python API users.** Scientists can immediately:

- Generate realistic simulations with sensible defaults
- Run filters with minimal configuration
- Validate results with comprehensive QA metrics
- Debug issues using clear error messages

**Minor polish needed** for CLI tools (planned post-M5) and configuration convenience methods, but these are enhancements, not blockers.

**Why this is excellent UX:**

- **It respects user time:** Good defaults mean most users never need to tweak 19 parameters
- **It prevents errors:** Validation catches mistakes early with clear guidance
- **It builds trust:** Comprehensive testing, documentation, and ground truth validation
- **It's discoverable:** Examples show the path forward; docstrings answer questions inline
- **It's professional:** Consistent style, mature error handling, scientific rigor

**Recommendation:** Ship current Python API to users. Gather feedback on pain points to prioritize CLI and configuration enhancements. The foundation is rock-solid.

---

## Appendix: UX Review Methodology

**Files Reviewed:**

- Core API: `models/ekf.py`, `models/ukf.py`, `models/filter_common.py`, `runtime/offline.py`
- Simulation: `sim/rat_imu.py`, `sim/simple.py`
- QA: `qa/metrics.py`
- Configuration: All `*Config` dataclasses
- Examples: `examples/01_*.py`, `examples/02_*.py`
- Documentation: README.md, docstrings throughout

**Evaluation Criteria:**

1. **Clarity:** Can target users understand intent without extensive documentation?
2. **Consistency:** Do similar concepts use similar patterns?
3. **Error Handling:** Are errors actionable with recovery steps?
4. **Discoverability:** Can users find what they need?
5. **Safety:** Are dangerous operations protected?
6. **Domain Alignment:** Does terminology match neuroscience conventions?

**Target Users:**

- Primary: Neuroscience researchers analyzing tracking data
- Secondary: Data engineers building analysis pipelines

**Review Focus:** User-facing APIs, configuration, error messages, documentation (not internal implementation details)

---

*End of UX Review Report*

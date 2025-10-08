# TrodesTrack Planning & Test Strategy

## Current State (as of 2025-10-08)

### Completed Simulation Infrastructure

**Implemented Modules:**
- `sim/simple.py` - Analytic simulations (stationary, constant velocity, circular)
- `sim/rat_imu.py` - Realistic rat motion with full IMU physics
- `sim/utils.py` - Shared utilities and SimOut TypedDict
- `examples/01_simple_simulations.py` - Demonstrates analytic scenarios
- `examples/02_rat_imu_simulation.py` - Demonstrates realistic scenarios
- `tests/filters/test_vision_robustness.py` - **NEW**: Tier 3 vision robustness tests

**Recent Improvements:**
- ✅ Gravity & tilt physics (specific force = a_body - g_body)
- ✅ Independent LED dropout with correlation parameter
- ✅ Timestamp jitter without clipping bias
- ✅ Smooth speed saturation (JAX-compatible tanh)
- ✅ Unified API with SimOut TypedDict
- ✅ Comprehensive gravity tests (9 tests in test_rat_imu_gravity.py)
- ✅ **LED swap simulation** (`led_swap_prob` config parameter in rat_imu.py)
- ✅ **Vision robustness test suite** (16 tests covering LED swaps, occlusions, confidence, dual LEDs)

---

## 5-Tier Test Plan Evaluation

### Overall Assessment

**Strengths:**
- Excellent progressive complexity model (analytic → stochastic → full realism)
- Clear isolation of concerns (dynamics, sensors, vision, environment)
- Well-aligned with TDD principles (test-driven development)
- Explicit RMSE/NEES targets match PRD acceptance criteria
- Ablation studies enable root-cause analysis of filter failures
- Comprehensive coverage from unit to integration levels

**Key Insight:**
This plan effectively implements a "simulation ladder" where each tier builds on validated components from previous tiers. Failures at higher tiers can be debugged by dropping down to simpler scenarios, making root-cause analysis tractable.

---

### Tier 0: Analytic Baselines

**Proposed Scenarios:**
- Stationary (x=0, v=0, θ=const)
- Constant velocity (v=const, θ=const)
- Uniform circular turn (θ̇=const, |v|=const)

**Current Coverage:**
✅ **FULLY IMPLEMENTED** in `sim/simple.py` with examples in `01_simple_simulations.py`

**Strengths:**
- Perfect for validating filter math (Kalman gain, covariance propagation)
- Closed-form ground truth eliminates simulation bugs as confound
- Fast execution enables property-based testing (hypothesis library)
- Already demonstrates EKF should collapse to linear KF in these cases

**Gaps:**
- ⚠️ No explicit tests yet for:
  - Zero-velocity update when stationary (should reject IMU drift)
  - Covariance consistency (NEES χ² test with known distributions)
  - Numerical stability at very small velocities (heading observability)

**Recommendation:**
- **Priority: HIGH** - Add unit tests validating filter behavior on these scenarios
- Create `tests/filters/test_ekf_analytic.py` with:
  - `test_stationary_rejects_imu_drift()` - verify position doesn't drift
  - `test_constant_velocity_observability()` - NEES for v, θ estimation
  - `test_circular_turn_bias_estimation()` - gyro bias should converge
- Add property tests: "for any constant velocity, RMSE < 2cm after convergence"

**Estimated Effort:** 2-3 hours (tests only; simulation already done)

---

### Tier 1: Controlled Stochasticity

**Proposed Addition:**
- Ornstein-Uhlenbeck velocity/yaw rate (mean-reverting, smooth)
- Known autocorrelation → predictable filter performance
- Bridge between deterministic and fully stochastic

**Current Coverage:**
✅ **ALREADY IMPLEMENTED** in `rat_imu.py` (OU for velocity, yaw rate)

**Strengths:**
- Realistic smooth motion without chaotic trajectories
- Tunable correlation time enables sensitivity analysis
- OU parameters (σ, λ) directly map to Q matrix tuning
- Differentiable dynamics support gradient-based Q optimization

**Gaps:**
- ⚠️ No tests validating OU parameter impact on filter performance
- ⚠️ No comparison of estimated vs true OU statistics (stationarity check)
- ⚠️ Missing "known-OU" scenario with matched process noise

**Recommendation:**
- **Priority: MEDIUM** - Add diagnostic tests, not critical for v1
- Create `tests/sim/test_ou_validation.py`:
  - `test_ou_matches_theoretical_variance()` - verify σ²/(2λ) steady-state
  - `test_ou_autocorrelation()` - check exp(-λτ) decay
- Create `tests/filters/test_ekf_ou_matching.py`:
  - `test_matched_process_noise_nees()` - NEES ≈ χ² when Q matches OU
  - `test_mismatched_q_degrades_gracefully()` - 2× Q error → bounded RMSE growth

**Estimated Effort:** 3-4 hours

---

### Tier 2: Sensor Realism

**Proposed Features:**
- Bias random walks (gyro, accel)
- White noise densities (discrete-time sampling)
- Timestamp jitter & latency
- Downsampling (30 kHz → 1 kHz IMU, 30 Hz camera)

**Current Coverage:**
✅ **FULLY IMPLEMENTED** in `rat_imu.py`:
- Bias RW with configurable σ_bias
- Noise densities converted via `density_to_sample_std()`
- Jitter + latency (no clipping bias)
- High-rate IMU generation (20 kHz default)

**Strengths:**
- Matches SpikeGadgets IMU specs (noise densities from datasheet)
- Bias estimation is core PRD requirement (state augmentation)
- Jitter fix (removing `np.clip`) ensures unbiased timestamp distribution
- Already includes gravity + tilt (added in recent work)

**Gaps:**
- ⚠️ No systematic tests for bias observability vs trajectory richness
  - Stationary → gyro bias unobservable
  - Straight line → lateral accel bias unobservable
  - Need "excitation" metrics (Grammian rank, condition number)
- ⚠️ Missing validation of noise density conversions against known distributions
- ⚠️ No timestamp jitter robustness tests (extreme jitter, latency >> dt_cam)

**Recommendation:**
- **Priority: HIGH** - Bias estimation is critical for PRD
- Create `tests/filters/test_bias_observability.py`:
  - `test_stationary_gyro_bias_unobservable()` - P_bias should grow
  - `test_circular_motion_gyro_bias_converges()` - P_bias should shrink
  - `test_straight_line_lateral_bias_unobservable()` - check Grammian rank
- Create `tests/sim/test_noise_statistics.py`:
  - `test_imu_noise_matches_density()` - empirical PSD vs theoretical
  - `test_bias_rw_variance_growth()` - σ²(t) = σ_bias² × t
- Add `tests/filters/test_timestamp_robustness.py`:
  - `test_large_jitter_bounded_error()` - jitter = 2×dt_cam → RMSE < threshold
  - `test_latency_compensation()` - verify prediction to t_cam_obs

**Estimated Effort:** 5-6 hours

---

### Tier 3: Vision Realism

**Proposed Features:**
- Dropout (occlusions, reflections)
- Confidence-dependent noise
- Dual LEDs (heading measurement, swap detection)
- LED spacing validation (Mahalanobis gating)

**Current Coverage:**
✅ **FULLY IMPLEMENTED** (as of Oct 8, 2025):
- Independent LED dropout with correlation parameter (Gaussian copula)
- Confidence → noise scaling via `confidence_to_noise_scale()`
- Dual LED support with configurable spacing
- Individual `mask_led1`, `mask_led2` in SimOut
- ✅ **LED swap simulation** via `led_swap_prob` parameter (NEW)
- ✅ **Comprehensive test suite** in `test_vision_robustness.py` (16 tests, NEW)

**Strengths:**
- Correlation parameter enables testing partial vs full dropouts
- Confidence scaling matches DLC output distributions
- Separate masks allow testing "heading from velocity" fallback
- Realistic for SpikeGadgets LED tracking (reflections common)
- LED swap simulation validates filter's ability to detect mislabeling
- Tests cover long occlusions, confidence scaling, dual LED heading accuracy

**Test Coverage Summary (test_vision_robustness.py):**
- ✅ `TestLEDSwap`: 4 tests validating swap behavior, confidence swapping
- ✅ `TestLongOcclusion`: 3 tests validating dropout durations (including ≥3s occlusions)
- ✅ `TestConfidenceScaling`: 4 tests validating noise inflation, zero-confidence rejection
- ✅ `TestDualLEDHeading`: 5 tests validating heading observability, LED spacing, independent/correlated dropouts

**Remaining Gaps:**
- ⚠️ No confidence validation against empirical DLC data
  - Current `confidence_to_noise_scale()` is heuristic (not data-driven)
  - **Action:** Document as configurable, revisit with real data
- ⚠️ No arena reflection model (wall proximity → spurious detections)
  - **Action:** Defer to Tier 4 (arena physics)

**Status: COMPLETED (Oct 8, 2025)**
- ✅ LED swap implemented with `led_swap_prob` config parameter
- ✅ Comprehensive test suite validates all vision features
- ✅ Tests ready for filter implementation (currently validate simulation only)
- ⏭️ **Next:** When filter exists, add filter performance tests to same file

---

### Tier 4: Environment & Dynamics

**Proposed Features:**
- Arena boundaries (reflection, gating)
- Speed saturation (realistic turning radius)
- Heading-dependent drag (forward vs lateral)
- Wall proximity effects

**Current Coverage:**
✅ **PARTIALLY IMPLEMENTED**:
- Speed saturation via smooth tanh (JAX-compatible)
- Isotropic drag (`vel_drag` parameter)

**Gaps:**
- ⚠️ No arena boundaries in simulation
  - Rat can wander off maze → unrealistic for ~2m track
  - Need soft walls (reflection) or hard bounds (truncation)
- ⚠️ No anisotropic drag (rats move easier forward than sideways)
- ⚠️ Missing wall reflection model for vision
  - LED reflections off walls → spurious detections outside arena
- ⚠️ No turning radius constraint
  - Current model allows instant direction changes at high speed (unrealistic)

**Recommendation:**
- **Priority: MEDIUM** - Important for realism, but filters should handle without explicit model
- Add to `rat_imu.py`:
  ```python
  # Arena bounds (2D box)
  arena_bounds: tuple[float, float, float, float] = (0, 200, 0, 150)  # xmin, xmax, ymin, ymax

  # Anisotropic drag (forward vs lateral in body frame)
  drag_forward: float = 0.5  # Lower drag forward
  drag_lateral: float = 2.0  # Higher drag sideways

  # Wall reflection for vision (spurious detections)
  wall_reflection_prob: float = 0.05  # Per-frame probability near walls
  ```
- Create `tests/sim/test_arena_physics.py`:
  - `test_rat_stays_in_bounds()` - with soft walls, position stays in arena
  - `test_wall_reflection_outside_bounds()` - reflected detections have x,y outside arena
  - `test_anisotropic_drag()` - lateral velocity decays faster than forward
- Create `tests/filters/test_arena_gating.py`:
  - `test_out_of_bounds_rejected()` - measurements outside arena masked
  - `test_reflection_detection()` - Mahalanobis gate rejects wall reflections

**Estimated Effort:** 4-5 hours

---

### Tier 5: Full PRD-like Session

**Proposed Scenarios:**
- 30-minute session with PRD-realistic parameters
- Stress tests: long occlusions, rapid turns, handling events
- Ablations: IMU-only, vision-only, fusion vs baselines
- NEES validation across full session
- Benchmark: throughput (10× realtime CPU, 50× GPU), latency (≤33ms online)

**Current Coverage:**
❌ **NOT YET IMPLEMENTED**

**Strengths of Proposed Plan:**
- Direct validation of PRD acceptance criteria
- Ablations isolate sensor contributions (quantify fusion benefit)
- Benchmarks ensure production readiness
- Full session length exposes long-term drift, bias convergence

**Gaps:**
- Currently no filter implementation to test against
- No NEES computation utilities
- No benchmark harness (pytest-benchmark exists but not configured)
- No "handling event" simulation (rat picked up, placed back down)

**Recommendation:**
- **Priority: LOW for now** - Defer until filter implementation exists
- This tier validates the *filter*, not the *simulation*
- Prerequisites:
  1. Implement EKF/UKF in `models/ekf.py`, `models/ukf.py`
  2. Implement RTS smoother in `runtime/offline.py`
  3. Create `qa/metrics.py` with NEES, RMSE, NIS computations
- When ready, create `tests/integration/test_prd_session.py`:
  - `test_30min_session_rmse_under_2cm()` - PRD target
  - `test_5s_occlusion_drift_under_15cm()` - PRD target
  - `test_imu_only_vs_vision_only_vs_fusion()` - ablation study
  - `test_nees_consistency()` - NEES ∈ [χ²(0.025), χ²(0.975)] for 95% of windows
- Add `tests/benchmark/test_throughput.py`:
  - `test_offline_smoother_10x_realtime_cpu(benchmark)` - pytest-benchmark
  - `test_online_filter_latency_under_33ms(benchmark)`

**Estimated Effort:** 10-12 hours (after filter implementation)

---

## Recommended Implementation Priority

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Validate simulation correctness and analytic filter behavior

1. **Tier 0 Tests** (HIGH priority, 2-3 hours)
   - Filter validation on analytic baselines
   - NEES for stationary, constant velocity, circular
   - Property-based tests with hypothesis

2. **Tier 2 Noise Tests** (HIGH priority, 3-4 hours)
   - Noise density validation
   - Bias random walk statistics
   - Timestamp jitter edge cases

3. ~~**Tier 3 Occlusion Tests**~~ ✅ **COMPLETED (Oct 8, 2025)**
   - ✅ LED swap implementation + tests
   - ✅ Long dropout drift bounds (PRD criterion)
   - ✅ Confidence scaling validation

**Deliverable:** ✅ Simulation validated, ready for filter development

---

### Phase 2: Filter Development (Weeks 3-5)
**Goal:** Implement EKF/UKF and smoother

1. **EKF Implementation** (`models/ekf.py`)
   - Prediction step with IMU pre-integration
   - Update step with vision measurements
   - Bias state augmentation
   - Tested against Tier 0 scenarios

2. **UKF Implementation** (`models/ukf.py`)
   - Sigma-point prediction
   - Unscented transform for nonlinear dynamics
   - Compared to EKF on Tier 1 scenarios

3. **RTS Smoother** (`runtime/offline.py`)
   - Backward pass with stored F_k, Q_k
   - NEES validation on Tier 2 scenarios

**Deliverable:** Working filter + smoother, tested on synthetic data

---

### Phase 3: Robustness (Weeks 6-7)
**Goal:** Handle realistic edge cases

1. **Tier 2 Bias Observability** (MEDIUM priority, 5-6 hours)
   - Excitation metrics (Grammian)
   - Trajectory-dependent convergence tests

2. ~~**Tier 3 LED Swap**~~ ✅ **COMPLETED (Oct 8, 2025)**
   - ✅ Already completed in this session

3. **Tier 4 Arena Physics** (MEDIUM priority, 4-5 hours)
   - Soft walls, anisotropic drag
   - Wall reflection gating

**Deliverable:** Robust filter handling occlusions, swaps, arena bounds

---

### Phase 4: Integration & Benchmarking (Week 8)
**Goal:** PRD acceptance criteria

1. **Tier 5 Full Session Tests** (10-12 hours)
   - 30-min session RMSE < 2cm
   - 5s dropout drift < 15cm
   - Ablation studies

2. **Benchmarks** (pytest-benchmark)
   - Throughput: 10× realtime (CPU), 50× (GPU)
   - Latency: ≤33ms (online EKF)

3. **NEES Diagnostics** (`qa/metrics.py`)
   - Tuning guide based on NEES violations
   - Parameter sensitivity analysis

**Deliverable:** Production-ready filter meeting all PRD targets

---

## Missing from Tier Plan

### Additional Test Categories to Consider:

1. **Corner Cases Not Covered:**
   - Zero-velocity updates (stationary detection)
   - Heading unobservability (single LED, no motion)
   - IMU saturation (very high angular rates)
   - Camera frame drops (not just occlusions, but missing timestamps)
   - Negative confidence values (DLC sometimes outputs these)

2. **Numerical Stability:**
   - Covariance matrix symmetry preservation
   - Positive-definiteness enforcement (Cholesky factorization)
   - Ill-conditioned updates (very low measurement noise)
   - JAX gradient numerical stability (for future gradient-based tuning)

3. **Reproducibility:**
   - Deterministic RNG seeding (already done in current sim)
   - Hardware-specific floating-point differences (CPU vs GPU)
   - JAX JIT compilation consistency

4. **Real Data Validation (outside tier plan scope):**
   - Hand-labeled ground truth for 1-2 real sessions
   - Cross-validation against external tracking (OptiTrack, etc.)
   - DLC confidence distribution matching

---

## Alignment with PRD

### Acceptance Criteria Coverage:

| PRD Criterion | Tier | Status | Gap |
|---------------|------|--------|-----|
| RMSE ≤2cm position | Tier 5 | ❌ Pending filter impl | None |
| RMSE ≤10cm/s velocity | Tier 5 | ❌ Pending filter impl | None |
| RMSE ≤7° heading | Tier 5 | ❌ Pending filter impl | None |
| 5s dropout → ≤15cm drift | Tier 3 | ✅ Sim ready, tests needed | Add `test_long_dropout_drift()` |
| Offline 10× realtime (CPU) | Tier 5 | ❌ Pending filter impl | None |
| Offline 50× realtime (GPU) | Tier 5 | ❌ Pending filter impl | None |
| Online ≤33ms latency | Tier 5 | ❌ Pending filter impl | None |
| Deterministic runs | All | ✅ Implemented | None |

**Conclusion:** Tier plan excellently covers all PRD criteria. Current simulation infrastructure is ready for Tier 0-3 tests. Tier 4-5 require filter implementation first.

---

## Recommendations Summary

### Immediate Next Steps (This Week):

1. **Add Tier 0 analytic filter tests** - validates filter math without simulation confounds
2. **Add Tier 3 LED swap** - critical for heading robustness
3. **Add long dropout test** - directly tests PRD criterion

### Before Filter Implementation:

1. **Complete Tier 2 noise validation** - ensures simulation matches hardware specs
2. **Add NEES utilities** in `qa/metrics.py` - needed for all filter tests
3. **Document test strategy** - this PLANNING.md serves as blueprint

### After Filter Implementation:

1. **Tier 1 OU matching tests** - validates process noise tuning
2. **Tier 4 arena physics** - optional for v1, but improves realism
3. **Tier 5 full PRD validation** - final acceptance testing

---

## Open Questions

1. **Should Tier 4 arena physics be in simulation or filter?**
   - Simulation: More realistic trajectories, but couples physics to test data
   - Filter: Arena bounds as measurement gating, but simulation unrealistic
   - **Recommendation:** Both - soft walls in simulation, hard gating in filter

2. **How to validate confidence scaling without real DLC data?**
   - Current `confidence_to_noise_scale()` is heuristic
   - Could fit to synthetic data with known noise, but won't match real DLC
   - **Recommendation:** Document as "placeholder pending real data", make configurable

3. **Should LED swap be per-frame random or event-based?**
   - Per-frame: Easier to implement, but unrealistic (swaps are persistent)
   - Event-based: Realistic (swap on close pass, persist until correction), but complex
   - **Recommendation:** Start with per-frame for Tier 3, add event-based in Tier 5

4. **GPU benchmarking infrastructure?**
   - JAX can run on CPU or GPU, but pytest-benchmark needs explicit device selection
   - Need CI with GPU runners (GitHub Actions doesn't have free GPU)
   - **Recommendation:** CPU benchmarks in CI, GPU benchmarks as manual script

---

## Conclusion

The proposed 5-tier test plan is **excellent and well-structured**. It provides:
- Clear progression from simple to complex
- Isolation of failure modes
- Direct mapping to PRD acceptance criteria
- TDD-compatible workflow (tests before implementation)

**Current simulation infrastructure (rat_imu.py, simple.py, utils.py) fully supports Tiers 0-3** and partially supports Tier 4. The main gaps are:
1. Missing test files (easy to add)
2. LED swap feature (small addition to rat_imu.py)
3. Arena physics (optional for v1)

**Recommended approach:**
- Implement Tier 0-3 tests immediately (simulation is ready)
- Use these tests to validate filter implementation (TDD)
- Add Tier 4-5 tests after filter works on simpler scenarios

This plan sets the project up for rigorous validation and high confidence in production deployment.

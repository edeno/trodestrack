# Examples

This section contains pedagogical examples demonstrating the TrodesTrack sensor-fusion tracking system. Each example is standalone and teaches specific concepts.

## Learning Path

The examples are numbered to form a progressive learning path:

| Example | Focus | Duration | Prerequisites |
|---------|-------|----------|---------------|
| 01 | Simulations | ~5s | None |
| 02 | Realistic IMU | ~10s | Example 01 |
| **03** | **EKF Basics** | **~5s** | **Example 02** |
| 03b | Plot Utilities | ~5s | Example 03 |
| 04 | UKF vs EKF | ~10s | Example 03 |
| 05 | EKF Dropouts | ~10s | Example 03 |
| 06 | UKF Dropouts | ~10s | Example 04 |
| 07 | Smoothing | ~15s | Example 03 |
| 08 | QA Reports | ~2s | Any filter example |

!!! tip "Start Here"
    If you're new to TrodesTrack, start with **Example 03** after completing the [Quick Start](../getting-started/quickstart.md).

## Getting Started: Simulations (01-02)

### Example 01: Simple Simulations

Learn analytic simulation fundamentals.

```bash
uv run python examples/01_simple_simulations.py
```

**Topics:**

- Stationary, constant velocity, and circular motion scenarios
- Ground truth validation (zero-variance checks)
- Camera measurement noise simulation
- IMU data generation

**Output:** 1 PNG showing all 3 scenarios side-by-side

---

### Example 02: Realistic Rat IMU Simulation

Learn realistic rat IMU simulation.

```bash
uv run python examples/02_rat_imu_simulation.py
```

**Topics:**

- Ornstein-Uhlenbeck motion dynamics
- IMU tilt, drag, and bias random walks
- Dual-LED configuration for heading measurements
- Camera dropout modeling with confidence scores
- LED swaps and occlusion simulation

**Output:** 5 PNGs demonstrating different simulation features

## Filter Fundamentals (03-04)

### Example 03: EKF Basic Scenarios

:star: **Start here for filtering!**

Learn Extended Kalman Filter fundamentals.

```bash
uv run python examples/03_ekf_basic_scenarios.py
```

**Topics:**

- EKF on stationary, constant velocity, and circular motion
- Position/velocity/heading accuracy vs PRD targets
- Bias observability (why gyro bias needs rotation)
- Filter consistency (NEES interpretation)
- Innovation statistics

**Output:** 3 comprehensive 9-panel diagnostic PNGs + detailed console metrics

!!! note "Key Learning"
    **Observability Insight:** Gyro bias is only observable during rotation. Stationary and straight-line motion cannot distinguish bias from heading drift.

---

### Example 03b: Using Plot Utilities

:star: **Recommended for analysis!**

Learn to use built-in plotting utilities and state layouts.

```bash
uv run python examples/03b_using_plot_utilities.py
```

**Topics:**

- **State Layout System:** Dimension-agnostic state extraction
- Using `qa.plot_*` functions instead of manual matplotlib code
- Covariance extraction and uncertainty visualization
- DRY principle for plotting

**Output:** 5 publication-quality plots using trodestrack utilities

!!! tip "Best Practice"
    Always use `layout = get_layout(config.state_mode)` and extract states with `result.filtered_means[:, layout.pos_idx]` instead of hardcoded indices.

---

### Example 04: UKF Basic Scenarios

Compare UKF vs EKF.

```bash
uv run python examples/04_ukf_basic_scenarios.py
```

**Topics:**

- Sigma-point transforms vs Jacobian linearization
- Computational cost analysis
- Accuracy comparison on same scenarios
- NEES-based filter consistency comparison

**Output:** 3 comparison PNGs + side-by-side metrics tables

!!! note "Verdict"
    EKF won 6/9 metrics vs UKF's 3/9. UKF is ~1-5x slower. **Recommendation:** Start with EKF; switch to UKF only if EKF fails to meet accuracy requirements.

## Robustness (05-06)

### Example 05: EKF with Dropouts

Test EKF robustness to camera dropouts.

```bash
uv run python examples/05_ekf_with_dropouts.py
```

**Topics:**

- 10%, 20%, and 30% camera dropout simulation
- IMU-only periods stress testing
- Adaptive process noise during dropout

---

### Example 06: UKF with Dropouts

Test UKF robustness to camera dropouts.

```bash
uv run python examples/06_ukf_with_dropouts.py
```

## Advanced Techniques (07-08)

### Example 07: Smoother Demonstration

Learn RTS/IEKS smoothing for drift reduction.

```bash
uv run python examples/07_smoother_demonstration.py
```

**Topics:**

- Forward filter vs backward smoother
- 5-second dropout scenario
- Filter drift (~1.5-2.0 m) vs smoothed drift (~0.5-0.7 m)
- IEKS (Iterated Extended Kalman Smoother)

**Output:** Comparison video showing improvement

!!! success "Smoother Benefit"
    On 5s dropout, smoother achieves **3x drift reduction** by using future vision measurements to correct past estimates.

---

### Example 08: QA Report Generation

Learn professional QA reporting workflow.

```bash
uv run python examples/08_qa_report_generation.py
```

**Topics:**

- Comprehensive PDF report generation
- All PRD metrics with pass/fail indicators
- NEES/NIS consistency checks
- Time series and trajectory visualizations

**Output:** `example_qa_report.pdf` with full diagnostics

## Key Concepts

### Observability

**Example 03** teaches which states are observable in different motion patterns:

- **Stationary:** Position only
- **Straight line:** Position + velocity + accel bias
- **Circular:** All states including gyro bias

### Filter Consistency

**Examples 03-04** use NEES (Normalized Estimation Error Squared) to check if the filter's uncertainty estimates are honest:

- NEES ~ state_dim = filter is consistent (e.g., ~2 for position-only NEES, ~8 for full 8D state)
- NEES well below state_dim = underconfident (covariance too large)
- NEES well above state_dim = overconfident (covariance too small)

### Computational Tradeoffs

**Example 04** demonstrates:

- EKF: 1 linearization point -> fast
- UKF: 17 sigma points -> 1-5x slower but handles nonlinearity better
- **Verdict:** EKF is sufficient for most scenarios

## Running All Examples

```bash
# Run all examples in sequence
for f in examples/0*.py; do
    echo "=== Running $f ==="
    uv run python "$f"
done
```

## Output Locations

- **PNGs:** `output/examples/` directory (created automatically)
- **Videos:** `diagnostics/videos/` directory
- **PDFs:** Same directory as the example script

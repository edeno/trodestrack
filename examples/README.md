# TrodesTrack Examples

This directory contains pedagogical examples demonstrating the TrodesTrack sensor-fusion tracking system. Each example is standalone and teaches specific concepts through clear code and educational output.

## 🎯 Learning Path

The examples are numbered to form a progressive learning path:

### **Getting Started: Simulations (01-02)**

Start here to understand how synthetic data is generated and validated.

#### [01_simple_simulations.py](01_simple_simulations.py)
**Learn:** Analytic simulation fundamentals
**Duration:** ~5 seconds
**Topics:**
- Stationary, constant velocity, and circular motion scenarios
- Ground truth validation (zero-variance checks)
- Camera measurement noise simulation
- IMU data generation

```bash
uv run python examples/01_simple_simulations.py
```

**Output:** 1 PNG showing all 3 scenarios side-by-side

---

#### [02_rat_imu_simulation.py](02_rat_imu_simulation.py)
**Learn:** Realistic rat IMU simulation
**Duration:** ~10 seconds
**Topics:**
- Ornstein-Uhlenbeck motion dynamics
- IMU tilt, drag, and bias random walks
- Dual-LED configuration for heading measurements
- Camera dropout modeling with confidence scores
- LED swaps and occlusion simulation

```bash
uv run python examples/02_rat_imu_simulation.py
```

**Output:** 5 PNGs demonstrating different simulation features

---

### **Filter Fundamentals: Clean Conditions (03-04)**

Learn how EKF and UKF perform under ideal conditions (no dropouts).

#### [03_ekf_basic_scenarios.py](03_ekf_basic_scenarios.py) ⭐ **Start Here for Filtering**
**Learn:** Extended Kalman Filter fundamentals
**Duration:** ~5 seconds
**Topics:**
- EKF on stationary, constant velocity, and circular motion
- Position/velocity/heading accuracy vs PRD targets
- Bias observability (why gyro bias needs rotation)
- Filter consistency (NEES interpretation)
- Innovation statistics

```bash
uv run python examples/03_ekf_basic_scenarios.py
```

**Output:** 3 comprehensive 9-panel diagnostic PNGs + detailed console metrics

**Key Learning:**
> **Observability Insight:** Gyro bias is only observable during rotation. Stationary and straight-line motion cannot distinguish bias from heading drift.

---

#### [03b_using_plot_utilities.py](03b_using_plot_utilities.py) ⭐ **Recommended for Analysis**
**Learn:** Using built-in plotting utilities and state layouts
**Duration:** ~5 seconds
**Topics:**
- **State Layout System:** Dimension-agnostic state extraction (works with 5D, 8D, 10D, 15D states)
- Using `qa.plot_*` functions instead of manual matplotlib code
- Covariance extraction and uncertainty visualization
- DRY principle for plotting (Don't Repeat Yourself)
- Comparing code complexity: utilities vs manual approach

```bash
uv run python examples/03b_using_plot_utilities.py
```

**Output:** 5 publication-quality plots using trodestrack utilities

**Key Learning:**
> **Best Practice:** Always use `layout = get_layout(config.state_mode)` and extract states with `result.filtered_means[:, layout.pos_idx]` instead of hardcoded indices like `[:, 0:2]`. This makes code work with any state dimension!

---

#### [04_ukf_basic_scenarios.py](04_ukf_basic_scenarios.py)
**Learn:** UKF vs EKF comparison
**Duration:** ~10 seconds
**Topics:**
- Sigma-point transforms vs Jacobian linearization
- Computational cost analysis (timing measurements)
- Accuracy comparison on same scenarios
- When UKF's advantages justify the cost
- NEES-based filter consistency comparison

```bash
uv run python examples/04_ukf_basic_scenarios.py
```

**Output:** 3 comparison PNGs + side-by-side metrics tables

**Key Learning:**
> **Verdict:** EKF won 6/9 metrics vs UKF's 3/9. UKF is ~1-5× slower. **Recommendation:** Start with EKF; switch to UKF only if EKF fails to meet accuracy requirements.

---

#### [03b_using_plot_utilities.py](03b_using_plot_utilities.py)
**Learn:** Build dimension-agnostic analysis code with `qa.plots` helpers
**Topics:** `get_layout(cfg.state_mode)`, layout-driven slicing, covariance
ellipses, error time series.

```bash
uv run python examples/03b_using_plot_utilities.py
```

---

### **Robustness to Camera Dropouts (05-06)**

#### [05_ekf_with_dropouts.py](05_ekf_with_dropouts.py)
**Learn:** EKF behavior under 10%, 20%, and 30% camera dropout.
**Topics:** Adaptive Q during dropout, bias-freeze policy, IMU-only drift.

```bash
uv run python examples/05_ekf_with_dropouts.py
```

#### [06_ukf_with_dropouts.py](06_ukf_with_dropouts.py)
**Learn:** Same dropout stress-test, UKF variant, for direct comparison
with example 05.

```bash
uv run python examples/06_ukf_with_dropouts.py
```

---

### **Advanced Techniques (07-08)**

Smoother and quality assurance workflows.

#### [07_smoother_demonstration.py](07_smoother_demonstration.py)
**Learn:** RTS/IEKS smoothing for drift reduction
**Duration:** ~15 seconds
**Topics:**
- Forward filter vs backward smoother
- 5-second dropout scenario
- Filter drift (~1.5-2.0 m) vs smoothed drift (~0.5-0.7 m)
- IEKS (Iterated Extended Kalman Smoother)
- Bias estimate correction via backward pass

```bash
uv run python examples/07_smoother_demonstration.py
```

**Output:** Comparison video showing improvement

**Key Learning:**
> **Smoother Benefit:** On 5s dropout, smoother achieves **3× drift reduction** by using future vision measurements to correct past estimates.

---

#### [08_qa_report_generation.py](08_qa_report_generation.py)
**Learn:** Professional QA reporting workflow
**Duration:** ~2 seconds
**Topics:**
- Comprehensive PDF report generation
- All PRD metrics with pass/fail indicators
- NEES/NIS consistency checks
- Time series and trajectory visualizations
- Filter configuration documentation

```bash
uv run python examples/08_qa_report_generation.py
```

**Output:** `example_qa_report.pdf` with full diagnostics

---

## 📊 Quick Reference

| Example | Focus | Output | Duration | Prerequisites |
|---------|-------|--------|----------|---------------|
| 01 | Simulations | 1 PNG | ~5s | None |
| 02 | Realistic IMU | 5 PNGs | ~10s | Example 01 |
| 03 ⭐ | **EKF Basics** | **3 PNGs** | **~5s** | **Example 02** |
| 03b | Plot utilities | PNGs | ~5s | Example 03 |
| 04 | UKF vs EKF | 3 PNGs | ~10s | Example 03 |
| 05 | EKF dropouts | PNGs | ~10s | Example 03 |
| 06 | UKF dropouts | PNGs | ~10s | Example 04 |
| 07 | Smoothing | 1 video | ~15s | Example 03 |
| 08 | QA Reports | 1 PDF | ~2s | Any filter example |

---

## 🎓 Key Concepts by Example

### Observability
**Example 03** teaches which states are observable in different motion patterns:
- **Stationary:** Position only
- **Straight line:** Position + velocity + accel bias
- **Circular:** All states including gyro bias

### Filter Consistency
**Examples 03-04** use NEES (Normalized Estimation Error Squared) to check if the filter's uncertainty estimates are honest:
- NEES ≈ state_dim (e.g. 8 for 8D state) → filter is consistent
- NEES > upper chi-square bound → overconfident (covariance too small)
- NEES < lower chi-square bound → underconfident (covariance too large)

### Computational Tradeoffs
**Example 04** demonstrates:
- EKF: 1 linearization point → fast
- UKF: 17 sigma points → 1-5× slower but handles nonlinearity better
- **Verdict:** EKF is sufficient for most scenarios

---

## 🛠️ Development Workflow

### Running Examples

All examples are standalone and use the `uv` package manager:

```bash
# Single example
uv run python examples/03_ekf_basic_scenarios.py

# All examples in sequence (for validation)
for f in examples/0*.py; do
    echo "=== Running $f ==="
    uv run python "$f"
done
```

### Example Output Locations

Each example writes to a different path. Quick reference:

- **Example 01** (`01_simple_simulations.py`) → `examples/01_simple_simulations.png` (next to the script).
- **Example 02** (`02_rat_imu_simulation.py`) → five PNGs in the current working directory (`02_basic_sim.png`, `02_two_led_sim.png`, `03_confidence_sim.png`, `04_noise_validation.png`, `05_vision_robustness.png`).
- **Examples 03 / 03b / 04 / 05 / 06** → PNGs next to each script (`examples/<name>.png`).
- **Example 07** (`07_smoother_demonstration.py`) → `output/dropout_smoother_comparison.mp4` (resolved relative to the process working directory).
- **Example 08** (`08_qa_report_generation.py`) → `example_qa_report.pdf` in the current working directory.

---

## 🤝 Contributing

When adding new examples:
1. Follow the pedagogical template (see Examples 03-04)
2. Include learning objectives in the docstring
3. Add educational console output explaining results
4. Create comprehensive visualizations (multi-panel layouts)
5. Compare metrics against PRD targets
6. Update this README with the new example

---

**Happy Learning! 🚀**

If you have questions or suggestions for improving these examples, please open an issue in the repository.

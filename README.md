# trodestrack

**Sensor-fused 2D rat tracking with JAX EKF/UKF for neuroscience research**

trodestrack combines video tracking (Trodes LEDs and/or DeepLabCut keypoints) with high-rate IMU data from SpikeGadgets headstages to provide accurate position, velocity, and heading estimates for freely-moving rats on behavioral mazes.

## Features

- **Sensor Fusion**: Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for combining video (~30 Hz) and IMU (~20-30 kHz) measurements
- **Online & Offline Processing**: Real-time filtering and RTS smoothing for offline analysis
- **Robust Handling**: Occlusions, LED swaps, reflections, and sensor dropout
- **JAX-Accelerated**: High-performance implementation using JAX with GPU support
- **Rich Simulation**: Comprehensive synthetic data generation for testing and validation
- **Diagnostic Visualization**: Publication-quality video output for quality control

## Installation

### Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager

### Install from source

```bash
git clone https://github.com/yourusername/trodestrack.git
cd trodestrack
uv sync
```

## Quick Start

### 1. Generate and Filter Synthetic Data (3 minutes)

The fastest way to understand TrodesTrack is to run the EKF example on synthetic data:

```bash
# Clone and setup
git clone https://github.com/yourusername/trodestrack.git
cd trodestrack
uv sync

# Run EKF on basic scenarios (stationary, constant velocity, circular)
uv run python examples/03_ekf_basic_scenarios.py
```

This generates 3 diagnostic PNGs showing filter performance, bias convergence, and NEES consistency checks. **Key insight:** gyro bias is only observable during rotation!

### 2. Compare EKF vs UKF

```bash
uv run python examples/04_ukf_basic_scenarios.py
```

Compares sigma-point (UKF) vs Jacobian (EKF) approaches. **Verdict:** EKF wins 5/9 metrics and is 1-5× faster—start with EKF!

### 3. Test Dropout Robustness

```bash
uv run python examples/05_ekf_with_dropouts.py
uv run python examples/06_ukf_with_dropouts.py
```

Simulates 10%, 20%, and 30% camera dropout to stress-test IMU-only periods.

### 4. Use Smoothing for Offline Analysis

```bash
uv run python examples/07_smoother_demonstration.py
```

Shows how backward RTS smoothing achieves **3× drift reduction** on 5-second dropout by using future observations.

### 5. Generate QA Reports

```bash
uv run python examples/08_qa_report_generation.py
```

Creates a publication-quality PDF with all PRD metrics, NEES/NIS checks, and time series plots.

### Python API Examples

#### Generate synthetic data

```python
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

config = RatIMUSimConfig(duration_s=10.0, seed=42)
sim = simulate_rat_imu(config)
```

#### Run EKF filter

```python
from trodestrack.models.ekf import ekf_forward, EKFConfig, ekf_initialize_state

# Initialize from simulation
cfg = EKFConfig()
x0, P0 = ekf_initialize_state(sim, cfg)

# Run filter
fwd = ekf_forward(x0, P0, cfg, sim)
```

#### Generate QA report

```python
from trodestrack.qa.report import generate_filter_report

generate_filter_report(
    states_fwd=fwd['x'],
    states_truth=sim['x_truth'],
    covariances=fwd['P'],
    config=cfg,
    output_path="report.pdf"
)
```

### Explore All Examples

See [`examples/README.md`](examples/README.md) for the complete learning path. Examples are numbered to teach concepts progressively:

- **01-02**: Simulation fundamentals
- **03-04**: Filter basics (EKF and UKF)
- **05-06**: Robustness (dropouts and occlusions)
- **07**: Smoothing techniques
- **08**: QA reporting

## Project Status

**Current Milestone**: Integration & QA (M4) - Ready for Production Testing

### Completed ✅

- ✅ **Simulation Foundation** (M1)
  - Realistic rat motion with Ornstein-Uhlenbeck dynamics
  - Full IMU physics (tilt, drag, bias random walks)
  - Camera dropout, LED swaps, occlusions, and reflections
  - Arena boundaries with inelastic collisions
- ✅ **Filter Implementation** (M2)
  - Extended Kalman Filter (EKF) with IMU pre-integration
  - Unscented Kalman Filter (UKF) with sigma-point transforms
  - Rauch-Tung-Striebel (RTS) smoothing for offline analysis
  - Iterated EKF/Smoother (IEKF/IEKS) for nonlinear accuracy
- ✅ **Robustness Features** (M3)
  - Mahalanobis gating for outlier rejection
  - Zero-velocity updates (ZUPT) for stationary periods
  - Adaptive process noise during camera dropout
  - Heading pseudo-measurements from dual LEDs
  - LED spacing validation and adaptive measurement noise
- ✅ **QA & Diagnostics** (M4)
  - Comprehensive metrics (RMSE, NEES, NIS, innovation statistics)
  - Publication-quality plots and multi-page PDF reports
  - CLI tool: `trodestrack report --run run1/ --pdf report.pdf`
  - Diagnostic videos with 9-panel filter state visualization
- ✅ **Testing & Validation**
  - 200+ unit, integration, and property tests (all passing)
  - PRD acceptance criteria achieved:
    - Position RMSE ≤ 2 cm ✓
    - Velocity RMSE ≤ 10 cm/s ✓
    - Heading RMSE ≤ 7° ✓
    - Throughput: 45× realtime (CPU), latency: 0.39 ms/frame ✓

### In Progress 🚧

- 🚧 **Documentation** (M5)
  - Tuning guide with NEES-based diagnostics
  - Troubleshooting guide for common failure modes
- 🚧 **I/O Loaders**
  - Trodes LED detection format
  - DeepLabCut keypoint format
  - SpikeGadgets raw IMU format
- 🚧 **CLI Tools**
  - `trodestrack smooth --config session.yaml`
  - `trodestrack online --config session.yaml`

## Documentation

### User Documentation

- **[Examples README](examples/README.md)** - Start here! Progressive learning path with 8 pedagogical examples
- **[Tuning Guide](TUNING.md)** - NEES-based diagnostics and parameter selection (coming soon)
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common filter failures and solutions (coming soon)

### Developer Documentation

- [Product Requirements Document (PRD.md)](PRD.md) - Full project specification
- [Development Guide (CLAUDE.md)](CLAUDE.md) - Commands and architecture
- [Task Tracking (TASKS.md)](TASKS.md) - Current roadmap and completion status

## Development

### Run tests

```bash
uv run pytest tests/ -v
```

### Code quality

```bash
# Type checking
uv run mypy src/trodestrack --ignore-missing-imports

# Linting
uv run ruff check src/ tests/

# Formatting
uv run black src/ tests/
```

### Development commands

See [CLAUDE.md](CLAUDE.md) for complete list of development commands and project architecture.

## Architecture

```
trodestrack/
  sim/          # Simulation: analytic scenarios + realistic rat IMU
  models/       # EKF, UKF, RTS/IEKS smoothers, state initialization
  runtime/      # Online filter API + offline smoother workflows
  qa/           # Metrics (RMSE, NEES, NIS), plots, PDF reports
  viz/          # Diagnostic videos with multi-panel state visualization
  cli/          # CLI: trodestrack report (more commands coming)
  io/           # Data loaders: Trodes, DLC, SpikeGadgets (coming soon)
  config/       # Configuration schemas (coming soon)
```

## Contributing

This project follows strict test-driven development (TDD) practices:

1. Write tests first
2. Run tests and verify they fail
3. Implement features
4. Run tests until they pass
5. Refactor for clarity

See [PRD.md](PRD.md) for development guidelines and code style requirements.

## Citation

If you use trodestrack in your research, please cite:

```bibtex
@software{trodestrack2025,
  title={trodestrack: Sensor-fused 2D rat tracking with JAX EKF/UKF},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/trodestrack}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- SpikeGadgets for hardware specifications
- DeepLabCut team for pose estimation framework
- JAX team for high-performance numerical computing

## Contact

For questions, issues, or feature requests, please open an issue on GitHub.

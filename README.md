# trodestrack

**Sensor-fused 2D rat tracking with JAX EKF/UKF for neuroscience research**

trodestrack combines video tracking (Trodes LEDs and/or DeepLabCut keypoints) with IMU data from SpikeGadgets headstages to provide accurate position, velocity, and heading estimates for freely-moving rats on behavioral mazes.

## Features

- **Sensor Fusion**: Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for combining video (~30 Hz) and IMU (100 Hz) measurements
- **3D IMU Support**: Full 6-axis IMU processing (3-axis gyro + 3-axis accel) with gravity compensation
- **Online & Offline Processing**: Real-time filtering and RTS smoothing for offline analysis
- **Robust Handling**: Occlusions, LED swaps, reflections, and sensor dropout
- **JAX-Accelerated**: High-performance implementation using JAX - **316× realtime** on CPU, GPU-ready
- **Rich Simulation**: Comprehensive synthetic data generation for testing and validation
- **Diagnostic Visualization**: Publication-quality video output for quality control

## Hardware Compatibility

**Supported IMU Hardware:**

trodestrack is designed for **SpikeGadgets headstages** with integrated 6-axis IMU sensors.

**Official Hardware Specifications** (source: [SpikeGadgets Product Manual](https://spikegadgets.com/documentation/)):
- **3-axis accelerometer**: ±2g range, 16-bit signed (0.000061g per LSB)
- **3-axis gyroscope**: ±2000 deg/s range, 16-bit signed (0.061 deg/s per LSB)
- **Sensor refresh rate**: 104 Hz (when both sensors enabled)
- **Internal sampling**: 500 Hz per sensor (both enabled), 1 kHz (single sensor)
- **Output format**: Sample-and-hold repeats expand 104 Hz data to ~20-30 kHz nominal rate

**Data Processing:**
- Preprocessing removes sample-and-hold duplicates → ~100 Hz effective rate
- Timestamp-based integration handles variable sampling rates
- Compatible with both 2D IMU mode (gyro-Z, accel-XY) and full 3D mode (all 6 axes)

**Simulation Defaults:**
- All synthetic data generation uses realistic SpikeGadgets specifications
- IMU rate: 104 Hz (matches hardware sensor refresh rate)
- Noise levels: 0.01 °/s/√Hz gyro, 0.2 mg/√Hz accel (per SpikeGadgets spec)
- Ensures simulations accurately predict real-world performance

**Video Tracking:**
- Trodes LED detection (dual LED setup for heading)
- DeepLabCut keypoint tracking (any pose estimation output)
- Camera rate: typically 30 Hz (configurable)

## Installation

### Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager

### Install from source

```bash
git clone https://github.com/edeno/trodestrack.git
cd trodestrack
uv sync
```

## Quick Start

### 1. Generate and Filter Synthetic Data (3 minutes)

The fastest way to understand TrodesTrack is to run the EKF example on synthetic data:

```bash
# Clone and setup
git clone https://github.com/edeno/trodestrack.git
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

Compares sigma-point (UKF) vs Jacobian (EKF) approaches. **Verdict:** EKF wins 6/9 metrics (UKF: 3/9) and is 1-5× faster—start with EKF!

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

# Default config matches SpikeGadgets hardware (104 Hz IMU, realistic noise)
config = RatIMUSimConfig(duration_s=10.0)
sim = simulate_rat_imu(config, seed=42)  # seed is an arg of simulate_rat_imu
```

#### Run EKF filter

```python
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig

cfg = EKFConfig()
result = extended_kalman_filter(
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)
# result.filtered_means: (N_cam, n_state)
# result.filtered_covariances: (N_cam, n_state, n_state)
```

#### Working with State Layouts (Recommended Pattern)

TrodesTrack uses an explicit **state layout system** to eliminate hardcoded dimension assumptions and support multiple tracking modes (5D, 8D, 10D, 15D states). **Always use state layouts** instead of magic indices like `[:, 0:2]`.

```python
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig
from trodestrack.models.state_layout import get_layout
from trodestrack.sim.simple import simulate_circular, SimpleSimConfig

# Generate simulation and run filter
sim_config = SimpleSimConfig(duration_s=10.0)
sim = simulate_circular(sim_config)
ekf_config = EKFConfig()
result = extended_kalman_filter(
    ekf_config,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)

# Get state layout from filter config (BEST PRACTICE!)
# EKFConfig defaults to "2d_cam_3d_imu" (10D: [x, y, vx, vy, vz, theta, biases...]).
layout = get_layout(ekf_config.state_mode)

# ✅ GOOD: Extract states using layout indices (dimension-agnostic)
positions = result.filtered_means[:, layout.pos_idx]      # (N, 2) in meters
velocities = result.filtered_means[:, layout.vel_idx]     # (N, 3) for 2d_cam_3d_imu, (N, 2) for 2d_full
headings = result.filtered_means[:, layout.heading_idx]   # (N,) in radians

# ❌ BAD: Hardcoded indices (breaks when switching state modes!)
# positions = result.filtered_means[:, 0:2]  # Fragile! Don't do this!

# Extract uncertainties (covariances) using layout indices
P = result.filtered_covariances                           # (N, layout.n, layout.n)
pos_cov = P[:, layout.pos_idx, :][:, :, layout.pos_idx] # (N, 2, 2) position covariance
pos_std = np.sqrt(np.diagonal(pos_cov, axis1=1, axis2=2)) # (N, 2) position uncertainty

# Plot position with ±2σ uncertainty bands
import matplotlib.pyplot as plt
t = sim['t_cam_exp']
plt.plot(t, positions[:, 0], label='x')
plt.fill_between(t,
                 positions[:, 0] - 2*pos_std[:, 0],
                 positions[:, 0] + 2*pos_std[:, 0],
                 alpha=0.3, label='±2σ')
plt.xlabel('Time (s)')
plt.ylabel('X Position (m)')
plt.legend()
plt.show()
```

**Available State Layouts:**

| Layout String | Dimensions | State Vector | Use Case |
|--------------|-----------|--------------|----------|
| `"2d_full"` | 8D | `[x, y, vx, vy, θ, b_gz, b_ax, b_ay]` | Standard sensor fusion (camera + IMU) |
| `"vision_only"` | 5D | `[x, y, vx, vy, θ]` | Camera-only tracking (no biases) |
| `"2d_cam_3d_imu"` | 10D | `[x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]` | 2D camera with 3D accel (detect rearing) |
| `"3d_euler"` | 15D | `[x, y, z, vx, vy, vz, roll, pitch, yaw, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]` | Full 3D tracking with Euler angles |
| `"3d_quat"` | 16D | `[x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]` | Full 3D tracking with quaternions |

**Why use state layouts?**

1. **Dimension-agnostic code**: Works with 5D, 8D, 10D, 15D states without modification
2. **Self-documenting**: `layout.pos_idx` is clearer than `[:, 0:2]`
3. **Robust to changes**: Switching state modes doesn't break your analysis code
4. **Matches internal implementation**: Filters use the same layout system

See [`src/trodestrack/models/state_layout.py`](src/trodestrack/models/state_layout.py) for full API documentation.

#### Generate QA report

```python
import numpy as np
from trodestrack.models.state_layout import get_layout
from trodestrack.qa.report import generate_qa_report
from trodestrack.qa.metrics import compute_nees

layout = get_layout(ekf_config.state_mode)

# Align ground truth (IMU rate, 5D [x, y, vx, vy, theta]) to camera frames.
X_truth_at_cam = np.array(
    [sim["X_truth"][np.argmin(np.abs(sim["t_imu"] - t_c))] for t_c in sim["t_cam_exp"]]
)
filtered = np.asarray(result.filtered_means)
filtered_cov = np.asarray(result.filtered_covariances)

# Layout-aware indexing handles every scalar-heading state mode.
pos_idx = list(layout.pos_idx)
vel_idx_2d = list(layout.vel_idx)[:2]  # X_truth has only vx, vy
heading_col = int(layout.heading_idx)

# Position-only NEES (state_dim=2): expected mean ~ 2 for a consistent filter.
nees = compute_nees(
    states_true=X_truth_at_cam[:, :2],
    states_est=filtered[:, pos_idx],
    covariances_est=filtered_cov[np.ix_(np.arange(filtered.shape[0]), pos_idx, pos_idx)],
)
generate_qa_report(
    pdf_path="report.pdf",
    t=sim["t_cam_exp"],
    positions_true=X_truth_at_cam[:, :2],
    positions_est=filtered[:, pos_idx],
    velocities_true=X_truth_at_cam[:, 2:4],
    velocities_est=filtered[:, vel_idx_2d],
    headings_true=X_truth_at_cam[:, 4],
    headings_est=filtered[:, heading_col],
    nees=nees,
    state_dim=2,
)
```

The CLI `trodestrack report` wraps this; see [`src/trodestrack/cli/report.py`](src/trodestrack/cli/report.py) for the full call site.

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
  - CLI tool: `trodestrack report --run qa_inputs/ --pdf report.pdf` (consumes a separately-prepared QA-input directory; see `--help`)
  - Diagnostic videos with 9-panel filter state visualization
- ✅ **Testing & Validation**
  - 660+ unit, integration, regression, and property tests (all passing)
  - PRD acceptance criteria achieved:
    - Position RMSE ≤ 2 cm ✓
    - Velocity RMSE ≤ 10 cm/s ✓
    - Heading RMSE ≤ 7° ✓
    - Throughput: 316× realtime (CPU), latency: 0.11 ms/frame ✓
- ✅ **3D IMU Support** (M5)
  - Full 6-axis IMU processing (gyro + accel)
  - Gravity-aware dynamics with 3D acceleration
  - 2D pose estimation with 3D IMU inputs
  - Improved drift handling during vision dropout
- ✅ **JAX Optimization** (M6)
  - JIT-compiled UKF (mirrors EKF pattern)
  - Vectorized operations (sigma points, bias freeze)
  - Host-side preprocessing for efficiency
  - 316× realtime speedup on 5-minute session

### In Progress 🚧

- 🚧 **I/O Loaders**
  - Trodes LED detection format
  - DeepLabCut keypoint format
  - SpikeGadgets raw IMU format
- 🚧 **CLI Tools**
  - YAML config loading for `trodestrack smooth` / `trodestrack online`
    (today both commands take individual `--imu-timestamps`,
    `--imu-measurements`, `--camera-timestamps`, `--led1-positions`, … flags;
    a single `--config session.yaml` shorthand is planned)

## Documentation

### User Documentation

- **[Examples README](examples/README.md)** - Start here! Progressive learning path with 9 pedagogical examples (01–08 plus 03b)
- **[Tuning Guide](docs/TUNING.md)** - NEES-based diagnostics and parameter selection
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common filter failures and solutions
- **[State Layouts](docs/user-guide/state-layouts.md)** - Available state modes and dimension-agnostic API

### Developer Documentation

- [Development Guide (CLAUDE.md)](CLAUDE.md) - Commands and architecture
- [Implementation plans](docs/plans/) - Active milestone plans (e.g., the tilt-orientation / 3D camera plan)

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
  cli/          # CLI: trodestrack online / smooth / report
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

See [CLAUDE.md](CLAUDE.md) for development guidelines and code style requirements.

## Citation

If you use trodestrack in your research, please cite:

```bibtex
@software{trodestrack2025,
  title={trodestrack: Sensor-fused 2D rat tracking with JAX EKF/UKF},
  author={Your Name},
  year={2025},
  url={https://github.com/edeno/trodestrack}
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

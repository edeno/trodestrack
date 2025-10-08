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

### Generate a synthetic simulation

```python
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

# Generate 10-second simulation with default parameters
config = RatIMUSimConfig(duration_s=10.0)
sim = simulate_rat_imu(config, seed=42)

# Simulation returns position, velocity, heading, IMU data, and camera observations
print(f"Generated {len(sim['t_cam_exp'])} camera frames")
print(f"Generated {len(sim['t_imu'])} IMU samples")
```

### Create a diagnostic video

```python
from trodestrack.viz.video import create_diagnostic_video

# Visualize the simulation with animated diagnostic plots
create_diagnostic_video(
    sim,
    "output.mp4",
    fps=30,
    speedup=1.0  # Real-time playback
)
```

### Explore examples

See the [`examples/`](examples/) directory for more detailed demonstrations:

- `01_simple_simulations.py` - Analytic trajectories (stationary, constant velocity, circular)
- `02_rat_imu_simulation.py` - Realistic rat motion with full IMU physics
- `03_video_diagnostic.py` - Generate diagnostic videos (requires simulation output)

## Project Status

**Current Phase**: Simulation Infrastructure (Complete ✓)

- ✅ Realistic rat motion simulation with IMU physics
- ✅ Camera dropout, LED swaps, and confidence modeling
- ✅ Comprehensive test suite (60+ tests)
- ✅ Publication-quality visualization
- 🚧 Filter implementation (EKF/UKF) - **In Progress**
- 🚧 I/O loaders for Trodes/DLC/SpikeGadgets data
- 🚧 CLI tools and configuration management

## Documentation

- [Product Requirements Document (PRD.md)](PRD.md) - Full project specification
- [Development Guide (CLAUDE.md)](CLAUDE.md) - Commands and architecture
- [Planning Document (PLANNING.md)](PLANNING.md) - Test strategy and roadmap
- Examples README: [`examples/README.md`](examples/README.md)

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
  sim/          # Simulation: analytic and realistic rat motion
  viz/          # Visualization: diagnostic videos and plots
  config/       # Configuration schemas (Pydantic, coming soon)
  io/           # Data loaders: Trodes, DLC, SpikeGadgets (coming soon)
  models/       # EKF, UKF, dynamics, measurements (coming soon)
  runtime/      # Online filter and offline smoother (coming soon)
  qa/           # Metrics, NEES, RMSE, tuning diagnostics (coming soon)
  cli/          # Command-line interface (coming soon)
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

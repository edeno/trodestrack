# Installation

This guide covers installing TrodesTrack for different use cases.

## Requirements

- **Python**: >= 3.11 (tested on 3.11, 3.12, 3.13)
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Install from Source (Recommended)

For development or the latest features:

```bash
# Clone the repository
git clone https://github.com/edeno/trodestrack.git
cd trodestrack

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Install from PyPI

!!! note "Coming Soon"
    PyPI release is planned for v1.0. For now, install from source.

```bash
# When available:
pip install trodestrack
```

## Optional Dependencies

TrodesTrack has optional dependency groups for different use cases:

=== "Development"

    ```bash
    # Install with development tools
    uv sync --extra dev

    # Includes: pytest, ruff, mypy, pre-commit, hypothesis
    ```

=== "Documentation"

    ```bash
    # Install with documentation tools
    uv sync --extra docs

    # Includes: mkdocs, mkdocs-material, mkdocstrings
    ```

=== "Notebooks"

    ```bash
    # Install with Jupyter support
    uv sync --extra notebooks

    # Includes: jupyter, jupytext, ipykernel
    ```

=== "Video"

    ```bash
    # Install with video generation tools
    uv sync --extra video

    # Includes: opencv-python, python-pptx
    ```

=== "All"

    ```bash
    # Install everything
    uv sync --all-extras
    ```

## GPU Acceleration

TrodesTrack uses JAX for high-performance computation. By default, JAX runs on CPU.

### CUDA (NVIDIA GPUs)

```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda12]"

# Verify GPU is available
python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)]
```

### Apple Silicon (Metal)

```bash
# Install JAX with Metal support
pip install --upgrade "jax[metal]"

# Verify Metal is available
python -c "import jax; print(jax.devices())"
# Should show: [MetalDevice(id=0)]
```

## Verify Installation

Run this quick check to verify everything is working:

```python
# Verify installation
from trodestrack.sim.simple import simulate_circular, SimpleSimConfig
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig

# Generate test data
sim = simulate_circular(SimpleSimConfig(duration_s=1.0))

# Run filter
result = extended_kalman_filter(
    EKFConfig(),
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)

print(f"Filtered {len(result.filtered_means)} frames successfully!")
print(f"Position shape: {result.filtered_means[:, :2].shape}")
```

## Troubleshooting

### JAX Not Found

```
ModuleNotFoundError: No module named 'jax'
```

**Solution**: Install JAX explicitly:

```bash
pip install jax jaxlib
```

### CUDA Not Available

```
JAX using CPU instead of GPU
```

**Solution**: Install the CUDA version of JAX:

```bash
pip install --upgrade "jax[cuda12]"
```

### Import Errors

```
ImportError: cannot import name 'extended_kalman_filter' from 'trodestrack.models.ekf'
```

**Solution**: Ensure you installed in development mode:

```bash
pip install -e .
```

## Next Steps

Now that you have TrodesTrack installed, proceed to the [Quick Start](quickstart.md) guide.

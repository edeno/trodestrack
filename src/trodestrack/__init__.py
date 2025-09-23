"""Sensor-fused 2D rat tracking with JAX EKF/UKF for SpikeGadgets/Trodes."""

# Configure JAX globally for the entire package
import jax
jax.config.update("jax_enable_x64", True)


def main() -> None:
    """Entry point for trodestrack CLI."""
    import sys

    from .cli.main import main as cli_main

    sys.exit(cli_main())

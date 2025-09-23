"""Sensor-fused 2D rat tracking with JAX EKF/UKF for SpikeGadgets/Trodes."""

# Keep __init__ side-effect free. Re-export public API if desired.


def main() -> None:
    """Entry point for trodestrack CLI."""
    import sys

    from .cli.main import main as cli_main

    sys.exit(cli_main())

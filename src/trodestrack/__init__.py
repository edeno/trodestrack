"""trodestrack: Sensor-fused 2D rat tracking with JAX EKF/UKF for SpikeGadgets/Trodes."""

from __future__ import annotations

import argparse
import sys

__version__ = "0.1.0"

# Public API surface - defines what's exported from the package
# Users import submodules directly: from trodestrack.models import ekf
__all__ = [
    "__version__",
    "main",
]


def main() -> None:
    """Main entry point for the trodestrack CLI.

    This function sets up the argument parser with all available subcommands
    and dispatches to the appropriate command handler.
    """
    parser = argparse.ArgumentParser(
        prog="trodestrack",
        description="Sensor-fused 2D rat tracking with JAX EKF/UKF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run offline smoothing on sensor data (best accuracy)
  trodestrack smooth --imu-timestamps t_imu.txt --imu-measurements U_imu.txt \\
                     --camera-timestamps t_cam.txt --led1-positions led1.txt \\
                     --output-dir run1/

  # Run online filtering only (faster, lower latency)
  trodestrack online --imu-timestamps t_imu.txt --imu-measurements U_imu.txt \\
                     --camera-timestamps t_cam.txt --led1-positions led1.txt \\
                     --output-dir run1/

  # Generate QA report (requires a separately-prepared QA directory containing
  # ground-truth-aligned timestamps.npy, positions_true.npy, positions_est.npy,
  # velocities_*.npy, headings_*.npy, nees.npy, and state_dim.txt; see
  # examples/08_qa_report_generation.py and tests/cli/test_report_command.py).
  trodestrack report --run qa_inputs/ --pdf report.pdf

  # Generate report with custom title
  trodestrack report --run qa_inputs/ --pdf report.pdf --title "Session 2024-10-11"

Note: ``online`` and ``smooth`` write the raw filter state
(filtered_means.txt, filtered_covariances.txt, marginal_loglik.txt) to
``--output-dir``. The ``report`` command operates on a different,
QA-prepared file layout that includes ground truth, so the two stages
are not directly chained — generate ``report``'s inputs from your own
analysis pipeline (see qa.metrics.compute_nees and qa.report.generate_qa_report).

For more information, visit: https://github.com/edeno/trodestrack
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"trodestrack {__version__}",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        help="Command to run",
    )

    # Import and register subcommands
    from trodestrack.cli.online import add_online_parser
    from trodestrack.cli.report import add_report_parser
    from trodestrack.cli.smooth import add_smooth_parser

    add_report_parser(subparsers)
    add_smooth_parser(subparsers)
    add_online_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)

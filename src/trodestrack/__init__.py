"""trodestrack: Sensor-fused 2D rat tracking with JAX EKF/UKF for SpikeGadgets/Trodes."""

from __future__ import annotations

import argparse
import sys

__version__ = "0.1.0"


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
  # Generate QA report from filter results
  trodestrack report --run run1/ --pdf report.pdf

  # Generate report with custom title
  trodestrack report --run run1/ --pdf report.pdf --title "Session 2024-10-11"

For more information, visit: https://github.com/yourusername/trodestrack
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
    from trodestrack.cli.report import add_report_parser

    add_report_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)

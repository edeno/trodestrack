"""Main CLI entry point for trodestrack."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ..config.loader import load_config

# Module-level logger
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="trodestrack", description="Sensor-fused 2D rat tracking with JAX EKF/UKF"
    )

    # Global verbosity options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Smooth command
    smooth_parser = subparsers.add_parser("smooth", help="Run offline smoothing on a session")
    smooth_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to session configuration YAML file",
    )
    smooth_parser.add_argument(
        "--output", "-o", type=Path, help="Override output directory from config"
    )

    # Online command
    online_parser = subparsers.add_parser("online", help="Run online filtering")
    online_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to session configuration YAML file",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate analysis report from results")
    report_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing tracking results",
    )
    report_parser.add_argument("--output", "-o", type=Path, help="Output path for report (PDF)")

    # Calibration command
    calib_parser = subparsers.add_parser(
        "calib-homography", help="Interactive homography calibration tool"
    )
    calib_parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to video frame image for calibration",
    )
    calib_parser.add_argument(
        "--output", "-o", type=Path, help="Output path for homography YAML file"
    )
    calib_parser.add_argument(
        "--arena-width", type=float, default=200.0, help="Arena width in cm (default: 200.0)"
    )
    calib_parser.add_argument(
        "--arena-height", type=float, default=150.0, help="Arena height in cm (default: 150.0)"
    )

    return parser


def cmd_smooth(args: argparse.Namespace) -> int:
    """Execute smooth command."""
    try:
        config = load_config(args.config)

        # Override output directory if specified
        if args.output:
            config.output.output_dir = args.output

        logger.info(f"Loading session config from: {args.config}")
        logger.info(f"Video file: {config.video_file}")
        logger.info(f"IMU file: {config.imu_file}")
        logger.info(f"Output directory: {config.output.output_dir}")
        logger.info(f"Filter type: {config.filter.filter_type}")

        # TODO: Implement actual smoothing pipeline
        logger.warning("Offline smoothing pipeline not yet implemented")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error in smooth command: {e}")
        return 1


def cmd_online(args: argparse.Namespace) -> int:
    """Execute online command."""
    try:
        config = load_config(args.config)

        logger.info(f"Loading session config from: {args.config}")
        logger.info(f"Starting online tracker with {config.filter.filter_type}")

        # TODO: Implement actual online tracker
        logger.warning("Online tracking not yet implemented")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error in online command: {e}")
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Execute report command."""
    try:
        if not args.run_dir.exists():
            logger.error(f"Run directory does not exist: {args.run_dir}")
            return 1

        output_path = args.output or args.run_dir / "report.pdf"

        logger.info(f"Generating report from: {args.run_dir}")
        logger.info(f"Output report: {output_path}")

        # TODO: Implement actual report generation
        logger.warning("Report generation not yet implemented")
        return 0

    except Exception as e:
        logger.error(f"Error in report command: {e}")
        return 1


def cmd_calib_homography(args: argparse.Namespace) -> int:
    """Execute calibration command."""
    try:
        if not args.video.exists():
            logger.error(f"Video file does not exist: {args.video}")
            return 1

        output_path = args.output or args.video.parent / "homography.yaml"

        logger.info(f"Starting homography calibration with: {args.video}")
        logger.info(f"Output homography file: {output_path}")

        # Import calibration tool
        from trodestrack.cli.calibration import (
            check_dependencies,
            run_interactive_calibration,
        )

        # Check dependencies
        deps_available, missing = check_dependencies()
        if not deps_available:
            logger.error(f"Missing required packages: {', '.join(missing)}")
            logger.error("Install with: uv add " + " ".join(missing))
            return 1

        # Get arena dimensions from command line args
        arena_width = args.arena_width
        arena_height = args.arena_height

        # Run interactive calibration
        success = run_interactive_calibration(
            image_path=args.video,
            output_path=output_path,
            arena_width_cm=arena_width,
            arena_height_cm=arena_height,
        )

        if success:
            logger.info(f"Homography calibration completed successfully: {output_path}")
            return 0
        else:
            logger.error("Homography calibration failed or was cancelled")
            return 1

    except Exception as e:
        logger.error(f"Error in calibration command: {e}")
        return 1


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging based on verbosity flags
    setup_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))

    # Dispatch to command handlers
    command_handlers = {
        "smooth": cmd_smooth,
        "online": cmd_online,
        "report": cmd_report,
        "calib-homography": cmd_calib_homography,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

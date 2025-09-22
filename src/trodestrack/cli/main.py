"""Main CLI entry point for trodestrack."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ..config.loader import load_config, create_default_config


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="trodestrack",
        description="Sensor-fused 2D rat tracking with JAX EKF/UKF"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )

    # Smooth command
    smooth_parser = subparsers.add_parser(
        "smooth",
        help="Run offline smoothing on a session"
    )
    smooth_parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to session configuration YAML file"
    )
    smooth_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Override output directory from config"
    )

    # Online command
    online_parser = subparsers.add_parser(
        "online",
        help="Run online filtering"
    )
    online_parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to session configuration YAML file"
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate analysis report from results"
    )
    report_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing tracking results"
    )
    report_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for report (PDF)"
    )

    # Calibration command
    calib_parser = subparsers.add_parser(
        "calib-homography",
        help="Interactive homography calibration tool"
    )
    calib_parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to video frame image for calibration"
    )
    calib_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for homography YAML file"
    )

    return parser


def cmd_smooth(args: argparse.Namespace) -> int:
    """Execute smooth command."""
    try:
        config = load_config(args.config)

        # Override output directory if specified
        if args.output:
            config.output.output_dir = args.output

        print(f"Loading session config from: {args.config}")
        print(f"Video file: {config.video_file}")
        print(f"IMU file: {config.imu_file}")
        print(f"Output directory: {config.output.output_dir}")
        print(f"Filter type: {config.filter.filter_type}")

        # TODO: Implement actual smoothing pipeline
        print("Offline smoothing pipeline not yet implemented")
        return 0

    except Exception as e:
        print(f"Error in smooth command: {e}", file=sys.stderr)
        return 1


def cmd_online(args: argparse.Namespace) -> int:
    """Execute online command."""
    try:
        config = load_config(args.config)

        print(f"Loading session config from: {args.config}")
        print(f"Starting online tracker with {config.filter.filter_type}")

        # TODO: Implement actual online tracker
        print("Online tracking not yet implemented")
        return 0

    except Exception as e:
        print(f"Error in online command: {e}", file=sys.stderr)
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Execute report command."""
    try:
        if not args.run_dir.exists():
            print(f"Run directory does not exist: {args.run_dir}", file=sys.stderr)
            return 1

        output_path = args.output or args.run_dir / "report.pdf"

        print(f"Generating report from: {args.run_dir}")
        print(f"Output report: {output_path}")

        # TODO: Implement actual report generation
        print("Report generation not yet implemented")
        return 0

    except Exception as e:
        print(f"Error in report command: {e}", file=sys.stderr)
        return 1


def cmd_calib_homography(args: argparse.Namespace) -> int:
    """Execute calibration command."""
    try:
        if not args.video.exists():
            print(f"Video file does not exist: {args.video}", file=sys.stderr)
            return 1

        output_path = args.output or args.video.parent / "homography.yaml"

        print(f"Starting homography calibration with: {args.video}")
        print(f"Output homography file: {output_path}")

        # TODO: Implement interactive homography tool
        print("Interactive homography calibration not yet implemented")
        return 0

    except Exception as e:
        print(f"Error in calibration command: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

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
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
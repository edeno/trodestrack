"""CLI command for online filtering of sensor-fused tracking data.

This module implements the `trodestrack online` command which runs real-time
EKF filtering (forward pass only, no smoothing) on IMU + camera data.

Usage:
    trodestrack online \\
        --imu-timestamps t_imu.txt \\
        --imu-measurements U_imu.txt \\
        --camera-timestamps t_cam.txt \\
        --led1-positions Z_cam_led1.txt \\
        --led2-positions Z_cam_led2.txt \\
        --output-dir run1/

Output files:
    run1/filtered_means.txt: Filter state estimates (N_cam, 8)
    run1/filtered_covariances.txt: Filter covariances (N_cam, 8, 8) flattened
    run1/marginal_loglik.txt: Marginal log-likelihood (scalar)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter


def add_online_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add online subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object from argparse.
    """
    parser = subparsers.add_parser(
        "online",
        help="Run online filtering (EKF forward pass only, no smoothing)",
        description="""
Run Extended Kalman Filter on sensor-fused tracking data (online mode).

This command performs forward filtering only (no backward smoothing):
1. Extended Kalman Filter (EKF) forward pass
2. Saves filter outputs to disk

Online mode is suitable for:
- Real-time applications (low latency)
- Situations where you don't have access to future observations
- Quick processing without smoothing overhead

For best accuracy with access to all data, use 'trodestrack smooth' instead.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required input arguments
    input_group = parser.add_argument_group("input files (required)")
    input_group.add_argument(
        "--imu-timestamps",
        type=Path,
        required=True,
        help="Path to IMU timestamps file (N_imu,) [seconds]",
        metavar="FILE",
    )
    input_group.add_argument(
        "--imu-measurements",
        type=Path,
        required=True,
        help="Path to IMU measurements file (N_imu, 3) [ω_z, f_x, f_y] in rad/s and m/s²",
        metavar="FILE",
    )
    input_group.add_argument(
        "--camera-timestamps",
        type=Path,
        required=True,
        help="Path to camera timestamps file (N_cam,) [seconds]",
        metavar="FILE",
    )
    input_group.add_argument(
        "--led1-positions",
        type=Path,
        required=True,
        help="Path to LED1 positions file (N_cam, 2) [x, y] in meters",
        metavar="FILE",
    )
    input_group.add_argument(
        "--led2-positions",
        type=Path,
        required=False,
        default=None,
        help="Path to LED2 positions file (N_cam, 2) [x, y] in meters (optional)",
        metavar="FILE",
    )
    input_group.add_argument(
        "--camera-mask",
        type=Path,
        required=False,
        default=None,
        help="Path to camera validity mask file (N_cam,) [0/1] (optional, default: all valid)",
        metavar="FILE",
    )

    # Output arguments
    output_group = parser.add_argument_group("output configuration")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save filter outputs",
        metavar="DIR",
    )

    # Filter configuration
    filter_group = parser.add_argument_group("filter parameters (optional)")
    filter_group.add_argument(
        "--process-noise-pos",
        type=float,
        default=0.02,
        help="Position process noise (default: 0.02 m²/s)",
    )
    filter_group.add_argument(
        "--process-noise-vel",
        type=float,
        default=2.0,
        help="Velocity process noise (default: 2.0 m²/s³)",
    )
    filter_group.add_argument(
        "--process-noise-heading",
        type=float,
        default=0.02,
        help="Heading process noise (default: 0.02 rad²/s)",
    )
    filter_group.add_argument(
        "--process-noise-gyro-bias",
        type=float,
        default=2e-6,
        help="Gyro bias random walk density (default: 2e-6 rad²/s³)",
    )
    filter_group.add_argument(
        "--process-noise-accel-bias",
        type=float,
        default=2e-4,
        help="Accel bias random walk density (default: 2e-4 m²/s⁵)",
    )
    filter_group.add_argument(
        "--measurement-noise-pos",
        type=float,
        default=0.005**2,
        help="Position measurement noise variance (default: 2.5e-5 m²)",
    )
    filter_group.add_argument(
        "--imu-gyro-noise-density",
        type=float,
        default=0.001,
        help="IMU gyro noise density (default: 0.001 rad/s/√Hz)",
    )
    filter_group.add_argument(
        "--imu-accel-noise-density",
        type=float,
        default=0.05,
        help="IMU accel noise density (default: 0.05 m/s²/√Hz)",
    )
    filter_group.add_argument(
        "--damping-coeff",
        type=float,
        default=0.5,
        help="Velocity damping coefficient (default: 0.5 s⁻¹)",
    )
    filter_group.add_argument(
        "--led-distance",
        type=float,
        default=None,
        help="Expected LED spacing in meters (default: auto-detect from first valid frame)",
    )
    filter_group.add_argument(
        "--use-heading-measurement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use heading pseudo-measurement from dual LEDs (default: True)",
    )

    parser.set_defaults(func=run_online)


def load_data_file(
    path: Path, name: str, expected_shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """Load a data file with validation.

    Parameters
    ----------
    path : Path
        Path to data file.
    name : str
        Descriptive name for error messages.
    expected_shape : tuple[int, ...] | None, optional
        Expected shape (None to skip validation).

    Returns
    -------
    np.ndarray
        Loaded numpy array.

    Raises
    ------
    SystemExit
        If file doesn't exist or has wrong shape.
    """
    if not path.exists():
        print(f"Error: {name} file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = np.loadtxt(path)
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if expected_shape is not None and data.shape != expected_shape:
        print(
            f"Error: {name} has shape {data.shape}, expected {expected_shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    return data


def run_online(args: argparse.Namespace) -> None:
    """Execute the online command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    print("=" * 80)
    print("trodestrack online — Real-time Filtering")
    print("=" * 80)

    # Load input data
    print("\nLoading input data...")
    t_imu = load_data_file(args.imu_timestamps, "IMU timestamps")
    U_imu = load_data_file(args.imu_measurements, "IMU measurements")
    t_cam = load_data_file(args.camera_timestamps, "Camera timestamps")
    Z_cam_led1 = load_data_file(args.led1_positions, "LED1 positions")

    n_imu = len(t_imu)
    n_cam = len(t_cam)

    # Validate IMU data shape
    if U_imu.shape != (n_imu, 3):
        print(
            f"Error: IMU measurements shape {U_imu.shape} doesn't match (n_imu={n_imu}, 3)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate camera data shape
    if Z_cam_led1.shape != (n_cam, 2):
        print(
            f"Error: LED1 positions shape {Z_cam_led1.shape} doesn't match (n_cam={n_cam}, 2)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load optional LED2 data
    if args.led2_positions is not None:
        Z_cam_led2 = load_data_file(args.led2_positions, "LED2 positions")
        if Z_cam_led2.shape != (n_cam, 2):
            print(
                f"Error: LED2 positions shape {Z_cam_led2.shape} doesn't match (n_cam={n_cam}, 2)",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Create NaN-filled array when LED2 not provided (required by filter signature)
        Z_cam_led2 = np.full((n_cam, 2), np.nan)

    # Load optional camera mask
    if args.camera_mask is not None:
        mask_cam = load_data_file(args.camera_mask, "Camera mask").astype(bool)
        if mask_cam.shape != (n_cam,):
            print(
                f"Error: Camera mask shape {mask_cam.shape} doesn't match (n_cam={n_cam},)",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        mask_cam = np.ones(n_cam, dtype=bool)

    print(f"  IMU samples: {n_imu} @ {1.0 / np.mean(np.diff(t_imu)):.1f} Hz")
    print(f"  Camera frames: {n_cam} @ {1.0 / np.mean(np.diff(t_cam)):.1f} Hz")
    print(f"  Valid camera frames: {mask_cam.sum()} ({100 * mask_cam.mean():.1f}%)")
    print(f"  Duration: {t_cam[-1] - t_cam[0]:.2f} s")

    # Configure filter
    print("\nConfiguring Extended Kalman Filter...")
    ekf_config = EKFConfig(
        process_noise_pos=args.process_noise_pos,
        process_noise_vel=args.process_noise_vel,
        process_noise_heading=args.process_noise_heading,
        process_noise_gyro_bias=args.process_noise_gyro_bias,
        process_noise_accel_bias=args.process_noise_accel_bias,
        measurement_noise_pos=args.measurement_noise_pos,
        imu_gyro_noise_density=args.imu_gyro_noise_density,
        imu_accel_noise_density=args.imu_accel_noise_density,
        damping_coeff=args.damping_coeff,
        led_distance=args.led_distance,
        use_heading_measurement=args.use_heading_measurement,
    )

    print(f"  Process noise (pos): {args.process_noise_pos:.4f} m²/s")
    print(f"  Process noise (vel): {args.process_noise_vel:.4f} m²/s³")
    print(f"  Damping coefficient: {args.damping_coeff:.2f} s⁻¹")
    print(f"  LED heading measurement: {args.use_heading_measurement}")

    # Run forward filter
    print("\nRunning Extended Kalman Filter (forward pass only, no smoothing)...")
    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=t_imu,
        U_imu=U_imu,
        t_cam=t_cam,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
    )
    print(f"  Marginal log-likelihood: {filter_result.marginal_loglik:.2f}")
    print("  Filtering complete")

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to {output_dir}/")

    # Save filter outputs
    np.savetxt(output_dir / "filtered_means.txt", filter_result.filtered_means)
    np.savetxt(
        output_dir / "filtered_covariances.txt",
        filter_result.filtered_covariances.reshape(n_cam, -1),
    )

    # Save marginal log-likelihood
    with open(output_dir / "marginal_loglik.txt", "w") as f:
        f.write(f"{filter_result.marginal_loglik:.6f}\n")

    # Save metadata for reproducibility
    with open(output_dir / "metadata.txt", "w") as f:
        f.write("trodestrack online — Real-time Filtering Results\n")
        f.write("=" * 80 + "\n\n")
        f.write("Input Files:\n")
        f.write(f"  IMU timestamps: {args.imu_timestamps}\n")
        f.write(f"  IMU measurements: {args.imu_measurements}\n")
        f.write(f"  Camera timestamps: {args.camera_timestamps}\n")
        f.write(f"  LED1 positions: {args.led1_positions}\n")
        f.write(f"  LED2 positions: {args.led2_positions}\n")
        f.write(f"  Camera mask: {args.camera_mask}\n\n")
        f.write("Filter Configuration:\n")
        f.write(f"  Process noise (pos): {args.process_noise_pos}\n")
        f.write(f"  Process noise (vel): {args.process_noise_vel}\n")
        f.write(f"  Process noise (heading): {args.process_noise_heading}\n")
        f.write(f"  Process noise (gyro bias): {args.process_noise_gyro_bias}\n")
        f.write(f"  Process noise (accel bias): {args.process_noise_accel_bias}\n")
        f.write(f"  Measurement noise (pos): {args.measurement_noise_pos}\n")
        f.write(f"  IMU gyro noise density: {args.imu_gyro_noise_density}\n")
        f.write(f"  IMU accel noise density: {args.imu_accel_noise_density}\n")
        f.write(f"  Damping coefficient: {args.damping_coeff}\n")
        f.write(f"  LED distance: {args.led_distance}\n")
        f.write(f"  Use heading measurement: {args.use_heading_measurement}\n\n")
        f.write("Results:\n")
        f.write(f"  Marginal log-likelihood: {filter_result.marginal_loglik:.6f}\n")
        f.write(f"  Output samples: {n_cam}\n")
        f.write(f"  State dimension: {filter_result.filtered_means.shape[1]}\n")

    print("\nOutput files:")
    print(f"  filtered_means.txt: Filter state estimates ({n_cam}, 8)")
    print(f"  filtered_covariances.txt: Filter covariances ({n_cam}, 64)")
    print("  marginal_loglik.txt: Marginal log-likelihood (scalar)")
    print("  metadata.txt: Run configuration and metadata")
    print("\nFiltering complete!")
    print("\nNOTE: For improved accuracy, use 'trodestrack smooth' to apply RTS smoothing.")

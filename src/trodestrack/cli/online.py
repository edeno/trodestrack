"""CLI command for forward-only filtering of sensor-fused tracking data.

This module implements the ``trodestrack online`` command, which runs the
EKF in **forward filter only** mode (no backward smoothing) on IMU +
camera data. "Online" here means "forward-pass / no future-frame
dependence" — the command is *not* a streaming ingest loop. It loads the
full IMU / camera / LED arrays from disk and calls
``extended_kalman_filter`` once over the batch. Per-frame ingest would
require driving ``trodestrack.models.ekf.predict_step`` /
``trodestrack.models.ekf.update_step`` directly from Python; that is not
exposed as a CLI today.

Usage:
    trodestrack online \\
        --imu-timestamps t_imu.txt \\
        --imu-measurements U_imu.txt \\
        --camera-timestamps t_cam.txt \\
        --led1-positions Z_cam_led1.txt \\
        --led2-positions Z_cam_led2.txt \\
        --output-dir run1/

Output files:
    run1/filtered_means.txt: Filter state estimates (N_cam, n)
    run1/filtered_covariances.txt: Filter covariances (N_cam, n, n) flattened
    run1/marginal_loglik.txt: Marginal log-likelihood (scalar)

Note:
    n is the state dimension (default: 10 for the "2d_cam_3d_imu" mode used
    by EKFConfig() out of the box; pass --led-distance, --use-heading-measurement,
    etc. to override individual filter parameters).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from trodestrack.cli.config_workflow import (
    prepare_config_filter_run,
    save_filter_outputs,
    write_config_metadata,
)
from trodestrack.cli.utils import (
    friendly_cli_errors,
    load_data_file,
    require_cli_inputs,
    validate_camera_mask,
    validate_finite_array,
    validate_monotonic_timestamps,
)
from trodestrack.io import write_session_diagnostics
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.filter_common import FilterCoreConfig

_FILTER_DEFAULTS = FilterCoreConfig()
_LEGACY_REQUIRED_ARGS = (
    "imu_timestamps",
    "imu_measurements",
    "camera_timestamps",
    "led1_positions",
    "output_dir",
)


def add_online_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add online subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object from argparse.
    """
    parser = subparsers.add_parser(
        "online",
        help="Run forward-only EKF filtering (no smoothing) over complete input files",
        description="""
Run Extended Kalman Filter on sensor-fused tracking data in forward-only
("online") mode.

Despite the name, this is a BATCH command: it loads complete IMU,
camera, LED, and mask arrays from disk and runs the EKF once over the
full session. There is no per-frame streaming ingest. "Online" here
means "no backward smoother" / "no future-frame dependence", not
"real-time per-frame".

What it does:
1. Reads input arrays from --imu-timestamps, --imu-measurements,
   --camera-timestamps, --led1-positions, --led2-positions.
2. Runs Extended Kalman Filter (EKF) forward pass over the full arrays
   via `extended_kalman_filter`.
3. Writes filter outputs to --output-dir.

Use this command for:
- Producing forward-only estimates without smoother latency / lookahead.
- Filtering datasets where future observations should not influence past
  estimates (online-style analyses on archived data).

For best accuracy when you do have access to all data, use
`trodestrack smooth` instead.

True per-frame / real-time streaming is not provided by this CLI; for
that, drive `trodestrack.models.ekf.predict_step` /
`trodestrack.models.ekf.update_step` directly via the Python API.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required input arguments
    input_group = parser.add_argument_group("input files (required)")
    input_group.add_argument(
        "--config",
        type=Path,
        required=False,
        default=None,
        help="YAML SessionConfig file. When provided, input files and output-dir may come from the config.",
        metavar="FILE",
    )
    input_group.add_argument(
        "--imu-timestamps",
        type=Path,
        required=False,
        help="Path to IMU timestamps file (N_imu,) [seconds]",
        metavar="FILE",
    )
    input_group.add_argument(
        "--imu-measurements",
        type=Path,
        required=False,
        help="Path to IMU measurements file. Channel count depends on --state-mode: (N_imu, 3) [ω_z, f_x, f_y] for 2d_full / vision_only and the default 2d_cam_3d_imu (degenerate, vz idle); (N_imu, 4) [ω_z, f_x, f_y, f_z] for 2d_cam_3d_imu with 3D velocity; (N_imu, 6) [ω_x, ω_y, ω_z, f_x, f_y, f_z] for 2d_cam_6dof_imu_orientation. Units: rad/s and m/s².",
        metavar="FILE",
    )
    input_group.add_argument(
        "--camera-timestamps",
        type=Path,
        required=False,
        help="Path to camera timestamps file (N_cam,) [seconds]",
        metavar="FILE",
    )
    input_group.add_argument(
        "--led1-positions",
        type=Path,
        required=False,
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
        required=False,
        help="Directory to save filter outputs",
        metavar="DIR",
    )

    # Filter configuration. Defaults are None sentinels so omitted flags fall
    # through to EKFConfig() / FilterCoreConfig defaults rather than being
    # overridden by stale CLI-side numbers. Help strings pull live values from
    # _FILTER_DEFAULTS so they stay in sync with the dataclass.
    filter_group = parser.add_argument_group("filter parameters (optional)")
    filter_group.add_argument(
        "--process-noise-pos",
        type=float,
        default=None,
        help=f"Position process noise (default: {_FILTER_DEFAULTS.process_noise_pos:.2e} m²/s)",
    )
    filter_group.add_argument(
        "--process-noise-vel",
        type=float,
        default=None,
        help=f"Velocity process noise (default: {_FILTER_DEFAULTS.process_noise_vel:.2e} m²/s³)",
    )
    filter_group.add_argument(
        "--process-noise-heading",
        type=float,
        default=None,
        help=f"Heading process noise (default: {_FILTER_DEFAULTS.process_noise_heading:.2e} rad²/s)",
    )
    filter_group.add_argument(
        "--process-noise-gyro-bias",
        type=float,
        default=None,
        help=f"Gyro bias random walk density (default: {_FILTER_DEFAULTS.process_noise_gyro_bias:.2e} rad²/s³)",
    )
    filter_group.add_argument(
        "--process-noise-accel-bias",
        type=float,
        default=None,
        help=f"Accel bias random walk density (default: {_FILTER_DEFAULTS.process_noise_accel_bias:.2e} m²/s⁵)",
    )
    filter_group.add_argument(
        "--measurement-noise-pos",
        type=float,
        default=None,
        help=f"Position measurement noise variance (default: {_FILTER_DEFAULTS.measurement_noise_pos:.2e} m²)",
    )
    filter_group.add_argument(
        "--imu-gyro-noise-density",
        type=float,
        default=None,
        help=f"IMU gyro noise density (default: {_FILTER_DEFAULTS.imu_gyro_noise_density:.2e} rad/s/√Hz)",
    )
    filter_group.add_argument(
        "--imu-accel-noise-density",
        type=float,
        default=None,
        help=f"IMU accel noise density (default: {_FILTER_DEFAULTS.imu_accel_noise_density:.2e} m/s²/√Hz)",
    )
    filter_group.add_argument(
        "--damping-coeff",
        type=float,
        default=None,
        help=f"Velocity damping coefficient (default: {_FILTER_DEFAULTS.damping_coeff} s⁻¹)",
    )
    filter_group.add_argument(
        "--led-distance",
        type=float,
        default=None,
        help="Expected LED spacing in meters (default: auto-detected from dual-LED frames; must be set explicitly via --led-distance or YAML config.filter.led_distance if no dual-LED frames exist.)",
    )
    filter_group.add_argument(
        "--use-heading-measurement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Use heading pseudo-measurement from dual LEDs (default: {_FILTER_DEFAULTS.use_heading_measurement})",
    )
    filter_group.add_argument(
        "--state-mode",
        type=str,
        default=None,
        choices=(
            "2d_full",
            "vision_only",
            "2d_cam_3d_imu",
            "2d_cam_6dof_imu_orientation",
        ),
        help=(
            f"State layout (default: {_FILTER_DEFAULTS.state_mode}). "
            "`2d_cam_6dof_imu_orientation` requires a 6-channel IMU input "
            "[gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]; "
            "`3d_cam_6dof_imu` requires the experimental "
            "`extended_kalman_filter_3d` entry point. Use the Python API "
            "for that 3D-camera mode."
        ),
    )

    parser.set_defaults(func=run_online)


@friendly_cli_errors
def run_online(args: argparse.Namespace) -> None:
    """Execute the online command.

    The :func:`friendly_cli_errors` decorator converts
    ``FileNotFoundError`` / ``ValueError`` raised by downstream
    library code (e.g. ``EKFConfig`` rejecting
    ``--process-noise-pos -1``) into a clean ``Error: ...`` stderr
    line, mirroring :func:`trodestrack.cli.report.run_report`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # Defer the banner until inputs are validated. Printing it
    # unconditionally up front meant Pydantic / file-load failures
    # appeared *under* a header advertising work that never started,
    # which read like the run had begun.
    if args.config is not None:
        _run_online_from_config(args)
        return

    require_cli_inputs(args, _LEGACY_REQUIRED_ARGS, command="online")
    print("=" * 80)
    print("trodestrack online — Forward-only EKF (batch over full input arrays)")
    print("=" * 80)

    # Load input data
    print("\nLoading input data...")
    t_imu = load_data_file(args.imu_timestamps, "IMU timestamps")
    U_imu = load_data_file(args.imu_measurements, "IMU measurements")
    t_cam = load_data_file(args.camera_timestamps, "Camera timestamps")
    Z_cam_led1 = load_data_file(args.led1_positions, "LED1 positions")

    # Reject malformed inputs at the boundary so the filter does not
    # silently produce NaN/poisoned states from non-finite IMU rows or
    # negative dt from out-of-order timestamps. LED arrays intentionally
    # tolerate NaN (handled via mask_cam downstream).
    validate_monotonic_timestamps(t_imu, "IMU timestamps")
    validate_monotonic_timestamps(t_cam, "Camera timestamps")
    validate_finite_array(U_imu, "IMU measurements")

    n_imu = len(t_imu)
    n_cam = len(t_cam)

    # Validate general IMU data shape. Layout-specific compatibility (e.g.
    # 4-channel only valid for 2d_cam_3d_imu, 6-channel only valid for
    # quaternion orientation) is enforced by validate_imu_input_shape
    # downstream.
    if U_imu.ndim != 2 or U_imu.shape[0] != n_imu or U_imu.shape[1] not in (3, 4, 6):
        print(
            f"Error: IMU measurements shape {U_imu.shape} must be "
            f"(n_imu={n_imu}, 3 | 4 | 6). Use 3 channels [ω_z, f_x, f_y], "
            "4 channels [ω_z, f_x, f_y, f_z] with --state-mode "
            "2d_cam_3d_imu, or 6 channels "
            "[ω_x, ω_y, ω_z, f_x, f_y, f_z] with --state-mode "
            "2d_cam_6dof_imu_orientation.",
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
        mask_cam = validate_camera_mask(
            load_data_file(args.camera_mask, "Camera mask"), n_cam
        )
    else:
        mask_cam = np.ones(n_cam, dtype=bool)

    print(f"  IMU samples: {n_imu} @ {1.0 / np.mean(np.diff(t_imu)):.1f} Hz")
    print(f"  Camera frames: {n_cam} @ {1.0 / np.mean(np.diff(t_cam)):.1f} Hz")
    print(f"  Valid camera frames: {mask_cam.sum()} ({100 * mask_cam.mean():.1f}%)")
    print(f"  Duration: {t_cam[-1] - t_cam[0]:.2f} s")

    # Configure filter. Filter only the kwargs the user explicitly set so
    # EKFConfig falls back to its own defaults for anything not on the CLI.
    # ``led_distance`` is special-cased: extended_kalman_filter auto-detects
    # LED spacing only when ekf_config.led_distance is None, so an omitted
    # ``--led-distance`` must pass ``None`` through (rather than fall back
    # to the FilterCoreConfig default of 0.04 m).
    print("\nConfiguring Extended Kalman Filter...")
    config_overrides = {
        "state_mode": args.state_mode,
        "process_noise_pos": args.process_noise_pos,
        "process_noise_vel": args.process_noise_vel,
        "process_noise_heading": args.process_noise_heading,
        "process_noise_gyro_bias": args.process_noise_gyro_bias,
        "process_noise_accel_bias": args.process_noise_accel_bias,
        "measurement_noise_pos": args.measurement_noise_pos,
        "imu_gyro_noise_density": args.imu_gyro_noise_density,
        "imu_accel_noise_density": args.imu_accel_noise_density,
        "damping_coeff": args.damping_coeff,
        "use_heading_measurement": args.use_heading_measurement,
    }
    ekf_config = EKFConfig(
        led_distance=args.led_distance,  # None -> auto-detect
        **{k: v for k, v in config_overrides.items() if v is not None},
    )

    print(f"  Process noise (pos): {ekf_config.process_noise_pos:.2e} m²/s")
    print(f"  Process noise (vel): {ekf_config.process_noise_vel:.2e} m²/s³")
    print(f"  Damping coefficient: {ekf_config.damping_coeff:.2f} s⁻¹")
    print(f"  LED heading measurement: {ekf_config.use_heading_measurement}")

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
        f.write(
            "trodestrack online — Forward-only EKF Results (batch over full inputs)\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write("Input Files:\n")
        f.write(f"  IMU timestamps: {args.imu_timestamps}\n")
        f.write(f"  IMU measurements: {args.imu_measurements}\n")
        f.write(f"  Camera timestamps: {args.camera_timestamps}\n")
        f.write(f"  LED1 positions: {args.led1_positions}\n")
        f.write(f"  LED2 positions: {args.led2_positions}\n")
        f.write(f"  Camera mask: {args.camera_mask}\n\n")
        f.write("Filter Configuration (effective values):\n")
        f.write(f"  Process noise (pos): {ekf_config.process_noise_pos}\n")
        f.write(f"  Process noise (vel): {ekf_config.process_noise_vel}\n")
        f.write(f"  Process noise (heading): {ekf_config.process_noise_heading}\n")
        f.write(f"  Process noise (gyro bias): {ekf_config.process_noise_gyro_bias}\n")
        f.write(
            f"  Process noise (accel bias): {ekf_config.process_noise_accel_bias}\n"
        )
        f.write(f"  Measurement noise (pos): {ekf_config.measurement_noise_pos}\n")
        f.write(f"  IMU gyro noise density: {ekf_config.imu_gyro_noise_density}\n")
        f.write(f"  IMU accel noise density: {ekf_config.imu_accel_noise_density}\n")
        f.write(f"  Damping coefficient: {ekf_config.damping_coeff}\n")
        if (
            args.led_distance is None
            and filter_result.estimated_led_distance is not None
        ):
            f.write(
                f"  LED distance: {filter_result.estimated_led_distance:.4f} m (auto-detected)\n"
            )
        else:
            f.write(f"  LED distance: {ekf_config.led_distance}\n")
        f.write(f"  Use heading measurement: {ekf_config.use_heading_measurement}\n")
        f.write(f"  State mode: {ekf_config.state_mode}\n\n")
        f.write("Results:\n")
        f.write(f"  Marginal log-likelihood: {filter_result.marginal_loglik:.6f}\n")
        f.write(f"  Output samples: {n_cam}\n")
        f.write(f"  State dimension: {filter_result.filtered_means.shape[1]}\n")

    # Derive state dimension from results
    n_state = filter_result.filtered_means.shape[1]
    n_cov_flat = n_state * n_state

    print("\nOutput files:")
    print(f"  filtered_means.txt: Filter state estimates ({n_cam}, {n_state})")
    print(f"  filtered_covariances.txt: Filter covariances ({n_cam}, {n_cov_flat})")
    print("  marginal_loglik.txt: Marginal log-likelihood (scalar)")
    print("  metadata.txt: Run configuration and metadata")
    print("\nFiltering complete!")
    print(
        "\nNOTE: For improved accuracy, use 'trodestrack smooth' to apply RTS smoothing."
    )


def _run_online_from_config(args: argparse.Namespace) -> None:
    run = prepare_config_filter_run(args)
    save_filter_outputs(run)
    with open(run.output_dir / "marginal_loglik.txt", "w") as f:
        f.write(f"{run.filter_result.marginal_loglik:.6f}\n")
    write_config_metadata(
        run,
        title="trodestrack online — Config-driven forward-only EKF",
        marginal_loglik=float(run.filter_result.marginal_loglik),
        state_dim=run.filter_result.filtered_means.shape[1],
    )
    if run.config.outputs.write_diagnostics:
        write_session_diagnostics(run.session, run.output_dir, run.safety_report)

    print(f"\nSaved config-driven online outputs to {run.output_dir}/")

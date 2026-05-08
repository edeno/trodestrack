"""Shared helpers for YAML-configured CLI runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trodestrack.config import SessionConfig, load_session_config
from trodestrack.io import (
    PreparedSession,
    SafetyReport,
    load_session,
    run_real_data_safety_check,
    uses_imu_fusion,
    write_session_diagnostics,
)
from trodestrack.models.ekf import EKFConfig, EKFResult, extended_kalman_filter


@dataclass(frozen=True)
class ConfigFilterRun:
    """Prepared config-driven run after forward filtering and safety checks."""

    config_path: Path
    config: SessionConfig
    session: PreparedSession
    ekf_config: EKFConfig
    filter_result: EKFResult
    safety_report: SafetyReport
    output_dir: Path


def prepare_config_filter_run(args: argparse.Namespace) -> ConfigFilterRun:
    """Load YAML config, run EKF, and apply calibration/safety gates.

    The CLI banner prints just before ``print_config_session_summary``
    so every pre-EKF failure mode surfaces without a header framing
    the error: YAML schema (Pydantic), missing input files, missing
    parquet columns (``load_session``), and the IMU-calibration gate
    all raise before the banner is printed.
    """

    config = load_session_config(args.config)
    if args.output_dir is not None:
        config = config.model_copy(
            update={
                "outputs": config.outputs.model_copy(
                    update={"output_dir": args.output_dir}
                )
            }
        )
    session = load_session(config)
    output_dir = config.outputs.output_dir

    calibration_error = session.diagnostics.get("imu_calibration_error")
    if (
        calibration_error is not None
        and config.imu.require_calibration_for_fusion
        and uses_imu_fusion(config)
    ):
        write_session_diagnostics(session, output_dir)
        raise ValueError(f"IMU calibration diagnostics failed: {calibration_error}")

    ekf_config = EKFConfig(
        **config.filter.to_ekf_kwargs(led_distance=session.led_distance)
    )
    print("=" * 80)
    print(f"trodestrack — config-driven run from {args.config}")
    print("=" * 80)
    print_config_session_summary(config, session, ekf_config)

    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=session.t_imu,
        U_imu=session.U_imu,
        t_cam=session.t_cam,
        Z_cam_led1=session.Z_cam_led1,
        Z_cam_led2=session.Z_cam_led2,
        mask_cam=session.mask_cam,
        conf_cam=session.conf_cam,
        event_source_anchors=session.event_source_anchors,
        event_source_covariances=session.event_source_covariances,
        event_indices_per_frame=session.event_indices_per_frame,
    )
    # ``run_real_data_safety_check`` can raise *before* returning a
    # ``SafetyReport`` (e.g. no finite dual-LED frame, or no finite
    # fused/vision position pair). The two ``raise ValueError`` sites
    # in ``io/session.py`` produce informative messages but skip the
    # diagnostics dump that the not-passed branch below performs, so
    # the user is left with a one-line error and no
    # ``session_diagnostics.json`` to inspect. Mirror the
    # not-passed handling: write loader/calibration diagnostics
    # (``safety_report=None`` since none was produced), then re-raise.
    try:
        safety_report = run_real_data_safety_check(session, ekf_config, filter_result)
    except ValueError:
        write_session_diagnostics(session, output_dir)
        raise
    if not safety_report.passed:
        write_session_diagnostics(session, output_dir, safety_report)
        raise ValueError(safety_report.message)

    return ConfigFilterRun(
        config_path=Path(args.config),
        config=config,
        session=session,
        ekf_config=ekf_config,
        filter_result=filter_result,
        safety_report=safety_report,
        output_dir=output_dir,
    )


def print_config_session_summary(
    config: SessionConfig, session: PreparedSession, ekf_config: EKFConfig
) -> None:
    """Print a compact summary matching the legacy CLI's verbose style."""

    print("\nLoaded config-driven session:")
    print(f"  Source: {config.inputs.format}")
    print(f"  IMU samples: {len(session.t_imu)}")
    print(f"  Camera frames: {len(session.t_cam)}")
    print(f"  Valid camera frames: {int(session.mask_cam.sum())}")
    print(f"  State mode: {ekf_config.state_mode}")
    led_diag = session.diagnostics.get("led_identity")
    if isinstance(led_diag, dict):
        print(
            "  LED identity: "
            f"{led_diag.get('n_swapped', 0)} swapped frames "
            f"(initial_state={led_diag.get('initial_state')})"
        )
        if led_diag.get("global_identity_ambiguous"):
            print(
                "  LED identity warning: initial_state='auto' cannot determine "
                "a whole-session front/back label convention."
            )
    ttl_diag = session.diagnostics.get("ttl_events")
    if isinstance(ttl_diag, dict):
        print(
            "  TTL events: "
            f"{ttl_diag.get('n_sources', 0)} configured source(s); "
            f"{ttl_diag.get('n_events_kept', 0)}/"
            f"{ttl_diag.get('n_events_total', 0)} events kept "
            f"(max {ttl_diag.get('max_events_per_frame', 0)}/frame)"
        )


def save_filter_outputs(run: ConfigFilterRun) -> None:
    """Write forward-filter output files shared by online and smooth.

    Writes the legacy text-format files (``filtered_means.txt``,
    ``filtered_covariances.txt``) plus the camera-frame side-data the
    filter consumed (timestamps, post-correction LED arrays, mask,
    optional confidence) so downstream code can reconstruct the
    inputs alongside the outputs without re-running the loader.
    Also bundles everything into ``filter_outputs.npz`` for easy
    NumPy / pandas consumption — the text files alone require the
    user to remember that ``filtered_covariances.txt`` is reshaped
    ``(n_cam, n*n)`` and to thread the camera timestamps in from a
    separate file.
    """

    n_cam = len(run.session.t_cam)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(run.output_dir / "filtered_means.txt", run.filter_result.filtered_means)
    np.savetxt(
        run.output_dir / "filtered_covariances.txt",
        run.filter_result.filtered_covariances.reshape(n_cam, -1),
    )

    np.savetxt(run.output_dir / "t_cam.txt", run.session.t_cam)
    np.savetxt(run.output_dir / "Z_cam_led1.txt", run.session.Z_cam_led1)
    np.savetxt(run.output_dir / "Z_cam_led2.txt", run.session.Z_cam_led2)
    np.savetxt(
        run.output_dir / "mask_cam.txt", run.session.mask_cam.astype(int), fmt="%d"
    )
    if run.session.conf_cam is not None:
        np.savetxt(run.output_dir / "conf_cam.txt", run.session.conf_cam)

    bundle = {
        "t_cam": run.session.t_cam,
        "Z_cam_led1": run.session.Z_cam_led1,
        "Z_cam_led2": run.session.Z_cam_led2,
        "mask_cam": run.session.mask_cam,
        "filtered_means": np.asarray(run.filter_result.filtered_means),
        "filtered_covariances": np.asarray(run.filter_result.filtered_covariances),
        "marginal_loglik": np.asarray(run.filter_result.marginal_loglik),
    }
    if run.session.conf_cam is not None:
        bundle["conf_cam"] = run.session.conf_cam
    if run.session.event_source_anchors is not None:
        bundle["event_source_anchors"] = run.session.event_source_anchors
        bundle["event_source_covariances"] = run.session.event_source_covariances
        bundle["event_indices_per_frame"] = run.session.event_indices_per_frame
        bundle["event_source_ids"] = np.asarray(
            [src.source_id for src in run.session.event_sources], dtype=np.int64
        )
        bundle["event_source_types"] = np.asarray(
            [src.source_type for src in run.session.event_sources]
        )
    np.savez(run.output_dir / "filter_outputs.npz", **bundle)


def write_config_metadata(
    run: ConfigFilterRun,
    *,
    title: str,
    marginal_loglik: float,
    state_dim: int,
    smoother_num_iter: int | None = None,
) -> None:
    """Write reproducibility metadata for YAML-configured CLI runs."""

    with (run.output_dir / "metadata.txt").open("w") as f:
        f.write(title + "\n")
        f.write("=" * 80 + "\n\n")
        f.write("Config:\n")
        f.write(f"  Path: {run.config_path}\n")
        f.write(f"  Input format: {run.config.inputs.format}\n")
        f.write(f"  Output directory: {run.output_dir}\n\n")
        f.write("Session:\n")
        f.write(f"  IMU samples: {len(run.session.t_imu)}\n")
        f.write(f"  Camera frames: {len(run.session.t_cam)}\n")
        f.write(f"  Valid camera frames: {int(run.session.mask_cam.sum())}\n")
        f.write(f"  LED distance: {run.ekf_config.led_distance}\n\n")
        f.write("Filter Configuration (effective values):\n")
        f.write(f"  State mode: {run.ekf_config.state_mode}\n")
        f.write(f"  Process noise (pos): {run.ekf_config.process_noise_pos}\n")
        f.write(f"  Process noise (vel): {run.ekf_config.process_noise_vel}\n")
        f.write(f"  Process noise (heading): {run.ekf_config.process_noise_heading}\n")
        f.write(
            f"  Process noise (gyro bias): {run.ekf_config.process_noise_gyro_bias}\n"
        )
        f.write(
            f"  Process noise (accel bias): {run.ekf_config.process_noise_accel_bias}\n"
        )
        f.write(f"  Measurement noise (pos): {run.ekf_config.measurement_noise_pos}\n")
        f.write(
            f"  Measurement noise (heading): "
            f"{run.ekf_config.measurement_noise_heading}\n"
        )
        f.write(f"  IMU gyro noise density: {run.ekf_config.imu_gyro_noise_density}\n")
        f.write(
            f"  IMU accel noise density: {run.ekf_config.imu_accel_noise_density}\n"
        )
        f.write(f"  IMU gravity body: {run.ekf_config.imu_gravity_body}\n")
        f.write(f"  Damping coefficient: {run.ekf_config.damping_coeff}\n")
        f.write(
            "  Experimental accel translation: "
            f"{run.ekf_config.enable_experimental_accel_translation}\n"
        )
        f.write(
            "  Gravity orientation update: "
            f"{run.ekf_config.use_gravity_orientation_update}\n"
        )
        f.write(
            f"  Use heading measurement: {run.ekf_config.use_heading_measurement}\n"
        )
        f.write(f"  Mahalanobis gating: {run.ekf_config.use_mahalanobis_gating}\n")
        f.write(f"  ZUPT enabled: {run.ekf_config.enable_zupt}\n\n")
        if smoother_num_iter is not None:
            f.write("Smoother Configuration:\n")
            f.write(f"  IEKS iterations: {smoother_num_iter}\n\n")
        led_diag = run.session.diagnostics.get("led_identity")
        if isinstance(led_diag, dict):
            f.write("LED Identity:\n")
            f.write(f"  Mode: {led_diag.get('mode')}\n")
            f.write(f"  Initial state: {led_diag.get('initial_state')}\n")
            f.write(f"  Swapped frames: {led_diag.get('n_swapped')}\n")
            f.write(f"  Fraction swapped: {led_diag.get('fraction_swapped')}\n")
            if led_diag.get("global_identity_ambiguous"):
                f.write(
                    "  Warning: initial_state='auto' resolves continuity breaks "
                    "but cannot determine a whole-session front/back convention.\n"
                )
            f.write("\n")
        ttl_diag = run.session.diagnostics.get("ttl_events")
        if isinstance(ttl_diag, dict):
            f.write("TTL Events:\n")
            f.write(f"  Events file: {ttl_diag.get('events_file')}\n")
            f.write(f"  Configured sources: {ttl_diag.get('n_sources')}\n")
            f.write(
                f"  Events kept / total: {ttl_diag.get('n_events_kept')} / "
                f"{ttl_diag.get('n_events_total')}\n"
            )
            f.write(f"  Max events per frame: {ttl_diag.get('max_events_per_frame')}\n")
            source_ids = ttl_diag.get("source_ids")
            source_types = ttl_diag.get("source_types")
            if source_ids is not None and source_types is not None:
                f.write(
                    f"  Source ids / types: "
                    f"{list(zip(source_ids, source_types, strict=False))}\n"
                )
            f.write("\n")
        f.write("Safety Check:\n")
        f.write(f"  Result: {run.safety_report.message}\n")
        f.write(f"  Dual-LED frames: {run.safety_report.dual_led_frame_count}\n")
        f.write(f"  Deviation frames: {run.safety_report.deviation_frame_count}\n")
        f.write(f"  Max fused speed: {run.safety_report.max_fused_speed_mps}\n")
        f.write(
            "  Max vision-position deviation: "
            f"{run.safety_report.max_vision_position_deviation_m}\n"
        )
        f.write(
            "  P95 vision-position deviation: "
            f"{run.safety_report.p95_vision_position_deviation_m}\n"
        )
        f.write(f"  Vision log-likelihood: {run.safety_report.vision_loglik}\n")
        f.write(f"  Fused log-likelihood: {run.safety_report.fused_loglik}\n\n")
        f.write("Results:\n")
        f.write(f"  Marginal log-likelihood: {marginal_loglik:.6f}\n")
        f.write(f"  Output samples: {len(run.session.t_cam)}\n")
        f.write(f"  State dimension: {state_dim}\n")

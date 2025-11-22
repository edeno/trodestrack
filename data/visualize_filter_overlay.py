"""Visualize Arthur session with filter overlay and IMU data.

The rendering pipeline supports two modes:
1) Single-process Matplotlib animation saved via FFMpegWriter
2) Multi-process PNG rendering stitched by ffmpeg (fast & robust for long clips)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

# Headless, deterministic rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_arthur_session import (
    SessionData,
    convert_meters_to_pixels,
    get_video_info,
    load_arthur_session,
    load_video_frame,
)
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlayStyle:
    led_marker_size: float = 10.0
    trajectory_length: int = 90  # frames
    heading_arrow_px: float = 40.0
    heading_flip: bool = True  # camera convention; add π to point forward
    show_uncertainty_circle: bool = False
    invert_yaxis: bool = False  # set True if (0,0) is top-left in the video


@dataclass(frozen=True)
class AxesLimits:
    gyro_ylim: tuple[float, float] = (-200.0, 200.0)
    accel_ylim: tuple[float, float] = (-15.0, 15.0)
    heading_ylim: tuple[float, float] = (-180.0, 180.0)
    vel_max_default: float = 150.0  # cm/s
    pos_unc_min: float = 2.0  # cm


@dataclass(frozen=True)
class RenderConfig:
    start_time: float = 0.0
    duration: float = 10.0
    fps: float = 30.0
    imu_window_s: float = 2.0
    state_window_s: float = 5.0
    dpi: int = 100
    render_mode: str = "single_process"  # "single_process" | "parallel_png"
    max_workers: int | None = None
    ffmpeg_threads: int | None = None
    bitrate_kbps: int = 4000
    style: OverlayStyle = field(default_factory=OverlayStyle)
    limits: AxesLimits = field(default_factory=AxesLimits)

    def n_frames(self) -> int:
        return int(self.duration * self.fps)


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Please install ffmpeg (e.g., brew install ffmpeg or apt-get install ffmpeg)."
        )


def nearest_indices(source_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Map each query time to the nearest index of source_times using searchsorted."""
    idx = np.searchsorted(source_times, query_times, side="left")
    idx = np.clip(idx, 0, len(source_times) - 1)
    return idx


def extract_time_window(
    timestamps: np.ndarray, data: np.ndarray, center_time: float, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    half = window_s / 2.0
    mask = (timestamps >= center_time - half) & (timestamps <= center_time + half)
    return timestamps[mask], data[mask]


def position_std_cm(P_filter_xy: np.ndarray) -> np.ndarray:
    """Euclidean positional std (cm) from 2x2 covariance slices.

    Uses sqrt(trace(P_xy)) which equals RMS spread in x and y.
    """
    # P_filter_xy: [T, 2, 2]
    tr = np.trace(P_filter_xy, axis1=1, axis2=2)
    pos_std = np.sqrt(np.clip(tr, 0.0, None))
    return pos_std * 100.0  # meters -> cm


# --------------------------------------------------------------------------------------
# Figure layout
# --------------------------------------------------------------------------------------


def setup_figure(style: OverlayStyle) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = fig.add_gridspec(
        3, 3, hspace=0.35, wspace=0.35, left=0.05, right=0.97, top=0.95, bottom=0.05
    )

    ax_video = fig.add_subplot(gs[:2, :2])
    ax_video.set_aspect("equal")
    ax_video.axis("off")
    ax_video.set_title(
        "Video with Filter Overlay", fontsize=12, fontweight="bold", pad=10
    )

    ax_gyro = fig.add_subplot(gs[0, 2])
    ax_gyro.set_ylabel("Gyro (deg/s)", fontsize=9)
    ax_gyro.grid(True, alpha=0.3)
    ax_gyro.set_title("Gyroscope (3-axis)", fontsize=10, fontweight="bold")
    ax_gyro.tick_params(labelsize=8)

    ax_accel = fig.add_subplot(gs[1, 2])
    ax_accel.set_ylabel("Accel (m/s²)", fontsize=9)
    ax_accel.grid(True, alpha=0.3)
    ax_accel.set_title("Accelerometer (3-axis)", fontsize=10, fontweight="bold")
    ax_accel.tick_params(labelsize=8)

    ax_pos_vel = fig.add_subplot(gs[2, 0])
    ax_pos_vel.set_xlabel("Time (s)", fontsize=9)
    ax_pos_vel.set_ylabel("Speed (cm/s)", fontsize=9)
    ax_pos_vel.grid(True, alpha=0.3)
    ax_pos_vel.set_title("Velocity Magnitude", fontsize=10, fontweight="bold")
    ax_pos_vel.tick_params(labelsize=8)

    ax_heading = fig.add_subplot(gs[2, 1])
    ax_heading.set_xlabel("Time (s)", fontsize=9)
    ax_heading.set_ylabel("Heading (deg)", fontsize=9)
    ax_heading.grid(True, alpha=0.3)
    ax_heading.set_title("Heading Estimate", fontsize=10, fontweight="bold")
    ax_heading.tick_params(labelsize=8)

    ax_uncertainty = fig.add_subplot(gs[2, 2])
    ax_uncertainty.set_xlabel("Time (s)", fontsize=9)
    ax_uncertainty.set_ylabel("Position std (cm)", fontsize=9)
    ax_uncertainty.grid(True, alpha=0.3)
    ax_uncertainty.set_title("Position Uncertainty", fontsize=10, fontweight="bold")
    ax_uncertainty.tick_params(labelsize=8)

    axes = {
        "video": ax_video,
        "gyro": ax_gyro,
        "accel": ax_accel,
        "pos_vel": ax_pos_vel,
        "heading": ax_heading,
        "uncertainty": ax_uncertainty,
    }

    if style.invert_yaxis:
        ax_video.invert_yaxis()

    return fig, axes


# --------------------------------------------------------------------------------------
# Rendering (single-process animation)
# --------------------------------------------------------------------------------------


def create_filter_overlay_video(
    *,
    video_path: str,
    data: SessionData,
    filter_result,
    t_filter: np.ndarray,
    position_df: pd.DataFrame,
    output_path: str,
    config: RenderConfig = RenderConfig(),
) -> None:
    """Create video with filter overlay and IMU/state visualization."""

    require_ffmpeg()

    video_info = get_video_info(video_path)
    if not video_info:
        raise RuntimeError("Could not read video info. Is opencv-python installed?")

    start_time = config.start_time
    duration = config.duration
    fps = config.fps

    end_time = start_time + duration
    max_time = min(data.t_cam[-1], video_info["duration_s"])
    if end_time > max_time:
        print(
            f"Warning: Requested end time {end_time:.1f}s exceeds data ({max_time:.1f}s)"
        )
        end_time = max_time
        duration = max(0.0, end_time - start_time)
        print(f"  Adjusted duration to {duration:.1f}s")

    # Extract filter state blocks
    try:
        X = np.asarray(filter_result.filtered_means)  # [T, n]
        P = np.asarray(filter_result.filtered_covariances)  # [T, n, n]
    except AttributeError:
        X = np.asarray(filter_result.smoothed_means)
        P = np.asarray(filter_result.smoothed_covariances)

    pos_filter_m = X[:, :2]  # meters
    vel_filter = X[:, 2:5]  # m/s (vx, vy, vz)
    heading_rad = X[:, 5]

    # Uncertainty (cm) and velocity magnitude (cm/s)
    pos_std_cm_series = position_std_cm(P[:, :2, :2])
    vel_mag_cms = np.hypot(vel_filter[:, 0], vel_filter[:, 1]) * 100.0

    # 1σ uncertainties for speed (cm/s) and heading (deg)
    # Project the (vx, vy) covariance onto the instantaneous speed direction
    Cov_v = P[:, 2:4, 2:4]  # covariance of (vx, vy)
    speed = np.hypot(vel_filter[:, 0], vel_filter[:, 1])
    eps = 1e-6
    ux = np.where(speed > eps, vel_filter[:, 0] / speed, 1.0)
    uy = np.where(speed > eps, vel_filter[:, 1] / speed, 0.0)
    u = np.stack([ux, uy], axis=1)  # [T, 2]
    # Robust σ_speed (m/s): project onto velocity direction; fall back to radial RMS near zero speed
    proj = np.einsum("ti,tij,tj->t", u, Cov_v, u)
    radial_rms = 0.5 * (Cov_v[:, 0, 0] + Cov_v[:, 1, 1])  # ≈ mean of Var(vx), Var(vy)
    vel_sigma_mps = np.where(
        speed > eps,
        np.sqrt(np.clip(proj, 0.0, None)),
        np.sqrt(np.clip(radial_rms, 0.0, None)),
    )
    vel_sigma_cms = vel_sigma_mps * 100.0  # m/s -> cm/s

    heading_sigma_deg = np.degrees(
        np.sqrt(np.clip(P[:, 5, 5], 0.0, None))
    )  # rad -> deg
    print(
        f"σ_speed cm/s — median: {np.nanmedian(vel_sigma_cms):.2f}, 95th: {np.nanpercentile(vel_sigma_cms, 95):.2f}"
    )
    print(
        f"σ_heading deg — median: {np.nanmedian(heading_sigma_deg):.2f}, 95th: {np.nanpercentile(heading_sigma_deg, 95):.2f}"
    )

    # Convert to pixels for overlay
    pos_filter_px = convert_meters_to_pixels(pos_filter_m, data.meters_per_pixel)
    led1_px = convert_meters_to_pixels(data.Z_cam_led1, data.meters_per_pixel)
    led2_px = convert_meters_to_pixels(data.Z_cam_led2, data.meters_per_pixel)

    # IMU: rad/s -> deg/s; accel already m/s²
    gyro = np.degrees(data.U_imu[:, :3])
    accel = data.U_imu[:, 3:6]

    # Frame indexing
    n_frames = int(duration * fps)
    frame_times = start_time + (np.arange(n_frames) / fps)

    fidx = nearest_indices(t_filter, frame_times)
    cidx = nearest_indices(data.t_cam, frame_times)

    pos_vid_inds = position_df["video_frame_ind"].to_numpy()
    vidx = pos_vid_inds[fidx]

    # Parallel PNG mode is handled elsewhere
    if config.render_mode == "parallel_png":
        _render_parallel_png(
            video_path=video_path,
            output_path=output_path,
            n_frames=n_frames,
            fps=fps,
            dpi=config.dpi,
            max_workers=config.max_workers,
            frame_times=frame_times,
            frame_to_filter_idx=fidx,
            frame_to_cam_idx=cidx,
            frame_to_video_ind=vidx,
            video_info=video_info,
            data=data,
            pos_filter_pixels=pos_filter_px,
            heading_filter=heading_rad,
            led1_pixels=led1_px,
            led2_pixels=led2_px,
            vel_mag=vel_mag_cms,
            pos_std_cm=pos_std_cm_series,
            vel_sigma=vel_sigma_cms,
            heading_sigma=heading_sigma_deg,
            config=config,
        )
        print(f"\n✓ Video saved to: {output_path}")
        return

    # Single-process animation path: cache frames with threads
    print(f"\nCaching {n_frames} video frames for animation…")
    cached_frames: list[np.ndarray | None] = [None] * n_frames

    def read_one(idx):
        return load_video_frame(video_path, int(idx))

    workers = (
        min(8, (os.cpu_count() or 4))
        if config.max_workers is None
        else config.max_workers
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(read_one, int(v)): i for i, v in enumerate(vidx)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                cached_frames[i] = fut.result()
            except Exception as exc:  # pragma: no cover
                print(f"Frame {i} read error: {exc}")
                cached_frames[i] = None

    fig, axes = setup_figure(config.style)

    # Video image
    video_frame = axes["video"].imshow(
        np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8),
        animated=True,
    )

    # Trajectory and markers
    (trajectory_line,) = axes["video"].plot(
        [], [], "c-", linewidth=1.5, alpha=0.6, label="Filter trajectory"
    )
    led1_circle = Circle(
        (0, 0),
        config.style.led_marker_size,
        color="red",
        alpha=0.5,
        label="LED1 (back)",
    )
    led2_circle = Circle(
        (0, 0),
        config.style.led_marker_size,
        color="yellow",
        alpha=0.5,
        label="LED2 (front)",
    )
    axes["video"].add_patch(led1_circle)
    axes["video"].add_patch(led2_circle)

    filter_circle = Circle(
        (0, 0),
        config.style.led_marker_size * 1.2,
        color="cyan",
        alpha=0.8,
        linewidth=2,
        fill=False,
        label="Filter estimate",
    )
    axes["video"].add_patch(filter_circle)
    (heading_line_video,) = axes["video"].plot(
        [], [], color="cyan", linewidth=2, alpha=0.7, label="Heading"
    )
    axes["video"].legend(loc="upper right", fontsize=9, framealpha=0.9)

    # IMU plots
    (gyro_x_line,) = axes["gyro"].plot([], [], "-", linewidth=1.2, label="X", alpha=0.8)
    (gyro_y_line,) = axes["gyro"].plot([], [], "-", linewidth=1.2, label="Y", alpha=0.8)
    (gyro_z_line,) = axes["gyro"].plot([], [], "-", linewidth=1.2, label="Z", alpha=0.8)
    gyro_marker = axes["gyro"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )
    axes["gyro"].legend(loc="upper right", fontsize=7)

    (accel_x_line,) = axes["accel"].plot(
        [], [], "-", linewidth=1.2, label="X", alpha=0.8
    )
    (accel_y_line,) = axes["accel"].plot(
        [], [], "-", linewidth=1.2, label="Y", alpha=0.8
    )
    (accel_z_line,) = axes["accel"].plot(
        [], [], "-", linewidth=1.2, label="Z", alpha=0.8
    )
    accel_marker = axes["accel"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )
    axes["accel"].legend(loc="upper right", fontsize=7)

    # State plots
    (vel_line,) = axes["pos_vel"].plot([], [], "-", linewidth=1.5, label="Speed")
    vel_fill = axes["pos_vel"].fill_between([], [], [], alpha=0.2, label="±1σ")
    vel_marker = axes["pos_vel"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    (heading_line_ax,) = axes["heading"].plot(
        [], [], "-", linewidth=1.5, label="Heading"
    )
    heading_fill = axes["heading"].fill_between([], [], [], alpha=0.2, label="±1σ")
    heading_marker = axes["heading"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    (unc_line,) = axes["uncertainty"].plot(
        [], [], "-", linewidth=1.5, label="Position σ"
    )
    unc_marker = axes["uncertainty"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    time_text = axes["video"].text(
        0.02,
        0.98,
        "",
        transform=axes["video"].transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    def init():
        """Initialize blit artists."""
        return (
            video_frame,
            trajectory_line,
            led1_circle,
            led2_circle,
            filter_circle,
            heading_line_video,
            gyro_x_line,
            gyro_y_line,
            gyro_z_line,
            gyro_marker,
            accel_x_line,
            accel_y_line,
            accel_z_line,
            accel_marker,
            vel_line,
            vel_fill,
            vel_marker,
            heading_line_ax,
            heading_fill,
            heading_marker,
            unc_line,
            unc_marker,
            time_text,
        )

    def update(k: int):
        """Advance all artists to frame index k."""
        current_time = frame_times[k]
        fi = fidx[k]
        ci = cidx[k]

        frame = cached_frames[k]
        if frame is not None:
            video_frame.set_data(frame)

        # Filter position & heading
        px, py = pos_filter_px[fi]
        trajectory_start = max(0, fi - config.style.trajectory_length)
        traj = pos_filter_px[trajectory_start : fi + 1]
        trajectory_line.set_data(traj[:, 0], traj[:, 1])

        led1_circle.center = tuple(led1_px[ci])
        led2_circle.center = tuple(led2_px[ci])
        filter_circle.center = (px, py)

        arrow_angle = heading_rad[fi] + (math.pi if config.style.heading_flip else 0.0)
        dx = config.style.heading_arrow_px * math.cos(arrow_angle)
        dy = config.style.heading_arrow_px * math.sin(arrow_angle)
        heading_line_video.set_data([px, px + dx], [py, py + dy])

        # Optional uncertainty circle
        if config.style.show_uncertainty_circle:
            # Remove existing circle if present by replacing the artist data-less trick
            # (for performance, a persistent artist with updated radius is cheaper)
            radius_px = (pos_std_cm_series[fi] / 100.0) / data.meters_per_pixel
            # ensure it exists once and update radius/center
            if not hasattr(update, "unc_patch"):
                update.unc_patch = Circle(
                    (px, py), radius_px, edgecolor="cyan", facecolor="none", alpha=0.3
                )
                axes["video"].add_patch(update.unc_patch)
            update.unc_patch.center = (px, py)
            update.unc_patch.radius = radius_px

        # IMU windows
        t_imu = data.t_imu
        half = config.imu_window_s / 2.0
        mask = (t_imu >= current_time - half) & (t_imu <= current_time + half)

        gx, gy, gz = gyro[mask].T if mask.any() else (np.array([]),) * 3
        tx = t_imu[mask]
        ax, ay, az = accel[mask].T if mask.any() else (np.array([]),) * 3

        gyro_x_line.set_data(tx, gx)
        gyro_y_line.set_data(tx, gy)
        gyro_z_line.set_data(tx, gz)
        axes["gyro"].set_xlim(current_time - half, current_time + half)
        axes["gyro"].set_ylim(*config.limits.gyro_ylim)
        gyro_marker.set_xdata([current_time, current_time])

        accel_x_line.set_data(tx, ax)
        accel_y_line.set_data(tx, ay)
        accel_z_line.set_data(tx, az)
        axes["accel"].set_xlim(current_time - half, current_time + half)
        axes["accel"].set_ylim(*config.limits.accel_ylim)
        accel_marker.set_xdata([current_time, current_time])

        # State windows (use t_filter for alignment with EKF output)
        half_s = config.state_window_s / 2.0
        mstate = (t_filter >= current_time - half_s) & (
            t_filter <= current_time + half_s
        )
        ts = t_filter[mstate]

        vel_w = vel_mag_cms[mstate]
        hdg_w = np.degrees(
            np.arctan2(np.sin(heading_rad[mstate]), np.cos(heading_rad[mstate]))
        )
        unc_w = pos_std_cm_series[mstate]

        # Update uncertainty fills (±2σ)
        k_sigma = 2.0  # for ±2σ fill
        vel_sig_w = k_sigma * vel_sigma_cms[mstate]
        hdg_sig_w = k_sigma * heading_sigma_deg[mstate]

        if len(ts):
            v_lower = vel_w - vel_sig_w
            v_upper = vel_w + vel_sig_w
            v_verts = np.concatenate(
                [
                    np.column_stack([ts, v_lower]),
                    np.column_stack([ts[::-1], v_upper[::-1]]),
                ]
            )
            vel_fill.set_verts([v_verts])

            h_lower = ((hdg_w - hdg_sig_w + 180.0) % 360.0) - 180.0
            h_upper = ((hdg_w + hdg_sig_w + 180.0) % 360.0) - 180.0
            h_verts = np.concatenate(
                [
                    np.column_stack([ts, h_lower]),
                    np.column_stack([ts[::-1], h_upper[::-1]]),
                ]
            )
            heading_fill.set_verts([h_verts])
        else:
            vel_fill.set_verts([])
            heading_fill.set_verts([])

        vel_line.set_data(ts, vel_w)
        axes["pos_vel"].set_xlim(current_time - half_s, current_time + half_s)
        vmax = (
            float(np.max(vel_w) * 1.2) if len(vel_w) else config.limits.vel_max_default
        )
        axes["pos_vel"].set_ylim(0.0, max(config.limits.vel_max_default, vmax))
        vel_marker.set_xdata([current_time, current_time])

        heading_line_ax.set_data(ts, hdg_w)
        axes["heading"].set_xlim(current_time - half_s, current_time + half_s)
        axes["heading"].set_ylim(*config.limits.heading_ylim)
        heading_marker.set_xdata([current_time, current_time])

        unc_line.set_data(ts, unc_w)
        axes["uncertainty"].set_xlim(current_time - half_s, current_time + half_s)
        umax = float(np.max(unc_w) * 1.2) if len(unc_w) else config.limits.pos_unc_min
        axes["uncertainty"].set_ylim(0.0, max(config.limits.pos_unc_min, umax))
        unc_marker.set_xdata([current_time, current_time])

        time_text.set_text(f"t = {current_time:.2f} s")

        return (
            video_frame,
            trajectory_line,
            led1_circle,
            led2_circle,
            filter_circle,
            heading_line_video,
            gyro_x_line,
            gyro_y_line,
            gyro_z_line,
            gyro_marker,
            accel_x_line,
            accel_y_line,
            accel_z_line,
            accel_marker,
            vel_line,
            vel_fill,
            vel_marker,
            heading_line_ax,
            heading_fill,
            heading_marker,
            unc_line,
            unc_marker,
            time_text,
        )

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=True
    )

    threads = (
        config.ffmpeg_threads
        if config.ffmpeg_threads is not None
        else min(4, os.cpu_count() or 2)
    )
    writer = FFMpegWriter(
        fps=fps,
        bitrate=config.bitrate_kbps,
        extra_args=["-threads", str(threads), "-pix_fmt", "yuv420p"],
    )

    t0 = time.time()
    anim.save(output_path, writer=writer, dpi=max(72, config.dpi))
    plt.close(fig)
    print(
        f"✓ Video generation complete in {time.time() - t0:.1f}s.\n✓ Video saved to: {output_path}"
    )


# --------------------------------------------------------------------------------------
# Rendering (multi-process PNG -> ffmpeg)
# --------------------------------------------------------------------------------------


def _render_parallel_png(
    *,
    video_path: str,
    output_path: str,
    n_frames: int,
    fps: float,
    dpi: int,
    max_workers: int | None,
    frame_times: np.ndarray,
    frame_to_filter_idx: np.ndarray,
    frame_to_cam_idx: np.ndarray,
    frame_to_video_ind: np.ndarray,
    video_info: dict,
    data: SessionData,
    pos_filter_pixels: np.ndarray,
    heading_filter: np.ndarray,
    led1_pixels: np.ndarray,
    led2_pixels: np.ndarray,
    vel_mag: np.ndarray,
    pos_std_cm: np.ndarray,
    vel_sigma: np.ndarray,
    heading_sigma: np.ndarray,
    config: RenderConfig,
) -> None:
    if max_workers is None:
        max_workers = max(2, (os.cpu_count() or 4) // 2)
    print(f"\n[parallel] Using {max_workers} workers")

    tmpdir = tempfile.mkdtemp(prefix="overlay_frames_")
    pattern = os.path.join(tmpdir, "frame_%06d.png")
    print(f"[parallel] Writing frames to {tmpdir}")

    # Partition frames evenly
    chunk_size = math.ceil(n_frames / max_workers)
    chunks = [
        (s, min(n_frames, s + chunk_size)) for s in range(0, n_frames, chunk_size)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                _render_chunk,
                video_path,
                video_info,
                s,
                e,
                fps,
                dpi,
                pattern,
                frame_times[s:e],
                frame_to_filter_idx[s:e],
                frame_to_cam_idx[s:e],
                frame_to_video_ind[s:e],
                data.t_imu,
                data.U_imu,
                data.t_cam,
                pos_filter_pixels,
                heading_filter,
                led1_pixels,
                led2_pixels,
                vel_mag,
                pos_std_cm,
                vel_sigma,
                heading_sigma,
                config,
            )
            for (s, e) in chunks
        ]
        for fut in futs:
            fut.result()

    _ffmpeg_stitch(
        pattern, output_path, fps, bitrate=config.bitrate_kbps, threads=max_workers
    )
    shutil.rmtree(tmpdir, ignore_errors=True)


def _render_chunk(
    video_path,
    video_info,
    start_f,
    end_f,
    fps,
    dpi,
    pattern,
    frame_times_chunk,
    fidx_chunk,
    cidx_chunk,
    vidx_chunk,
    t_imu,
    U_imu,
    t_cam,
    pos_filter_px,
    heading_rad,
    led1_px,
    led2_px,
    vel_mag_cms,
    pos_std_cm_series,
    vel_sigma_cms,
    heading_sigma_deg,
    config: RenderConfig,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = setup_figure(config.style)
    video_frame = axes["video"].imshow(
        np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8)
    )
    (trajectory_line,) = axes["video"].plot(
        [], [], "c-", linewidth=1.5, alpha=0.6, label="Filter trajectory"
    )
    led1_circle = Circle(
        (0, 0),
        config.style.led_marker_size,
        color="red",
        alpha=0.5,
        label="LED1 (back)",
    )
    led2_circle = Circle(
        (0, 0),
        config.style.led_marker_size,
        color="yellow",
        alpha=0.5,
        label="LED2 (front)",
    )
    axes["video"].add_patch(led1_circle)
    axes["video"].add_patch(led2_circle)
    filter_circle = Circle(
        (0, 0),
        config.style.led_marker_size * 1.2,
        color="cyan",
        alpha=0.8,
        linewidth=2,
        fill=False,
        label="Filter estimate",
    )
    axes["video"].add_patch(filter_circle)
    (heading_line_video,) = axes["video"].plot(
        [], [], color="cyan", linewidth=2, alpha=0.7, label="Heading"
    )
    axes["video"].legend(loc="upper right", fontsize=9, framealpha=0.9)

    (gyro_x_line,) = axes["gyro"].plot([], [], "-", linewidth=1.2, label="X", alpha=0.8)
    (gyro_y_line,) = axes["gyro"].plot([], [], "-", linewidth=1.2, label="Y", alpha=0.8)
    (gyro_z_line,) = axes["gyro"].plot([], [], "-", linewidth=1.2, label="Z", alpha=0.8)
    gyro_marker = axes["gyro"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    (accel_x_line,) = axes["accel"].plot(
        [], [], "-", linewidth=1.2, label="X", alpha=0.8
    )
    (accel_y_line,) = axes["accel"].plot(
        [], [], "-", linewidth=1.2, label="Y", alpha=0.8
    )
    (accel_z_line,) = axes["accel"].plot(
        [], [], "-", linewidth=1.2, label="Z", alpha=0.8
    )
    accel_marker = axes["accel"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    (vel_line,) = axes["pos_vel"].plot([], [], "-", linewidth=1.5, label="Speed")
    vel_marker = axes["pos_vel"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )
    vel_fill = axes["pos_vel"].fill_between([], [], [], alpha=0.2, label="±1σ")

    (heading_line_ax,) = axes["heading"].plot(
        [], [], "-", linewidth=1.5, label="Heading"
    )
    heading_marker = axes["heading"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )
    heading_fill = axes["heading"].fill_between([], [], [], alpha=0.2, label="±1σ")

    (unc_line,) = axes["uncertainty"].plot(
        [], [], "-", linewidth=1.5, label="Position σ"
    )
    unc_marker = axes["uncertainty"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    time_text = axes["video"].text(
        0.02,
        0.98,
        "",
        transform=axes["video"].transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    gyro = np.degrees(U_imu[:, :3])
    accel = U_imu[:, 3:6]

    for i in range(start_f, end_f):
        j = i - start_f
        t = frame_times_chunk[j]
        fi = fidx_chunk[j]
        ci = cidx_chunk[j]
        vi = int(vidx_chunk[j])

        frame = load_video_frame(video_path, vi)
        if frame is not None:
            video_frame.set_data(frame)

        px, py = pos_filter_px[fi]
        t0 = max(0, fi - config.style.trajectory_length)
        traj = pos_filter_px[t0 : fi + 1]
        trajectory_line.set_data(traj[:, 0], traj[:, 1])
        filter_circle.center = (px, py)

        led1_circle.center = tuple(led1_px[ci])
        led2_circle.center = tuple(led2_px[ci])

        angle = heading_rad[fi] + (math.pi if config.style.heading_flip else 0.0)
        dx = config.style.heading_arrow_px * math.cos(angle)
        dy = config.style.heading_arrow_px * math.sin(angle)
        heading_line_video.set_data([px, px + dx], [py, py + dy])

        # IMU windows
        half = config.imu_window_s / 2.0
        m = (t_imu >= t - half) & (t_imu <= t + half)
        tx = t_imu[m]
        gx, gy, gz = gyro[m].T if m.any() else (np.array([]),) * 3
        ax, ay, az = accel[m].T if m.any() else (np.array([]),) * 3

        gyro_x_line.set_data(tx, gx)
        gyro_y_line.set_data(tx, gy)
        gyro_z_line.set_data(tx, gz)
        axes["gyro"].set_xlim(t - half, t + half)
        axes["gyro"].set_ylim(*config.limits.gyro_ylim)
        gyro_marker.set_xdata([t, t])

        accel_x_line.set_data(tx, ax)
        accel_y_line.set_data(tx, ay)
        accel_z_line.set_data(tx, az)
        axes["accel"].set_xlim(t - half, t + half)
        axes["accel"].set_ylim(*config.limits.accel_ylim)
        accel_marker.set_xdata([t, t])

        # State windows (aligned to t_cam / t_filter)
        half_s = config.state_window_s / 2.0
        ms = (t_cam >= t - half_s) & (t_cam <= t + half_s)
        ts = t_cam[ms]
        vel_w = vel_mag_cms[ms]
        hdg_w = np.degrees(np.arctan2(np.sin(heading_rad[ms]), np.cos(heading_rad[ms])))

        unc_w = pos_std_cm_series[ms]

        k_sigma = 2.0  # for ±2σ fill
        vel_sig_w = k_sigma * vel_sigma_cms[ms]
        hdg_sig_w = k_sigma * heading_sigma_deg[ms]

        if len(ts):
            v_lower = vel_w - vel_sig_w
            v_upper = vel_w + vel_sig_w
            v_verts = np.concatenate(
                [
                    np.column_stack([ts, v_lower]),
                    np.column_stack([ts[::-1], v_upper[::-1]]),
                ]
            )
            vel_fill.set_verts([v_verts])

            h_lower = ((hdg_w - hdg_sig_w + 180.0) % 360.0) - 180.0
            h_upper = ((hdg_w + hdg_sig_w + 180.0) % 360.0) - 180.0
            h_verts = np.concatenate(
                [
                    np.column_stack([ts, h_lower]),
                    np.column_stack([ts[::-1], h_upper[::-1]]),
                ]
            )
            heading_fill.set_verts([h_verts])
        else:
            vel_fill.set_verts([])
            heading_fill.set_verts([])

        vel_line.set_data(ts, vel_w)
        axes["pos_vel"].set_xlim(t - half_s, t + half_s)
        vmax = (
            float(np.max(vel_w) * 1.2) if len(vel_w) else config.limits.vel_max_default
        )
        axes["pos_vel"].set_ylim(0.0, max(config.limits.vel_max_default, vmax))
        vel_marker.set_xdata([t, t])

        heading_line_ax.set_data(ts, hdg_w)
        axes["heading"].set_xlim(t - half_s, t + half_s)
        axes["heading"].set_ylim(*config.limits.heading_ylim)
        heading_marker.set_xdata([t, t])

        unc_line.set_data(ts, unc_w)
        axes["uncertainty"].set_xlim(t - half_s, t + half_s)
        umax = float(np.max(unc_w) * 1.2) if len(unc_w) else config.limits.pos_unc_min
        axes["uncertainty"].set_ylim(0.0, max(config.limits.pos_unc_min, umax))
        unc_marker.set_xdata([t, t])

        time_text.set_text(f"t = {t:.2f} s")

        out_path = pattern % (i + 1)  # 1-based for ffmpeg %06d
        fig.savefig(out_path, dpi=max(72, dpi))

    plt.close(fig)


def _ffmpeg_stitch(
    pattern: str, output_path: str, fps: float, bitrate: int = 4000, threads: int = 4
):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(int(fps)),
        "-b:v",
        str(bitrate * 1000),
        "-threads",
        str(max(1, threads)),
        output_path,
    ]
    print("[ffmpeg]", " ".join(cmd))
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------------------


def main(*, smoother: bool = False, render_mode: str = "single_process") -> int:
    """Example: create filter overlay video for the bundled Arthur session."""
    import sys

    sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
    from trodestrack.runtime.offline import rts_smoother

    script_dir = Path(__file__).parent

    print("Loading session data…")
    meters_per_pixel = 0.0022  # camera calibration
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220324_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220324_imu_info.parquet"),
        imu_mode="3d",
        meters_per_pixel=meters_per_pixel,
        verbose=False,
    )
    print(f"✓ Loaded {len(data.t_cam):,} frames and {len(data.t_imu):,} IMU samples\n")

    print("Running Extended Kalman Filter with 3D IMU…")
    t0 = time.time()
    ekf_config = EKFConfig(
        # State: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
        state_mode="2d_cam_3d_imu",
        # --- Process model (how much you allow the state to wander per second) ---
        process_noise_pos=0.05,  # m RMS per √s  (random walk on position; small but nonzero)
        process_noise_vel=0.10,  # m/s RMS per √s (lets velocity adapt to turns/sprints)
        process_noise_gyro_bias=2e-6,  # (rad/s) per √s
        process_noise_accel_bias=2e-4,  # (m/s^2) per √s
        # --- Camera measurement model ---
        # Set this from pixel noise: measurement_noise_pos = (sigma_px * meters_per_pixel)**2
        # If sigma_px ≈ 1.5 px and mpp ≈ 0.0022 m/px -> (1.5*0.0022)^2 ≈ 1.1e-5 m^2
        measurement_noise_pos=(2.0 * meters_per_pixel) ** 2,  # m^2
        # Heading from LED geometry (in radians): 3–8° std is common -> ~0.05–0.14 rad
        # If your code auto-computes this or uses adaptive noise, keep this enabled:
        use_heading_measurement=True,
        adaptive_heading_noise=True,
        # Robustness to LED glitches / occlusions
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.99,  # gate obvious outliers
        led_distance_tolerance=0.10,  # meters; ~10 cm allows perspective/slight tracking error
        # Adaptive Q during camera dropouts (lean on IMU when vision is missing or untrusted)
        adaptive_q_during_dropout=True,
        dropout_q_pos_multiplier=4.0,
        dropout_q_vel_multiplier=4.0,
        # Optionally temper IMU when vision is gone, to avoid drift blow-ups
        reduce_imu_noise_during_blackout=True,
        blackout_imu_noise_scale=0.5,
        # Mild physical damping to suppress tiny oscillations
        damping_coeff=0.10,
        # Iterated EKF helps linearization on quick turns
        num_iter=2,
        # Zero-velocity updates (very helpful during brief pauses)
        enable_zupt=True,
        zupt_velocity_threshold=0.03,  # m/s  (≈ 5 cm/s)
        zupt_measurement_noise=0.006**2,  # (m/s)^2
    )

    # want to be able to compare IMU prediction, camera prediction (no kalman), camera prediction (with kalman)

    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=data.t_imu,
        U_imu=data.U_imu,
        t_cam=data.t_cam,
        Z_cam_led1=data.Z_cam_led1,
        Z_cam_led2=data.Z_cam_led2,
        mask_cam=data.mask_cam,
    )
    print(
        f"✓ Filter complete in {time.time() - t0:.1f}s: {len(result.filtered_means):,} timesteps\n"
    )

    if smoother:
        print("Running RTS smoother…")
        t1 = time.time()
        result = rts_smoother(
            result, ekf_config, t_imu=data.t_imu, U_imu=data.U_imu, t_cam=data.t_cam
        )
        print(
            f"✓ Smoothing complete in {time.time() - t1:.1f}s: {len(result.smoothed_means):,} timesteps\n"
        )

    video_path = script_dir / "20220324_arthur_02_r1.mp4"
    output_path = script_dir / "arthur_filter_overlay.mp4"

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    position_df = pd.read_parquet(script_dir / "arthur20220324_position_info.parquet")

    style = OverlayStyle(
        led_marker_size=10.0,
        trajectory_length=90,
        heading_arrow_px=40.0,
        heading_flip=True,  # camera convention
        show_uncertainty_circle=False,
        invert_yaxis=False,
    )

    config = RenderConfig(
        start_time=120.0,
        duration=60.0,  # 60-second clip
        fps=30.0,
        imu_window_s=2.0,
        state_window_s=5.0,
        dpi=100,
        render_mode=render_mode,
        ffmpeg_threads=4,
        style=style,
    )

    create_filter_overlay_video(
        video_path=str(video_path),
        data=data,
        filter_result=result,
        t_filter=data.t_cam,  # align with camera timestamps
        position_df=position_df,
        output_path=str(output_path),
        config=config,
    )

    print(f"\n✓ Complete! View output: {output_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize EKF results with uncertainty overlays."
    )
    p.add_argument(
        "--smooth",
        action="store_true",
        help="Run RTS smoother (if available) before visualization.",
    )
    p.add_argument(
        "--render_mode",
        choices=["single_process", "parallel_png"],
        default="single_process",
        help="Choose the rendering mode for the visualization.",
    )
    return p.parse_args()


if __name__ == "__main__":
    import sys

    args = _parse_args()
    sys.exit(main(smoother=args.smooth, render_mode=args.render_mode))

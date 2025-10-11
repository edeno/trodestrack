"""Basic Rat IMU + Vision Simulation Demo.

This script demonstrates the rat IMU simulator with various configuration options:
1. Basic simulation (single LED, no confidence)
2. Two-LED simulation for heading measurements
3. Confidence-enabled simulation with dropout modeling
4. Measurement noise validation
5. Vision robustness (LED swaps + long occlusions) **NEW**

Generates publication-quality plots following Tufte/Gelman principles:
- Minimal chartjunk, maximum data-ink ratio
- Direct labeling over legends where possible
- Muted colors, emphasis on data
- Clear, informative axes

Usage:
    uv run python examples/02_rat_imu_simulation.py

Output:
    Saves 5 PNG files to examples/ directory with comprehensive visualizations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

# =============================================================================
# Matplotlib Configuration (Tufte/Gelman best practices)
# =============================================================================

plt.rcParams.update(
    {
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        # Axes styling (minimal spines, thin lines)
        "axes.linewidth": 0.5,
        "axes.spines.top": False,  # Remove top spine (Tufte principle)
        "axes.spines.right": False,  # Remove right spine
        # Grid (subtle, unobtrusive)
        "axes.grid": True,
        "grid.alpha": 0.12,  # Very transparent
        "grid.linewidth": 0.4,  # Thin lines
        # Ticks
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Legend (frameless, small)
        "legend.frameon": False,
        "legend.fontsize": 8,
        # Figure quality
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# ColorBrewer-inspired palette (color-blind safe, print-friendly)
# https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=6
COLORS = {
    "blue": "#2166AC",
    "red": "#B2182B",
    "gray": "#666666",
    "light_gray": "#CCCCCC",
    "orange": "#D6604D",
    "green": "#1B7837",
    "purple": "#762A83",
}


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_simulation_overview(
    sim_data: dict[str, np.ndarray], title: str = "Simulation Overview"
) -> plt.Figure:
    """Create comprehensive visualization of simulation data.

    Displays 4 rows × 3 columns of plots showing:
    - Row 1: Trajectory, speed
    - Row 2: Position time series
    - Row 3: IMU measurements (gyro, accel X, accel Y)
    - Row 4: Biases and camera status

    Args:
        sim_data: Dictionary returned by simulate_rat_imu()
        title: Figure title

    Returns:
        Matplotlib figure object

    Note:
        Follows Tufte/Gelman principles: high data-ink ratio, minimal
        chartjunk, direct labeling, muted colors.
    """
    # Create figure with 4×3 grid
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Extract data from simulation results
    t_imu = sim_data["t_imu"]  # IMU timestamps (high rate, e.g., 200 Hz)
    t_cam = sim_data["t_cam_exp"]  # Camera exposure times (lower rate, e.g., 30 Hz)
    X = sim_data["X_truth"]  # Ground truth: [x, y, vx, vy, θ]
    U = sim_data["U_imu"]  # IMU measurements: [ω_z, a_x, a_y]
    led1 = sim_data["Z_cam_led1"]  # LED1 camera observations
    led2 = sim_data["Z_cam_led2"]  # LED2 camera observations (if enabled)
    mask = sim_data["mask_cam"]  # Boolean mask: True = valid detection
    conf1 = sim_data["confidence_led1"]  # Confidence scores (if enabled)
    config: RatIMUSimConfig = sim_data["config"]  # type: ignore[assignment]

    # -------------------------------------------------------------------------
    # Row 1, Col 1-2: 2D Trajectory in Arena
    # -------------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[0, :2])

    # Ground truth path (subtle gray line)
    ax_traj.plot(
        X[:, 0],
        X[:, 1],
        "-",
        linewidth=1,
        color=COLORS["gray"],
        alpha=0.3,
        zorder=1,
    )

    # Camera observations (small colored points, color = time)
    sc = ax_traj.scatter(
        led1[mask, 0],
        led1[mask, 1],
        c=t_cam[mask],
        s=6,
        cmap="YlOrBr",  # Yellow-Orange-Brown colormap
        alpha=0.7,
        edgecolors="none",
        zorder=2,
    )

    # Second LED (if enabled)
    if config.use_second_led:
        ax_traj.scatter(
            led2[mask, 0],
            led2[mask, 1],
            c=t_cam[mask],
            s=6,
            cmap="YlGnBu",  # Yellow-Green-Blue colormap
            alpha=0.7,
            marker="^",  # Triangle marker for distinction
            edgecolors="none",
            zorder=2,
        )

    # Arena boundaries
    ax_traj.set_xlim(-0.05, config.arena_w + 0.05)
    ax_traj.set_ylim(-0.05, config.arena_h + 0.05)
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Trajectory", fontweight="normal", loc="left", fontsize=10)
    ax_traj.set_aspect("equal")  # Equal aspect ratio for spatial accuracy

    # Minimal colorbar (time indicator)
    cbar = plt.colorbar(sc, ax=ax_traj, pad=0.02, aspect=25, shrink=0.85)
    cbar.set_label("time (s)", fontsize=8)
    cbar.outline.set_linewidth(0.5)  # type: ignore[operator]

    # -------------------------------------------------------------------------
    # Row 1, Col 3: Speed Over Time
    # -------------------------------------------------------------------------
    ax_speed = fig.add_subplot(gs[0, 2])

    # Compute speed magnitude from velocity components
    speed = np.hypot(X[:, 2], X[:, 3])  # sqrt(vx² + vy²)

    ax_speed.plot(t_imu, speed, "-", linewidth=0.9, color=COLORS["blue"])

    # Reference line for speed clip limit
    ax_speed.axhline(
        config.speed_clip, color=COLORS["red"], linestyle="--", linewidth=0.7, alpha=0.5
    )

    # Direct label (Tufte principle: avoid legends when possible)
    ax_speed.text(
        t_imu[-1] * 0.02,
        config.speed_clip * 1.08,
        "clip limit",
        fontsize=7,
        color=COLORS["red"],
        va="bottom",
    )

    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.set_title("Speed", fontweight="normal", loc="left", fontsize=10)
    ax_speed.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Row 2: Position Time Series (x and y components)
    # -------------------------------------------------------------------------
    ax_pos = fig.add_subplot(gs[1, :])

    # Ground truth position (smooth lines, muted colors)
    ax_pos.plot(t_imu, X[:, 0], "-", linewidth=0.9, color=COLORS["red"], alpha=0.6, label="x")
    ax_pos.plot(t_imu, X[:, 1], "-", linewidth=0.9, color=COLORS["blue"], alpha=0.6, label="y")

    # Camera observations (small semi-transparent markers)
    ax_pos.scatter(
        t_cam[mask],
        led1[mask, 0],
        s=3,
        c=COLORS["red"],
        alpha=0.3,
        edgecolors="none",
    )
    ax_pos.scatter(
        t_cam[mask],
        led1[mask, 1],
        s=3,
        c=COLORS["blue"],
        alpha=0.3,
        edgecolors="none",
    )

    ax_pos.set_xlabel("time (s)")
    ax_pos.set_ylabel("position (m)")
    ax_pos.set_title(
        "Position (truth + observations)", fontweight="normal", loc="left", fontsize=10
    )
    ax_pos.legend(loc="upper right", ncol=2)

    # -------------------------------------------------------------------------
    # Row 3, Col 1: Gyroscope (Yaw Rate)
    # -------------------------------------------------------------------------
    ax_gyro = fig.add_subplot(gs[2, 0])

    # Ground truth (gray, subtle)
    ax_gyro.plot(
        t_imu,
        np.rad2deg(sim_data["yaw_rate_truth"]),
        "-",
        linewidth=0.7,
        color=COLORS["gray"],
        alpha=0.4,
        label="truth",
    )

    # Measurement (colored, more prominent)
    ax_gyro.plot(
        t_imu,
        np.rad2deg(U[:, 0]),
        "-",
        linewidth=0.5,
        color=COLORS["purple"],
        alpha=0.7,
        label="measured",
    )

    ax_gyro.set_xlabel("time (s)")
    ax_gyro.set_ylabel("yaw rate (°/s)")
    ax_gyro.set_title("Gyroscope", fontweight="normal", loc="left", fontsize=10)
    ax_gyro.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # Row 3, Col 2: Accelerometer X (Body Frame)
    # -------------------------------------------------------------------------
    ax_accel_x = fig.add_subplot(gs[2, 1])

    ax_accel_x.plot(
        t_imu,
        sim_data["accel_body_truth"][:, 0],
        "-",
        linewidth=0.7,
        color=COLORS["gray"],
        alpha=0.4,
    )
    ax_accel_x.plot(t_imu, U[:, 1], "-", linewidth=0.5, color=COLORS["blue"], alpha=0.7)

    ax_accel_x.set_xlabel("time (s)")
    ax_accel_x.set_ylabel("accel (m/s²)")
    ax_accel_x.set_title("Accel X (body)", fontweight="normal", loc="left", fontsize=10)

    # -------------------------------------------------------------------------
    # Row 3, Col 3: Accelerometer Y (Body Frame)
    # -------------------------------------------------------------------------
    ax_accel_y = fig.add_subplot(gs[2, 2])

    ax_accel_y.plot(
        t_imu,
        sim_data["accel_body_truth"][:, 1],
        "-",
        linewidth=0.7,
        color=COLORS["gray"],
        alpha=0.4,
    )
    ax_accel_y.plot(t_imu, U[:, 2], "-", linewidth=0.5, color=COLORS["red"], alpha=0.7)

    ax_accel_y.set_xlabel("time (s)")
    ax_accel_y.set_ylabel("accel (m/s²)")
    ax_accel_y.set_title("Accel Y (body)", fontweight="normal", loc="left", fontsize=10)

    # -------------------------------------------------------------------------
    # Row 4, Col 1: Gyroscope Bias (Random Walk)
    # -------------------------------------------------------------------------
    ax_bias_gyro = fig.add_subplot(gs[3, 0])

    ax_bias_gyro.plot(
        t_imu,
        np.rad2deg(sim_data["bias_gyro"]),
        "-",
        linewidth=0.8,
        color=COLORS["purple"],
    )

    # Zero reference line
    ax_bias_gyro.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.2)

    ax_bias_gyro.set_xlabel("time (s)")
    ax_bias_gyro.set_ylabel("bias (°/s)")
    ax_bias_gyro.set_title("Gyro Bias (random walk)", fontweight="normal", loc="left", fontsize=10)

    # -------------------------------------------------------------------------
    # Row 4, Col 2: Accelerometer Biases
    # -------------------------------------------------------------------------
    ax_bias_accel = fig.add_subplot(gs[3, 1])

    ax_bias_accel.plot(
        t_imu,
        sim_data["bias_accel_x"],
        "-",
        linewidth=0.8,
        color=COLORS["blue"],
        alpha=0.7,
        label="x",
    )
    ax_bias_accel.plot(
        t_imu,
        sim_data["bias_accel_y"],
        "-",
        linewidth=0.8,
        color=COLORS["red"],
        alpha=0.7,
        label="y",
    )

    ax_bias_accel.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.2)

    ax_bias_accel.set_xlabel("time (s)")
    ax_bias_accel.set_ylabel("bias (m/s²)")
    ax_bias_accel.set_title("Accel Biases", fontweight="normal", loc="left", fontsize=10)
    ax_bias_accel.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # Row 4, Col 3: Camera Confidence or Dropout Pattern
    # -------------------------------------------------------------------------
    ax_conf = fig.add_subplot(gs[3, 2])

    if config.use_confidence:
        # Confidence score over time
        ax_conf.plot(t_cam, conf1, "-", linewidth=0.9, color=COLORS["green"], alpha=0.7)

        # Base confidence reference
        ax_conf.axhline(
            config.confidence_base,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=0.6,
            alpha=0.4,
        )

        ax_conf.set_ylabel("confidence")
        ax_conf.set_ylim(0, 1)
    else:
        # Show dropout pattern as filled regions (step function)
        ax_conf.fill_between(
            t_cam,
            0,
            1,
            where=mask,  # type: ignore[arg-type]
            alpha=0.4,
            color=COLORS["green"],
            step="mid",
            label="valid",
        )
        ax_conf.fill_between(
            t_cam,
            0,
            1,
            where=~mask,  # type: ignore[arg-type]
            alpha=0.4,
            color=COLORS["red"],
            step="mid",
            label="dropout",
        )

        ax_conf.set_ylabel("detection")
        ax_conf.set_yticks([0, 1])
        ax_conf.set_yticklabels(["dropout", "valid"])
        ax_conf.legend(loc="upper right")

    ax_conf.set_xlabel("time (s)")
    ax_conf.set_title("Camera Status", fontweight="normal", loc="left", fontsize=10)

    # Overall title
    fig.suptitle(title, fontsize=11, fontweight="normal", y=0.995)

    return fig


def plot_measurement_errors(sim_data: dict[str, np.ndarray]) -> plt.Figure:
    """Plot measurement errors for noise validation.

    Creates 2×3 grid showing:
    - Row 1: Gyro error (time series, histogram, accel error)
    - Row 2: Camera error (time series, histogram, 2D scatter)

    Args:
        sim_data: Dictionary returned by simulate_rat_imu()

    Returns:
        Matplotlib figure object

    Note:
        All plots include reference lines (zero, 1σ circle) for context.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    # Extract data
    t_imu = sim_data["t_imu"]
    t_cam = sim_data["t_cam_exp"]
    X = sim_data["X_truth"]
    U = sim_data["U_imu"]
    led1 = sim_data["Z_cam_led1"]
    mask = sim_data["mask_cam"]
    config: RatIMUSimConfig = sim_data["config"]  # type: ignore[assignment]

    # -------------------------------------------------------------------------
    # Compute Measurement Errors (measured - bias - truth)
    # -------------------------------------------------------------------------

    # Gyro error (zero-mean white noise)
    gyro_err = U[:, 0] - sim_data["bias_gyro"] - sim_data["yaw_rate_truth"]

    # Accelerometer errors
    accel_x_err = U[:, 1] - sim_data["bias_accel_x"] - sim_data["accel_body_truth"][:, 0]

    # Camera position error (interpolate truth to camera timestamps)
    px_interp = np.interp(t_cam, t_imu, X[:, 0])
    py_interp = np.interp(t_cam, t_imu, X[:, 1])
    cam_err_x = led1[:, 0] - px_interp
    cam_err_y = led1[:, 1] - py_interp

    # -------------------------------------------------------------------------
    # Row 1, Col 1: Gyro Error Time Series
    # -------------------------------------------------------------------------
    axes[0, 0].plot(
        t_imu, np.rad2deg(gyro_err), "-", linewidth=0.4, color=COLORS["purple"], alpha=0.6
    )
    axes[0, 0].axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("error (°/s)")
    axes[0, 0].set_title("Gyro Measurement Error", fontweight="normal", loc="left", fontsize=10)

    # -------------------------------------------------------------------------
    # Row 1, Col 2: Gyro Error Distribution
    # -------------------------------------------------------------------------
    axes[0, 1].hist(
        np.rad2deg(gyro_err),
        bins=40,
        alpha=0.6,
        color=COLORS["purple"],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0, 1].axvline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    axes[0, 1].set_xlabel("error (°/s)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title(
        f"Gyro Error (σ={np.std(gyro_err):.5f} rad/s)",
        fontweight="normal",
        loc="left",
        fontsize=10,
    )

    # -------------------------------------------------------------------------
    # Row 1, Col 3: Accel X Error Time Series
    # -------------------------------------------------------------------------
    axes[0, 2].plot(t_imu, accel_x_err, "-", linewidth=0.4, color=COLORS["blue"], alpha=0.6)
    axes[0, 2].axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    axes[0, 2].set_xlabel("time (s)")
    axes[0, 2].set_ylabel("error (m/s²)")
    axes[0, 2].set_title("Accel X Error", fontweight="normal", loc="left", fontsize=10)

    # -------------------------------------------------------------------------
    # Row 2, Col 1: Camera Position Error Time Series
    # -------------------------------------------------------------------------

    # Convert to mm for readability
    valid_err_x = cam_err_x[mask] * 1000
    valid_err_y = cam_err_y[mask] * 1000

    axes[1, 0].scatter(
        t_cam[mask],
        valid_err_x,
        s=2,
        alpha=0.5,
        c=COLORS["red"],
        edgecolors="none",
        label="x",
    )
    axes[1, 0].scatter(
        t_cam[mask],
        valid_err_y,
        s=2,
        alpha=0.5,
        c=COLORS["blue"],
        edgecolors="none",
        label="y",
    )
    axes[1, 0].axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("error (mm)")
    axes[1, 0].set_title("Camera Position Error", fontweight="normal", loc="left", fontsize=10)
    axes[1, 0].legend(loc="upper right", markerscale=2)

    # -------------------------------------------------------------------------
    # Row 2, Col 2: Camera Error Distribution
    # -------------------------------------------------------------------------
    axes[1, 1].hist(
        valid_err_x,
        bins=30,
        alpha=0.5,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.5,
        label="x",
    )
    axes[1, 1].hist(
        valid_err_y,
        bins=30,
        alpha=0.5,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.5,
        label="y",
    )
    axes[1, 1].axvline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    axes[1, 1].set_xlabel("error (mm)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title(
        f"Camera Error (σ_x={np.std(valid_err_x):.2f} mm)",
        fontweight="normal",
        loc="left",
        fontsize=10,
    )
    axes[1, 1].legend(loc="upper right")

    # -------------------------------------------------------------------------
    # Row 2, Col 3: 2D Camera Error Pattern (with 1σ circle)
    # -------------------------------------------------------------------------
    axes[1, 2].scatter(
        valid_err_x,
        valid_err_y,
        s=3,
        alpha=0.3,
        c=t_cam[mask],
        cmap="YlOrBr",
        edgecolors="none",
    )

    # Zero reference lines
    axes[1, 2].axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.2)
    axes[1, 2].axvline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.2)

    # 1σ reference circle (expected error magnitude)
    circle = plt.Circle(
        (0, 0),
        config.cam_sigma_m * 1000,  # Convert to mm
        fill=False,
        edgecolor=COLORS["gray"],
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )
    axes[1, 2].add_patch(circle)

    # Direct label for circle
    axes[1, 2].text(
        0,
        -config.cam_sigma_m * 1000 * 1.15,
        f"1σ = {config.cam_sigma_m * 1000:.1f} mm",
        ha="center",
        fontsize=7,
        color=COLORS["gray"],
    )

    axes[1, 2].set_xlabel("x error (mm)")
    axes[1, 2].set_ylabel("y error (mm)")
    axes[1, 2].set_title("2D Error Pattern", fontweight="normal", loc="left", fontsize=10)
    axes[1, 2].set_aspect("equal")

    fig.suptitle("Measurement Noise Validation", fontsize=11, fontweight="normal", y=0.995)

    return fig


def plot_vision_robustness(sim_data: dict[str, np.ndarray]) -> plt.Figure:
    """Plot vision robustness features: LED swaps, long occlusions, confidence.

    Creates 4×2 grid showing:
    - Row 1: Trajectory with occlusion periods, LED spacing over time
    - Row 2: LED1 vs LED2 dropouts (showing independence), confidence time series
    - Row 3: Occlusion duration histogram, confidence distribution
    - Row 4: LED position residuals (detect swaps), camera status timeline

    Args:
        sim_data: Dictionary returned by simulate_rat_imu()

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Extract data
    t_imu = sim_data["t_imu"]
    t_cam = sim_data["t_cam_exp"]
    X = sim_data["X_truth"]
    led1 = sim_data["Z_cam_led1"]
    led2 = sim_data["Z_cam_led2"]
    mask_cam = sim_data["mask_cam"]
    mask_led1 = sim_data["mask_led1"]
    mask_led2 = sim_data["mask_led2"]
    conf_led1 = sim_data["confidence_led1"]
    conf_led2 = sim_data["confidence_led2"]
    config: RatIMUSimConfig = sim_data["config"]  # type: ignore[assignment]

    # -------------------------------------------------------------------------
    # Row 1, Col 1: Trajectory with Occlusion Shading
    # -------------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[0, 0])

    # Interpolate position to camera times
    px_cam = np.interp(t_cam, t_imu, X[:, 0])
    py_cam = np.interp(t_cam, t_imu, X[:, 1])

    # Plot trajectory colored by occlusion status
    visible_idx = np.where(mask_cam)[0]
    dropout_idx = np.where(~mask_cam)[0]

    ax_traj.plot(
        px_cam[visible_idx],
        py_cam[visible_idx],
        "o",
        markersize=2,
        color=COLORS["blue"],
        alpha=0.3,
        label="visible",
    )
    ax_traj.plot(
        px_cam[dropout_idx],
        py_cam[dropout_idx],
        "x",
        markersize=3,
        color=COLORS["red"],
        alpha=0.6,
        label="dropout",
    )

    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Trajectory (colored by visibility)", fontweight="normal", loc="left")
    ax_traj.legend(loc="best")
    ax_traj.set_aspect("equal")

    # -------------------------------------------------------------------------
    # Row 1, Col 2: LED Spacing Over Time (Detect Swaps)
    # -------------------------------------------------------------------------
    ax_spacing = fig.add_subplot(gs[0, 1])

    # Compute LED spacing when both visible
    both_visible = mask_led1 & mask_led2
    if np.any(both_visible):
        led_spacing = np.linalg.norm(led1 - led2, axis=1)
        expected_spacing = np.linalg.norm(config.led1_offset_body - config.led2_offset_body)

        ax_spacing.plot(
            t_cam[both_visible],
            led_spacing[both_visible] * 100,
            "o",
            markersize=2,
            color=COLORS["blue"],
            alpha=0.5,
        )
        ax_spacing.axhline(
            expected_spacing * 100,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1,
            label=f"expected: {expected_spacing*100:.1f} cm",
        )

        ax_spacing.set_xlabel("time (s)")
        ax_spacing.set_ylabel("LED spacing (cm)")
        ax_spacing.set_title("LED Spacing (both visible)", fontweight="normal", loc="left")
        ax_spacing.legend(loc="best")
    else:
        ax_spacing.text(0.5, 0.5, "No frames with both LEDs", ha="center", va="center")
        ax_spacing.set_xlabel("time (s)")

    # -------------------------------------------------------------------------
    # Row 2, Col 1: LED1 vs LED2 Dropout Status (Show Independence)
    # -------------------------------------------------------------------------
    ax_dropout = fig.add_subplot(gs[1, 0])

    # Create dropout status time series (offset for visibility)
    led1_status = mask_led1.astype(float)
    led2_status = mask_led2.astype(float) + 2  # Offset by 2 for visual separation

    ax_dropout.plot(
        t_cam, led1_status, linewidth=0.5, color=COLORS["blue"], label="LED1", alpha=0.7
    )
    ax_dropout.plot(
        t_cam, led2_status, linewidth=0.5, color=COLORS["orange"], label="LED2", alpha=0.7
    )

    ax_dropout.set_xlabel("time (s)")
    ax_dropout.set_ylabel("LED status (offset for clarity)")
    ax_dropout.set_yticks([0, 1, 2, 3])
    ax_dropout.set_yticklabels(["LED1\ndropout", "LED1\nvisible", "LED2\ndropout", "LED2\nvisible"])
    ax_dropout.set_title(
        "Independent LED Dropouts (correlation={:.2f})".format(config.cam_dropout_correlation),
        fontweight="normal",
        loc="left",
    )
    ax_dropout.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # Row 2, Col 2: Confidence Time Series
    # -------------------------------------------------------------------------
    ax_conf = fig.add_subplot(gs[1, 1])

    if config.use_confidence:
        ax_conf.plot(
            t_cam[mask_led1],
            conf_led1[mask_led1],
            "o",
            markersize=2,
            color=COLORS["blue"],
            alpha=0.5,
            label="LED1",
        )
        if config.use_second_led:
            ax_conf.plot(
                t_cam[mask_led2],
                conf_led2[mask_led2],
                "o",
                markersize=2,
                color=COLORS["orange"],
                alpha=0.5,
                label="LED2",
            )

        ax_conf.axhline(config.confidence_base, color=COLORS["gray"], linestyle="--", linewidth=1)
        ax_conf.set_ylim([0, 1.05])
        ax_conf.set_xlabel("time (s)")
        ax_conf.set_ylabel("confidence")
        ax_conf.set_title("Confidence Scores", fontweight="normal", loc="left")
        ax_conf.legend(loc="best")
    else:
        ax_conf.text(0.5, 0.5, "Confidence disabled", ha="center", va="center")
        ax_conf.set_xlabel("time (s)")

    # -------------------------------------------------------------------------
    # Row 3, Col 1: Occlusion Duration Histogram
    # -------------------------------------------------------------------------
    ax_occ_hist = fig.add_subplot(gs[2, 0])

    # Find occlusion runs
    dropout_runs = []
    current_run = 0
    for visible in mask_cam:
        if not visible:
            current_run += 1
        else:
            if current_run > 0:
                dropout_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        dropout_runs.append(current_run)

    if dropout_runs:
        dropout_durations = np.array(dropout_runs) / config.fs_cam
        ax_occ_hist.hist(
            dropout_durations,
            bins=20,
            color=COLORS["red"],
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        # Add reference lines for PRD targets
        ax_occ_hist.axvline(3.0, color=COLORS["orange"], linestyle="--", linewidth=1, label="3s")
        ax_occ_hist.axvline(5.0, color=COLORS["red"], linestyle="--", linewidth=1, label="5s (PRD)")

        ax_occ_hist.set_xlabel("occlusion duration (s)")
        ax_occ_hist.set_ylabel("count")
        ax_occ_hist.set_title(
            f"Occlusion Durations (max={max(dropout_durations):.2f}s)",
            fontweight="normal",
            loc="left",
        )
        ax_occ_hist.legend(loc="upper right")
    else:
        ax_occ_hist.text(0.5, 0.5, "No occlusions", ha="center", va="center")
        ax_occ_hist.set_xlabel("occlusion duration (s)")

    # -------------------------------------------------------------------------
    # Row 3, Col 2: Confidence Distribution
    # -------------------------------------------------------------------------
    ax_conf_hist = fig.add_subplot(gs[2, 1])

    if config.use_confidence and np.any(mask_led1):
        conf_visible = conf_led1[mask_led1]
        ax_conf_hist.hist(
            conf_visible,
            bins=30,
            color=COLORS["blue"],
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_conf_hist.axvline(
            conf_visible.mean(),
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1,
            label=f"mean={conf_visible.mean():.3f}",
        )

        ax_conf_hist.set_xlabel("confidence")
        ax_conf_hist.set_ylabel("count")
        ax_conf_hist.set_title("Confidence Distribution (LED1)", fontweight="normal", loc="left")
        ax_conf_hist.legend(loc="best")
    else:
        ax_conf_hist.text(0.5, 0.5, "Confidence disabled", ha="center", va="center")
        ax_conf_hist.set_xlabel("confidence")

    # -------------------------------------------------------------------------
    # Row 4, Col 1: LED Position Residuals (X & Y)
    # -------------------------------------------------------------------------
    ax_resid = fig.add_subplot(gs[3, 0])

    # Compute LED1 residuals (measurement - interpolated truth)
    px_interp = np.interp(t_cam, t_imu, X[:, 0])
    py_interp = np.interp(t_cam, t_imu, X[:, 1])

    resid_x = (led1[:, 0] - px_interp) * 1000  # mm
    resid_y = (led1[:, 1] - py_interp) * 1000  # mm

    ax_resid.plot(
        t_cam[mask_led1],
        resid_x[mask_led1],
        "o",
        markersize=2,
        color=COLORS["blue"],
        alpha=0.4,
        label="x residual",
    )
    ax_resid.plot(
        t_cam[mask_led1],
        resid_y[mask_led1],
        "o",
        markersize=2,
        color=COLORS["orange"],
        alpha=0.4,
        label="y residual",
    )
    ax_resid.axhline(0, color=COLORS["gray"], linestyle="-", linewidth=0.5)

    ax_resid.set_xlabel("time (s)")
    ax_resid.set_ylabel("position error (mm)")
    ax_resid.set_title("LED1 Position Residuals", fontweight="normal", loc="left")
    ax_resid.legend(loc="best")

    # -------------------------------------------------------------------------
    # Row 4, Col 2: Camera Status Timeline (Both LEDs)
    # -------------------------------------------------------------------------
    ax_status = fig.add_subplot(gs[3, 1])

    # Create status categories
    # 0 = both dropped, 1 = LED1 only, 2 = LED2 only, 3 = both visible
    status = np.zeros(len(mask_cam), dtype=int)
    status[mask_led1 & ~mask_led2] = 1
    status[~mask_led1 & mask_led2] = 2
    status[mask_led1 & mask_led2] = 3

    colors_status = [COLORS["red"], COLORS["blue"], COLORS["orange"], COLORS["green"]]
    labels_status = ["both dropout", "LED1 only", "LED2 only", "both visible"]

    for i in range(4):
        mask_status = status == i
        if np.any(mask_status):
            ax_status.scatter(
                t_cam[mask_status],
                np.zeros(np.sum(mask_status)) + i,
                c=colors_status[i],
                s=10,
                marker="|",
                alpha=0.7,
            )

    ax_status.set_xlabel("time (s)")
    ax_status.set_yticks([0, 1, 2, 3])
    ax_status.set_yticklabels(labels_status)
    ax_status.set_title("Camera Status Timeline", fontweight="normal", loc="left")
    ax_status.set_ylim([-0.5, 3.5])

    # Overall title
    fig.suptitle(
        "Example 5: Vision Robustness (LED Swaps + Long Occlusions)",
        fontsize=11,
        fontweight="normal",
        y=0.995,
    )

    return fig


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Run all simulation examples and generate plots."""
    print("=" * 70)
    print("Rat IMU + Vision Simulation Examples")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Example 1: Basic Simulation (Single LED, No Confidence)
    # -------------------------------------------------------------------------
    print("\n1. Basic simulation (single LED, 10s)...")

    config1 = RatIMUSimConfig(
        duration_s=10.0,
        fs_imu=200.0,  # IMU at 200 Hz
        fs_cam=30.0,  # Camera at 30 Hz
        cam_dropout_prob=0.15,  # 15% dropout rate
        use_confidence=False,  # Uniform noise (no confidence scores)
        use_second_led=False,  # Single LED only
    )

    sim1 = simulate_rat_imu(config1, seed=42)

    print(
        f"   Generated {len(sim1['t_imu'])} IMU samples, " f"{len(sim1['t_cam_exp'])} camera frames"
    )
    print(
        f"   Valid frames: {sim1['mask_cam'].sum()}/{len(sim1['mask_cam'])} "
        f"({100 * sim1['mask_cam'].mean():.1f}%)"
    )

    fig1 = plot_simulation_overview(sim1, "Example 1: Basic Simulation")
    fig1.savefig("02_basic_sim.png", dpi=150)
    plt.close(fig1)
    print("   → 02_basic_sim.png")

    # -------------------------------------------------------------------------
    # Example 2: Two-LED Simulation (For Heading Measurements)
    # -------------------------------------------------------------------------
    print("\n2. Two-LED simulation (10s)...")

    config2 = RatIMUSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        fs_cam=30.0,
        use_second_led=True,  # Enable second LED
        led1_offset_body=np.array([0.03, 0.0]),  # 3 cm forward
        led2_offset_body=np.array([-0.03, 0.0]),  # 3 cm backward
        cam_dropout_prob=0.10,
    )

    sim2 = simulate_rat_imu(config2, seed=123)

    led_sep = np.linalg.norm(config2.led1_offset_body - config2.led2_offset_body)
    print(f"   LED separation: {led_sep * 100:.1f} cm")

    fig2 = plot_simulation_overview(sim2, "Example 2: Two-LED Simulation")
    fig2.savefig("02_two_led_sim.png", dpi=150)
    plt.close(fig2)
    print("   → 02_two_led_sim.png")

    # -------------------------------------------------------------------------
    # Example 3: Confidence-Enabled Simulation (Dropout Modeling)
    # -------------------------------------------------------------------------
    print("\n3. Confidence-enabled simulation (10s)...")

    config3 = RatIMUSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        fs_cam=30.0,
        use_confidence=True,  # Enable confidence scores
        confidence_base=0.95,  # Base confidence when detection is good
        confidence_dropout_decay=0.25,  # Multiply by 0.25 near dropouts
        cam_dropout_prob=0.20,  # Higher dropout rate (20%)
        use_second_led=True,
    )

    sim3 = simulate_rat_imu(config3, seed=456)

    mean_conf = sim3["confidence_led1"][sim3["mask_cam"]].mean()
    print(f"   Mean confidence: {mean_conf:.3f}")

    fig3 = plot_simulation_overview(sim3, "Example 3: Confidence-Enabled Simulation")
    fig3.savefig("03_confidence_sim.png", dpi=150)
    plt.close(fig3)
    print("   → 03_confidence_sim.png")

    # -------------------------------------------------------------------------
    # Example 4: Noise Validation (Long Run for Statistics)
    # -------------------------------------------------------------------------
    print("\n4. Noise validation (60s run)...")

    config4 = RatIMUSimConfig(
        duration_s=60.0,  # Longer duration for statistical analysis
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=0.10,
    )

    sim4 = simulate_rat_imu(config4, seed=789)

    fig4 = plot_measurement_errors(sim4)
    fig4.savefig("04_noise_validation.png", dpi=150)
    plt.close(fig4)
    print("   → 04_noise_validation.png")

    # -------------------------------------------------------------------------
    # Example 5: Vision Robustness (LED Swaps + Long Occlusions)
    # -------------------------------------------------------------------------
    print("\n5. Vision robustness: LED swaps + long occlusions (30s)...")

    config5 = RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=200.0,
        fs_cam=30.0,
        use_second_led=True,
        led1_offset_body=np.array([0.025, 0.0]),  # 2.5 cm forward
        led2_offset_body=np.array([-0.025, 0.0]),  # 2.5 cm backward (5cm total)
        # LED swap parameters
        led_swap_prob=0.15,  # 15% of visible frames have swapped labels
        # Long occlusion parameters
        cam_dropout_prob=0.25,  # 25% dropout rate
        cam_dropout_correlation=0.85,  # High correlation → long consecutive dropouts
        # Confidence scaling
        use_confidence=True,
        confidence_base=0.90,
        confidence_dropout_decay=0.3,  # Strong decay near dropouts
    )

    sim5 = simulate_rat_imu(config5, seed=999)

    # Analyze occlusion durations
    mask_cam = sim5["mask_cam"]
    dropout_runs = []
    current_run = 0
    for visible in mask_cam:
        if not visible:
            current_run += 1
        else:
            if current_run > 0:
                dropout_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        dropout_runs.append(current_run)

    if dropout_runs:
        max_dropout_frames = max(dropout_runs)
        max_dropout_s = max_dropout_frames / config5.fs_cam
        mean_dropout_s = np.mean(dropout_runs) / config5.fs_cam
        print(f"   Occlusion stats: max={max_dropout_s:.2f}s, mean={mean_dropout_s:.2f}s")
    else:
        print("   No occlusions detected")

    # Count LED swaps (approximate by checking if both LEDs visible)
    both_visible = sim5["mask_led1"] & sim5["mask_led2"]
    n_swap_candidates = np.sum(both_visible)
    expected_swaps = int(n_swap_candidates * config5.led_swap_prob)
    print(f"   LED swap candidates: {n_swap_candidates}, expected swaps: ~{expected_swaps}")

    # Mean confidence
    conf_led1 = sim5["confidence_led1"][sim5["mask_led1"]]
    conf_led2 = sim5["confidence_led2"][sim5["mask_led2"]]
    print(f"   Mean confidence: LED1={conf_led1.mean():.3f}, LED2={conf_led2.mean():.3f}")

    fig5 = plot_vision_robustness(sim5)
    fig5.savefig("05_vision_robustness.png", dpi=150)
    plt.close(fig5)
    print("   → 05_vision_robustness.png")

    # -------------------------------------------------------------------------
    # Print Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary Statistics (60s validation run)")
    print("=" * 70)

    # Compute measurement errors (truth is known in simulation)
    gyro_err = sim4["U_imu"][:, 0] - sim4["bias_gyro"] - sim4["yaw_rate_truth"]
    accel_x_err = sim4["U_imu"][:, 1] - sim4["bias_accel_x"] - sim4["accel_body_truth"][:, 0]
    accel_y_err = sim4["U_imu"][:, 2] - sim4["bias_accel_y"] - sim4["accel_body_truth"][:, 1]

    print("IMU Measurement Noise (std):")
    print(
        f"  Gyroscope:      {np.std(gyro_err):.6f} rad/s = "
        f"{np.rad2deg(np.std(gyro_err)):.4f} °/s"
    )
    print(f"  Accelerometer X: {np.std(accel_x_err):.6f} m/s²")
    print(f"  Accelerometer Y: {np.std(accel_y_err):.6f} m/s²")

    # Camera position error
    t_cam = sim4["t_cam_exp"]
    t_imu = sim4["t_imu"]
    X = sim4["X_truth"]
    led1 = sim4["Z_cam_led1"]
    mask = sim4["mask_cam"]

    px_interp = np.interp(t_cam, t_imu, X[:, 0])
    py_interp = np.interp(t_cam, t_imu, X[:, 1])
    cam_err = np.hypot(led1[:, 0] - px_interp, led1[:, 1] - py_interp)

    print("\nCamera Measurement Noise:")
    print(f"  Position error (std): {np.std(cam_err[mask]) * 1000:.2f} mm")
    print(f"  Dropout rate: {100 * (1 - mask.mean()):.1f}%")

    # Bias drift over time
    print("\nBias Random Walks:")
    print(f"  Gyro bias range: {np.rad2deg(np.ptp(sim4['bias_gyro'])):.4f} °/s")
    print(f"  Accel X range: {np.ptp(sim4['bias_accel_x']):.6f} m/s²")
    print(f"  Accel Y range: {np.ptp(sim4['bias_accel_y']):.6f} m/s²")

    print("\n" + "=" * 70)
    print("✓ All plots saved to examples/")
    print("=" * 70)


if __name__ == "__main__":
    main()

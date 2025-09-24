"""
Quality assurance plotting functions for trajectory visualization and diagnostics.

This module provides comprehensive plotting capabilities for evaluating tracking
performance, including trajectory plots, residual analysis, bias traces, and NEES
consistency diagnostics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import jax.numpy as jnp
from typing import Optional, Dict, Tuple
from pathlib import Path

from .metrics import compute_nees, compute_position_nees


def plot_trajectory_comparison(
    estimated_states: jnp.ndarray,
    ground_truth_states: jnp.ndarray,
    timestamps: Optional[jnp.ndarray] = None,
    occlusion_mask: Optional[jnp.ndarray] = None,
    arena_bounds: Optional[Tuple[float, float, float, float]] = None,
    title: str = "Trajectory Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Plot estimated vs ground truth trajectories with occlusion periods highlighted.

    Args:
        estimated_states: Shape (N, 8) estimated states [x, y, vx, vy, theta, ...]
        ground_truth_states: Shape (N, 8) ground truth states
        timestamps: Optional timestamps for trajectory
        occlusion_mask: Optional mask where True = occluded
        arena_bounds: Optional (x_min, x_max, y_min, y_max) for arena
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, (ax_traj, ax_error) = plt.subplots(1, 2, figsize=figsize)

    # Extract positions
    est_pos = np.array(estimated_states[:, :2])
    gt_pos = np.array(ground_truth_states[:, :2])

    # Main trajectory plot
    ax_traj.plot(gt_pos[:, 0], gt_pos[:, 1], "k-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax_traj.plot(est_pos[:, 0], est_pos[:, 1], "r-", linewidth=1.5, label="Estimated", alpha=0.7)

    # Highlight occlusion periods
    if occlusion_mask is not None:
        occlusion_indices = np.where(occlusion_mask)[0]
        if len(occlusion_indices) > 0:
            ax_traj.scatter(
                est_pos[occlusion_indices, 0],
                est_pos[occlusion_indices, 1],
                c="orange",
                s=20,
                alpha=0.6,
                label="Occluded",
                zorder=5,
            )

    # Mark start and end points
    ax_traj.scatter(
        gt_pos[0, 0],
        gt_pos[0, 1],
        c="green",
        s=100,
        marker="o",
        label="Start",
        zorder=6,
        edgecolors="black",
    )
    ax_traj.scatter(
        gt_pos[-1, 0],
        gt_pos[-1, 1],
        c="red",
        s=100,
        marker="s",
        label="End",
        zorder=6,
        edgecolors="black",
    )

    # Arena bounds
    if arena_bounds is not None:
        x_min, x_max, y_min, y_max = arena_bounds
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
            linestyle="--",
            label="Arena",
        )
        ax_traj.add_patch(rect)

    ax_traj.set_xlabel("X Position (cm)")
    ax_traj.set_ylabel("Y Position (cm)")
    ax_traj.set_title("2D Trajectory")
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect("equal")

    # Position error over time
    position_error = np.linalg.norm(est_pos - gt_pos, axis=1)

    if timestamps is not None:
        time_axis = np.array(timestamps)
        ax_error.set_xlabel("Time (s)")
    else:
        time_axis = np.arange(len(position_error))
        ax_error.set_xlabel("Frame")

    ax_error.plot(time_axis, position_error, "b-", linewidth=1.5, alpha=0.8)

    # Highlight occlusion periods
    if occlusion_mask is not None:
        ax_error.fill_between(
            time_axis,
            0,
            position_error,
            where=occlusion_mask,
            alpha=0.3,
            color="orange",
            label="Occluded",
        )

    # PRD threshold line
    ax_error.axhline(y=2.0, color="red", linestyle="--", alpha=0.7, label="PRD Threshold (2 cm)")

    ax_error.set_ylabel("Position Error (cm)")
    ax_error.set_title("Position Error Over Time")
    ax_error.legend()
    ax_error.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_velocity_and_heading(
    estimated_states: jnp.ndarray,
    ground_truth_states: jnp.ndarray,
    timestamps: Optional[jnp.ndarray] = None,
    occlusion_mask: Optional[jnp.ndarray] = None,
    title: str = "Velocity and Heading Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot velocity and heading comparison over time.

    Args:
        estimated_states: Shape (N, 8) estimated states
        ground_truth_states: Shape (N, 8) ground truth states
        timestamps: Optional timestamps
        occlusion_mask: Optional occlusion mask
        title: Plot title
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if timestamps is not None:
        time_axis = np.array(timestamps)
        time_label = "Time (s)"
    else:
        time_axis = np.arange(len(estimated_states))
        time_label = "Frame"

    # Extract components
    est_vel = np.array(estimated_states[:, 2:4])
    gt_vel = np.array(ground_truth_states[:, 2:4])
    est_heading = np.array(estimated_states[:, 4])
    gt_heading = np.array(ground_truth_states[:, 4])

    # Velocity magnitude
    est_speed = np.linalg.norm(est_vel, axis=1)
    gt_speed = np.linalg.norm(gt_vel, axis=1)

    axes[0, 0].plot(time_axis, gt_speed, "k-", linewidth=2, label="Ground Truth", alpha=0.8)
    axes[0, 0].plot(time_axis, est_speed, "r-", linewidth=1.5, label="Estimated", alpha=0.7)

    if occlusion_mask is not None:
        axes[0, 0].fill_between(
            time_axis,
            0,
            np.max([gt_speed.max(), est_speed.max()]),
            where=occlusion_mask,
            alpha=0.2,
            color="orange",
        )

    axes[0, 0].set_xlabel(time_label)
    axes[0, 0].set_ylabel("Speed (cm/s)")
    axes[0, 0].set_title("Speed Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Velocity error
    vel_error = np.linalg.norm(est_vel - gt_vel, axis=1)
    axes[0, 1].plot(time_axis, vel_error, "b-", linewidth=1.5, alpha=0.8)
    axes[0, 1].axhline(
        y=10.0, color="red", linestyle="--", alpha=0.7, label="PRD Threshold (10 cm/s)"
    )

    if occlusion_mask is not None:
        axes[0, 1].fill_between(
            time_axis, 0, vel_error, where=occlusion_mask, alpha=0.3, color="orange"
        )

    axes[0, 1].set_xlabel(time_label)
    axes[0, 1].set_ylabel("Velocity Error (cm/s)")
    axes[0, 1].set_title("Velocity Error Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Heading comparison
    axes[1, 0].plot(
        time_axis, np.degrees(gt_heading), "k-", linewidth=2, label="Ground Truth", alpha=0.8
    )
    axes[1, 0].plot(
        time_axis, np.degrees(est_heading), "r-", linewidth=1.5, label="Estimated", alpha=0.7
    )

    if occlusion_mask is not None:
        y_min, y_max = axes[1, 0].get_ylim()
        axes[1, 0].fill_between(
            time_axis, y_min, y_max, where=occlusion_mask, alpha=0.2, color="orange"
        )

    axes[1, 0].set_xlabel(time_label)
    axes[1, 0].set_ylabel("Heading (degrees)")
    axes[1, 0].set_title("Heading Over Time")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Heading error
    heading_error = np.abs(np.degrees(_wrap_angle_difference(est_heading - gt_heading)))
    axes[1, 1].plot(time_axis, heading_error, "g-", linewidth=1.5, alpha=0.8)
    axes[1, 1].axhline(y=7.0, color="red", linestyle="--", alpha=0.7, label="PRD Threshold (7°)")

    if occlusion_mask is not None:
        axes[1, 1].fill_between(
            time_axis, 0, heading_error, where=occlusion_mask, alpha=0.3, color="orange"
        )

    axes[1, 1].set_xlabel(time_label)
    axes[1, 1].set_ylabel("Heading Error (degrees)")
    axes[1, 1].set_title("Heading Error Over Time")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_nees_analysis(
    estimated_states: jnp.ndarray,
    ground_truth_states: jnp.ndarray,
    covariances: jnp.ndarray,
    timestamps: Optional[jnp.ndarray] = None,
    title: str = "NEES Consistency Analysis",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot NEES consistency analysis including time series and histograms.

    Args:
        estimated_states: Shape (N, 8) estimated states
        ground_truth_states: Shape (N, 8) ground truth states
        covariances: Shape (N, 8, 8) covariance matrices
        timestamps: Optional timestamps
        title: Plot title
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Compute NEES metrics
    full_nees = compute_nees(estimated_states, ground_truth_states, covariances)
    pos_nees = compute_position_nees(estimated_states, ground_truth_states, covariances)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if timestamps is not None:
        time_axis = np.array(timestamps)
        time_label = "Time (s)"
    else:
        time_axis = np.arange(len(estimated_states))
        time_label = "Frame"

    # Full state NEES over time
    nees_values = full_nees["nees_values"]
    expected_nees = full_nees["nees_expected"]

    axes[0, 0].plot(time_axis, nees_values, "b-", linewidth=1.5, alpha=0.7, label="NEES")
    axes[0, 0].axhline(
        y=expected_nees,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Expected ({expected_nees:.0f})",
    )
    axes[0, 0].axhline(
        y=full_nees["nees_mean"],
        color="green",
        linestyle="-",
        alpha=0.7,
        label=f'Mean ({full_nees["nees_mean"]:.1f})',
    )

    # 95% confidence interval for chi-squared with 8 DOF
    chi2_lower = 2.18  # 2.5th percentile
    chi2_upper = 15.51  # 97.5th percentile
    axes[0, 0].axhline(y=chi2_lower, color="orange", linestyle=":", alpha=0.7, label="95% CI")
    axes[0, 0].axhline(y=chi2_upper, color="orange", linestyle=":", alpha=0.7)

    axes[0, 0].set_xlabel(time_label)
    axes[0, 0].set_ylabel("NEES Value")
    axes[0, 0].set_title("Full State NEES Over Time (8-DOF)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Position NEES over time
    pos_nees_values = pos_nees["position_nees_values"]
    pos_expected = pos_nees["position_nees_expected"]

    axes[0, 1].plot(
        time_axis, pos_nees_values, "g-", linewidth=1.5, alpha=0.7, label="Position NEES"
    )
    axes[0, 1].axhline(
        y=pos_expected,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Expected ({pos_expected:.0f})",
    )
    axes[0, 1].axhline(
        y=pos_nees["position_nees_mean"],
        color="blue",
        linestyle="-",
        alpha=0.7,
        label=f'Mean ({pos_nees["position_nees_mean"]:.1f})',
    )

    # 95% confidence interval for chi-squared with 2 DOF
    chi2_2dof_lower = 0.051  # 2.5th percentile
    chi2_2dof_upper = 7.378  # 97.5th percentile
    axes[0, 1].axhline(y=chi2_2dof_lower, color="orange", linestyle=":", alpha=0.7, label="95% CI")
    axes[0, 1].axhline(y=chi2_2dof_upper, color="orange", linestyle=":", alpha=0.7)

    axes[0, 1].set_xlabel(time_label)
    axes[0, 1].set_ylabel("Position NEES Value")
    axes[0, 1].set_title("Position NEES Over Time (2-DOF)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Full state NEES histogram
    axes[1, 0].hist(nees_values, bins=30, density=True, alpha=0.7, color="blue", label="Observed")

    # Theoretical chi-squared distribution
    x_theory = np.linspace(0, nees_values.max(), 100)
    from scipy.stats import chi2

    y_theory = chi2.pdf(x_theory, df=8)
    axes[1, 0].plot(x_theory, y_theory, "r-", linewidth=2, label="Chi² (8-DOF)")

    axes[1, 0].axvline(
        x=expected_nees,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Expected ({expected_nees:.0f})",
    )
    axes[1, 0].axvline(
        x=full_nees["nees_mean"],
        color="green",
        linestyle="-",
        alpha=0.7,
        label=f'Mean ({full_nees["nees_mean"]:.1f})',
    )

    axes[1, 0].set_xlabel("NEES Value")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Full State NEES Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Position NEES histogram
    axes[1, 1].hist(
        pos_nees_values, bins=30, density=True, alpha=0.7, color="green", label="Observed"
    )

    # Theoretical chi-squared distribution for 2-DOF
    x_theory_2 = np.linspace(0, pos_nees_values.max(), 100)
    y_theory_2 = chi2.pdf(x_theory_2, df=2)
    axes[1, 1].plot(x_theory_2, y_theory_2, "r-", linewidth=2, label="Chi² (2-DOF)")

    axes[1, 1].axvline(
        x=pos_expected,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Expected ({pos_expected:.0f})",
    )
    axes[1, 1].axvline(
        x=pos_nees["position_nees_mean"],
        color="blue",
        linestyle="-",
        alpha=0.7,
        label=f'Mean ({pos_nees["position_nees_mean"]:.1f})',
    )

    axes[1, 1].set_xlabel("Position NEES Value")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Position NEES Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_bias_traces(
    estimated_states: jnp.ndarray,
    ground_truth_states: Optional[jnp.ndarray] = None,
    timestamps: Optional[jnp.ndarray] = None,
    title: str = "IMU Bias Estimates",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Plot IMU bias estimates over time.

    Args:
        estimated_states: Shape (N, 8) estimated states [..., b_gz, b_ax, b_ay]
        ground_truth_states: Optional ground truth states for comparison
        timestamps: Optional timestamps
        title: Plot title
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    if timestamps is not None:
        time_axis = np.array(timestamps)
        time_label = "Time (s)"
    else:
        time_axis = np.arange(len(estimated_states))
        time_label = "Frame"

    # Extract bias estimates
    bias_gz = np.array(estimated_states[:, 5])  # Gyro z bias
    bias_ax = np.array(estimated_states[:, 6])  # Accel x bias
    bias_ay = np.array(estimated_states[:, 7])  # Accel y bias

    # Gyro bias
    axes[0].plot(time_axis, bias_gz, "b-", linewidth=1.5, label="Estimated", alpha=0.8)

    if ground_truth_states is not None:
        gt_bias_gz = np.array(ground_truth_states[:, 5])
        axes[0].plot(time_axis, gt_bias_gz, "k--", linewidth=2, label="Ground Truth", alpha=0.7)

    axes[0].set_xlabel(time_label)
    axes[0].set_ylabel("Gyro Z Bias (rad/s)")
    axes[0].set_title("Gyroscope Z-axis Bias")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accelerometer biases
    axes[1].plot(time_axis, bias_ax, "r-", linewidth=1.5, label="Accel X", alpha=0.8)
    axes[1].plot(time_axis, bias_ay, "g-", linewidth=1.5, label="Accel Y", alpha=0.8)

    if ground_truth_states is not None:
        gt_bias_ax = np.array(ground_truth_states[:, 6])
        gt_bias_ay = np.array(ground_truth_states[:, 7])
        axes[1].plot(time_axis, gt_bias_ax, "k--", linewidth=2, label="GT Accel X", alpha=0.7)
        axes[1].plot(time_axis, gt_bias_ay, "k:", linewidth=2, label="GT Accel Y", alpha=0.7)

    axes[1].set_xlabel(time_label)
    axes[1].set_ylabel("Accel Bias (m/s²)")
    axes[1].set_title("Accelerometer X/Y Biases")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_measurement_residuals(
    residuals: Dict[str, jnp.ndarray],
    timestamps: Optional[jnp.ndarray] = None,
    measurement_validity: Optional[Dict[str, jnp.ndarray]] = None,
    title: str = "Measurement Residuals",
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 10),
) -> plt.Figure:
    """
    Plot measurement residuals over time.

    Args:
        residuals: Dictionary with 'position' and 'heading' residual arrays
        timestamps: Optional timestamps
        measurement_validity: Optional validity masks for measurements
        title: Plot title
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if timestamps is not None:
        time_axis = np.array(timestamps)
        time_label = "Time (s)"
    else:
        # Use indices from residuals
        if "position" in residuals:
            time_axis = np.arange(len(residuals["position"]))
        else:
            time_axis = np.arange(100)  # fallback
        time_label = "Frame"

    # Position residuals
    if "position" in residuals:
        pos_residuals = np.array(residuals["position"])

        if pos_residuals.ndim == 2 and pos_residuals.shape[1] >= 2:
            # Separate x and y components
            axes[0, 0].plot(
                time_axis, pos_residuals[:, 0], "r-", linewidth=1.5, label="X residual", alpha=0.8
            )
            axes[0, 1].plot(
                time_axis, pos_residuals[:, 1], "g-", linewidth=1.5, label="Y residual", alpha=0.8
            )

            # Mark invalid measurements
            if measurement_validity and "position" in measurement_validity:
                invalid_mask = ~measurement_validity["position"]
                if np.any(invalid_mask):
                    axes[0, 0].scatter(
                        time_axis[invalid_mask],
                        pos_residuals[invalid_mask, 0],
                        c="red",
                        s=20,
                        alpha=0.5,
                        label="Invalid",
                    )
                    axes[0, 1].scatter(
                        time_axis[invalid_mask],
                        pos_residuals[invalid_mask, 1],
                        c="red",
                        s=20,
                        alpha=0.5,
                        label="Invalid",
                    )
        else:
            # Single residual magnitude
            axes[0, 0].plot(
                time_axis, pos_residuals, "b-", linewidth=1.5, label="Position residual", alpha=0.8
            )

        axes[0, 0].set_xlabel(time_label)
        axes[0, 0].set_ylabel("X Position Residual (cm)")
        axes[0, 0].set_title("X Position Residual")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel(time_label)
        axes[0, 1].set_ylabel("Y Position Residual (cm)")
        axes[0, 1].set_title("Y Position Residual")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Heading residuals
    if "heading" in residuals:
        heading_residuals = np.array(residuals["heading"])
        heading_residuals_deg = np.degrees(heading_residuals)

        axes[1, 0].plot(
            time_axis,
            heading_residuals_deg,
            "purple",
            linewidth=1.5,
            label="Heading residual",
            alpha=0.8,
        )

        # Mark invalid measurements
        if measurement_validity and "heading" in measurement_validity:
            invalid_mask = ~measurement_validity["heading"]
            if np.any(invalid_mask):
                axes[1, 0].scatter(
                    time_axis[invalid_mask],
                    heading_residuals_deg[invalid_mask],
                    c="red",
                    s=20,
                    alpha=0.5,
                    label="Invalid",
                )

        axes[1, 0].set_xlabel(time_label)
        axes[1, 0].set_ylabel("Heading Residual (degrees)")
        axes[1, 0].set_title("Heading Residual")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Combined residual magnitude
    if "position" in residuals:
        pos_residuals = np.array(residuals["position"])
        if pos_residuals.ndim == 2:
            residual_mag = np.linalg.norm(pos_residuals, axis=1)
        else:
            residual_mag = np.abs(pos_residuals)

        axes[1, 1].plot(
            time_axis, residual_mag, "blue", linewidth=1.5, label="Position magnitude", alpha=0.8
        )

    if "heading" in residuals:
        heading_residuals = np.array(residuals["heading"])
        heading_mag = np.abs(heading_residuals)

        # Scale to be comparable with position (rough heuristic)
        heading_mag_scaled = heading_mag * 10  # 10 cm per radian roughly
        axes[1, 1].plot(
            time_axis,
            heading_mag_scaled,
            "purple",
            linewidth=1.5,
            label="Heading magnitude (scaled)",
            alpha=0.8,
        )

    axes[1, 1].set_xlabel(time_label)
    axes[1, 1].set_ylabel("Residual Magnitude")
    axes[1, 1].set_title("Combined Residual Magnitudes")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _wrap_angle_difference(angle_diff: np.ndarray) -> np.ndarray:
    """Wrap angle difference to [-π, π] range."""
    return np.remainder(angle_diff + np.pi, 2 * np.pi) - np.pi

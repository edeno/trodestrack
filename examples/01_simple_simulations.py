"""Simple Analytic Simulations Demo.

This script demonstrates three minimal simulation scenarios with known
analytic solutions, ideal for unit testing and filter development:

1. Stationary: No motion, constant position
2. Constant velocity: Straight line motion
3. Circular: Constant angular velocity

Each simulation provides perfect ground truth for validation.

Usage:
    uv run python examples/01_simple_simulations.py

Output:
    Saves 1 PNG file showing all three scenarios side-by-side.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)

# =============================================================================
# Matplotlib Configuration
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    }
)

# Color palette (ColorBrewer)
COLORS = {
    "blue": "#2166AC",
    "red": "#B2182B",
    "green": "#1B7837",
    "orange": "#D6604D",
    "purple": "#762A83",
    "gray": "#666666",
}


# =============================================================================
# Plotting Function
# =============================================================================


def plot_simple_simulations(
    sim_stat: dict[str, np.ndarray],
    sim_vel: dict[str, np.ndarray],
    sim_circ: dict[str, np.ndarray],
) -> plt.Figure:
    """Create side-by-side comparison of three simple simulations.

    Args:
        sim_stat: Stationary simulation results
        sim_vel: Constant velocity simulation results
        sim_circ: Circular motion simulation results

    Returns:
        Matplotlib figure with 3×4 grid of subplots
    """
    fig, axes = plt.subplots(4, 3, figsize=(13, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    simulations = [sim_stat, sim_vel, sim_circ]
    titles = ["Stationary", "Constant Velocity", "Circular"]

    for col, (sim, title) in enumerate(zip(simulations, titles, strict=False)):
        # Extract data
        t_imu = sim["t_imu"]
        t_cam = sim["t_cam_exp"]
        X = sim["X_truth"]
        U = sim["U_imu"]
        Z = sim["Z_cam_led1"]
        mask = sim["mask_cam"]

        # -----------------------------------------------------------------------
        # Row 0: Trajectory
        # -----------------------------------------------------------------------
        ax = axes[0, col]

        # Ground truth path
        ax.plot(
            X[:, 0], X[:, 1], "-", linewidth=1.5, color=COLORS["blue"], label="truth"
        )

        # Camera measurements (valid only)
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=8,
            c=COLORS["orange"],
            alpha=0.6,
            label="camera",
            edgecolors="none",
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title, fontweight="normal", fontsize=10)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")

        # -----------------------------------------------------------------------
        # Row 1: Position Time Series
        # -----------------------------------------------------------------------
        ax = axes[1, col]

        ax.plot(t_imu, X[:, 0], "-", linewidth=1, color=COLORS["blue"], label="x")
        ax.plot(t_imu, X[:, 1], "-", linewidth=1, color=COLORS["red"], label="y")

        # Camera measurements
        ax.scatter(t_cam[mask], Z[mask, 0], s=6, c=COLORS["blue"], alpha=0.5)
        ax.scatter(t_cam[mask], Z[mask, 1], s=6, c=COLORS["red"], alpha=0.5)

        ax.set_xlabel("time (s)")
        ax.set_ylabel("position (m)")
        ax.legend(loc="upper right")

        # -----------------------------------------------------------------------
        # Row 2: Velocity
        # -----------------------------------------------------------------------
        ax = axes[2, col]

        speed = np.hypot(X[:, 2], X[:, 3])
        ax.plot(t_imu, speed, "-", linewidth=1, color=COLORS["green"])

        ax.set_xlabel("time (s)")
        ax.set_ylabel("speed (m/s)")

        # -----------------------------------------------------------------------
        # Row 3: IMU (Gyro + Accel magnitude)
        # -----------------------------------------------------------------------
        ax = axes[3, col]

        # Gyro (convert to deg/s)
        gyro_deg = np.rad2deg(U[:, 0])
        ax.plot(t_imu, gyro_deg, "-", linewidth=0.5, color=COLORS["purple"], alpha=0.7)

        ax.set_xlabel("time (s)")
        ax.set_ylabel("gyro (°/s)")

        # Twin axis for accel magnitude
        ax2 = ax.twinx()
        accel_mag = np.sqrt(U[:, 1] ** 2 + U[:, 2] ** 2)
        ax2.plot(
            t_imu, accel_mag, "-", linewidth=0.5, color=COLORS["orange"], alpha=0.7
        )
        ax2.set_ylabel("accel mag (m/s²)", color=COLORS["orange"])
        ax2.tick_params(axis="y", labelcolor=COLORS["orange"])

    fig.suptitle(
        "Simple Analytic Simulations (10s duration)", fontsize=11, fontweight="normal"
    )

    return fig


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Generate and plot simple simulations."""
    print("=" * 70)
    print("Simple Analytic Simulations")
    print("=" * 70)

    # Shared configuration (low noise, 10s duration)
    config = SimpleSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.0001,
        accel_noise_density=0.03,
        gyro_bias_std=0.001,
        accel_bias_std=0.01,
        cam_noise_std=0.002,  # 2 mm
        cam_dropout_prob=0.10,
    )

    # -------------------------------------------------------------------------
    # 1. Stationary Simulation
    # -------------------------------------------------------------------------
    print("\n1. Stationary simulation...")
    sim_stat = simulate_stationary(
        config, position=np.array([0.5, 0.5]), heading=np.pi / 6, seed=42
    )

    print(
        f"   Position: ({sim_stat['X_truth'][0, 0]:.2f}, {sim_stat['X_truth'][0, 1]:.2f}) m"
    )
    print(f"   Heading: {np.rad2deg(sim_stat['X_truth'][0, 4]):.1f}°")
    print(
        f"   Valid camera frames: {sim_stat['mask_cam'].sum()}/{len(sim_stat['mask_cam'])}"
    )

    # -------------------------------------------------------------------------
    # 2. Constant Velocity Simulation
    # -------------------------------------------------------------------------
    print("\n2. Constant velocity simulation...")
    sim_vel = simulate_constant_velocity(
        config,
        initial_position=np.array([0.1, 0.2]),
        velocity=np.array([0.15, 0.08]),
        seed=42,
    )

    vx = sim_vel["X_truth"][0, 2]
    vy = sim_vel["X_truth"][0, 3]
    speed = np.hypot(vx, vy)
    print(f"   Velocity: ({vx:.3f}, {vy:.3f}) m/s")
    print(f"   Speed: {speed:.3f} m/s = {speed * 100:.1f} cm/s")
    print(f"   Distance traveled: {speed * config.duration_s:.2f} m")

    # -------------------------------------------------------------------------
    # 3. Circular Motion Simulation
    # -------------------------------------------------------------------------
    print("\n3. Circular motion simulation...")
    radius = 0.25
    omega = 1.5  # rad/s

    sim_circ = simulate_circular(
        config,
        center=np.array([0.5, 0.5]),
        radius=radius,
        angular_velocity=omega,
        seed=42,
    )

    print(f"   Radius: {radius:.2f} m")
    print(f"   Angular velocity: {omega:.2f} rad/s = {np.rad2deg(omega):.1f}°/s")
    print(f"   Linear speed: {radius * omega:.3f} m/s")
    print(f"   Rotations: {omega * config.duration_s / (2 * np.pi):.2f}")

    # -------------------------------------------------------------------------
    # Generate Plot
    # -------------------------------------------------------------------------
    print("\nGenerating plots...")
    from pathlib import Path

    fig = plot_simple_simulations(sim_stat, sim_vel, sim_circ)
    output_path = Path(__file__).parent / "01_simple_simulations.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"   → {output_path}")

    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Ground Truth Validation (No Noise)")
    print("=" * 70)

    # Stationary: position should never change
    pos_std = np.std(sim_stat["X_truth"][:, :2], axis=0)
    print(
        f"\nStationary position std: ({pos_std[0]:.2e}, {pos_std[1]:.2e}) m (expect 0)"
    )

    # Constant velocity: velocity should be constant
    vel_std = np.std(sim_vel["X_truth"][:, 2:4], axis=0)
    print(f"Constant velocity std: ({vel_std[0]:.2e}, {vel_std[1]:.2e}) m/s (expect 0)")

    # Circular: distance from center should be constant
    center = np.array([0.5, 0.5])
    dist = np.sqrt(
        (sim_circ["X_truth"][:, 0] - center[0]) ** 2
        + (sim_circ["X_truth"][:, 1] - center[1]) ** 2
    )
    dist_std = np.std(dist)
    print(f"Circular radius std: {dist_std:.2e} m (expect 0)")

    print("\n" + "=" * 70)
    print("✓ All simulations completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()

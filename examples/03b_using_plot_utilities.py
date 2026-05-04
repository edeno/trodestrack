"""Using TrodesTrack's Built-in Plotting Utilities - Educational Demonstration.

LEARNING OBJECTIVES:
-------------------
This example teaches you how to use trodestrack's pre-built plotting and diagnostic
utilities instead of manually creating matplotlib figures. You will learn:

1. **State Layouts**: How to extract states robustly using the StateLayout system
2. **Plot Utilities**: Using qa.plot_* functions for common visualizations
3. **DRY Principle**: Avoiding code duplication with reusable plotting functions
4. **Covariance Visualization**: Plotting uncertainty ellipses and ±2σ bands

WHY THIS MATTERS:
-----------------
Writing custom matplotlib code for every analysis is time-consuming and error-prone.
TrodesTrack provides publication-quality plotting functions that handle:
- Proper axis labels and units
- PRD threshold lines
- Statistical bounds (NEES, NIS chi-squared limits)
- Consistent styling across all plots

This example shows you the RECOMMENDED way to visualize filter results.

KEY PRINCIPLE: Use State Layouts, Not Magic Indices
----------------------------------------------------
❌ BAD:  positions = result.filtered_means[:, 0:2]  # Fragile!
✅ GOOD: positions = result.filtered_means[:, layout.pos_idx]  # Robust!

State layouts make your code:
- Dimension-agnostic (works with 5D, 8D, 10D, 15D states)
- Self-documenting (layout.pos_idx is clearer than [:, 0:2])
- Robust to changes (switching state modes doesn't break your code)

USAGE:
------
    uv run python examples/03b_using_plot_utilities.py

OUTPUT:
-------
    - Console: Summary of available plotting utilities
    - Files: 03b_position_error.png, 03b_velocity_error.png, etc.

ESTIMATED RUNTIME: ~5 seconds

WHAT YOU'LL LEARN:
------------------
- How to import and use qa.plot_* functions
- How to extract states using StateLayout for dimension-agnostic code
- How to plot covariance ellipses for uncertainty visualization
- How to avoid matplotlib boilerplate

NEXT STEPS:
-----------
After mastering plotting utilities, you can:
- Use generate_qa_report() for comprehensive PDF reports (see example 08)
- Create custom multi-panel figures by composing plot utilities
- Extend utilities for project-specific visualizations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import EKF and state layout system
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout

# Import plotting utilities (DRY - don't reinvent the wheel!)
from trodestrack.qa import (
    plot_covariance_ellipse,
    plot_heading_error,
    plot_nees_histogram,
    plot_position_error,
    plot_velocity_error,
)

# Import simulation
from trodestrack.sim.simple import SimpleSimConfig, simulate_circular

# Output directory
OUTPUT_DIR = Path(__file__).parent


# =============================================================================
# Helper Functions
# =============================================================================


def print_section_header(title: str) -> None:
    """Print a clear section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Demonstrate trodestrack's built-in plotting utilities."""
    print_section_header("Using TrodesTrack Plotting Utilities")

    print(
        """
    This example shows the RECOMMENDED way to visualize filter results:
    • Use state layouts for robust state extraction
    • Use qa.plot_* functions instead of custom matplotlib code
    • Follow DRY principle (Don't Repeat Yourself)

    Let's begin!
    """
    )

    # -------------------------------------------------------------------------
    # Step 1: Generate Data and Run Filter
    # -------------------------------------------------------------------------
    print_section_header("Step 1: Generate Data and Run EKF")

    # Simulation config using REALISTIC SpikeGadgets IMU specifications
    sim_config = SimpleSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        fs_cam=30.0,
        # SpikeGadgets IMU specs (from PRD.md):
        gyro_noise_density=0.000175,  # 0.01 °/s/√Hz (SpikeGadgets spec)
        accel_noise_density=0.00196,  # 0.2 mg/√Hz (SpikeGadgets spec)
        gyro_bias_std=0.001,  # ~0.06 °/s bias std
        accel_bias_std=0.01,  # ~1 mg bias std
        cam_noise_std=0.005,  # 5 mm camera noise
        cam_dropout_prob=0.0,
    )

    print("   Simulating circular motion...")
    sim = simulate_circular(
        sim_config,
        center=np.array([1.0, 1.0]),
        radius=0.5,
        angular_velocity=0.5,
        seed=42,
    )

    # EKF configuration using REALISTIC SpikeGadgets IMU specifications.
    # Pin to the 8D ``2d_full`` layout so layout.heading_idx (4) is in range
    # for the simulator's 5D X_truth array.
    ekf_config = EKFConfig(
        state_mode="2d_full",
        # Process noise spectral densities (tuned for good performance)
        process_noise_pos=2e-3,  # m^2/s^3
        process_noise_vel=1e-1,  # (m/s)^2/s
        process_noise_heading=1e-3,  # rad^2/s
        process_noise_gyro_bias=5e-7,  # (rad/s)^2/s
        process_noise_accel_bias=5e-5,  # (m/s²)^2/s
        # Measurement noise
        measurement_noise_pos=0.005**2,  # m^2
        measurement_noise_heading=0.05**2,  # rad^2
        # SpikeGadgets IMU specs (MUST match simulation!)
        imu_gyro_noise_density=0.000175,  # rad/s/√Hz
        imu_accel_noise_density=0.00196,  # m/s²/√Hz
        # Physics
        damping_coeff=0.5,
        led_distance=0.04,
        use_heading_measurement=True,
        enable_zupt=False,
    )

    print("   Running EKF...")
    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    print("   ✓ Filter complete")

    # -------------------------------------------------------------------------
    # Step 2: Extract States Using State Layouts (BEST PRACTICE!)
    # -------------------------------------------------------------------------
    print_section_header("Step 2: Extract States Using State Layouts")

    # Get state layout from config (dimension-agnostic!)
    layout = get_layout(ekf_config.state_mode)

    print(
        f"""
    State Layout Information:
      Mode:            {ekf_config.state_mode}
      Dimension:       {layout.n}D
      Position indices: {layout.pos_idx}
      Velocity indices: {layout.vel_idx}
      Heading index:    {layout.heading_idx}
      Gyro bias:        {layout.bias_gyro_idx}
      Accel bias:       {layout.bias_accel_idx}

    KEY INSIGHT: Using layout indices makes code work with ANY state dimension!
    """
    )

    # Extract filter results using layout (ROBUST - dimension-agnostic!)
    t_cam = sim["t_cam_exp"]
    positions_est = result.filtered_means[:, layout.pos_idx]  # (N, 2) meters
    velocities_est = result.filtered_means[:, layout.vel_idx]  # (N, 2) m/s
    headings_est = result.filtered_means[:, layout.heading_idx]  # (N,) radians

    # Extract ground truth (interpolate IMU-rate truth to camera times)
    from trodestrack.sim.utils import interp_angle

    t_imu = sim["t_imu"]
    X_truth = sim["X_truth"]
    positions_true = np.column_stack(
        [np.interp(t_cam, t_imu, X_truth[:, i]) for i in layout.pos_idx]
    )
    velocities_true = np.column_stack(
        [np.interp(t_cam, t_imu, X_truth[:, i]) for i in layout.vel_idx]
    )
    headings_true = interp_angle(t_cam, t_imu, X_truth[:, layout.heading_idx])

    # Extract covariances using layout (for uncertainty visualization)
    P = np.array(result.filtered_covariances)  # (N, 8, 8) - convert from JAX to numpy
    # Extract position covariance submatrix using layout indices
    pos_idx = list(layout.pos_idx)
    pos_cov = P[:, pos_idx, :][:, :, pos_idx]  # (N, 2, 2)

    print(
        f"""
    Extracted Data Shapes:
      Positions:       {positions_est.shape} (time, spatial_dim)
      Velocities:      {velocities_est.shape}
      Headings:        {headings_est.shape}
      Position cov:    {pos_cov.shape} (time, spatial_dim, spatial_dim)

    ✓ All extractions used layout indices - code will work with 10D or 15D states!
    """
    )

    # -------------------------------------------------------------------------
    # Step 3: Use Plotting Utilities (No matplotlib boilerplate!)
    # -------------------------------------------------------------------------
    print_section_header("Step 3: Use Pre-built Plotting Utilities")

    print(
        """
    Instead of writing 100+ lines of matplotlib code, we use trodestrack's
    built-in plotting functions. These provide:
    • Consistent styling and labeling
    • PRD threshold lines
    • Statistical bounds (chi-squared limits for NEES)
    • Publication-quality output
    """
    )

    # -------------------------------------------------------------------------
    # Plot 1: Position Error Time Series
    # -------------------------------------------------------------------------
    print("\n   Creating position error plot...")
    fig1, ax1 = plot_position_error(
        t=t_cam,
        positions_true=positions_true,
        positions_est=positions_est,
    )
    # Add PRD threshold line manually (2 cm)
    ax1.axhline(
        0.02,
        linestyle="--",
        color="red",
        linewidth=1,
        alpha=0.5,
        label="PRD Target (2 cm)",
    )
    ax1.legend()
    fig1.savefig(OUTPUT_DIR / "03b_position_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("   💾 Saved: 03b_position_error.png")

    # -------------------------------------------------------------------------
    # Plot 2: Velocity Error Time Series
    # -------------------------------------------------------------------------
    print("   Creating velocity error plot...")
    fig2, ax2 = plot_velocity_error(
        t=t_cam,
        velocities_true=velocities_true,
        velocities_est=velocities_est,
    )
    # Add PRD threshold line manually (10 cm/s = 0.1 m/s)
    ax2.axhline(
        0.10,
        linestyle="--",
        color="red",
        linewidth=1,
        alpha=0.5,
        label="PRD Target (10 cm/s)",
    )
    ax2.legend()
    fig2.savefig(OUTPUT_DIR / "03b_velocity_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("   💾 Saved: 03b_velocity_error.png")

    # -------------------------------------------------------------------------
    # Plot 3: Heading Error Time Series
    # -------------------------------------------------------------------------
    print("   Creating heading error plot...")
    fig3, ax3 = plot_heading_error(
        t=t_cam,
        headings_true=headings_true,
        headings_est=headings_est,
    )
    # Add PRD threshold line manually (7°)
    ax3.axhline(
        7.0,
        linestyle="--",
        color="red",
        linewidth=1,
        alpha=0.5,
        label="PRD Target (7°)",
    )
    ax3.legend()
    fig3.savefig(OUTPUT_DIR / "03b_heading_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("   💾 Saved: 03b_heading_error.png")

    # -------------------------------------------------------------------------
    # Plot 4: NEES Histogram (Filter Consistency Check)
    # -------------------------------------------------------------------------
    print("   Creating NEES histogram...")

    # Compute NEES manually (for educational purposes)
    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    # Build full truth state at camera times (including biases)
    bias_gyro_truth = np.interp(t_cam, t_imu, sim["bias_gyro"])
    bias_ax_truth = np.interp(t_cam, t_imu, sim["bias_accel_x"])
    bias_ay_truth = np.interp(t_cam, t_imu, sim["bias_accel_y"])
    truth_full = np.column_stack(
        [
            positions_true,
            velocities_true,
            headings_true,
            bias_gyro_truth,
            bias_ax_truth,
            bias_ay_truth,
        ]
    )

    # Compute NEES
    nees_values = []
    X_est_np = np.array(result.filtered_means)  # Convert to numpy for modification
    P_np = np.array(result.filtered_covariances)
    for i in range(len(t_cam)):
        err = X_est_np[i] - truth_full[i]
        # Wrap heading error
        err[layout.heading_idx] = angle_diff(
            X_est_np[i, layout.heading_idx], truth_full[i, layout.heading_idx]
        )
        nees = err @ np.linalg.solve(P_np[i], err)
        nees_values.append(nees)
    nees_values = np.array(nees_values)

    fig4, _ax4 = plot_nees_histogram(
        nees=nees_values,
        state_dim=layout.n,  # Use layout dimension (8D for "2d_full")
    )
    fig4.savefig(OUTPUT_DIR / "03b_nees_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("   💾 Saved: 03b_nees_histogram.png")

    # -------------------------------------------------------------------------
    # Plot 5: Covariance Ellipse (Uncertainty Visualization)
    # -------------------------------------------------------------------------
    print("   Creating covariance ellipse plot...")

    # Use the final position and covariance for the ellipse plot
    fig5, ax5 = plot_covariance_ellipse(
        mean=positions_est[-1],  # Final position
        cov=pos_cov[-1],  # Final covariance
        n_std=[1.0, 2.0, 3.0],  # 1σ, 2σ, 3σ ellipses
        trajectory=positions_est,  # Show full trajectory
        color="blue",
        alpha=0.3,
    )

    # Add ground truth trajectory for comparison
    ax5.plot(
        positions_true[:, 0],
        positions_true[:, 1],
        "-",
        linewidth=2,
        color="gray",
        alpha=0.4,
        label="Ground Truth",
        zorder=1,
    )
    ax5.legend(loc="best")

    fig5.savefig(
        OUTPUT_DIR / "03b_covariance_ellipse.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig5)
    print("   💾 Saved: 03b_covariance_ellipse.png")

    # -------------------------------------------------------------------------
    # Step 4: Compare with Manual Approach
    # -------------------------------------------------------------------------
    print_section_header("Step 4: Lines of Code Comparison")

    print(
        """
    📊 CODE COMPLEXITY COMPARISON:

    Manual matplotlib approach (see example 03):
      • ~120 lines of plotting code per scenario
      • Must handle axis creation, styling, labeling manually
      • Easy to introduce inconsistencies across plots

    Using plot utilities (this example):
      • ~5 lines per plot
      • Automatic styling, labeling, and PRD thresholds
      • Consistent across all analyses

    RECOMMENDATION: Always use plot utilities for standard visualizations!
    Reserve custom matplotlib only for novel plot types.
    """
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section_header("Summary: What You Learned")

    print(
        """
    🎓 KEY TAKEAWAYS:

    1. STATE LAYOUTS ARE ESSENTIAL:
       • ✅ DO:  layout = get_layout(config.state_mode)
                 pos = states[:, layout.pos_idx]
       • ❌ DON'T: pos = states[:, 0:2]  # Fragile!

    2. USE PLOTTING UTILITIES:
       • plot_position_error() - Position vs time with PRD threshold
       • plot_velocity_error() - Velocity vs time with PRD threshold
       • plot_heading_error() - Heading vs time with PRD threshold
       • plot_nees_histogram() - NEES with chi-squared bounds
       • plot_covariance_ellipse() - Uncertainty visualization

    3. COVARIANCE EXTRACTION:
       • Full: P = result.filtered_covariances  # (N, n, n)
       • Position: pos_cov = P[:, layout.pos_idx, :][:, :, layout.pos_idx]
       • Std dev: pos_std = np.sqrt(np.diagonal(pos_cov, axis1=1, axis2=2))

    4. DRY PRINCIPLE:
       • Don't write matplotlib boilerplate for common plots
       • Use trodestrack's functions for consistency
       • Focus on analysis, not plotting code

    NEXT STEPS:
    • See example 08 for generate_qa_report() (comprehensive PDF)
    • See README for full plotting API documentation
    • Build custom multi-panel figures by composing utilities

    Happy analyzing! 🚀
    """
    )


if __name__ == "__main__":
    main()

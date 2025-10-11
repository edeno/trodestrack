"""Example: Generate a sample QA report from synthetic filter results.

This script demonstrates how to use the QA report generation module to create
a comprehensive PDF report with filter diagnostics.
"""

from pathlib import Path

import numpy as np

from trodestrack.qa import generate_qa_report

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
duration = 30.0  # seconds
dt = 1 / 30.0  # 30 Hz camera rate
N = int(duration / dt)
t = np.linspace(0, duration, N)

print(f"Generating example filter results ({duration}s, {N} samples)...")

# Generate synthetic circular trajectory
angular_velocity = 0.3  # rad/s
radius = 0.3  # meters
theta_traj = t * angular_velocity

# Ground truth states
pos_true = np.column_stack([radius * np.cos(theta_traj), radius * np.sin(theta_traj)])
vel_true = np.column_stack(
    [
        -radius * angular_velocity * np.sin(theta_traj),
        radius * angular_velocity * np.cos(theta_traj),
    ]
)
heading_true = theta_traj + np.pi / 2  # Tangent to circle

# Add realistic filter errors
pos_noise_std = 0.01  # 1 cm position noise
vel_noise_std = 0.02  # 2 cm/s velocity noise
heading_noise_std = np.deg2rad(3)  # 3 degree heading noise

pos_est = pos_true + np.random.randn(N, 2) * pos_noise_std
vel_est = vel_true + np.random.randn(N, 2) * vel_noise_std
heading_est = heading_true + np.random.randn(N) * heading_noise_std

# Generate synthetic NEES values (chi-squared with df=8)
# Add some variation to make it realistic
nees = np.random.chisquare(df=8, size=N) * 1.2  # Slightly overconfident filter

# Generate synthetic NIS values (chi-squared with df=4 for dual-LED measurements)
nis = np.random.chisquare(df=4, size=int(N * 0.8))  # Some measurements dropped

# Filter configuration
config = {
    "filter_type": "EKF",
    "process_noise_pos": 0.02,
    "process_noise_vel": 0.5,
    "process_noise_heading": 0.02,
    "measurement_noise": 0.005**2,
    "imu_gyro_noise_density": 0.001,
    "imu_accel_noise_density": 0.05,
    "damping_coefficient": 0.4,
    "led_distance": 0.04,
    "enable_zupt": True,
    "use_heading_measurement": True,
}

# Output path
output_path = Path(__file__).parent / "example_qa_report.pdf"

print(f"Generating PDF report: {output_path}")

# Generate comprehensive QA report
generate_qa_report(
    pdf_path=output_path,
    t=t,
    positions_true=pos_true,
    positions_est=pos_est,
    velocities_true=vel_true,
    velocities_est=vel_est,
    headings_true=heading_true,
    headings_est=heading_est,
    nees=nees,
    state_dim=8,
    nis=nis,
    measurement_dim=4,
    config=config,
    title="Example EKF Performance Report - Circular Trajectory",
)

print("\n✓ Report generated successfully!")
print(f"  File: {output_path}")
print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
print("\nOpen the PDF to view:")
print("  - Summary statistics (position/velocity/heading RMSE)")
print("  - PRD threshold comparisons")
print("  - NEES/NIS consistency checks")
print("  - Time series plots")
print("  - 2D trajectory visualization")
print("  - Filter configuration")

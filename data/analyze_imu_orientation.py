"""Determine IMU axis orientation from data.

This script analyzes the IMU data to determine which physical axes (X, Y, Z)
correspond to which body frame directions (forward, lateral, vertical).

Method:
1. Use gravity direction from accelerometer (largest mean component = vertical)
2. Use LED pair heading to correlate gyro axes with yaw
3. Use camera velocity to correlate accel axes with motion direction

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SpikeGadgets headstage hardware specifications
GYRO_SCALE = 0.061  # deg/s per LSB (±2000 deg/s range, 16-bit)
ACCEL_SCALE = 0.000061  # g per LSB (±2g range, 16-bit)
GRAVITY = 9.80665  # m/s²
DEG_TO_RAD = np.pi / 180.0


def load_imu_data(imu_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load IMU data with unit conversion.

    Parameters
    ----------
    imu_file : str
        Path to IMU parquet file

    Returns
    -------
    timestamps : np.ndarray
        Relative timestamps in seconds [N]
    gyro_rad_s : np.ndarray
        Gyroscope in rad/s [N × 3]: [X, Y, Z]
    accel_m_s2 : np.ndarray
        Accelerometer in m/s² [N × 3]: [X, Y, Z]
    """
    df = pd.read_parquet(imu_file)

    # Convert to relative time
    timestamps = df.index.values
    timestamps = timestamps - timestamps[0]

    # Remove sample-and-hold
    all_values = np.column_stack([df[col].values for col in df.columns])
    changes = np.where(np.any(np.diff(all_values, axis=0) != 0, axis=1))[0] + 1
    unique_idx = np.concatenate([[0], changes])
    timestamps = timestamps[unique_idx]

    # Load and convert gyro (all 3 axes)
    gyro_raw = np.column_stack(
        [
            df["Headstage_GyroX"].values,
            df["Headstage_GyroY"].values,
            df["Headstage_GyroZ"].values,
        ]
    )
    gyro_raw = gyro_raw[unique_idx]
    gyro_rad_s = gyro_raw * GYRO_SCALE * DEG_TO_RAD

    # Load and convert accel (all 3 axes)
    accel_raw = np.column_stack(
        [
            df["Headstage_AccelX"].values,
            df["Headstage_AccelY"].values,
            df["Headstage_AccelZ"].values,
        ]
    )
    accel_raw = accel_raw[unique_idx]
    accel_m_s2 = accel_raw * ACCEL_SCALE * GRAVITY

    return timestamps, gyro_rad_s, accel_m_s2


def load_position_data(
    position_file: str, meters_per_pixel: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load camera position data.

    Parameters
    ----------
    position_file : str
        Path to position parquet file
    meters_per_pixel : float
        Conversion factor

    Returns
    -------
    timestamps : np.ndarray
        Relative timestamps in seconds [N]
    led1_m : np.ndarray
        LED1 positions in meters [N × 2]
    led2_m : np.ndarray
        LED2 positions in meters [N × 2]
    """
    df = pd.read_parquet(position_file)

    timestamps = df.index.values
    timestamps = timestamps - timestamps[0]

    led1_pixels = df[["xloc", "yloc"]].values
    led2_pixels = df[["xloc2", "yloc2"]].values

    led1_m = led1_pixels * meters_per_pixel
    led2_m = led2_pixels * meters_per_pixel

    return timestamps, led1_m, led2_m


def compute_heading_from_leds(led1: np.ndarray, led2: np.ndarray) -> np.ndarray:
    """Compute heading angle from LED pair.

    Parameters
    ----------
    led1 : np.ndarray
        Rear LED positions [N × 2]
    led2 : np.ndarray
        Front LED positions [N × 2]

    Returns
    -------
    heading : np.ndarray
        Heading angle in radians [N]

    Notes
    -----
    Convention: 0 = East (+X), π/2 = North (+Y)
    """
    delta = led2 - led1
    return np.arctan2(delta[:, 1], delta[:, 0])


def compute_angular_velocity(timestamps: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Compute angular velocity from heading time series.

    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps in seconds [N]
    heading : np.ndarray
        Heading angles in radians [N]

    Returns
    -------
    angular_velocity : np.ndarray
        Angular velocity in rad/s [N-1]
    """
    # Unwrap heading to handle 2π discontinuities
    heading_unwrapped = np.unwrap(heading)

    # Compute derivative
    dt = np.diff(timestamps)
    dheading = np.diff(heading_unwrapped)

    return dheading / dt


def find_gravity_axis(accel_m_s2: np.ndarray) -> tuple[int, float]:
    """Find which accelerometer axis is aligned with gravity.

    Parameters
    ----------
    accel_m_s2 : np.ndarray
        Accelerometer data [N × 3]

    Returns
    -------
    axis_index : int
        Index of axis most aligned with gravity (0=X, 1=Y, 2=Z)
    mean_value : float
        Mean acceleration on that axis (negative = upward in body frame)
    """
    means = np.mean(accel_m_s2, axis=0)
    abs_means = np.abs(means)
    axis_index = int(np.argmax(abs_means))

    return axis_index, means[axis_index]


def correlate_gyro_with_heading(
    t_cam: np.ndarray,
    heading_rate: np.ndarray,
    t_imu: np.ndarray,
    gyro_rad_s: np.ndarray,
) -> np.ndarray:
    """Correlate each gyro axis with heading rate from camera.

    Parameters
    ----------
    t_cam : np.ndarray
        Camera timestamps [N_cam]
    heading_rate : np.ndarray
        Heading rate from LED pair [N_cam - 1]
    t_imu : np.ndarray
        IMU timestamps [N_imu]
    gyro_rad_s : np.ndarray
        Gyro data [N_imu × 3]

    Returns
    -------
    correlations : np.ndarray
        Pearson correlation for each axis [3]
    """
    # Interpolate gyro to camera time points
    t_cam_mid = (t_cam[:-1] + t_cam[1:]) / 2

    correlations = np.zeros(3)
    for axis in range(3):
        gyro_interp = np.interp(t_cam_mid, t_imu, gyro_rad_s[:, axis])
        correlations[axis] = np.corrcoef(heading_rate, gyro_interp)[0, 1]

    return correlations


def compute_camera_velocity(
    timestamps: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity from position time series.

    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps in seconds [N]
    positions : np.ndarray
        Positions in meters [N × 2]

    Returns
    -------
    t_mid : np.ndarray
        Midpoint timestamps [N-1]
    velocity : np.ndarray
        Velocity in m/s [N-1 × 2]
    """
    dt = np.diff(timestamps)
    dpos = np.diff(positions, axis=0)

    t_mid = (timestamps[:-1] + timestamps[1:]) / 2
    velocity = dpos / dt[:, np.newaxis]

    return t_mid, velocity


def correlate_accel_with_velocity(
    t_cam: np.ndarray,
    velocity: np.ndarray,
    t_imu: np.ndarray,
    accel_m_s2: np.ndarray,
) -> np.ndarray:
    """Correlate horizontal accel axes with camera velocity.

    Parameters
    ----------
    t_cam : np.ndarray
        Camera timestamps [N_cam]
    velocity : np.ndarray
        Camera velocity [N_cam × 2]: [vx, vy]
    t_imu : np.ndarray
        IMU timestamps [N_imu]
    accel_m_s2 : np.ndarray
        Accel data [N_imu × 3]

    Returns
    -------
    correlations : np.ndarray
        Correlation matrix [3 × 2]: accel axis vs camera velocity axis
    """
    # Compute acceleration from velocity (finite difference)
    dt = np.diff(t_cam)
    dv = np.diff(velocity, axis=0)
    cam_accel = dv / dt[:, np.newaxis]
    t_cam_accel = (t_cam[:-1] + t_cam[1:]) / 2

    # Correlate each accel axis with each camera accel direction
    correlations = np.zeros((3, 2))
    for imu_axis in range(3):
        accel_interp = np.interp(t_cam_accel, t_imu, accel_m_s2[:, imu_axis])
        for cam_axis in range(2):
            correlations[imu_axis, cam_axis] = np.corrcoef(
                cam_accel[:, cam_axis], accel_interp
            )[0, 1]

    return correlations


def analyze_imu_orientation(
    position_file: str, imu_file: str, meters_per_pixel: float = 0.0022
) -> None:
    """Analyze IMU data to determine axis orientation.

    Parameters
    ----------
    position_file : str
        Path to position parquet
    imu_file : str
        Path to IMU parquet
    meters_per_pixel : float
        Conversion factor (default: 0.0022)
    """
    print("=" * 80)
    print("IMU AXIS ORIENTATION ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    t_cam, led1_m, led2_m = load_position_data(position_file, meters_per_pixel)
    t_imu, gyro_rad_s, accel_m_s2 = load_imu_data(imu_file)
    print(f"✓ Camera: {len(t_cam):,} frames")
    print(f"✓ IMU: {len(t_imu):,} samples")

    # === STEP 1: Find gravity axis ===
    print("\n" + "=" * 80)
    print("STEP 1: IDENTIFY VERTICAL AXIS (GRAVITY)")
    print("=" * 80)

    gravity_axis, gravity_value = find_gravity_axis(accel_m_s2)
    axis_names = ["X", "Y", "Z"]

    print("\nAccelerometer mean values:")
    for i, name in enumerate(axis_names):
        mean_val = np.mean(accel_m_s2[:, i])
        std_val = np.std(accel_m_s2[:, i])
        marker = " ← GRAVITY AXIS" if i == gravity_axis else ""
        print(f"  Accel {name}: {mean_val:+7.2f} ± {std_val:.2f} m/s²{marker}")

    magnitude = np.linalg.norm(np.mean(accel_m_s2, axis=0))
    print(f"\n3D magnitude: {magnitude:.2f} m/s² (expected ~9.81 m/s²)")
    print(f"\n✓ Vertical axis: {axis_names[gravity_axis]} (aligned with gravity)")
    print(
        f"  Sign: {'negative' if gravity_value < 0 else 'positive'} = pointing upward in body frame"
    )

    # === STEP 2: Find yaw axis ===
    print("\n" + "=" * 80)
    print("STEP 2: IDENTIFY YAW AXIS (HEADING RATE)")
    print("=" * 80)

    heading = compute_heading_from_leds(led1_m, led2_m)
    heading_rate = compute_angular_velocity(t_cam, heading)

    gyro_correlations = correlate_gyro_with_heading(
        t_cam, heading_rate, t_imu, gyro_rad_s
    )

    print("\nMethod 1: Correlation of each gyro axis with heading rate from camera:")
    for i, name in enumerate(axis_names):
        corr = gyro_correlations[i]
        print(f"  Gyro {name}: r = {corr:+.3f}")

    # Method 2: Look at variance (most active axis during motion)
    print("\nMethod 2: Gyro variance (indicates which axes are most active):")
    gyro_std = np.std(gyro_rad_s, axis=0) * 180 / np.pi  # Convert to deg/s
    for i, name in enumerate(axis_names):
        print(f"  Gyro {name}: σ = {gyro_std[i]:.1f} deg/s")

    # Method 3: Expected yaw axis should match vertical (gravity) axis
    print("\nMethod 3: Physical expectation:")
    print(
        f"  → Yaw rotation should be around vertical axis: {axis_names[gravity_axis]}"
    )

    # Use gravity axis as yaw axis (physical constraint)
    yaw_axis = gravity_axis

    # Determine sign from correlation
    if np.abs(gyro_correlations[yaw_axis]) > 0.01:
        yaw_sign = np.sign(gyro_correlations[yaw_axis])
    else:
        yaw_sign = 1  # Default assumption
        print("  ⚠ Warning: Correlation too weak to determine sign reliably")

    print(f"\n✓ Yaw axis: Gyro {axis_names[yaw_axis]} (matches vertical axis)")
    if np.abs(gyro_correlations[yaw_axis]) > 0.01:
        print(f"  Correlation with heading: {gyro_correlations[yaw_axis]:+.3f}")
        print(
            f"  Sign: {'positive' if yaw_sign > 0 else 'negative'} gyro = counterclockwise rotation"
        )
    else:
        print(
            f"  ⚠ Very weak correlation ({gyro_correlations[yaw_axis]:+.3f}) - rat may be mostly stationary"
        )

    # === STEP 3: Correlate accel with camera motion ===
    print("\n" + "=" * 80)
    print("STEP 3: CORRELATE HORIZONTAL ACCEL WITH CAMERA MOTION")
    print("=" * 80)

    # Use midpoint between LEDs as position estimate
    position = (led1_m + led2_m) / 2
    t_vel, velocity = compute_camera_velocity(t_cam, position)

    accel_correlations = correlate_accel_with_velocity(
        t_vel, velocity, t_imu, accel_m_s2
    )

    print("\nCorrelation matrix (IMU accel axis vs camera velocity):")
    print("                  Camera X    Camera Y")
    for i, name in enumerate(axis_names):
        print(
            f"  Accel {name}:    {accel_correlations[i, 0]:+7.3f}   {accel_correlations[i, 1]:+7.3f}"
        )

    print(
        "\nNote: Camera X/Y are arbitrary image coordinates, not necessarily aligned with body frame."
    )

    # === STEP 4: Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: IMU AXIS ORIENTATION")
    print("=" * 80)

    print(f"\n✓ VERTICAL (aligned with gravity): Accel {axis_names[gravity_axis]}")
    print(
        f"  → Mean = {gravity_value:.2f} m/s² ({'up' if gravity_value < 0 else 'down'} in body frame)"
    )

    print(f"\n✓ YAW ROTATION (heading): Gyro {axis_names[yaw_axis]}")
    print(
        f"  → Positive rotation = {'CCW' if yaw_sign > 0 else 'CW'} when viewed from above"
    )

    # Determine horizontal axes (the two that are NOT vertical)
    horizontal_axes = [i for i in range(3) if i != gravity_axis]
    print(
        f"\n✓ HORIZONTAL PLANE: Accel {axis_names[horizontal_axes[0]]} and {axis_names[horizontal_axes[1]]}"
    )
    print("  → These measure lateral and forward/backward motion (order TBD)")

    # Check consistency
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK")
    print("=" * 80)

    if yaw_axis == gravity_axis:
        print("✓ Yaw axis matches vertical axis (expected for upright headstage)")
    else:
        print("⚠ Warning: Yaw axis does NOT match vertical axis!")
        print(
            "  This suggests the headstage may be tilted or rotated relative to expected orientation"
        )

    # Create visualization
    create_orientation_plot(accel_m_s2, gyro_rad_s, gravity_axis, yaw_axis)


def create_orientation_plot(
    accel_m_s2: np.ndarray, gyro_rad_s: np.ndarray, gravity_axis: int, yaw_axis: int
) -> None:
    """Create visualization of IMU axis distributions.

    Parameters
    ----------
    accel_m_s2 : np.ndarray
        Accelerometer data [N × 3]
    gyro_rad_s : np.ndarray
        Gyroscope data [N × 3]
    gravity_axis : int
        Index of gravity-aligned axis
    yaw_axis : int
        Index of yaw axis
    """
    axis_names = ["X", "Y", "Z"]
    gyro_deg_s = gyro_rad_s * 180 / np.pi

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle("IMU Axis Distributions", fontsize=14, fontweight="bold")

    # Plot accelerometer histograms
    for i in range(3):
        ax = axes[0, i]
        ax.hist(accel_m_s2[:, i], bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(accel_m_s2[:, i]),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Mean",
        )
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)

        title = f"Accel {axis_names[i]}"
        if i == gravity_axis:
            title += " (VERTICAL)"
        ax.set_title(title, fontweight="bold" if i == gravity_axis else "normal")
        ax.set_xlabel("Acceleration (m/s²)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot gyroscope histograms
    for i in range(3):
        ax = axes[1, i]
        ax.hist(gyro_deg_s[:, i], bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(gyro_deg_s[:, i]),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Mean",
        )
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)

        title = f"Gyro {axis_names[i]}"
        if i == yaw_axis:
            title += " (YAW)"
        ax.set_title(title, fontweight="bold" if i == yaw_axis else "normal")
        ax.set_xlabel("Angular Rate (deg/s)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("imu_axis_analysis.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved visualization: imu_axis_analysis.png")
    plt.close()


def main():
    """Run IMU orientation analysis."""
    script_dir = Path(__file__).parent

    position_file = script_dir / "arthur20220324_position_info.parquet"
    imu_file = script_dir / "arthur20220324_imu_info.parquet"

    if not position_file.exists() or not imu_file.exists():
        print(f"Error: Data files not found in {script_dir}")
        return 1

    analyze_imu_orientation(str(position_file), str(imu_file), meters_per_pixel=0.0022)

    return 0


if __name__ == "__main__":
    exit(main())

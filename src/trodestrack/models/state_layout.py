"""State dimension layouts for different tracking modes.

This module defines explicit state vector layouts to eliminate hardcoded
dimension assumptions ("magic 8s") and enable extensibility to:
- Vision-only tracking (5D)
- IMU-only tracking (8D)
- 2D camera + 3D IMU (10D)
- Full 3D tracking (16D)

Each layout explicitly specifies which indices correspond to position,
velocity, orientation, and bias states.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateLayout:
    """Explicit state dimension mapping for Kalman filters.

    Defines the structure of the state vector [x₀, x₁, ..., xₙ] by
    specifying which indices correspond to position, velocity, orientation,
    and bias states.

    This abstraction eliminates hardcoded assumptions (e.g., "n == 8")
    and enables dimension-agnostic code that works for 2D, 3D, vision-only,
    and IMU-only tracking modes.

    Attributes:
        n: Total state dimension
        pos_idx: Indices for position states (x, y) or (x, y, z)
        vel_idx: Indices for velocity states (vx, vy) or (vx, vy, vz)
        heading_idx: Index/indices for orientation
            - 2D: single index for θ
            - 3D Euler: tuple (roll, pitch, yaw)
            - 3D Quaternion: tuple (qw, qx, qy, qz)
        bias_gyro_idx: Indices for gyroscope biases (b_gz) or (b_gx, b_gy, b_gz)
        bias_accel_idx: Indices for accelerometer biases (b_ax, b_ay) or (b_ax, b_ay, b_az)

    Example:
        >>> # 2D tracking with heading and biases (8D)
        >>> layout = LAYOUT_2D_FULL
        >>> layout.n
        8
        >>> layout.pos_idx
        (0, 1)
        >>> layout.heading_idx
        4

        >>> # Vision-only tracking (5D, no biases)
        >>> layout = LAYOUT_VISION_ONLY
        >>> layout.n
        5
        >>> layout.bias_gyro_idx
        ()
    """

    n: int
    pos_idx: tuple[int, ...]
    vel_idx: tuple[int, ...]
    heading_idx: int | tuple[int, ...]
    bias_gyro_idx: tuple[int, ...]
    bias_accel_idx: tuple[int, ...]

    @property
    def has_biases(self) -> bool:
        """Check if state includes bias estimation."""
        return len(self.bias_gyro_idx) > 0 or len(self.bias_accel_idx) > 0

    @property
    def spatial_dim(self) -> int:
        """Spatial dimension (2D or 3D)."""
        return len(self.pos_idx)

    @property
    def has_heading_2d(self) -> bool:
        """Check if state uses 2D heading (single angle)."""
        return isinstance(self.heading_idx, int)

    @property
    def has_orientation_3d(self) -> bool:
        """Check if state uses 3D orientation (Euler or quaternion)."""
        return isinstance(self.heading_idx, tuple) and len(self.heading_idx) > 1


def get_heading_index(layout: "StateLayout") -> int:
    """Return the 2D heading index, raising for non-2D layouts.

    This helper centralizes the assumption that EKF/UKF in this repository
    currently operate on 2D heading states for measurement and wrapping.
    """
    if isinstance(layout.heading_idx, int):
        return layout.heading_idx
    raise NotImplementedError(
        "Heading index is not a single scalar; 3D orientation not yet supported here."
    )


# =============================================================================
# Standard Layouts
# =============================================================================

LAYOUT_2D_FULL = StateLayout(
    n=8,
    pos_idx=(0, 1),  # x, y
    vel_idx=(2, 3),  # vx, vy
    heading_idx=4,  # θ
    bias_gyro_idx=(5,),  # b_gz
    bias_accel_idx=(6, 7),  # b_ax, b_ay
)
"""Standard 2D tracking with heading and biases.

State: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]

- Position: (x, y) in meters
- Velocity: (vx, vy) in m/s
- Heading: θ in radians
- Gyro bias: b_gz in rad/s (yaw rate)
- Accel bias: (b_ax, b_ay) in m/s²

Used for: Standard sensor-fused tracking with camera + IMU
"""


LAYOUT_VISION_ONLY = StateLayout(
    n=5,
    pos_idx=(0, 1),  # x, y
    vel_idx=(2, 3),  # vx, vy
    heading_idx=4,  # θ
    bias_gyro_idx=(),  # No biases
    bias_accel_idx=(),
)
"""Vision-only tracking without IMU bias estimation.

State: [x, y, vx, vy, θ]

- Position: (x, y) from camera
- Velocity: (vx, vy) from camera deltas + constant-velocity model
- Heading: θ from LED pair geometry
- No biases (no IMU integration)

Used for: Camera-only tracking when IMU is unavailable or untrusted
"""


LAYOUT_2D_CAM_3D_IMU = StateLayout(
    n=10,
    pos_idx=(0, 1),  # x, y (from 2D camera)
    vel_idx=(2, 3, 4),  # vx, vy, vz (from 3D IMU)
    heading_idx=5,  # θ
    bias_gyro_idx=(6,),  # b_gz
    bias_accel_idx=(7, 8, 9),  # b_ax, b_ay, b_az
)
"""2D camera with 3D accelerometer (detect vertical motion).

State: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]

- Position: (x, y) from overhead camera (no z)
- Velocity: (vx, vy, vz) from 3D accel (vz for rearing detection)
- Heading: θ in radians (2D rotation)
- Gyro bias: b_gz (only yaw rate)
- Accel bias: (b_ax, b_ay, b_az) for 3D accel

Used for: Detecting vertical motion (rearing, jumping) with overhead camera

Note: vz and b_az are weakly observable (no position measurement for z)
"""


LAYOUT_3D_EULER = StateLayout(
    n=15,
    pos_idx=(0, 1, 2),  # x, y, z
    vel_idx=(3, 4, 5),  # vx, vy, vz
    heading_idx=(6, 7, 8),  # roll, pitch, yaw
    bias_gyro_idx=(9, 10, 11),  # b_gx, b_gy, b_gz
    bias_accel_idx=(12, 13, 14),  # b_ax, b_ay, b_az
)
"""Full 3D tracking with Euler angle orientation.

State: [x, y, z, vx, vy, vz, roll, pitch, yaw, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

- Position: (x, y, z) in meters
- Velocity: (vx, vy, vz) in m/s
- Orientation: (roll, pitch, yaw) in radians
- Gyro bias: (b_gx, b_gy, b_gz) in rad/s
- Accel bias: (b_ax, b_ay, b_az) in m/s²

Used for: Full 6-DOF pose tracking with 3D camera + 6-axis IMU

Warning: Euler angles suffer from gimbal lock at pitch = ±90°
Consider using quaternion representation (LAYOUT_3D_QUAT) for full 3D.
"""


LAYOUT_3D_QUAT = StateLayout(
    n=16,
    pos_idx=(0, 1, 2),  # x, y, z
    vel_idx=(3, 4, 5),  # vx, vy, vz
    heading_idx=(6, 7, 8, 9),  # qw, qx, qy, qz
    bias_gyro_idx=(10, 11, 12),  # b_gx, b_gy, b_gz
    bias_accel_idx=(13, 14, 15),  # b_ax, b_ay, b_az
)
"""Full 3D tracking with quaternion orientation (preferred for 3D).

State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

- Position: (x, y, z) in meters
- Velocity: (vx, vy, vz) in m/s
- Orientation: (qw, qx, qy, qz) unit quaternion
- Gyro bias: (b_gx, b_gy, b_gz) in rad/s
- Accel bias: (b_ax, b_ay, b_az) in m/s²

Used for: Full 6-DOF pose tracking with 3D camera + 6-axis IMU

Preferred over Euler angles:
- No gimbal lock
- Smooth dynamics
- Standard in robotics/aerospace

Note: Requires quaternion normalization constraint |q| = 1
"""


# =============================================================================
# Lookup Table
# =============================================================================

LAYOUT_REGISTRY = {
    "2d_full": LAYOUT_2D_FULL,
    "vision_only": LAYOUT_VISION_ONLY,
    "imu_only": LAYOUT_2D_FULL,  # Same as 2d_full (8D with biases)
    "2d_cam_3d_imu": LAYOUT_2D_CAM_3D_IMU,
    "3d_euler": LAYOUT_3D_EULER,
    "3d_quat": LAYOUT_3D_QUAT,
}
"""Lookup table mapping mode strings to StateLayout instances.

Use this to get the appropriate layout from a configuration string:

    >>> layout = LAYOUT_REGISTRY["vision_only"]
    >>> layout.n
    5
"""


def get_layout(mode: str) -> StateLayout:
    """Get StateLayout for a given tracking mode.

    Args:
        mode: Tracking mode string, one of:
            - "2d_full": Standard 2D tracking with IMU biases (8D)
            - "vision_only": Camera-only tracking without biases (5D)
            - "imu_only": IMU-only tracking (same as 2d_full, 8D)
            - "2d_cam_3d_imu": 2D camera + 3D IMU for vertical motion (10D)
            - "3d_euler": Full 3D with Euler angles (15D)
            - "3d_quat": Full 3D with quaternions (16D)

    Returns:
        StateLayout instance with explicit dimension mapping

    Raises:
        KeyError: If mode is not recognized

    Example:
        >>> layout = get_layout("vision_only")
        >>> layout.n
        5
        >>> layout.has_biases
        False
    """
    return LAYOUT_REGISTRY[mode]

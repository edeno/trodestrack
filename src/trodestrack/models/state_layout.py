"""State dimension layouts for different tracking modes.

This module defines explicit state vector layouts to eliminate hardcoded
dimension assumptions ("magic 8s") and enable extensibility to:
- Vision-only tracking (5D)
- IMU-only tracking (8D)
- 2D camera + 3D IMU (10D)
    - 2D camera + 6-DOF IMU orientation (14D)
    - 3D state vectors (15D Euler / 16D quaternion); the experimental
        ``extended_kalman_filter_3d`` consumes the 16D ``3d_cam_6dof_imu``
        alias.

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

    @property
    def has_quaternion_orientation(self) -> bool:
        """Check if state uses scalar-first quaternion orientation."""
        return isinstance(self.heading_idx, tuple) and len(self.heading_idx) == 4


def get_heading_index(layout: StateLayout) -> int:
    """Return the 2D heading index, raising for non-2D layouts.

    Parameters
    ----------
    layout : StateLayout
        State layout describing index mapping.

    Returns
    -------
    int
        Index of heading state θ in 2D layouts.

    Raises
    ------
    NotImplementedError
        If ``layout`` does not use a single-angle 2D heading.
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


LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION = StateLayout(
    n=14,
    pos_idx=(0, 1),  # x, y
    vel_idx=(2, 3),  # vx, vy
    heading_idx=(4, 5, 6, 7),  # qw, qx, qy, qz
    bias_gyro_idx=(8, 9, 10),  # b_gx, b_gy, b_gz
    bias_accel_idx=(11, 12, 13),  # b_ax, b_ay, b_az
)
"""2D camera with full 6-DOF IMU orientation.

State: [x, y, vx, vy, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

- Position: (x, y) from overhead camera
- Velocity: (vx, vy) from camera-derived motion / constant-velocity prediction
- Orientation: scalar-first body-to-world quaternion
- Gyro bias: (b_gx, b_gy, b_gz) in rad/s
- Accel bias: (b_ax, b_ay, b_az) in m/s²

Used for: Experimental 2D camera tracking with full IMU orientation support.

Note: Accelerometer-driven x/y translation is disabled by default in the filter
configuration for this mode. The accelerometer is retained for orientation and
future experimental translation work, not as a default position driver.
"""


LAYOUT_3D_EULER = StateLayout(
    n=15,
    pos_idx=(0, 1, 2),  # x, y, z
    vel_idx=(3, 4, 5),  # vx, vy, vz
    heading_idx=(6, 7, 8),  # roll, pitch, yaw
    bias_gyro_idx=(9, 10, 11),  # b_gx, b_gy, b_gz
    bias_accel_idx=(12, 13, 14),  # b_ax, b_ay, b_az
)
"""15D state vector for 3D pose with Euler-angle orientation.

State: [x, y, z, vx, vy, vz, roll, pitch, yaw, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

- Position: (x, y, z) in meters
- Velocity: (vx, vy, vz) in m/s
- Orientation: (roll, pitch, yaw) in radians
- Gyro bias: (b_gx, b_gy, b_gz) in rad/s
- Accel bias: (b_ax, b_ay, b_az) in m/s²

No public filter entry point currently consumes this layout (the 2D
``extended_kalman_filter`` rejects 15D states; the 3D path uses
``LAYOUT_3D_CAM_6DOF_IMU``). Provided as a registered state vector for
custom analyses or future filter work.

Warning: Euler angles suffer from gimbal lock at pitch = ±90°. Prefer
``LAYOUT_3D_QUAT`` / ``LAYOUT_3D_CAM_6DOF_IMU`` for tracking work.
"""


LAYOUT_3D_QUAT = StateLayout(
    n=16,
    pos_idx=(0, 1, 2),  # x, y, z
    vel_idx=(3, 4, 5),  # vx, vy, vz
    heading_idx=(6, 7, 8, 9),  # qw, qx, qy, qz
    bias_gyro_idx=(10, 11, 12),  # b_gx, b_gy, b_gz
    bias_accel_idx=(13, 14, 15),  # b_ax, b_ay, b_az
)
"""16D state vector for 3D pose with quaternion orientation.

State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

- Position: (x, y, z) in meters
- Velocity: (vx, vy, vz) in m/s
- Orientation: (qw, qx, qy, qz) unit quaternion
- Gyro bias: (b_gx, b_gy, b_gz) in rad/s
- Accel bias: (b_ax, b_ay, b_az) in m/s²

The UKF rejects quaternion layouts and the 2D ``extended_kalman_filter``
rejects 16D states. To run 3D tracking on this state vector, use the
separately registered ``LAYOUT_3D_CAM_6DOF_IMU`` alias with the
experimental ``extended_kalman_filter_3d`` entry point.

Preferred over Euler angles:
- No gimbal lock
- Smooth dynamics
- Standard in robotics/aerospace

Note: Requires quaternion normalization constraint |q| = 1
"""


LAYOUT_3D_CAM_6DOF_IMU = StateLayout(
    n=LAYOUT_3D_QUAT.n,
    pos_idx=LAYOUT_3D_QUAT.pos_idx,
    vel_idx=LAYOUT_3D_QUAT.vel_idx,
    heading_idx=LAYOUT_3D_QUAT.heading_idx,
    bias_gyro_idx=LAYOUT_3D_QUAT.bias_gyro_idx,
    bias_accel_idx=LAYOUT_3D_QUAT.bias_accel_idx,
)
"""Full 3D camera with 6-DOF IMU fusion.

Separate layout instance with the same 16D quaternion state structure as
``LAYOUT_3D_QUAT`` and the explicit state-mode name for 3D camera +
6-DOF IMU.
"""


# =============================================================================
# Lookup Table
# =============================================================================

LAYOUT_REGISTRY = {
    "2d_full": LAYOUT_2D_FULL,
    "vision_only": LAYOUT_VISION_ONLY,
    "imu_only": LAYOUT_2D_FULL,  # Same as 2d_full (8D with biases)
    "2d_cam_3d_imu": LAYOUT_2D_CAM_3D_IMU,
    "2d_cam_6dof_imu_orientation": LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION,
    "3d_euler": LAYOUT_3D_EULER,
    "3d_quat": LAYOUT_3D_QUAT,
    "3d_cam_6dof_imu": LAYOUT_3D_CAM_6DOF_IMU,
}
"""Lookup table mapping mode strings to StateLayout instances.

Use this to get the appropriate layout from a configuration string:

    >>> layout = LAYOUT_REGISTRY["vision_only"]
    >>> layout.n
    5
"""


def get_layout(mode: str) -> StateLayout:
    """Get ``StateLayout`` for a given tracking mode.

    Parameters
    ----------
    mode : str
        One of {"2d_full", "vision_only", "imu_only", "2d_cam_3d_imu",
        "2d_cam_6dof_imu_orientation", "3d_euler", "3d_quat",
        "3d_cam_6dof_imu"}.

    Returns
    -------
    StateLayout
        Layout instance with explicit dimension mapping.

    Raises
    ------
    KeyError
        If ``mode`` is not recognized.
    """
    return LAYOUT_REGISTRY[mode]

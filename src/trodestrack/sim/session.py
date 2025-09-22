"""Complete synthetic session generation."""

import numpy as np
from typing import Dict, Any
from .config import SimConfig
from .imu import generate_synthetic_imu
from .video import generate_synthetic_video
from ..constants import (
    SPATIAL_DIMENSIONS,
    ARENA_BOUNDARY_MARGIN_CM,
    MIN_INITIAL_SPEED_CM_S,
    MAX_INITIAL_SPEED_CM_S,
    TURN_ANGLE_STD_RAD,
    ACCELERATION_DIVISOR,
    MIN_VELOCITY_THRESHOLD,
    PI_RADIANS,
)


def generate_synthetic_session(config: SimConfig) -> Dict[str, Any]:
    """Generate a complete synthetic session with ground truth, IMU, and video data.

    Args:
        config: Simulation configuration

    Returns:
        Dictionary containing 'ground_truth', 'imu_data', and 'video_data'
    """
    # Generate ground truth trajectory
    ground_truth = _generate_ground_truth_trajectory(config)

    # Generate synthetic IMU data
    imu_data = generate_synthetic_imu(ground_truth, config)

    # Generate synthetic video data
    video_data = generate_synthetic_video(ground_truth, config)

    return {
        "ground_truth": ground_truth,
        "imu_data": imu_data,
        "video_data": video_data,
    }


def _generate_ground_truth_trajectory(config: SimConfig) -> Dict[str, np.ndarray]:
    """Generate ground truth trajectory for the synthetic session.

    Args:
        config: Simulation configuration

    Returns:
        Dictionary with 'timestamps', 'positions', 'velocities', 'headings'
    """
    # Set random seed for reproducibility
    np.random.seed(config.seed)

    # Generate timestamps at IMU rate for high resolution
    n_samples = int(config.duration * config.imu_rate)
    timestamps = np.linspace(0, config.duration, n_samples)
    dt = timestamps[1] - timestamps[0]

    # Initialize trajectory arrays
    positions = np.zeros((n_samples, SPATIAL_DIMENSIONS))  # [x, y] in cm
    velocities = np.zeros((n_samples, SPATIAL_DIMENSIONS))  # [vx, vy] in cm/s
    headings = np.zeros(n_samples)  # heading in radians

    # Start in center of arena
    arena_width, arena_height = config.arena_size
    positions[0] = [arena_width / 2, arena_height / 2]

    # Initial heading (random)
    headings[0] = np.random.uniform(0, 2 * PI_RADIANS)

    # Initial velocity (random direction, moderate speed)
    initial_speed = np.random.uniform(MIN_INITIAL_SPEED_CM_S, MAX_INITIAL_SPEED_CM_S)  # cm/s
    velocities[0] = [
        initial_speed * np.cos(headings[0]),
        initial_speed * np.sin(headings[0]),
    ]

    # Generate trajectory using random walk with physics constraints
    for i in range(1, n_samples):
        # Current state
        pos = positions[i - 1]
        vel = velocities[i - 1]
        heading = headings[i - 1]
        speed = np.linalg.norm(vel)

        # Check for boundary collisions and bounce
        margin = ARENA_BOUNDARY_MARGIN_CM  # cm from edge

        if pos[0] < margin or pos[0] > arena_width - margin:
            vel[0] = -vel[0]  # Reverse x velocity
        if pos[1] < margin or pos[1] > arena_height - margin:
            vel[1] = -vel[1]  # Reverse y velocity

        # Random direction changes
        if np.random.random() < config.trajectory.turn_probability:
            # Apply random turn
            turn_angle = np.random.normal(0, TURN_ANGLE_STD_RAD)  # radians
            new_heading = heading + turn_angle

            # Update velocity direction but keep similar speed
            velocities[i] = speed * np.array([np.cos(new_heading), np.sin(new_heading)])
        else:
            # Apply small random acceleration
            accel_magnitude = np.random.normal(
                0, config.trajectory.max_acceleration / ACCELERATION_DIVISOR
            )
            accel_direction = np.random.uniform(0, 2 * PI_RADIANS)

            accel = accel_magnitude * np.array(
                [np.cos(accel_direction), np.sin(accel_direction)]
            )

            # Update velocity with acceleration
            new_vel = vel + accel * dt

            # Limit maximum speed
            new_speed = np.linalg.norm(new_vel)
            if new_speed > config.trajectory.max_speed:
                new_vel = (config.trajectory.max_speed / new_speed) * new_vel

            velocities[i] = new_vel

        # Update position
        positions[i] = pos + velocities[i] * dt

        # Update heading from velocity
        if np.linalg.norm(velocities[i]) > MIN_VELOCITY_THRESHOLD:  # Avoid division by zero
            headings[i] = np.arctan2(velocities[i, 1], velocities[i, 0])
        else:
            headings[i] = headings[i - 1]  # Keep previous heading if not moving

        # Ensure positions stay within arena bounds
        positions[i, 0] = np.clip(positions[i, 0], margin, arena_width - margin)
        positions[i, 1] = np.clip(positions[i, 1], margin, arena_height - margin)

    return {
        "timestamps": timestamps,
        "positions": positions,
        "velocities": velocities,
        "headings": headings,
    }

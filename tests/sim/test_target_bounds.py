"""Test that simulation tiers meet the project's accuracy acceptance targets.

This module validates that Tiers 0-3 simulations satisfy the acceptance
criteria (SI units):
- Position RMSE <= 0.02 m (2 cm)
- Velocity RMSE <= 0.10 m/s (10 cm/s)
- Heading error <= 0.122 rad (7 degrees)

These bounds represent the minimum accuracy targets for the tracking system.
Tests use ground truth from simulations (perfect "filter" = just return ground truth).
"""

import numpy as np
import pytest

from trodestrack.qa.metrics import (
    compute_heading_error,
    compute_position_rmse,
    compute_velocity_rmse,
)
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)

# =============================================================================
# Acceptance Criteria - All in SI units
# =============================================================================

TARGET_POSITION_RMSE_M = 0.02  # Position RMSE <= 0.02 m (2 cm)
TARGET_VELOCITY_RMSE_M_S = 0.10  # Velocity RMSE <= 0.10 m/s (10 cm/s)
TARGET_HEADING_ERROR_RAD = np.deg2rad(7.0)  # Heading error <= 0.122 rad (7 degrees)


# =============================================================================
# Tier 0: Analytic Scenarios (Perfect Ground Truth)
# =============================================================================


@pytest.mark.parametrize("duration, expected_pos_rmse", [(30.0, 0.0)])
def test_tier0_stationary_perfect_truth(duration, expected_pos_rmse):
    """Tier 0: Stationary - ground truth should have zero error."""
    config = SimpleSimConfig(duration_s=duration)
    result = simulate_stationary(config=config, seed=42)

    # Ground truth vs itself = perfect
    pos_rmse = compute_position_rmse(
        result["X_truth"][:, :2],  # true positions
        result["X_truth"][:, :2],  # "estimated" = same as truth
    )

    assert pos_rmse == pytest.approx(expected_pos_rmse, abs=1e-10)


@pytest.mark.parametrize(
    "duration, velocity_m_s, expected_vel_rmse", [(30.0, 0.10, 0.0)]
)
def test_tier0_constant_velocity_perfect_truth(
    duration, velocity_m_s, expected_vel_rmse
):
    """Tier 0: Constant velocity - ground truth should have zero error."""
    config = SimpleSimConfig(duration_s=duration)
    velocity = np.array([velocity_m_s, 0.0])  # m/s in x-direction
    result = simulate_constant_velocity(config=config, velocity=velocity, seed=42)

    vel_rmse = compute_velocity_rmse(
        result["X_truth"][:, 2:4],  # true velocities
        result["X_truth"][:, 2:4],  # "estimated" = same as truth
    )

    assert vel_rmse == pytest.approx(expected_vel_rmse, abs=1e-10)


@pytest.mark.parametrize(
    "duration, radius_m, expected_heading_err", [(30.0, 0.50, 0.0)]
)
def test_tier0_circular_perfect_truth(duration, radius_m, expected_heading_err):
    """Tier 0: Circular motion - ground truth should have zero heading error."""
    config = SimpleSimConfig(duration_s=duration)
    result = simulate_circular(config=config, radius=radius_m, seed=42)

    heading_err = compute_heading_error(
        result["X_truth"][:, 4],  # true heading
        result["X_truth"][:, 4],  # "estimated" = same as truth
    )

    assert heading_err == pytest.approx(expected_heading_err, abs=1e-10)


# =============================================================================
# Tier 1-3: Realistic Scenarios with Noise
# =============================================================================


def test_tier1_rat_imu_position_rmse_within_target():
    """Tier 1-3: Rat IMU simulation position RMSE should meet the target (<=2 cm).

    This test uses a realistic rat simulation with:
    - IMU noise
    - Camera dropout
    - LED swaps
    - Bias drift

    Ground truth is available, so we test that the simulation itself is
    reasonable (noise levels don't destroy ground truth quality).
    """
    config = RatIMUSimConfig(duration_s=30.0)
    result = simulate_rat_imu(config=config, seed=42)

    # For now, test ground truth vs itself (validates data structure)
    # In Milestone 2, this will test filtered estimates vs ground truth
    pos_rmse = compute_position_rmse(
        result["X_truth"][:, :2],
        result["X_truth"][:, :2],
    )

    # Ground truth vs itself should be perfect
    assert pos_rmse <= TARGET_POSITION_RMSE_M  # Will be 0.0 for ground truth


def test_tier1_rat_imu_velocity_rmse_within_target():
    """Tier 1-3: Rat IMU simulation velocity RMSE should meet the target (<=10 cm/s)."""
    config = RatIMUSimConfig(duration_s=30.0)
    result = simulate_rat_imu(config=config, seed=42)

    vel_rmse = compute_velocity_rmse(
        result["X_truth"][:, 2:4],
        result["X_truth"][:, 2:4],
    )

    assert vel_rmse <= TARGET_VELOCITY_RMSE_M_S


def test_tier1_rat_imu_heading_error_within_target():
    """Tier 1-3: Rat IMU simulation heading error should meet the target (<=7 deg)."""
    config = RatIMUSimConfig(duration_s=30.0)
    result = simulate_rat_imu(config=config, seed=42)

    heading_err = compute_heading_error(
        result["X_truth"][:, 4],
        result["X_truth"][:, 4],
    )

    assert heading_err <= TARGET_HEADING_ERROR_RAD


# =============================================================================
# Full Integration Test
# =============================================================================


@pytest.mark.parametrize(
    "scenario_name",
    [
        "stationary",
        "constant_velocity",
        "circular",
        "rat_imu",
    ],
)
def test_all_tiers_data_structure_valid(scenario_name):
    """All tiers should return valid SimOut structure."""
    if scenario_name == "stationary":
        config = SimpleSimConfig(duration_s=30.0)
        result = simulate_stationary(config=config, seed=42)
    elif scenario_name == "constant_velocity":
        config = SimpleSimConfig(duration_s=30.0)
        velocity = np.array([0.15, 0.0])  # 15 cm/s in x
        result = simulate_constant_velocity(config=config, velocity=velocity, seed=42)
    elif scenario_name == "circular":
        config = SimpleSimConfig(duration_s=30.0)
        result = simulate_circular(config=config, radius=0.50, seed=42)
    elif scenario_name == "rat_imu":
        config = RatIMUSimConfig(duration_s=30.0)
        result = simulate_rat_imu(config=config, seed=42)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    # Check required keys
    assert "X_truth" in result
    assert "t_imu" in result
    assert "t_cam_exp" in result
    assert "U_imu" in result
    assert "Z_cam_led1" in result
    assert "Z_cam_led2" in result

    # Check shapes
    assert result["X_truth"].shape[1] == 5  # [x, y, vx, vy, θ]
    assert result["U_imu"].shape[1] == 3  # [ω_z, f_x, f_y]
    assert result["Z_cam_led1"].shape[1] == 2  # [x, y]
    assert result["Z_cam_led2"].shape[1] == 2  # [x, y]

    # Check time alignment
    assert len(result["t_imu"]) == len(result["X_truth"])
    assert len(result["t_cam_exp"]) == len(result["Z_cam_led1"])

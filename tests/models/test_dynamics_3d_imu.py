"""Test dynamics_function with 3D IMU inputs (gravity compensation).

This test suite validates that dynamics_function correctly handles:
    1. 3D IMU inputs [ω_z, fx, fy, fz] instead of 2D [ω_z, fx, fy]
    2. Rotation from body frame to world frame using yaw angle
    3. Gravity compensation to remove vertical acceleration component
    4. Backward compatibility with 2D IMU (no fz provided)

The 2D camera + 3D IMU mode (LAYOUT_2D_CAM_3D_IMU) estimates:
    - 2D position (x, y) from camera
    - 3D velocity (vx, vy, vz) from 3D accelerometer
    - 2D heading (θ) from gyroscope
    - 3D accel bias (b_ax, b_ay, b_az)

This enables detection of vertical motion (rearing, jumping) even though
the overhead camera only provides 2D position.

PRD Reference:
    - Section 6: Mathematical Model (2D v1) - future 3D extension
    - Milestone M5: 2D Pose + 3D IMU (Gravity-Aware)
"""

from __future__ import annotations

import jax.numpy as jnp

from trodestrack.models.filter_common import dynamics_function
from trodestrack.models.state_layout import (
    LAYOUT_2D_CAM_3D_IMU,
    LAYOUT_2D_FULL,
    LAYOUT_VISION_ONLY,
)

# =============================================================================
# Test: 3D IMU Input Shape and Gravity Compensation
# =============================================================================


def test_dynamics_function_accepts_3d_imu():
    """Test that dynamics_function accepts 3D IMU input [ω_z, fx, fy, fz]."""
    # Use 2D camera + 3D IMU layout (10D state)
    layout = LAYOUT_2D_CAM_3D_IMU
    state = jnp.zeros(layout.n)  # [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]

    # 3D IMU: [ω_z (rad/s), fx (m/s²), fy (m/s²), fz (m/s²)]
    imu_3d = jnp.array([0.0, 0.0, 0.0, 9.81])  # Stationary, reading gravity

    dt = 0.033  # ~30 Hz camera
    damping = 0.1

    # Should not crash
    next_state = dynamics_function(state, imu_3d, dt, damping, layout)

    # Output shape should match input
    assert next_state.shape == state.shape
    assert next_state.shape == (10,)


def test_dynamics_function_gravity_compensation_at_rest():
    """Test that gravity is correctly removed from vertical acceleration.

    At rest with no motion, the IMU reads [0, 0, +9.81] due to the normal
    force. After gravity compensation, the kinematic acceleration should be
    zero, so velocity should not change.
    """
    layout = LAYOUT_2D_CAM_3D_IMU

    # Initial state: at origin, zero velocity, zero biases
    state = jnp.zeros(layout.n)
    # Set position to [1.0, 2.0] to verify it propagates correctly
    state = state.at[layout.pos_idx[0]].set(1.0)  # x
    state = state.at[layout.pos_idx[1]].set(2.0)  # y

    # IMU at rest: gyro=0, accel reads gravity [0, 0, +g]
    # Note: body frame aligned with world frame (θ=0)
    g = 9.81
    imu = jnp.array([0.0, 0.0, 0.0, g])  # [ω_z, fx, fy, fz]

    dt = 0.1
    damping = 0.0  # No damping for this test

    next_state = dynamics_function(state, imu, dt, damping, layout)

    # Velocities should remain ~zero (gravity compensated)
    vx_next = next_state[layout.vel_idx[0]]
    vy_next = next_state[layout.vel_idx[1]]
    vz_next = next_state[layout.vel_idx[2]]

    # Allow small numerical error
    assert jnp.abs(vx_next) < 1e-6, f"vx should be ~0, got {vx_next}"
    assert jnp.abs(vy_next) < 1e-6, f"vy should be ~0, got {vy_next}"
    assert jnp.abs(vz_next) < 1e-6, f"vz should be ~0, got {vz_next}"

    # Position should remain constant (v=0 throughout)
    x_next = next_state[layout.pos_idx[0]]
    y_next = next_state[layout.pos_idx[1]]
    assert jnp.allclose(x_next, 1.0, atol=1e-6)
    assert jnp.allclose(y_next, 2.0, atol=1e-6)


def test_dynamics_function_vertical_acceleration():
    """Test that vertical acceleration (after gravity compensation) integrates correctly.

    If the rat is accelerating upward (e.g., jumping), the IMU will read
    a_z > g. After gravity compensation, this should increase vz.
    """
    layout = LAYOUT_2D_CAM_3D_IMU

    # Initial state: zero velocity
    state = jnp.zeros(layout.n)

    # IMU: stationary in x-y, accelerating upward at 2 m/s² above gravity
    # Raw IMU reads: [0, 0, 0, 9.81 + 2.0] = [0, 0, 0, 11.81]
    imu = jnp.array([0.0, 0.0, 0.0, 11.81])

    dt = 0.1
    damping = 0.0

    next_state = dynamics_function(state, imu, dt, damping, layout)

    # After gravity compensation: a_z = 11.81 - 9.81 = 2.0 m/s²
    # Expected vz = 0 + 2.0 * 0.1 = 0.2 m/s
    vz_next = next_state[layout.vel_idx[2]]
    expected_vz = 2.0 * dt

    assert jnp.allclose(
        vz_next, expected_vz, atol=1e-6
    ), f"Expected vz={expected_vz}, got {vz_next}"

    # Horizontal velocities should remain zero
    vx_next = next_state[layout.vel_idx[0]]
    vy_next = next_state[layout.vel_idx[1]]
    assert jnp.abs(vx_next) < 1e-6
    assert jnp.abs(vy_next) < 1e-6


def test_dynamics_function_body_to_world_rotation():
    """Test that acceleration is correctly rotated from body to world frame.

    If the rat is oriented at θ=90° (facing +y) and the IMU measures forward
    acceleration (body-frame +x), this should become world-frame +y acceleration.
    """
    layout = LAYOUT_2D_CAM_3D_IMU

    # Initial state: oriented at θ=90° (π/2), zero velocity
    state = jnp.zeros(layout.n)
    theta = jnp.pi / 2  # 90 degrees, facing +y direction
    h_idx = layout.heading_idx
    state = state.at[h_idx].set(theta)

    # IMU: accelerating forward in body frame (+x body)
    # Body frame: [ax=1.0, ay=0, az=9.81] (at rest vertically)
    imu = jnp.array([0.0, 1.0, 0.0, 9.81])  # [ω_z, fx, fy, fz]

    dt = 0.1
    damping = 0.0

    next_state = dynamics_function(state, imu, dt, damping, layout)

    # After rotation by θ=90°:
    # World frame: [ax_w ≈ 0, ay_w ≈ 1.0, az_w ≈ 0] (after gravity compensation)
    #
    # R_z(90°) @ [1, 0, 0]ᵀ = [cos(90°), sin(90°), 0]ᵀ = [0, 1, 0]ᵀ
    #
    # Expected velocities:
    # vx = 0 + 0 * dt ≈ 0
    # vy = 0 + 1.0 * dt = 0.1
    # vz = 0 + 0 * dt ≈ 0
    vx_next = next_state[layout.vel_idx[0]]
    vy_next = next_state[layout.vel_idx[1]]
    vz_next = next_state[layout.vel_idx[2]]

    assert jnp.abs(vx_next) < 1e-6, f"vx should be ~0, got {vx_next}"
    assert jnp.allclose(vy_next, 1.0 * dt, atol=1e-6), f"vy should be ~{1.0 * dt}, got {vy_next}"
    assert jnp.abs(vz_next) < 1e-6, f"vz should be ~0, got {vz_next}"


def test_dynamics_function_3d_bias_correction():
    """Test that 3D accelerometer biases are correctly subtracted."""
    layout = LAYOUT_2D_CAM_3D_IMU

    # Initial state with biases
    state = jnp.zeros(layout.n)
    # Set biases: b_ax=0.1, b_ay=0.2, b_az=0.3 m/s²
    state = state.at[layout.bias_accel_idx[0]].set(0.1)  # b_ax
    state = state.at[layout.bias_accel_idx[1]].set(0.2)  # b_ay
    state = state.at[layout.bias_accel_idx[2]].set(0.3)  # b_az

    # IMU: raw readings include biases + gravity
    # Raw: [ω_z=0, fx=1.1, fy=1.2, fz=10.11]
    # True accel (body): [1.0, 1.0, 9.81] (after bias removal)
    imu = jnp.array([0.0, 1.1, 1.2, 10.11])

    dt = 0.1
    damping = 0.0

    next_state = dynamics_function(state, imu, dt, damping, layout)

    # After bias removal: [fx-b_ax, fy-b_ay, fz-b_az] = [1.0, 1.0, 9.81]
    # After gravity compensation: [1.0, 1.0, 0.0]
    # After rotation (θ=0): [1.0, 1.0, 0.0]
    # Expected velocities:
    # vx = 0 + 1.0 * 0.1 = 0.1
    # vy = 0 + 1.0 * 0.1 = 0.1
    # vz = 0 + 0.0 * 0.1 = 0.0

    vx_next = next_state[layout.vel_idx[0]]
    vy_next = next_state[layout.vel_idx[1]]
    vz_next = next_state[layout.vel_idx[2]]

    assert jnp.allclose(vx_next, 0.1, atol=1e-6), f"Expected vx=0.1, got {vx_next}"
    assert jnp.allclose(vy_next, 0.1, atol=1e-6), f"Expected vy=0.1, got {vy_next}"
    assert jnp.allclose(vz_next, 0.0, atol=1e-6), f"Expected vz=0.0, got {vz_next}"


# =============================================================================
# Test: Backward Compatibility with 2D IMU
# =============================================================================


def test_dynamics_function_2d_imu_backward_compatibility():
    """Test that 2D IMU mode (LAYOUT_2D_FULL) still works with [ω_z, fx, fy].

    This ensures we don't break existing functionality when adding 3D support.
    The function should detect 2D layouts and handle 3-element IMU vectors.
    """
    layout = LAYOUT_2D_FULL  # 8D: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]

    state = jnp.zeros(layout.n)

    # 2D IMU: [ω_z, fx, fy] (no fz)
    imu_2d = jnp.array([0.0, 1.0, 0.0])

    dt = 0.1
    damping = 0.0

    # Should not crash
    next_state = dynamics_function(state, imu_2d, dt, damping, layout)

    # Check output shape
    assert next_state.shape == (8,)

    # Check that horizontal acceleration integrates correctly
    # (no gravity compensation needed in 2D mode)
    vx_next = next_state[layout.vel_idx[0]]
    expected_vx = 1.0 * dt  # ax=1.0 m/s²

    assert jnp.allclose(
        vx_next, expected_vx, atol=1e-6
    ), f"Expected vx={expected_vx}, got {vx_next}"


def test_dynamics_function_vision_only_mode():
    """Test that vision-only mode (LAYOUT_VISION_ONLY) works without IMU.

    Vision-only mode uses a simplified dynamics model without IMU integration.
    This test ensures that the dynamics function handles this layout correctly.
    """
    layout = LAYOUT_VISION_ONLY  # 5D: [x, y, vx, vy, θ]

    state = jnp.zeros(layout.n)
    state = state.at[layout.vel_idx[0]].set(1.0)  # vx = 1.0 m/s
    state = state.at[layout.vel_idx[1]].set(0.5)  # vy = 0.5 m/s

    # Minimal IMU (zeros, should not be used in vision-only mode)
    imu = jnp.array([0.0, 0.0, 0.0])

    dt = 0.1
    damping = 0.1  # Apply damping

    next_state = dynamics_function(state, imu, dt, damping, layout)

    # Check output shape
    assert next_state.shape == (5,)

    # Verify no NaNs or Infs (basic sanity check)
    assert jnp.all(jnp.isfinite(next_state)), "Vision-only mode should produce finite state values"


# =============================================================================
# Test: Edge Cases and Robustness
# =============================================================================


def test_dynamics_function_handles_large_rotation():
    """Test that rotation works correctly for arbitrary heading angles.

    The rotation matrix should handle angles outside [0, 2π] and wrap correctly.
    """
    layout = LAYOUT_2D_CAM_3D_IMU

    # Initial state: large heading angle (e.g., 10π)
    state = jnp.zeros(layout.n)
    state = state.at[layout.heading_idx].set(10.0 * jnp.pi)

    # IMU: forward acceleration in body frame
    imu = jnp.array([0.0, 1.0, 0.0, 9.81])

    dt = 0.1
    damping = 0.0

    # Should not crash; rotation should handle large angles via cos/sin
    next_state = dynamics_function(state, imu, dt, damping, layout)

    # Verify output is finite (no NaN or Inf)
    assert jnp.all(jnp.isfinite(next_state))


def test_dynamics_function_gyro_bias_integration():
    """Test that gyro bias is correctly subtracted from angular velocity.

    The heading update should use (ω_z - b_gz), not raw ω_z.
    """
    layout = LAYOUT_2D_CAM_3D_IMU

    # Initial state: θ=0, b_gz=0.1 rad/s
    state = jnp.zeros(layout.n)
    state = state.at[layout.bias_gyro_idx[0]].set(0.1)  # b_gz

    # IMU: raw gyro reads 0.5 rad/s
    # True angular velocity: ω_z - b_gz = 0.5 - 0.1 = 0.4 rad/s
    imu = jnp.array([0.5, 0.0, 0.0, 9.81])

    dt = 0.1
    damping = 0.0

    next_state = dynamics_function(state, imu, dt, damping, layout)

    # Expected heading: θ = 0 + 0.4 * 0.1 = 0.04 rad
    theta_next = next_state[layout.heading_idx]
    expected_theta = 0.4 * dt

    assert jnp.allclose(
        theta_next, expected_theta, atol=1e-6
    ), f"Expected θ={expected_theta}, got {theta_next}"


def test_dynamics_function_position_velocity_coupling():
    """Test that position updates correctly include both velocity and acceleration terms.

    The position update should be:
    p_{k+1} = p_k + v_k * dt + 0.5 * a_world * dt²
    """
    layout = LAYOUT_2D_CAM_3D_IMU

    # Initial state: p=[0, 0], v=[1, 2, 0] m/s
    state = jnp.zeros(layout.n)
    state = state.at[layout.vel_idx[0]].set(1.0)  # vx
    state = state.at[layout.vel_idx[1]].set(2.0)  # vy

    # IMU: constant acceleration [ax=0.5, ay=1.0, az=0] (after gravity comp)
    # Raw IMU: [0, 0.5, 1.0, 9.81]
    imu = jnp.array([0.0, 0.5, 1.0, 9.81])

    dt = 0.1
    damping = 0.0

    next_state = dynamics_function(state, imu, dt, damping, layout)

    # Expected position:
    # x = 0 + 1.0*0.1 + 0.5*0.5*0.1² = 0.1 + 0.0025 = 0.1025
    # y = 0 + 2.0*0.1 + 0.5*1.0*0.1² = 0.2 + 0.005 = 0.205

    x_next = next_state[layout.pos_idx[0]]
    y_next = next_state[layout.pos_idx[1]]

    expected_x = 1.0 * dt + 0.5 * 0.5 * dt**2
    expected_y = 2.0 * dt + 0.5 * 1.0 * dt**2

    assert jnp.allclose(x_next, expected_x, atol=1e-6), f"Expected x={expected_x}, got {x_next}"
    assert jnp.allclose(y_next, expected_y, atol=1e-6), f"Expected y={expected_y}, got {y_next}"


# =============================================================================
# Test: Numerical Stability
# =============================================================================


def test_dynamics_function_no_nans_with_extreme_inputs():
    """Test that dynamics function doesn't produce NaNs with extreme but valid inputs."""
    layout = LAYOUT_2D_CAM_3D_IMU

    # Extreme but physically plausible state
    state = jnp.array(
        [
            100.0,  # x (far from origin)
            -50.0,  # y
            10.0,  # vx (fast)
            -5.0,  # vy
            2.0,  # vz
            3.0,  # θ (arbitrary angle)
            0.01,  # b_gz
            0.1,  # b_ax
            -0.1,  # b_ay
            0.05,  # b_az
        ]
    )

    # Extreme but valid IMU
    imu = jnp.array([5.0, 20.0, -10.0, 20.0])  # High angular velocity and accel

    dt = 0.01  # Small time step
    damping = 0.5

    next_state = dynamics_function(state, imu, dt, damping, layout)

    # Verify no NaNs or Infs
    assert jnp.all(
        jnp.isfinite(next_state)
    ), "dynamics_function produced NaN or Inf with extreme inputs"


def test_dynamics_function_deterministic():
    """Test that dynamics function produces identical outputs for identical inputs."""
    layout = LAYOUT_2D_CAM_3D_IMU

    state = jnp.ones(layout.n) * 0.5
    imu = jnp.array([0.1, 0.2, 0.3, 9.81])
    dt = 0.033
    damping = 0.1

    # Run twice
    result1 = dynamics_function(state, imu, dt, damping, layout)
    result2 = dynamics_function(state, imu, dt, damping, layout)

    # Should be bit-for-bit identical (deterministic)
    assert jnp.array_equal(result1, result2), "dynamics_function should be deterministic"

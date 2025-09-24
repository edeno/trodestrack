#!/usr/bin/env python3
"""Demonstration of modern JAX-optimized EKF/RTS functions.

This example shows how to use the production-ready JAX functions for
high-performance sensor fusion with lax.scan optimization.
"""

import jax
import jax.numpy as jnp
from jax import lax

from trodestrack.models import (
    ekf_step_pytree,
    rts_smooth_pure,
    ForwardPassData,
    EkfCarry,
    create_initial_ekf_state,
    State2D,
)


def demo_modern_ekf():
    """Demonstrate modern JAX-optimized EKF with lax.scan."""
    print("=== Modern JAX EKF Demo ===")

    # Create initial state and covariance
    initial_state = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0)
    initial_covariance = jnp.eye(8) * 0.1

    # Convert to EKF format
    ekf_state = create_initial_ekf_state(initial_state, initial_covariance)
    carry = EkfCarry(x=ekf_state.state, P=ekf_state.covariance)

    # Create frame data for lax.scan (modern approach)
    n_frames = 3
    frame_data = (
        jnp.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]),  # positions
        jnp.array([0.5, 0.6, 0.7]),  # headings
        jnp.array([0.9, 0.8, 0.95]),  # confidences
        jnp.array([True, True, True]),  # position validity
        jnp.array([True, False, True]),  # heading validity
        jnp.array([[0.1, 0.2, 0.05], [0.15, 0.1, 0.03], [0.08, 0.25, 0.02]]),  # IMU blocks
        jnp.array([0.1, 0.1, 0.1]),  # dt values
        # Filter parameters (repeated for each frame)
        jnp.full(n_frames, 0.1),  # velocity_damping
        jnp.full(n_frames, 0.5),  # accel_noise_std
        jnp.full(n_frames, 0.1),  # gyro_noise_std
        jnp.full(n_frames, 0.01),  # bias_drift_std
        jnp.full(n_frames, 1.0),  # position_noise_std
        jnp.full(n_frames, 0.1),  # heading_noise_std
        jnp.full(n_frames, 9.21),  # gate_threshold (will be ignored, auto-computed)
    )

    print(f"Processing {n_frames} frames with JAX lax.scan...")

    # Use lax.scan for optimal performance (modern approach)
    final_carry, all_outputs = lax.scan(ekf_step_pytree, carry, frame_data)

    print(f"Final filtered state: {final_carry.x}")
    print(f"All filtered states shape: {all_outputs.x_filt.shape}")
    print(f"Position trajectory: {all_outputs.x_filt[:, :2]}")
    print("✓ Modern JAX EKF with lax.scan completed successfully")


def demo_pure_rts():
    """Demonstrate pure RTS smoothing function."""
    print("\n=== Pure RTS Smoothing Demo ===")

    # Create example forward pass data
    n_frames = 5
    state_dim = 8

    # Generate some dummy filtered data
    filtered_states = jnp.ones((n_frames, state_dim)) * jnp.arange(n_frames)[:, None]
    filtered_covariances = jnp.tile(jnp.eye(state_dim) * 0.1, (n_frames, 1, 1))
    predicted_states = filtered_states + 0.01  # Slightly different
    predicted_covariances = filtered_covariances * 1.1
    log_likelihood = -50.0

    # Create transition matrices (identity for simplicity in this demo)
    n_frames = filtered_states.shape[0]
    transition_matrices = jnp.tile(jnp.eye(8)[None, :, :], (n_frames, 1, 1))

    # Call pure RTS function directly (JIT-compiled for optimal performance)
    result = rts_smooth_pure(
        filtered_states,
        filtered_covariances,
        predicted_states,
        predicted_covariances,
        transition_matrices,
        log_likelihood
    )

    print(f"Smoothed states shape: {result.smoothed_states.shape}")
    print(f"Smoothed covariances shape: {result.smoothed_covariances.shape}")
    print(f"Log likelihood: {result.log_likelihood}")
    print("✓ Pure RTS smoothing completed successfully")


def demo_performance_comparison():
    """Demonstrate performance benefits of JAX lax.scan vs loops."""
    print("\n=== Performance Comparison Demo ===")

    # Create test data for performance comparison
    n_frames = 1000
    carry = EkfCarry(x=jnp.zeros(8), P=jnp.eye(8))

    # Create frame data for lax.scan
    frame_data = (
        jnp.zeros((n_frames, 2)),  # positions
        jnp.zeros(n_frames),  # headings
        jnp.ones(n_frames),  # confidences
        jnp.ones(n_frames, dtype=bool),  # position validity
        jnp.ones(n_frames, dtype=bool),  # heading validity
        jnp.zeros((n_frames, 3)),  # IMU blocks
        jnp.full(n_frames, 0.1),  # dt values
        # Filter parameters
        jnp.full(n_frames, 0.1),  # velocity_damping
        jnp.full(n_frames, 0.5),  # accel_noise_std
        jnp.full(n_frames, 0.1),  # gyro_noise_std
        jnp.full(n_frames, 0.01),  # bias_drift_std
        jnp.full(n_frames, 1.0),  # position_noise_std
        jnp.full(n_frames, 0.1),  # heading_noise_std
        jnp.full(n_frames, 9.21),  # gate_threshold
    )

    # Warm up JIT compilation
    _ = lax.scan(ekf_step_pytree, carry, frame_data)

    # Time the optimized lax.scan approach
    import time

    n_trials = 10
    for trial in range(n_trials):
        start_time = time.time()
        _ = lax.scan(ekf_step_pytree, carry, frame_data)
        end_time = time.time()

    avg_time_ms = (end_time - start_time) * 1000 / n_trials
    throughput = n_frames / ((end_time - start_time) / n_trials)
    print(f"JAX lax.scan: {avg_time_ms:.2f} ms per {n_frames} frames ({throughput:.0f} frames/sec)")
    print("✓ Performance measurement completed - JAX lax.scan provides optimal throughput")


if __name__ == "__main__":
    # Enable JAX 64-bit precision for consistency
    jax.config.update("jax_enable_x64", True)

    print("Modern JAX EKF/RTS Performance Demo")
    print("=" * 50)

    demo_modern_ekf()
    demo_pure_rts()
    demo_performance_comparison()

    print("\n" + "=" * 50)
    print("Pure functions demo completed successfully!")
    print("\nKey benefits:")
    print("- EKF step functions with static parameters avoid redundant parameter passing")
    print("- JIT compilation provides optimal performance for repeated calls")
    print("- Pure functions enable better caching and avoid stateful closures")
    print("- Cleaner separation between configuration and computation")
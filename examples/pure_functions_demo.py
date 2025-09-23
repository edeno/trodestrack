#!/usr/bin/env python3
"""Demonstration of pure EKF/RTS functions for optimal JAX performance.

This example shows how to use the new pure, JIT-compiled versions of the EKF
and RTS functions for maximum performance when calling repeatedly with
different data but same parameters.
"""

import jax
import jax.numpy as jnp

from trodestrack.models import (
    create_ekf_step_arrays_optimized,
    rts_smooth_pure,
    ForwardPassData,
    EkfCarry,
)


def demo_optimized_ekf():
    """Demonstrate optimized EKF step function with static parameters."""
    print("=== Optimized EKF Step Function Demo ===")

    # Create optimized EKF step function with static filter parameters
    # This creates a JIT-compiled function with baked-in parameters for optimal performance
    ekf_step_optimized = create_ekf_step_arrays_optimized(
        velocity_damping=0.1,
        accel_noise_std=0.5,
        gyro_noise_std=0.1,
        bias_drift_std=0.01,
        position_noise_std=1.0,
        heading_noise_std=0.1,
        gate_threshold=9.21,
    )

    # Example usage with simplified input (no repeated filter parameters)
    initial_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_covariance = jnp.eye(8) * 0.1
    carry = EkfCarry(x=initial_state, P=initial_covariance)

    # Simplified input tuple (no filter parameters needed)
    inp = (
        jnp.array([1.0, 2.0]),  # position
        0.1,  # dt
        jnp.array([0.1, 0.2, 0.05]),  # imu_block [ax, ay, gz]
        0.5,  # heading
        1.0,  # confidence
        True,  # pos_mask
        True,  # head_mask
    )

    # Call the optimized function - this is JIT-compiled and very fast
    new_carry, outputs = ekf_step_optimized(carry, inp)

    print(f"Filtered state: {outputs.x_filt}")
    print(f"Predicted state: {outputs.x_pred}")
    print("✓ Optimized EKF step completed successfully")


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

    # Call pure RTS function directly (JIT-compiled for optimal performance)
    result = rts_smooth_pure(
        filtered_states,
        filtered_covariances,
        predicted_states,
        predicted_covariances,
        log_likelihood
    )

    print(f"Smoothed states shape: {result.smoothed_states.shape}")
    print(f"Smoothed covariances shape: {result.smoothed_covariances.shape}")
    print(f"Log likelihood: {result.log_likelihood}")
    print("✓ Pure RTS smoothing completed successfully")


def demo_performance_comparison():
    """Demonstrate performance benefits of pure functions."""
    print("\n=== Performance Comparison Demo ===")

    # Create optimized EKF function
    ekf_step_optimized = create_ekf_step_arrays_optimized(
        velocity_damping=0.1,
        accel_noise_std=0.5,
        gyro_noise_std=0.1,
        bias_drift_std=0.01,
        position_noise_std=1.0,
        heading_noise_std=0.1,
        gate_threshold=9.21,
    )

    # Warm up JIT compilation
    carry = EkfCarry(x=jnp.zeros(8), P=jnp.eye(8))
    inp = (jnp.array([0.0, 0.0]), 0.1, jnp.zeros(3), 0.0, 1.0, True, False)
    _ = ekf_step_optimized(carry, inp)

    # Time the optimized function
    import time
    n_calls = 1000

    start_time = time.time()
    for _ in range(n_calls):
        _ = ekf_step_optimized(carry, inp)
    end_time = time.time()

    avg_time_us = (end_time - start_time) / n_calls * 1e6
    print(f"Optimized EKF step: {avg_time_us:.2f} μs per call (average over {n_calls} calls)")
    print("✓ Performance measurement completed")


if __name__ == "__main__":
    # Enable JAX 64-bit precision for consistency
    jax.config.update("jax_enable_x64", True)

    print("JAX Pure Functions Performance Demo")
    print("=" * 50)

    demo_optimized_ekf()
    demo_pure_rts()
    demo_performance_comparison()

    print("\n" + "=" * 50)
    print("Pure functions demo completed successfully!")
    print("\nKey benefits:")
    print("- EKF step functions with static parameters avoid redundant parameter passing")
    print("- JIT compilation provides optimal performance for repeated calls")
    print("- Pure functions enable better caching and avoid stateful closures")
    print("- Cleaner separation between configuration and computation")
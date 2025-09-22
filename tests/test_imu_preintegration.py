"""Tests for IMU pre-integration functionality."""

import jax

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from trodestrack.constants import DEGREES_TO_RADIANS, STANDARD_GRAVITY_MS2
from trodestrack.imu.preintegration import (
    IMUPreintegrationResult,
    PreintegrationState,
    convert_spikegadgets_to_preintegration_units,
    preintegrate_between_frames,
    preintegrate_imu_scan,
    preintegration_step,
    rotation_matrix_2d,
)


class TestRotationMatrix2D:
    """Tests for 2D rotation matrix computation."""

    def test_rotation_matrix_identity(self):
        """Test that zero rotation gives identity matrix."""
        R = rotation_matrix_2d(0.0)
        expected = jnp.eye(2, dtype=jnp.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_rotation_matrix_90_degrees(self):
        """Test 90-degree rotation matrix."""
        R = rotation_matrix_2d(jnp.pi / 2)
        expected = jnp.array([[0, -1], [1, 0]], dtype=jnp.float64)
        np.testing.assert_allclose(R, expected, atol=1e-8)

    def test_rotation_matrix_180_degrees(self):
        """Test 180-degree rotation matrix."""
        R = rotation_matrix_2d(jnp.pi)
        expected = jnp.array([[-1, 0], [0, -1]], dtype=jnp.float64)
        np.testing.assert_allclose(R, expected, atol=1e-8)

    def test_rotation_matrix_orthogonal(self):
        """Test that rotation matrix is orthogonal."""
        angle = 0.7  # arbitrary angle
        R = rotation_matrix_2d(angle)

        # R @ R.T should be identity
        identity = R @ R.T
        np.testing.assert_allclose(identity, jnp.eye(2), atol=1e-10)

        # Determinant should be 1
        det = jnp.linalg.det(R)
        np.testing.assert_allclose(det, 1.0, atol=1e-10)

    @given(st.floats(-2 * np.pi, 2 * np.pi, allow_nan=False, allow_infinity=False))
    def test_rotation_matrix_properties(self, angle):
        """Property test for rotation matrix properties."""
        R = rotation_matrix_2d(angle)

        # Should be 2x2
        assert R.shape == (2, 2)

        # Should be orthogonal (R @ R.T = I)
        identity = R @ R.T
        np.testing.assert_allclose(identity, jnp.eye(2), atol=1e-10)

        # Determinant should be 1
        det = jnp.linalg.det(R)
        np.testing.assert_allclose(det, 1.0, atol=1e-10)


class TestPreintegrationStep:
    """Tests for single step pre-integration."""

    def test_preintegration_step_zero_motion(self):
        """Test step with zero IMU measurements."""
        initial_state = PreintegrationState(
            position=jnp.zeros(2), velocity=jnp.zeros(2), heading=0.0, time=0.0
        )

        # Zero IMU sample
        imu_sample = jnp.zeros(6)
        dt = 0.01

        new_state = preintegration_step(
            initial_state,
            imu_sample,
            gyro_bias=0.0,
            accel_bias=jnp.zeros(2),
            damping_lambda=0.0,
            dt=dt,
        )

        # Should remain at origin with zero motion
        np.testing.assert_allclose(new_state.position, jnp.zeros(2), atol=1e-8)
        np.testing.assert_allclose(new_state.velocity, jnp.zeros(2), atol=1e-8)
        assert new_state.heading == 0.0
        assert new_state.time == dt

    def test_preintegration_step_constant_acceleration(self):
        """Test step with constant acceleration."""
        initial_state = PreintegrationState(
            position=jnp.zeros(2), velocity=jnp.zeros(2), heading=0.0, time=0.0
        )

        # Constant acceleration in x direction
        accel_x = 1.0  # m/s²
        imu_sample = jnp.array([accel_x, 0.0, 0.0, 0.0, 0.0, 0.0])
        dt = 0.1

        new_state = preintegration_step(
            initial_state,
            imu_sample,
            gyro_bias=0.0,
            accel_bias=jnp.zeros(2),
            damping_lambda=0.0,
            dt=dt,
        )

        # Expected: v = a*t, x = 0.5*a*t²
        expected_velocity = jnp.array([accel_x * dt, 0.0])  # m/s
        expected_position = jnp.array([0.5 * accel_x * dt**2, 0.0])  # m

        np.testing.assert_allclose(new_state.velocity, expected_velocity, atol=1e-8)
        np.testing.assert_allclose(new_state.position, expected_position, atol=1e-8)

    def test_preintegration_step_constant_rotation(self):
        """Test step with constant angular velocity."""
        initial_state = PreintegrationState(
            position=jnp.zeros(2), velocity=jnp.zeros(2), heading=0.0, time=0.0
        )

        # Constant angular velocity
        omega_z = 1.0  # rad/s
        imu_sample = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, omega_z])
        dt = 0.1

        new_state = preintegration_step(
            initial_state,
            imu_sample,
            gyro_bias=0.0,
            accel_bias=jnp.zeros(2),
            damping_lambda=0.0,
            dt=dt,
        )

        # Expected heading change
        expected_heading = omega_z * dt
        np.testing.assert_allclose(new_state.heading, expected_heading, atol=1e-8)

    def test_preintegration_step_with_bias(self):
        """Test step with gyroscope and accelerometer bias."""
        initial_state = PreintegrationState(
            position=jnp.zeros(2), velocity=jnp.zeros(2), heading=0.0, time=0.0
        )

        # IMU measurements with bias
        gyro_bias = 0.1  # rad/s
        accel_bias = jnp.array([0.2, -0.1])  # m/s²

        omega_z_raw = 1.0 + gyro_bias  # Raw measurement includes bias
        accel_raw = jnp.array([0.5, 0.3]) + accel_bias  # Raw includes bias

        imu_sample = jnp.array([accel_raw[0], accel_raw[1], 0.0, 0.0, 0.0, omega_z_raw])
        dt = 0.1

        new_state = preintegration_step(
            initial_state,
            imu_sample,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            damping_lambda=0.0,
            dt=dt,
        )

        # After bias correction, should get original values
        expected_omega = 1.0
        expected_accel = jnp.array([0.5, 0.3])

        expected_heading = expected_omega * dt

        # For velocity, need to account for rotation during the step
        # Use mid-point heading for rotation (as implemented in the function)
        mid_heading = 0.5 * expected_omega * dt
        cos_mid = jnp.cos(mid_heading)
        sin_mid = jnp.sin(mid_heading)
        expected_velocity = (
            jnp.array(
                [
                    cos_mid * expected_accel[0] - sin_mid * expected_accel[1],
                    sin_mid * expected_accel[0] + cos_mid * expected_accel[1],
                ]
            )
            * dt
        )

        np.testing.assert_allclose(new_state.heading, expected_heading, atol=1e-8)
        np.testing.assert_allclose(new_state.velocity, expected_velocity, atol=1e-8)

    def test_preintegration_step_with_damping(self):
        """Test step with velocity damping."""
        initial_velocity = jnp.array([1.0, 0.5]) / 100  # Convert to m/s
        initial_state = PreintegrationState(
            position=jnp.zeros(2), velocity=initial_velocity, heading=0.0, time=0.0
        )

        # Zero acceleration, but with damping
        imu_sample = jnp.zeros(6)
        damping_lambda = 2.0  # 1/s
        dt = 0.1

        new_state = preintegration_step(
            initial_state,
            imu_sample,
            gyro_bias=0.0,
            accel_bias=jnp.zeros(2),
            damping_lambda=damping_lambda,
            dt=dt,
        )

        # Velocity should decay: v(t) = v0 * exp(-λt) ≈ v0 * (1 - λt) for small t
        expected_velocity_factor = 1.0 - damping_lambda * dt
        expected_velocity = initial_velocity * expected_velocity_factor

        np.testing.assert_allclose(new_state.velocity, expected_velocity, atol=1e-6)


class TestPreintegrationScan:
    """Tests for full pre-integration using scan."""

    def test_preintegrate_empty_data(self):
        """Test pre-integration with empty data."""
        imu_data = jnp.zeros((0, 6))
        timestamps = jnp.zeros(0)

        result = preintegrate_imu_scan(imu_data, timestamps)

        assert isinstance(result, IMUPreintegrationResult)
        np.testing.assert_allclose(result.delta_position, jnp.zeros(2))
        np.testing.assert_allclose(result.delta_velocity, jnp.zeros(2))
        assert result.delta_heading == 0.0
        assert result.dt == 0.0
        assert result.n_samples == 0

    def test_preintegrate_single_sample(self):
        """Test pre-integration with single sample."""
        imu_data = jnp.zeros((1, 6))
        timestamps = jnp.array([0.0])

        result = preintegrate_imu_scan(imu_data, timestamps)

        assert isinstance(result, IMUPreintegrationResult)
        np.testing.assert_allclose(result.delta_position, jnp.zeros(2))
        np.testing.assert_allclose(result.delta_velocity, jnp.zeros(2))
        assert result.delta_heading == 0.0
        assert result.dt == 0.0
        assert result.n_samples == 1

    def test_preintegrate_constant_acceleration(self):
        """Test pre-integration with constant acceleration."""
        # Create synthetic data with constant acceleration
        dt = 0.01
        duration = 1.0
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Constant acceleration in x direction
        accel_x = 2.0  # m/s²
        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(accel_x)  # ax

        result = preintegrate_imu_scan(imu_data, timestamps)

        # Expected kinematics: v = at, x = 0.5*a*t²
        expected_velocity_x = accel_x * duration * 100  # Convert to cm/s
        expected_position_x = 0.5 * accel_x * duration**2 * 100  # Convert to cm

        # Allow for slight differences due to discrete time steps and scan implementation
        np.testing.assert_allclose(result.delta_velocity[0], expected_velocity_x, rtol=2e-2)
        np.testing.assert_allclose(result.delta_position[0], expected_position_x, rtol=2e-2)
        np.testing.assert_allclose(result.delta_velocity[1], 0.0, atol=1e-8)
        np.testing.assert_allclose(result.delta_position[1], 0.0, atol=1e-8)

    def test_preintegrate_pure_rotation(self):
        """Test pre-integration with pure rotation."""
        # Create data with constant angular velocity
        dt = 0.01
        duration = 1.0
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Constant angular velocity
        omega_z = 0.5  # rad/s
        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 5].set(omega_z)  # gz

        result = preintegrate_imu_scan(imu_data, timestamps)

        # Expected heading change
        expected_heading = omega_z * duration

        np.testing.assert_allclose(result.delta_heading, expected_heading, rtol=2e-2)
        # No translation expected
        np.testing.assert_allclose(result.delta_position, jnp.zeros(2), atol=1e-6)

    def test_preintegrate_initial_velocity(self):
        """Test pre-integration with non-zero initial velocity."""
        dt = 0.01
        duration = 0.5
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Zero acceleration
        imu_data = jnp.zeros((n_samples, 6))

        # Initial velocity
        initial_velocity = jnp.array([10.0, 5.0])  # cm/s

        result = preintegrate_imu_scan(imu_data, timestamps, initial_velocity=initial_velocity)

        # Expected: constant velocity motion
        expected_position = initial_velocity * duration
        expected_velocity_change = jnp.zeros(2)  # No acceleration

        np.testing.assert_allclose(result.delta_position, expected_position, rtol=3e-2)
        np.testing.assert_allclose(result.delta_velocity, expected_velocity_change, atol=1e-6)


class TestPreintegrationBetweenFrames:
    """Tests for pre-integration between specific time points."""

    def test_preintegrate_between_frames_invalid_range(self):
        """Test error handling for invalid time range."""
        imu_data = jnp.zeros((100, 6))
        timestamps = jnp.linspace(0, 1, 100)

        with pytest.raises(ValueError, match="Invalid time range"):
            preintegrate_between_frames(imu_data, timestamps, start_time=1.0, end_time=0.5)

    def test_preintegrate_between_frames_no_data(self):
        """Test handling when no IMU samples in time range."""
        imu_data = jnp.zeros((100, 6))
        timestamps = jnp.linspace(0, 1, 100)

        # Time range with no samples
        result = preintegrate_between_frames(imu_data, timestamps, start_time=1.5, end_time=2.0)

        assert result.n_samples == 0
        np.testing.assert_allclose(result.delta_position, jnp.zeros(2))
        np.testing.assert_allclose(result.delta_velocity, jnp.zeros(2))
        assert result.dt == 0.5

    def test_preintegrate_between_frames_subset(self):
        """Test pre-integration on time subset."""
        # Create longer dataset
        dt = 0.01
        total_duration = 2.0
        timestamps = jnp.arange(0, total_duration, dt)
        n_samples = len(timestamps)

        # Constant acceleration
        accel_x = 1.0  # m/s²
        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(accel_x)

        # Integrate subset from 0.5s to 1.0s
        start_time = 0.5
        end_time = 1.0
        subset_duration = end_time - start_time

        result = preintegrate_between_frames(
            imu_data, timestamps, start_time=start_time, end_time=end_time
        )

        # Expected motion for subset duration
        expected_velocity_x = accel_x * subset_duration * 100  # cm/s
        expected_position_x = 0.5 * accel_x * subset_duration**2 * 100  # cm

        np.testing.assert_allclose(result.delta_velocity[0], expected_velocity_x, rtol=1e-3)
        np.testing.assert_allclose(result.delta_position[0], expected_position_x, rtol=1e-3)
        np.testing.assert_allclose(result.dt, subset_duration, atol=1e-6)


class TestNumericalIntegrationBaseline:
    """Tests comparing JAX pre-integration against numerical baselines."""

    def numerical_integration_baseline(
        self,
        imu_data: np.ndarray,
        timestamps: np.ndarray,
        initial_heading: float = 0.0,
        initial_velocity: np.ndarray = None,
        gyro_bias: float = 0.0,
        accel_bias: np.ndarray = None,
        damping_lambda: float = 0.0,
    ) -> dict:
        """Reference numerical integration implementation."""
        if initial_velocity is None:
            initial_velocity = np.zeros(2)
        if accel_bias is None:
            accel_bias = np.zeros(2)

        # Convert initial velocity to m/s
        velocity = initial_velocity.copy() / 100.0
        position = np.zeros(2)
        heading = initial_heading

        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]

            # Extract measurements
            accel_xy = imu_data[i, :2] - accel_bias
            gyro_z = imu_data[i, 5] - gyro_bias

            # Update heading
            heading += gyro_z * dt

            # Rotate acceleration to world frame
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            accel_world = np.array(
                [
                    cos_h * accel_xy[0] - sin_h * accel_xy[1],
                    sin_h * accel_xy[0] + cos_h * accel_xy[1],
                ]
            )

            # Apply damping and update velocity
            velocity_damping = -damping_lambda * velocity
            total_accel = accel_world + velocity_damping
            velocity += total_accel * dt

            # Update position
            position += velocity * dt

        return {
            "delta_position": position * 100,  # Convert to cm
            "delta_velocity": (velocity - initial_velocity / 100.0) * 100,  # Convert to cm/s
            "delta_heading": heading - initial_heading,
            "dt": timestamps[-1] - timestamps[0],
        }

    def test_jax_vs_numerical_constant_motion(self):
        """Compare JAX implementation against numerical baseline for constant motion."""
        # Create test data
        dt = 0.01
        duration = 1.0
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Constant acceleration and rotation
        accel_x, accel_y = 1.0, 0.5  # m/s²
        omega_z = 0.2  # rad/s

        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(accel_x)
        imu_data = imu_data.at[:, 1].set(accel_y)
        imu_data = imu_data.at[:, 5].set(omega_z)

        # JAX implementation
        jax_result = preintegrate_imu_scan(imu_data, timestamps)

        # Numerical baseline
        numerical_result = self.numerical_integration_baseline(
            np.array(imu_data), np.array(timestamps)
        )

        # Compare results (should be very close for simple motion)
        np.testing.assert_allclose(
            jax_result.delta_position, numerical_result["delta_position"], rtol=2e-2
        )
        np.testing.assert_allclose(
            jax_result.delta_velocity, numerical_result["delta_velocity"], rtol=2e-2
        )
        np.testing.assert_allclose(
            jax_result.delta_heading, numerical_result["delta_heading"], rtol=2e-2
        )

    def test_jax_vs_numerical_with_bias(self):
        """Compare JAX vs numerical with bias correction."""
        # Create test data
        dt = 0.005  # Higher resolution
        duration = 0.5
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Motion with bias
        gyro_bias = 0.05
        accel_bias = jnp.array([0.1, -0.02])

        # Raw measurements include bias
        omega_z_true = 0.3
        accel_true = jnp.array([0.8, 0.4])

        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(accel_true[0] + accel_bias[0])
        imu_data = imu_data.at[:, 1].set(accel_true[1] + accel_bias[1])
        imu_data = imu_data.at[:, 5].set(omega_z_true + gyro_bias)

        # JAX implementation
        jax_result = preintegrate_imu_scan(
            imu_data, timestamps, gyro_bias=gyro_bias, accel_bias=accel_bias
        )

        # Numerical baseline
        numerical_result = self.numerical_integration_baseline(
            np.array(imu_data),
            np.array(timestamps),
            gyro_bias=gyro_bias,
            accel_bias=np.array(accel_bias),
        )

        # Compare results
        np.testing.assert_allclose(
            jax_result.delta_position, numerical_result["delta_position"], rtol=2e-2
        )
        np.testing.assert_allclose(
            jax_result.delta_velocity, numerical_result["delta_velocity"], rtol=2e-2
        )
        np.testing.assert_allclose(
            jax_result.delta_heading, numerical_result["delta_heading"], rtol=2e-2
        )

    def test_jax_vs_numerical_complex_motion(self):
        """Compare JAX vs numerical for complex motion profile."""
        # Create complex motion with time-varying acceleration and rotation
        dt = 0.002  # High resolution for accuracy
        duration = 1.0
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Time-varying motion
        t = timestamps
        accel_x = 0.5 * jnp.sin(2 * jnp.pi * t)  # Sinusoidal acceleration
        accel_y = 0.3 * jnp.cos(2 * jnp.pi * t)
        omega_z = 0.2 * jnp.sin(4 * jnp.pi * t)  # Higher frequency rotation

        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(accel_x)
        imu_data = imu_data.at[:, 1].set(accel_y)
        imu_data = imu_data.at[:, 5].set(omega_z)

        # JAX implementation
        jax_result = preintegrate_imu_scan(imu_data, timestamps)

        # Numerical baseline
        numerical_result = self.numerical_integration_baseline(
            np.array(imu_data), np.array(timestamps)
        )

        # For complex motion, allow slightly larger tolerance due to integration differences
        np.testing.assert_allclose(
            jax_result.delta_position, numerical_result["delta_position"], rtol=5e-2, atol=1e-12
        )
        np.testing.assert_allclose(
            jax_result.delta_velocity, numerical_result["delta_velocity"], rtol=5e-2, atol=1e-12
        )
        np.testing.assert_allclose(
            jax_result.delta_heading, numerical_result["delta_heading"], rtol=5e-2, atol=1e-12
        )


class TestConversionUtilities:
    """Tests for SpikeGadgets data conversion utilities."""

    def test_convert_spikegadgets_to_preintegration_units(self):
        """Test conversion from SpikeGadgets format to pre-integration format."""
        n_samples = 100

        # Create sample SpikeGadgets data
        accel_ms2 = np.random.randn(n_samples, 3) * 2.0  # m/s²
        gyro_rad_s = np.random.randn(n_samples, 3) * 0.5  # rad/s

        # Convert
        imu_combined = convert_spikegadgets_to_preintegration_units(accel_ms2, gyro_rad_s)

        # Check shape and contents
        assert imu_combined.shape == (n_samples, 6)
        np.testing.assert_allclose(imu_combined[:, :3], accel_ms2)
        np.testing.assert_allclose(imu_combined[:, 3:], gyro_rad_s)

    def test_convert_spikegadgets_mismatched_shapes(self):
        """Test error handling for mismatched input shapes."""
        accel_ms2 = np.random.randn(100, 3)
        gyro_rad_s = np.random.randn(90, 3)  # Different length

        with pytest.raises(Exception):  # JAX will raise an error for mismatched shapes
            convert_spikegadgets_to_preintegration_units(accel_ms2, gyro_rad_s)

    def test_convert_spikegadgets_wrong_dimensions(self):
        """Test error handling for wrong input dimensions."""
        accel_ms2 = np.random.randn(100, 2)  # Wrong shape
        gyro_rad_s = np.random.randn(100, 3)

        with pytest.raises(Exception):  # Chex assertion will fail
            convert_spikegadgets_to_preintegration_units(accel_ms2, gyro_rad_s)


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        st.floats(0.001, 0.1, allow_nan=False, allow_infinity=False),
    )
    def test_zero_motion_gives_zero_deltas(self, initial_vx, initial_vy, dt_step):
        """Property test: zero IMU input should give zero motion change."""
        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            n_steps = 10
            # Ensure timestamps and data have exactly the same length
            timestamps = jnp.linspace(0, n_steps * dt_step, n_steps, endpoint=False)
            imu_data = jnp.zeros((n_steps, 6))

            initial_velocity = jnp.array([initial_vx, initial_vy])

            result = preintegrate_imu_scan(imu_data, timestamps, initial_velocity=initial_velocity)

            # With zero acceleration and rotation, velocity should remain constant
            # So delta_velocity should be zero
            np.testing.assert_allclose(result.delta_velocity, jnp.zeros(2), atol=1e-10)

            # Position change should be initial_velocity * total_time
            total_time = timestamps[-1] - timestamps[0]
            expected_position_change = initial_velocity * total_time
            np.testing.assert_allclose(
                result.delta_position, expected_position_change, rtol=1e-8, atol=1e-8
            )

    @given(
        st.floats(-2, 2, allow_nan=False, allow_infinity=False),
        st.floats(0.001, 0.1, allow_nan=False, allow_infinity=False),
    )
    def test_constant_rotation_is_linear(self, omega_z, dt_step):
        """Property test: constant rotation should be linear in time."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            n_steps = 20
            timestamps = jnp.arange(0, n_steps * dt_step, dt_step)

            imu_data = jnp.zeros((n_steps, 6))
            imu_data = imu_data.at[:, 5].set(omega_z)  # Constant gyro_z

            result = preintegrate_imu_scan(imu_data, timestamps)

            total_time = timestamps[-1] - timestamps[0]
            expected_heading_change = omega_z * total_time

            # Allow small tolerance for numerical precision
            np.testing.assert_allclose(
                result.delta_heading, expected_heading_change, rtol=1e-6, atol=1e-8
            )

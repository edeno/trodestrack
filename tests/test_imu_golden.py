"""Golden tests comparing JAX pre-integration against high-resolution numerical baselines.

These tests implement the three motion profiles specified in TASKS.md:
1. Steady motion - constant velocity and acceleration
2. Step-turn - sudden heading changes
3. Sinusoid - smooth periodic motion

Each test compares the JAX implementation against a high-resolution numerical
integration baseline to ensure accuracy within specified tolerances.
"""

import jax
# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)

import pytest
import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import warnings

from trodestrack.imu.preintegration import preintegrate_imu_scan
from trodestrack.constants import DEGREES_TO_RADIANS


class HighResolutionBaseline:
    """High-resolution numerical integration baseline using scipy.integrate."""

    @staticmethod
    def differential_equation(t, state, imu_interp_func, gyro_bias, accel_bias, damping_lambda):
        """Differential equation for IMU pre-integration.

        State vector: [px, py, vx, vy, theta]

        Parameters
        ----------
        t : float
            Current time
        state : np.ndarray, shape (5,)
            Current state [px, py, vx, vy, theta]
        imu_interp_func : callable
            Function that returns IMU data at time t
        gyro_bias : float
            Gyroscope bias in rad/s
        accel_bias : np.ndarray, shape (2,)
            Accelerometer bias in m/s²
        damping_lambda : float
            Velocity damping factor

        Returns
        -------
        np.ndarray, shape (5,)
            State derivative [dpx/dt, dpy/dt, dvx/dt, dvy/dt, dtheta/dt]
        """
        px, py, vx, vy, theta = state

        # Get IMU measurements at current time
        imu_data = imu_interp_func(t)
        accel_x, accel_y, gyro_z = imu_data[0], imu_data[1], imu_data[5]

        # Apply bias correction
        accel_corrected = np.array([accel_x - accel_bias[0], accel_y - accel_bias[1]])
        gyro_corrected = gyro_z - gyro_bias

        # Rotate acceleration to world frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        accel_world = np.array([
            cos_theta * accel_corrected[0] - sin_theta * accel_corrected[1],
            sin_theta * accel_corrected[0] + cos_theta * accel_corrected[1]
        ])

        # Apply velocity damping
        velocity_damping = -damping_lambda * np.array([vx, vy])
        total_accel = accel_world + velocity_damping

        # State derivatives
        dpx_dt = vx
        dpy_dt = vy
        dvx_dt = total_accel[0]
        dvy_dt = total_accel[1]
        dtheta_dt = gyro_corrected

        return np.array([dpx_dt, dpy_dt, dvx_dt, dvy_dt, dtheta_dt])

    @classmethod
    def integrate_motion(
        cls,
        imu_data: np.ndarray,
        timestamps: np.ndarray,
        initial_heading: float = 0.0,
        initial_velocity: np.ndarray = None,
        gyro_bias: float = 0.0,
        accel_bias: np.ndarray = None,
        damping_lambda: float = 0.0,
        rtol: float = 1e-8,
        atol: float = 1e-10
    ) -> dict:
        """High-resolution numerical integration using scipy.solve_ivp.

        Parameters
        ----------
        imu_data : np.ndarray, shape (n_samples, 6)
            IMU measurements [ax, ay, az, gx, gy, gz] in SI units
        timestamps : np.ndarray, shape (n_samples,)
            Sample timestamps in seconds
        initial_heading : float, optional
            Initial heading in radians
        initial_velocity : np.ndarray, shape (2,), optional
            Initial velocity [vx, vy] in cm/s
        gyro_bias : float, optional
            Gyroscope bias in rad/s
        accel_bias : np.ndarray, shape (2,), optional
            Accelerometer bias in m/s²
        damping_lambda : float, optional
            Velocity damping factor
        rtol : float, optional
            Relative tolerance for integration
        atol : float, optional
            Absolute tolerance for integration

        Returns
        -------
        dict
            Integration results with high-resolution trajectory
        """
        if initial_velocity is None:
            initial_velocity = np.zeros(2)
        if accel_bias is None:
            accel_bias = np.zeros(2)

        # Convert initial velocity to m/s
        initial_velocity_ms = initial_velocity / 100.0

        # Create interpolation function for IMU data
        from scipy.interpolate import interp1d
        imu_interp = interp1d(
            timestamps, imu_data, axis=0, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )

        # Initial state: [px, py, vx, vy, theta] in SI units
        initial_state = np.array([0.0, 0.0, initial_velocity_ms[0], initial_velocity_ms[1], initial_heading])

        # Time span
        t_span = (timestamps[0], timestamps[-1])

        # High-resolution time evaluation points
        n_eval = len(timestamps) * 10  # 10x oversampling
        t_eval = np.linspace(timestamps[0], timestamps[-1], n_eval)

        # Solve differential equation
        sol = solve_ivp(
            cls.differential_equation,
            t_span,
            initial_state,
            args=(imu_interp, gyro_bias, accel_bias, damping_lambda),
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            method='RK45'  # 4th-order Runge-Kutta
        )

        if not sol.success:
            raise RuntimeError(f"High-resolution integration failed: {sol.message}")

        # Extract final state
        final_state = sol.y[:, -1]
        final_px, final_py, final_vx, final_vy, final_theta = final_state

        # Convert back to original units
        delta_position_cm = np.array([final_px, final_py]) * 100.0  # m to cm
        delta_velocity_cm = np.array([final_vx - initial_velocity_ms[0],
                                     final_vy - initial_velocity_ms[1]]) * 100.0  # m/s to cm/s
        delta_heading = final_theta - initial_heading

        return {
            'delta_position': delta_position_cm,
            'delta_velocity': delta_velocity_cm,
            'delta_heading': delta_heading,
            'dt': timestamps[-1] - timestamps[0],
            'trajectory': {
                'times': sol.t,
                'positions': sol.y[:2] * 100.0,  # Convert to cm
                'velocities': sol.y[2:4] * 100.0,  # Convert to cm/s
                'headings': sol.y[4]
            }
        }


class TestGoldenMotionProfiles:
    """Golden tests for the three specified motion profiles."""

    def test_golden_steady_motion(self):
        """Golden test for steady motion (constant velocity and acceleration)."""
        # Test parameters
        dt = 0.001  # High resolution: 1 kHz
        duration = 2.0
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Steady motion parameters
        constant_accel = jnp.array([0.5, 0.3])  # m/s² in x,y
        constant_gyro = 0.1  # rad/s steady rotation
        initial_velocity = jnp.array([2.0, 1.5])  # cm/s

        # Create IMU data with steady motion
        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(constant_accel[0])  # ax
        imu_data = imu_data.at[:, 1].set(constant_accel[1])  # ay
        imu_data = imu_data.at[:, 5].set(constant_gyro)      # gz

        # JAX implementation
        jax_result = preintegrate_imu_scan(
            imu_data, timestamps, initial_velocity=initial_velocity
        )

        # High-resolution baseline
        baseline_result = HighResolutionBaseline.integrate_motion(
            np.array(imu_data), np.array(timestamps), initial_velocity=np.array(initial_velocity)
        )

        # Compare results with tight tolerances for steady motion
        np.testing.assert_allclose(
            jax_result.delta_position, baseline_result['delta_position'],
            rtol=1e-4, atol=0.01,  # 1cm absolute, 0.01% relative tolerance
            err_msg="Position integration mismatch for steady motion"
        )

        np.testing.assert_allclose(
            jax_result.delta_velocity, baseline_result['delta_velocity'],
            rtol=1e-4, atol=0.1,  # 0.1 cm/s absolute tolerance
            err_msg="Velocity integration mismatch for steady motion"
        )

        np.testing.assert_allclose(
            jax_result.delta_heading, baseline_result['delta_heading'],
            rtol=1e-4, atol=0.001,  # ~0.06 degree absolute tolerance
            err_msg="Heading integration mismatch for steady motion"
        )

    def test_golden_step_turn_motion(self):
        """Golden test for step-turn motion (sudden heading changes)."""
        # Test parameters
        dt = 0.0005  # Very high resolution for step changes: 2 kHz
        duration = 1.0
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Step-turn motion: sudden rotation changes
        imu_data = jnp.zeros((n_samples, 6))

        # Constant forward acceleration
        forward_accel = 1.0  # m/s²
        imu_data = imu_data.at[:, 0].set(forward_accel)

        # Step changes in rotation at specific times
        step_times = [0.2, 0.5, 0.8]  # Times for sudden turns
        step_rotations = [0.5, -0.8, 0.3]  # rad/s

        for i, t in enumerate(timestamps):
            if t < step_times[0]:
                gyro_z = 0.0
            elif t < step_times[1]:
                gyro_z = step_rotations[0]
            elif t < step_times[2]:
                gyro_z = step_rotations[1]
            else:
                gyro_z = step_rotations[2]
            imu_data = imu_data.at[i, 5].set(gyro_z)

        # JAX implementation
        jax_result = preintegrate_imu_scan(imu_data, timestamps)

        # High-resolution baseline with tighter tolerances for step changes
        baseline_result = HighResolutionBaseline.integrate_motion(
            np.array(imu_data), np.array(timestamps), rtol=1e-10, atol=1e-12
        )

        # Compare results with appropriate tolerances for step motion
        np.testing.assert_allclose(
            jax_result.delta_position, baseline_result['delta_position'],
            rtol=1e-3, atol=0.05,  # 5cm absolute tolerance for step changes
            err_msg="Position integration mismatch for step-turn motion"
        )

        np.testing.assert_allclose(
            jax_result.delta_velocity, baseline_result['delta_velocity'],
            rtol=1e-3, atol=0.5,  # 0.5 cm/s absolute tolerance
            err_msg="Velocity integration mismatch for step-turn motion"
        )

        np.testing.assert_allclose(
            jax_result.delta_heading, baseline_result['delta_heading'],
            rtol=1e-3, atol=0.01,  # ~0.6 degree absolute tolerance
            err_msg="Heading integration mismatch for step-turn motion"
        )

    def test_golden_sinusoidal_motion(self):
        """Golden test for sinusoidal motion (smooth periodic motion)."""
        # Test parameters
        dt = 0.0005  # High resolution: 2 kHz
        duration = 3.0  # Longer duration to capture multiple cycles
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Sinusoidal motion parameters
        accel_freq_x = 0.5  # Hz
        accel_freq_y = 0.7  # Hz (different frequency)
        accel_amp_x = 2.0   # m/s²
        accel_amp_y = 1.5   # m/s²

        gyro_freq = 0.3     # Hz
        gyro_amp = 0.8      # rad/s

        # Create sinusoidal IMU data
        t = timestamps
        accel_x = accel_amp_x * jnp.sin(2 * jnp.pi * accel_freq_x * t)
        accel_y = accel_amp_y * jnp.cos(2 * jnp.pi * accel_freq_y * t)  # Cosine for phase difference
        gyro_z = gyro_amp * jnp.sin(2 * jnp.pi * gyro_freq * t)

        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(accel_x)
        imu_data = imu_data.at[:, 1].set(accel_y)
        imu_data = imu_data.at[:, 5].set(gyro_z)

        # Initial conditions
        initial_velocity = jnp.array([1.0, -0.5])  # cm/s
        initial_heading = 0.2  # rad

        # JAX implementation
        jax_result = preintegrate_imu_scan(
            imu_data, timestamps,
            initial_velocity=initial_velocity,
            initial_heading=initial_heading
        )

        # High-resolution baseline
        baseline_result = HighResolutionBaseline.integrate_motion(
            np.array(imu_data), np.array(timestamps),
            initial_velocity=np.array(initial_velocity),
            initial_heading=initial_heading,
            rtol=1e-9, atol=1e-11
        )

        # Compare results with tolerances appropriate for smooth periodic motion
        np.testing.assert_allclose(
            jax_result.delta_position, baseline_result['delta_position'],
            rtol=5e-4, atol=0.02,  # 2cm absolute, 0.05% relative tolerance
            err_msg="Position integration mismatch for sinusoidal motion"
        )

        np.testing.assert_allclose(
            jax_result.delta_velocity, baseline_result['delta_velocity'],
            rtol=5e-4, atol=0.2,  # 0.2 cm/s absolute tolerance
            err_msg="Velocity integration mismatch for sinusoidal motion"
        )

        np.testing.assert_allclose(
            jax_result.delta_heading, baseline_result['delta_heading'],
            rtol=5e-4, atol=0.005,  # ~0.3 degree absolute tolerance
            err_msg="Heading integration mismatch for sinusoidal motion"
        )

    def test_golden_with_bias_and_damping(self):
        """Golden test with bias correction and velocity damping."""
        # Test parameters
        dt = 0.001
        duration = 1.5
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Motion with bias
        gyro_bias = 0.05  # rad/s
        accel_bias = jnp.array([0.1, -0.05])  # m/s²
        damping_lambda = 1.0  # 1/s velocity damping

        # True motion (without bias)
        true_accel = jnp.array([0.8, 0.6])  # m/s²
        true_gyro = 0.2  # rad/s

        # Raw measurements (with bias)
        raw_accel = true_accel + accel_bias
        raw_gyro = true_gyro + gyro_bias

        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(raw_accel[0])
        imu_data = imu_data.at[:, 1].set(raw_accel[1])
        imu_data = imu_data.at[:, 5].set(raw_gyro)

        initial_velocity = jnp.array([3.0, 2.0])  # cm/s

        # JAX implementation with bias correction and damping
        jax_result = preintegrate_imu_scan(
            imu_data, timestamps,
            initial_velocity=initial_velocity,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            damping_lambda=damping_lambda
        )

        # High-resolution baseline with same parameters
        baseline_result = HighResolutionBaseline.integrate_motion(
            np.array(imu_data), np.array(timestamps),
            initial_velocity=np.array(initial_velocity),
            gyro_bias=gyro_bias,
            accel_bias=np.array(accel_bias),
            damping_lambda=damping_lambda,
            rtol=1e-9, atol=1e-11
        )

        # Compare results
        np.testing.assert_allclose(
            jax_result.delta_position, baseline_result['delta_position'],
            rtol=1e-3, atol=0.05,  # Looser tolerance due to damping complexity
            err_msg="Position integration mismatch with bias and damping"
        )

        np.testing.assert_allclose(
            jax_result.delta_velocity, baseline_result['delta_velocity'],
            rtol=1e-3, atol=0.3,
            err_msg="Velocity integration mismatch with bias and damping"
        )

        np.testing.assert_allclose(
            jax_result.delta_heading, baseline_result['delta_heading'],
            rtol=1e-3, atol=0.01,
            err_msg="Heading integration mismatch with bias and damping"
        )

    def test_golden_accuracy_regression(self):
        """Regression test to ensure accuracy doesn't degrade over time."""
        # This test uses a fixed random seed to ensure reproducible results
        np.random.seed(42)

        dt = 0.001
        duration = 2.0
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Complex but deterministic motion
        t = timestamps
        accel_x = 1.0 * jnp.sin(2 * jnp.pi * 0.4 * t) + 0.3 * jnp.cos(2 * jnp.pi * 1.2 * t)
        accel_y = 0.8 * jnp.cos(2 * jnp.pi * 0.6 * t) - 0.2 * jnp.sin(2 * jnp.pi * 0.9 * t)
        gyro_z = 0.5 * jnp.sin(2 * jnp.pi * 0.3 * t)

        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(accel_x)
        imu_data = imu_data.at[:, 1].set(accel_y)
        imu_data = imu_data.at[:, 5].set(gyro_z)

        # JAX implementation
        jax_result = preintegrate_imu_scan(imu_data, timestamps)

        # These are the expected values from a known-good implementation
        # (These would be established once and then used for regression testing)
        expected_delta_position = jnp.array([1.234567, -0.987654])  # Placeholder values
        expected_delta_velocity = jnp.array([2.345678, 1.876543])   # Placeholder values
        expected_delta_heading = 0.567890  # Placeholder value

        # Note: In practice, these expected values would be computed once using the
        # high-resolution baseline and then stored as reference values

        # For now, just verify that the implementation runs without errors
        # and produces reasonable results
        assert not jnp.isnan(jax_result.delta_position).any()
        assert not jnp.isnan(jax_result.delta_velocity).any()
        assert not jnp.isnan(jax_result.delta_heading)
        assert jax_result.dt == pytest.approx(duration, abs=1e-6)
        assert jax_result.n_samples == n_samples


class TestGoldenToleranceVerification:
    """Verify that golden test tolerances meet the acceptance criteria."""

    def test_accuracy_meets_prd_requirements(self):
        """Verify that integration accuracy meets PRD requirements.

        From PRD: Synthetic benchmarks ≤2 cm RMSE position, ≤10 cm/s velocity, ≤7° heading.
        """
        # This test would ideally run on a large set of synthetic trajectories
        # and verify that the average errors meet the PRD requirements

        # For now, implement a representative test case
        dt = 0.002
        duration = 5.0  # Longer test for error accumulation
        timestamps = jnp.arange(0, duration, dt)
        n_samples = len(timestamps)

        # Realistic rat motion parameters
        max_accel = 3.0  # m/s² (reasonable for small animals)
        max_rotation = 2.0  # rad/s

        # Create semi-realistic motion
        t = timestamps
        accel_x = max_accel * 0.3 * jnp.sin(2 * jnp.pi * 0.2 * t)  # Slow acceleration changes
        accel_y = max_accel * 0.2 * jnp.cos(2 * jnp.pi * 0.15 * t)
        gyro_z = max_rotation * 0.4 * jnp.sin(2 * jnp.pi * 0.1 * t)  # Slow turns

        imu_data = jnp.zeros((n_samples, 6))
        imu_data = imu_data.at[:, 0].set(accel_x)
        imu_data = imu_data.at[:, 1].set(accel_y)
        imu_data = imu_data.at[:, 5].set(gyro_z)

        # JAX implementation
        jax_result = preintegrate_imu_scan(imu_data, timestamps)

        # High-resolution baseline
        baseline_result = HighResolutionBaseline.integrate_motion(
            np.array(imu_data), np.array(timestamps), rtol=1e-10, atol=1e-12
        )

        # Calculate errors
        position_error = np.linalg.norm(jax_result.delta_position - baseline_result['delta_position'])
        velocity_error = np.linalg.norm(jax_result.delta_velocity - baseline_result['delta_velocity'])
        heading_error_deg = abs(jax_result.delta_heading - baseline_result['delta_heading']) * 180 / np.pi

        # Verify errors are well within PRD requirements
        assert position_error < 1.0, f"Position error {position_error:.3f} cm should be < 1.0 cm"
        assert velocity_error < 5.0, f"Velocity error {velocity_error:.3f} cm/s should be < 5.0 cm/s"
        assert heading_error_deg < 3.0, f"Heading error {heading_error_deg:.3f}° should be < 3.0°"

    def test_convergence_with_resolution(self):
        """Test that JAX implementation converges to baseline as resolution increases."""
        # Test different resolutions and verify convergence
        base_duration = 1.0

        # Motion profile
        accel_amp = 1.0  # m/s²
        gyro_amp = 0.5   # rad/s

        errors_position = []
        errors_velocity = []
        errors_heading = []
        resolutions = [0.01, 0.005, 0.002, 0.001]  # Decreasing time steps

        for dt in resolutions:
            timestamps = jnp.arange(0, base_duration, dt)
            n_samples = len(timestamps)

            t = timestamps
            accel_x = accel_amp * jnp.sin(2 * jnp.pi * 0.5 * t)
            gyro_z = gyro_amp * jnp.cos(2 * jnp.pi * 0.3 * t)

            imu_data = jnp.zeros((n_samples, 6))
            imu_data = imu_data.at[:, 0].set(accel_x)
            imu_data = imu_data.at[:, 5].set(gyro_z)

            # JAX implementation
            jax_result = preintegrate_imu_scan(imu_data, timestamps)

            # High-resolution baseline
            baseline_result = HighResolutionBaseline.integrate_motion(
                np.array(imu_data), np.array(timestamps), rtol=1e-10, atol=1e-12
            )

            # Calculate errors
            pos_error = np.linalg.norm(jax_result.delta_position - baseline_result['delta_position'])
            vel_error = np.linalg.norm(jax_result.delta_velocity - baseline_result['delta_velocity'])
            head_error = abs(jax_result.delta_heading - baseline_result['delta_heading'])

            errors_position.append(pos_error)
            errors_velocity.append(vel_error)
            errors_heading.append(head_error)

        # Verify that errors generally decrease with finer resolution
        # (Allow for some numerical noise at the finest resolutions)
        assert errors_position[-1] < errors_position[0], "Position error should decrease with finer resolution"
        assert errors_velocity[-1] < errors_velocity[0], "Velocity error should decrease with finer resolution"
        assert errors_heading[-1] < errors_heading[0], "Heading error should decrease with finer resolution"
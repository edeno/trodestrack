"""Tests for IMU utility functions."""

import pytest
import numpy as np
from unittest.mock import patch
import warnings

from trodestrack.imu.utils import (
    remove_gravity_estimate,
    detect_imu_misalignment,
    estimate_gyroscope_bias,
    compute_imu_alignment_matrix,
    validate_imu_data_quality,
)
from trodestrack.constants import STANDARD_GRAVITY_MS2


class TestRemoveGravityEstimate:
    """Tests for gravity removal functionality."""

    def test_remove_gravity_estimate_basic(self):
        """Test basic gravity removal."""
        n_samples = 1000
        gravity_offset = 9.8

        # Create accelerometer data with gravity in z-axis
        accel_data = np.random.randn(n_samples, 3) * 0.1  # Small noise
        accel_data[:, 2] += gravity_offset  # Add gravity to z-axis

        result = remove_gravity_estimate(accel_data, axis=2, window_size=100)

        # Z-axis should have reduced mean (gravity removed)
        original_mean_z = np.mean(accel_data[:, 2])
        result_mean_z = np.mean(result[:, 2])

        assert abs(original_mean_z - gravity_offset) < 0.1
        assert abs(result_mean_z) < 0.5  # Should be much smaller after gravity removal

        # X and Y should be unchanged
        np.testing.assert_allclose(result[:, 0], accel_data[:, 0], rtol=1e-10)
        np.testing.assert_allclose(result[:, 1], accel_data[:, 1], rtol=1e-10)

    def test_remove_gravity_estimate_wrong_shape(self):
        """Test error handling for wrong input shape."""
        accel_data = np.random.randn(100, 2)  # Wrong shape

        with pytest.raises(ValueError, match="must have 3 columns"):
            remove_gravity_estimate(accel_data)

    def test_remove_gravity_estimate_short_sequence(self):
        """Test gravity removal with short sequence."""
        accel_data = np.random.randn(10, 3)
        accel_data[:, 2] += 9.8

        # Should fallback to global mean
        result = remove_gravity_estimate(accel_data, window_size=100)

        # Check that gravity was removed (approximately)
        assert abs(np.mean(result[:, 2])) < abs(np.mean(accel_data[:, 2]))

    def test_remove_gravity_estimate_different_axis(self):
        """Test gravity removal on different axis."""
        n_samples = 500
        gravity_offset = -9.8

        accel_data = np.random.randn(n_samples, 3) * 0.1
        accel_data[:, 1] += gravity_offset  # Gravity in y-axis

        result = remove_gravity_estimate(accel_data, axis=1, window_size=50)

        # Y-axis should have gravity removed
        assert abs(np.mean(result[:, 1])) < abs(np.mean(accel_data[:, 1]))

        # X and Z should be unchanged
        np.testing.assert_allclose(result[:, 0], accel_data[:, 0], rtol=1e-10)
        np.testing.assert_allclose(result[:, 2], accel_data[:, 2], rtol=1e-10)


class TestDetectIMUMisalignment:
    """Tests for IMU misalignment detection."""

    def test_detect_misalignment_well_aligned(self):
        """Test detection with well-aligned IMU."""
        n_samples = 1000

        # Create data with correct gravity magnitude
        accel_data = np.random.randn(n_samples, 3) * 0.5
        accel_data[:, 2] += STANDARD_GRAVITY_MS2  # Correct gravity in z

        is_misaligned, error = detect_imu_misalignment(accel_data)

        assert not is_misaligned
        assert error < 0.1  # Low error

    def test_detect_misalignment_badly_aligned(self):
        """Test detection with badly aligned IMU."""
        n_samples = 1000

        # Create data with incorrect gravity magnitude
        wrong_gravity = STANDARD_GRAVITY_MS2 * 1.5  # 50% error
        accel_data = np.random.randn(n_samples, 3) * 0.5
        accel_data[:, 2] += wrong_gravity

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            is_misaligned, error = detect_imu_misalignment(accel_data)

            assert is_misaligned
            assert error > 0.2  # High error
            assert len(w) == 1  # Should have issued warning
            assert "misaligned" in str(w[0].message)

    def test_detect_misalignment_empty_data(self):
        """Test misalignment detection with empty data."""
        accel_data = np.zeros((0, 3))

        is_misaligned, error = detect_imu_misalignment(accel_data)

        assert not is_misaligned
        assert error == 0.0

    def test_detect_misalignment_wrong_shape(self):
        """Test error handling for wrong input shape."""
        accel_data = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="must have 3 columns"):
            detect_imu_misalignment(accel_data)

    def test_detect_misalignment_custom_tolerance(self):
        """Test misalignment detection with custom tolerance."""
        n_samples = 500

        # Create data with 15% gravity error
        wrong_gravity = STANDARD_GRAVITY_MS2 * 1.15
        accel_data = np.random.randn(n_samples, 3) * 0.3
        accel_data[:, 2] += wrong_gravity

        # Should not be misaligned with lenient tolerance
        is_misaligned_lenient, _ = detect_imu_misalignment(accel_data, tolerance=0.3)
        assert not is_misaligned_lenient

        # Should be misaligned with strict tolerance
        is_misaligned_strict, _ = detect_imu_misalignment(accel_data, tolerance=0.1)
        assert is_misaligned_strict


class TestEstimateGyroscopeBias:
    """Tests for gyroscope bias estimation."""

    def test_estimate_bias_stationary(self):
        """Test bias estimation with stationary data."""
        n_samples = 1000
        true_bias = np.array([0.05, -0.02, 0.03])  # rad/s

        # Create stationary data (just bias + small noise)
        gyro_data = np.random.randn(n_samples, 3) * 0.01 + true_bias

        estimated_bias, is_reliable = estimate_gyroscope_bias(
            gyro_data, stationary_threshold=0.1
        )

        assert is_reliable
        np.testing.assert_allclose(estimated_bias, true_bias, atol=0.01)

    def test_estimate_bias_moving(self):
        """Test bias estimation with moving data."""
        n_samples = 1000
        true_bias = np.array([0.02, 0.01, -0.01])

        # Create data with large motion
        gyro_data = np.random.randn(n_samples, 3) * 2.0 + true_bias

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            estimated_bias, is_reliable = estimate_gyroscope_bias(
                gyro_data, stationary_threshold=0.1
            )

            assert not is_reliable
            assert len(w) == 1  # Should warn about insufficient stationary samples

    def test_estimate_bias_mixed_motion(self):
        """Test bias estimation with mixed stationary and moving periods."""
        n_samples = 1000
        true_bias = np.array([0.03, -0.01, 0.02])

        gyro_data = np.zeros((n_samples, 3))

        # First 300 samples: stationary (bias + small noise)
        gyro_data[:300] = np.random.randn(300, 3) * 0.02 + true_bias

        # Remaining samples: large motion
        gyro_data[300:] = np.random.randn(700, 3) * 1.0 + true_bias

        estimated_bias, is_reliable = estimate_gyroscope_bias(
            gyro_data, stationary_threshold=0.1, min_samples=100
        )

        assert is_reliable
        # Should estimate bias from stationary period
        np.testing.assert_allclose(estimated_bias, true_bias, atol=0.05)

    def test_estimate_bias_empty_data(self):
        """Test bias estimation with empty data."""
        gyro_data = np.zeros((0, 3))

        estimated_bias, is_reliable = estimate_gyroscope_bias(gyro_data)

        assert not is_reliable
        np.testing.assert_allclose(estimated_bias, np.zeros(3))

    def test_estimate_bias_wrong_shape(self):
        """Test error handling for wrong input shape."""
        gyro_data = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="must have 3 columns"):
            estimate_gyroscope_bias(gyro_data)


class TestComputeIMUAlignmentMatrix:
    """Tests for IMU alignment matrix computation."""

    def test_alignment_matrix_identity(self):
        """Test alignment matrix when gravity is already aligned."""
        gravity_vector = np.array([0, 0, 1])  # Already aligned with z-axis
        R = compute_imu_alignment_matrix(gravity_vector, target_axis=2)

        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_alignment_matrix_90_degree_rotation(self):
        """Test alignment matrix for 90-degree rotation."""
        gravity_vector = np.array([1, 0, 0])  # Gravity in x-axis
        R = compute_imu_alignment_matrix(gravity_vector, target_axis=2)

        # Should rotate x-axis to z-axis
        rotated_gravity = R @ gravity_vector
        expected = np.array([0, 0, 1])

        np.testing.assert_allclose(rotated_gravity, expected, atol=1e-10)

    def test_alignment_matrix_orthogonal(self):
        """Test that alignment matrix is orthogonal."""
        gravity_vector = np.array([0.6, 0.8, 0])  # Arbitrary gravity direction
        R = compute_imu_alignment_matrix(gravity_vector, target_axis=1)

        # Check orthogonality: R @ R.T = I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

        # Check determinant = 1
        det = np.linalg.det(R)
        np.testing.assert_allclose(det, 1.0, atol=1e-10)

    def test_alignment_matrix_anti_aligned(self):
        """Test alignment when gravity is anti-aligned."""
        gravity_vector = np.array([0, 0, -1])  # Anti-aligned with z-axis
        R = compute_imu_alignment_matrix(gravity_vector, target_axis=2)

        # Should flip gravity to positive z
        rotated_gravity = R @ gravity_vector
        expected = np.array([0, 0, 1])

        np.testing.assert_allclose(rotated_gravity, expected, atol=1e-10)

    def test_alignment_matrix_different_target_axes(self):
        """Test alignment to different target axes."""
        gravity_vector = np.array([0, 0, 1])

        for target_axis in [0, 1, 2]:
            R = compute_imu_alignment_matrix(gravity_vector, target_axis=target_axis)

            # Check that gravity aligns with target axis
            rotated_gravity = R @ gravity_vector
            expected = np.zeros(3)
            expected[target_axis] = 1.0

            np.testing.assert_allclose(rotated_gravity, expected, atol=1e-10)

    def test_alignment_matrix_wrong_shape(self):
        """Test error handling for wrong gravity vector shape."""
        gravity_vector = np.array([1, 1])  # Wrong shape

        with pytest.raises(ValueError, match="must have 3 components"):
            compute_imu_alignment_matrix(gravity_vector)


class TestValidateIMUDataQuality:
    """Tests for IMU data quality validation."""

    def test_validate_good_quality_data(self):
        """Test validation with good quality data."""
        n_samples = 10000
        sampling_rate = 1000.0
        dt = 1.0 / sampling_rate
        timestamps = np.arange(0, n_samples * dt, dt)[:n_samples]

        # Good quality IMU data
        accel_data = np.random.randn(n_samples, 3) * 0.5  # Smaller noise for better gravity alignment
        accel_data[:, 2] += STANDARD_GRAVITY_MS2  # Add gravity
        gyro_data = np.random.randn(n_samples, 3) * 0.05  # Small gyro noise for stationary periods

        diagnostics = validate_imu_data_quality(
            accel_data, gyro_data, timestamps, sampling_rate
        )

        assert diagnostics['quality_good']
        assert len(diagnostics['quality_issues']) == 0
        assert diagnostics['n_samples'] == n_samples
        assert diagnostics['timing_error_percent'] < 1.0
        assert not diagnostics['accel_saturated']
        assert not diagnostics['gyro_saturated']

    def test_validate_timing_issues(self):
        """Test validation with timing issues."""
        n_samples = 1000
        sampling_rate = 1000.0

        # Create irregular timestamps
        timestamps = np.cumsum(np.random.exponential(1.0/sampling_rate, n_samples))
        accel_data = np.random.randn(n_samples, 3) * 2.0
        gyro_data = np.random.randn(n_samples, 3) * 0.5

        diagnostics = validate_imu_data_quality(
            accel_data, gyro_data, timestamps, sampling_rate
        )

        assert not diagnostics['quality_good']
        assert 'timing_inconsistent' in diagnostics['quality_issues']

    def test_validate_saturated_data(self):
        """Test validation with saturated data."""
        n_samples = 1000
        sampling_rate = 1000.0
        dt = 1.0 / sampling_rate
        timestamps = np.arange(0, n_samples * dt, dt)[:n_samples]

        # Create saturated accelerometer data
        accel_data = np.random.randn(n_samples, 3) * 2.0
        accel_data[100:200, 0] = 200.0  # Saturated samples

        gyro_data = np.random.randn(n_samples, 3) * 0.5

        diagnostics = validate_imu_data_quality(
            accel_data, gyro_data, timestamps, sampling_rate
        )

        assert not diagnostics['quality_good']
        assert 'accel_saturated' in diagnostics['quality_issues']
        assert diagnostics['accel_saturated']

    def test_validate_gravity_misalignment(self):
        """Test validation with gravity misalignment."""
        n_samples = 1000
        sampling_rate = 1000.0
        dt = 1.0 / sampling_rate
        timestamps = np.arange(0, n_samples * dt, dt)[:n_samples]

        # Create misaligned gravity
        accel_data = np.random.randn(n_samples, 3) * 0.5
        accel_data[:, 2] += STANDARD_GRAVITY_MS2 * 1.5  # Wrong gravity magnitude

        gyro_data = np.random.randn(n_samples, 3) * 0.1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress misalignment warning
            diagnostics = validate_imu_data_quality(
                accel_data, gyro_data, timestamps, sampling_rate
            )

        assert not diagnostics['quality_good']
        assert 'gravity_misaligned' in diagnostics['quality_issues']
        assert diagnostics['gravity_misaligned']

    def test_validate_unreliable_bias(self):
        """Test validation with unreliable bias estimation."""
        n_samples = 1000
        sampling_rate = 1000.0
        dt = 1.0 / sampling_rate
        timestamps = np.arange(0, n_samples * dt, dt)[:n_samples]

        accel_data = np.random.randn(n_samples, 3) * 2.0
        accel_data[:, 2] += STANDARD_GRAVITY_MS2

        # High motion gyro data (no stationary periods)
        gyro_data = np.random.randn(n_samples, 3) * 5.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress bias warning
            diagnostics = validate_imu_data_quality(
                accel_data, gyro_data, timestamps, sampling_rate
            )

        assert not diagnostics['quality_good']
        assert 'unreliable_bias' in diagnostics['quality_issues']
        assert not diagnostics['gyro_bias_reliable']

    def test_validate_timing_gaps(self):
        """Test validation with timing gaps."""
        n_samples = 1000
        sampling_rate = 1000.0
        dt = 1.0 / sampling_rate

        # Create timestamps with large gap
        timestamps = np.arange(0, n_samples * dt, dt)[:n_samples]
        timestamps[500:] += 10.0  # Large gap

        accel_data = np.random.randn(n_samples, 3) * 2.0
        gyro_data = np.random.randn(n_samples, 3) * 0.5

        diagnostics = validate_imu_data_quality(
            accel_data, gyro_data, timestamps, sampling_rate
        )

        assert not diagnostics['quality_good']
        assert 'timing_gaps' in diagnostics['quality_issues']
        assert diagnostics['timing_gaps_count'] > 0
        assert diagnostics['max_gap_s'] > 5.0

    def test_validate_comprehensive_diagnostics(self):
        """Test that all diagnostic fields are present."""
        n_samples = 100
        sampling_rate = 1000.0
        dt = 1.0 / sampling_rate
        timestamps = np.arange(0, n_samples * dt, dt)[:n_samples]

        accel_data = np.random.randn(n_samples, 3) * 2.0
        gyro_data = np.random.randn(n_samples, 3) * 0.5

        diagnostics = validate_imu_data_quality(
            accel_data, gyro_data, timestamps, sampling_rate
        )

        # Check that all expected fields are present
        expected_fields = [
            'n_samples', 'duration_s', 'timing_error_percent', 'timing_gaps_count',
            'max_gap_s', 'accel_range_ms2', 'gyro_range_rad_s', 'accel_saturated',
            'gyro_saturated', 'gravity_misaligned', 'gravity_error_percent',
            'gyro_bias_rad_s', 'gyro_bias_reliable', 'quality_issues', 'quality_good'
        ]

        for field in expected_fields:
            assert field in diagnostics, f"Missing diagnostic field: {field}"
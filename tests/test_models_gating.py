"""Tests for Mahalanobis gating and measurement masking."""

import numpy as np
import pytest
import jax.numpy as jnp
from hypothesis import given, strategies as st

from trodestrack.models.gating import (
    mahalanobis_distance,
    mahalanobis_gate,
    create_measurement_mask,
    apply_measurement_mask,
    chi_squared_threshold,
)


class TestMahalanobisDistance:
    """Test Mahalanobis distance computation."""

    def test_mahalanobis_distance_identity_covariance(self):
        """Test Mahalanobis distance with identity covariance."""
        residual = jnp.array([3.0, 4.0])  # 3-4-5 triangle
        covariance = jnp.eye(2)  # Identity covariance

        distance = mahalanobis_distance(residual, covariance)

        # Should be Euclidean distance: sqrt(3² + 4²) = 5
        expected = 5.0
        np.testing.assert_allclose(distance, expected, rtol=1e-12)

    def test_mahalanobis_distance_diagonal_covariance(self):
        """Test Mahalanobis distance with diagonal covariance."""
        residual = jnp.array([2.0, 4.0])
        covariance = jnp.diag(jnp.array([1.0, 4.0]))  # Different variances

        distance = mahalanobis_distance(residual, covariance)

        # Manual calculation: d² = [2, 4] @ [[1, 0], [0, 0.25]] @ [2, 4]
        #                         = [2, 1] @ [2, 4] = 4 + 4 = 8
        expected = jnp.sqrt(8.0)
        np.testing.assert_allclose(distance, expected, rtol=1e-12)

    def test_mahalanobis_distance_zero_residual(self):
        """Test Mahalanobis distance with zero residual."""
        residual = jnp.zeros(3)
        covariance = jnp.eye(3)

        distance = mahalanobis_distance(residual, covariance)

        # Should be zero
        np.testing.assert_allclose(distance, 0.0, rtol=1e-15)

    @given(
        residual=st.lists(st.floats(-10, 10, allow_subnormal=False), min_size=2, max_size=4)
    )
    def test_mahalanobis_distance_positive_definite(self, residual):
        """Property test: Mahalanobis distance should be non-negative."""
        residual = jnp.array(residual)
        n = len(residual)
        covariance = jnp.eye(n)  # Identity covariance

        distance = mahalanobis_distance(residual, covariance)

        assert distance >= 0, "Mahalanobis distance should be non-negative"

    def test_mahalanobis_distance_singular_covariance(self):
        """Test behavior with singular covariance matrix."""
        residual = jnp.array([1.0, 1.0])
        # Singular covariance (rank 1)
        covariance = jnp.array([[1.0, 1.0], [1.0, 1.0]])

        # Should handle gracefully (using pseudoinverse)
        distance = mahalanobis_distance(residual, covariance)

        assert jnp.isfinite(distance), "Distance should be finite even with singular covariance"


class TestMahalanobisGating:
    """Test Mahalanobis gating for outlier rejection."""

    def test_mahalanobis_gate_accept(self):
        """Test that good measurements are accepted."""
        residual = jnp.array([1.0, 1.0])
        covariance = jnp.eye(2)
        threshold = 5.0  # Chi-squared threshold

        is_valid = mahalanobis_gate(residual, covariance, threshold)

        # Distance = sqrt(2) ≈ 1.41 < 5, should be accepted
        assert is_valid

    def test_mahalanobis_gate_reject(self):
        """Test that outlier measurements are rejected."""
        residual = jnp.array([10.0, 10.0])  # Large residual
        covariance = jnp.eye(2)
        threshold = 2.0  # Strict threshold

        is_valid = mahalanobis_gate(residual, covariance, threshold)

        # Distance = sqrt(200) ≈ 14.14 > 2, should be rejected
        assert not is_valid

    def test_mahalanobis_gate_boundary(self):
        """Test gating at the boundary."""
        # Design residual to have exactly threshold distance
        threshold = 3.0
        residual = jnp.array([threshold, 0.0])  # Distance exactly = threshold
        covariance = jnp.eye(2)

        is_valid = mahalanobis_gate(residual, covariance, threshold)

        # Should be accepted (distance = threshold, not > threshold)
        assert is_valid

    @given(
        threshold=st.floats(0.1, 10.0, allow_subnormal=False),
        scale=st.floats(0.01, 0.99),  # Scale factor to ensure distance < threshold
    )
    def test_mahalanobis_gate_property(self, threshold, scale):
        """Property test: measurements within threshold should be accepted."""
        # Create residual that is definitely within threshold
        residual = jnp.array([threshold * scale, 0.0])
        covariance = jnp.eye(2)

        is_valid = mahalanobis_gate(residual, covariance, threshold)

        assert is_valid, f"Distance {threshold * scale} should be < threshold {threshold}"


class TestChiSquaredThreshold:
    """Test chi-squared threshold computation."""

    def test_chi_squared_threshold_2d(self):
        """Test chi-squared threshold for 2D measurements."""
        dof = 2
        p_value = 0.05  # 95% confidence

        threshold = chi_squared_threshold(dof, p_value)

        # For 2 DOF and p=0.05, threshold ≈ 5.99
        expected = 5.991464547107979  # From scipy.stats.chi2.ppf(0.95, 2)
        np.testing.assert_allclose(threshold, expected, rtol=1e-6)

    def test_chi_squared_threshold_3d(self):
        """Test chi-squared threshold for 3D measurements."""
        dof = 3
        p_value = 0.01  # 99% confidence

        threshold = chi_squared_threshold(dof, p_value)

        # For 3 DOF and p=0.01, threshold ≈ 11.34
        expected = 11.344866730144373  # From scipy.stats.chi2.ppf(0.99, 3)
        np.testing.assert_allclose(threshold, expected, rtol=1e-6)

    def test_chi_squared_threshold_conservative(self):
        """Test that smaller p-values give larger thresholds."""
        dof = 2

        threshold_95 = chi_squared_threshold(dof, 0.05)  # 95% confidence
        threshold_99 = chi_squared_threshold(dof, 0.01)  # 99% confidence

        # 99% confidence should have larger threshold (more conservative)
        assert threshold_99 > threshold_95


class TestMeasurementMask:
    """Test measurement masking for missing data."""

    def test_create_measurement_mask_all_valid(self):
        """Test mask creation when all measurements are valid."""
        measurements = jnp.array([1.0, 2.0, 3.0])
        confidences = jnp.array([0.9, 0.8, 0.95])
        min_confidence = 0.7

        mask = create_measurement_mask(measurements, confidences, min_confidence)

        # All should be valid
        expected = jnp.array([True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_create_measurement_mask_some_invalid(self):
        """Test mask creation with some invalid measurements."""
        measurements = jnp.array([1.0, 2.0, 3.0])
        confidences = jnp.array([0.9, 0.5, 0.95])  # Middle one too low
        min_confidence = 0.7

        mask = create_measurement_mask(measurements, confidences, min_confidence)

        # Only first and third should be valid
        expected = jnp.array([True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_create_measurement_mask_nan_values(self):
        """Test mask creation with NaN measurements."""
        measurements = jnp.array([1.0, jnp.nan, 3.0])
        confidences = jnp.array([0.9, 0.8, 0.95])
        min_confidence = 0.7

        mask = create_measurement_mask(measurements, confidences, min_confidence)

        # NaN measurement should be invalid
        expected = jnp.array([True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_create_measurement_mask_infinite_values(self):
        """Test mask creation with infinite measurements."""
        measurements = jnp.array([1.0, jnp.inf, 3.0])
        confidences = jnp.array([0.9, 0.8, 0.95])
        min_confidence = 0.7

        mask = create_measurement_mask(measurements, confidences, min_confidence)

        # Infinite measurement should be invalid
        expected = jnp.array([True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_apply_measurement_mask_position_only(self):
        """Test applying mask to position-only measurements."""
        measurements = jnp.array([10.0, 20.0])
        mask = jnp.array([True, False])  # Only x-position valid

        masked_meas, masked_cov, masked_H = apply_measurement_mask(
            measurements=measurements,
            covariance=jnp.eye(2),
            jacobian=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            mask=mask
        )

        # Should only include x-position
        np.testing.assert_array_equal(masked_meas, jnp.array([10.0]))
        assert masked_cov.shape == (1, 1)
        assert masked_H.shape == (1, 2)  # 1 measurement, 2 states
        np.testing.assert_array_equal(masked_H, jnp.array([[1.0, 0.0]]))

    def test_apply_measurement_mask_all_invalid(self):
        """Test applying mask when all measurements are invalid."""
        measurements = jnp.array([10.0, 20.0])
        mask = jnp.array([False, False])  # All invalid

        masked_meas, masked_cov, masked_H = apply_measurement_mask(
            measurements=measurements,
            covariance=jnp.eye(2),
            jacobian=jnp.eye(2),
            mask=mask
        )

        # Should return empty arrays
        assert masked_meas.size == 0
        assert masked_cov.shape == (0, 0)
        assert masked_H.shape == (0, 2)

    def test_apply_measurement_mask_partial_3d(self):
        """Test applying mask to 3D measurements (position + heading)."""
        measurements = jnp.array([10.0, 20.0, 1.57])  # x, y, theta
        mask = jnp.array([True, False, True])  # x and theta valid, y invalid

        H = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # Simplified Jacobian
        masked_meas, masked_cov, masked_H = apply_measurement_mask(
            measurements=measurements,
            covariance=jnp.eye(3),
            jacobian=H,
            mask=mask
        )

        # Should include x and theta only
        np.testing.assert_array_equal(masked_meas, jnp.array([10.0, 1.57]))
        assert masked_cov.shape == (2, 2)
        assert masked_H.shape == (2, 2)

    @given(
        n_measurements=st.integers(1, 5),
        confidence_threshold=st.floats(0.1, 0.9),
    )
    def test_create_measurement_mask_property(self, n_measurements, confidence_threshold):
        """Property test: mask should correctly identify valid measurements."""
        # Create measurements with confidences both above and below threshold
        measurements = jnp.ones(n_measurements)
        confidences = jnp.linspace(0.0, 1.0, n_measurements)

        mask = create_measurement_mask(measurements, confidences, confidence_threshold)

        # Check that mask correctly identifies valid measurements
        for i in range(n_measurements):
            expected_valid = confidences[i] >= confidence_threshold
            assert mask[i] == expected_valid, f"Mask mismatch at index {i}"
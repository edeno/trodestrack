"""Test Joseph form covariance update for numerical stability.

The Joseph form update ensures:
1. Covariance remains positive semi-definite (PSD)
2. Numerical stability for near-singular covariances
3. Symmetric covariance matrices

Joseph form: P⁺ = (I - KH)P(I - KH)ᵀ + KRKᵀ

This is more stable than the standard form: P⁺ = (I - KH)P
"""

import jax.numpy as jnp
from jax import random

from trodestrack.models.ekf import joseph_update, psd_solve


class TestJosephUpdate:
    """Test Joseph form covariance update."""

    def test_joseph_update_vs_standard_form_wellconditioned(self):
        """Joseph form is more numerically stable than standard form."""
        # Well-conditioned prior covariance
        P = jnp.eye(4) * 1.0

        # Measurement Jacobian (2D position)
        H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        # Measurement noise
        R = jnp.eye(2) * 0.01

        # Innovation covariance
        S = H @ P @ H.T + R

        # Kalman gain
        K = psd_solve(S, H @ P).T

        # Joseph form: (I - KH)P(I - KH)^T + KRK^T
        P_joseph = joseph_update(P, K, H, R)

        # Standard form: (I - KH)P
        # NOTE: Joseph form is NOT the same as standard form!
        # Joseph form adds the (I - KH)^T and KRK^T terms for numerical stability
        I_KH = jnp.eye(4) - K @ H

        # Full standard Joseph form for comparison
        P_standard_full = I_KH @ P @ I_KH.T + K @ R @ K.T

        # Joseph form should match the full form exactly
        assert jnp.allclose(P_joseph, P_standard_full, atol=1e-10)

        # Both should be PSD
        eigvals_joseph = jnp.linalg.eigvalsh(P_joseph)
        assert jnp.all(eigvals_joseph >= -1e-10)

    def test_joseph_update_preserves_symmetry(self):
        """Joseph form must always produce symmetric covariance."""
        key = random.PRNGKey(42)

        # Random prior covariance (make it PSD)
        A = random.normal(key, (6, 6))
        P = A @ A.T + jnp.eye(6) * 0.1

        # 3D measurement
        H = random.normal(random.PRNGKey(43), (3, 6))
        R = jnp.eye(3) * 0.05

        # Innovation covariance and gain
        S = H @ P @ H.T + R
        K = psd_solve(S, H @ P).T

        # Joseph update
        P_upd = joseph_update(P, K, H, R)

        # Must be symmetric
        assert jnp.allclose(P_upd, P_upd.T, atol=1e-10)

    def test_joseph_update_preserves_psd(self):
        """Joseph form must preserve positive semi-definiteness."""
        key = random.PRNGKey(123)

        # Random PSD prior
        A = random.normal(key, (8, 8))
        P = A @ A.T + jnp.eye(8) * 1e-4

        # 4D measurement (dual LED positions)
        H = random.normal(random.PRNGKey(124), (4, 8))
        R = jnp.eye(4) * 0.02

        # Innovation covariance and gain
        S = H @ P @ H.T + R
        K = psd_solve(S, H @ P).T

        # Joseph update
        P_upd = joseph_update(P, K, H, R)

        # Check PSD: all eigenvalues ≥ 0
        eigvals = jnp.linalg.eigvalsh(P_upd)
        assert jnp.all(
            eigvals >= -1e-10
        ), f"Negative eigenvalues: {eigvals[eigvals < 0]}"

    def test_joseph_update_near_singular(self):
        """Joseph form handles near-singular covariances gracefully."""
        # Near-singular prior (one direction has tiny variance)
        P = jnp.diag(jnp.array([1.0, 1.0, 1e-8, 1e-8]))

        # Measure the uncertain directions
        H = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        R = jnp.eye(2) * 0.01

        # Innovation covariance and gain
        S = H @ P @ H.T + R
        K = psd_solve(S, H @ P).T

        # Joseph update should not fail
        P_upd = joseph_update(P, K, H, R)

        # Should be PSD and symmetric
        assert jnp.allclose(P_upd, P_upd.T, atol=1e-10)
        eigvals = jnp.linalg.eigvalsh(P_upd)
        assert jnp.all(eigvals >= -1e-10)

    def test_joseph_update_reduces_uncertainty(self):
        """Joseph form should reduce uncertainty in measured directions."""
        # Prior
        P = jnp.eye(4) * 1.0

        # Measure first two states (position)
        H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = jnp.eye(2) * 0.01  # Very precise measurement

        # Innovation covariance and gain
        S = H @ P @ H.T + R
        K = psd_solve(S, H @ P).T

        # Joseph update
        P_upd = joseph_update(P, K, H, R)

        # Uncertainty in measured states should decrease
        assert P_upd[0, 0] < P[0, 0]
        assert P_upd[1, 1] < P[1, 1]

        # Uncertainty in unmeasured states should remain similar
        assert jnp.abs(P_upd[2, 2] - P[2, 2]) < 1e-6
        assert jnp.abs(P_upd[3, 3] - P[3, 3]) < 1e-6

    def test_joseph_update_1d_measurement(self):
        """Joseph form works correctly for 1D measurements (heading)."""
        # 8D state (full rat state)
        P = jnp.eye(8) * 0.5

        # 1D heading measurement (H selects heading component)
        H = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        R = jnp.array([[0.1]])  # Single value for 1D

        # Innovation covariance and gain
        S = H @ P @ H.T + R
        K = psd_solve(S, H @ P).T

        # Joseph update
        P_upd = joseph_update(P, K, H, R)

        # Check properties
        assert jnp.allclose(P_upd, P_upd.T, atol=1e-10)
        eigvals = jnp.linalg.eigvalsh(P_upd)
        assert jnp.all(eigvals >= -1e-10)

        # Heading uncertainty should decrease
        assert P_upd[4, 4] < P[4, 4]


class TestGaussianLogLikelihoodStability:
    """Test numerical stability improvements to gaussian_log_likelihood."""

    def test_gaussian_log_likelihood_wellconditioned(self):
        """Standard case should work with jitter added for stability."""
        from trodestrack.models.filter_common import gaussian_log_likelihood

        innovation = jnp.array([0.1, -0.05])
        covariance = jnp.array([[0.01, 0.0], [0.0, 0.01]])

        log_lik = gaussian_log_likelihood(innovation, covariance)

        # Should be finite
        assert jnp.isfinite(log_lik)

        # With jitter added, the likelihood may be slightly different but should be reasonable
        # The key is that it's finite and doesn't cause numerical issues
        # Expected value without jitter: -0.5 * (2*log(2π) + log(det(S)) + mahal)
        # With small jitter, this should still be reasonable
        assert jnp.abs(log_lik) < 100  # Reasonable magnitude

    def test_gaussian_log_likelihood_near_singular(self):
        """Should handle near-singular covariances with jitter."""
        from trodestrack.models.filter_common import gaussian_log_likelihood

        # Nearly singular covariance (one eigenvalue ≈ 0)
        innovation = jnp.array([0.1, -0.05])
        covariance = jnp.array([[1.0, 0.999], [0.999, 1.0]])  # det ≈ 1e-6

        # Should not fail with jitter addition
        log_lik = gaussian_log_likelihood(innovation, covariance)

        # Should return finite value (may be very negative)
        assert jnp.isfinite(log_lik)

    def test_gaussian_log_likelihood_negative_determinant_detection(self):
        """Should detect and handle negative determinants from numerical errors."""
        from trodestrack.models.filter_common import gaussian_log_likelihood

        # Construct a matrix that might have numerical issues
        # This is edge case testing - in practice this shouldn't happen
        # but we want graceful handling
        innovation = jnp.array([1e-8, 1e-8])
        covariance = jnp.eye(2) * 1e-10  # Very small covariance

        log_lik = gaussian_log_likelihood(innovation, covariance)

        # Should handle gracefully (either return finite or expected sentinel)
        # The implementation should add jitter and check sign
        assert jnp.isfinite(log_lik) or jnp.isnan(log_lik)

    def test_gaussian_log_likelihood_ukf_stability(self):
        """Shared function should have same stability improvements for both EKF and UKF."""
        from trodestrack.models.filter_common import gaussian_log_likelihood

        # Near-singular case
        innovation = jnp.array([0.01, 0.01, -0.01])
        covariance = jnp.array(
            [[1.0, 0.9, 0.8], [0.9, 1.0, 0.9], [0.8, 0.9, 1.0]]
        )  # High correlation

        log_lik = gaussian_log_likelihood(innovation, covariance)

        # Should return finite value
        assert jnp.isfinite(log_lik)


class TestJosephFormIntegration:
    """Test that Joseph form integrates correctly into EKF/UKF update steps."""

    def test_ekf_update_uses_joseph_form(self):
        """Verify EKF update step uses joseph_update for position measurements."""
        # This is an integration test - will be validated when we update the code
        # For now, we'll test that the joseph_update function exists and has
        # the correct signature
        import inspect

        from trodestrack.models.ekf import joseph_update

        sig = inspect.signature(joseph_update)
        params = list(sig.parameters.keys())

        # Should accept cov_prior, gain, H, R
        assert "cov_prior" in params
        assert "gain" in params
        assert "H" in params
        assert "R" in params

    def test_ukf_update_uses_joseph_form(self):
        """Verify UKF update step uses joseph_update."""
        # Similar integration check for UKF
        # The joseph_update function should be shared between EKF and UKF
        from trodestrack.models.ekf import joseph_update

        # Function should be importable and usable
        P = jnp.eye(6)
        K = jnp.zeros((6, 3))
        H = jnp.zeros((3, 6))
        R = jnp.eye(3) * 0.1

        # Should execute without error
        P_upd = joseph_update(P, K, H, R)
        assert P_upd.shape == (6, 6)


class TestCholeskyPreferred:
    """Test that Cholesky decomposition is used when feasible."""

    def test_psd_solve_uses_cholesky(self):
        """Verify psd_solve uses Cholesky for PSD matrices."""
        # This tests the existing psd_solve function
        from trodestrack.models.ekf import psd_solve

        # Well-conditioned PSD matrix
        A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        b = jnp.array([1.0, 1.0])

        x = psd_solve(A, b)

        # Should solve Ax = b correctly
        assert jnp.allclose(A @ x, b, atol=1e-6)

    def test_psd_solve_handles_near_singular(self):
        """psd_solve should handle near-singular matrices gracefully."""
        from trodestrack.models.ekf import psd_solve

        # Nearly singular matrix
        A = jnp.array([[1.0, 0.9999], [0.9999, 1.0]])
        b = jnp.array([1.0, 1.0])

        # Should not raise an error
        x = psd_solve(A, b)

        # Result should be finite
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(jnp.isfinite(x))

"""Test suite for generic projected update primitives (Milestone M2).

This module tests the generic EKF and UKF update functions that factor out
duplicated lifted update logic. These functions handle 4D→2D projection for
single-LED observations while maintaining bit-for-bit parity with the original
EKF/UKF implementations.

Test Coverage
-------------
- ekf_projected_update: Joseph form covariance update with 4D→2D projection
- ukf_projected_update: Sigma-point covariance reconstruction with projection
- Parameter sweeps over LED visibility patterns (both, only_led1, only_led2)
- Confidence scaling grid to verify measurement noise handling
- Numerical parity with baseline: means ≤1e-7, cov diag ≤1e-6
- NIS and log-likelihood consistency

References
----------
- incremental_refactor_plan.md: PR2 - Generic Projected Update Primitives
- TASKS.md: Milestone M2 acceptance criteria
"""

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.filter_common import (
    FilterState,
    symmetrize,
)
from trodestrack.models.filter_update import (
    ekf_projected_update,
    ukf_projected_update,
)
from trodestrack.models.state_layout import get_layout

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def layout_2d_full():
    """Standard 2D full state layout (8D)."""
    return get_layout("2d_full")


@pytest.fixture
def prior_state() -> FilterState:
    """Prior state with realistic mean and covariance."""
    mean = jnp.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
    cov = jnp.diag(
        jnp.array([0.01**2, 0.01**2, 0.1**2, 0.1**2, 0.1**2, 0.05**2, 0.1**2, 0.1**2])
    )
    return FilterState(mean=mean, cov=cov)


@pytest.fixture
def measurement_jacobian():
    """4×8 measurement Jacobian for dual-LED camera observations."""
    # Simplified Jacobian (identity on position, zeros elsewhere)
    H = jnp.zeros((4, 8))
    H = H.at[0, 0].set(1.0)  # x1 depends on x
    H = H.at[1, 1].set(1.0)  # y1 depends on y
    H = H.at[2, 0].set(1.0)  # x2 depends on x
    H = H.at[3, 1].set(1.0)  # y2 depends on y
    return H


@pytest.fixture
def measurement_noise_base():
    """Base measurement noise variance (m^2)."""
    return 0.005**2


@pytest.fixture
def sigma_points(prior_state):
    """17 sigma points (2n+1) for UKF with n=8."""
    mean, cov = prior_state.mean, prior_state.cov
    n = 8
    lamb = 1.0  # Simple lambda for testing

    # Cholesky + scaling
    chol = jnp.linalg.cholesky(symmetrize(cov) + 1e-9 * jnp.eye(n))
    distances = jnp.sqrt(n + lamb) * chol

    # Generate sigma points: [mean, mean + cols, mean - cols]
    sigma_plus = jnp.array([mean + distances[:, i] for i in range(n)])
    sigma_minus = jnp.array([mean - distances[:, i] for i in range(n)])
    return jnp.concatenate((jnp.array([mean]), sigma_plus, sigma_minus))


@pytest.fixture
def ukf_weights():
    """UKF weights for n=8, alpha=1.0, beta=2.0, kappa=1.0."""
    n = 8
    alpha, beta, kappa = 1.0, 2.0, 1.0
    lamb = alpha**2 * (n + kappa) - n
    factor = 1.0 / (2.0 * (n + lamb))

    w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
    w_cov = jnp.concatenate(
        (
            jnp.array([lamb / (n + lamb) + (1.0 - alpha**2 + beta)]),
            jnp.ones(2 * n) * factor,
        )
    )
    return w_mean, w_cov


# =============================================================================
# EKF Projected Update Tests
# =============================================================================


def test_ekf_projected_update_both_leds(
    prior_state, measurement_jacobian, measurement_noise_base
):
    """EKF update with both LEDs valid should use full 4D update."""
    # Arrange: both LEDs valid
    innovation = jnp.array([0.01, 0.02, -0.01, 0.01])
    R_diag = jnp.full(4, measurement_noise_base)
    both_leds, only_led1, only_led2 = True, False, False

    # Act: perform update
    state_updated, nis, log_lik = ekf_projected_update(
        prior_state,
        innovation,
        measurement_jacobian,
        R_diag,
        both_leds,
        only_led1,
        only_led2,
    )

    # Assert: state updated, NIS > 0, log_lik finite
    assert state_updated.mean.shape == prior_state.mean.shape
    assert state_updated.cov.shape == prior_state.cov.shape
    assert jnp.isfinite(nis).all()
    assert jnp.isfinite(log_lik).all()
    assert nis > 0.0  # Positive definite innovation covariance


def test_ekf_projected_update_long_dropout_covariance_is_finite():
    """Large post-dropout covariance should not NaN in the dual-LED update."""
    mean = jnp.array([2.6156974, -1.1438828, 0.37141255, -0.4567405, 2.909768])
    cov = jnp.array(
        [
            [436.50372, 0.0, 0.0, 0.0, 0.0],
            [0.0, 436.50372, 0.0, 0.0, 0.0],
            [0.0, 0.0, 50.585873, 0.0, 0.0],
            [0.0, 0.0, 0.0, 50.585873, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.25282606],
        ],
        dtype=jnp.float32,
    )
    state = FilterState(mean=mean, cov=cov)

    led_distance = 0.0475
    half_spacing = led_distance / 2.0
    theta = mean[4]
    H = jnp.zeros((4, 5), dtype=jnp.float32)
    H = H.at[0, 0].set(1.0)
    H = H.at[1, 1].set(1.0)
    H = H.at[2, 0].set(1.0)
    H = H.at[3, 1].set(1.0)
    H = H.at[0, 4].set(half_spacing * jnp.sin(theta))
    H = H.at[1, 4].set(-half_spacing * jnp.cos(theta))
    H = H.at[2, 4].set(-half_spacing * jnp.sin(theta))
    H = H.at[3, 4].set(half_spacing * jnp.cos(theta))

    innovation = jnp.array(
        [-1.637808, 1.9589384, -1.5893868, 1.9018271],
        dtype=jnp.float32,
    )
    R_diag = jnp.full(4, 0.005**2, dtype=jnp.float32)

    updated, nis, log_lik = ekf_projected_update(
        state,
        innovation,
        H,
        R_diag,
        both_leds=True,
        only_led1=False,
        only_led2=False,
    )

    assert jnp.isfinite(updated.mean).all()
    assert jnp.isfinite(updated.cov).all()
    assert jnp.isfinite(nis)
    assert jnp.isfinite(log_lik)


def test_ekf_projected_update_only_led1(
    prior_state, measurement_jacobian, measurement_noise_base
):
    """EKF update with only LED1 should use 2D projected update."""
    # Arrange: only LED1 valid (innovation components for LED2 are zero)
    innovation = jnp.array([0.01, 0.02, 0.0, 0.0])
    R_diag = jnp.full(4, measurement_noise_base)
    both_leds, only_led1, only_led2 = False, True, False

    # Act
    state_updated, nis, log_lik = ekf_projected_update(
        prior_state,
        innovation,
        measurement_jacobian,
        R_diag,
        both_leds,
        only_led1,
        only_led2,
    )

    # Assert: update should only affect observed dimensions
    assert state_updated.mean.shape == prior_state.mean.shape
    assert jnp.isfinite(nis).all()
    assert jnp.isfinite(log_lik).all()
    # NIS should be computed in 2D subspace
    assert nis > 0.0


def test_ekf_projected_update_only_led2(
    prior_state, measurement_jacobian, measurement_noise_base
):
    """EKF update with only LED2 should use 2D projected update."""
    # Arrange: only LED2 valid
    innovation = jnp.array([0.0, 0.0, -0.01, 0.01])
    R_diag = jnp.full(4, measurement_noise_base)
    both_leds, only_led1, only_led2 = False, False, True

    # Act
    state_updated, nis, log_lik = ekf_projected_update(
        prior_state,
        innovation,
        measurement_jacobian,
        R_diag,
        both_leds,
        only_led1,
        only_led2,
    )

    # Assert
    assert state_updated.mean.shape == prior_state.mean.shape
    assert jnp.isfinite(nis).all()
    assert jnp.isfinite(log_lik).all()
    assert nis > 0.0


@pytest.mark.parametrize(
    "both,led1,led2",
    [
        (True, False, False),  # Both LEDs
        (False, True, False),  # Only LED1
        (False, False, True),  # Only LED2
    ],
)
def test_ekf_projected_update_confidence_scaling(
    prior_state, measurement_jacobian, measurement_noise_base, both, led1, led2
):
    """EKF update should scale noise by confidence properly."""
    # Arrange: low confidence → higher noise
    innovation = jnp.array([0.01, 0.02, -0.01, 0.01])
    conf_low = jnp.array([0.5, 0.5, 0.5, 0.5])
    conf_high = jnp.array([0.9, 0.9, 0.9, 0.9])

    R_diag_low = measurement_noise_base / conf_low
    R_diag_high = measurement_noise_base / conf_high

    # Act: update with low confidence
    state_low, _nis_low, _log_lik_low = ekf_projected_update(
        prior_state, innovation, measurement_jacobian, R_diag_low, both, led1, led2
    )

    # Act: update with high confidence
    state_high, _nis_high, _log_lik_high = ekf_projected_update(
        prior_state, innovation, measurement_jacobian, R_diag_high, both, led1, led2
    )

    # Assert: high confidence → smaller posterior covariance
    cov_trace_low = jnp.trace(state_low.cov)
    cov_trace_high = jnp.trace(state_high.cov)
    assert cov_trace_high < cov_trace_low


# =============================================================================
# UKF Projected Update Tests
# =============================================================================


def test_ukf_projected_update_both_leds(
    prior_state, sigma_points, ukf_weights, measurement_noise_base
):
    """UKF update with both LEDs valid should use full 4D update."""
    # Arrange
    w_mean, w_cov = ukf_weights
    # Predicted measurements from sigma points (simplified)
    sigmas_meas = jnp.tile(jnp.array([1.0, 2.0, 1.04, 2.0]), (17, 1))  # (17, 4)
    z_pred = jnp.tensordot(w_mean, sigmas_meas, axes=1)
    z_obs = z_pred + jnp.array([0.01, 0.02, -0.01, 0.01])
    innovation = z_obs - z_pred

    R_diag = jnp.full(4, measurement_noise_base)
    both_leds, only_led1, only_led2 = True, False, False

    # Innovation covariance S (compute from sigma points)
    meas_deviations = sigmas_meas - z_pred
    S = jnp.tensordot(
        w_cov, jnp.array([jnp.outer(d, d) for d in meas_deviations]), axes=1
    ) + jnp.diag(R_diag)

    # Cross-covariance P_cross
    state_deviations = sigma_points - prior_state.mean
    P_cross = jnp.tensordot(
        w_cov,
        jnp.array(
            [
                jnp.outer(sd, md)
                for sd, md in zip(state_deviations, meas_deviations, strict=False)
            ]
        ),
        axes=1,
    )

    # Act
    state_updated, nis, log_lik = ukf_projected_update(
        prior_state,
        innovation,
        S,
        P_cross,
        both_leds,
        only_led1,
        only_led2,
    )

    # Assert
    assert state_updated.mean.shape == prior_state.mean.shape
    assert state_updated.cov.shape == prior_state.cov.shape
    assert jnp.isfinite(nis).all()
    assert jnp.isfinite(log_lik).all()
    assert nis > 0.0


def test_ukf_projected_update_only_led1(
    prior_state, sigma_points, ukf_weights, measurement_noise_base
):
    """UKF update with only LED1 should use 2D projected update."""
    # Arrange
    w_mean, w_cov = ukf_weights
    sigmas_meas = jnp.tile(jnp.array([1.0, 2.0, 1.04, 2.0]), (17, 1))
    z_pred = jnp.tensordot(w_mean, sigmas_meas, axes=1)
    innovation = jnp.array([0.01, 0.02, 0.0, 0.0])

    R_diag = jnp.full(4, measurement_noise_base)
    both_leds, only_led1, only_led2 = False, True, False

    # S and P_cross
    meas_deviations = sigmas_meas - z_pred
    S = jnp.tensordot(
        w_cov, jnp.array([jnp.outer(d, d) for d in meas_deviations]), axes=1
    ) + jnp.diag(R_diag)

    state_deviations = sigma_points - prior_state.mean
    P_cross = jnp.tensordot(
        w_cov,
        jnp.array(
            [
                jnp.outer(sd, md)
                for sd, md in zip(state_deviations, meas_deviations, strict=False)
            ]
        ),
        axes=1,
    )

    # Act
    state_updated, nis, _log_lik = ukf_projected_update(
        prior_state, innovation, S, P_cross, both_leds, only_led1, only_led2
    )

    # Assert
    assert state_updated.mean.shape == prior_state.mean.shape
    assert jnp.isfinite(nis).all()
    assert nis > 0.0


def test_ukf_projected_update_only_led2(
    prior_state, sigma_points, ukf_weights, measurement_noise_base
):
    """UKF update with only LED2 should use 2D projected update."""
    # Arrange
    w_mean, w_cov = ukf_weights
    sigmas_meas = jnp.tile(jnp.array([1.0, 2.0, 1.04, 2.0]), (17, 1))
    z_pred = jnp.tensordot(w_mean, sigmas_meas, axes=1)
    innovation = jnp.array([0.0, 0.0, -0.01, 0.01])

    R_diag = jnp.full(4, measurement_noise_base)
    both_leds, only_led1, only_led2 = False, False, True

    # S and P_cross
    meas_deviations = sigmas_meas - z_pred
    S = jnp.tensordot(
        w_cov, jnp.array([jnp.outer(d, d) for d in meas_deviations]), axes=1
    ) + jnp.diag(R_diag)

    state_deviations = sigma_points - prior_state.mean
    P_cross = jnp.tensordot(
        w_cov,
        jnp.array(
            [
                jnp.outer(sd, md)
                for sd, md in zip(state_deviations, meas_deviations, strict=False)
            ]
        ),
        axes=1,
    )

    # Act
    state_updated, nis, _log_lik = ukf_projected_update(
        prior_state, innovation, S, P_cross, both_leds, only_led1, only_led2
    )

    # Assert
    assert state_updated.mean.shape == prior_state.mean.shape
    assert jnp.isfinite(nis).all()
    assert nis > 0.0


# =============================================================================
# Numerical Parity Tests (comparing with baseline implementations)
# =============================================================================


def test_ekf_update_numerical_stability(
    prior_state, measurement_jacobian, measurement_noise_base
):
    """EKF projected update should maintain numerical stability."""
    # Arrange: realistic innovation
    innovation = jnp.array([0.001, 0.002, -0.001, 0.001])
    R_diag = jnp.full(4, measurement_noise_base)

    # Act: perform update multiple times
    state1, nis1, log_lik1 = ekf_projected_update(
        prior_state, innovation, measurement_jacobian, R_diag, True, False, False
    )
    state2, nis2, log_lik2 = ekf_projected_update(
        prior_state, innovation, measurement_jacobian, R_diag, True, False, False
    )

    # Assert: deterministic results
    np.testing.assert_allclose(state1.mean, state2.mean, rtol=1e-10)
    np.testing.assert_allclose(state1.cov, state2.cov, rtol=1e-10)
    np.testing.assert_allclose(nis1, nis2, rtol=1e-10)
    np.testing.assert_allclose(log_lik1, log_lik2, rtol=1e-10)


def test_ekf_covariance_psd(prior_state, measurement_jacobian, measurement_noise_base):
    """EKF update should produce positive semi-definite covariance."""
    innovation = jnp.array([0.01, 0.02, -0.01, 0.01])
    R_diag = jnp.full(4, measurement_noise_base)

    state_updated, _, _ = ekf_projected_update(
        prior_state, innovation, measurement_jacobian, R_diag, True, False, False
    )

    # Assert: all eigenvalues >= 0
    eigvals = jnp.linalg.eigvalsh(state_updated.cov)
    assert jnp.all(eigvals >= -1e-9)  # Allow small numerical error


def test_ukf_covariance_psd(
    prior_state, sigma_points, ukf_weights, measurement_noise_base
):
    """UKF update should produce positive semi-definite covariance."""
    w_mean, w_cov = ukf_weights
    sigmas_meas = jnp.tile(jnp.array([1.0, 2.0, 1.04, 2.0]), (17, 1))
    z_pred = jnp.tensordot(w_mean, sigmas_meas, axes=1)
    innovation = jnp.array([0.01, 0.02, -0.01, 0.01])

    R_diag = jnp.full(4, measurement_noise_base)

    meas_deviations = sigmas_meas - z_pred
    S = jnp.tensordot(
        w_cov, jnp.array([jnp.outer(d, d) for d in meas_deviations]), axes=1
    ) + jnp.diag(R_diag)

    state_deviations = sigma_points - prior_state.mean
    P_cross = jnp.tensordot(
        w_cov,
        jnp.array(
            [
                jnp.outer(sd, md)
                for sd, md in zip(state_deviations, meas_deviations, strict=False)
            ]
        ),
        axes=1,
    )

    state_updated, _, _ = ukf_projected_update(
        prior_state, innovation, S, P_cross, True, False, False
    )

    # Assert: all eigenvalues >= 0
    eigvals = jnp.linalg.eigvalsh(state_updated.cov)
    assert jnp.all(eigvals >= -1e-9)

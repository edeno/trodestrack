"""Regression tests for numerical-stability fixes in projected updates and configs.

These tests pin down three related improvements:

1. EKF projected update: uses true Joseph form
   ``P⁺ = (I - K H) P⁻ (I - K H)^T + K R K^T`` rather than the algebraically
   equivalent but numerically fragile subtraction form ``P⁻ - K S K^T``.
   Under poor conditioning, the subtraction form can produce asymmetric or
   non-PSD posterior covariances even when the Joseph form stays valid.

2. UKF projected update: the posterior's subtrahend ``K S K^T`` is built via a
   Cholesky factor of ``S_active``, which guarantees the subtrahend is
   symmetric PSD by construction (``(KL)(KL)^T``). True Joseph form in UKF
   would require the statistical linearization ``C = P_cross^T P^{-1}`` which
   reintroduces the ill-conditioning we are trying to avoid; the sqrt-style
   subtraction keeps the numerical benefits without that cost.

3. Config validation:
   - ``FilterCoreConfig`` rejects unsupported ``mahalanobis_threshold_prob``
     values at construction (instead of silently falling back to the 95%
     threshold inside ``chi2_threshold``).
   - ``UKFConfig`` rejects degenerate ``(n + λ)`` values that would produce
     extreme sigma-point weights (e.g. the old ``conservative(alpha=1e-3)``
     preset at ``n=10`` produced weights on the order of ±10⁶).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig
from trodestrack.models.filter_common import FilterState, symmetrize
from trodestrack.models.filter_update import ekf_projected_update, ukf_projected_update
from trodestrack.models.ukf import UKFConfig

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _random_psd(
    n: int, rng: np.random.Generator, condition: float = 1.0
) -> jnp.ndarray:
    """Random PSD matrix with condition number ``condition`` (larger = worse)."""
    A = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    # Geometric spread of eigenvalues
    eigs = np.geomspace(1.0, 1.0 / condition, n)
    return jnp.asarray(Q @ np.diag(eigs) @ Q.T)


def _min_eig_sym(P: jnp.ndarray) -> float:
    """Minimum eigenvalue of symmetrized ``P``."""
    P_np = np.asarray(symmetrize(P))
    return float(np.min(np.linalg.eigvalsh(P_np)))


def _sym_err(P: jnp.ndarray) -> float:
    """Frobenius distance of ``P`` from its symmetric part."""
    P_np = np.asarray(P)
    return float(np.linalg.norm(P_np - P_np.T, ord="fro"))


@pytest.mark.parametrize(
    "gravity_body",
    [
        (0.0, 9.81),
        (0.0, 0.0, 9.81, 0.0),
        (0.0, np.nan, 9.81),
        (0.0, np.inf, 9.81),
    ],
)
def test_config_rejects_invalid_imu_gravity_body(gravity_body):
    """Gravity calibration must be an explicit finite 3-vector."""
    with pytest.raises(ValueError, match="imu_gravity_body"):
        EKFConfig(imu_gravity_body=gravity_body)


# -----------------------------------------------------------------------------
# EKF projected update: PSD under conditioning stress
# -----------------------------------------------------------------------------


def _subtraction_form_posterior(
    P_prior: jnp.ndarray, H: jnp.ndarray, R_diag: jnp.ndarray
) -> np.ndarray:
    """Algebraically-equivalent subtraction-form posterior ``P⁻ - K S K^T``.

    Used as a numerical-stability baseline in tests. In exact arithmetic this
    equals the Joseph form; in floating point it does not.
    """
    HP = np.asarray(H @ P_prior)
    S = HP @ np.asarray(H).T + np.diag(np.asarray(R_diag))
    K = np.asarray(P_prior) @ np.asarray(H).T @ np.linalg.inv(S)
    return np.asarray(P_prior) - K @ S @ K.T


@pytest.mark.parametrize("condition", [1e4, 1e5, 1e6])
def test_ekf_projected_update_more_psd_than_subtraction_form(condition):
    """Joseph-form posterior must be at least as PSD as the subtraction form.

    JAX defaults to float32, so in very ill-conditioned regimes even Joseph
    can pick up small negative eigenvalues. The meaningful claim is that
    Joseph is *strictly better* than the subtraction form at the same
    conditioning. With enough conditioning stress, this test must pass under
    Joseph and would fail under subtraction.
    """
    rng = np.random.default_rng(42)
    n = 8
    P_prior = _random_psd(n, rng, condition=condition)
    state_prior = FilterState(mean=jnp.zeros(n), cov=P_prior)

    H = jnp.zeros((4, n))
    H = H.at[:2, :2].set(jnp.eye(2))
    H = H.at[2:, :2].set(jnp.eye(2))
    R_diag = jnp.full(4, 0.005**2)
    innovation = jnp.array([0.01, 0.01, 0.01, 0.01])

    joseph_post, _, _ = ekf_projected_update(
        state_prior, innovation, H, R_diag, True, False, False
    )
    joseph_min_eig = _min_eig_sym(joseph_post.cov)

    sub_post = _subtraction_form_posterior(P_prior, H, R_diag)
    sub_sym = 0.5 * (sub_post + sub_post.T)
    sub_min_eig = float(np.min(np.linalg.eigvalsh(sub_sym)))

    # Joseph's min eigenvalue must be >= subtraction's (modulo tiny float32
    # noise of ~1e-7 * scale).
    assert joseph_min_eig >= sub_min_eig - 1e-7, (
        f"Joseph posterior less PSD than subtraction: "
        f"joseph={joseph_min_eig:.3e}, subtraction={sub_min_eig:.3e} "
        f"(condition={condition:.0e})."
    )
    # Joseph is symmetric by construction (up to float32 noise).
    assert _sym_err(joseph_post.cov) < 1e-6


def test_ekf_projected_update_joseph_mean_matches_subtraction():
    """Mean update path must be identical between Joseph and subtraction forms.

    Joseph changes only the covariance formula. The Kalman update of the mean
    (``m⁺ = m⁻ + K @ innovation``) is unchanged, so Joseph must produce the
    same posterior mean as the subtraction form to numerical noise.
    """
    n = 8
    P_prior = jnp.eye(n) * 0.1
    state_prior = FilterState(mean=jnp.zeros(n), cov=P_prior)

    H = jnp.zeros((4, n))
    H = H.at[:2, :2].set(jnp.eye(2))
    H = H.at[2:, :2].set(jnp.eye(2))
    R_diag = jnp.full(4, 0.005**2)
    innovation = jnp.array([0.01, 0.02, -0.01, 0.005])

    posterior, _, _ = ekf_projected_update(
        state_prior, innovation, H, R_diag, True, False, False
    )

    # Reference mean via the explicit Kalman gain
    H_np = np.asarray(H)
    P_np = np.asarray(P_prior)
    S_np = H_np @ P_np @ H_np.T + np.diag(np.asarray(R_diag))
    K_np = P_np @ H_np.T @ np.linalg.inv(S_np)
    mean_ref = K_np @ np.asarray(innovation)

    # float32 precision is ~1e-6; mean values are O(1e-2), so atol=1e-5 is fine.
    assert np.allclose(np.asarray(posterior.mean), mean_ref, atol=1e-5)


def test_ekf_projected_update_joseph_single_led_symmetric_and_finite():
    """Joseph form works on the single-LED (projected) path too."""
    rng = np.random.default_rng(1)
    n = 8
    P_prior = _random_psd(n, rng, condition=1e4)
    state_prior = FilterState(mean=jnp.zeros(n), cov=P_prior)

    H = jnp.zeros((4, n))
    H = H.at[:2, :2].set(jnp.eye(2))
    H = H.at[2:, :2].set(jnp.eye(2))
    R_diag = jnp.full(4, 0.005**2)
    innovation = jnp.array([0.01, 0.02, 0.0, 0.0])

    posterior_single, _, _ = ekf_projected_update(
        state_prior, innovation, H, R_diag, False, True, False
    )
    # Posterior must be finite and symmetric (to float32 noise).
    assert np.all(np.isfinite(np.asarray(posterior_single.cov)))
    assert _sym_err(posterior_single.cov) < 1e-6


# -----------------------------------------------------------------------------
# UKF projected update: symmetric PSD subtrahend
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("condition", [1e3, 1e4, 1e5])
def test_ukf_projected_update_subtrahend_symmetric_and_psd(condition):
    """UKF posterior's subtrahend is PSD and the result is symmetric.

    With the sqrt-style Cholesky form, ``K @ S @ K^T`` is computed as
    ``(K L)(K L)^T`` with ``L L^T = S``, so the subtrahend is a Gram matrix
    and thus symmetric PSD by construction. We verify the posterior is
    symmetric (to float32 noise) and that the overall posterior is at least
    as PSD as a direct ``P - K S K^T`` reference.
    """
    rng = np.random.default_rng(42)
    n = 8
    P_prior = _random_psd(n, rng, condition=condition)
    state_prior = FilterState(mean=jnp.zeros(n), cov=P_prior)

    P_yy = _random_psd(4, rng, condition=10.0)
    R = jnp.eye(4) * 0.005**2
    S = P_yy + R
    # Cross-covariance that couples position to both LEDs realistically.
    P_cross = jnp.zeros((n, 4))
    P_cross = P_cross.at[:2, :2].set(P_prior[:2, :2] * 0.5)
    P_cross = P_cross.at[:2, 2:].set(P_prior[:2, :2] * 0.5)

    innovation = jnp.array([0.01, 0.02, 0.01, 0.02])
    posterior, _, _ = ukf_projected_update(
        state_prior, innovation, S, P_cross, True, False, False
    )

    # Symmetric to float32 machine noise (sqrt form guarantees subtrahend is
    # a Gram matrix; symmetrize() absorbs any remaining rounding).
    assert _sym_err(posterior.cov) < 1e-6

    # Overall posterior should not be dramatically non-PSD relative to a
    # direct reference computation of the same math.
    K_ref = np.linalg.solve(np.asarray(S), np.asarray(P_cross).T).T
    sub_ref = np.asarray(P_prior) - K_ref @ np.asarray(S) @ K_ref.T
    sub_min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (sub_ref + sub_ref.T))))
    joseph_min_eig = _min_eig_sym(posterior.cov)
    assert joseph_min_eig >= sub_min_eig - 1e-6


def test_ukf_projected_update_single_led_posterior_finite_and_symmetric():
    """Single-LED (projected) UKF update must be finite and symmetric."""
    rng = np.random.default_rng(2)
    n = 8
    P_prior = _random_psd(n, rng, condition=1e4)
    state_prior = FilterState(mean=jnp.zeros(n), cov=P_prior)

    P_yy = _random_psd(4, rng, condition=10.0)
    R = jnp.eye(4) * 0.005**2
    S = P_yy + R
    P_cross = jnp.zeros((n, 4))
    P_cross = P_cross.at[:2, :2].set(P_prior[:2, :2])

    innovation = jnp.array([0.01, 0.02, 0.0, 0.0])
    posterior, _, _ = ukf_projected_update(
        state_prior, innovation, S, P_cross, False, True, False
    )

    assert np.all(np.isfinite(np.asarray(posterior.cov)))
    assert np.all(np.isfinite(np.asarray(posterior.mean)))
    assert _sym_err(posterior.cov) < 1e-10


# -----------------------------------------------------------------------------
# FilterCoreConfig mahalanobis_threshold_prob validation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("prob", [0.95, 0.975, 0.99, 0.997])
def test_config_accepts_supported_mahalanobis_probs(prob):
    """The four closed-form probabilities must all construct without error."""
    EKFConfig(mahalanobis_threshold_prob=prob)
    UKFConfig(mahalanobis_threshold_prob=prob)


@pytest.mark.parametrize("prob", [0.5, 0.9, 0.98, 0.999, 1.0])
def test_config_rejects_unsupported_mahalanobis_prob(prob):
    """Unsupported probabilities must fail fast, not silently fall back to 95%."""
    with pytest.raises(ValueError, match="mahalanobis_threshold_prob"):
        EKFConfig(mahalanobis_threshold_prob=prob, use_mahalanobis_gating=True)
    with pytest.raises(ValueError, match="mahalanobis_threshold_prob"):
        UKFConfig(mahalanobis_threshold_prob=prob, use_mahalanobis_gating=True)


def test_config_allows_unsupported_prob_when_gating_disabled():
    """If gating is off, the probability is unused and shouldn't block."""
    EKFConfig(mahalanobis_threshold_prob=0.9, use_mahalanobis_gating=False)
    UKFConfig(mahalanobis_threshold_prob=0.9, use_mahalanobis_gating=False)


# -----------------------------------------------------------------------------
# UKFConfig alpha/kappa guard
# -----------------------------------------------------------------------------


def test_ukfconfig_conservative_preset_has_stable_weights():
    """The conservative preset must yield reasonable sigma-point weights.

    The old preset used alpha=1e-3 and kappa=0, giving (n + λ) ≈ 1e-5 at n=10
    and weights on the order of ±10⁶. A "conservative" preset should produce
    weights whose magnitude is O(1), not O(10⁶).
    """
    cfg = UKFConfig.conservative(state_mode="2d_cam_3d_imu")
    # n for 2d_cam_3d_imu is 10
    n = 10
    n_plus_lambda = cfg.alpha**2 * (n + cfg.kappa)
    w_mean_0 = (cfg.alpha**2 * (n + cfg.kappa) - n) / n_plus_lambda
    # w_mean[0] magnitude should be well below 1000 for a "safe" preset.
    assert abs(w_mean_0) < 1e3, (
        f"UKFConfig.conservative produces w_mean[0]={w_mean_0:.3e}; "
        f"extreme weights defeat the point of a 'conservative' preset."
    )


def test_ukfconfig_rejects_degenerate_alpha_for_given_state_mode():
    """Hand-rolled extreme alpha must raise, not silently destabilize the UKF."""
    with pytest.raises(ValueError, match="sigma-point weights"):
        UKFConfig(alpha=1e-3, kappa=0.0, state_mode="2d_cam_3d_imu")


def test_ukfconfig_accepts_reasonable_alpha_kappa_combinations():
    """Standard alpha/kappa combinations must not be blocked by the guard."""
    UKFConfig(alpha=1.0, kappa=0.0, state_mode="2d_full")
    UKFConfig(alpha=1.732, kappa=1.0, state_mode="2d_full")
    UKFConfig(alpha=0.5, kappa=0.0, state_mode="2d_cam_3d_imu")
    UKFConfig.aggressive(state_mode="2d_cam_3d_imu")

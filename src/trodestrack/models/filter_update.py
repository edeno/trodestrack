"""Generic projected update primitives for EKF and UKF.

This module implements reusable update functions for EKF and UKF implementations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from trodestrack.models.filter_common import (
    FilterState,
    apply_lifted_inverse,
    compute_nis_and_loglik,
    make_led_selector,
    psd_solve,
    symmetrize,
)


def ekf_projected_update(
    state_prior: FilterState,
    innovation: jnp.ndarray,
    jacobian_H: jnp.ndarray,
    R_diag: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> tuple[FilterState, jnp.ndarray, jnp.ndarray]:
    """EKF measurement update with 4D→2D projection for single-LED observations.

    Implements the Joseph-form covariance update with lifted inverse operator to
    handle partial observations (single LED) without inflating measurement noise.
    Uses exact 2D/4D mathematics based on LED validity flags.

    Parameters
    ----------
    state_prior : FilterState
        Prior state with mean (n,) and covariance (n, n).
    innovation : jnp.ndarray
        Innovation vector (4,) in full measurement space [x1, y1, x2, y2] (m).
        Invalid components should be zero-filled to avoid NaN propagation.
    jacobian_H : jnp.ndarray
        Measurement Jacobian (4, n) evaluated at prior mean.
    R_diag : jnp.ndarray
        Diagonal entries of measurement noise covariance (4,) in m^2.
        May be confidence-scaled per dimension.
    both_leds : bool
        True if both LEDs valid (4D update path).
    only_led1 : bool
        True if only LED1 valid (2D update via projection).
    only_led2 : bool
        True if only LED2 valid (2D update via projection).

    Returns
    -------
    state_posterior : FilterState
        Posterior state with updated mean and covariance.
    nis : jnp.ndarray
        Normalized Innovation Squared (scalar) in active subspace.
    log_likelihood : jnp.ndarray
        Gaussian log-likelihood (scalar) for this measurement.

    Notes
    -----
    **Algorithm:**

    1. Compute innovation covariance: S = H P H^T + R
    2. Apply lifted inverse: v = S_eff^{-1} @ innovation
       - 4D path: v = S^{-1} @ innovation (both LEDs)
       - 2D path: v = M^T (M S M^T)^{-1} (M @ innovation) (single LED)
    3. Update mean: m⁺ = m⁻ + (P H^T) @ v
    4. Joseph-form covariance: P⁺ = (I - K H_eff) P⁻ (I - K H_eff)^T + K R_eff K^T
       where (K, H_eff, R_eff) live in the active measurement subspace (4D if
       both LEDs are valid, 2D subspace via selector M otherwise).
    5. Compute NIS and log-likelihood in active subspace (2D or 4D)

    **Numerical Stability:**

    - Joseph form: covariance update is the sum of two structurally PSD terms
      (congruence of PSD prior + Gram of gain @ chol(R)), so the posterior
      stays PSD even when the prior is ill-conditioned or the innovation
      covariance is nearly singular. The equivalent subtraction form
      ``P - K S K^T`` can produce a small negative eigenvalue under the same
      conditions because floating-point cancellation propagates unchecked.
    - Single-LED case projects H and R into the 2D active subspace before
      applying Joseph so the closed-loop ``I - K H_eff`` has the right rank.
    - Symmetrization after covariance update to absorb any remaining
      floating-point asymmetry.

    **Parity Targets:**

    - Mean difference ≤ 1e-7
    - Covariance diagonal difference ≤ 1e-6
    - NIS and log-likelihood match within floating-point precision

    Examples
    --------
    >>> from trodestrack.models.filter_common import FilterState
    >>> import jax.numpy as jnp
    >>> # Prior state (8D)
    >>> mean_prior = jnp.zeros(8)
    >>> cov_prior = jnp.eye(8) * 0.01**2
    >>> state_prior = FilterState(mean=mean_prior, cov=cov_prior)
    >>>
    >>> # Measurement (4D camera)
    >>> innovation = jnp.array([0.01, 0.02, -0.01, 0.01])
    >>> H = jnp.zeros((4, 8))
    >>> H = H.at[:2, :2].set(jnp.eye(2))  # LED1 observes position
    >>> H = H.at[2:, :2].set(jnp.eye(2))  # LED2 observes position
    >>> R_diag = jnp.full(4, 0.005**2)
    >>>
    >>> # Update with both LEDs
    >>> state_post, nis, log_lik = ekf_projected_update(
    ...     state_prior, innovation, H, R_diag, True, False, False
    ... )
    >>> state_post.mean.shape
    (8,)
    >>> state_post.cov.shape
    (8, 8)
    """
    m_prior, P_prior = state_prior.mean, state_prior.cov
    n = P_prior.shape[0]

    # Innovation covariance S (always 4×4)
    HP = jacobian_H @ P_prior  # (4, n)
    R4 = jnp.diag(R_diag)  # (4, 4)
    S4 = HP @ jacobian_H.T + R4  # (4, 4)

    # Lifted inverse: v = S_eff^{-1} @ innovation
    # Automatically handles 2D/4D based on LED validity. Used for mean update
    # only; the Joseph covariance update works directly with (K, H_eff, R_eff)
    # in the active subspace.
    v = apply_lifted_inverse(S4, innovation, both_leds, only_led1, only_led2)

    # Kalman update (mean): δx = (P H^T) @ v
    PH_t = P_prior @ jacobian_H.T  # (n, 4)
    m_posterior = m_prior + PH_t @ v

    # Joseph form of the covariance update. Two branches:
    #   - Both LEDs valid: work in the full 4-D measurement space.
    #   - Single LED valid: project through selector M into 2-D subspace so
    #     that (I - K H_sub) has the correct rank-2 deflation.
    identity = jnp.eye(n)

    def joseph_4d() -> jnp.ndarray:
        K4 = psd_solve(S4, PH_t.T).T  # (n, 4)
        closed_loop = identity - K4 @ jacobian_H
        return closed_loop @ P_prior @ closed_loop.T + K4 @ R4 @ K4.T

    def joseph_2d() -> jnp.ndarray:
        M = make_led_selector(only_led1, only_led2)  # (2, 4)
        H_sub = M @ jacobian_H  # (2, n)
        R_sub = M @ R4 @ M.T  # (2, 2)
        S_sub = M @ S4 @ M.T  # (2, 2)
        PH_sub = PH_t @ M.T  # (n, 2)
        K_sub = psd_solve(S_sub, PH_sub.T).T  # (n, 2)
        closed_loop = identity - K_sub @ H_sub
        return closed_loop @ P_prior @ closed_loop.T + K_sub @ R_sub @ K_sub.T

    P_posterior = lax.cond(both_leds, joseph_4d, joseph_2d)
    P_posterior = symmetrize(P_posterior)

    # Compute exact NIS and log-likelihood in active subspace
    nis, log_likelihood = compute_nis_and_loglik(
        innovation, S4, both_leds, only_led1, only_led2
    )

    return FilterState(mean=m_posterior, cov=P_posterior), nis, log_likelihood


def ukf_projected_update(
    state_prior: FilterState,
    innovation: jnp.ndarray,
    S: jnp.ndarray,
    P_cross: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> tuple[FilterState, jnp.ndarray, jnp.ndarray]:
    """UKF measurement update with 4D→2D projection for single-LED observations.

    Implements the unscented Kalman filter update using sigma-point covariance
    reconstruction. Handles partial observations (single LED) via subspace
    projection without inflating measurement noise.

    Parameters
    ----------
    state_prior : FilterState
        Prior state with mean (n,) and covariance (n, n).
    innovation : jnp.ndarray
        Innovation vector (4,) in full measurement space [x1, y1, x2, y2] (m).
        Invalid components should be zero-filled.
    S : jnp.ndarray
        Innovation covariance (4, 4) in m^2, reconstructed from sigma points.
    P_cross : jnp.ndarray
        Cross-covariance between state and measurements (n, 4).
        Computed as Σ w_cov[i] * (state_dev[i] ⊗ meas_dev[i]).
    both_leds : bool
        True if both LEDs valid (4D update path).
    only_led1 : bool
        True if only LED1 valid (2D update via projection).
    only_led2 : bool
        True if only LED2 valid (2D update via projection).

    Returns
    -------
    state_posterior : FilterState
        Posterior state with updated mean and covariance.
    nis : jnp.ndarray
        Normalized Innovation Squared (scalar) in active subspace.
    log_likelihood : jnp.ndarray
        Gaussian log-likelihood (scalar) for this measurement.

    Notes
    -----
    **Algorithm:**

    1. **Kalman gain computation:**
       - 4D path: K = P_cross @ S^{-1}
       - 2D path: K_sub = P_cross_sub @ S_sub^{-1}, then lift to 4D via M^T
    2. **Mean update:** m⁺ = m⁻ + K @ innovation_active
    3. **Covariance update:** P⁺ = P⁻ - (K L)(K L)^T with L L^T = S_active
       (Cholesky of the active-subspace innovation covariance). This is the
       square-root style equivalent of ``P⁻ - K S K^T`` -- algebraically the
       same in exact arithmetic, but the subtrahend is structurally symmetric
       PSD by construction (it is a Gram matrix), so floating-point rounding
       cannot introduce asymmetry into the subtrahend itself.
    4. **NIS and log-likelihood:** Computed in active subspace (2D or 4D)

    **Projection details:**

    - Single LED: Project to 2D subspace using selector matrix M (2, 4)
      - S_sub = M @ S @ M^T (2, 2)
      - P_cross_sub = P_cross @ M^T (n, 2)
      - K_sub = P_cross_sub @ S_sub^{-1} (n, 2)
      - Lift back: K = K_sub @ M (n, 4)

    - Both LEDs: Direct 4D update (no projection)

    **Numerical Stability:**

    - Symmetrization after covariance update
    - All branches return identical shapes for JAX tracing
    - Uses Cholesky-based solves for inversion
    - Unlike the EKF projected update, full Joseph form ``(I − K C) P (I − K C)^T
      + K R K^T`` is not used here because the UKF has no explicit measurement
      Jacobian. The statistical linearization ``C = P_cross^T P^{-1}`` would
      reintroduce an explicit inverse of the prior covariance, partially
      defeating the numerical improvement. The sqrt-style subtraction keeps
      the subtrahend PSD without that cost.

    **Parity Targets:**

    - Mean difference ≤ 1e-7
    - Covariance diagonal difference ≤ 1e-6
    - NIS and log-likelihood match within floating-point precision

    Examples
    --------
    >>> from trodestrack.models.filter_common import FilterState
    >>> import jax.numpy as jnp
    >>> # Prior state (8D)
    >>> mean_prior = jnp.zeros(8)
    >>> cov_prior = jnp.eye(8) * 0.01**2
    >>> state_prior = FilterState(mean=mean_prior, cov=cov_prior)
    >>>
    >>> # Innovation and covariances from sigma-point transform
    >>> innovation = jnp.array([0.01, 0.02, -0.01, 0.01])
    >>> S = jnp.eye(4) * 0.005**2  # Simplified
    >>> P_cross = jnp.zeros((8, 4))
    >>> P_cross = P_cross.at[:2, :2].set(cov_prior[:2, :2])  # Position→LED1
    >>>
    >>> # Update with both LEDs
    >>> state_post, nis, log_lik = ukf_projected_update(
    ...     state_prior, innovation, S, P_cross, True, False, False
    ... )
    >>> state_post.mean.shape
    (8,)
    """
    m_prior, P_prior = state_prior.mean, state_prior.cov

    # Kalman gain and covariance update using lifted subspace operator.
    # Project to active measurement subspace (2D or 4D) to avoid spurious
    # covariance reduction from missing observations:
    #   - Both LEDs: standard 4D update
    #   - Single LED: compute in 2D subspace, lift back to 4D
    # Each branch returns:
    #   K_lifted   (n, 4)  — gain applied to a 4-D lifted innovation
    #   innov_lift (4,)    — lifted innovation (zeros outside active subspace)
    #   subtrahend (n, n)  — structurally PSD Gram matrix (K_active @ L)(K_active @ L)^T
    #                        where L L^T = S_active. Replaces the less stable
    #                        direct product K @ S_active @ K^T.
    M = make_led_selector(only_led1, only_led2)  # (2, 4)
    # Jitter used in both branches to harden Cholesky for near-singular S.
    jitter_4 = 1e-12 * jnp.eye(4)
    jitter_2 = 1e-12 * jnp.eye(2)

    def compute_in_full_space():
        """Both LEDs valid: standard 4D update."""
        K_full = psd_solve(S, P_cross.T).T  # (n, 4)
        L = jnp.linalg.cholesky(symmetrize(S) + jitter_4)  # (4, 4)
        KL = K_full @ L  # (n, 4)
        subtrahend = KL @ KL.T  # (n, n), PSD by construction
        return K_full, innovation, subtrahend

    def compute_in_subspace():
        """Single LED valid: compute in 2D subspace, lift to 4D."""
        S_sub = M @ S @ M.T  # (2, 2)
        P_cross_sub = P_cross @ M.T  # (n, 2)
        innov_sub = M @ innovation  # (2,)

        K_sub = psd_solve(S_sub, P_cross_sub.T).T  # (n, 2)
        L_sub = jnp.linalg.cholesky(symmetrize(S_sub) + jitter_2)  # (2, 2)
        KL_sub = K_sub @ L_sub  # (n, 2)
        subtrahend = KL_sub @ KL_sub.T  # (n, n), PSD by construction

        K_lifted = K_sub @ M  # (n, 4) — only affects active dims
        innov_lifted = M.T @ innov_sub  # (4,)
        return K_lifted, innov_lifted, subtrahend

    K, innov_active, ksk_subtrahend = lax.cond(
        both_leds,
        compute_in_full_space,
        compute_in_subspace,
    )

    # Apply update in full n-dimensional state space
    m_posterior = m_prior + K @ innov_active
    P_posterior = P_prior - ksk_subtrahend
    P_posterior = symmetrize(P_posterior)

    # Compute exact NIS and log-likelihood in active subspace
    nis, log_likelihood = compute_nis_and_loglik(
        innovation, S, both_leds, only_led1, only_led2
    )

    return FilterState(mean=m_posterior, cov=P_posterior), nis, log_likelihood

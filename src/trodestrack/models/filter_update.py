"""Generic projected update primitives for EKF and UKF.

This module implements reusable update functions for EKF and UKF implementations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import vmap

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
    4. Joseph-form covariance: P⁺ = P⁻ - (P H^T) @ S_eff^{-1} @ (H P⁻)
    5. Compute NIS and log-likelihood in active subspace (2D or 4D)

    **Numerical Stability:**

    - Uses vectorized lifted inverse (`vmap`) for covariance update
    - All operations maintain static shapes for JAX tracing
    - Symmetrization after covariance update to enforce PSD

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

    # Innovation covariance S (always 4×4)
    HP = jacobian_H @ P_prior  # (4, n)
    S4 = HP @ jacobian_H.T + jnp.diag(R_diag)  # (4, 4)

    # Lifted inverse: v = S_eff^{-1} @ innovation
    # Automatically handles 2D/4D based on LED validity
    v = apply_lifted_inverse(S4, innovation, both_leds, only_led1, only_led2)

    # Kalman update without forming K explicitly
    # δx = (P H^T) @ v
    PH_t = P_prior @ jacobian_H.T  # (n, 4)
    delta_x = PH_t @ v  # (n,)

    # Update mean
    m_posterior = m_prior + delta_x

    # Joseph-form covariance update using vectorized lifted inverse
    # P⁺ = P⁻ - (P H^T) @ S_eff^{-1} @ (H P⁻)
    # Apply lifted inverse to each column of HP
    # Note: Uses 1e-9 diagonal jitter in psd_solve for numerical stability
    # This balances accuracy vs. robustness for near-singular covariances
    def apply_inv_to_col(col: jnp.ndarray) -> jnp.ndarray:
        """Apply S_eff^{-1} to a column vector (4,) → (4,)."""
        return apply_lifted_inverse(S4, col, both_leds, only_led1, only_led2)

    # Vectorize over columns (axis 1)
    inv_S_HP = vmap(apply_inv_to_col, in_axes=1, out_axes=1)(HP)  # (4, n)

    # Complete Joseph form
    PH_t_inv_S_HP = PH_t @ inv_S_HP  # (n, n)
    P_posterior = P_prior - PH_t_inv_S_HP
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
    3. **Covariance update:** P⁺ = P⁻ - K @ S_active @ K^T
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
    from jax import lax

    m_prior, P_prior = state_prior.mean, state_prior.cov

    # Kalman gain and covariance update using lifted subspace operator
    # Project to active measurement subspace (2D or 4D) to avoid spurious
    # covariance reduction from missing observations
    #
    # Algorithm:
    #   - Both LEDs: standard 4D update
    #   - Single LED: compute in 2D subspace, lift back to 4D

    # Project innovation, covariance, and cross-correlation to active subspace
    M = make_led_selector(only_led1, only_led2)  # (2, 4)

    def compute_in_full_space():
        """Both LEDs valid: standard 4D update."""
        K_full = psd_solve(S, P_cross.T).T  # (n, 4)
        innov_4d = innovation  # (4,)
        return K_full, innov_4d, S

    def compute_in_subspace():
        """Single LED valid: compute in 2D subspace, lift to 4D."""
        # Project to 2D subspace
        S_sub = M @ S @ M.T  # (2, 2)
        P_cross_sub = P_cross @ M.T  # (n, 2)
        innov_sub = M @ innovation  # (2,)

        # Compute gain in subspace
        K_sub = psd_solve(S_sub, P_cross_sub.T).T  # (n, 2)

        # Lift back to 4D (pad with zeros)
        K_lifted = K_sub @ M  # (n, 4) - only affects active dims
        innov_lifted = M.T @ innov_sub  # (4,)
        S_lifted = M.T @ S_sub @ M  # (4, 4) - only active block

        return K_lifted, innov_lifted, S_lifted

    K, innov_active, S_active = lax.cond(
        both_leds,
        compute_in_full_space,
        compute_in_subspace,
    )

    # Apply update in full n-dimensional state space
    m_posterior = m_prior + K @ innov_active
    P_posterior = P_prior - K @ S_active @ K.T
    P_posterior = symmetrize(P_posterior)

    # Compute exact NIS and log-likelihood in active subspace
    nis, log_likelihood = compute_nis_and_loglik(
        innovation, S, both_leds, only_led1, only_led2
    )

    return FilterState(mean=m_posterior, cov=P_posterior), nis, log_likelihood

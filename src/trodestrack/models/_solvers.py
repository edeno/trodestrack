"""JAX-based linear solvers with PSD hygiene."""

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax


def _symmetrize_and_stabilize(A: jnp.ndarray, jitter: float = 1e-12) -> jnp.ndarray:
    """Symmetrize matrix and add jitter for numerical stability.

    Parameters
    ----------
    A : jnp.ndarray
        Matrix to stabilize (typically covariance)
    jitter : float
        Jitter amount to add to diagonal

    Returns
    -------
    jnp.ndarray
        Stabilized symmetric matrix
    """
    A_sym = 0.5 * (A + A.T)
    return A_sym + jitter * jnp.eye(A.shape[0])


def safe_solve(A: jnp.ndarray, b: jnp.ndarray, jitter: float = 1e-12) -> jnp.ndarray:
    """Safely solve Ax = b with PSD hygiene.

    Parameters
    ----------
    A : jnp.ndarray
        Matrix A (assumed to be positive definite)
    b : jnp.ndarray
        Right-hand side b
    jitter : float
        Jitter for numerical stability

    Returns
    -------
    jnp.ndarray
        Solution x
    """
    A_stable = _symmetrize_and_stabilize(A, jitter)
    return jlinalg.solve(A_stable, b)


def safe_cho_solve(A: jnp.ndarray, b: jnp.ndarray, jitter: float = 1e-12) -> jnp.ndarray:
    """Safely solve Ax = b using Cholesky decomposition.

    Parameters
    ----------
    A : jnp.ndarray
        Symmetric positive definite matrix A
    b : jnp.ndarray
        Right-hand side b
    jitter : float
        Jitter for numerical stability

    Returns
    -------
    jnp.ndarray
        Solution x
    """
    A_stable = _symmetrize_and_stabilize(A, jitter)
    L = jnp.linalg.cholesky(A_stable)
    return jlinalg.cho_solve((L, True), b)


def mahalanobis_distance(residual: jnp.ndarray, covariance: jnp.ndarray,
                        jitter: float = 1e-12) -> jnp.ndarray:
    """Compute Mahalanobis distance using safe solve.

    Parameters
    ----------
    residual : jnp.ndarray
        Residual vector
    covariance : jnp.ndarray
        Covariance matrix
    jitter : float
        Jitter for numerical stability

    Returns
    -------
    jnp.ndarray
        Mahalanobis distance squared
    """
    cov_stable = _symmetrize_and_stabilize(covariance, jitter)
    # Solve S^{-1} @ residual instead of computing inverse
    solved = safe_solve(cov_stable, residual, jitter=0.0)  # Already stabilized
    return residual.T @ solved


def kalman_gain(state_cov: jnp.ndarray, H: jnp.ndarray, measurement_cov: jnp.ndarray,
                jitter: float = 1e-12) -> jnp.ndarray:
    """Compute Kalman gain K = P @ H.T @ S^{-1} using safe solve.

    Parameters
    ----------
    state_cov : jnp.ndarray
        State covariance P
    H : jnp.ndarray
        Measurement Jacobian
    measurement_cov : jnp.ndarray
        Measurement covariance R
    jitter : float
        Jitter for numerical stability

    Returns
    -------
    jnp.ndarray
        Kalman gain matrix
    """
    # Innovation covariance: S = H @ P @ H.T + R
    S = H @ state_cov @ H.T + measurement_cov
    S_stable = _symmetrize_and_stabilize(S, jitter)

    # K = P @ H.T @ S^{-1} = (S^{-1} @ H @ P).T
    # Solve S @ X = H @ P, then K = X.T
    HP = H @ state_cov
    return safe_solve(S_stable, HP, jitter=0.0).T
"""Zero-Velocity Update (ZUPT) measurement model.

This module provides a MeasurementModel implementation for Zero-Velocity Updates,
which apply pseudo-measurements constraining velocity to zero when the rat is
nearly stationary. This prevents IMU drift during stationary periods.

References
----------
- PRD.md: Section 6 (Mathematical Model - Robustness)
- incremental_refactor_plan.md: PR4 - ZUPT as First-Class Sensor
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from trodestrack.models.state_layout import StateLayout


class ZUPTModel:
    """Zero-Velocity Update (ZUPT) measurement model.

    ZUPT applies a pseudo-measurement constraining velocity to zero when the
    animal is nearly stationary (velocity magnitude below threshold). This
    prevents IMU-induced velocity drift during stationary periods.

    Parameters
    ----------
    enable_zupt : bool
        Enable ZUPT updates. If False, R is set to 1e6 (gated out).
    velocity_threshold : float
        Speed threshold (m/s) below which ZUPT applies.
    measurement_noise : float
        ZUPT measurement noise variance ((m/s)^2) when active.
    layout : StateLayout
        State index mapping for velocity extraction.
    dtype : jnp.dtype, optional
        Array dtype, default jnp.float32.

    Attributes
    ----------
    meas_dim : int
        Measurement dimension (always 2 for 2D velocity).

    Notes
    -----
    **Gating Logic:**
    - Stationary (v < threshold) AND enabled → R = measurement_noise (small)
    - Moving (v >= threshold) OR disabled → R = 1e6 (gated out)

    **JAX Compatibility:**
    - Uses `lax.select` for branchless gating (JIT-friendly)
    - No Python `if` statements inside JAX-traced functions
    - Pure functions (no mutable state) safe for jax.jit and lax.scan

    **Integration with Filter:**
    - `predict(state_mean)` extracts [vx, vy] from state
    - `meas_cov_from_pred(meas_pred)` computes R based on velocity magnitude (PURE)
    - `innovation(frame_idx, meas_pred)` returns -[vx, vy] (measuring zero velocity)
    - `jacobian(state_mean)` returns velocity selector matrix H (2, n)

    Examples
    --------
    >>> from trodestrack.models.sensors.zupt import ZUPTModel
    >>> from trodestrack.models.state_layout import get_layout
    >>> layout = get_layout("2d_full")
    >>> model = ZUPTModel(
    ...     enable_zupt=True,
    ...     velocity_threshold=0.05,
    ...     measurement_noise=0.01**2,
    ...     layout=layout,
    ... )
    >>> state = jnp.array([0.5, 0.5, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0])
    >>> meas_pred = model.predict(state)  # [0.02, 0.01]
    >>> R = model.meas_cov_from_pred(meas_pred)  # Small R (stationary, pure)
    """

    def __init__(
        self,
        enable_zupt: bool,
        velocity_threshold: float,
        measurement_noise: float,
        layout: StateLayout,
        dtype=jnp.float32,
    ):
        # Validate parameters
        if measurement_noise <= 0:
            raise ValueError(f"measurement_noise must be > 0, got {measurement_noise}")
        if velocity_threshold < 0:
            raise ValueError(f"velocity_threshold must be >= 0, got {velocity_threshold}")

        self.enable_zupt = enable_zupt
        self.velocity_threshold = velocity_threshold
        self.measurement_noise = measurement_noise
        self.layout = layout
        self.dtype = dtype

    @property
    def meas_dim(self) -> int:
        """Measurement dimension (2 for 2D velocity)."""
        return 2

    def predict(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Extract velocity [vx, vy] from state.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean (n,).

        Returns
        -------
        jnp.ndarray
            Velocity prediction (2,) [vx, vy] in m/s.
        """
        vx_idx, vy_idx = self.layout.vel_idx[0], self.layout.vel_idx[1]
        return jnp.array([state_mean[vx_idx], state_mean[vy_idx]], dtype=self.dtype)

    def jacobian(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Return velocity selector matrix H (2, n).

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean (n,).

        Returns
        -------
        jnp.ndarray
            Jacobian H (2, n) selecting [vx, vy] from state.

        Notes
        -----
        H is a sparse matrix with H[0, vx_idx] = 1, H[1, vy_idx] = 1.
        """
        n = state_mean.shape[0]
        vx_idx, vy_idx = self.layout.vel_idx[0], self.layout.vel_idx[1]

        H = jnp.zeros((2, n), dtype=self.dtype)
        H = H.at[0, vx_idx].set(1.0)
        H = H.at[1, vy_idx].set(1.0)
        return H

    def meas_cov_from_pred(self, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Return measurement noise covariance R based on predicted velocity (PURE).

        Parameters
        ----------
        meas_pred : jnp.ndarray
            Predicted velocity (2,) [vx, vy] from `predict()`.

        Returns
        -------
        jnp.ndarray
            Measurement covariance R (2, 2) diagonal matrix.
            - Stationary + enabled: R = diag([measurement_noise, measurement_noise])
            - Moving or disabled: R = diag([1e6, 1e6]) (gated out)

        Notes
        -----
        **Fully pure function:** No mutable state, safe for JAX JIT/scan.

        **Stationarity logic:** Uses strict < (not <=) to avoid flapping.
        Tie goes to "moving" (ZUPT disabled). This prevents rapid on/off
        toggling when velocity hovers exactly at threshold.

        **Usage in filter:**

        ```python
        meas_pred = model.predict(state_mean)
        R = model.meas_cov_from_pred(meas_pred)  # Pure, JIT-safe
        innovation = model.innovation(frame_idx, meas_pred)
        ```
        """
        v_mag = jnp.linalg.norm(meas_pred)

        # Stationarity check (branchless via lax.select)
        is_stationary = (v_mag < self.velocity_threshold) & self.enable_zupt

        R_scalar = lax.select(
            is_stationary,
            jnp.asarray(self.measurement_noise, dtype=self.dtype),
            jnp.asarray(1e6, dtype=self.dtype),
        )

        return jnp.diag(jnp.array([R_scalar, R_scalar], dtype=self.dtype))

    def meas_cov(self, frame_idx: int) -> jnp.ndarray:
        """Protocol-compliant fallback for meas_cov (NOT RECOMMENDED).

        Parameters
        ----------
        frame_idx : int
            Frame index (unused for ZUPT).

        Returns
        -------
        jnp.ndarray
            Always returns large R (1e6) to gate out update.

        Notes
        -----
        **DEPRECATED:** Use `meas_cov_from_pred()` instead for pure, stateless R.

        This method exists only for protocol compliance. Since ZUPT's R depends
        on runtime velocity (not preloaded frame data), calling meas_cov(frame_idx)
        without state information always returns gated-out R = 1e6 * I.

        Filters should call `meas_cov_from_pred(meas_pred)` after `predict()`.
        """
        # Always gate out - caller should use meas_cov_from_pred() instead
        return jnp.diag(jnp.array([1e6, 1e6], dtype=self.dtype))

    def innovation(self, frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Compute innovation (0 - h(x)) for ZUPT.

        Parameters
        ----------
        frame_idx : int
            Frame index (unused for ZUPT, included for protocol compliance).
        meas_pred : jnp.ndarray
            Predicted velocity (2,) [vx, vy] from `predict()`.

        Returns
        -------
        jnp.ndarray
            Innovation (2,) = -[vx, vy] (measuring zero velocity).

        Notes
        -----
        ZUPT measures zero velocity, so innovation = 0 - h(x) = -h(x).
        """
        return -meas_pred

    def subspace(self, frame_idx: int) -> tuple[bool, bool, bool, jnp.ndarray]:
        """Return subspace flags and identity projector.

        Parameters
        ----------
        frame_idx : int
            Frame index (unused for ZUPT, included for protocol compliance).

        Returns
        -------
        both_leds : bool
            Always True (no projection needed for ZUPT).
        only_led1 : bool
            Always False (not applicable for ZUPT).
        only_led2 : bool
            Always False (not applicable for ZUPT).
        projector : jnp.ndarray
            Identity matrix (2, 2) for consistency with protocol.

        Notes
        -----
        ZUPT doesn't use LED projection logic. Returns identity for consistency.
        Filter update primitives check `both_leds=True` → direct 2D update.
        """
        return True, False, False, jnp.eye(2, dtype=self.dtype)

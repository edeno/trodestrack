"""Zero-Velocity Update (ZUPT) measurement model.

This module provides a MeasurementModel implementation for Zero-Velocity Updates,
which apply pseudo-measurements constraining velocity to zero when the rat is
nearly stationary. This prevents IMU drift during stationary periods.

"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from trodestrack.models.state_layout import StateLayout


class ZUPTModel:
    """Zero-Velocity Update (ZUPT) measurement model.

    ZUPT applies a pseudo-measurement constraining velocity to zero when the
    caller has external stationarity evidence. The filter-level caller is
    responsible for measured stationarity detection; this model only supplies
    the zero-velocity measurement.

    Parameters
    ----------
    enable_zupt : bool
        Enable ZUPT updates. If False, R is set to 1e6 (gated out).
    measurement_noise : float
        ZUPT measurement noise variance ((m/s)^2) when active.
    layout : StateLayout
        State index mapping for velocity extraction.
    dtype : jnp.dtype, optional
        Array dtype, default jnp.float32.

    Attributes
    ----------
    meas_dim : int
        Measurement dimension, matching ``len(layout.vel_idx)``.

    Notes
    -----
    **Gating Logic:**
    - Enabled → R = measurement_noise (small)
    - Disabled → R = 1e6 (gated out)

    **JAX Compatibility:**
    - Uses `lax.select` for branchless gating (JIT-friendly)
    - No Python `if` statements inside JAX-traced functions
    - Pure functions (no mutable state) safe for jax.jit and lax.scan

    **Integration with Filter:**
    - `predict(state_mean)` extracts velocity components from state
    - `meas_cov_from_pred(meas_pred)` computes enabled/disabled R (PURE)
    - `innovation(frame_idx, meas_pred)` returns ``-velocity``
    - `jacobian(state_mean)` returns velocity selector matrix H

    Examples
    --------
    >>> from trodestrack.models.sensors.zupt import ZUPTModel
    >>> from trodestrack.models.state_layout import get_layout
    >>> layout = get_layout("2d_full")
    >>> model = ZUPTModel(
    ...     enable_zupt=True,
    ...     measurement_noise=0.01**2,
    ...     layout=layout,
    ... )
    >>> state = jnp.array([0.5, 0.5, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0])
    >>> meas_pred = model.predict(state)  # [0.02, 0.01]
    >>> R = model.meas_cov_from_pred(meas_pred)  # Small R when enabled
    """

    def __init__(
        self,
        enable_zupt: bool,
        measurement_noise: float,
        layout: StateLayout,
        dtype=jnp.float32,
    ):
        # Use np.isfinite explicitly: NaN compares False to both ``<= 0``
        # and ``< 0``, so the bare comparisons accepted NaN previously.
        # NaN measurement_noise produced an all-NaN ZUPT covariance; NaN
        # covariance would poison the innovation covariance and posterior.
        if not np.isfinite(measurement_noise) or measurement_noise <= 0:
            raise ValueError(
                "measurement_noise must be a finite strictly-positive "
                f"variance (m²/s²); got {measurement_noise!r}."
            )
        # Strict-bool validation. ``enable_zupt`` later combines with a
        # JAX predicate via ``&``; a string ``"False"`` (truthy) crashes
        # ``meas_cov_from_pred`` with ``TypeError: unsupported operand
        # type(s) for &: 'jaxlib.xla_extension.ArrayImpl' and 'str'``,
        # while ``0`` / ``1`` silently look like ``False`` / ``True``
        # without going through the documented bool contract.
        if not isinstance(enable_zupt, bool):
            raise ValueError(
                f"enable_zupt must be a Python ``bool`` (True/False); "
                f"got {enable_zupt!r} (type {type(enable_zupt).__name__})."
            )

        self.enable_zupt = enable_zupt
        self.measurement_noise = measurement_noise
        self.layout = layout
        self.dtype = dtype

    @property
    def meas_dim(self) -> int:
        """Measurement dimension, matching the layout velocity dimension."""
        return len(self.layout.vel_idx)

    def predict(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Extract velocity components from state.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean (n,).

        Returns
        -------
        jnp.ndarray
            Velocity prediction, shape ``(len(layout.vel_idx),)`` in m/s.
        """
        vel_idx = jnp.array(self.layout.vel_idx, dtype=jnp.int32)
        return state_mean[vel_idx].astype(self.dtype)

    def jacobian(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Return velocity selector matrix H.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean (n,).

        Returns
        -------
        jnp.ndarray
            Jacobian H selecting all layout velocity components from state.

        Notes
        -----
        H is a sparse matrix with one selector row per velocity index.
        """
        n = state_mean.shape[0]
        H = jnp.zeros((self.meas_dim, n), dtype=self.dtype)
        for row, vel_idx in enumerate(self.layout.vel_idx):
            H = H.at[row, vel_idx].set(1.0)
        return H

    def meas_cov_from_pred(self, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Return measurement noise covariance R for a zero-velocity update.

        Parameters
        ----------
        meas_pred : jnp.ndarray
            Predicted velocity from `predict()`; accepted for protocol
            compatibility but not used for stationarity detection.

        Returns
        -------
        jnp.ndarray
            Diagonal measurement covariance R with one row/column per velocity.
            - Enabled: diagonal entries are ``measurement_noise``.
            - Disabled: diagonal entries are ``1e6`` (gated out).

        Notes
        -----
        **Fully pure function:** No mutable state, safe for JAX JIT/scan.

        **Usage in filter:**

        ```python
        meas_pred = model.predict(state_mean)
        R = model.meas_cov_from_pred(meas_pred)  # Pure, JIT-safe
        innovation = model.innovation(frame_idx, meas_pred)
        ```
        """
        R_scalar = lax.select(
            jnp.asarray(self.enable_zupt, dtype=bool),
            jnp.asarray(self.measurement_noise, dtype=self.dtype),
            jnp.asarray(1e6, dtype=self.dtype),
        )

        return jnp.eye(self.meas_dim, dtype=self.dtype) * R_scalar

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
        return jnp.eye(self.meas_dim, dtype=self.dtype) * jnp.asarray(
            1e6,
            dtype=self.dtype,
        )

    def innovation(self, frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Compute innovation (0 - h(x)) for ZUPT.

        Parameters
        ----------
        frame_idx : int
            Frame index (unused for ZUPT, included for protocol compliance).
        meas_pred : jnp.ndarray
            Predicted velocity from `predict()`.

        Returns
        -------
        jnp.ndarray
            Innovation equal to ``-meas_pred`` (measuring zero velocity).

        Notes
        -----
        ZUPT measures zero velocity, so innovation = 0 - h(x) = -h(x).
        """
        return -meas_pred

    def subspace(self, frame_idx: int) -> tuple[bool, bool, bool, jnp.ndarray]:
        """Return subspace flags and identity selector.

        Parameters
        ----------
        frame_idx : int
            Frame index (unused for ZUPT, included for protocol compliance).

        Returns
        -------
        both_leds : bool
            Always True (no selection needed for ZUPT).
        only_led1 : bool
            Always False (not applicable for ZUPT).
        only_led2 : bool
            Always False (not applicable for ZUPT).
        selector : jnp.ndarray
            Identity matrix for protocol consistency.

        Notes
        -----
        ZUPT does not use LED selection logic. Returns an identity selector
        matching the velocity dimension.
        """
        return True, False, False, jnp.eye(self.meas_dim, dtype=self.dtype)

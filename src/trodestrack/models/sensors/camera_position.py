"""Camera position measurement model wrapping existing dual-LED helpers.

This module implements the MeasurementModel protocol for camera-based position
measurements from dual-LED tracking. It wraps existing filter_common helpers
while providing a JAX-traceable interface via preallocated arrays.

Design
------
- Wraps `measurement_function()` for prediction h(x)
- Wraps `confidence_to_R_diagonal()` for adaptive measurement noise
- Wraps `make_led_selector()` for 2D/4D projection handling
- **Preallocated JAX arrays** for all frame data (JAX-compatible)
- Maintains static 4D shapes for JAX compatibility
- **Projection-only approach:** NaN → meas_pred (zero residual), no R inflation

References
----------
- incremental_refactor_plan.md: PR1 - MeasurementModel Protocol
- filter_common.py: measurement_function, confidence_to_R_diagonal, make_led_selector
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, jacfwd

from trodestrack.models.filter_common import (
    confidence_to_R_diagonal,
    make_led_selector,
    measurement_function,
)
from trodestrack.models.state_layout import StateLayout, get_heading_index


class CameraPositionModel:
    """Dual-LED camera position measurement model with preallocated arrays.

    This model predicts LED positions from state (position + heading) and
    handles partial observations (single LED) via 2D projection matrices.

    Parameters
    ----------
    led_distance : float
        Nominal LED spacing (m).
    measurement_noise_base : float
        Base position measurement noise variance (m²).
    layout : StateLayout
        State index mapping for accessing position/heading components.
    z_led1_all : jnp.ndarray
        LED1 positions (T, 2) [x, y] in meters. Use NaN for invalid frames.
    z_led2_all : jnp.ndarray
        LED2 positions (T, 2) [x, y] in meters. Use NaN for invalid frames.
    conf_all : jnp.ndarray | None, default None
        Confidence scores (T, 4) [x1, y1, x2, y2] in [0, 1]. If None, uniform.
    confidence_clip_min : float, default 1e-2
        Minimum confidence value for noise scaling (avoids division by zero).

    Attributes
    ----------
    z_led1_all : jnp.ndarray
        Preallocated LED1 positions (T, 2).
    z_led2_all : jnp.ndarray
        Preallocated LED2 positions (T, 2).
    conf_all : jnp.ndarray | None
        Preallocated confidence scores (T, 4) or None.

    Methods
    -------
    meas_dim : int
        Measurement dimension (always 4 for dual-LED).
    predict(state_mean)
        Predict dual-LED positions [x1, y1, x2, y2] from state.
    jacobian(state_mean)
        Compute analytic measurement Jacobian H (4, n).
    meas_cov(frame_idx)
        Return confidence-scaled measurement noise R (4, 4).
    innovation(frame_idx, meas_pred)
        Compute innovation z - h(x) with NaN → meas_pred replacement.
    subspace(frame_idx)
        Return LED validity and (2, 4) selector matrix for lifted updates.

    Notes
    -----
    - **JAX-traceable:** Uses preallocated arrays, no Python dict lookups
    - **Static shapes:** Always returns (2, 4) selector, never (4, 4)
    - **Projection-only:** Invalid LED components → zero residual via NaN replacement
    - **Analytic Jacobian:** No AD cost, exact derivatives
    - Confidence scaling: R_i = base / clip(conf_i, min, 1)

    Examples
    --------
    >>> from trodestrack.models.sensors import CameraPositionModel
    >>> from trodestrack.models.state_layout import get_layout
    >>> import jax.numpy as jnp
    >>> layout = get_layout("2d_full")
    >>> # Preallocate arrays for T=100 frames
    >>> z_led1 = jnp.ones((100, 2))
    >>> z_led2 = jnp.ones((100, 2)) * 1.04
    >>> conf = jnp.ones((100, 4)) * 0.9
    >>> model = CameraPositionModel(
    ...     led_distance=0.04,
    ...     measurement_noise_base=0.005**2,
    ...     layout=layout,
    ...     z_led1_all=z_led1,
    ...     z_led2_all=z_led2,
    ...     conf_all=conf,
    ... )
    >>> # Predict from state
    >>> state = jnp.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
    >>> meas_pred = model.predict(state)
    >>> meas_pred.shape
    (4,)
    >>> # Get innovation for frame 0
    >>> innov = model.innovation(frame_idx=0, meas_pred=meas_pred)
    >>> innov.shape
    (4,)
    """

    def __init__(
        self,
        led_distance: float,
        measurement_noise_base: float,
        layout: StateLayout,
        z_led1_all: jnp.ndarray,
        z_led2_all: jnp.ndarray,
        conf_all: jnp.ndarray | None = None,
        confidence_clip_min: float = 1e-2,
    ) -> None:
        """Initialize camera position model with preallocated arrays.

        Parameters
        ----------
        led_distance : float
            Nominal LED spacing (m).
        measurement_noise_base : float
            Base position measurement noise variance (m²).
        layout : StateLayout
            State index mapping.
        z_led1_all : jnp.ndarray
            LED1 positions (T, 2) [x, y] in meters.
        z_led2_all : jnp.ndarray
            LED2 positions (T, 2) [x, y] in meters.
        conf_all : jnp.ndarray | None
            Confidence scores (T, 4) or None for uniform confidence.
        confidence_clip_min : float, default 1e-2
            Minimum confidence for noise scaling.
        """
        self.led_distance = led_distance
        self.measurement_noise_base = measurement_noise_base
        self.layout = layout
        self.confidence_clip_min = confidence_clip_min

        # Preallocated arrays (JAX-traceable)
        self.z_led1_all = z_led1_all
        self.z_led2_all = z_led2_all
        self.conf_all = conf_all

    @property
    def meas_dim(self) -> int:
        """Measurement dimension (always 4 for dual-LED)."""
        return 4

    def predict(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Predict dual-LED positions from state.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean (n,).

        Returns
        -------
        jnp.ndarray
            Predicted measurement (4,) [x1, y1, x2, y2] in meters.
        """
        return measurement_function(state_mean, self.led_distance, self.layout)

    def jacobian(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Compute analytic measurement Jacobian H = ∂h/∂x.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean (n,).

        Returns
        -------
        jnp.ndarray
            Jacobian matrix (4, n).

        Notes
        -----
        **Analytic derivation** (no AD cost):

        Measurement function:
            h(x) = [px - d·cos(θ), py - d·sin(θ), px + d·cos(θ), py + d·sin(θ)]
        where d = led_distance / 2.

        Jacobian:
            ∂h/∂[x, y, vx, vy, θ, ...] =
            [[1, 0, 0, 0,  d·sin(θ), 0, ...],
             [0, 1, 0, 0, -d·cos(θ), 0, ...],
             [1, 0, 0, 0, -d·sin(θ), 0, ...],
             [0, 1, 0, 0,  d·cos(θ), 0, ...]]
        """
        if self.layout.has_quaternion_orientation:
            return jacfwd(
                lambda state: measurement_function(
                    state, self.led_distance, self.layout
                )
            )(state_mean)

        h_idx = get_heading_index(self.layout)
        theta = state_mean[h_idx]
        d = self.led_distance / 2.0

        # Initialize Jacobian (4, n) with zeros
        n = state_mean.shape[0]
        H = jnp.zeros((4, n))

        # Position derivatives (identity for x, y)
        H = H.at[0, self.layout.pos_idx[0]].set(1.0)  # ∂x₁/∂x
        H = H.at[1, self.layout.pos_idx[1]].set(1.0)  # ∂y₁/∂y
        H = H.at[2, self.layout.pos_idx[0]].set(1.0)  # ∂x₂/∂x
        H = H.at[3, self.layout.pos_idx[1]].set(1.0)  # ∂y₂/∂y

        # Heading derivatives
        H = H.at[0, h_idx].set(d * jnp.sin(theta))  # ∂x₁/∂θ
        H = H.at[1, h_idx].set(-d * jnp.cos(theta))  # ∂y₁/∂θ
        H = H.at[2, h_idx].set(-d * jnp.sin(theta))  # ∂x₂/∂θ
        H = H.at[3, h_idx].set(d * jnp.cos(theta))  # ∂y₂/∂θ

        return H

    def meas_cov(self, frame_idx: int) -> jnp.ndarray:
        """Return confidence-scaled measurement noise R.

        Parameters
        ----------
        frame_idx : int
            Frame index.

        Returns
        -------
        jnp.ndarray
            Measurement noise covariance (4, 4), diagonal.

        Notes
        -----
        - R_i = base / clip(conf_i, min, 1) for each dimension
        - If no confidence, R = base * I
        - No R inflation for invalid LEDs (projection-only approach)
        """
        # Extract confidence for this frame (or None)
        confidence = None if self.conf_all is None else self.conf_all[frame_idx]

        # Compute diagonal using shared helper
        R_diag = confidence_to_R_diagonal(
            confidence,
            base=self.measurement_noise_base,
            size=4,
            clip_min=self.confidence_clip_min,
        )

        return jnp.diag(R_diag)

    def innovation(self, frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Compute innovation z - h(x) with NaN replacement (projection-only).

        Parameters
        ----------
        frame_idx : int
            Frame index.
        meas_pred : jnp.ndarray
            Predicted measurement (4,).

        Returns
        -------
        jnp.ndarray
            Innovation (4,) with NaN components replaced by meas_pred (zero residual).

        Notes
        -----
        **Projection-only approach:**
        - Invalid LED components (NaN in z) are replaced with meas_pred
        - This yields zero residual for those components: (meas_pred - meas_pred) = 0
        - No explicit zeroing or R inflation needed
        - The 2D projection will extract only valid components when single LED
        """
        z_led1 = self.z_led1_all[frame_idx]
        z_led2 = self.z_led2_all[frame_idx]

        # Concatenate observations [x1, y1, x2, y2]
        z_obs = jnp.concatenate([z_led1, z_led2])

        # Replace NaN with meas_pred to yield zero residual for invalid components
        z_obs_sanitized = jnp.where(jnp.isfinite(z_obs), z_obs, meas_pred)

        return z_obs_sanitized - meas_pred

    def subspace(self, frame_idx: int) -> tuple[Array, Array, Array, Array]:
        """Return LED validity flags and (2, 4) selector matrix.

        Parameters
        ----------
        frame_idx : int
            Frame index.

        Returns
        -------
        both_leds : Array
            Boolean scalar array. True if both LEDs valid (4D update, selector ignored).
        only_led1 : Array
            Boolean scalar array. True if only LED1 valid (2D update via projection).
        only_led2 : Array
            Boolean scalar array. True if only LED2 valid (2D update via projection).
        selector_M2 : Array
            **Static shape (2, 4)** selector matrix (never (4, 4)).
            Selects active 2D subspace from 4D measurement space.
            For dual-LED, returns conventional LED1 selector (ignored by update).

        Notes
        -----
        **Critical for PR2/PR3 JAX compatibility:**
        - Always returns (2, 4) shape, even for dual-LED case
        - Generic update primitive uses `lax.cond(both_leds, ...)` to choose
          between 4D direct update vs 2D projected update
        - When `both_leds=True`, the selector is not used (update works in 4D)
        - Uses `make_led_selector()` from filter_common for single-LED cases
        """
        z_led1 = self.z_led1_all[frame_idx]
        z_led2 = self.z_led2_all[frame_idx]

        # Determine LED validity
        led1_valid = jnp.isfinite(z_led1).all()
        led2_valid = jnp.isfinite(z_led2).all()

        both_leds = led1_valid & led2_valid
        only_led1 = led1_valid & (~led2_valid)
        only_led2 = (~led1_valid) & led2_valid

        # Invariant: exactly one of {both_leds, only_led1, only_led2} must be True,
        # or all False (no valid LEDs). Sum should be 0 or 1.
        # This ensures valid LED configuration for update logic.

        # Always return (2, 4) selector, even for dual-LED case
        # For dual-LED: return conventional LED1 selector (arbitrary, will be ignored)
        # For single-LED: return appropriate selector
        selector_M2 = make_led_selector(only_led1, only_led2)

        return both_leds, only_led1, only_led2, selector_M2

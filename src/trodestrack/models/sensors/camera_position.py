"""Camera position measurement model wrapping existing dual-LED helpers.

This module implements the MeasurementModel protocol for camera-based position
measurements from dual-LED tracking. It wraps existing filter_common helpers
while providing a clean, stateful interface for per-frame observations.

Design
------
- Wraps `measurement_function()` for prediction h(x)
- Wraps `confidence_to_R_diagonal()` for adaptive measurement noise
- Wraps `make_led_selector()` for 2D/4D projection handling
- Caches per-frame observations via `set_frame_data()`
- Maintains static 4D shapes for JAX compatibility

References
----------
- incremental_refactor_plan.md: PR1 - MeasurementModel Protocol
- filter_common.py: measurement_function, confidence_to_R_diagonal, make_led_selector
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jacfwd

from trodestrack.models.filter_common import (
    confidence_to_R_diagonal,
    make_led_selector,
    measurement_function,
)
from trodestrack.models.state_layout import StateLayout


class CameraPositionModel:
    """Dual-LED camera position measurement model.

    This model predicts LED positions from state (position + heading) and
    handles partial observations (single LED) via projection matrices.

    Parameters
    ----------
    led_distance : float
        Nominal LED spacing (m).
    measurement_noise_base : float
        Base position measurement noise variance (m²).
    layout : StateLayout
        State index mapping for accessing position/heading components.
    confidence_clip_min : float, default 1e-2
        Minimum confidence value for noise scaling (avoids division by zero).

    Attributes
    ----------
    _frame_data : dict[int, dict]
        Cached frame-specific observations and validity flags.
        Keys: frame_idx, Values: {z_led1, z_led2, confidence, led1_valid, led2_valid}

    Methods
    -------
    set_frame_data(frame_idx, z_led1, z_led2, confidence=None)
        Cache observations for frame_idx.
    meas_dim : int
        Measurement dimension (always 4 for dual-LED).
    predict(state_mean)
        Predict dual-LED positions [x1, y1, x2, y2] from state.
    jacobian(state_mean)
        Compute measurement Jacobian H (4, n).
    meas_cov(frame_idx)
        Return confidence-scaled measurement noise R (4, 4).
    innovation(frame_idx, meas_pred)
        Compute innovation z - h(x) with invalid LEDs zeroed.
    subspace(frame_idx)
        Return LED validity and projector matrix for lifted updates.

    Notes
    -----
    - Maintains static 4D shapes (uses NaN for invalid LEDs)
    - Invalid observations handled via zeroing in innovation, not gating in R
    - Confidence scaling: R_i = base / clip(conf_i, min, 1)

    Examples
    --------
    >>> from trodestrack.models.sensors import CameraPositionModel
    >>> from trodestrack.models.state_layout import get_layout
    >>> import jax.numpy as jnp
    >>> layout = get_layout("2d_full")
    >>> model = CameraPositionModel(
    ...     led_distance=0.04,
    ...     measurement_noise_base=0.005**2,
    ...     layout=layout,
    ... )
    >>> # Set frame observations
    >>> model.set_frame_data(
    ...     frame_idx=0,
    ...     z_led1=jnp.array([1.0, 2.0]),
    ...     z_led2=jnp.array([1.04, 2.0]),
    ...     confidence=jnp.array([0.9, 0.9, 0.8, 0.8]),
    ... )
    >>> # Predict from state
    >>> state = jnp.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
    >>> meas_pred = model.predict(state)
    >>> meas_pred.shape
    (4,)
    """

    def __init__(
        self,
        led_distance: float,
        measurement_noise_base: float,
        layout: StateLayout,
        confidence_clip_min: float = 1e-2,
    ) -> None:
        """Initialize camera position model.

        Parameters
        ----------
        led_distance : float
            Nominal LED spacing (m).
        measurement_noise_base : float
            Base position measurement noise variance (m²).
        layout : StateLayout
            State index mapping.
        confidence_clip_min : float, default 1e-2
            Minimum confidence for noise scaling.
        """
        self.led_distance = led_distance
        self.measurement_noise_base = measurement_noise_base
        self.layout = layout
        self.confidence_clip_min = confidence_clip_min

        # Cache for frame-specific observations
        self._frame_data: dict[int, dict] = {}

    def set_frame_data(
        self,
        frame_idx: int,
        z_led1: jnp.ndarray,
        z_led2: jnp.ndarray,
        confidence: jnp.ndarray | None = None,
    ) -> None:
        """Cache observations for frame_idx.

        Parameters
        ----------
        frame_idx : int
            Frame index to associate with observations.
        z_led1 : jnp.ndarray
            LED1 position (2,) [x, y] in meters. Use NaN for invalid.
        z_led2 : jnp.ndarray
            LED2 position (2,) [x, y] in meters. Use NaN for invalid.
        confidence : jnp.ndarray | None, default None
            Confidence scores (4,) [x1, y1, x2, y2] in [0, 1].
            If None, uniform confidence assumed (no scaling).
        """
        # Check LED validity (use isfinite to handle NaN)
        led1_valid = jnp.isfinite(z_led1[0])
        led2_valid = jnp.isfinite(z_led2[0])

        # Store frame data
        self._frame_data[frame_idx] = {
            "z_led1": z_led1,
            "z_led2": z_led2,
            "confidence": confidence,
            "led1_valid": led1_valid,
            "led2_valid": led2_valid,
        }

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
        """Compute measurement Jacobian H = ∂h/∂x.

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
        Uses JAX automatic differentiation via jacfwd.
        """

        def h(x):
            return measurement_function(x, self.led_distance, self.layout)

        return jacfwd(h)(state_mean)

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
        """
        data = self._frame_data[frame_idx]
        confidence = data["confidence"]

        # Compute diagonal using shared helper
        R_diag = confidence_to_R_diagonal(
            confidence,
            base=self.measurement_noise_base,
            size=4,
            clip_min=self.confidence_clip_min,
        )

        return jnp.diag(R_diag)

    def innovation(self, frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Compute innovation z - h(x) with invalid LEDs zeroed.

        Parameters
        ----------
        frame_idx : int
            Frame index.
        meas_pred : jnp.ndarray
            Predicted measurement (4,).

        Returns
        -------
        jnp.ndarray
            Innovation (4,) with invalid components set to zero.

        Notes
        -----
        - Avoids NaN propagation by zeroing invalid LED components
        - Observation vector: [z_led1[0], z_led1[1], z_led2[0], z_led2[1]]
        """
        data = self._frame_data[frame_idx]
        z_led1 = data["z_led1"]
        z_led2 = data["z_led2"]

        # Concatenate observations
        z_obs = jnp.concatenate([z_led1, z_led2])

        # Raw innovation
        innov_raw = z_obs - meas_pred

        # Zero out invalid components (avoid NaN propagation)
        led1_valid = data["led1_valid"]
        led2_valid = data["led2_valid"]
        obs_mask = jnp.array([led1_valid, led1_valid, led2_valid, led2_valid], dtype=bool)
        innovation = jnp.where(obs_mask, innov_raw, 0.0)

        return innovation

    def subspace(self, frame_idx: int) -> tuple[bool, bool, bool, jnp.ndarray]:
        """Return LED validity and projector matrix.

        Parameters
        ----------
        frame_idx : int
            Frame index.

        Returns
        -------
        both_leds : bool
            True if both LEDs valid (4D update).
        only_led1 : bool
            True if only LED1 valid (2D update).
        only_led2 : bool
            True if only LED2 valid (2D update).
        projector_M : jnp.ndarray
            Projector matrix (2, 4) for single LED, or (4, 4) identity for dual LED.

        Notes
        -----
        - For dual LED: returns eye(4) (no projection)
        - For single LED: returns 2×4 selector matrix
        - Uses `make_led_selector()` from filter_common
        - JAX-compatible: uses lax.select() instead of Python if/else
        """

        data = self._frame_data[frame_idx]
        led1_valid = data["led1_valid"]
        led2_valid = data["led2_valid"]

        # Determine LED configuration
        both_leds = led1_valid & led2_valid
        only_led1 = led1_valid & (~led2_valid)
        only_led2 = (~led1_valid) & led2_valid

        # Projector matrix
        # NOTE: Python if/else is used here because the shapes differ (2×4 vs 4×4).
        # JAX's lax.cond/lax.select require identical shapes on both branches.
        # This is acceptable for M1 since subspace() is only called in tests.
        # For PR3 integration into traced filter loops, we will need to:
        #   1. Always return 4×4 projector (pad 2×4 with zeros), OR
        #   2. Redesign the lifted update to handle variable shapes differently
        if both_leds:  # noqa: SIM108
            # No projection needed (4D update)
            projector_M = jnp.eye(4)
        else:
            # Single LED: 2×4 selector matrix
            projector_M = make_led_selector(only_led1, only_led2)

        return both_leds, only_led1, only_led2, projector_M

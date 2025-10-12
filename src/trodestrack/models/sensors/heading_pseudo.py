"""Heading pseudo-measurement model wrapping existing LED geometry helpers.

This module implements the MeasurementModel protocol for heading pseudo-measurements
derived from dual-LED geometry. It wraps existing filter_common helpers while providing
a clean interface for per-frame LED observations and adaptive gating logic.

Design
------
- Wraps `prepare_heading_measurement()` for heading observation and gating
- Uses large-R gating (R=1e6) for invalid observations (JAX-compatible)
- Supports adaptive noise scaling by LED spacing ratio
- Maintains 1D measurement shape for heading angle
- Handles angle wrapping in innovation computation

References
----------
- incremental_refactor_plan.md: PR1 - MeasurementModel Protocol
- filter_common.py: prepare_heading_measurement
- ekf.py: update_heading
"""

from __future__ import annotations

import jax.numpy as jnp

from trodestrack.models.filter_common import (
    FilterCoreConfig,
    prepare_heading_measurement,
    wrap_angle,
)
from trodestrack.models.state_layout import StateLayout, get_heading_index


class HeadingPseudoModel:
    """Heading pseudo-measurement model from dual-LED geometry.

    This model extracts heading from the LED pair vector and applies
    adaptive gating based on LED spacing validity.

    Parameters
    ----------
    config : FilterCoreConfig
        Full filter configuration (heading parameters extracted internally).
    layout : StateLayout
        State index mapping for accessing heading component.

    Attributes
    ----------
    _frame_data : dict[int, dict]
        Cached frame-specific LED observations.
        Keys: frame_idx, Values: {z_led1, z_led2, heading_obs, R_heading, use_heading}

    Methods
    -------
    set_frame_data(frame_idx, z_led1, z_led2)
        Cache LED observations for frame_idx and precompute heading measurement.
    meas_dim : int
        Measurement dimension (always 1 for heading).
    predict(state_mean)
        Extract heading component from state.
    jacobian(state_mean)
        Return 1×n Jacobian selecting heading component.
    meas_cov(frame_idx)
        Return heading measurement noise R (possibly gated).
    innovation(frame_idx, meas_pred)
        Compute angle-wrapped innovation (z_heading - predicted_heading).
    subspace(frame_idx)
        Return identity projector (not applicable for heading).

    Notes
    -----
    - Uses prepare_heading_measurement() for LED spacing validation and gating
    - Invalid observations gated via R=1e6 (no Python branching)
    - Innovation wrapped to [-π, π] via wrap_angle()
    - Accepts full FilterCoreConfig for type compatibility with prepare_heading_measurement()

    Examples
    --------
    >>> from trodestrack.models.sensors import HeadingPseudoModel
    >>> from trodestrack.models.filter_common import FilterCoreConfig
    >>> from trodestrack.models.state_layout import get_layout
    >>> import jax.numpy as jnp
    >>> layout = get_layout("2d_full")
    >>> config = FilterCoreConfig(
    ...     use_heading_measurement=True,
    ...     measurement_noise_heading=0.05**2,
    ...     led_distance=0.04,
    ...     led_distance_tolerance=0.3,
    ...     adaptive_heading_noise=True,
    ... )
    >>> model = HeadingPseudoModel(config=config, layout=layout)
    >>> # Set frame observations
    >>> model.set_frame_data(
    ...     frame_idx=0,
    ...     z_led1=jnp.array([1.0, 2.0]),
    ...     z_led2=jnp.array([1.04, 2.0]),
    ... )
    >>> # Predict from state
    >>> state = jnp.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
    >>> meas_pred = model.predict(state)
    >>> meas_pred.shape
    (1,)
    """

    def __init__(
        self,
        config: FilterCoreConfig,
        layout: StateLayout,
    ) -> None:
        """Initialize heading pseudo-measurement model.

        Parameters
        ----------
        config : FilterCoreConfig
            Full filter configuration. Heading-specific parameters:
            - use_heading_measurement: Enable heading pseudo-measurement
            - measurement_noise_heading: Base heading noise (rad²)
            - led_distance: Expected LED spacing (m)
            - led_distance_tolerance: Relative spacing tolerance (fraction)
            - adaptive_heading_noise: Enable adaptive noise scaling
        layout : StateLayout
            State index mapping.
        """
        self.config = config
        self.layout = layout

        # Cache for frame-specific observations and precomputed measurements
        self._frame_data: dict[int, dict] = {}

    def set_frame_data(
        self,
        frame_idx: int,
        z_led1: jnp.ndarray,
        z_led2: jnp.ndarray,
    ) -> None:
        """Cache LED observations and precompute heading measurement.

        Parameters
        ----------
        frame_idx : int
            Frame index.
        z_led1 : jnp.ndarray
            LED1 position (2,) [x, y] in meters.
        z_led2 : jnp.ndarray
            LED2 position (2,) [x, y] in meters.

        Notes
        -----
        Calls prepare_heading_measurement() to compute:
        - heading_obs: arctan2(dy, dx) in radians
        - R_heading: measurement noise (gated to 1e6 if invalid)
        - use_heading: boolean flag (True if valid)
        """
        # Use existing helper to precompute heading measurement
        heading_obs, R_heading, use_heading = prepare_heading_measurement(
            z_led1, z_led2, self.config
        )

        # Store frame data
        self._frame_data[frame_idx] = {
            "z_led1": z_led1,
            "z_led2": z_led2,
            "heading_obs": heading_obs,
            "R_heading": R_heading,
            "use_heading": use_heading,
        }

    @property
    def meas_dim(self) -> int:
        """Measurement dimension (always 1 for heading)."""
        return 1

    def predict(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Extract heading component from state.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean (n,).

        Returns
        -------
        jnp.ndarray
            Predicted heading (1,) [θ] in radians.

        Notes
        -----
        Measurement function: h(x) = x[heading_idx]
        """
        h_idx = get_heading_index(self.layout)
        return state_mean[h_idx : h_idx + 1]  # Keep as 1D array

    def jacobian(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Return Jacobian selecting heading component.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean (n,).

        Returns
        -------
        jnp.ndarray
            Jacobian (1, n) with 1 at heading index, 0 elsewhere.

        Notes
        -----
        H = [0, 0, 0, 0, 1, 0, 0, 0] for 8D state (heading at index 4).
        """
        n = state_mean.shape[0]
        h_idx = get_heading_index(self.layout)
        H = jnp.zeros((1, n))
        H = H.at[0, h_idx].set(1.0)
        return H

    def meas_cov(self, frame_idx: int) -> jnp.ndarray:
        """Return heading measurement noise R.

        Parameters
        ----------
        frame_idx : int
            Frame index.

        Returns
        -------
        jnp.ndarray
            Measurement noise (1, 1).

        Notes
        -----
        - Valid observations: R ≈ 0.05² (small, enables update)
        - Invalid observations: R = 1e6 (large, gates update)
        - Adaptive scaling: R *= (expected/observed)² if enabled
        """
        data = self._frame_data[frame_idx]
        R_heading = data["R_heading"]
        return jnp.array([[R_heading]])

    def innovation(self, frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Compute angle-wrapped innovation.

        Parameters
        ----------
        frame_idx : int
            Frame index.
        meas_pred : jnp.ndarray
            Predicted heading (1,).

        Returns
        -------
        jnp.ndarray
            Innovation (1,) wrapped to [-π, π].

        Notes
        -----
        - Raw innovation: z_heading - h_pred
        - Wraps result to [-π, π] via wrap_angle()
        - Replaces NaN with 0 (for gated observations)
        """
        data = self._frame_data[frame_idx]
        heading_obs = data["heading_obs"]

        # Raw innovation with angle wrapping
        innov_raw = wrap_angle(heading_obs - meas_pred[0])

        # Replace NaN with 0 (for gated cases)
        innovation = jnp.where(jnp.isfinite(innov_raw), innov_raw, 0.0)

        return jnp.array([innovation])

    def subspace(self, frame_idx: int) -> tuple[bool, bool, bool, jnp.ndarray]:
        """Return identity projector (not applicable for heading).

        Parameters
        ----------
        frame_idx : int
            Frame index.

        Returns
        -------
        both_leds : bool
            False (not applicable).
        only_led1 : bool
            False (not applicable).
        only_led2 : bool
            False (not applicable).
        projector_M : jnp.ndarray
            Identity (1, 1).

        Notes
        -----
        Heading is always 1D, no LED subspace projection needed.
        """
        return False, False, False, jnp.eye(1)

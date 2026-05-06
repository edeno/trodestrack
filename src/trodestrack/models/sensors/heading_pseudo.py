"""Heading pseudo-measurement model wrapping existing LED geometry helpers.

This module implements the MeasurementModel protocol for heading pseudo-measurements
derived from dual-LED geometry. It wraps existing filter_common helpers while providing
a JAX-traceable interface via preallocated arrays.

Design
------
- Wraps `prepare_heading_measurement()` for heading observation and gating
- **Preallocated JAX arrays** for all frame data (JAX-compatible)
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
from jax import jacfwd

from trodestrack.models.filter_common import (
    FilterCoreConfig,
    prepare_heading_measurement,
    state_yaw,
    wrap_angle,
)
from trodestrack.models.state_layout import StateLayout, get_heading_index

HEADING_GATE_THRESHOLD = 1e5


class HeadingPseudoModel:
    """Heading pseudo-measurement model from dual-LED geometry with preallocated arrays.

    This model extracts heading from the LED pair vector and applies
    adaptive gating based on LED spacing validity.

    Parameters
    ----------
    config : FilterCoreConfig
        Full filter configuration (heading parameters extracted internally).
    layout : StateLayout
        State index mapping for accessing heading component.
    z_led1_all : jnp.ndarray
        LED1 positions (T, 2) [x, y] in meters. Use NaN for invalid frames.
    z_led2_all : jnp.ndarray
        LED2 positions (T, 2) [x, y] in meters. Use NaN for invalid frames.

    Attributes
    ----------
    z_led1_all : jnp.ndarray
        Preallocated LED1 positions (T, 2).
    z_led2_all : jnp.ndarray
        Preallocated LED2 positions (T, 2).

    Methods
    -------
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
        Return identity selector (1, 1) - not applicable for heading.

    Notes
    -----
    - **JAX-traceable:** Uses preallocated arrays, no Python dict lookups
    - Uses prepare_heading_measurement() for LED spacing validation and gating
    - Invalid observations gated via R=1e6 (no Python branching)
    - Innovation wrapped to [-π, π] via wrap_angle()
    - Ensures heading_obs is finite (replaces NaN with 0.0)

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
    >>> # Preallocate arrays for T=100 frames
    >>> z_led1 = jnp.ones((100, 2))
    >>> z_led2 = jnp.ones((100, 2)) * 1.04
    >>> model = HeadingPseudoModel(
    ...     config=config,
    ...     layout=layout,
    ...     z_led1_all=z_led1,
    ...     z_led2_all=z_led2,
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
        z_led1_all: jnp.ndarray,
        z_led2_all: jnp.ndarray,
    ) -> None:
        """Initialize heading pseudo-measurement model with preallocated arrays.

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
        z_led1_all : jnp.ndarray
            LED1 positions (T, 2) [x, y] in meters.
        z_led2_all : jnp.ndarray
            LED2 positions (T, 2) [x, y] in meters.
        """
        # Shape gate: ``meas_cov`` / ``use_measurement`` / ``innovation``
        # index the frame arrays by ``frame_idx``. JAX silently clamps
        # out-of-range indices to the last row, so an undersized z_led1_all
        # reuses frame 0 for every later step. Validate (n_time, 2) for
        # both LEDs at the constructor boundary so direct callers (tests,
        # custom pipelines) see the same gate the public EKF/UKF entry
        # points already enforce via validate_camera_input_shapes.
        z_led1_arr = jnp.asarray(z_led1_all)
        z_led2_arr = jnp.asarray(z_led2_all)
        if z_led1_arr.ndim != 2 or z_led1_arr.shape[1] != 2:
            raise ValueError(
                f"z_led1_all must have shape (n_time, 2); got {z_led1_arr.shape}."
            )
        if z_led2_arr.shape != z_led1_arr.shape:
            raise ValueError(
                "z_led1_all and z_led2_all must share shape (n_time, 2); "
                f"got z_led1_all={z_led1_arr.shape}, z_led2_all={z_led2_arr.shape}."
            )

        self.config = config
        self.layout = layout

        # Preallocated arrays (JAX-traceable)
        self.z_led1_all = z_led1_all
        self.z_led2_all = z_led2_all

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
        Measurement function: h(x) = yaw(x), where yaw is either the scalar
        heading state or the yaw extracted from a quaternion orientation.
        """
        return jnp.array([state_yaw(state_mean, self.layout)])

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
        H selects the scalar heading for 2D layouts. Quaternion layouts use
        automatic differentiation through the yaw extraction.
        """
        if self.layout.has_quaternion_orientation:
            return jacfwd(lambda state: self.predict(state))(state_mean)

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
        - Computed on-demand from LED arrays via prepare_heading_measurement()
        """
        z_led1 = self.z_led1_all[frame_idx]
        z_led2 = self.z_led2_all[frame_idx]

        # Compute heading measurement on-demand
        _, R_heading, _ = prepare_heading_measurement(z_led1, z_led2, self.config)

        return jnp.array([[R_heading]])

    def use_measurement(self, frame_idx: int) -> jnp.ndarray:
        """Return whether the heading pseudo-measurement should be applied."""
        z_led1 = self.z_led1_all[frame_idx]
        z_led2 = self.z_led2_all[frame_idx]
        _, _, use_heading = prepare_heading_measurement(z_led1, z_led2, self.config)
        return use_heading

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
        - Ensures heading_obs is finite (replaces NaN with 0.0 for gated cases)
        - Computed on-demand from LED arrays via prepare_heading_measurement()
        """
        z_led1 = self.z_led1_all[frame_idx]
        z_led2 = self.z_led2_all[frame_idx]

        # Compute heading measurement on-demand
        heading_obs, _, _ = prepare_heading_measurement(z_led1, z_led2, self.config)

        # Ensure heading_obs is finite (replace NaN with 0.0)
        heading_obs = jnp.where(jnp.isfinite(heading_obs), heading_obs, 0.0)

        # Raw innovation with angle wrapping
        innov_raw = wrap_angle(heading_obs - meas_pred[0])

        # Replace NaN with 0 (additional safety for gated cases)
        innovation = jnp.where(jnp.isfinite(innov_raw), innov_raw, 0.0)

        return jnp.array([innovation])

    def subspace(self, frame_idx: int) -> tuple[bool, bool, bool, jnp.ndarray]:
        """Return identity selector (not applicable for heading).

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
        selector_M : jnp.ndarray
            Identity (1, 1).

        Notes
        -----
        Heading is always 1D, no LED subspace projection needed.
        Selector terminology used for consistency with camera model.
        """
        return False, False, False, jnp.eye(1)

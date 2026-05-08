"""Measurement model protocol for sensor fusion.

This module defines the MeasurementModel protocol used by sensors with a
fixed per-frame measurement dimension: camera position, heading
pseudo-measurements, ZUPT. TTL event sources (beam break, zone trigger,
RFID reader) deliberately do *not* implement this protocol because their
per-frame source count is variable; that path lives in
``models/sensors/event_location.py`` and is wired through the EKF/UKF as a
separate ``update_event_location`` call.

Design Philosophy
-----------------
- **Static shapes for JAX**: Protocol measurements use fixed-size arrays.
- **Explicit frame indexing**: Sensor models cache frame-specific observations.
- **Selector-based updates**: Partial observations are projected to valid rows.
- **Model-local gating**: Missing/invalid observations are handled by each model.

Protocol Methods
----------------
- `meas_dim`: Measurement dimension (e.g., 4 for dual-LED camera, 1 for heading)
- `predict()`: Measurement prediction h(x) from state
- `jacobian()`: Measurement Jacobian H (optional, returns None for UKF)
- `meas_cov()`: Measurement noise covariance R for given frame
- `innovation()`: Innovation z - h(x) with sensor-specific processing (e.g., angle wrapping)
- `subspace()`: LED validity flags and selector matrix for lifted updates

"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class MeasurementModel(Protocol):
    """Protocol for sensor measurement models in Kalman filtering.

    This protocol defines the interface for fixed-shape, frame-indexed
    measurement models that are compatible with the shared EKF/UKF projected
    update path. The protocol uses structural subtyping (duck typing) via
    `@runtime_checkable`.

    Methods
    -------
    meas_dim : property
        Return measurement dimension (e.g., 4 for camera, 1 for heading).
    predict(state_mean: jnp.ndarray) -> jnp.ndarray
        Compute measurement prediction h(x) from state mean.
    jacobian(state_mean: jnp.ndarray) -> jnp.ndarray | None
        Compute measurement Jacobian H = ∂h/∂x at state_mean.
        Returns None for UKF (uses sigma points instead).
    meas_cov(frame_idx: int) -> jnp.ndarray
        Return measurement noise covariance R for frame_idx.
        May depend on confidence, LED validity, or gating logic.
    innovation(frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray
        Compute innovation (z - h(x)) for frame_idx with sensor-specific
        processing (e.g., angle wrapping for heading).
    subspace(frame_idx: int) -> tuple[bool, bool, bool, jnp.ndarray]
        Return LED validity flags and selector matrix:
        ``(both_leds, only_led1, only_led2, selector_M)``
        For camera: selector is always (2, 4) regardless of LED validity.
        For heading: (1, 1) identity. For ZUPT: (2, 2) identity.

    Notes
    -----
    - Implementations preallocate the per-frame arrays (LED positions,
      masks, confidences) at construction time and index them by
      ``frame_idx`` inside the JIT-traced filter.
    - All arrays use static shapes for JAX compatibility.
    - Missing or invalid observations are handled inside each model without
      Python branching in JAX scans.
    - Protocol is `@runtime_checkable` for isinstance() checks.
    - ``EventLocationModel`` (TTL beam / zone / RFID) intentionally does
      *not* implement this protocol because it has variable per-frame
      source counts and a stacked ``(2K, n_state)`` Jacobian; the
      EKF/UKF call ``update_event_location`` for that channel rather
      than the protocol path. See ``models/sensors/event_location.py``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from trodestrack.models.sensors import CameraPositionModel
    >>> from trodestrack.models.sensors.protocols import MeasurementModel
    >>> from trodestrack.models.state_layout import get_layout
    >>> layout = get_layout("2d_full")
    >>> n_frames = 4
    >>> model = CameraPositionModel(
    ...     led_distance=0.04,
    ...     measurement_noise_base=0.005**2,
    ...     layout=layout,
    ...     z_led1_all=jnp.zeros((n_frames, 2)),
    ...     z_led2_all=jnp.zeros((n_frames, 2)),
    ... )
    >>> isinstance(model, MeasurementModel)
    True
    >>> model.meas_dim
    4
    """

    @property
    def meas_dim(self) -> int:
        """Measurement dimension.

        Returns
        -------
        int
            Number of measurement dimensions. Examples:
            - Camera dual-LED: 4
            - Heading: 1
            - ZUPT velocity: 2
        """
        ...

    def predict(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Compute measurement prediction h(x) from state.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean vector (n,).

        Returns
        -------
        jnp.ndarray
            Predicted measurement (meas_dim,).

        Notes
        -----
        - For camera: returns [x1, y1, x2, y2] from position and heading
        - For heading: returns [θ] extracted from state
        - For ZUPT: returns [vx, vy] extracted from state
        """
        ...

    def jacobian(self, state_mean: jnp.ndarray) -> jnp.ndarray | None:
        """Compute measurement Jacobian H = ∂h/∂x.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean vector (n,) at which to evaluate Jacobian.

        Returns
        -------
        jnp.ndarray | None
            Jacobian matrix (meas_dim, n), or None if not applicable.
            - EKF: requires explicit Jacobian
            - UKF: returns None (uses sigma points for implicit linearization)

        Notes
        -----
        - For camera: 4×n matrix with derivatives w.r.t. position, heading
        - For heading: 1×n matrix selecting heading component
        - UKF models return None (no explicit Jacobian needed)
        """
        ...

    def meas_cov(self, frame_idx: int) -> jnp.ndarray:
        """Return measurement noise covariance R for frame_idx.

        Parameters
        ----------
        frame_idx : int
            Frame index identifying the observation.

        Returns
        -------
        jnp.ndarray
            Measurement noise covariance (meas_dim, meas_dim).

        Notes
        -----
        - For camera: R may be confidence-scaled per dimension
        - For heading: R may be adaptively scaled by LED spacing ratio
        - Missing or invalid observations are handled by the concrete model.
        - Implementations cache frame data at construction time.
        """
        ...

    def innovation(self, frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Compute innovation (z - h(x)) for frame_idx.

        Parameters
        ----------
        frame_idx : int
            Frame index identifying the observation.
        meas_pred : jnp.ndarray
            Predicted measurement h(x) (meas_dim,).

        Returns
        -------
        jnp.ndarray
            Innovation vector (meas_dim,) with sensor-specific processing.

        Notes
        -----
        - For camera: z - h(x) with NaN replacement (sanitized to meas_pred)
        - For heading: wraps innovation to [-π, π]
        - For velocity: simple z - h(x)
        - **Projection-only approach**: invalid components replaced with meas_pred
          to yield zero residual; no explicit zeroing or large-R inflation
        """
        ...

    def subspace(self, frame_idx: int) -> tuple[bool, bool, bool, jnp.ndarray]:
        """Return LED validity flags and selector matrix for frame_idx.

        Parameters
        ----------
        frame_idx : int
            Frame index identifying the observation.

        Returns
        -------
        both_leds : bool
            True if both LEDs valid (4D update, selector ignored).
        only_led1 : bool
            True if only LED1 valid (2D update via selector).
        only_led2 : bool
            True if only LED2 valid (2D update via selector).
        selector_M2 : jnp.ndarray
            Selector matrix with **static shape** for measurements:
            - Camera: (2, 4) selector mapping 4D → 2D subspace when single LED valid
            - Heading: (1, 1) identity
            - ZUPT: (2, 2) identity
            For dual-LED camera case, returns conventional (2, 4) selector (ignored by update).

        Notes
        -----
        **Critical for PR2/PR3 JAX compatibility:**
        - Shape must be **static** and **known at trace time**
        - Camera: always returns (2, 4) selector, never (4, 4)
        - Heading: returns (1, 1) identity. ZUPT: returns (2, 2) identity
        - Generic update primitive uses `lax.cond(both_leds, ...)` to choose
          4D direct update vs 2D selector-based update
        - See `filter_common.make_led_selector()` for camera implementation

        **Naming:**
        - Called "selector" (not "projector") because it selects a 2D subspace
          from 4D measurement space via matrix multiplication: M2 @ z4 → z2
        """
        ...

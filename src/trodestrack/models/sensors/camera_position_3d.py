"""3D LED camera measurement model for quaternion pose states.

This module defines the camera-side measurement convention needed for future
full 3D camera + 6-DOF IMU tracking. It is intentionally independent of any
loader or Arthur-specific data shape.

Convention
----------
``z_leds_all`` has shape ``(n_time, n_leds, 3)`` and stores observed LED
positions in world coordinates. ``mask_leds_all`` has shape
``(n_time, n_leds)`` and marks whole-LED visibility. Individual coordinates
can also be set to NaN; non-finite coordinates are ignored independently.

Predictions use body-frame LED offsets and a scalar-first body-to-world
quaternion from a 3D quaternion state layout.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jacfwd

from trodestrack.models.filter_common import confidence_to_R_diagonal
from trodestrack.models.quaternion import rotate_vector_body_to_world
from trodestrack.models.state_layout import StateLayout


class Camera3DPositionModel:
    """3D LED position measurement model with fixed-shape masked outputs.

    Parameters
    ----------
    led_offsets_body : jnp.ndarray
        LED positions in the body/headstage frame, shape ``(n_leds, 3)``.
    measurement_noise_base : float
        Base position measurement noise variance in m^2.
    layout : StateLayout
        3D quaternion state layout. Position indices must be 3D and
        orientation indices must be scalar-first quaternion indices.
    z_leds_all : jnp.ndarray
        Observed 3D LED positions, shape ``(n_time, n_leds, 3)``. Use NaN for
        missing coordinates.
    mask_leds_all : jnp.ndarray | None, default None
        Optional whole-LED visibility mask, shape ``(n_time, n_leds)``. If not
        provided, finite coordinates alone determine validity.
    conf_all : jnp.ndarray | None, default None
        Optional confidence scores. Shape may be ``(n_time, n_leds)`` for
        per-LED confidence or ``(n_time, n_leds, 3)`` for per-coordinate
        confidence.
    confidence_clip_min : float, default 1e-2
        Lower bound for confidence-to-noise scaling.
    invalid_measurement_noise : float, default 1e6
        Diagonal variance used for masked or non-finite coordinates.
    """

    def __init__(
        self,
        *,
        led_offsets_body: jnp.ndarray,
        measurement_noise_base: float,
        layout: StateLayout,
        z_leds_all: jnp.ndarray,
        mask_leds_all: jnp.ndarray | None = None,
        conf_all: jnp.ndarray | None = None,
        confidence_clip_min: float = 1e-2,
        invalid_measurement_noise: float = 1e6,
    ) -> None:
        if layout.spatial_dim != 3 or not layout.has_quaternion_orientation:
            raise ValueError(
                "Camera3DPositionModel requires a 3D quaternion state layout."
            )
        if measurement_noise_base <= 0:
            raise ValueError(
                f"measurement_noise_base must be > 0, got {measurement_noise_base}."
            )
        if invalid_measurement_noise <= 0:
            raise ValueError(
                "invalid_measurement_noise must be > 0, got "
                f"{invalid_measurement_noise}."
            )

        led_offsets = jnp.asarray(led_offsets_body)
        z_leds = jnp.asarray(z_leds_all)
        if led_offsets.ndim != 2 or led_offsets.shape[1] != 3:
            raise ValueError(
                "led_offsets_body must have shape (n_leds, 3); got "
                f"{led_offsets.shape}."
            )
        if led_offsets.shape[0] < 2:
            raise ValueError("Camera3DPositionModel requires at least two LED offsets.")
        if z_leds.ndim != 3 or z_leds.shape[1:] != led_offsets.shape:
            raise ValueError(
                "z_leds_all must have shape (n_time, n_leds, 3) matching "
                f"led_offsets_body; got {z_leds.shape} and {led_offsets.shape}."
            )

        if mask_leds_all is not None:
            mask_leds = jnp.asarray(mask_leds_all)
            if mask_leds.shape != z_leds.shape[:2]:
                raise ValueError(
                    "mask_leds_all must have shape (n_time, n_leds); got "
                    f"{mask_leds.shape} for z_leds_all {z_leds.shape}."
                )
        else:
            mask_leds = None

        if conf_all is not None:
            conf = jnp.asarray(conf_all)
            if conf.shape not in (z_leds.shape[:2], z_leds.shape):
                raise ValueError(
                    "conf_all must have shape (n_time, n_leds) or "
                    f"(n_time, n_leds, 3); got {conf.shape}."
                )
        else:
            conf = None

        self.led_offsets_body = led_offsets
        self.measurement_noise_base = measurement_noise_base
        self.layout = layout
        self.z_leds_all = z_leds
        self.mask_leds_all = mask_leds
        self.conf_all = conf
        self.confidence_clip_min = confidence_clip_min
        self.invalid_measurement_noise = invalid_measurement_noise

    @property
    def n_leds(self) -> int:
        """Number of LED markers represented by the model."""
        return int(self.led_offsets_body.shape[0])

    @property
    def meas_dim(self) -> int:
        """Flattened measurement dimension, ``n_leds * 3``."""
        return self.n_leds * 3

    def predict(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Predict flattened 3D LED positions from state.

        Parameters
        ----------
        state_mean : jnp.ndarray
            State mean, shape ``(n_state,)``.

        Returns
        -------
        jnp.ndarray
            Predicted LED coordinates flattened in LED-major order,
            shape ``(n_leds * 3,)``.
        """
        pos_idx = jnp.array(self.layout.pos_idx, dtype=jnp.int32)
        quat_idx = jnp.array(self.layout.heading_idx, dtype=jnp.int32)
        position_world = state_mean[pos_idx]
        quat_body_to_world = state_mean[quat_idx]
        led_world = position_world[None, :] + rotate_vector_body_to_world(
            quat_body_to_world,
            self.led_offsets_body,
        )
        return led_world.reshape(-1)

    def jacobian(self, state_mean: jnp.ndarray, frame_idx: int) -> jnp.ndarray:
        """Return frame-masked measurement Jacobian.

        Invalid coordinates have zero Jacobian rows. This makes missing
        coordinates exact no-ops in Kalman updates instead of relying on a large
        finite measurement variance, which can still shrink covariance when the
        prior uncertainty is very large.
        """
        H = self.geometric_jacobian(state_mean)
        valid = self.valid_coordinates(frame_idx).astype(H.dtype)
        return H * valid[:, None]

    def geometric_jacobian(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        """Return unmasked geometric Jacobian with shape ``(meas_dim, n)``."""
        return jacfwd(lambda state: self.predict(state))(state_mean)

    def observed(self, frame_idx: int) -> jnp.ndarray:
        """Return flattened raw observations for one camera frame."""
        return self.z_leds_all[frame_idx].reshape(-1)

    def valid_coordinates(self, frame_idx: int) -> jnp.ndarray:
        """Return flattened coordinate-validity mask for one frame."""
        finite = jnp.isfinite(self.z_leds_all[frame_idx])
        if self.mask_leds_all is None:
            return finite.reshape(-1)
        led_mask = self.mask_leds_all[frame_idx].astype(bool)
        return (finite & led_mask[:, None]).reshape(-1)

    def meas_cov(self, frame_idx: int) -> jnp.ndarray:
        """Return diagonal measurement covariance with invalid coordinates gated.

        Invalid coordinates receive ``invalid_measurement_noise`` on the
        diagonal. This keeps measurement shapes static while making missing
        coordinates contribute effectively no update.
        """
        confidence = self._confidence_for_frame(frame_idx)
        R_diag = confidence_to_R_diagonal(
            confidence,
            base=self.measurement_noise_base,
            size=self.meas_dim,
            clip_min=self.confidence_clip_min,
        )
        valid = self.valid_coordinates(frame_idx)
        R_diag = jnp.where(
            valid,
            R_diag,
            jnp.asarray(self.invalid_measurement_noise, dtype=R_diag.dtype),
        )
        return jnp.diag(R_diag)

    def innovation(self, frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray:
        """Return finite innovation with missing coordinates set to zero residual."""
        z_obs = self.observed(frame_idx)
        valid = self.valid_coordinates(frame_idx)
        z_sanitized = jnp.where(valid, z_obs, meas_pred)
        return z_sanitized - meas_pred

    def _confidence_for_frame(self, frame_idx: int) -> jnp.ndarray | None:
        if self.conf_all is None:
            return None
        confidence = self.conf_all[frame_idx]
        if confidence.ndim == 1:
            return jnp.repeat(confidence, 3)
        return confidence.reshape(-1)

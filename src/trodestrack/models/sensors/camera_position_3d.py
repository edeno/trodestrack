"""3D LED camera measurement model for quaternion pose states.

This module defines the camera-side measurement convention used today by
the experimental ``extended_kalman_filter_3d`` entry point for 3D camera +
6-DOF IMU tracking. It is intentionally independent of any loader or
Arthur-specific data shape.

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

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd

from trodestrack.models.filter_common import confidence_to_R_diagonal
from trodestrack.models.quaternion import rotate_vector_body_to_world
from trodestrack.models.state_layout import StateLayout


def _is_traced(arr) -> bool:
    """True when ``arr`` is a JAX tracer (cannot run host-side numeric checks)."""

    return isinstance(arr, jax.core.Tracer)


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
        # Use np.isfinite explicitly: NaN compares False to ``<= 0`` and
        # would otherwise propagate through every covariance / prediction
        # entry. confidence_clip_min appears in the denominator of
        # confidence_to_R_diagonal (R = base / clip(conf, clip_min, 1.0)),
        # so it must also be strictly positive.
        if not np.isfinite(measurement_noise_base) or measurement_noise_base <= 0:
            raise ValueError(
                "measurement_noise_base must be a finite strictly-positive "
                f"variance; got {measurement_noise_base!r}."
            )
        if not np.isfinite(invalid_measurement_noise) or invalid_measurement_noise <= 0:
            raise ValueError(
                "invalid_measurement_noise must be a finite strictly-positive "
                f"variance; got {invalid_measurement_noise!r}."
            )
        if not np.isfinite(confidence_clip_min) or confidence_clip_min <= 0:
            raise ValueError(
                "confidence_clip_min must be a finite strictly-positive "
                f"floor; got {confidence_clip_min!r}."
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
        # Reject non-finite LED offsets — predict() rotates them into
        # world frame and adds them to the predicted body position, so a
        # single NaN/inf poisons every predicted LED. Only run this on
        # concrete (host) arrays; the model is also reconstructed inside
        # the JIT-traced filter where the array is a tracer and has
        # already been validated at the public entry point.
        if not _is_traced(led_offsets_body) and not np.all(
            np.isfinite(np.asarray(led_offsets_body))
        ):
            raise ValueError(
                "led_offsets_body must contain only finite values; got "
                "non-finite entries (NaN/inf)."
            )
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
            # Reject non-bool / non-0-or-1-integer masks for concrete
            # (non-traced) inputs. ``valid_coordinates()`` later does
            # ``astype(bool)`` which silently coerces 2 / -1 / NaN into
            # True and treats invalid LEDs as visible. The 3D filter
            # entry point already enforces this contract via
            # ``validate_camera_3d_input_shapes``; mirror it here so
            # callers constructing Camera3DPositionModel directly (e.g.
            # in tests or custom pipelines) get the same gate.
            if not _is_traced(mask_leds_all):
                mask_arr_host = np.asarray(mask_leds_all)
                if mask_arr_host.dtype != np.bool_:
                    if not np.issubdtype(mask_arr_host.dtype, np.integer):
                        raise ValueError(
                            "mask_leds_all must be boolean or 0/1 integer; "
                            f"got dtype {mask_arr_host.dtype!r}."
                        )
                    if not np.all(np.isin(mask_arr_host, (0, 1))):
                        bad = mask_arr_host[~np.isin(mask_arr_host, (0, 1))]
                        raise ValueError(
                            "mask_leds_all must contain only 0 or 1 (or be "
                            f"boolean); found {len(bad)} other value(s) "
                            f"(e.g. {bad[:5].tolist()})."
                        )
            mask_leds = mask_leds.astype(bool)
        else:
            mask_leds = None

        if conf_all is not None:
            conf = jnp.asarray(conf_all)
            if conf.shape not in (z_leds.shape[:2], z_leds.shape):
                raise ValueError(
                    "conf_all must have shape (n_time, n_leds) or "
                    f"(n_time, n_leds, 3); got {conf.shape}."
                )
            # Reject non-finite confidences — confidence_to_R_diagonal
            # uses ``base / clip(conf, clip_min, 1.0)``; np.clip of NaN
            # is NaN, which propagates into every R entry. The 2D EKF /
            # UKF entry points already validate conf_cam at ingress, but
            # this model is also constructed directly elsewhere.
            if not _is_traced(conf_all) and not np.all(
                np.isfinite(np.asarray(conf_all))
            ):
                raise ValueError(
                    "conf_all must contain only finite values; got "
                    "non-finite entries (NaN/inf)."
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

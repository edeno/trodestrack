"""Shared configuration, state containers, and helpers for Kalman filters.

This module provides common dataclasses, utilities, and math helpers used by
both EKF and UKF implementations. All public functions use NumPy-style
docstrings and include array shapes and physical units where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array, jacfwd, lax, tree_util
from jax.scipy.linalg import cho_factor, cho_solve

from trodestrack.models.quaternion import (
    integrate_body_gyro,
    normalize_quaternion,
    quaternion_from_rotation_vector,
    quaternion_to_yaw,
    rotate_vector_body_to_world,
    rotate_vector_world_to_body,
)
from trodestrack.models.state_layout import (
    STATE_MODES,
    StateLayout,
    StateMode,
    get_heading_index,
    get_layout,
)


@dataclass(frozen=True)
class FilterCoreConfig:
    """Core filter configuration shared by EKF and UKF.

    Parameters
    ----------
    process_noise_pos : float
        Position random-walk spectral density (m^2/s) used in Q. The
        process-noise rate diagonal places this directly on the position
        entries (see ``build_Q_rate`` in ``process_noise.py``) and the
        per-step assembled Q multiplies it by ``dt``, so the per-step
        position variance contribution has units m^2.
    process_noise_vel : float
        Velocity random-walk spectral density ((m/s)^2/s).
    process_noise_heading : float
        Heading random-walk spectral density (rad^2/s).
    process_noise_gyro_bias : float
        Gyro bias random-walk spectral density ((rad/s)^2/s).
    process_noise_accel_bias : float
        Accelerometer bias random-walk spectral density ((m/s^2)^2/s).

    measurement_noise_pos : float
        Per-dimension position measurement noise variance (m^2).
    measurement_noise_heading : float
        Heading measurement noise variance (rad^2).

    imu_gyro_noise_density : float
        IMU gyro noise density (rad/s/√Hz). Default: 0.01 deg/s/√Hz (SpikeGadgets spec).
    imu_accel_noise_density : float
        IMU accel noise density (m/s²/√Hz). Default: 0.2 mg/√Hz (SpikeGadgets spec).
    imu_gravity_body : tuple[float, float, float]
        Expected gravity vector in the tracking/world frame (m/s²). The field
        keeps its legacy name for API compatibility. Default assumes level
        mounting with gravity on +z.

    damping_coeff : float
        Linear velocity damping coefficient (1/s) in dynamics model.
    led_distance : float | None
        Nominal LED spacing (m). If None, spacing is estimated from data.

    use_mahalanobis_gating : bool
        Enable χ²-based outlier rejection on measurement updates.
    mahalanobis_threshold_prob : float
        Probability mass for χ² threshold (e.g., 0.997 ≈ 3σ).

    use_heading_measurement : bool
        Enable heading pseudo-measurement from LED geometry.
    led_distance_tolerance : float
        Relative tolerance for observed LED spacing vs expected (fraction).
    adaptive_heading_noise : bool
        If True, scales heading R by (expected/observed spacing)^2.

    adaptive_q_during_dropout : bool
        If True, increase position/velocity Q during vision dropouts.
    dropout_q_pos_multiplier : float
        Multiplier on position Q during dropout.
    dropout_q_vel_multiplier : float
        Multiplier on velocity Q during dropout.
    dropout_q_bias_multiplier : float
        Multiplier on bias Q during dropout (often < 1 to freeze biases).
    freeze_bias_during_blackout : bool
        If True, set bias Q≈0 during dropout to prevent drift.
    reduce_imu_noise_during_blackout : bool
        If True, scale IMU input noise when vision is absent.
    blackout_imu_noise_scale : float
        Scale applied to IMU noise during blackout when enabled.
    enable_experimental_accel_translation : bool
        If True, allow experimental quaternion-orientation modes to integrate
        accelerometer samples into velocity. Default is False.
    use_gravity_orientation_update : bool
        If True, quaternion-orientation EKF modes use a gated accelerometer
        gravity-direction pseudo-measurement to constrain roll/pitch.
    gravity_orientation_measurement_noise : float
        Variance of the unit-vector gravity-direction pseudo-measurement.
    gravity_accel_magnitude_tolerance_m_s2 : float
        Maximum stationary-acceleration magnitude deviation from gravity.
    gravity_gyro_norm_threshold_rad_s : float
        Maximum gyro norm for accepting gravity-direction updates.

    enable_zupt : bool
        Enable zero-velocity pseudo-measurements when measured stationarity gates pass.
    zupt_velocity_threshold : float
        Camera-derived speed threshold (m/s) below which ZUPT may apply.
    zupt_measurement_noise : float
        ZUPT measurement noise variance ((m/s)^2).
    zupt_gyro_threshold_rad_s : float
        IMU gyro norm threshold for stationary detection.
    zupt_accel_threshold_m_s2 : float
        IMU accelerometer mean magnitude residual threshold for stationary detection.
    zupt_camera_stationary_window_frames : int
        Camera-frame lookback window used for visual stationarity detection.
    zupt_visual_context_hold_frames : int
        Maximum number of camera frames to carry the last stationary visual
        decision through LED dropout.

    state_mode : str
        State layout key, e.g. "2d_full" (8D), "vision_only" (5D), or
        "2d_cam_3d_imu" (10D).
    """

    process_noise_pos: float = 1e-4
    process_noise_vel: float = 5e-3
    process_noise_heading: float = 5e-4
    process_noise_gyro_bias: float = 5e-8
    process_noise_accel_bias: float = 2e-5

    measurement_noise_pos: float = 0.01**2
    measurement_noise_heading: float = 0.05**2

    # SpikeGadgets IMU noise specifications (Product Manual)
    imu_gyro_noise_density: float = 0.00017453  # 0.01 deg/s/√Hz → rad/s/√Hz
    imu_accel_noise_density: float = 0.00196133  # 0.2 mg/√Hz → 0.0002g * 9.80665
    imu_gravity_body: tuple[float, float, float] = (0.0, 0.0, 9.81)

    damping_coeff: float = 0.2
    led_distance: float | None = 0.04

    use_mahalanobis_gating: bool = True  # Default to robust outlier rejection
    mahalanobis_threshold_prob: float = 0.997  # Reject ~0.3% of measurements (3σ)

    use_heading_measurement: bool = True
    led_distance_tolerance: float = 0.3
    adaptive_heading_noise: bool = True

    adaptive_q_during_dropout: bool = True
    dropout_q_pos_multiplier: float = 2.0
    dropout_q_vel_multiplier: float = 2.0
    dropout_q_bias_multiplier: float = 0.5
    freeze_bias_during_blackout: bool = True
    reduce_imu_noise_during_blackout: bool = True
    blackout_imu_noise_scale: float = 0.3
    enable_experimental_accel_translation: bool = False
    use_gravity_orientation_update: bool = True
    gravity_orientation_measurement_noise: float = 0.05**2
    gravity_accel_magnitude_tolerance_m_s2: float = 0.5
    gravity_gyro_norm_threshold_rad_s: float = 0.2

    enable_zupt: bool = True
    zupt_velocity_threshold: float = 0.02  # m/s
    zupt_measurement_noise: float = 0.01**2
    zupt_gyro_threshold_rad_s: float = 0.02
    zupt_accel_threshold_m_s2: float = 1.5
    zupt_camera_stationary_window_frames: int = 10
    zupt_visual_context_hold_frames: int = 10

    # State layout mode (controls state dimension and index mapping).
    # ``StateMode`` is the Literal alias owned by
    # ``trodestrack.models.state_layout``; ``STATE_MODES`` is the runtime
    # tuple of valid values. mypy enforces the Literal statically; the
    # runtime guard in ``__post_init__`` rejects typos at construction.
    state_mode: StateMode = "2d_cam_3d_imu"

    # PyTree support: treat `state_mode` as static auxiliary data.
    _TREE_STATIC_FIELDS: ClassVar[tuple[str, ...]] = ("state_mode",)

    # Probabilities supported by chi2_threshold's closed-form table.
    _SUPPORTED_MAHALANOBIS_PROBS: ClassVar[tuple[float, ...]] = (
        0.95,
        0.975,
        0.99,
        0.997,
    )

    def __post_init__(self) -> None:
        """Validate configuration fields at construction.

        Why: ``chi2_threshold`` only has closed-form values for the four listed
        probabilities and silently falls back to the 95% threshold for anything
        else. That silent fallback is easy to miss in tuning experiments, so we
        catch the mistake at config construction when gating is enabled.
        """
        # Strict-bool validation. Plain Python truthiness silently
        # accepts strings like ``"False"`` (truthy), integers, and
        # lists. The simulator/filter then either silently takes the
        # wrong branch (e.g. integer ``0`` looks like ``False``) or
        # crashes deep in JAX with ``TypeError: unsupported operand
        # type(s) for &: 'str' and 'jaxlib.xla_extension.ArrayImpl'``
        # when a string flows into a JAX boolean op. CLI / YAML / env
        # loaders are an obvious source — require ``bool`` exactly.
        bool_fields = (
            "use_mahalanobis_gating",
            "use_heading_measurement",
            "adaptive_heading_noise",
            "adaptive_q_during_dropout",
            "freeze_bias_during_blackout",
            "reduce_imu_noise_during_blackout",
            "enable_experimental_accel_translation",
            "use_gravity_orientation_update",
            "enable_zupt",
        )
        for fname in bool_fields:
            value = getattr(self, fname)
            if not isinstance(value, bool):
                raise ValueError(
                    f"{fname} must be a Python ``bool`` (True/False); "
                    f"got {value!r} (type {type(value).__name__}). "
                    f"If you're loading from YAML / env / CLI, parse "
                    f"the string to a bool before constructing the "
                    f"config."
                )

        # Runtime check on ``state_mode``: mypy enforces the Literal at
        # static-analysis time, but dataclasses bypass that at runtime.
        # Catch typos (e.g. ``"vison_only"``) at construction so callers
        # get a helpful message naming the allowed values instead of a
        # KeyError later in ``get_layout``.
        if self.state_mode not in STATE_MODES:
            raise ValueError(
                f"state_mode must be one of {STATE_MODES}; got "
                f"{self.state_mode!r}. Add to LAYOUT_REGISTRY and StateMode "
                "(in state_layout.py) if introducing a new mode."
            )

        if self.state_mode == "vision_only" and self.enable_zupt:
            raise ValueError(
                "enable_zupt=True is incompatible with state_mode='vision_only': "
                "ZUPT requires IMU stationarity detection. Set enable_zupt=False "
                "explicitly when using vision_only, or use a state mode that "
                "consumes IMU data."
            )

        if self.use_mahalanobis_gating:
            prob = self.mahalanobis_threshold_prob
            supported = self._SUPPORTED_MAHALANOBIS_PROBS
            if not any(abs(prob - p) < 1e-3 for p in supported):
                raise ValueError(
                    f"mahalanobis_threshold_prob={prob!r} is not in the closed-form "
                    f"table supported by chi2_threshold. Choose one of {supported} "
                    f"or set use_mahalanobis_gating=False."
                )

        gravity = np.asarray(self.imu_gravity_body, dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError(
                "imu_gravity_body must be a length-3 finite sequence "
                f"[g_x, g_y, g_z] in world-frame m/s²; got "
                f"{self.imu_gravity_body!r}."
            )
        # Use np.isfinite explicitly: NaN compares False to both <= 0 and
        # < 0, so the previous bare comparisons silently accepted NaN and
        # the value then poisoned the gravity / quaternion update path
        # (a NaN gravity_orientation_measurement_noise made
        # quaternion predict_step fail with LinAlgError mid-filter).
        if (
            not np.isfinite(self.gravity_orientation_measurement_noise)
            or self.gravity_orientation_measurement_noise <= 0
        ):
            raise ValueError(
                "gravity_orientation_measurement_noise must be a finite "
                f"strictly-positive value; got "
                f"{self.gravity_orientation_measurement_noise!r}."
            )
        if (
            not np.isfinite(self.gravity_accel_magnitude_tolerance_m_s2)
            or self.gravity_accel_magnitude_tolerance_m_s2 < 0
        ):
            raise ValueError(
                "gravity_accel_magnitude_tolerance_m_s2 must be a finite "
                f"non-negative value; got "
                f"{self.gravity_accel_magnitude_tolerance_m_s2!r}."
            )
        if (
            not np.isfinite(self.gravity_gyro_norm_threshold_rad_s)
            or self.gravity_gyro_norm_threshold_rad_s < 0
        ):
            raise ValueError(
                "gravity_gyro_norm_threshold_rad_s must be a finite "
                f"non-negative value; got "
                f"{self.gravity_gyro_norm_threshold_rad_s!r}."
            )

        # ZUPT measurement-noise variance and velocity threshold flow into
        # update_zupt as R_scalar / gating threshold respectively. NaN
        # propagates through R into the innovation covariance and every
        # downstream state; update_zupt's own checks were <= 0 / < 0 only.
        if (
            not np.isfinite(self.zupt_measurement_noise)
            or self.zupt_measurement_noise <= 0
        ):
            raise ValueError(
                "zupt_measurement_noise must be a finite strictly-positive "
                f"variance (m²/s²); got {self.zupt_measurement_noise!r}."
            )
        if (
            not np.isfinite(self.zupt_velocity_threshold)
            or self.zupt_velocity_threshold < 0
        ):
            raise ValueError(
                "zupt_velocity_threshold must be a finite non-negative speed "
                f"in m/s; got {self.zupt_velocity_threshold!r}."
            )
        if (
            not np.isfinite(self.zupt_gyro_threshold_rad_s)
            or self.zupt_gyro_threshold_rad_s < 0
        ):
            raise ValueError(
                "zupt_gyro_threshold_rad_s must be a finite non-negative angular "
                f"speed in rad/s; got {self.zupt_gyro_threshold_rad_s!r}."
            )
        if (
            not np.isfinite(self.zupt_accel_threshold_m_s2)
            or self.zupt_accel_threshold_m_s2 < 0
        ):
            raise ValueError(
                "zupt_accel_threshold_m_s2 must be a finite non-negative "
                f"acceleration in m/s²; got {self.zupt_accel_threshold_m_s2!r}."
            )
        if (
            not isinstance(self.zupt_camera_stationary_window_frames, int)
            or self.zupt_camera_stationary_window_frames < 1
        ):
            raise ValueError(
                "zupt_camera_stationary_window_frames must be a positive integer; "
                f"got {self.zupt_camera_stationary_window_frames!r}."
            )
        if (
            not isinstance(self.zupt_visual_context_hold_frames, int)
            or self.zupt_visual_context_hold_frames < 0
        ):
            raise ValueError(
                "zupt_visual_context_hold_frames must be a non-negative integer; "
                f"got {self.zupt_visual_context_hold_frames!r}."
            )

        # Process- and measurement-noise fields are variances or spectral
        # densities — they must be non-negative or build_Q_rate would write
        # negative entries straight into the diagonal of Q and produce a
        # non-PSD process covariance (verified: process_noise_pos=-1e-4
        # under n=8 gives min eig ≈ -1e-6). Strict positivity is required
        # for the measurement-noise fields because they appear in
        # innovation-covariance inverses and gating thresholds, where 0
        # collapses the gate.
        non_negative_fields = (
            "process_noise_pos",
            "process_noise_vel",
            "process_noise_heading",
            "process_noise_gyro_bias",
            "process_noise_accel_bias",
            "imu_gyro_noise_density",
            "imu_accel_noise_density",
        )
        for fname in non_negative_fields:
            value = getattr(self, fname)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{fname} must be a finite non-negative variance/spectral "
                    f"density (squared physical units); got {value!r}."
                )

        positive_fields = (
            "measurement_noise_pos",
            "measurement_noise_heading",
        )
        for fname in positive_fields:
            value = getattr(self, fname)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{fname} must be a finite strictly-positive variance "
                    f"(squared physical units); got {value!r}."
                )

        # led_distance is None when auto-detection is desired and is
        # otherwise the dual-LED spacing in meters used by the camera
        # measurement model. A NaN/inf or non-positive value silently
        # poisons every predicted LED position.
        if self.led_distance is not None:
            value = self.led_distance
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "led_distance must be None (auto-detect) or a finite "
                    f"strictly-positive value in meters; got {value!r}."
                )

        if (
            not np.isfinite(self.led_distance_tolerance)
            or self.led_distance_tolerance < 0
        ):
            raise ValueError(
                "led_distance_tolerance must be a finite non-negative "
                f"fraction; got {self.led_distance_tolerance!r}."
            )

        # damping_coeff (1/s) appears multiplicatively in the velocity
        # propagation step (`vel_next = vel + accel * dt - damping *
        # vel * dt`). NaN here propagates into every state and covariance
        # entry from the first prediction onward; a negative value would
        # invert damping into a destabilizing feedback. Require finite
        # non-negative.
        if not np.isfinite(self.damping_coeff) or self.damping_coeff < 0:
            raise ValueError(
                "damping_coeff must be a finite non-negative value (1/s); "
                f"got {self.damping_coeff!r}."
            )

        # Dropout / blackout multipliers scale the per-step Q diagonal in
        # process_noise.assemble_Q during camera blackouts. A negative
        # multiplier flips the sign of the corresponding Q diagonal entry
        # (verified: dropout_q_vel_multiplier=-1.0 → min eig ≈ -5e-5),
        # NaN propagates through Q and breaks every downstream solve, and
        # the IMU-noise scale appears multiplicatively in G Q_u Gᵀ. All
        # four are unitless scaling factors; require finite-non-negative.
        scale_fields = (
            "dropout_q_pos_multiplier",
            "dropout_q_vel_multiplier",
            "dropout_q_bias_multiplier",
            "blackout_imu_noise_scale",
        )
        for fname in scale_fields:
            value = getattr(self, fname)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{fname} must be a finite non-negative unitless "
                    f"multiplier; got {value!r}."
                )

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten config for JAX PyTree registration.

        Returns
        -------
        tuple[tuple, dict]
            Children tuple containing dynamic fields and static data dictionary
            containing class reference and static field values.

        Notes
        -----
        This method is part of the JAX PyTree protocol. It separates config
        fields into dynamic (children) and static (auxiliary) data for JIT
        compilation and tracing.
        """
        field_names = list(self.__dataclass_fields__.keys())
        children = []
        static_data = {"cls": self.__class__}

        for name in field_names:
            if name.startswith("_"):
                continue
            value = getattr(self, name)
            if name in self._TREE_STATIC_FIELDS:
                static_data[name] = value
            else:
                children.append(value)

        return tuple(children), static_data

    @classmethod
    def tree_unflatten(cls, static_data: dict, children: tuple) -> FilterCoreConfig:
        """Reconstruct config from PyTree children.

        Parameters
        ----------
        static_data : dict
            Static auxiliary data containing class reference and static field values.
            Must include 'cls' key with the target class.
        children : tuple
            Dynamic field values to reconstruct.

        Returns
        -------
        FilterCoreConfig
            Reconstructed configuration instance with all fields restored.

        Notes
        -----
        This method is part of the JAX PyTree protocol. It reconstructs a config
        instance from the separated dynamic (children) and static (auxiliary) data
        produced by tree_flatten().
        """
        target_cls = static_data.get("cls", cls)
        field_names = list(target_cls.__dataclass_fields__.keys())
        child_iter = iter(children)
        static_fields = getattr(target_cls, "_TREE_STATIC_FIELDS", ())

        kwargs = {}
        for name in field_names:
            if name.startswith("_"):
                continue
            if name in static_fields:
                kwargs[name] = static_data[name]
            else:
                kwargs[name] = next(child_iter)

        return target_cls(**kwargs)


tree_util.register_pytree_node_class(FilterCoreConfig)


class FilterState(NamedTuple):
    """Kalman filter state comprising mean vector and covariance matrix.

    Attributes
    ----------
    mean : jnp.ndarray
        State mean (n,). Units depend on layout; typically
        [x(m), y(m), vx(m/s), vy(m/s), θ(rad), b_gz(rad/s), b_ax(m/s^2), b_ay(m/s^2)].
    cov : jnp.ndarray
        State covariance (n, n).

    Notes
    -----
    Raw ``FilterState(mean, cov)`` construction is intentionally still
    supported: the filter cores build ``FilterState`` instances inside
    JIT-compiled scan bodies, where the Python-side validation done by
    :meth:`create` cannot be traced. For host-side construction (CLI
    initialization, tests, user code building an initial state), prefer
    :meth:`FilterState.create` so shape and PSD violations are caught at
    construction instead of crashing deep in a JAX trace.
    """

    mean: jnp.ndarray
    cov: jnp.ndarray

    @classmethod
    def create(
        cls,
        mean: jnp.ndarray,
        cov: jnp.ndarray,
        layout: StateLayout | None = None,
    ) -> FilterState:
        """Construct a ``FilterState`` with shape and PSD validation.

        Validates shape, finiteness, symmetry, and strict positive-
        definiteness of ``cov`` before constructing. When ``layout`` is
        provided, additionally checks that ``mean`` has length
        ``layout.n``. Raises ``ValueError`` on any violation.

        Parameters
        ----------
        mean : jnp.ndarray
            State mean (n,).
        cov : jnp.ndarray
            State covariance (n, n). Must be symmetric and positive
            definite.
        layout : StateLayout, optional
            If provided, additionally validates that ``mean.shape[0] ==
            layout.n`` so the state vector matches the active state mode.

        Returns
        -------
        FilterState
            The constructed state container.

        Raises
        ------
        ValueError
            If ``mean`` is not 1-D, ``cov`` is not square, the shapes
            disagree, any value is non-finite, ``cov`` is non-symmetric,
            ``cov`` is not strictly positive definite, or (when
            ``layout`` is given) ``mean.shape[0] != layout.n``.
        """
        mean_np = np.asarray(mean)
        cov_np = np.asarray(cov)
        if mean_np.ndim != 1:
            raise ValueError(
                f"FilterState.mean must be 1-D; got shape {mean_np.shape}."
            )
        n = mean_np.shape[0]
        if cov_np.shape != (n, n):
            raise ValueError(
                f"FilterState.cov must have shape ({n}, {n}) to match mean of "
                f"shape ({n},); got {cov_np.shape}."
            )
        if not np.all(np.isfinite(mean_np)):
            raise ValueError("FilterState.mean contains non-finite value(s) (NaN/inf).")
        if not np.all(np.isfinite(cov_np)):
            raise ValueError("FilterState.cov contains non-finite value(s) (NaN/inf).")
        if not np.allclose(cov_np, cov_np.T, atol=1e-10):
            raise ValueError(
                "FilterState.cov must be symmetric (cov == cov.T); got an "
                "asymmetric matrix."
            )
        try:
            np.linalg.cholesky(cov_np)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "FilterState.cov must be symmetric and strictly positive "
                "definite (the filter's covariance solves require a non-"
                "singular prior; add a small diagonal floor to express "
                "deterministic components instead of zero variance). "
                f"Cholesky factorization failed: {exc}."
            ) from exc
        if layout is not None and n != layout.n:
            raise ValueError(
                f"FilterState.mean has shape ({n},); layout requires "
                f"({layout.n},). Use the StateLayout matching the state_mode."
            )
        return cls(mean=mean, cov=cov)


def symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    """Enforce numerical symmetry on a square matrix.

    Parameters
    ----------
    matrix : jnp.ndarray
        Input matrix (n, n).

    Returns
    -------
    jnp.ndarray
        Symmetrized matrix (n, n): 0.5·(A + Aᵀ).
    """

    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def adaptive_diagonal_boost(
    matrix: jnp.ndarray,
    *,
    absolute_floor: float = 1e-9,
    relative_scale: float = 1e-6,
) -> jnp.ndarray:
    """Return a scale-aware diagonal boost for PSD Cholesky operations."""

    mean_diag = jnp.trace(matrix) / matrix.shape[-1]
    relative_boost = relative_scale * jnp.maximum(jnp.abs(mean_diag), 1.0)
    return jnp.maximum(jnp.asarray(absolute_floor, dtype=matrix.dtype), relative_boost)


def psd_solve(
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    diagonal_boost: float = 1e-9,
    relative_diagonal_boost: float = 1e-6,
) -> jnp.ndarray:
    """Solve A x = b for PSD matrices via Cholesky factorization.

    Parameters
    ----------
    matrix : jnp.ndarray
        Positive semi-definite matrix A (k, k).
    rhs : jnp.ndarray
        Right-hand side b. Shape (k,) or (k, m).
    diagonal_boost : float, optional
        Small value added to diag(A) to improve numerical stability.

    Returns
    -------
    jnp.ndarray
        Solution x with shape matching rhs.
    """

    matrix = symmetrize(matrix)
    boost = adaptive_diagonal_boost(
        matrix,
        absolute_floor=diagonal_boost,
        relative_scale=relative_diagonal_boost,
    )
    stabilized = matrix + boost * jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    chol, lower = cho_factor(stabilized, lower=True)
    return cho_solve((chol, lower), rhs)


def joseph_update(
    cov_prior: jnp.ndarray,
    gain: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
) -> jnp.ndarray:
    """Joseph-form covariance update that preserves PSD and symmetry.

    Parameters
    ----------
    cov_prior : jnp.ndarray
        Prior covariance P⁻ (n, n).
    gain : jnp.ndarray
        Kalman gain K (n, k).
    H : jnp.ndarray
        Measurement Jacobian H (k, n).
    R : jnp.ndarray
        Measurement noise covariance R (k, k).

    Returns
    -------
    jnp.ndarray
        Posterior covariance P⁺ (n, n).

    Notes
    -----
    Uses the numerically stable Joseph form:

    P⁺ = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ
    """

    n = cov_prior.shape[0]
    identity = jnp.eye(n)
    I_minus_KH = identity - gain @ H
    return symmetrize(I_minus_KH @ cov_prior @ I_minus_KH.T + gain @ R @ gain.T)


def validate_initial_state(
    initial_state,
    layout: StateLayout,
    *,
    func_name: str = "Kalman filter",
) -> None:
    """Validate that ``initial_state.mean`` / ``cov`` match the active layout.

    The filter wrappers pass ``initial_state`` straight into the JIT'd
    core. A wrong-dimension mean (e.g. a 5-D state from a previous
    ``vision_only`` run plugged into an ``2d_full`` filter) silently
    produced ``filtered_means`` with the wrong second axis instead of
    failing loudly, and a NaN/inf entry poisoned every downstream state.
    Validate at the public entry point so the contract failure has a
    clear ValueError message.
    """
    n = layout.n
    mean = np.asarray(initial_state.mean)
    if mean.shape != (n,):
        raise ValueError(
            f"{func_name}: initial_state.mean must have shape ({n},) for "
            f"state_mode='{layout}' (n={n}); got {mean.shape}."
        )
    if not np.all(np.isfinite(mean)):
        raise ValueError(
            f"{func_name}: initial_state.mean contains non-finite value(s) "
            "(NaN/inf); the initial state must be a finite vector."
        )

    cov = np.asarray(initial_state.cov)
    if cov.shape != (n, n):
        raise ValueError(
            f"{func_name}: initial_state.cov must have shape ({n}, {n}) for "
            f"state_mode='{layout}' (n={n}); got {cov.shape}."
        )
    if not np.all(np.isfinite(cov)):
        raise ValueError(
            f"{func_name}: initial_state.cov contains non-finite value(s) "
            "(NaN/inf); the initial covariance must be a finite matrix."
        )
    # Catch invalid covariance content at the public boundary. An
    # asymmetric or non-PSD prior previously slipped through and either
    # crashed deep in eigvalsh / Cholesky inside the JIT'd core or
    # silently produced negative-eigenvalue covariances.
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError(
            f"{func_name}: initial_state.cov must be symmetric (cov == cov.T); "
            "got an asymmetric matrix."
        )
    # The JIT'd filter inverts the innovation covariance ``H P Hᵀ + R``
    # and propagates ``F P Fᵀ + Q`` through psd_solve / triangular solves
    # that expect a strictly-positive-definite ``P``. A singular
    # (zero-row / zero-column) prior expresses "this state component is
    # known exactly" but causes those solves to lose rank downstream, so
    # we require strict positive-definiteness here. If you genuinely
    # need a deterministic component, add a small floor (e.g. 1e-9 on the
    # corresponding diagonal) before constructing initial_state.cov.
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{func_name}: initial_state.cov must be symmetric and "
            "STRICTLY positive definite (the filter's covariance solves "
            "require a non-singular prior; add a small diagonal floor "
            "to express deterministic components instead of zero "
            f"variance). Cholesky factorization failed: {exc}."
        ) from exc


def validate_imu_input_shape(
    U_imu,
    layout: StateLayout,
    *,
    t_imu=None,
    func_name: str = "Kalman filter",
) -> None:
    """Validate that ``U_imu`` has a shape ``dynamics_function`` can consume.

    Raises a ``ValueError`` with an actionable message if the shape is wrong.
    This is a cheap check performed at the filter/smoother entrypoint so that
    silent channel misinterpretations (the classic "6-channel IMU from the
    loader into a 4-channel 3D-IMU filter" bug, where ``dynamics_function``
    auto-detects 3D from ``imu.shape[0] >= 4`` and reads wrong columns) fail
    loudly at ingress rather than propagating into plausible-looking but
    incorrect estimates.

    Parameters
    ----------
    U_imu : array-like
        IMU measurements intended to be passed into the filter.
    layout : StateLayout
        State layout used by the filter. Only ``vel_idx`` is inspected, to
        catch the specific case of a 4-channel (3D IMU) array combined with a
        2D-velocity state layout (which silently drops ``f_z``).
    func_name : str, optional
        Name of the caller, used as a prefix in error messages.

    Raises
    ------
    ValueError
        If ``U_imu`` is not 2-D, has an unsupported channel count, or carries
        channels into a layout that would misinterpret or silently drop them.

    Notes
    -----
    Expected channel conventions (matching ``dynamics_function`` in this
    module):

    - 3 channels ``[ω_z (rad/s), f_x (m/s²), f_y (m/s²)]`` — runs the 2D
      branch. Valid for non-quaternion layouts.
    - 4 channels ``[ω_z, f_x, f_y, f_z]`` — runs the 3D branch. Valid only
      when ``layout`` has 3D velocity (e.g. ``LAYOUT_2D_CAM_3D_IMU``).
    - 6 channels ``[ω_x, ω_y, ω_z, f_x, f_y, f_z]`` — runs the experimental
      6-DOF orientation branch. Valid only for quaternion-orientation layouts,
      including ``3d_cam_6dof_imu``.
    """
    arr = np.asarray(U_imu)

    if arr.ndim != 2:
        raise ValueError(
            f"{func_name}: U_imu must be a 2-D array of shape "
            f"(N_imu, n_channels); got ndim={arr.ndim}, shape={arr.shape}."
        )

    # Length and finiteness checks. The filter pre-integration loop
    # indexes ``U_imu_jax[imu_idx]`` where imu_idx comes from
    # compute_imu_index_arrays(t_imu, t_cam); a length mismatch would
    # silently clamp via JAX's out-of-bounds clamp, and a NaN/inf
    # sample would propagate through every downstream state from that
    # frame onward. The CLI already enforces these via
    # validate_finite_array / validate_monotonic_timestamps; mirror
    # them here so direct Python-API callers get the same contract.
    if t_imu is not None:
        t_arr = np.asarray(t_imu)
        if t_arr.ndim != 1:
            raise ValueError(f"{func_name}: t_imu must be 1D, got shape {t_arr.shape}.")
        if arr.shape[0] != t_arr.shape[0]:
            raise ValueError(
                f"{func_name}: U_imu.shape[0]={arr.shape[0]} does not match "
                f"len(t_imu)={t_arr.shape[0]}."
            )
    if not np.all(np.isfinite(arr)):
        bad = ~np.isfinite(arr).all(axis=1)
        n_bad = int(bad.sum())
        first_bad = int(np.argmax(bad))
        raise ValueError(
            f"{func_name}: U_imu contains non-finite value(s) (NaN/inf) "
            f"in {n_bad} row(s); first offending row at index {first_bad}. "
            "The IMU integration path does not support partial-sample "
            "masking — clean U_imu before calling the filter."
        )

    got = arr.shape[1]
    has_3d_velocity = len(layout.vel_idx) >= 3
    has_quaternion_orientation = layout.has_quaternion_orientation

    if has_quaternion_orientation:
        if got == 6:
            return
        raise ValueError(
            f"{func_name}: state layout uses 6-DOF quaternion orientation and "
            f"requires 6-channel IMU [ω_x, ω_y, ω_z, f_x, f_y, f_z]; got "
            f"{got} channels."
        )

    if got in (3, 4):
        if got == 4 and not has_3d_velocity:
            raise ValueError(
                f"{func_name}: U_imu has 4 channels [ω_z, f_x, f_y, f_z] but "
                f"the state layout has only 2D velocity; f_z would be "
                f"silently dropped. Use a 3D-velocity state_mode (e.g. "
                f"'2d_cam_3d_imu') or slice U_imu to 3 channels "
                f"[ω_z, f_x, f_y]."
            )
        return

    # Wrong channel count — build a helpful message.
    msg = (
        f"{func_name}: U_imu has {got} channels; expected 3 "
        f"[ω_z, f_x, f_y] (2D IMU), 4 [ω_z, f_x, f_y, f_z] (3D IMU), "
        f"or 6 [ω_x, ω_y, ω_z, f_x, f_y, f_z] for quaternion orientation. "
        f"Selected state_mode maps to a "
        f"{'3D-velocity' if has_3d_velocity else '2D-velocity'} layout."
    )
    if got == 6:
        # Most common real-data tripwire: load_arthur_session(mode='3d')
        # returns 6 channels [ω_x, ω_y, ω_z, f_x, f_y, f_z]; the filter wants
        # only the yaw gyro + accel triad.
        if has_3d_velocity:
            msg += (
                " Hint: load_arthur_session(mode='3d') returns 6 channels "
                "[ω_x, ω_y, ω_z, f_x, f_y, f_z]; select columns [2, 3, 4, 5] "
                "to get [ω_z, f_x, f_y, f_z] before passing to the filter."
            )
        else:
            msg += (
                " Hint: load_arthur_session(mode='3d') returns 6 channels; "
                "select columns [2, 3, 4] to get [ω_z, f_x, f_y] for 2D "
                "filtering, or pass mode='2d' when loading."
            )
    raise ValueError(msg)


def validate_timestamps(
    t,
    *,
    name: str,
    func_name: str = "Kalman filter",
    min_size: int = 1,
) -> None:
    """Reject timestamp arrays that are non-finite or not strictly increasing.

    Mirrors the CLI's :func:`trodestrack.cli.utils.validate_monotonic_timestamps`
    so the same contract is enforced when callers reach the filter through
    the Python API rather than the CLI. ``np.diff(t)`` is used to derive
    sample periods inside ``compute_imu_index_arrays`` and the IMU
    pre-integration step; non-finite or decreasing entries produce NaN /
    negative dt and silently poison the filter outputs.

    Parameters
    ----------
    t : array-like
        Timestamp array (N,) in seconds.
    name : str
        Field name for error messages (e.g. ``"t_imu"`` / ``"t_cam"``).
    func_name : str, optional
        Caller name for error-message prefixing.
    min_size : int, default 1
        Minimum acceptable array length. Pass ``min_size=2`` for IMU
        timestamps — the filter computes ``dt_imu_mean = mean(diff(t_imu))``
        which is NaN for a single-sample array and silently poisons every
        downstream prediction.

    Raises
    ------
    ValueError
        If ``t`` is not 1-D, contains a non-finite value, is shorter than
        ``min_size``, or is not strictly increasing.
    """
    arr = np.asarray(t)
    if arr.ndim != 1:
        raise ValueError(f"{func_name}: {name} must be 1D, got shape {arr.shape}.")
    if arr.size < min_size:
        if min_size == 1:
            raise ValueError(
                f"{func_name}: {name} must have at least one sample, got "
                f"shape {arr.shape}."
            )
        raise ValueError(
            f"{func_name}: {name} must have at least {min_size} samples to "
            "derive a sample period (mean(diff(t)) is NaN for shorter "
            f"arrays); got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"{func_name}: {name} contains {n_bad} non-finite value(s) "
            "(NaN/inf); timestamps must be finite seconds."
        )
    if arr.size >= 2:
        diffs = np.diff(arr)
        if not np.all(diffs > 0):
            first_bad = int(np.argmax(diffs <= 0))
            raise ValueError(
                f"{func_name}: {name} must be strictly increasing; first "
                f"non-increasing step at index {first_bad + 1} "
                f"({name}[{first_bad}]={arr[first_bad]!r}, "
                f"{name}[{first_bad + 1}]={arr[first_bad + 1]!r}, "
                f"dt={diffs[first_bad]!r})."
            )


def validate_camera_input_shapes(
    t_cam,
    Z_cam_led1,
    Z_cam_led2,
    mask_cam,
    *,
    conf_cam=None,
    func_name: str = "Kalman filter",
) -> None:
    """Validate that all camera-aligned arrays match ``len(t_cam)``.

    Why this exists: the camera measurement models (``CameraPositionModel``
    and friends) index ``Z_cam_led*[frame_idx]``, ``mask_cam[frame_idx]``,
    and ``conf_cam[frame_idx]`` with a JAX scalar. Out-of-bounds JAX
    indexing silently clamps to the last in-range element, so a
    too-short camera-aligned array reuses its last row for every later
    frame and the filter returns finite but wrong outputs. Catch the
    mismatch at the Python entrypoint instead.

    Parameters
    ----------
    t_cam : array-like
        Camera timestamps; defines the expected frame count ``N_cam``.
    Z_cam_led1, Z_cam_led2 : array-like
        LED position arrays. Must have shape ``(N_cam, 2)``.
    mask_cam : array-like
        Per-frame validity mask. Must have shape ``(N_cam,)``.
    conf_cam : array-like or None, optional
        Per-frame confidence array. If provided, must have shape
        ``(N_cam, 4)`` (``[x1, y1, x2, y2]`` weights).
    func_name : str, optional
        Caller name for error-message prefixing.

    Raises
    ------
    ValueError
        If ``t_cam`` is not 1-D or any camera-aligned array has a shape
        that does not match ``(N_cam, ...)``.
    """
    t_cam_arr = np.asarray(t_cam)
    if t_cam_arr.ndim != 1:
        raise ValueError(f"{func_name}: t_cam must be 1D, got shape {t_cam_arr.shape}.")
    n_cam = int(t_cam_arr.shape[0])

    led1_arr = np.asarray(Z_cam_led1)
    if led1_arr.shape != (n_cam, 2):
        raise ValueError(
            f"{func_name}: Z_cam_led1 must have shape ({n_cam}, 2) to match "
            f"t_cam, got {led1_arr.shape}."
        )

    led2_arr = np.asarray(Z_cam_led2)
    if led2_arr.shape != (n_cam, 2):
        raise ValueError(
            f"{func_name}: Z_cam_led2 must have shape ({n_cam}, 2) to match "
            f"t_cam, got {led2_arr.shape}."
        )

    mask_arr = np.asarray(mask_cam)
    if mask_arr.shape != (n_cam,):
        raise ValueError(
            f"{func_name}: mask_cam must have shape ({n_cam},) to match "
            f"t_cam, got {mask_arr.shape}."
        )
    # Tighten the mask contract at the Python API: only bool-typed masks
    # or 0/1 integer masks are accepted. Initialization uses mask_cam in
    # bitwise expressions (mask & finite-row-check), so mask=2 silently
    # changes initialization semantics while still producing finite
    # (but wrong) outputs. The CLI already enforces this via its own
    # camera-mask loader; mirror the contract for direct API callers.
    if mask_arr.dtype != np.bool_:
        if not np.issubdtype(mask_arr.dtype, np.integer):
            raise ValueError(
                f"{func_name}: mask_cam must be boolean or 0/1 integer; "
                f"got dtype {mask_arr.dtype!r}."
            )
        if not np.all(np.isin(mask_arr, (0, 1))):
            bad = mask_arr[~np.isin(mask_arr, (0, 1))]
            raise ValueError(
                f"{func_name}: mask_cam must contain only 0 or 1 (or be "
                f"boolean); found {len(bad)} other value(s) "
                f"(e.g. {bad[:5].tolist()})."
            )

    if conf_cam is not None:
        conf_arr = np.asarray(conf_cam)
        if conf_arr.shape != (n_cam, 4):
            raise ValueError(
                f"{func_name}: conf_cam must have shape ({n_cam}, 4) "
                "([x1, y1, x2, y2] per frame) to match t_cam, got "
                f"{conf_arr.shape}."
            )
        # Confidence is later clipped to [1e-2, 1.0] and used as the
        # denominator in confidence_to_R_diagonal (R = base / conf). NaN
        # / inf survive np.clip and propagate into R, the innovation
        # covariance, and every downstream state. With gating disabled
        # there is no second-line defense, so reject non-finite
        # confidence at ingress.
        if not np.all(np.isfinite(conf_arr)):
            n_bad = int(np.sum(~np.isfinite(conf_arr)))
            raise ValueError(
                f"{func_name}: conf_cam contains {n_bad} non-finite "
                "value(s) (NaN/inf); confidence must be finite in [0, 1]."
            )


def validate_camera_3d_input_shapes(
    t_cam,
    Z_cam_leds,
    led_offsets_body,
    *,
    mask_cam_leds=None,
    conf_cam=None,
    func_name: str = "Kalman filter (3D)",
) -> None:
    """Validate 3D camera-aligned arrays against ``len(t_cam)`` and offsets.

    Sibling of :func:`validate_camera_input_shapes` for the experimental
    3D camera path (``extended_kalman_filter_3d``), where the LED-position
    array is ``(n_cam, n_leds, 3)`` rather than the 2D contract's
    ``(n_cam, 2)``. Without this guard, JAX out-of-bounds indexing would
    silently clamp a too-short ``Z_cam_leds`` (or ``mask_cam_leds`` /
    ``conf_cam``) to its last in-range row and the filter would return
    finite-but-wrong outputs (verified: a 4-frame ``t_cam`` with
    ``Z_cam_leds.shape == (1, 3, 3)`` previously produced a finite
    ``(4, 16)`` filtered_means).

    Parameters
    ----------
    t_cam : array-like
        Camera timestamps; defines ``N_cam``.
    Z_cam_leds : array-like
        3D LED observations. Must have shape ``(N_cam, n_leds, 3)`` matching
        ``led_offsets_body``.
    led_offsets_body : array-like
        Body-frame LED offsets, ``(n_leds, 3)``.
    mask_cam_leds : array-like or None, optional
        Per-frame, per-LED validity mask; must have shape ``(N_cam, n_leds)``.
    conf_cam : array-like or None, optional
        Per-frame confidence. The 3D camera model accepts either
        ``(N_cam, n_leds)`` or ``(N_cam, n_leds, 3)``. Non-finite values
        are rejected (they propagate through R = base/conf otherwise).
    func_name : str, optional
        Caller name for error-message prefixing.

    Raises
    ------
    ValueError
        On any shape or finiteness mismatch.
    """
    t_cam_arr = np.asarray(t_cam)
    if t_cam_arr.ndim != 1:
        raise ValueError(f"{func_name}: t_cam must be 1D, got shape {t_cam_arr.shape}.")
    n_cam = int(t_cam_arr.shape[0])

    offsets = np.asarray(led_offsets_body)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError(
            f"{func_name}: led_offsets_body must have shape (n_leds, 3); "
            f"got {offsets.shape}."
        )
    # Camera3DPositionModel.predict rotates these offsets into world frame
    # and adds them to the predicted body position; a single NaN entry
    # poisons every predicted LED, every residual, R, and the loglik.
    if not np.all(np.isfinite(offsets)):
        n_bad = int(np.sum(~np.isfinite(offsets)))
        raise ValueError(
            f"{func_name}: led_offsets_body contains {n_bad} non-finite "
            "value(s) (NaN/inf); offsets must be finite body-frame meters."
        )
    n_leds = int(offsets.shape[0])

    leds = np.asarray(Z_cam_leds)
    if leds.shape != (n_cam, n_leds, 3):
        raise ValueError(
            f"{func_name}: Z_cam_leds must have shape ({n_cam}, {n_leds}, 3) "
            f"to match t_cam and led_offsets_body, got {leds.shape}."
        )

    if mask_cam_leds is not None:
        mask_arr = np.asarray(mask_cam_leds)
        if mask_arr.shape != (n_cam, n_leds):
            raise ValueError(
                f"{func_name}: mask_cam_leds must have shape "
                f"({n_cam}, {n_leds}) to match t_cam and led_offsets_body, "
                f"got {mask_arr.shape}."
            )
        # Mirror the 2D mask_cam contract: only bool or 0/1 integer values
        # are accepted. The downstream call jnp.asarray(..., dtype=bool)
        # would otherwise silently coerce 2/-1/NaN to True and treat
        # invalid LEDs as visible.
        if mask_arr.dtype != np.bool_:
            if not np.issubdtype(mask_arr.dtype, np.integer):
                raise ValueError(
                    f"{func_name}: mask_cam_leds must be boolean or 0/1 "
                    f"integer; got dtype {mask_arr.dtype!r}."
                )
            if not np.all(np.isin(mask_arr, (0, 1))):
                bad = mask_arr[~np.isin(mask_arr, (0, 1))]
                raise ValueError(
                    f"{func_name}: mask_cam_leds must contain only 0 or 1 "
                    f"(or be boolean); found {len(bad)} other value(s) "
                    f"(e.g. {bad[:5].tolist()})."
                )

    if conf_cam is not None:
        conf_arr = np.asarray(conf_cam)
        if conf_arr.shape not in ((n_cam, n_leds), (n_cam, n_leds, 3)):
            raise ValueError(
                f"{func_name}: conf_cam must have shape ({n_cam}, {n_leds}) "
                f"or ({n_cam}, {n_leds}, 3), got {conf_arr.shape}."
            )
        if not np.all(np.isfinite(conf_arr)):
            n_bad = int(np.sum(~np.isfinite(conf_arr)))
            raise ValueError(
                f"{func_name}: conf_cam contains {n_bad} non-finite "
                "value(s) (NaN/inf); confidence must be finite in [0, 1]."
            )


def wrap_angle(theta: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to (-π, π] using numerically stable trigonometric method.

    JAX version of the canonical wrap_angle implementation in sim/utils.py.
    Uses atan2(sin(θ), cos(θ)) for numerical stability.

    Parameters
    ----------
    theta : jnp.ndarray
        Angle(s) in radians, arbitrary shape.

    Returns
    -------
    jnp.ndarray
        Wrapped angle(s), same shape as input.

    Notes
    -----
    This implementation is identical to sim/utils.wrap_angle but uses JAX
    arrays for JIT compilation and automatic differentiation.

    See Also
    --------
    trodestrack.sim.utils.wrap_angle : NumPy version
    """
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def rotate_body_accel_to_world(
    accel_body: jnp.ndarray, yaw_heading: float | jnp.ndarray
) -> jnp.ndarray:
    """Rotate 3D body-frame acceleration to world frame using yaw angle.

    Parameters
    ----------
    accel_body : jnp.ndarray
        Acceleration in body frame, shape (3,), as ``[ax, ay, az]`` in m/s².
    yaw_heading : float or jnp.ndarray
        Yaw heading angle (rad). Rotation about vertical (z) axis.

    Returns
    -------
    jnp.ndarray
        Acceleration in world frame, shape (3,), as ``[ax_w, ay_w, az_w]`` in m/s².

    Notes
    -----
    Applies R_z(θ) rotation matrix to the x-y plane while preserving z::

        [ax_w]   [cos(θ)  -sin(θ)  0] [ax]
        [ay_w] = [sin(θ)   cos(θ)  0] [ay]
        [az_w]   [0        0       1] [az]

    This is the correct transformation for converting IMU measurements from
    the body frame (where the IMU is mounted on the rat) to the world frame
    (the fixed laboratory coordinate system).
    """
    yaw = jnp.asarray(yaw_heading)
    cos_yaw = jnp.cos(yaw)
    sin_yaw = jnp.sin(yaw)

    # Rotation matrix R_z(yaw) for yaw-only rotation (3x3)
    # Only affects x-y plane; z is unchanged
    R_z = jnp.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
    )

    return R_z @ accel_body


def state_yaw(state: jnp.ndarray, layout: StateLayout) -> jnp.ndarray:
    """Return yaw angle from either scalar-heading or quaternion state layout."""

    if layout.has_heading_2d:
        return state[get_heading_index(layout)]
    if layout.has_quaternion_orientation:
        quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
        return quaternion_to_yaw(state[quat_idx])
    raise NotImplementedError(
        "Yaw extraction is implemented for 2D heading and quaternion layouts only."
    )


def normalize_state_orientation(
    state: jnp.ndarray,
    layout: StateLayout,
) -> jnp.ndarray:
    """Normalize quaternion state components when the layout has them."""

    if not layout.has_quaternion_orientation:
        return state
    quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
    quat = normalize_quaternion(state[quat_idx])
    return state.at[quat_idx].set(quat)


def gravity_compensate(accel_world: jnp.ndarray, g: float = 9.81) -> jnp.ndarray:
    """Remove gravity from world-frame acceleration.

    Parameters
    ----------
    accel_world : jnp.ndarray
        Acceleration in world frame (3,) [ax, ay, az] in m/s².
    g : float, default 9.81
        Gravitational acceleration magnitude (m/s²).

    Returns
    -------
    jnp.ndarray
        Gravity-compensated acceleration (3,) in m/s².

    Notes
    -----
    Removes the gravitational acceleration vector [0, 0, g] from the
    world-frame measurement. This is necessary because IMUs measure
    specific force (proper acceleration), which includes gravity.

    **IMU Convention:** Assumes IMU reports specific force (proper acceleration).
    Kinematic acceleration is: a = R f_body - g_world

    The IMU at rest reads [0, 0, +g] due to the normal force from the
    surface. Motion creates additional accelerations that add to this
    baseline. Subtracting [0, 0, g] recovers the kinematic acceleration
    (coordinate acceleration) needed for state propagation.
    """
    gravity_vector = jnp.array([0.0, 0.0, g])
    return accel_world - gravity_vector


def chi2_threshold(dof: int, prob: float) -> jnp.ndarray:
    """Closed-form χ² thresholds for common degrees of freedom.

    Parameters
    ----------
    dof : int
        Degrees of freedom, {2, 4} for single-LED vs dual-LED measurements.
    prob : float
        Probability mass for threshold, one of {0.95, 0.975, 0.99, 0.997}.

    Returns
    -------
    jnp.ndarray
        χ² threshold value (scalar).

    Notes
    -----
    Values match ``scipy.stats.chi2.ppf(prob, dof)`` for the listed pairs.
    - dof=2: Mahalanobis distance² for single LED (x, y)
    - dof=4: Mahalanobis distance² for dual LEDs (x1, y1, x2, y2)
    """
    # Thresholds for dof=2 (single LED)
    threshold_2 = lax.select(
        jnp.abs(prob - 0.997) < 0.001,
        11.618,  # 99.7% (3σ) - very tight gate
        lax.select(
            jnp.abs(prob - 0.99) < 0.001,
            9.210,  # 99% - tight gate
            lax.select(
                jnp.abs(prob - 0.975) < 0.001,
                7.378,  # 97.5% (2.5σ) - moderate gate
                5.991,  # 95% (2σ) - loose gate (default fallback)
            ),
        ),
    )

    # Thresholds for dof=4 (dual LEDs)
    threshold_4 = lax.select(
        jnp.abs(prob - 0.997) < 0.001,
        16.014,  # 99.7% (3σ) - very tight gate
        lax.select(
            jnp.abs(prob - 0.99) < 0.001,
            13.277,  # 99% - tight gate
            lax.select(
                jnp.abs(prob - 0.975) < 0.001,
                11.143,  # 97.5% (2.5σ) - moderate gate
                9.488,  # 95% (2σ) - loose gate (default fallback)
            ),
        ),
    )

    return lax.select(dof == 2, threshold_2, threshold_4)


def dynamics_function(
    state: jnp.ndarray,
    imu: jnp.ndarray,
    dt: float,
    damping: float,
    layout: StateLayout,
    gravity_body: tuple[float, float, float] | jnp.ndarray | None = None,
    enable_experimental_accel_translation: bool = False,
) -> jnp.ndarray:
    """Constant-acceleration dynamics with linear damping (layout-aware).

    Supports 2D, yaw+3D-accel, and full 6-axis orientation IMU inputs:
    - 2D IMU: [ω_z(rad/s), f_x(m/s²), f_y(m/s²)] (3,)
    - 3D IMU: [ω_z(rad/s), f_x(m/s²), f_y(m/s²), f_z(m/s²)] (4,)
    - 6-DOF IMU: [ω_x, ω_y, ω_z, f_x, f_y, f_z] (6,)

    For 3D IMU, applies gravity compensation and 3D rotation.

    Parameters
    ----------
    state : jnp.ndarray
        State vector (n,). Layout-dependent structure.
    imu : jnp.ndarray
        IMU measurements. Either 3-element (2D) or 4-element (3D).
    dt : float
        Time step (s).
    damping : float
        Linear velocity damping coefficient (1/s).
    layout : StateLayout
        State index mapping.
    gravity_body : tuple[float, float, float] or jnp.ndarray, optional
        Expected gravity vector in the tracking/world frame (m/s²). The
        parameter keeps its legacy name for API compatibility. Defaults to
        ``[0, 0, 9.81]`` for level mounting.
    enable_experimental_accel_translation : bool, default False
        For quaternion orientation layouts, whether to integrate accelerometer
        samples into x/y velocity. Default False keeps position dynamics
        camera/constant-velocity driven.

    Returns
    -------
    jnp.ndarray
        Next state (n,).

    Notes
    -----
    For 2D IMU mode:
        vₖ₊₁ = vₖ + (R₂ₓ₂ f − γ vₖ) dt
        pₖ₊₁ = pₖ + vₖ dt + 1/2 (R₂ₓ₂ f − γ vₖ) dt²

    For 3D IMU mode:
        f_world = R₃ₓ₃(θ) @ f_body
        a_kinematic = f_world - gravity_world  (gravity compensation)
        vₖ₊₁ = vₖ + (a_kinematic − γ vₖ) dt
        pₖ₊₁ = pₖ + vₖ dt + 1/2 (a_kinematic − γ vₖ) dt²

    Heading update:
        θₖ₊₁ = θₖ + (ω_z − b_gz) dt

    Unused layout components are propagated as identity.
    """

    px_i, py_i = layout.pos_idx[0], layout.pos_idx[1]
    vx_i, vy_i = layout.vel_idx[0], layout.vel_idx[1]

    # Check if layout has 3D velocity (vz)
    has_3d_velocity = len(layout.vel_idx) >= 3

    # Bias indices (may be empty for vision-only)
    b_gz = state[layout.bias_gyro_idx[0]] if len(layout.bias_gyro_idx) >= 1 else 0.0
    b_ax = state[layout.bias_accel_idx[0]] if len(layout.bias_accel_idx) >= 1 else 0.0
    b_ay = state[layout.bias_accel_idx[1]] if len(layout.bias_accel_idx) >= 2 else 0.0
    b_az = state[layout.bias_accel_idx[2]] if len(layout.bias_accel_idx) >= 3 else 0.0

    # Current values
    px, py = state[px_i], state[py_i]
    vx, vy = state[vx_i], state[vy_i]

    # Vision-only layouts do not integrate IMU channels. During dropouts they
    # extrapolate from camera-derived velocity only.
    if not layout.has_biases:
        next_state = state
        next_state = next_state.at[px_i].set(px + vx * dt)
        next_state = next_state.at[py_i].set(py + vy * dt)
        return next_state

    gravity_world_vec = (
        jnp.array([0.0, 0.0, 9.81])
        if gravity_body is None
        else jnp.asarray(gravity_body)
    )

    if layout.has_quaternion_orientation:
        quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
        gyro_bias = jnp.array(
            [
                state[layout.bias_gyro_idx[0]],
                state[layout.bias_gyro_idx[1]],
                state[layout.bias_gyro_idx[2]],
            ]
        )
        accel_bias = jnp.array(
            [
                state[layout.bias_accel_idx[0]],
                state[layout.bias_accel_idx[1]],
                state[layout.bias_accel_idx[2]],
            ]
        )

        omega_body = imu[:3] - gyro_bias
        quat_next = integrate_body_gyro(state[quat_idx], omega_body, dt)

        pos_idx = jnp.array(layout.pos_idx, dtype=jnp.int32)
        vel_idx = jnp.array(layout.vel_idx, dtype=jnp.int32)
        pos = state[pos_idx]
        vel = state[vel_idx]
        use_accel_translation = enable_experimental_accel_translation
        if use_accel_translation:
            expected_gravity_body = rotate_vector_world_to_body(
                quat_next,
                gravity_world_vec.astype(state.dtype),
            )
            accel_body_kinematic = imu[3:6] - accel_bias - expected_gravity_body
            accel_world = rotate_vector_body_to_world(quat_next, accel_body_kinematic)
            accel = accel_world[: layout.spatial_dim]
            vel_next = vel + accel * dt - damping * vel * dt
            pos_next = (
                pos + vel * dt + 0.5 * accel * dt**2 - 0.5 * damping * vel * dt**2
            )
        else:
            vel_next = vel
            pos_next = pos + vel * dt

        next_state = state
        next_state = next_state.at[pos_idx].set(pos_next)
        next_state = next_state.at[vel_idx].set(vel_next)
        next_state = next_state.at[quat_idx].set(quat_next)
        return next_state

    # Detect IMU dimension: 3-element (2D) or 4-element (3D)
    imu_is_3d = imu.shape[0] >= 4

    # Update heading
    h_idx = get_heading_index(layout)
    theta = state[h_idx]
    omega_z = imu[0]
    omega_z_unbiased = omega_z - b_gz
    theta_next = theta + omega_z_unbiased * dt

    if imu_is_3d and has_3d_velocity:
        # 3D IMU mode: [ω_z, fx, fy, fz]
        # Consistency check: 3D velocity requires 3D accel bias
        if len(layout.bias_accel_idx) < 3:
            raise ValueError(
                f"3D IMU mode requires 3D accel bias (b_ax, b_ay, b_az), "
                f"but layout has only {len(layout.bias_accel_idx)} accel bias terms. "
                f"Use LAYOUT_2D_CAM_3D_IMU or ensure bias_accel_idx has length 3."
            )

        # Extract 3D velocity
        vz_i = layout.vel_idx[2]
        vz = state[vz_i]

        # Extract 3D accelerations from IMU
        fx, fy, fz = imu[1], imu[2], imu[3]

        # Remove biases, rotate to world frame, then subtract calibrated world gravity.
        accel_body = jnp.array([fx - b_ax, fy - b_ay, fz - b_az])
        accel_world = rotate_body_accel_to_world(accel_body, theta)
        accel_kinematic = accel_world - gravity_world_vec

        # Update 3D velocity with damping
        vel = jnp.array([vx, vy, vz])
        vel_next = vel + accel_kinematic * dt - damping * vel * dt

        # Update 2D position (only x, y; no z position in LAYOUT_2D_CAM_3D_IMU)
        pos = jnp.array([px, py])
        # Only use horizontal components for position update
        accel_horizontal = accel_kinematic[:2]
        vel_horizontal = vel[:2]
        pos_next = (
            pos
            + vel_horizontal * dt
            + 0.5 * accel_horizontal * dt**2
            - 0.5 * damping * vel_horizontal * dt**2
        )

        # Update state
        next_state = state
        next_state = next_state.at[px_i].set(pos_next[0])
        next_state = next_state.at[py_i].set(pos_next[1])
        next_state = next_state.at[vx_i].set(vel_next[0])
        next_state = next_state.at[vy_i].set(vel_next[1])
        next_state = next_state.at[vz_i].set(vel_next[2])
        next_state = next_state.at[h_idx].set(theta_next)

    else:
        # 2D IMU mode: [ω_z, fx, fy] (backward compatible)
        fx, fy = imu[1], imu[2]
        accel_body = jnp.array([fx - b_ax, fy - b_ay, 0.0])

        # 2D rotation (yaw only)
        accel_world_3d = rotate_body_accel_to_world(accel_body, theta)
        accel_world = accel_world_3d[:2] - gravity_world_vec[:2]

        # Update 2D velocity
        vel = jnp.array([vx, vy])
        vel_next = vel + accel_world * dt - damping * vel * dt

        # Update 2D position
        pos = jnp.array([px, py])
        pos_next = (
            pos + vel * dt + 0.5 * accel_world * dt**2 - 0.5 * damping * vel * dt**2
        )

        # Update state
        next_state = state
        next_state = next_state.at[px_i].set(pos_next[0])
        next_state = next_state.at[py_i].set(pos_next[1])
        next_state = next_state.at[vx_i].set(vel_next[0])
        next_state = next_state.at[vy_i].set(vel_next[1])
        next_state = next_state.at[h_idx].set(theta_next)

    return next_state


def build_quaternion_transition_jacobian(
    linearization_mean: jnp.ndarray,
    linearization_pred: jnp.ndarray,
    dt_imu: float | Array,
    damping_coeff: float,
    layout: StateLayout,
    *,
    u_imu: jnp.ndarray | None = None,
    enable_experimental_accel_translation: bool = False,
) -> jnp.ndarray:
    """Build the first-order transition Jacobian for quaternion layouts.

    Parameters
    ----------
    linearization_mean : jnp.ndarray
        State at the start of the IMU step, shape ``(n,)``.
    linearization_pred : jnp.ndarray
        Propagated linearization state after the IMU step, shape ``(n,)``.
    dt_imu : float or Array
        IMU timestep in seconds.
    damping_coeff : float
        Linear velocity damping coefficient.
    layout : StateLayout
        Quaternion state layout.
    u_imu : jnp.ndarray or None, optional
        IMU sample at this step (gyro in the first three channels). When
        provided the quaternion-vs-quaternion block is set to the proper
        first-order Jacobian ``I_4 + 0.5·dt·Ω_R(ω)`` of
        ``q_next = q_prev ⊗ exp(ω_unbiased · dt)``. Without it the block
        defaults to identity, which is only correct at exactly zero
        rotation; passing ``None`` is supported for legacy callers but
        makes covariance / gating / smoothing inconsistent with the mean
        for non-zero gyro samples.
    enable_experimental_accel_translation : bool, default False
        Whether accelerometer-driven translation is active.

    Returns
    -------
    jnp.ndarray
        Transition Jacobian ``F_x`` with shape ``(n, n)``.
    """
    n = linearization_mean.shape[0]
    dtype = linearization_mean.dtype
    F_x = jnp.eye(n, dtype=dtype)
    dt_arr = jnp.asarray(dt_imu, dtype=dtype)

    if enable_experimental_accel_translation:
        vel_self = 1.0 - damping_coeff * dt_arr
        pos_vel = dt_arr - 0.5 * damping_coeff * dt_arr**2
    else:
        vel_self = 1.0
        pos_vel = dt_arr

    for pos_i, vel_i in zip(layout.pos_idx, layout.vel_idx, strict=True):
        F_x = F_x.at[vel_i, vel_i].set(vel_self)
        F_x = F_x.at[pos_i, vel_i].set(pos_vel)

    quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
    gyro_bias_idx = jnp.array(layout.bias_gyro_idx, dtype=jnp.int32)
    qw, qx, qy, qz = linearization_mean[quat_idx]

    # Quaternion blocks. The mean propagation is
    #   q_intermediate = q_prev ⊗ exp(ω_unbiased · dt)
    #   q_next         = normalize(q_intermediate)
    # with ω_unbiased = u_imu[gyro] - bias_gyro_state. The first-order
    # Jacobian of the un-normalized step is
    #   ∂q_intermediate/∂q_prev   = I_4 + 0.5·dt·Ω_R(ω_unbiased)
    #   ∂q_intermediate/∂bias_gyro = -0.5·dt · M_q(q_prev)
    # where Ω_R is the right-quaternion-product matrix and M_q is the
    # left-quaternion-product matrix used in the q ⊗ ω term. The
    # post-normalize step composes a unit-sphere projection
    # P = (I - q_pred q_predᵀ) on the left, since q_pred is unit-norm.
    # Without this projection the manual Jacobian disagreed with
    # jacfwd(dynamics_function) by O(1) for nonzero gyro samples
    # (verified: quaternion-self diagonal was 1.0 vs autodiff ~0.6–0.9
    # for a 0.5–0.7 rad/s sample at dt=5 ms).
    q_pred = linearization_pred[quat_idx]
    projector = jnp.eye(4, dtype=dtype) - jnp.outer(q_pred, q_pred)

    quat_gyro_matrix = jnp.array(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ],
        dtype=dtype,
    )
    bias_gyro_block_unnormalized = -0.5 * dt_arr * quat_gyro_matrix
    F_x = F_x.at[quat_idx[:, None], gyro_bias_idx[None, :]].set(
        projector @ bias_gyro_block_unnormalized
    )

    quat_self_block = projector
    if u_imu is not None:
        gyro_meas = jnp.asarray(u_imu, dtype=dtype)[:3]
        bias_gyro = linearization_mean[gyro_bias_idx]
        omega_unbiased = gyro_meas - bias_gyro
        wx, wy, wz = omega_unbiased[0], omega_unbiased[1], omega_unbiased[2]
        omega_right_matrix = jnp.array(
            [
                [0.0, -wx, -wy, -wz],
                [wx, 0.0, wz, -wy],
                [wy, -wz, 0.0, wx],
                [wz, wy, -wx, 0.0],
            ],
            dtype=dtype,
        )
        quat_self_unnormalized = (
            jnp.eye(4, dtype=dtype) + 0.5 * dt_arr * omega_right_matrix
        )
        quat_self_block = projector @ quat_self_unnormalized
        F_x = F_x.at[quat_idx[:, None], quat_idx[None, :]].set(quat_self_block)

    if enable_experimental_accel_translation:
        basis_body = jnp.eye(3, dtype=dtype)
        rotation_world_from_body = rotate_vector_body_to_world(
            linearization_pred[quat_idx],
            basis_body,
        ).T
        for dim, (pos_i, vel_i) in enumerate(
            zip(layout.pos_idx, layout.vel_idx, strict=True)
        ):
            for axis, bias_i in enumerate(layout.bias_accel_idx):
                coeff = -rotation_world_from_body[dim, axis]
                F_x = F_x.at[vel_i, bias_i].set(dt_arr * coeff)
                F_x = F_x.at[pos_i, bias_i].set(0.5 * dt_arr**2 * coeff)

        # Position / velocity dependence on the quaternion. The mean
        # dynamics rotate accel through ``q_next``:
        #   accel_world = R(q_next) · (imu_accel - bias) - g_world
        #   vel_next   = vel + accel_world[:dim] · dt - damping · vel · dt
        #   pos_next   = pos + vel · dt + 0.5 · accel_world[:dim] · dt² - …
        # so position and velocity *do* depend on q_next through R(q_next).
        # Without this block, EKF covariance propagation and RTS smoothing
        # silently dropped the pos/vel-vs-quat coupling (autodiff parity
        # check: max abs diff ≈ 0.37 with all-zero entries here vs
        # autodiff entries up to 0.366). Build d(accel_world)/d(q_next)
        # via a local jax.jacfwd of the rotation-of-accel lambda, then
        # compose with the same first-order q_prev -> q_next transition
        # block used above. Requires ``u_imu`` to read the accel measurement.
        if u_imu is not None and len(layout.bias_accel_idx) >= 3:
            accel_meas = jnp.asarray(u_imu, dtype=dtype)[3:6]
            bias_accel = linearization_mean[
                jnp.array(layout.bias_accel_idx, dtype=jnp.int32)
            ]
            a_meas = accel_meas - bias_accel  # body-frame, bias-removed

            def _accel_world_of_q(q4):
                # accel_world = R(q) @ a_meas - g_world. The constant
                # g_world drops out under d/dq, so we only need the
                # rotated component here. q4 is treated as unit-norm
                # post-integration (the projector is applied below).
                return rotate_vector_body_to_world(q4, a_meas)

            d_accel_world_d_q = jacfwd(_accel_world_of_q)(q_pred)
            # d_accel_world_d_q has shape (3, 4) with respect to q_next.
            # Compose it with the same q_prev -> q_next transition block used
            # above so this F_x block is also with respect to the prior state.
            d_accel_world_d_q_prev = d_accel_world_d_q @ quat_self_block

            for dim, (pos_i, vel_i) in enumerate(
                zip(layout.pos_idx, layout.vel_idx, strict=True)
            ):
                F_x = F_x.at[vel_i, quat_idx].set(dt_arr * d_accel_world_d_q_prev[dim])
                F_x = F_x.at[pos_i, quat_idx].set(
                    0.5 * dt_arr**2 * d_accel_world_d_q_prev[dim]
                )

    return F_x


def measurement_function(
    state: jnp.ndarray, led_distance: float, layout: StateLayout
) -> jnp.ndarray:
    """Project state into dual-LED measurement space (layout-aware).

    Parameters
    ----------
    state : jnp.ndarray
        State (n,).
    led_distance : float
        LED spacing (m).
    layout : StateLayout
        State index mapping.

    Returns
    -------
    jnp.ndarray
        Measurement vector (4,) ordered as [x1, y1, x2, y2] in meters.
    """

    px = state[layout.pos_idx[0]]
    py = state[layout.pos_idx[1]]
    theta = state_yaw(state, layout)
    dx = 0.5 * led_distance * jnp.cos(theta)
    dy = 0.5 * led_distance * jnp.sin(theta)
    return jnp.array([px - dx, py - dy, px + dx, py + dy])


def make_led_selector(only_led1: bool | Array, only_led2: bool | Array) -> Array:
    """Create 2×4 selector matrix for single-LED observations.

    Parameters
    ----------
    only_led1 : bool | Array
        Boolean value or boolean scalar array. True if only LED1 is valid.
    only_led2 : bool | Array
        Boolean value or boolean scalar array. True if only LED2 is valid.

    Returns
    -------
    Array
        Selector matrix ``M`` (2, 4) such that ``M @ [x1,y1,x2,y2]`` extracts
        the active LED's 2D subspace.
    """
    # LED1 selector: picks first 2 dimensions
    M_led1 = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    # LED2 selector: picks last 2 dimensions
    M_led2 = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    # Select based on which LED is valid
    return lax.select(only_led1, M_led1, M_led2)


def apply_lifted_inverse(
    S4: jnp.ndarray,
    w4: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> jnp.ndarray:
    """Apply effective inverse in active subspace then lift to 4D.

    Parameters
    ----------
    S4 : jnp.ndarray
        Innovation covariance in full space (4, 4).
    w4 : jnp.ndarray
        Vector to multiply (4,).
    both_leds : bool
        True if both LEDs are valid (4D update).
    only_led1 : bool
        True if only LED1 is valid (2D update).
    only_led2 : bool
        True if only LED2 is valid (2D update).

    Returns
    -------
    jnp.ndarray
        Result x = S_eff⁻¹ @ w in 4D with static shape (4,).

    Notes
    -----
    For single-LED updates, compute in the 2D active subspace and lift:

    x₄ = Mᵀ · (M S₄ Mᵀ)⁻¹ · (M w₄)
    """
    # 4D path: both LEDs valid
    x4_full = psd_solve(S4, w4)

    # 2D path: single LED valid
    M2 = make_led_selector(only_led1, only_led2)  # (2, 4)
    S2 = M2 @ S4 @ M2.T  # (2, 2) - subspace innovation covariance
    w2 = M2 @ w4  # (2,) - project to subspace
    x2 = psd_solve(S2, w2)  # (2,) - solve in subspace
    x4_lifted = M2.T @ x2  # (4,) - lift back to 4D

    # Select based on LED validity (both branches return same shape)
    return lax.select(both_leds, x4_full, x4_lifted)


def initialize_state(
    led1_obs: jnp.ndarray,
    led2_obs: jnp.ndarray,
    observation_mask: jnp.ndarray,
    dt_cam: float | jnp.ndarray,
    led_distance: float = 0.04,
    *,
    layout: StateLayout | None = None,
) -> FilterState:
    """Bootstrap filter state from early LED observations.

    Parameters
    ----------
    led1_obs : jnp.ndarray
        LED1 observations (N, 2) in meters.
    led2_obs : jnp.ndarray
        LED2 observations (N, 2) in meters.
    observation_mask : jnp.ndarray
        Observation validity mask (N,), boolean.
    dt_cam : float or jnp.ndarray
        Camera frame interval (s). JAX scalar allowed for JIT.
    led_distance : float, default 0.04
        LED spacing (m) used to infer heading when both LEDs visible.
    layout : StateLayout, optional
        State mapping; defaults to "2d_full" if not provided.

    Returns
    -------
    FilterState
        Initial mean (n,) and covariance (n, n).

    Notes
    -----
    If all observations are invalid, initializes near the origin with large
    uncertainty, allowing prediction-only filtering to proceed.
    """

    # Find frames with valid observation mask AND finite LED observations
    # (the mask alone isn't sufficient -- LEDs can be NaN even when observation_mask=True)
    led1_finite_mask = jnp.isfinite(led1_obs).all(axis=1)
    led2_finite_mask = jnp.isfinite(led2_obs).all(axis=1)
    any_led_finite = led1_finite_mask | led2_finite_mask
    valid_with_data = observation_mask & any_led_finite

    valid_indices = jnp.where(valid_with_data)[0]
    has_valid_obs = len(valid_indices) > 0
    first_valid = valid_indices[0] if has_valid_obs else 0

    # Check LED validity at first valid frame
    led1_valid = jnp.isfinite(led1_obs[first_valid]).all() if has_valid_obs else False
    led2_valid = jnp.isfinite(led2_obs[first_valid]).all() if has_valid_obs else False

    # Replace NaN with zero to prevent propagation (only used if marked invalid)
    pos_led1 = jnp.where(
        jnp.isfinite(led1_obs[first_valid]),
        led1_obs[first_valid],
        jnp.array([0.0, 0.0]),
    )
    pos_led2 = jnp.where(
        jnp.isfinite(led2_obs[first_valid]),
        led2_obs[first_valid],
        jnp.array([0.0, 0.0]),
    )

    pos_init = jnp.where(
        led1_valid & led2_valid,
        (pos_led1 + pos_led2) / 2.0,
        jnp.where(
            led1_valid,
            pos_led1,
            jnp.where(led2_valid, pos_led2, jnp.array([0.0, 0.0])),
        ),
    )

    def compute_velocity() -> jnp.ndarray:
        idx1 = valid_indices[0]
        idx2 = valid_indices[1]
        dt = (idx2 - idx1) * dt_cam

        led1_1, led2_1 = led1_obs[idx1], led2_obs[idx1]
        led1_2, led2_2 = led1_obs[idx2], led2_obs[idx2]

        led1_1_valid = jnp.isfinite(led1_1).all()
        led2_1_valid = jnp.isfinite(led2_1).all()
        led1_2_valid = jnp.isfinite(led1_2).all()
        led2_2_valid = jnp.isfinite(led2_2).all()

        pos1 = jnp.where(
            led1_1_valid & led2_1_valid,
            (led1_1 + led2_1) / 2.0,
            jnp.where(led1_1_valid, led1_1, led2_1),
        )
        pos2 = jnp.where(
            led1_2_valid & led2_2_valid,
            (led1_2 + led2_2) / 2.0,
            jnp.where(led1_2_valid, led1_2, led2_2),
        )
        return (pos2 - pos1) / dt

    # Only compute velocity if we have at least 2 valid frames
    vel_init = compute_velocity() if len(valid_indices) >= 2 else jnp.zeros(2)

    led_vec = pos_led2 - pos_led1
    heading_from_leds = jnp.arctan2(led_vec[1], led_vec[0])
    heading_init = jnp.where(led1_valid & led2_valid, heading_from_leds, 0.0)

    # Build 8D default mean/cov, then adapt to layout based on desired state_mode
    heading_std = jnp.where(led1_valid & led2_valid, jnp.pi / 4, jnp.pi / 2)

    mean8 = jnp.array(
        [
            pos_init[0],
            pos_init[1],
            vel_init[0],
            vel_init[1],
            heading_init,
            0.0,
            0.0,
            0.0,
        ]
    )

    # When *no* LED frame is valid we fall back to (0, 0) for position
    # and (0, 0) for velocity — that mean is arbitrary, so the prior has
    # to advertise that arbitrariness through a wide variance instead of
    # the tight 0.01 m / 0.1 m/s defaults used when at least one frame
    # observed an LED. Without this widening the docstring's "large
    # uncertainty" claim was wrong: an all-invalid LED stream produced
    # an over-confident origin prior (pos_var = 1e-4) that the filter
    # then anchored against.
    pos_var_valid = 0.01**2
    pos_var_uninformative = 10.0**2  # ~10 m std covers any reasonable arena
    pos_var = jnp.where(has_valid_obs, pos_var_valid, pos_var_uninformative)
    vel_var_valid = 0.1**2
    vel_var_uninformative = 1.0**2  # 1 m/s std covers any reasonable speed
    vel_var = jnp.where(has_valid_obs, vel_var_valid, vel_var_uninformative)

    cov8 = jnp.diag(
        jnp.array(
            [
                pos_var,
                pos_var,
                vel_var,
                vel_var,
                heading_std**2,
                0.05**2,
                0.1**2,
                0.1**2,
            ]
        )
    )

    # Determine layout (defaults to 2D full if not provided)
    layout = get_layout("2d_full") if layout is None else layout
    n = layout.n
    mean = jnp.zeros(n)
    cov = jnp.eye(n) * 1.0

    # Map 2D pos/vel/orientation
    mean = mean.at[layout.pos_idx[0]].set(mean8[0])
    mean = mean.at[layout.pos_idx[1]].set(mean8[1])
    mean = mean.at[layout.vel_idx[0]].set(mean8[2])
    mean = mean.at[layout.vel_idx[1]].set(mean8[3])
    if layout.has_heading_2d:
        mean = mean.at[get_heading_index(layout)].set(mean8[4])
    elif layout.has_quaternion_orientation:
        quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
        quat = quaternion_from_rotation_vector(jnp.array([0.0, 0.0, mean8[4]]))
        mean = mean.at[quat_idx].set(quat)
    else:
        raise NotImplementedError(
            "initialize_state supports scalar-heading and quaternion layouts only."
        )

    cov = cov.at[layout.pos_idx[0], layout.pos_idx[0]].set(cov8[0, 0])
    cov = cov.at[layout.pos_idx[1], layout.pos_idx[1]].set(cov8[1, 1])
    cov = cov.at[layout.vel_idx[0], layout.vel_idx[0]].set(cov8[2, 2])
    cov = cov.at[layout.vel_idx[1], layout.vel_idx[1]].set(cov8[3, 3])
    if layout.has_heading_2d:
        cov = cov.at[get_heading_index(layout), get_heading_index(layout)].set(
            cov8[4, 4]
        )
    elif layout.has_quaternion_orientation:
        for idx in layout.heading_idx:
            cov = cov.at[idx, idx].set(cov8[4, 4])

    # Bias variances if present
    for idx in layout.bias_gyro_idx:
        cov = cov.at[idx, idx].set(cov8[5, 5])
    if len(layout.bias_accel_idx) >= 1:
        cov = cov.at[layout.bias_accel_idx[0], layout.bias_accel_idx[0]].set(cov8[6, 6])
    if len(layout.bias_accel_idx) >= 2:
        cov = cov.at[layout.bias_accel_idx[1], layout.bias_accel_idx[1]].set(cov8[7, 7])
    if len(layout.bias_accel_idx) >= 3:
        cov = cov.at[layout.bias_accel_idx[2], layout.bias_accel_idx[2]].set(cov8[7, 7])

    return FilterState(mean=mean, cov=cov)


def update_zupt(
    state: FilterState,
    config: FilterCoreConfig,
    *,
    active: bool | jnp.ndarray = True,
    layout: StateLayout,
) -> tuple[FilterState, jnp.ndarray]:
    """Apply zero-velocity pseudo-measurement when nearly stationary.

    Parameters
    ----------
    state : FilterState
        Current state.
    config : FilterCoreConfig
        ZUPT parameters in config.
    active : bool | jnp.ndarray, optional
        Caller-side stationarity gate. Filters should pass measured
        stationarity evidence here (IMU quietness plus camera speed/context),
        not a predicate derived from the velocity state being updated.
    layout : StateLayout
        Explicit state layout used to construct the ZUPT measurement model.
        Required so callers cannot rely on a brittle state-dimension lookup
        that could silently miswire two layouts sharing the same ``n``.

    Returns
    -------
    tuple[FilterState, jnp.ndarray]
        Updated state and log-likelihood (scalar).

    Notes
    -----
    This function is a compatibility wrapper that uses ZUPTModel internally.
    It maintains the existing API while adopting the new sensor architecture.
    """

    from trodestrack.models.sensors.zupt import ZUPTModel

    mean, cov = state
    if mean.shape[0] != layout.n:
        raise ValueError(
            f"update_zupt: state has dim {mean.shape[0]} but layout.n={layout.n}. "
            f"Check that the FilterState was built from the same state_mode the "
            f"rest of the pipeline uses."
        )

    # Create ZUPT model (fully pure, no mutable state)
    zupt_model = ZUPTModel(
        enable_zupt=config.enable_zupt,
        measurement_noise=config.zupt_measurement_noise,
        layout=layout,
        dtype=mean.dtype,
    )

    # Extract measurement components (all pure functions, JIT-safe)
    meas_pred = zupt_model.predict(mean)
    is_active = jnp.asarray(active, dtype=bool) & jnp.asarray(
        config.enable_zupt, dtype=bool
    )

    def do_update(_: None) -> tuple[FilterState, jnp.ndarray]:
        H = zupt_model.jacobian(mean)
        R = jnp.eye(zupt_model.meas_dim, dtype=mean.dtype) * jnp.asarray(
            config.zupt_measurement_noise,
            dtype=mean.dtype,
        )
        innovation = zupt_model.innovation(frame_idx=0, meas_pred=meas_pred)

        # Standard Kalman update
        S = H @ cov @ H.T + R
        K = psd_solve(S, H @ cov).T

        mean_updated = mean + K @ innovation
        cov_updated = joseph_update(cov, K, H, R)

        # Log-likelihood
        log_det = jnp.linalg.slogdet(S)[1]
        innov_quad = innovation @ psd_solve(S, innovation)
        meas_dim = innovation.shape[0]
        log_likelihood = -0.5 * (meas_dim * jnp.log(2 * jnp.pi) + log_det + innov_quad)
        return FilterState(mean=mean_updated, cov=cov_updated), log_likelihood

    def no_update(_: None) -> tuple[FilterState, jnp.ndarray]:
        return state, jnp.asarray(0.0, dtype=mean.dtype)

    return lax.cond(is_active, do_update, no_update, operand=None)


def imu_stationary_zupt_gate(
    imu_samples: jnp.ndarray,
    valid_samples: jnp.ndarray,
    config: FilterCoreConfig,
    layout: StateLayout,
) -> jnp.ndarray:
    """Return True when IMU samples in a camera interval look stationary.

    The detector uses measured IMU quietness, not the filter's current velocity
    estimate. For 2D IMU inputs, the accelerometer gate expects horizontal
    acceleration near zero. For 3D/6-DOF inputs, it expects specific-force
    magnitude near gravity.
    """

    valid = jnp.asarray(valid_samples, dtype=bool)
    sample_count = jnp.sum(valid.astype(imu_samples.dtype))
    has_samples = sample_count > 0
    denom = jnp.maximum(sample_count, jnp.asarray(1.0, dtype=imu_samples.dtype))
    mean_sample = jnp.sum(jnp.where(valid[:, None], imu_samples, 0.0), axis=0) / denom

    if layout.has_quaternion_orientation:
        gyro = mean_sample[:3]
        accel = mean_sample[3:6]
        expected_accel_norm = jnp.linalg.norm(
            jnp.asarray(config.imu_gravity_body, dtype=imu_samples.dtype)
        )
    else:
        gyro = mean_sample[:1]
        if imu_samples.shape[1] >= 4 and len(layout.vel_idx) >= 3:
            accel = mean_sample[1:4]
            expected_accel_norm = jnp.linalg.norm(
                jnp.asarray(config.imu_gravity_body, dtype=imu_samples.dtype)
            )
        else:
            accel = mean_sample[1:3]
            expected_accel_norm = jnp.asarray(0.0, dtype=imu_samples.dtype)

    gyro_norm = jnp.linalg.norm(gyro)
    accel_norm_error = jnp.abs(jnp.linalg.norm(accel) - expected_accel_norm)
    gyro_ok = gyro_norm <= jnp.asarray(
        config.zupt_gyro_threshold_rad_s,
        dtype=imu_samples.dtype,
    )
    accel_ok = accel_norm_error <= jnp.asarray(
        config.zupt_accel_threshold_m_s2,
        dtype=imu_samples.dtype,
    )
    return has_samples & gyro_ok & accel_ok


def camera_stationary_zupt_gate_2d(
    t_cam: jnp.ndarray,
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    mask_cam: jnp.ndarray,
    frame_idx: jnp.ndarray,
    config: FilterCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(has_visual_speed, speed_is_stationary)`` for 2D camera frames."""

    lookback = jnp.asarray(
        config.zupt_camera_stationary_window_frames,
        dtype=frame_idx.dtype,
    )
    prev_idx = jnp.maximum(frame_idx - lookback, 0)
    has_prev = frame_idx >= lookback

    def camera_point(idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        led1 = z_led1[idx]
        led2 = z_led2[idx]
        led1_valid = jnp.isfinite(led1).all()
        led2_valid = jnp.isfinite(led2).all()
        count = led1_valid.astype(led1.dtype) + led2_valid.astype(led1.dtype)
        point = (
            jnp.where(led1_valid, led1, jnp.zeros_like(led1))
            + jnp.where(led2_valid, led2, jnp.zeros_like(led2))
        ) / jnp.maximum(count, jnp.asarray(1.0, dtype=led1.dtype))
        valid = mask_cam[idx] & (count > 0)
        return point, valid

    current_point, current_valid = camera_point(frame_idx)
    previous_point, previous_valid = camera_point(prev_idx)
    dt = jnp.maximum(
        t_cam[frame_idx] - t_cam[prev_idx],
        jnp.asarray(1e-6, dtype=t_cam.dtype),
    )
    speed = jnp.linalg.norm(current_point - previous_point) / dt
    has_visual_speed = has_prev & current_valid & previous_valid
    speed_is_stationary = speed <= jnp.asarray(
        config.zupt_velocity_threshold,
        dtype=speed.dtype,
    )
    return has_visual_speed, speed_is_stationary


def camera_stationary_zupt_gate_3d(
    t_cam: jnp.ndarray,
    z_leds: jnp.ndarray,
    valid_coords: jnp.ndarray,
    prev_valid_coords: jnp.ndarray,
    frame_idx: jnp.ndarray,
    config: FilterCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(has_visual_speed, speed_is_stationary)`` for 3D camera frames."""

    lookback = jnp.asarray(
        config.zupt_camera_stationary_window_frames,
        dtype=frame_idx.dtype,
    )
    prev_idx = jnp.maximum(frame_idx - lookback, 0)
    has_prev = frame_idx >= lookback
    n_leds = z_leds.shape[1]
    valid_current = valid_coords.reshape((n_leds, 3)).all(axis=1)
    valid_previous = prev_valid_coords.reshape((n_leds, 3)).all(axis=1)

    def centroid(
        frame: jnp.ndarray,
        valid_leds: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        count = jnp.sum(valid_leds.astype(frame.dtype))
        point = jnp.sum(
            jnp.where(valid_leds[:, None], frame, 0.0), axis=0
        ) / jnp.maximum(count, jnp.asarray(1.0, dtype=frame.dtype))
        return point, count > 0

    current_point, current_valid = centroid(z_leds[frame_idx], valid_current)
    previous_point, previous_valid = centroid(z_leds[prev_idx], valid_previous)
    dt = jnp.maximum(
        t_cam[frame_idx] - t_cam[prev_idx],
        jnp.asarray(1e-6, dtype=t_cam.dtype),
    )
    speed = jnp.linalg.norm(current_point - previous_point) / dt
    has_visual_speed = has_prev & current_valid & previous_valid
    speed_is_stationary = speed <= jnp.asarray(
        config.zupt_velocity_threshold,
        dtype=speed.dtype,
    )
    return has_visual_speed, speed_is_stationary


def update_zupt_visual_context(
    visual_speed_valid: jnp.ndarray,
    visual_stationary: jnp.ndarray,
    stationary_context_prev: jnp.ndarray,
    stationary_context_age_prev: jnp.ndarray,
    config: FilterCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Update visual stationary context with bounded dropout carry-forward."""

    missing_visual_age = stationary_context_age_prev + jnp.asarray(
        1,
        dtype=stationary_context_age_prev.dtype,
    )
    zupt_visual_hold_frames = jnp.asarray(
        config.zupt_visual_context_hold_frames,
        dtype=stationary_context_age_prev.dtype,
    )
    return lax.cond(
        visual_speed_valid,
        lambda _: (
            jnp.asarray(visual_stationary, dtype=bool),
            jnp.asarray(0, dtype=stationary_context_age_prev.dtype),
        ),
        lambda _: (
            jnp.asarray(stationary_context_prev, dtype=bool)
            & (missing_visual_age <= zupt_visual_hold_frames),
            missing_visual_age,
        ),
        operand=None,
    )


def confidence_to_R_diagonal(
    confidence: jnp.ndarray | None,
    *,
    base: float,
    size: int,
    clip_min: float = 1e-2,
) -> jnp.ndarray:
    """Map confidence scores to per-dimension measurement noise.

    Parameters
    ----------
    confidence : jnp.ndarray or None
        Confidence per measurement dimension, shape (size,) in [0, 1].
        If None, no scaling is applied.
    base : float
        Base variance per dimension (units^2).
    size : int
        Number of measurement dimensions.
    clip_min : float, default 1e-2
        Lower bound for confidence to avoid division by zero.

    Returns
    -------
    jnp.ndarray
        Diagonal entries of R (size,), where R_i = base / clip(conf_i, clip_min, 1).
    """
    if confidence is None:
        return jnp.full(size, base)
    conf = jnp.clip(confidence, clip_min, 1.0)
    return base / conf


def gaussian_log_likelihood(
    innovation: jnp.ndarray, covariance: jnp.ndarray
) -> jnp.ndarray:
    """Gaussian log-likelihood of an innovation with stability tweaks.

    Parameters
    ----------
    innovation : jnp.ndarray
        Innovation vector v (k,).
    covariance : jnp.ndarray
        Innovation covariance S (k, k).

    Returns
    -------
    jnp.ndarray
        Log-likelihood log p(v | 0, S) (scalar).

    Notes
    -----
    Computes ``-0.5 * (k log(2π) + log det S + vᵀ S⁻¹ v)`` with small diagonal
    jitter to improve conditioning.
    """
    k = innovation.shape[0]

    # Add small jitter to diagonal for numerical stability
    # Scale by mean diagonal value to be adaptive
    jitter = 1e-8 * jnp.trace(covariance) / k
    S_stable = symmetrize(covariance) + jnp.eye(k) * jitter

    # Log determinant using slogdet (more stable than det)
    sign, logdet = jnp.linalg.slogdet(S_stable)

    # Check for numerical issues (sign should be +1 for PSD matrix)
    # If sign <= 0, increase jitter and recompute
    def add_more_jitter():
        jitter_large = 1e-6 * jnp.trace(covariance) / k
        S_jittered = symmetrize(covariance) + jnp.eye(k) * jitter_large
        _sign_j, logdet_j = jnp.linalg.slogdet(S_jittered)
        return logdet_j

    # Use original logdet if sign is positive, otherwise use jittered version
    logdet_safe = lax.cond(sign > 0, lambda: logdet, add_more_jitter)

    # Mahalanobis distance: v^T S^{-1} v
    # psd_solve computes S^{-1} @ v, then we dot with v
    S_inv_v = psd_solve(S_stable, innovation)
    mahal = jnp.dot(innovation, S_inv_v)

    # Gaussian log-likelihood
    log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + logdet_safe + mahal)

    return log_prob


def gaussian_log_likelihood_masked(
    innovation: jnp.ndarray,
    covariance: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Gaussian log-likelihood over a masked subset of measurement rows.

    Inactive rows (where ``active_mask`` is False) get covariance replaced by
    identity and innovation replaced by zero, contributing exactly nothing
    to the returned log-likelihood. Used by sensor models with variable
    per-frame measurement dimensions (camera 3D LEDs, TTL events).

    Parameters
    ----------
    innovation : jnp.ndarray, shape (k,)
        Stacked innovation vector.
    covariance : jnp.ndarray, shape (k, k)
        Stacked innovation covariance.
    active_mask : jnp.ndarray, shape (k,)
        Per-row activity mask. ``True`` rows contribute to the likelihood.
    """
    active = active_mask.astype(bool)
    active_outer = active[:, None] & active[None, :]
    eye = jnp.eye(covariance.shape[0], dtype=covariance.dtype)
    covariance_masked = jnp.where(active_outer, covariance, eye)
    innovation_masked = jnp.where(active, innovation, 0.0)
    dim = jnp.sum(active_mask.astype(innovation.dtype))
    solved = psd_solve(covariance_masked, innovation_masked)
    sign, logdet = jnp.linalg.slogdet(symmetrize(covariance_masked))
    logdet = jnp.where(sign > 0, logdet, jnp.asarray(0.0, dtype=innovation.dtype))
    return -0.5 * (
        dim * jnp.log(jnp.asarray(2.0 * np.pi, dtype=innovation.dtype))
        + logdet
        + jnp.dot(innovation_masked, solved)
    )


def compute_nis_and_loglik(
    innov4: jnp.ndarray,
    S4: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact NIS and log-likelihood in active measurement subspace.

    Parameters
    ----------
    innov4 : jnp.ndarray
        Innovation in 4D measurement space (4,) [x1,y1,x2,y2] (m).
    S4 : jnp.ndarray
        Innovation covariance in 4D (4, 4) (m^2).
    both_leds : bool
        True if both LEDs valid → 4D.
    only_led1 : bool
        True if only LED1 valid → 2D.
    only_led2 : bool
        True if only LED2 valid → 2D.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(nis, log_likelihood)`` scalars.

    Notes
    -----
    Uses Cholesky-based solves. For 2D cases, projects via selector matrix
    before computing statistics, ensuring exact results without diagonal
    approximations.
    """
    from jax.scipy.linalg import cho_solve

    # 4D branch: both LEDs valid
    def compute_4d():
        S4s = symmetrize(S4)
        eps = adaptive_diagonal_boost(S4s)
        L4 = jnp.linalg.cholesky(S4s + eps * jnp.eye(4, dtype=S4s.dtype))
        x4 = cho_solve((L4, True), innov4)
        nis = jnp.dot(innov4, x4)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L4)))
        loglik = -0.5 * (logdet + nis + 4 * jnp.log(2 * jnp.pi))
        return nis, loglik

    # 2D branch: single LED valid
    def compute_2d():
        M2 = make_led_selector(only_led1, only_led2)  # (2, 4)
        S2 = M2 @ symmetrize(S4) @ M2.T  # (2, 2)
        innov2 = M2 @ innov4  # (2,)

        eps = adaptive_diagonal_boost(S2)
        L2 = jnp.linalg.cholesky(S2 + eps * jnp.eye(2, dtype=S2.dtype))
        x2 = cho_solve((L2, True), innov2)
        nis = jnp.dot(innov2, x2)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L2)))
        loglik = -0.5 * (logdet + nis + 2 * jnp.log(2 * jnp.pi))
        return nis, loglik

    # Select based on LED validity
    return lax.cond(both_leds, compute_4d, compute_2d)


def prepare_heading_measurement(
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    config: FilterCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare heading pseudo-measurement from LED geometry.

    Parameters
    ----------
    z_led1 : jnp.ndarray
        LED1 observation (2,) [x, y] in meters.
    z_led2 : jnp.ndarray
        LED2 observation (2,) [x, y] in meters.
    config : FilterCoreConfig
        Heading measurement configuration.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(heading_obs, R_heading, use_heading)`` where
        ``heading_obs`` is in radians, ``R_heading`` is variance (rad^2), and
        ``use_heading`` is a boolean JAX scalar.

    Notes
    -----
    Computes θ_obs = arctan2(dy, dx). If both LEDs are valid and the observed
    spacing is within the tolerance of the expected spacing, returns a small
    ``R_heading`` (possibly adapted by spacing ratio). Otherwise, returns a
    large ``R_heading`` (1e6) which effectively gates out the update.
    """
    # Check LED validity
    led1_valid = jnp.isfinite(z_led1).all()
    led2_valid = jnp.isfinite(z_led2).all()
    both_leds = led1_valid & led2_valid

    # Compute heading observation (always compute, gate via R)
    dx = z_led2[0] - z_led1[0]
    dy = z_led2[1] - z_led1[1]
    heading_obs = jnp.arctan2(dy, dx)

    # Check LED spacing validity
    obs_spacing = jnp.sqrt(dx**2 + dy**2)
    obs_spacing_valid = jnp.isfinite(obs_spacing) & (obs_spacing > 1e-6)

    if config.led_distance is not None:
        expected_spacing = jnp.asarray(config.led_distance, dtype=obs_spacing.dtype)
        spacing_ratio = jnp.where(
            obs_spacing_valid,
            obs_spacing / expected_spacing,
            jnp.zeros_like(obs_spacing),
        )
        spacing_valid = obs_spacing_valid & (
            (spacing_ratio > (1 - config.led_distance_tolerance))
            & (spacing_ratio < (1 + config.led_distance_tolerance))
        )
    else:
        expected_spacing = jnp.where(
            obs_spacing_valid,
            obs_spacing,
            jnp.asarray(1.0, dtype=obs_spacing.dtype),
        )
        spacing_valid = obs_spacing_valid

    # Overall validity: both LEDs + spacing OK + feature enabled
    use_heading = config.use_heading_measurement & both_leds & spacing_valid

    # Base heading measurement noise
    R_base = config.measurement_noise_heading

    # Adaptive noise scaling (if enabled and spacing is valid)
    # Clip obs_spacing to avoid division by zero/NaN
    obs_spacing_safe = jnp.where(
        obs_spacing_valid,
        obs_spacing,
        jnp.maximum(expected_spacing, jnp.asarray(1e-3, dtype=obs_spacing.dtype)),
    )
    R_heading_adapted = lax.cond(
        config.adaptive_heading_noise,
        lambda: R_base * (expected_spacing / obs_spacing_safe) ** 2,
        lambda: R_base,
    )

    # Gate via large R (JAX-friendly: no branching)
    # Valid: R ≈ 0.05² → strong update
    # Invalid: R = 1e6 → K ≈ 0 → no update
    R_heading = lax.select(use_heading, R_heading_adapted, 1e6)

    return heading_obs, R_heading, use_heading


# =============================================================================
# LED Spacing Estimation (shared by EKF and UKF)
# =============================================================================


def estimate_led_spacing(
    Z_cam_led1: jnp.ndarray,
    Z_cam_led2: jnp.ndarray,
    mask_cam: jnp.ndarray,
) -> float:
    """Estimate LED spacing from camera observations.

    Uses host-side NumPy to avoid expensive JIT compilation of nanmedian.

    Parameters
    ----------
    Z_cam_led1 : jnp.ndarray
        LED1 positions (N_cam, 2) in meters.
    Z_cam_led2 : jnp.ndarray
        LED2 positions (N_cam, 2) in meters.
    mask_cam : jnp.ndarray
        Camera validity mask (N_cam,), boolean.

    Returns
    -------
    float
        Median LED spacing (m). Falls back to 0.04 m if no valid dual-LED frames.
    """
    # Convert to NumPy for host-side computation (avoid JIT nanmedian)
    import numpy as np

    led1_np = np.asarray(Z_cam_led1)
    led2_np = np.asarray(Z_cam_led2)
    mask_np = np.asarray(mask_cam)

    # Find frames where both LEDs are visible
    led1_valid = np.isfinite(led1_np).all(axis=1)
    led2_valid = np.isfinite(led2_np).all(axis=1)
    both_valid = led1_valid & led2_valid & mask_np

    # Compute distances for valid frames
    distances = np.linalg.norm(led2_np - led1_np, axis=1)

    # Median of valid distances
    valid_distances = np.where(both_valid, distances, np.nan)

    # Use nanmedian, with fallback if all NaN
    median_spacing = np.nanmedian(valid_distances)

    # Fallback to 4 cm if no valid observations
    spacing = 0.04 if not np.isfinite(median_spacing) else float(median_spacing)
    return spacing


# =============================================================================
# IMU Index Computation
# =============================================================================


def compute_imu_index_arrays(
    t_imu: np.ndarray | jnp.ndarray, t_cam: np.ndarray | jnp.ndarray
) -> jnp.ndarray:
    """Build padded index arrays for IMU samples between camera frames.

    Parameters
    ----------
    t_imu : jnp.ndarray
        IMU timestamps (N_imu,) in seconds.
    t_cam : jnp.ndarray
        Camera timestamps (N_cam,) in seconds.

    Returns
    -------
    jnp.ndarray
        Index array (N_cam, max_imu_per_frame) of IMU indices; -1 indicates padding
        (no IMU sample). Returned as a JAX array for device use.

    Notes
    -----
    Host-side precomputation using NumPy avoids dynamic loop unrolling inside JIT.
    For each frame i, finds IMU indices in the half-open interval (t_cam[i-1], t_cam[i]].
    """
    t_imu_np = np.asarray(t_imu)
    t_cam_np = np.asarray(t_cam)

    n_cam = len(t_cam_np)
    all_indices = []

    # First pass: collect all valid index arrays to find max length
    for i in range(n_cam):
        if i == 0:
            # First frame: no IMU propagation
            valid_indices = np.array([], dtype=np.int32)
        else:
            # Find IMU samples in (t_prev, t_current]
            interval_mask = (t_imu_np > t_cam_np[i - 1]) & (t_imu_np <= t_cam_np[i])
            valid_indices = np.nonzero(interval_mask)[0]

        all_indices.append(valid_indices)

    # Compute max length from actual data
    max_imu_per_frame = max(len(idx) for idx in all_indices)

    # Second pass: pad all arrays to max length
    padded_indices = []
    for valid_indices in all_indices:
        indices = np.full(max_imu_per_frame, -1, dtype=np.int32)
        if len(valid_indices) > 0:
            indices[: len(valid_indices)] = valid_indices
        padded_indices.append(indices)

    # Convert to JAX array for device use
    return jnp.array(padded_indices, dtype=jnp.int32)


# =============================================================================
# IMU Noise Propagation Matrices
# =============================================================================


def build_G_matrix_generic(
    n: int,
    theta: float | Array,
    dt: float | Array,
    *,
    pos_idx: tuple[int, int] = (0, 1),
    vel_idx: tuple[int, int] | tuple[int, int, int] = (2, 3),
    theta_idx: int = 4,
    n_accel: int = 2,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Generic IMU input noise mapping G for arbitrary layouts.

    Parameters
    ----------
    n : int
        State dimension.
    theta : float
        Heading angle (rad).
    dt : float
        Time step (s).
    pos_idx : tuple[int, int], default (0, 1)
        Position indices (x, y). Only 2D positions supported.
    vel_idx : tuple[int, int] or tuple[int, int, int], default (2, 3)
        Velocity indices (vx, vy) or (vx, vy, vz). Length must match n_accel.
    theta_idx : int, default 4
        Heading index.
    n_accel : int, default 2
        Number of accelerometer axes (2 for 2D, 3 for 3D).
    dtype : jnp.dtype, default jnp.float32
        Array dtype.

    Returns
    -------
    jnp.ndarray
        G matrix (n, n_accel+1) for [ω_z, f_x, f_y, ...].

    Notes
    -----
    Places ∂θ/∂ω_z = dt at ``theta_idx``, ∂v/∂f = R(θ)·dt at ``vel_idx``,
    and ∂p/∂f = R(θ)·0.5·dt² at ``pos_idx``. Missing/out-of-bounds indices
    are ignored.

    For n_accel=3 (3D accel), the z-axis velocity is affected directly by f_z
    without rotation: ∂vz/∂f_z = dt (since z-axis is vertical).

    Raises
    ------
    ValueError
        If n_accel is not 2 or 3.
        If n_accel=3 but len(vel_idx) != 3 (inconsistent configuration).
        If n_accel=2 but len(vel_idx) not in (2, 3).
    """
    if n_accel not in (2, 3):
        raise ValueError(f"n_accel must be 2 or 3, got {n_accel}")

    # Validate consistency between n_accel and vel_idx
    if n_accel == 3 and len(vel_idx) != 3:
        raise ValueError(
            f"n_accel=3 requires 3D velocity (len(vel_idx)=3), got {len(vel_idx)}"
        )
    if n_accel == 2 and len(vel_idx) not in (2, 3):
        raise ValueError(
            f"n_accel=2 requires 2D velocity, got len(vel_idx)={len(vel_idx)}"
        )

    G = jnp.zeros((n, n_accel + 1), dtype=dtype)
    theta_arr = jnp.asarray(theta, dtype=dtype)
    dt_arr = jnp.asarray(dt, dtype=dtype)
    c, s = jnp.cos(theta_arr), jnp.sin(theta_arr)
    R_2d = jnp.array([[c, -s], [s, c]], dtype=dtype)

    # Heading: ∂θ/∂ω_z = dt
    if 0 <= theta_idx < n:
        G = G.at[theta_idx, 0].set(dt_arr)

    # Velocity (x, y components affected by 2D rotated accel)
    if len(vel_idx) >= 2:
        vx_i, vy_i = vel_idx[0], vel_idx[1]
        if 0 <= vx_i < n and 0 <= vy_i < n:
            G = G.at[vx_i : vy_i + 1, 1:3].set(R_2d * dt_arr)

    # Velocity (z component, if present, affected directly by f_z)
    if n_accel == 3 and len(vel_idx) == 3:
        vz_i = vel_idx[2]
        if 0 <= vz_i < n:
            # ∂vz/∂f_z = dt (no rotation, since yaw-only model preserves z)
            # R_z(θ) rotates only in x-y plane; world z ≈ body z
            G = G.at[vz_i, 3].set(dt_arr)

    # Position (only x, y components affected, since we track 2D position)
    px_i, py_i = pos_idx
    if 0 <= px_i < n and 0 <= py_i < n:
        G = G.at[px_i : py_i + 1, 1:3].set(R_2d * (0.5 * dt_arr * dt_arr))

    return G

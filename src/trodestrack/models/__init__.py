"""State-space models for 2D tracking with IMU integration."""

from .state import (
    State2D,
    STATE_DIM,
    state_to_array,
    array_to_state,
    create_initial_state,
)

from .dynamics import (
    predict_state,
    predict_covariance,
    compute_state_jacobian,
    compute_process_noise,
)

from .measurements import (
    position_measurement,
    heading_measurement,
    compute_position_jacobian,
    compute_heading_jacobian,
    create_measurement_noise,
    validate_led_measurement,
    create_combined_measurement,
    create_combined_jacobian,
)

from .gating import (
    mahalanobis_distance,
    mahalanobis_gate,
    chi_squared_threshold,
    create_measurement_mask,
    apply_measurement_mask,
    validate_and_gate_measurement,
    compute_innovation_covariance,
)

from .velocity import (
    velocity_measurement,
    compute_velocity_jacobian,
    create_velocity_noise,
    estimate_velocity_from_positions,
    velocity_pseudo_measurement_update,
    should_use_velocity_constraint,
    compute_velocity_from_recent_positions,
)

from .ekf import (
    EKFState,
    EKFResult,
    EKFFilter,
    ekf_predict,
    ekf_update,
    create_initial_ekf_state,
)

from .ukf import (
    UKFState,
    UKFResult,
    UKFFilter,
    UKFParams,
    ukf_predict,
    ukf_update,
    create_initial_ukf_state,
    generate_sigma_points,
)

from .rts_smoother import (
    RTSResult,
    RTSSmoother,
    ForwardPassData,
    rts_smooth,
    rts_backward_step,
    compute_smoothing_improvement,
)

from .cached_ekf import (
    CachedEKFFilter,
    CachedComputations,
    efficient_rts_smooth_with_cache,
    compute_cache_efficiency_stats,
)

__all__ = [
    # State representation
    "State2D",
    "STATE_DIM",
    "state_to_array",
    "array_to_state",
    "create_initial_state",
    # Dynamics
    "predict_state",
    "predict_covariance",
    "compute_state_jacobian",
    "compute_process_noise",
    # Measurements
    "position_measurement",
    "heading_measurement",
    "compute_position_jacobian",
    "compute_heading_jacobian",
    "create_measurement_noise",
    "validate_led_measurement",
    "create_combined_measurement",
    "create_combined_jacobian",
    # Gating
    "mahalanobis_distance",
    "mahalanobis_gate",
    "chi_squared_threshold",
    "create_measurement_mask",
    "apply_measurement_mask",
    "validate_and_gate_measurement",
    "compute_innovation_covariance",
    # Velocity
    "velocity_measurement",
    "compute_velocity_jacobian",
    "create_velocity_noise",
    "estimate_velocity_from_positions",
    "velocity_pseudo_measurement_update",
    "should_use_velocity_constraint",
    "compute_velocity_from_recent_positions",
    # Kalman Filtering
    "EKFState",
    "EKFResult",
    "EKFFilter",
    "ekf_predict",
    "ekf_update",
    "create_initial_ekf_state",
    # Unscented Kalman Filtering
    "UKFState",
    "UKFResult",
    "UKFFilter",
    "UKFParams",
    "ukf_predict",
    "ukf_update",
    "create_initial_ukf_state",
    "generate_sigma_points",
    # RTS Smoothing
    "RTSResult",
    "RTSSmoother",
    "ForwardPassData",
    "rts_smooth",
    "rts_backward_step",
    "compute_smoothing_improvement",
    # Cached EKF
    "CachedEKFFilter",
    "CachedComputations",
    "efficient_rts_smooth_with_cache",
    "compute_cache_efficiency_stats",
]

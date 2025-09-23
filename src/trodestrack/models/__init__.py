"""State-space models for 2D tracking with IMU integration."""

from .cached_ekf import (
    CachedComputations,
    CachedEKFFilter,
    compute_cache_efficiency_stats,
    efficient_rts_smooth_with_cache,
)
from .dynamics import (
    compute_process_noise,
    compute_state_jacobian,
    predict_covariance,
    predict_state,
)
from .ekf import (
    EkfCarry,
    EKFFilter,
    EKFResult,
    EKFState,
    create_ekf_step_arrays_optimized,
    create_initial_ekf_state,
    ekf_predict,
    ekf_step_arrays_pure,
    ekf_update,
)
from .gating import (
    apply_measurement_mask,
    chi_squared_threshold,
    compute_innovation_covariance,
    create_measurement_mask,
    mahalanobis_distance,
    mahalanobis_gate,
    validate_and_gate_measurement,
)
from .measurements import (
    compute_heading_jacobian,
    compute_position_jacobian,
    create_combined_jacobian,
    create_combined_measurement,
    create_measurement_noise,
    heading_measurement,
    position_measurement,
    validate_led_measurement,
)
from .rts_smoother import (
    ForwardPassData,
    RTSResult,
    RTSSmoother,
    compute_smoothing_improvement,
    rts_backward_step,
    rts_smooth,
    rts_smooth_pure,
)
from .state import (
    STATE_DIM,
    State2D,
    array_to_state,
    create_initial_state,
    state_to_array,
)
from .ukf import (
    UKFFilter,
    UKFParams,
    UKFResult,
    UKFState,
    create_initial_ukf_state,
    generate_sigma_points,
    ukf_predict,
    ukf_update,
)
from .velocity import (
    compute_velocity_from_recent_positions,
    compute_velocity_jacobian,
    create_velocity_noise,
    estimate_velocity_from_positions,
    should_use_velocity_constraint,
    velocity_measurement,
    velocity_pseudo_measurement_update,
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
    "EkfCarry",
    "EKFState",
    "EKFResult",
    "EKFFilter",
    "ekf_predict",
    "ekf_update",
    "create_initial_ekf_state",
    "ekf_step_arrays_pure",
    "create_ekf_step_arrays_optimized",
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
    "rts_smooth_pure",
    "rts_backward_step",
    "compute_smoothing_improvement",
    # Cached EKF
    "CachedEKFFilter",
    "CachedComputations",
    "efficient_rts_smooth_with_cache",
    "compute_cache_efficiency_stats",
]

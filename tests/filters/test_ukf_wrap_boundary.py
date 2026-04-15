"""Regression tests for UKF angle-wrap handling at heading = ±π.

Primary bug (heading pseudo-measurement update): ``update_heading`` computes
the predicted heading ``h_pred`` via circular mean (atan2) which wraps to
(-π, π], but the sigma-point heading values remain in whatever numeric space
they were generated in (typically ``m[h_idx] + offset``, unwrapped). When the
state sits near the ±π boundary, ``h_pred`` can land on the opposite side of
the wrap from the sigma points, making the raw subtraction
``sigmas_heading - h_pred`` produce ~2π-sized deviations. This inflates the
innovation covariance ``S`` by ~(2π)² and collapses the Kalman gain.

The parallel fix already exists in the UKF predict step at ukf.py:362, which
wraps heading deviations before covariance reconstruction (predict step uses
circular mean for m_pred and raw sigmas_prop from dynamics).

Secondary observations (no fix needed):
    - Camera update ``state_deviations = sigmas - m_in``: ``m_in`` comes
      straight from the filtered state and sigmas are generated from it
      algebraically (no wrapping), so they stay numerically consistent and
      the deviations are always small. No wrap needed in this path.
    - Sigma-point smoother ``predict_between_frames_sigma``: uses arithmetic
      mean (not circular mean) and dynamics does not wrap heading, so
      ``m_pred`` and ``sigmas_prop`` stay in a consistent numeric space. The
      smoother residual at offline.py:487 already wraps the cross-boundary
      difference. No wrap needed in the deviation/cross-cov computation.

The camera and smoother tests below are kept as no-regression guards.

Strategy
--------
For a measurement that exactly matches the predicted state, a correct update
must reduce the heading variance by a fixed factor determined by R and P. That
factor must NOT depend on the absolute heading angle -- it should be the same
whether the state sits at θ=0 (no wrap) or θ=π (straddles wrap). We run each
update at both locations and compare the posterior covariance reduction ratio.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.sensors.heading_pseudo import HeadingPseudoModel
from trodestrack.models.state_layout import LAYOUT_2D_FULL, get_heading_index
from trodestrack.models.ukf import UKFConfig, UKFState, update_heading, update_step

H_IDX = get_heading_index(LAYOUT_2D_FULL)  # = 4 for 2d_full
LED_D = 0.04


def _leds_for_heading(theta: float, midpoint: tuple[float, float] = (1.0, 1.0)):
    """Build (LED1, LED2) positions such that observed heading equals ``theta``.

    Heading convention: heading_obs = atan2(LED2.y - LED1.y, LED2.x - LED1.x).
    """
    cx, cy = midpoint
    half_d = LED_D / 2.0
    led1 = jnp.array([cx - half_d * np.cos(theta), cy - half_d * np.sin(theta)])
    led2 = jnp.array([cx + half_d * np.cos(theta), cy + half_d * np.sin(theta)])
    return led1, led2


def _make_state(theta: float, heading_var: float = 0.04) -> UKFState:
    """Build an 8D state at position (1, 1) with heading = ``theta``.

    ``heading_var`` = 0.04 gives σ_θ = 0.2 rad, wide enough that with the
    UKFConfig used in the tests below (α=1, κ=0, n=8) some sigma points land
    beyond ±π when ``theta`` is at the wrap boundary.
    """
    mean = jnp.array([1.0, 1.0, 0.0, 0.0, theta, 0.0, 0.0, 0.0])
    # Diagonal cov: small position/velocity/bias variance, larger heading variance
    P = jnp.eye(8) * 1e-4
    P = P.at[H_IDX, H_IDX].set(heading_var)
    return UKFState(mean=mean, cov=P)


def _heading_model_for_theta(theta: float, config: UKFConfig) -> HeadingPseudoModel:
    """Single-frame HeadingPseudoModel whose observation equals ``theta``."""
    led1, led2 = _leds_for_heading(theta, midpoint=(1.0, 1.0))
    return HeadingPseudoModel(
        config=config,
        layout=LAYOUT_2D_FULL,
        z_led1_all=led1.reshape(1, 2),
        z_led2_all=led2.reshape(1, 2),
    )


def _camera_model_for_theta(theta: float, config: UKFConfig) -> CameraPositionModel:
    """Single-frame CameraPositionModel whose LED observations match ``theta``."""
    led1, led2 = _leds_for_heading(theta, midpoint=(1.0, 1.0))
    return CameraPositionModel(
        led_distance=LED_D,
        measurement_noise_base=config.measurement_noise_pos,
        layout=LAYOUT_2D_FULL,
        z_led1_all=led1.reshape(1, 2),
        z_led2_all=led2.reshape(1, 2),
        conf_all=None,
        confidence_clip_min=1e-2,
    )


def _reduction_ratio(prior: UKFState, posterior: UKFState) -> float:
    """Ratio of posterior heading variance to prior heading variance."""
    return float(posterior.cov[H_IDX, H_IDX] / prior.cov[H_IDX, H_IDX])


# =============================================================================
# Heading pseudo-measurement update: wrap boundary
# =============================================================================


def test_update_heading_variance_reduction_independent_of_wrap():
    """update_heading should reduce heading variance identically at θ=0 and θ=π.

    With a measurement matching the state exactly, the innovation covariance S
    and Kalman gain K depend only on R and P -- not on the absolute heading.
    If heading deviations are not wrapped before computing S, at θ=π the
    deviations include ~2π values and S becomes ~(2π)² times too large,
    collapsing K toward 0 and leaving the posterior variance unchanged.
    """
    config = UKFConfig(
        use_heading_measurement=True,
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.05**2,
        led_distance=LED_D,
        led_distance_tolerance=0.3,
        adaptive_heading_noise=False,
        state_mode="2d_full",
    )

    reductions = {}
    for label, theta in [("far_from_wrap", 0.0), ("at_wrap", float(np.pi))]:
        prior = _make_state(theta)
        model = _heading_model_for_theta(theta, config)
        posterior, _loglik = update_heading(
            state=prior,
            heading_model=model,
            frame_idx=0,
            observation_is_valid=True,
            config=config,
            layout=LAYOUT_2D_FULL,
        )
        reductions[label] = _reduction_ratio(prior, posterior)

    # The update should actually reduce variance -- sanity check.
    assert reductions["far_from_wrap"] < 0.5, (
        f"Baseline heading update did not reduce variance enough: "
        f"ratio={reductions['far_from_wrap']:.3f}"
    )
    # The reduction at the wrap boundary should match the baseline reduction.
    # Both runs are zero-innovation scalar Kalman updates with identical R and
    # P, so the posterior variance ratio is analytically the same value
    # (R / (R + P) ≈ 0.059 for these parameters); any difference reflects the
    # wrap-boundary bug in S/P_cross. 0.01 is a generous numerical tolerance.
    # Without wrap fix, at_wrap ratio ≈ 1.0 (S inflated by ~(2π)², K ≈ 0).
    assert abs(reductions["at_wrap"] - reductions["far_from_wrap"]) < 0.01, (
        f"Heading variance reduction differs at wrap boundary: "
        f"far_from_wrap={reductions['far_from_wrap']:.4f}, "
        f"at_wrap={reductions['at_wrap']:.4f} "
        f"(likely missing wrap on sigma-point heading deviations)"
    )


def test_update_heading_posterior_finite_at_wrap():
    """Posterior mean and covariance must be finite at the wrap boundary."""
    config = UKFConfig(
        use_heading_measurement=True,
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.05**2,
        led_distance=LED_D,
        led_distance_tolerance=0.3,
        state_mode="2d_full",
    )
    prior = _make_state(float(np.pi))
    model = _heading_model_for_theta(float(np.pi), config)
    posterior, _ = update_heading(
        state=prior,
        heading_model=model,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )
    assert np.all(np.isfinite(np.asarray(posterior.mean)))
    assert np.all(np.isfinite(np.asarray(posterior.cov)))
    theta_post = float(posterior.mean[H_IDX])
    assert -np.pi - 1e-6 <= theta_post <= np.pi + 1e-6


# =============================================================================
# Camera LED update: cross-covariance wrap
# =============================================================================


def test_update_step_heading_reduction_independent_of_wrap():
    """No-regression guard: camera update is wrap-safe by construction.

    An external review suggested that ``state_deviations = sigmas - m_in`` in
    the camera update's cross-covariance might inflate at the wrap boundary.
    In reality, ``sigmas`` are generated algebraically from ``m_in`` via
    ``compute_sigma_points`` (no wrapping), so ``sigmas[:, h_idx] - m_in[h_idx]``
    is exactly the sigma-point offset with no wrap discontinuity possible.

    This test would only start failing if someone later introduced a code path
    that wraps ``m_in`` or ``sigmas`` inconsistently between generation and the
    cross-cov computation.
    """
    config = UKFConfig(
        use_heading_measurement=False,  # only camera update here
        measurement_noise_pos=0.005**2,
        led_distance=LED_D,
        use_mahalanobis_gating=False,
        state_mode="2d_full",
    )

    reductions = {}
    for label, theta in [("far_from_wrap", 0.0), ("at_wrap", float(np.pi))]:
        prior = _make_state(theta)
        camera_model = _camera_model_for_theta(theta, config)
        posterior, _loglik = update_step(
            state=prior,
            camera_model=camera_model,
            frame_idx=0,
            observation_is_valid=True,
            config=config,
        )
        reductions[label] = _reduction_ratio(prior, posterior)

    # Camera update should couple to heading via LED geometry and reduce variance.
    assert reductions["far_from_wrap"] < 0.99, (
        f"Baseline camera update did not reduce heading variance: "
        f"ratio={reductions['far_from_wrap']:.3f}"
    )
    # Reduction at wrap boundary should match baseline (within ~5%).
    rel = abs(reductions["at_wrap"] - reductions["far_from_wrap"])
    assert rel < 0.05, (
        f"Camera-update heading reduction differs at wrap boundary: "
        f"far={reductions['far_from_wrap']:.4f}, "
        f"wrap={reductions['at_wrap']:.4f} "
        f"(likely missing wrap on state_deviations[:, h_idx] in cross-cov)"
    )


# =============================================================================
# Smoother: circular mean + wrapped deviations
# =============================================================================


@pytest.mark.slow
def test_sigma_smoother_matches_filter_near_wrap_boundary():
    """No-regression guard: sigma-point smoother is wrap-safe by construction.

    An external review suggested the smoother's ``predict_between_frames_sigma``
    might corrupt covariance at heading wrap crossings because it uses
    arithmetic mean rather than circular mean. In reality, ``dynamics_function``
    in filter_common.py integrates heading without applying ``wrap_angle``, so
    ``sigmas_prop`` and the arithmetic ``m_pred`` stay in the same unwrapped
    numeric space and deviations remain consistent. The only place the smoother
    needs to wrap is the cross-boundary residual ``smoothed_mean_next - m_pred``
    at offline.py:487, which is already wrapped.

    This test runs a full forward+backward pass on a circular trajectory (which
    crosses ±π many times) and asserts smoother output is finite, not worse
    than the filter, and variance does not blow up. It would start failing if
    someone later introduced a path that wraps mean or sigmas inconsistently.
    """
    from trodestrack.models.ukf import unscented_kalman_filter
    from trodestrack.runtime.offline import sigma_point_smoother
    from trodestrack.sim.simple import SimpleSimConfig, simulate_circular
    from trodestrack.sim.utils import interp_angle

    # 3 s is enough: a circular trajectory crosses ±π within the first
    # revolution regardless of radius/speed.
    sim_cfg = SimpleSimConfig(
        duration_s=3.0, fs_cam=30.0, fs_imu=200.0, cam_dropout_prob=0.0
    )
    sim = simulate_circular(config=sim_cfg, radius=0.5, seed=7)

    ukf_cfg = UKFConfig(
        use_heading_measurement=True,
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.05**2,
        led_distance=LED_D,
        state_mode="2d_full",
        alpha=1.0,
    )

    filt = unscented_kalman_filter(
        ukf_config=ukf_cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_obs"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    smoothed = sigma_point_smoother(
        filter_result=filt,
        ukf_config=ukf_cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_obs"],
        mask_cam=sim["mask_cam"],
    )

    smoothed_means = np.asarray(smoothed.smoothed_means)
    smoothed_covs = np.asarray(smoothed.smoothed_covariances)

    # Must be finite everywhere.
    assert np.all(np.isfinite(smoothed_means))
    assert np.all(np.isfinite(smoothed_covs))

    # Heading RMSE vs truth.
    heading_truth = interp_angle(sim["t_cam_obs"], sim["t_imu"], sim["X_truth"][:, 4])
    theta_filt = np.asarray(filt.filtered_means[:, H_IDX])
    theta_smooth = smoothed_means[:, H_IDX]

    def ang_err(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    rmse_filt = float(np.sqrt(np.mean(ang_err(theta_filt, heading_truth) ** 2)))
    rmse_smooth = float(np.sqrt(np.mean(ang_err(theta_smooth, heading_truth) ** 2)))

    # Smoother should not be worse than filter on heading when heading crosses ±π.
    # Allow small numerical slack.
    assert rmse_smooth <= rmse_filt * 1.05 + 1e-3, (
        f"Smoother heading RMSE worse than filter: "
        f"filter={np.rad2deg(rmse_filt):.2f}°, "
        f"smoother={np.rad2deg(rmse_smooth):.2f}° "
        f"(likely missing wrap in predict_between_frames_sigma)"
    )

    # Smoother heading variance should be <= filter heading variance on average.
    var_filt = float(np.mean(filt.filtered_covariances[:, H_IDX, H_IDX]))
    var_smooth = float(np.mean(smoothed_covs[:, H_IDX, H_IDX]))
    assert var_smooth <= var_filt * 1.2 + 1e-6, (
        f"Smoother heading variance inflated vs filter: "
        f"filter_mean_var={var_filt:.4e}, smoother_mean_var={var_smooth:.4e}"
    )

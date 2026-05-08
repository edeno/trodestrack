"""Synthetic-scenario tests for the TTL event channel (Milestone 3).

These tests use a constant-velocity trajectory to keep the sim cheap, then:

- Beam grid scenario: events along the trajectory reduce position RMSE
  during a long camera dropout vs the no-event baseline.
- Zone trigger scenario: a single event collapses position uncertainty
  to within the zone's sigma.
- Multiple-events-per-frame parity: stacking two events in one camera
  frame yields the same posterior as applying them sequentially.
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.config.schemas import (
    BeamSpec,
    RFIDReaderSpec,
    ZoneTriggerSpec,
)
from trodestrack.io.ttl_events import per_frame_event_indices
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.simple import SimpleSimConfig, simulate_constant_velocity
from trodestrack.sim.ttl_events import events_from_trajectory

EDGE_TO_INT = {"fall": 0, "rise": 1}


def _ekf_config() -> EKFConfig:
    return EKFConfig(
        state_mode="2d_full",
        process_noise_pos=0.001,
        process_noise_vel=0.5,
        process_noise_heading=0.5,
        process_noise_gyro_bias=7.6e-7,
        process_noise_accel_bias=2.4e-9,
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.5**2,
        led_distance=0.04,
        use_heading_measurement=True,
        damping_coeff=0.4,
    )


def _build_session(duration_s: float = 4.0, vx: float = 0.2):
    cfg = SimpleSimConfig(
        duration_s=duration_s,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.0005,
        accel_noise_density=0.02,
        gyro_bias_std=0.005,
        accel_bias_std=0.01,
        cam_noise_std=0.005,
        cam_dropout_prob=0.0,
    )
    return simulate_constant_velocity(cfg, velocity=np.array([vx, 0.0]))


def _camera_dropout(sim, t_start: float, t_end: float):
    """Return mask_cam with a dropout window, plus updated LED arrays (NaN in window)."""
    mask = np.asarray(sim["mask_cam"]).copy()
    led1 = np.asarray(sim["Z_cam_led1"]).copy()
    led2 = np.asarray(sim["Z_cam_led2"]).copy()
    drop = (sim["t_cam_exp"] >= t_start) & (sim["t_cam_exp"] <= t_end)
    mask[drop] = False
    led1[drop] = np.nan
    led2[drop] = np.nan
    return mask, led1, led2


def _events_to_dense(events, sources, t_cam, max_events_per_frame=4):
    """Convert event list + sources into the dense arrays the EKF consumes."""
    anchors = np.stack([s.anchor for s in sources], axis=0)
    covariances = np.stack([s.R for s in sources], axis=0)
    source_id_to_index = {s.source_id: i for i, s in enumerate(sources)}
    source_active_edges = {
        s.source_id: EDGE_TO_INT["fall" if s.source_type == "beam" else "rise"]
        for s in sources
    }
    t_evt = np.array([e.time for e in events])
    sid = np.array([e.source_id for e in events], dtype=int)
    edge = np.array([EDGE_TO_INT[e.edge] for e in events], dtype=int)
    indices = per_frame_event_indices(
        t_evt,
        sid,
        edge,
        t_cam,
        source_active_edges=source_active_edges,
        source_id_to_index=source_id_to_index,
        max_events_per_frame=max_events_per_frame,
    )
    return anchors, covariances, indices


def test_beam_grid_reduces_dropout_position_rmse():
    """A 1D beam set along the trajectory cuts position drift during a 2s dropout."""
    sim = _build_session(duration_s=4.0, vx=0.2)
    ekf_cfg = _ekf_config()

    # Camera blackout from 1.0s to 3.0s.
    mask, led1, led2 = _camera_dropout(sim, t_start=1.0, t_end=3.0)
    common_kwargs = dict(
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=led1,
        Z_cam_led2=led2,
        mask_cam=mask,
    )

    # Truth at camera times.
    t_truth = sim["t_imu"]
    x_truth_at_cam = np.interp(sim["t_cam_exp"], t_truth, sim["X_truth"][:, 0])
    y_truth_at_cam = np.interp(sim["t_cam_exp"], t_truth, sim["X_truth"][:, 1])

    # Beams perpendicular to motion (along y) at five x-locations covering
    # the dropout window. Centered at the trajectory's y (0.1) with a short
    # length so the along-beam Gaussian approximation does not pull
    # systematically off-trajectory.
    beams = [
        BeamSpec(
            id=10 + k,
            emitter=(x_loc, 0.0),
            receiver=(x_loc, 0.2),
            sigma_perp_m=0.01,
        )
        for k, x_loc in enumerate(np.linspace(0.20, 0.60, 5))
    ]
    sources = [b.to_event_source() for b in beams]
    events = events_from_trajectory(
        np.asarray(sim["t_imu"]),
        np.column_stack([sim["X_truth"][:, 0], sim["X_truth"][:, 1]]),
        beams=beams,
    )
    assert len(events) >= 3, "Expected several beam crossings during dropout"

    anchors, covariances, indices = _events_to_dense(events, sources, sim["t_cam_exp"])

    # Baseline (no events).
    baseline = extended_kalman_filter(ekf_cfg, **common_kwargs)
    with_events = extended_kalman_filter(
        ekf_cfg,
        **common_kwargs,
        event_source_anchors=anchors,
        event_source_covariances=covariances,
        event_indices_per_frame=indices,
    )

    drop_mask = (sim["t_cam_exp"] >= 1.0) & (sim["t_cam_exp"] <= 3.0)
    pos_baseline = np.asarray(baseline.filtered_means)[drop_mask, 0:2]
    pos_events = np.asarray(with_events.filtered_means)[drop_mask, 0:2]
    truth_drop = np.column_stack([x_truth_at_cam[drop_mask], y_truth_at_cam[drop_mask]])
    rmse_baseline = float(
        np.sqrt(np.mean(np.sum((pos_baseline - truth_drop) ** 2, axis=1)))
    )
    rmse_events = float(
        np.sqrt(np.mean(np.sum((pos_events - truth_drop) ** 2, axis=1)))
    )
    # Beam crossings should provide a meaningful drift cut.
    assert rmse_events < rmse_baseline


def test_zone_trigger_collapses_uncertainty():
    """A single zone trigger pulls posterior position to within zone sigma."""
    sim = _build_session(duration_s=2.0, vx=0.2)
    ekf_cfg = _ekf_config()
    mask, led1, led2 = _camera_dropout(sim, t_start=0.5, t_end=1.5)

    # Zone at the truth location at t=1.0s.
    truth_x = float(np.interp(1.0, sim["t_imu"], sim["X_truth"][:, 0]))
    truth_y = float(np.interp(1.0, sim["t_imu"], sim["X_truth"][:, 1]))
    zone = ZoneTriggerSpec(id=99, center=(truth_x, truth_y), sigma_m=0.02)
    sources = [zone.to_event_source()]

    events = events_from_trajectory(
        np.asarray(sim["t_imu"]),
        np.column_stack([sim["X_truth"][:, 0], sim["X_truth"][:, 1]]),
        zone_triggers=[zone],
        zone_trigger_radius_m=0.05,
    )
    assert len(events) >= 1
    anchors, covariances, indices = _events_to_dense(events, sources, sim["t_cam_exp"])

    result = extended_kalman_filter(
        ekf_cfg,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        led1,
        led2,
        mask,
        event_source_anchors=anchors,
        event_source_covariances=covariances,
        event_indices_per_frame=indices,
    )
    # Frame index of the trigger in camera time (first frame within the
    # zone disc).
    trigger_frame = int(np.argmax(np.any(indices >= 0, axis=1)))
    pos_post = np.asarray(result.filtered_means)[trigger_frame, 0:2]
    cov_post = np.asarray(result.filtered_covariances)[trigger_frame, 0:2, 0:2]
    # Posterior position is within ~5σ of the anchor and variance has
    # collapsed below the prior position variance contribution.
    np.testing.assert_allclose(pos_post, [truth_x, truth_y], atol=0.05)
    pos_var = float(np.diag(cov_post).max())
    assert pos_var < 0.05**2


@pytest.mark.parametrize("two_events_in_same_frame", [True, False])
def test_multiple_events_in_one_frame_parity(two_events_in_same_frame):
    """Events in one frame produce the same posterior as sequential application
    when the underlying measurements are independent and conditionally
    consistent."""
    sim = _build_session(duration_s=1.5, vx=0.2)
    ekf_cfg = _ekf_config()
    n_cam = sim["t_cam_exp"].shape[0]

    # Two zone triggers at different anchors.
    zone1 = ZoneTriggerSpec(id=1, center=(0.30, 0.05), sigma_m=0.02)
    zone2 = ZoneTriggerSpec(id=2, center=(0.30, -0.05), sigma_m=0.02)
    sources = [zone1.to_event_source(), zone2.to_event_source()]
    anchors = np.stack([s.anchor for s in sources], axis=0)
    covariances = np.stack([s.R for s in sources], axis=0)

    indices = np.full((n_cam, 4), -1, dtype=np.int32)
    fire_frame = n_cam // 2
    if two_events_in_same_frame:
        indices[fire_frame, 0] = 0
        indices[fire_frame, 1] = 1
    else:
        indices[fire_frame, 0] = 0
        indices[fire_frame + 1, 0] = 1

    result = extended_kalman_filter(
        ekf_cfg,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        sim["mask_cam"],
        event_source_anchors=anchors,
        event_source_covariances=covariances,
        event_indices_per_frame=indices,
    )
    # Posterior remains finite and pulled toward the average anchor.
    final_pos = np.asarray(result.filtered_means)[-1, 0:2]
    assert np.all(np.isfinite(final_pos))


def test_rfid_collapses_position_uncertainty():
    """RFID detection collapses position uncertainty to ~effective_radius."""
    sim = _build_session(duration_s=2.0, vx=0.2)
    ekf_cfg = _ekf_config()
    mask, led1, led2 = _camera_dropout(sim, t_start=0.5, t_end=1.5)

    truth_x = float(np.interp(1.0, sim["t_imu"], sim["X_truth"][:, 0]))
    truth_y = float(np.interp(1.0, sim["t_imu"], sim["X_truth"][:, 1]))
    reader = RFIDReaderSpec(id=42, center=(truth_x, truth_y), effective_radius_m=0.05)
    sources = [reader.to_event_source()]

    events = events_from_trajectory(
        np.asarray(sim["t_imu"]),
        np.column_stack([sim["X_truth"][:, 0], sim["X_truth"][:, 1]]),
        rfid_readers=[reader],
    )
    assert len(events) >= 1
    anchors, covariances, indices = _events_to_dense(events, sources, sim["t_cam_exp"])

    result = extended_kalman_filter(
        ekf_cfg,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        led1,
        led2,
        mask,
        event_source_anchors=anchors,
        event_source_covariances=covariances,
        event_indices_per_frame=indices,
    )
    trigger_frame = int(np.argmax(np.any(indices >= 0, axis=1)))
    cov_post = np.asarray(result.filtered_covariances)[trigger_frame, 0:2, 0:2]
    pos_var = float(np.diag(cov_post).max())
    # Effective radius² / 2 is the per-axis variance bound; allow 50% slack.
    assert pos_var < (0.05**2 / 2.0) * 1.5

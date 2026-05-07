"""Offline LED identity correction for persistent front/back swaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trodestrack.config.schemas import LedIdentityConfig


@dataclass(frozen=True)
class CorrectedLEDIdentity:
    """Corrected LED arrays and swap diagnostics."""

    led1: np.ndarray
    led2: np.ndarray
    swapped: np.ndarray
    diagnostics: dict[str, float | int | str]


def resolve_led_identity(
    t_cam: np.ndarray,
    led1: np.ndarray,
    led2: np.ndarray,
    mask_cam: np.ndarray,
    *,
    led_distance: float | None,
    config: LedIdentityConfig,
    t_imu: np.ndarray | None = None,
    gyro_z: np.ndarray | None = None,
) -> CorrectedLEDIdentity:
    """Resolve persistent LED swaps with a two-state dynamic program.

    State 0 keeps the observed labels. State 1 swaps LED1/LED2. The transition
    cost favors temporal continuity of each physical LED, with optional gyro
    heading consistency. Missing/non-finite dual-LED frames are carried through
    unchanged and do not force identity switches. ``config.initial_state`` can
    anchor the first valid dual-LED frame when the whole session has a known
    original/swapped label convention; without that prior, a global all-session
    swap is unidentifiable from continuity alone.
    """

    led1_arr = _validate_led_array(led1, "led1")
    led2_arr = _validate_led_array(led2, "led2")
    t_arr = np.asarray(t_cam, dtype=float)
    mask_arr = np.asarray(mask_cam, dtype=bool)
    if led1_arr.shape != led2_arr.shape:
        raise ValueError(f"led1 shape {led1_arr.shape} != led2 shape {led2_arr.shape}.")
    if t_arr.shape != (led1_arr.shape[0],):
        raise ValueError(
            f"t_cam must have shape ({led1_arr.shape[0]},), got {t_arr.shape}."
        )
    if mask_arr.shape != t_arr.shape:
        raise ValueError(
            f"mask_cam shape {mask_arr.shape} != t_cam shape {t_arr.shape}."
        )
    if config.mode == "none":
        return CorrectedLEDIdentity(
            led1=led1_arr.copy(),
            led2=led2_arr.copy(),
            swapped=np.zeros(t_arr.shape, dtype=bool),
            diagnostics={"mode": "none", "n_swapped": 0},
        )
    if led1_arr.shape[0] == 0:
        return CorrectedLEDIdentity(
            led1=led1_arr.copy(),
            led2=led2_arr.copy(),
            swapped=np.zeros(t_arr.shape, dtype=bool),
            diagnostics={"mode": "auto", "n_swapped": 0},
        )

    dual_valid = (
        mask_arr & np.isfinite(led1_arr).all(axis=1) & np.isfinite(led2_arr).all(axis=1)
    )
    if led_distance is None:
        distances = np.linalg.norm(led2_arr[dual_valid] - led1_arr[dual_valid], axis=1)
        led_distance = float(np.nanmedian(distances)) if distances.size else 0.04

    assignments = np.stack(
        [
            np.stack([led1_arr, led2_arr], axis=1),
            np.stack([led2_arr, led1_arr], axis=1),
        ],
        axis=1,
    )
    headings = np.arctan2(
        assignments[:, :, 1, 1] - assignments[:, :, 0, 1],
        assignments[:, :, 1, 0] - assignments[:, :, 0, 0],
    )

    n = t_arr.shape[0]
    valid_indices = np.flatnonzero(dual_valid)
    if valid_indices.size == 0:
        return CorrectedLEDIdentity(
            led1=led1_arr.copy(),
            led2=led2_arr.copy(),
            swapped=np.zeros(t_arr.shape, dtype=bool),
            diagnostics={
                "mode": "auto",
                "n_swapped": 0,
                "dual_led_frame_count": 0,
                "fraction_swapped": 0.0,
                "initial_state": config.initial_state,
                "global_identity_ambiguous": True,
                "message": (
                    "LED identity correction did not run because the session "
                    "has no dual-LED frames; auto mode requires at least one "
                    "frame with both LEDs finite to estimate the swap pattern."
                ),
                "transition_penalty": float(config.transition_penalty),
                "gyro_weight": float(config.gyro_weight),
            },
        )

    costs = np.full((valid_indices.size, 2), np.inf)
    back = np.zeros((valid_indices.size, 2), dtype=np.int8)
    first_idx = int(valid_indices[0])
    costs[0] = _observation_cost(
        assignments[first_idx], True, led_distance
    ) + _initial_state_cost(config.initial_state)

    for k in range(1, valid_indices.size):
        prev_idx = int(valid_indices[k - 1])
        idx = int(valid_indices[k])
        obs_cost = _observation_cost(assignments[idx], True, led_distance)
        for state in (0, 1):
            best_cost = np.inf
            best_prev = 0
            for prev in (0, 1):
                trans_cost = _transition_cost(
                    t_arr,
                    assignments,
                    headings,
                    dual_valid,
                    prev_idx,
                    idx,
                    prev,
                    state,
                    config=config,
                    t_imu=t_imu,
                    gyro_z=gyro_z,
                )
                candidate = costs[k - 1, prev] + trans_cost + obs_cost[state]
                if candidate < best_cost:
                    best_cost = candidate
                    best_prev = prev
            costs[k, state] = best_cost
            back[k, state] = best_prev

    valid_states = np.zeros(valid_indices.size, dtype=np.int8)
    valid_states[-1] = int(np.argmin(costs[-1]))
    for k in range(valid_indices.size - 1, 0, -1):
        valid_states[k - 1] = back[k, valid_states[k]]

    # Propagate the inferred state to *every* frame, not just the
    # dual-valid ones. Single-LED frames inside a swapped interval
    # carry the wrong physical-LED label too; gating swaps by
    # ``dual_valid`` used to leave them mislabeled, so the EKF
    # consumed a partial observation in the original (wrong)
    # identity. Carry-forward the most recent dual-valid state, and
    # for frames before the first dual-valid frame use that frame's
    # resolved state (the DP already picked it given the
    # ``initial_state`` prior).
    positions = np.searchsorted(valid_indices, np.arange(n), side="right") - 1
    positions = np.maximum(positions, 0)
    states = valid_states[positions]
    swapped = states == 1
    corrected1 = led1_arr.copy()
    corrected2 = led2_arr.copy()
    # For single-LED swapped frames, the finite LED moves to the
    # other slot and the NaN follows; ``led2_arr[swapped]`` carries
    # the NaN naturally so no special-casing is needed.
    corrected1[swapped] = led2_arr[swapped]
    corrected2[swapped] = led1_arr[swapped]
    return CorrectedLEDIdentity(
        led1=corrected1,
        led2=corrected2,
        swapped=swapped,
        diagnostics={
            "mode": "auto",
            "n_swapped": int(swapped.sum()),
            "fraction_swapped": float(swapped.mean()) if swapped.size else 0.0,
            "dual_led_frame_count": int(valid_indices.size),
            "led_distance_m": float(led_distance),
            "initial_state": config.initial_state,
            "global_identity_ambiguous": config.initial_state == "auto",
            "message": (
                "initial_state='auto' resolves continuity breaks but cannot "
                "determine a whole-session front/back label convention; set "
                "initial_state to 'original' or 'swapped' when that convention "
                "is known."
                if config.initial_state == "auto"
                else "initial_state prior applied."
            ),
            "transition_penalty": float(config.transition_penalty),
            "gyro_weight": float(config.gyro_weight),
        },
    )


def _validate_led_array(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_time, 2); got {arr.shape}.")
    return arr


def _observation_cost(
    assignment: np.ndarray, dual_valid: bool, led_distance: float
) -> np.ndarray:
    if not dual_valid:
        return np.zeros(2)
    spacing = np.linalg.norm(assignment[:, 1] - assignment[:, 0], axis=1)
    sigma = max(0.02, 0.25 * led_distance)
    return ((spacing - led_distance) / sigma) ** 2


def _initial_state_cost(initial_state: str) -> np.ndarray:
    if initial_state == "original":
        return np.array([0.0, np.inf])
    if initial_state == "swapped":
        return np.array([np.inf, 0.0])
    return np.zeros(2)


def _transition_cost(
    t_cam: np.ndarray,
    assignments: np.ndarray,
    headings: np.ndarray,
    dual_valid: np.ndarray,
    prev_i: int,
    i: int,
    prev_state: int,
    state: int,
    *,
    config: LedIdentityConfig,
    t_imu: np.ndarray | None,
    gyro_z: np.ndarray | None,
) -> float:
    penalty = config.transition_penalty if state != prev_state else 0.0
    if not (dual_valid[prev_i] and dual_valid[i]):
        return penalty

    dt = max(float(t_cam[i] - t_cam[prev_i]), 1e-6)
    prev = assignments[prev_i, prev_state]
    cur = assignments[i, state]
    led_step = np.linalg.norm(cur - prev, axis=1)
    continuity = float(np.sum((led_step / max(config.max_speed_mps * dt, 0.02)) ** 2))

    midpoint_speed = np.linalg.norm(cur.mean(axis=0) - prev.mean(axis=0)) / dt
    speed_cost = 0.0
    if midpoint_speed > config.max_speed_mps:
        speed_cost = (
            (midpoint_speed - config.max_speed_mps) / config.max_speed_mps
        ) ** 2

    heading_delta = _wrap(headings[i, state] - headings[prev_i, prev_state])
    expected_delta = 0.0
    if (
        config.gyro_weight > 0
        and t_imu is not None
        and gyro_z is not None
        and len(t_imu) >= 2
    ):
        t_mid = 0.5 * (t_cam[prev_i] + t_cam[i])
        expected_delta = float(np.interp(t_mid, t_imu, gyro_z) * dt)
    heading_continuity = (_wrap(heading_delta) / 0.75) ** 2
    heading_cost = (
        config.gyro_weight * (_wrap(heading_delta - expected_delta) / 0.5) ** 2
    )
    return float(penalty + continuity + speed_cost + heading_continuity + heading_cost)


def _wrap(angle: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))

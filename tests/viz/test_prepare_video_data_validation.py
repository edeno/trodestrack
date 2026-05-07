"""Regression tests for prepare_video_data input validation.

``prepare_video_data`` is exported from ``trodestrack.viz.utils`` and
called both directly (e.g. by tests and example scripts) and from
``create_diagnostic_video``. The top-level wrapper validates fps /
speedup at the public boundary, but direct callers used to fall
through into ``speedup / fps`` and trigger raw ``ZeroDivisionError``
or ``TypeError`` before any clear ``ValueError`` was raised.
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.utils import SimOut
from trodestrack.viz.utils import prepare_video_data


def _minimal_sim() -> SimOut:
    cfg = RatIMUSimConfig(
        duration_s=2.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,
        use_second_led=False,
        led_swap_prob=0.0,
        led_wall_reflection_prob=0.0,
    )
    return simulate_rat_imu(cfg, seed=0)


def test_prepare_video_data_rejects_zero_fps() -> None:
    """``fps=0`` must raise ValueError, not ZeroDivisionError."""
    sim = _minimal_sim()
    with pytest.raises(ValueError, match=r"fps must be a finite strictly-positive"):
        prepare_video_data(sim, fps=0, speedup=1.0)


def test_prepare_video_data_rejects_zero_speedup() -> None:
    """``speedup=0`` must raise ValueError, not produce a NaN-length arange."""
    sim = _minimal_sim()
    with pytest.raises(ValueError, match=r"speedup must be a finite strictly-positive"):
        prepare_video_data(sim, fps=30, speedup=0.0)


def test_prepare_video_data_rejects_string_fps() -> None:
    """Non-numeric fps must raise ValueError, not raw TypeError from division."""
    sim = _minimal_sim()
    with pytest.raises(ValueError, match=r"fps must be a finite strictly-positive"):
        prepare_video_data(sim, fps="30", speedup=1.0)  # type: ignore[arg-type]


def test_prepare_video_data_rejects_string_speedup() -> None:
    """Non-numeric speedup must raise ValueError, not raw TypeError."""
    sim = _minimal_sim()
    with pytest.raises(ValueError, match=r"speedup must be a finite strictly-positive"):
        prepare_video_data(sim, fps=30, speedup="1.0")  # type: ignore[arg-type]


def test_prepare_video_data_rejects_negative_inputs() -> None:
    """Negative fps / speedup are also rejected."""
    sim = _minimal_sim()
    with pytest.raises(ValueError, match=r"fps must be a finite strictly-positive"):
        prepare_video_data(sim, fps=-30, speedup=1.0)
    with pytest.raises(ValueError, match=r"speedup must be a finite strictly-positive"):
        prepare_video_data(sim, fps=30, speedup=-1.0)


def test_prepare_video_data_accepts_valid_inputs() -> None:
    """Sanity: valid fps / speedup still produce a populated video frame array."""
    sim = _minimal_sim()
    out = prepare_video_data(sim, fps=30, speedup=1.0)
    assert out["fps"] == 30
    assert out["n_frames"] > 0
    assert np.all(np.isfinite(out["t_video"]))

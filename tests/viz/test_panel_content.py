"""Panel-content tests for diagnostic-video artists.

The existing viz tests largely cover negative-space behaviour (custom colors,
non-PSD covariance, NaN handling, scrolling xlim). These tests cover the
positive-content path: when an artist is given known inputs, the rendered
line/text data must reflect those inputs. A regression that silently feeds
zeros, wrong-axis values, or stale buffers into the plots would otherwise
ship through the existing "file exists" smoke tests undetected.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trodestrack.viz.components import (
    BiasEstimatePanelArtist,
    HUDArtist,
    IMUPanelArtist,
    StateErrorPanelArtist,
)


class TestHUDArtist:
    def test_renders_speed_and_heading_in_state_text(self) -> None:
        fig, ax = plt.subplots()
        try:
            hud = HUDArtist(ax)
            hud.update(
                t=1.234,
                state={
                    "speed": 0.456,
                    "theta": np.deg2rad(45.0),
                    "led1_visible": True,
                    "led2_visible": False,
                },
            )
            rendered = hud.state_text.get_text()
            assert "t = 1.23 s" in rendered
            assert "v = 0.46 m/s" in rendered
            assert "45.0" in rendered  # heading
            # LED status — exact glyphs match the renderer convention.
            assert "LED1: ✓" in rendered
            assert "LED2: ✗" in rendered
        finally:
            plt.close(fig)

    def test_treats_missing_state_keys_as_zero_defaults(self) -> None:
        """A partial state dict must not raise; defaults render as zero."""
        fig, ax = plt.subplots()
        try:
            hud = HUDArtist(ax)
            hud.update(t=0.0, state={})
            rendered = hud.state_text.get_text()
            # Speed defaults to 0; missing LED flags render as ✗.
            assert "v = 0.00 m/s" in rendered
            assert "LED1: ✗" in rendered
            assert "LED2: ✗" in rendered
        finally:
            plt.close(fig)


class TestStateErrorPanelArtist:
    def test_velocity_error_buffers_reflect_input_in_cm_per_s(self) -> None:
        """Per-update velocity errors must appear as the rendered y-data."""
        fig, (ax_vel, ax_h) = plt.subplots(2, 1)
        try:
            artist = StateErrorPanelArtist(ax_vel, ax_h, window_s=10.0, fps=30)
            inputs = [
                (0.0, 1.0, -2.0, 3.0),
                (0.1, 4.0, -5.0, -6.0),
                (0.2, 7.0, -8.0, 9.0),
            ]
            for t, evx, evy, eh in inputs:
                artist.update(t, evx, evy, eh)

            x_vx, y_vx = artist.line_vx.get_data()
            _x_vy, y_vy = artist.line_vy.get_data()
            x_h, y_h = artist.line_heading.get_data()
            np.testing.assert_allclose(x_vx, [t for t, *_ in inputs])
            np.testing.assert_allclose(y_vx, [evx for _, evx, *_ in inputs])
            np.testing.assert_allclose(y_vy, [evy for _, _, evy, _ in inputs])
            np.testing.assert_allclose(x_h, [t for t, *_ in inputs])
            np.testing.assert_allclose(y_h, [eh for *_, eh in inputs])
        finally:
            plt.close(fig)


class TestBiasEstimatePanelArtist:
    def test_bias_line_y_data_matches_inputs(self) -> None:
        """Each line's y-data must match the corresponding bias series exactly."""
        fig, ax = plt.subplots()
        try:
            artist = BiasEstimatePanelArtist(ax, window_s=10.0, fps=30)
            samples = [
                (0.0, 0.001, 0.02, -0.03),
                (0.1, 0.002, 0.05, -0.04),
                (0.2, 0.003, 0.08, -0.05),
            ]
            for t, gb, abx, aby in samples:
                artist.update(t, gb, abx, aby)

            _, gyro_y = artist.line_gyro.get_data()
            _, ax_y = artist.line_ax.get_data()
            _, ay_y = artist.line_ay.get_data()
            np.testing.assert_allclose(gyro_y, [s[1] for s in samples])
            np.testing.assert_allclose(ax_y, [s[2] for s in samples])
            np.testing.assert_allclose(ay_y, [s[3] for s in samples])
        finally:
            plt.close(fig)

    def test_diverged_inf_bias_does_not_crash_ylim(self) -> None:
        """±inf bias estimates must not propagate into matplotlib's ylim call."""
        fig, ax = plt.subplots()
        try:
            artist = BiasEstimatePanelArtist(ax, window_s=10.0, fps=30)
            # Valid sample first to set buffer state.
            artist.update(0.0, 0.001, 0.02, -0.01)
            # Diverged sample with infinite bias.
            artist.update(0.1, np.inf, -np.inf, 0.05)
            # Render didn't raise — verify ylim remains finite.
            y_lo, y_hi = ax.get_ylim()
            assert np.isfinite(y_lo) and np.isfinite(y_hi)
        finally:
            plt.close(fig)


class TestIMUPanelArtist:
    def test_single_sample_mode_lines_reflect_inputs(self) -> None:
        """Each buffered sample must appear on the gyro/accel-x/accel-y lines."""
        fig, axes = plt.subplots(3, 1)
        try:
            artist = IMUPanelArtist(list(axes), window_s=10.0, fps=30)
            samples = [
                (0.0, {"gyro": 0.1, "accel_x": 1.0, "accel_y": 2.0}),
                (0.1, {"gyro": 0.2, "accel_x": 1.5, "accel_y": 2.5}),
                (0.2, {"gyro": 0.3, "accel_x": 2.0, "accel_y": 3.0}),
            ]
            for t, imu in samples:
                artist.update(t, imu_data=imu)

            _, gyro_y = artist.gyro_line.get_data()
            _, ax_y = artist.accel_x_line.get_data()
            _, ay_y = artist.accel_y_line.get_data()
            np.testing.assert_allclose(gyro_y, [imu["gyro"] for _, imu in samples])
            np.testing.assert_allclose(ax_y, [imu["accel_x"] for _, imu in samples])
            np.testing.assert_allclose(ay_y, [imu["accel_y"] for _, imu in samples])
        finally:
            plt.close(fig)

    def test_high_rate_mode_passes_raw_arrays_to_lines(self) -> None:
        """High-rate mode must set each line's data to the raw input arrays."""
        fig, axes = plt.subplots(3, 1)
        try:
            artist = IMUPanelArtist(list(axes), window_s=10.0, fps=30)
            t_raw = np.linspace(0.0, 0.5, 50)
            imu_raw = {
                "gyro": np.sin(t_raw * 5.0),
                "accel_x": np.cos(t_raw * 3.0),
                "accel_y": np.linspace(-1.0, 1.0, 50),
            }
            artist.update(t=0.5, t_raw=t_raw, imu_raw=imu_raw)

            np.testing.assert_allclose(artist.gyro_line.get_xdata(), t_raw)
            np.testing.assert_allclose(artist.gyro_line.get_ydata(), imu_raw["gyro"])
            np.testing.assert_allclose(
                artist.accel_x_line.get_ydata(), imu_raw["accel_x"]
            )
            np.testing.assert_allclose(
                artist.accel_y_line.get_ydata(), imu_raw["accel_y"]
            )
        finally:
            plt.close(fig)

"""Regression tests for diagnostic-video panel artists."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trodestrack.viz.components import NEESPanelArtist


def test_nees_panel_clears_line_when_window_is_nan_only():
    """A NaN-only rolling window must clear the rendered NEES line.

    The panel filters NaN samples out of its rolling buffer before
    plotting (so the NEES line stays continuous through finite values).
    Previously, when the window dropped to all-NaN it skipped
    ``set_data`` entirely, leaving the previously-plotted NEES value
    visible — which read as a current consistency value during
    dropouts where NEES is undefined.
    """
    fig, ax = plt.subplots()
    try:
        panel = NEESPanelArtist(ax, window_s=2.0, fps=10, state_dim=2)

        # Seed with a valid sample.
        panel.update(0.0, 1.5)
        y_after_valid = np.asarray(panel.line_nees.get_ydata())
        assert y_after_valid.size == 1 and y_after_valid[0] == 1.5

        # Advance well past window_s so the rolling buffer drops the
        # valid sample and contains only NaN.
        for i in range(1, 30):
            panel.update(i * 0.1, np.nan)
        assert all(np.isnan(v) for v in panel.nees_buffer), (
            "test setup: buffer should be all-NaN at this point"
        )

        x_after_nan = np.asarray(panel.line_nees.get_xdata())
        y_after_nan = np.asarray(panel.line_nees.get_ydata())
        assert x_after_nan.size == 0 and y_after_nan.size == 0, (
            f"NaN-only window must clear the line; got x={list(x_after_nan)}, "
            f"y={list(y_after_nan)} (stale reading visible during dropout)."
        )

        # And a fresh valid sample after the dropout must re-render.
        panel.update(3.5, 2.7)
        y_recovered = np.asarray(panel.line_nees.get_ydata())
        assert y_recovered.size >= 1 and y_recovered[-1] == 2.7
    finally:
        plt.close(fig)

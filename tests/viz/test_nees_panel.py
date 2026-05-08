"""Regression tests for diagnostic-video panel artists."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trodestrack.viz.components import (
    BiasEstimatePanelArtist,
    FilterArtist,
    NEESPanelArtist,
    ResidualPanelArtist,
)


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


def test_panels_tolerate_inf_diagnostics_without_crashing():
    """Inf residuals, biases, or NEES must not abort the panel update.

    A diverged filter or near-singular covariance can emit ±Inf
    diagnostics. Previously the panels filtered NaN but kept Inf, which
    propagated into matplotlib's ``set_ylim`` and raised
    ``Axis limits cannot be NaN or Inf`` — aborting the entire
    diagnostic-video render.
    """
    # ResidualPanelArtist: Inf residuals
    fig, ax = plt.subplots()
    try:
        panel = ResidualPanelArtist(ax, window_s=2.0, fps=10)
        panel.update(0.0, np.inf, np.inf)  # Both LEDs diverged
        # Follow up with a finite sample to confirm autoscaling recovers.
        panel.update(0.1, 0.5, 0.7)
        ylim = ax.get_ylim()
        assert np.all(np.isfinite(ylim)), f"residual y-limits not finite: {ylim}"
    finally:
        plt.close(fig)

    # BiasEstimatePanelArtist: Inf biases
    fig, ax = plt.subplots()
    try:
        panel = BiasEstimatePanelArtist(ax, window_s=2.0, fps=10)
        panel.update(0.0, np.inf, np.inf, np.inf)
        panel.update(0.1, 0.001, -0.005, 0.002)
        ylim = ax.get_ylim()
        assert np.all(np.isfinite(ylim)), f"bias y-limits not finite: {ylim}"
    finally:
        plt.close(fig)

    # NEESPanelArtist: Inf NEES
    fig, ax = plt.subplots()
    try:
        panel = NEESPanelArtist(ax, window_s=2.0, fps=10, state_dim=2)
        panel.update(0.0, np.inf)
        panel.update(0.1, 1.5)
        ylim = ax.get_ylim()
        assert np.all(np.isfinite(ylim)), f"NEES y-limits not finite: {ylim}"
        # Ensure the Inf sample wasn't stroked into the line (line should
        # contain only the finite sample).
        y = np.asarray(panel.line_nees.get_ydata())
        assert np.all(np.isfinite(y)), f"NEES line contains non-finite: {y}"
    finally:
        plt.close(fig)


def test_filter_artist_hides_overlay_on_non_finite_state():
    """Diverged filter inputs (NaN/Inf in mean or covariance) must not produce a
    bogus overlay.

    The previous code only validated the covariance shape, then wrote the
    NaN/Inf marker position straight into ``set_data`` and propagated
    them through ``np.linalg.eigh`` into the ellipse width / height.
    Hide the overlay (clear marker, zero ellipse) instead — same
    semantics as the residual / bias / NEES panels under non-finite
    inputs.
    """
    cases = [
        ("nan_mean_x", np.nan, 0.5, np.eye(2) * 1e-3),
        ("inf_mean_x", np.inf, 0.5, np.eye(2) * 1e-3),
        ("nan_mean_y", 0.5, np.nan, np.eye(2) * 1e-3),
        ("nan_cov", 0.5, 0.5, np.full((2, 2), np.nan)),
        ("inf_cov", 0.5, 0.5, np.full((2, 2), np.inf)),
    ]
    for label, x, y, P in cases:
        fig, ax = plt.subplots()
        try:
            artist = FilterArtist(ax)
            artist.update(x, y, P)
            mx = np.asarray(artist.pred_marker.get_xdata())
            my = np.asarray(artist.pred_marker.get_ydata())
            w = artist.uncertainty_ellipse.width
            h = artist.uncertainty_ellipse.height
            assert mx.size == 0 and my.size == 0, (
                f"{label}: marker should be hidden, got x={list(mx)}, y={list(my)}"
            )
            assert w == 0.0 and h == 0.0, (
                f"{label}: ellipse should be zeroed, got w={w}, h={h}"
            )
        finally:
            plt.close(fig)

    # Sanity: a finite update after a diverged update must restore the overlay.
    fig, ax = plt.subplots()
    try:
        artist = FilterArtist(ax)
        artist.update(np.nan, np.nan, np.eye(2) * 1e-3)
        artist.update(0.5, 0.5, np.eye(2) * 1e-3)
        mx = np.asarray(artist.pred_marker.get_xdata())
        assert mx.size == 1 and mx[0] == 0.5
        assert artist.uncertainty_ellipse.width > 0.0
    finally:
        plt.close(fig)


def test_filter_panels_quiet_on_first_sample_xlim():
    """Per-step filter panels must not warn on identical xlim for the
    first rendered sample.

    Each rolling-buffer panel sees exactly one sample on its first
    update, and previously called ``set_xlim(t, t)`` directly, which
    triggers matplotlib's "Attempting to set identical low and high
    xlims" warning. The shared ``_set_scrolling_xlim`` helper pads the
    range ±1 ms in that degenerate case. Run with warnings-as-errors
    so any reintroduction would fail the test.
    """
    import warnings

    from trodestrack.viz.components import (
        BiasEstimatePanelArtist,
        NEESPanelArtist,
        ResidualPanelArtist,
        StateErrorPanelArtist,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Attempting to set identical low and high xlims"
        )

        fig, ax = plt.subplots()
        try:
            ResidualPanelArtist(ax, window_s=2.0, fps=10).update(0.0, 0.5, 0.6)
        finally:
            plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        try:
            StateErrorPanelArtist(ax1, ax2, window_s=2.0, fps=10).update(
                0.0, 0.0, 0.0, 0.0
            )
        finally:
            plt.close(fig)

        fig, ax = plt.subplots()
        try:
            BiasEstimatePanelArtist(ax, window_s=2.0, fps=10).update(
                0.0, 0.001, -0.01, 0.005
            )
        finally:
            plt.close(fig)

        fig, ax = plt.subplots()
        try:
            NEESPanelArtist(ax, window_s=2.0, fps=10, state_dim=2).update(0.0, 1.5)
        finally:
            plt.close(fig)

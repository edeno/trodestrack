"""Regression tests for diagnostic-video TrailArtist and FilterArtist.

Both regressions guard previously silent failures in the overlay
artists:

* ``TrailArtist`` accepted a ``color=`` kwarg but its per-frame fade
  used hard-coded blue RGBA values, so any non-default trail color
  silently rendered as blue after the second sample.
* ``FilterArtist`` clipped negative eigenvalues of the position
  covariance to zero, masking non-PSD covariance failures by drawing
  a degenerate ellipse on top of the prediction marker.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from trodestrack.viz.components import FilterArtist, TrailArtist


def test_trail_artist_respects_custom_color() -> None:
    """``TrailArtist(color="red")`` must render red segments, not blue.

    The previous implementation stored the configured color but
    ignored it inside ``update``, where the per-segment RGBA list was
    hard-coded to blue (``(0.133, 0.4, 0.675, alpha)``). A red trail
    therefore rendered as blue from the second sample onward, which
    silently broke any callsite trying to distinguish multiple
    trails by color.
    """
    fig, ax = plt.subplots()
    try:
        artist = TrailArtist(ax, trail_length_s=1.0, fps=30, color="red")
        artist.update(0.0, 0.0)
        artist.update(0.1, 0.1)
        artist.update(0.2, 0.2)

        rendered = np.asarray(artist.lines.get_colors())
        assert rendered.shape[0] >= 1, (
            f"Expected at least one rendered segment after multiple updates, "
            f"got shape {rendered.shape}."
        )
        expected_rgb = np.asarray(to_rgb("red"))
        for row in rendered:
            assert np.allclose(row[:3], expected_rgb, atol=1e-6), (
                f"Trail RGB {row[:3].tolist()} does not match requested "
                f"color 'red' = {expected_rgb.tolist()}."
            )
    finally:
        plt.close(fig)


def test_filter_artist_clears_overlay_on_non_psd_covariance() -> None:
    """Non-PSD covariance must clear the overlay rather than silently render.

    A covariance with a negative eigenvalue is mathematically
    invalid. Clipping to zero used to draw a degenerate ellipse on
    top of the prediction marker, so a diverged filter looked
    healthy in the diagnostic video. The artist must instead clear
    both marker and ellipse so the failure is visible.
    """
    fig, ax = plt.subplots()
    try:
        artist = FilterArtist(ax)

        # Sanity: a valid PSD covariance still renders the overlay.
        artist.update(0.5, 0.5, np.diag([1e-3, 1e-3]))
        x_after_valid = np.asarray(artist.pred_marker.get_xdata())
        y_after_valid = np.asarray(artist.pred_marker.get_ydata())
        assert x_after_valid.tolist() == [0.5] and y_after_valid.tolist() == [0.5]
        assert artist.uncertainty_ellipse.width > 0.0
        assert artist.uncertainty_ellipse.height > 0.0

        # Non-PSD: the negative eigenvalue must trigger a clear.
        artist.update(0.5, 0.5, np.diag([-1e-3, 1e-3]))
        x = np.asarray(artist.pred_marker.get_xdata())
        y = np.asarray(artist.pred_marker.get_ydata())
        assert x.size == 0 and y.size == 0, (
            f"Non-PSD covariance must clear the prediction marker; got "
            f"x={list(x)}, y={list(y)}."
        )
        assert artist.uncertainty_ellipse.width == 0.0
        assert artist.uncertainty_ellipse.height == 0.0
    finally:
        plt.close(fig)

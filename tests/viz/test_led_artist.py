"""Regression tests for the diagnostic-video LEDArtist."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trodestrack.viz.components import LEDArtist


def test_dropout_before_first_visible_sample_does_not_render_at_origin():
    """An invisible-first session must not render a phantom red X at (0, 0).

    ``LEDArtist`` stores a "last known position" used to anchor the
    dropout marker. Initializing it to ``(0.0, 0.0)`` made the first
    invisible update draw a red X at the origin even though no LED had
    ever been detected. Initialize as ``None`` and only render the
    dropout marker once a real visible sample has been seen.
    """
    fig, ax = plt.subplots()
    try:
        artist = LEDArtist(ax, led_id=1, color="blue")

        # Update with visible=False before any visible sample.
        artist.update(0.0, 0.0, visible=False)
        x = np.asarray(artist.dropout_marker.get_xdata())
        y = np.asarray(artist.dropout_marker.get_ydata())
        assert x.size == 0 and y.size == 0, (
            f"Pre-visible dropout should leave the marker hidden; got "
            f"x={list(x)}, y={list(y)} (phantom render at origin)."
        )

        # After a real visible sample, the next dropout marks that point.
        artist.update(0.4, 0.6, visible=True)
        artist.update(0.4, 0.6, visible=False)
        x = np.asarray(artist.dropout_marker.get_xdata())
        y = np.asarray(artist.dropout_marker.get_ydata())
        assert x.tolist() == [0.4] and y.tolist() == [0.6]
    finally:
        plt.close(fig)

"""Visualization styles and color schemes.

Consistent with example plotting style (Tufte/Gelman principles).
"""

import matplotlib.pyplot as plt

# ColorBrewer-inspired palette (color-blind safe, print-friendly).
# The Okabe-Ito entries (``okabe_ito_blue`` / ``okabe_ito_orange``) are taken
# from the Okabe-Ito 8-color palette and are used where a distinctly
# colorblind-safe pair is required (e.g. start/end trajectory markers).
COLORS = {
    "blue": "#2166AC",
    "red": "#B2182B",
    "gray": "#666666",
    "light_gray": "#CCCCCC",
    "orange": "#D6604D",
    "green": "#1B7837",
    "purple": "#762A83",
    "yellow": "#FDB863",
    "okabe_ito_blue": "#0072B2",
    "okabe_ito_orange": "#E69F00",
    "verdict_pass": "#2C7A2C",
    "verdict_fail": "#C13030",
}


def apply_tufte_style() -> None:
    """Apply Tufte/Gelman plotting style to matplotlib.

    Principles:
    - Minimal chartjunk, maximum data-ink ratio
    - Remove unnecessary spines
    - Subtle grids
    - Small, clean fonts
    """
    plt.rcParams.update(
        {
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            # Axes styling (minimal spines, thin lines)
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid (subtle, unobtrusive)
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linewidth": 0.4,
            # Ticks
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Legend (frameless, small)
            "legend.frameon": False,
            "legend.fontsize": 8,
            # Figure quality
            "figure.dpi": 100,
        }
    )

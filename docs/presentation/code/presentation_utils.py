"""
Presentation Figure Utilities

MANDATORY STANDARDS (must follow for all figures):
1. ALWAYS use constrained_layout=True
2. ALWAYS place legends OUTSIDE plot area: bbox_to_anchor=(1.02, 1)
3. ALWAYS use labelpad=10 for axis labels
4. CHECK for warnings before saving

Additional standards:
- 16:9 aspect ratio (10" × 5.625")
- Large fonts (title 32-48pt, labels 20-24pt, legend 20pt)
- Bold, thick lines (4-5pt width)
- High contrast colors
- No overlapping elements
- Clean, minimal style (no grids)
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Standard presentation slide size (16:9)
SLIDE_WIDTH = 10.0  # inches
SLIDE_HEIGHT = 5.625  # inches
SLIDE_DPI = 150

# Wong colorblind-friendly palette (high contrast)
WONG = {
    "blue": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "red": "#D55E00",
    "black": "#000000",
    "sky_blue": "#0072B2",
}

# Project color palette (for consistency with slides)
COLORS = {
    "blue": "#2E86AB",
    "orange": "#F77F00",
    "green": "#06A77D",
    "red": "#D62828",
    "gray": "#6C757D",
}


def set_presentation_style():
    """Set matplotlib rcParams for presentation figures"""
    plt.style.use("default")  # Start clean

    # Font sizes (optimized for projection)
    rcParams["font.size"] = 20
    rcParams["axes.titlesize"] = 32
    rcParams["axes.labelsize"] = 24
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18
    rcParams["legend.fontsize"] = 20
    rcParams["figure.titlesize"] = 36

    # Font weights
    rcParams["font.weight"] = "normal"
    rcParams["axes.titleweight"] = "bold"
    rcParams["axes.labelweight"] = "bold"

    # Line widths (bold for projection)
    rcParams["lines.linewidth"] = 4
    rcParams["axes.linewidth"] = 2
    rcParams["grid.linewidth"] = 1.5
    rcParams["patch.linewidth"] = 2

    # Marker sizes
    rcParams["lines.markersize"] = 10
    rcParams["lines.markeredgewidth"] = 2

    # Grid (disabled by default for clean look)
    rcParams["axes.grid"] = False
    rcParams["grid.alpha"] = 0.3

    # Figure appearance
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    rcParams["savefig.dpi"] = SLIDE_DPI
    rcParams["savefig.bbox"] = "tight"

    # Spines (clean look)
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False

    # Font
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def create_presentation_figure(ncols=1, nrows=1, **kwargs):
    """
    Create figure sized for 16:9 presentation slides

    Args:
        ncols: Number of columns (default 1)
        nrows: Number of rows (default 1)
        **kwargs: Additional arguments passed to plt.subplots()

    Returns:
        fig, axes
    """
    # Set style first
    set_presentation_style()

    # Create figure
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(SLIDE_WIDTH, SLIDE_HEIGHT),
        dpi=SLIDE_DPI,
        constrained_layout=True,  # Prevents overlapping labels
        **kwargs,
    )

    return fig, axes


def clean_axis(ax, grid=False):
    """
    Clean up axis for presentation (remove clutter)

    Args:
        ax: Matplotlib axis
        grid: Show grid (default False for clean look)
    """
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bold remaining spines
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    # Grid (subtle if enabled)
    if grid:
        ax.grid(True, alpha=0.3, linewidth=1, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)

    # Tick parameters
    ax.tick_params(
        labelsize=18,
        width=2,
        length=6,
        direction="out",
    )

    # Limit number of ticks for readability
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))


def add_title(ax, title, fontsize=32, color="black", pad=15):
    """
    Add large, bold title optimized for projection

    Args:
        ax: Matplotlib axis
        title: Title text
        fontsize: Font size (default 32pt)
        color: Title color
        pad: Padding above title (default 15pt)
    """
    ax.set_title(title, fontsize=fontsize, fontweight="bold", color=color, pad=pad)


def add_legend(
    ax, fontsize=20, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, **kwargs
):
    """
    Add large, readable legend OUTSIDE plot area (MANDATORY)

    IMPORTANT: Legends are ALWAYS placed outside to prevent overlap with data.
    Default placement is to the right of the plot.

    Args:
        ax: Matplotlib axis
        fontsize: Font size (default 20pt)
        loc: Legend location (default 'upper left')
        bbox_to_anchor: Position relative to axes (default (1.02, 1) = outside right)
        frameon: Show frame (default True for visibility)
        **kwargs: Additional arguments passed to ax.legend()

    Returns:
        legend object
    """
    # MANDATORY: Place legend outside plot area
    legend = ax.legend(
        fontsize=fontsize,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=frameon,
        fancybox=True,
        **kwargs,
    )

    return legend


def annotate_with_arrow(
    ax, text, xy, xytext, fontsize=20, color="black", bbox_color="yellow", arrow_width=3
):
    """
    Add annotation with arrow (ensures no overlap with bbox)

    Args:
        ax: Matplotlib axis
        text: Annotation text
        xy: Point to annotate (x, y)
        xytext: Text position (x, y) or offset
        fontsize: Font size (default 20pt)
        color: Text color
        bbox_color: Background box color
        arrow_width: Arrow line width
    """
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=arrow_width, color=color),
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor=bbox_color,
            alpha=0.8,
            edgecolor=color,
            linewidth=2,
        ),
        zorder=100,  # Ensure annotation is on top
    )


def save_presentation_figure(filepath, dpi=SLIDE_DPI, **kwargs):
    """
    Save figure with correct settings for presentations

    Args:
        filepath: Output path
        dpi: Resolution (default 150 for screen)
        **kwargs: Additional arguments passed to plt.savefig()
    """
    plt.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        **kwargs,
    )
    print(f"✓ Saved: {filepath}")


def ensure_no_overlap(ax, margin_factor=0.1):
    """
    Ensure plot elements don't overlap by adding margins

    Args:
        ax: Matplotlib axis
        margin_factor: Margin as fraction of data range (default 10%)
    """
    # Get current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Add margins
    x_margin = (xlim[1] - xlim[0]) * margin_factor
    y_margin = (ylim[1] - ylim[0]) * margin_factor

    ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
    ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)


# Presentation-optimized plotting functions


def plot_trajectory(
    ax,
    x,
    y,
    color=COLORS["blue"],
    linewidth=4,
    label=None,
    marker=None,
    markersize=10,
    alpha=1.0,
):
    """Plot trajectory with presentation-optimized styling"""
    return ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        label=label,
        marker=marker,
        markersize=markersize,
        alpha=alpha,
        markeredgecolor="black" if marker else None,
        markeredgewidth=2 if marker else 0,
        zorder=10,
    )


def scatter_points(
    ax, x, y, color=COLORS["red"], size=100, label=None, alpha=0.7, edgecolor="black"
):
    """Scatter plot with presentation-optimized styling"""
    return ax.scatter(
        x,
        y,
        s=size,
        color=color,
        alpha=alpha,
        label=label,
        edgecolors=edgecolor,
        linewidths=2,
        zorder=10,
    )


def add_reference_line(
    ax, value, axis="x", color=COLORS["orange"], linewidth=3, label=None
):
    """Add horizontal or vertical reference line"""
    if axis == "x":
        line = ax.axvline(
            value, color=color, linewidth=linewidth, linestyle="--", label=label
        )
    else:  # y
        line = ax.axhline(
            value, color=color, linewidth=linewidth, linestyle="--", label=label
        )
    return line


def format_comparison_panels(axes, titles, xlabel="", ylabel=""):
    """
    Format side-by-side comparison panels

    Args:
        axes: List of axes
        titles: List of titles (one per axis)
        xlabel: X-axis label (applied to all)
        ylabel: Y-axis label (applied to leftmost only)
    """
    for i, (ax, title) in enumerate(zip(axes, titles, strict=False)):
        add_title(ax, title, fontsize=28, pad=10)
        clean_axis(ax)

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=20, fontweight="bold")

        # Only add ylabel to leftmost panel
        if ylabel and i == 0:
            ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")

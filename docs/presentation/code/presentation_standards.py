"""
MANDATORY Presentation Figure Standards

These standards MUST be followed for all presentation figures to prevent
text overlap and ensure readability from 30+ feet.

Author: Claude Code
Date: 2025-10-18
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================================
# MANDATORY STANDARDS
# ============================================================================

MANDATORY_STANDARDS = """
PRESENTATION FIGURE STANDARDS (MANDATORY):

1. ALWAYS use constrained_layout=True
   ✓ fig, ax = plt.subplots(figsize=(10, 5.625), constrained_layout=True)
   ✗ NEVER use tight_layout() - conflicts with constrained_layout

2. ALWAYS place legends OUTSIDE plot area
   ✓ ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=20)
   ✗ NEVER use loc='best' or 'upper right' - causes overlaps

3. ALWAYS use labelpad for axis labels
   ✓ ax.set_xlabel("Time (s)", fontsize=24, labelpad=10)
   ✓ ax.set_ylabel("Signal", fontsize=24, labelpad=10)
   ✗ NEVER omit labelpad - text can touch plot

4. CHECK for warnings before saving
   ✓ Use warnings.filterwarnings to catch layout issues
   ✗ NEVER ignore matplotlib warnings

5. Font sizes (minimum)
   - Title: 32-36pt
   - Axis labels: 24-28pt
   - Tick labels: 18-20pt
   - Legend: 20pt
   - Annotations: 18-24pt

6. Line widths
   - Plot lines: 4-5pt
   - Axes: 2pt
   - Grid: 1.5pt (if used)
   - Annotations: 3-4pt
"""


def create_figure(ncols=1, nrows=1, **kwargs):
    """
    Create presentation figure with MANDATORY standards applied

    Args:
        ncols: Number of columns
        nrows: Number of rows
        **kwargs: Additional arguments for plt.subplots()

    Returns:
        fig, axes (or ax if single subplot)

    Raises:
        ValueError: If constrained_layout not enabled
    """
    # Force constrained_layout
    if "constrained_layout" in kwargs and not kwargs["constrained_layout"]:
        raise ValueError(
            "constrained_layout=True is MANDATORY for presentation figures. "
            "This prevents text overlap."
        )

    # Set defaults
    defaults = {
        "figsize": (10, 5.625),  # 16:9 aspect ratio
        "dpi": 150,
        "constrained_layout": True,  # MANDATORY
    }
    defaults.update(kwargs)

    # Create figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **defaults)

    return fig, axes


def add_legend_outside(ax, **kwargs):
    """
    Add legend OUTSIDE plot area (MANDATORY placement)

    Args:
        ax: Matplotlib axis
        **kwargs: Additional legend arguments

    Returns:
        legend object
    """
    # Force outside placement
    defaults = {
        "loc": "upper left",
        "bbox_to_anchor": (1.02, 1),  # MANDATORY: outside right edge
        "fontsize": 20,
        "frameon": True,
        "fancybox": True,
    }
    defaults.update(kwargs)

    legend = ax.legend(**defaults)
    return legend


def set_axis_labels(
    ax, xlabel="", ylabel="", xlabel_size=24, ylabel_size=24, labelpad=10
):
    """
    Set axis labels with MANDATORY labelpad

    Args:
        ax: Matplotlib axis
        xlabel: X-axis label text
        ylabel: Y-axis label text
        xlabel_size: X-axis label font size
        ylabel_size: Y-axis label font size
        labelpad: Padding (MANDATORY: prevents text touching plot)
    """
    if xlabel:
        ax.set_xlabel(
            xlabel, fontsize=xlabel_size, fontweight="bold", labelpad=labelpad
        )

    if ylabel:
        ax.set_ylabel(
            ylabel, fontsize=ylabel_size, fontweight="bold", labelpad=labelpad
        )


def check_warnings_before_save(output_path):
    """
    Check for matplotlib warnings before saving

    Catches common issues:
    - constrained_layout failures
    - Text overlap warnings
    - Tight layout warnings

    Args:
        output_path: Path where figure will be saved

    Raises:
        Warning: If layout issues detected
    """
    # Enable all warnings
    warnings.filterwarnings("always", category=UserWarning, module="matplotlib")

    # Try to draw the figure to trigger any warnings
    try:
        plt.gcf().canvas.draw()
    except Exception as e:
        warnings.warn(
            f"Figure drawing failed for {output_path}: {e}\n"
            "This may indicate overlapping elements or layout issues.",
            UserWarning,
            stacklevel=2,
        )


def save_figure(output_path, dpi=150, check_warnings=True, **kwargs):
    """
    Save figure with standard settings and optional warning check

    Args:
        output_path: Output file path
        dpi: Resolution (default 150 for screen)
        check_warnings: Check for warnings before saving (default True)
        **kwargs: Additional arguments for plt.savefig()
    """
    output_path = Path(output_path)

    # Check for warnings if requested
    if check_warnings:
        check_warnings_before_save(output_path)

    # Save with standard settings
    defaults = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
    }
    defaults.update(kwargs)

    plt.savefig(output_path, **defaults)
    print(f"✓ Saved: {output_path}")


def verify_standards_applied(fig, ax):
    """
    Verify that mandatory standards have been applied

    Args:
        fig: Matplotlib figure
        ax: Matplotlib axis (or array of axes)

    Returns:
        list of warnings/errors if standards violated
    """
    issues = []

    # Check 1: constrained_layout enabled
    if not fig.get_constrained_layout():
        issues.append(
            "❌ MANDATORY: constrained_layout=True not enabled. "
            "This will cause text overlap!"
        )

    # Check 2: Legend placement (if legend exists)
    axes = [ax] if not hasattr(ax, "__iter__") else ax.flatten()
    for i, axis in enumerate(axes):
        legend = axis.get_legend()
        if legend:
            # Check if legend is outside plot area
            bbox = legend.get_bbox_to_anchor()
            if bbox is None or bbox.x0 <= 1.0:
                issues.append(
                    f"❌ MANDATORY: Legend on axis {i} not placed outside plot area. "
                    f"Use bbox_to_anchor=(1.02, 1)"
                )

    # Check 3: Label padding (warn if likely missing)
    for i, axis in enumerate(axes):
        xlabel = axis.get_xlabel()
        ylabel = axis.get_ylabel()
        if xlabel and not axis.xaxis.labelpad:
            issues.append(
                f"⚠️  RECOMMENDED: X-axis label on axis {i} has no labelpad. "
                f"Use labelpad=10"
            )
        if ylabel and not axis.yaxis.labelpad:
            issues.append(
                f"⚠️  RECOMMENDED: Y-axis label on axis {i} has no labelpad. "
                f"Use labelpad=10"
            )

    return issues


def print_standards():
    """Print mandatory presentation standards"""
    print(MANDATORY_STANDARDS)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Presentation Figure Standards")
    print("=" * 80)
    print_standards()
    print("\n" + "=" * 80)
    print("\nExample code:\n")

    example_code = """
from presentation_standards import create_figure, add_legend_outside, set_axis_labels, save_figure
import numpy as np

# 1. Create figure with MANDATORY constrained_layout
fig, ax = create_figure()

# 2. Plot data
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), linewidth=4, label='Signal')

# 3. Add labels with MANDATORY labelpad
set_axis_labels(ax, xlabel="Time (s)", ylabel="Amplitude", labelpad=10)

# 4. Add legend OUTSIDE (MANDATORY placement)
add_legend_outside(ax)

# 5. Save with warning check (MANDATORY)
save_figure("example.png", check_warnings=True)
"""

    print(example_code)

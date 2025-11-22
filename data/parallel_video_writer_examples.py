"""Examples demonstrating how to use parallel_video_writer.

This script shows several common use cases for creating videos with matplotlib
using parallel rendering for improved performance.

Note: Functions passed to create_parallel_video must be defined at module level
(not as nested functions) to be pickle-able for multiprocessing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from parallel_video_writer import VideoConfig, create_parallel_video

# ============================================================================
# Example 1: Simple sine wave
# ============================================================================


def setup_figure_sine():
    """Setup figure for sine wave example."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 4 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    ax.grid(True, alpha=0.3)
    axes = {"main": ax}
    return fig, axes


def render_frame_sine(fig, axes, frame_idx, data):
    """Update plot for sine wave frame."""
    ax = axes["main"]
    ax.clear()

    # Calculate how many points to show
    n_show = int((frame_idx / 100) * data["n_points_total"])

    # Plot growing sine wave
    ax.plot(data["x"][:n_show], data["y"][:n_show], "b-", linewidth=2, label="sin(x)")

    # Restore axis properties
    ax.set_xlim(0, 4 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Frame {frame_idx + 1}/100")


def example_simple_sine_wave():
    """Example 1: Simple growing sine wave."""
    print("\n=== Example 1: Simple sine wave ===")

    # Prepare data
    n_frames = 100
    x = np.linspace(0, 4 * np.pi, 200)
    y = np.sin(x)
    frame_data = {"x": x, "y": y, "n_points_total": len(x)}

    # Create video
    output_dir = Path(__file__).parent / "example_videos"
    output_dir.mkdir(exist_ok=True)

    create_parallel_video(
        n_frames=n_frames,
        output_path=str(output_dir / "sine_wave.mp4"),
        render_frame_func=render_frame_sine,
        setup_figure_func=setup_figure_sine,
        frame_data=frame_data,
        config=VideoConfig(fps=30.0, dpi=100, max_workers=4),
    )


# ============================================================================
# Example 2: Multiple subplots
# ============================================================================


def setup_figure_subplots():
    """Setup figure for multiple subplots example."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Synchronized Waveforms", fontsize=14, fontweight="bold")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 4 * np.pi)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("sin(t)")
    ax2.set_ylabel("cos(t)")
    ax3.set_ylabel("sin(2t)")
    ax3.set_xlabel("t")

    axes = {"sin": ax1, "cos": ax2, "sin2": ax3}
    return fig, axes


def render_frame_subplots(fig, axes, frame_idx, data):
    """Update all subplots for this frame."""
    n_show = int((frame_idx / 120) * data["n_points"])
    t_show = data["t"][:n_show]

    # Update each subplot
    axes["sin"].clear()
    axes["sin"].plot(t_show, np.sin(t_show), "b-", linewidth=2)
    axes["sin"].set_xlim(0, 4 * np.pi)
    axes["sin"].set_ylim(-1.5, 1.5)
    axes["sin"].set_ylabel("sin(t)")
    axes["sin"].grid(True, alpha=0.3)

    axes["cos"].clear()
    axes["cos"].plot(t_show, np.cos(t_show), "r-", linewidth=2)
    axes["cos"].set_xlim(0, 4 * np.pi)
    axes["cos"].set_ylim(-1.5, 1.5)
    axes["cos"].set_ylabel("cos(t)")
    axes["cos"].grid(True, alpha=0.3)

    axes["sin2"].clear()
    axes["sin2"].plot(t_show, np.sin(2 * t_show), "g-", linewidth=2)
    axes["sin2"].set_xlim(0, 4 * np.pi)
    axes["sin2"].set_ylim(-1.5, 1.5)
    axes["sin2"].set_ylabel("sin(2t)")
    axes["sin2"].set_xlabel("t")
    axes["sin2"].grid(True, alpha=0.3)

    # Update suptitle with frame info
    fig.suptitle(
        f"Synchronized Waveforms (Frame {frame_idx + 1}/120)",
        fontsize=14,
        fontweight="bold",
    )


def example_multiple_subplots():
    """Example 2: Multiple subplots with different data."""
    print("\n=== Example 2: Multiple subplots ===")

    # Prepare data
    n_frames = 120
    t = np.linspace(0, 4 * np.pi, 300)
    frame_data = {"t": t, "n_points": len(t)}

    # Create video
    output_dir = Path(__file__).parent / "example_videos"
    output_dir.mkdir(exist_ok=True)

    create_parallel_video(
        n_frames=n_frames,
        output_path=str(output_dir / "multiple_subplots.mp4"),
        render_frame_func=render_frame_subplots,
        setup_figure_func=setup_figure_subplots,
        frame_data=frame_data,
        config=VideoConfig(fps=30.0, dpi=100, max_workers=4),
    )


# ============================================================================
# Example 3: Particle animation
# ============================================================================


def setup_figure_particles():
    """Setup figure for particle animation example."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    axes = {"main": ax}
    return fig, axes


def render_frame_particles(fig, axes, frame_idx, data):
    """Update particle positions for this frame."""
    ax = axes["main"]
    ax.clear()

    # Calculate particle positions
    t = (frame_idx / 150) * 2 * np.pi
    radius = 0.5 + 0.3 * np.sin(3 * t)
    x = radius * np.cos(data["angles"] + t)
    y = radius * np.sin(data["angles"] + t)

    # Color by angle
    colors = plt.cm.hsv(data["angles"] / (2 * np.pi))

    # Plot particles
    ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors="black", linewidth=1)

    # Restore axis properties
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Rotating Particles (Frame {frame_idx + 1}/150)")


def example_particle_animation():
    """Example 3: Animated particles with scatter plot."""
    print("\n=== Example 3: Particle animation ===")

    # Prepare data
    n_frames = 150
    n_particles = 50
    angles = np.linspace(0, 2 * np.pi, n_particles)
    frame_data = {"angles": angles, "n_particles": n_particles}

    # Create video
    output_dir = Path(__file__).parent / "example_videos"
    output_dir.mkdir(exist_ok=True)

    create_parallel_video(
        n_frames=n_frames,
        output_path=str(output_dir / "particles.mp4"),
        render_frame_func=render_frame_particles,
        setup_figure_func=setup_figure_particles,
        frame_data=frame_data,
        config=VideoConfig(fps=30.0, dpi=100, max_workers=4),
    )


# ============================================================================
# Example 4: Heatmap evolution
# ============================================================================


def setup_figure_heatmap():
    """Setup figure for heatmap evolution example."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    axes = {"main": ax}
    return fig, axes


def render_frame_heatmap(fig, axes, frame_idx, data):
    """Update heatmap for this frame."""
    ax = axes["main"]

    # Calculate evolving Gaussian
    t = (frame_idx / 100) * 2 * np.pi
    center_x = 0.5 * np.cos(2 * t)
    center_y = 0.5 * np.sin(2 * t)
    sigma = 0.5 + 0.3 * np.sin(3 * t)

    Z = np.exp(-((data["X"] - center_x) ** 2 + (data["Y"] - center_y) ** 2) / (2 * sigma**2))

    # Reuse or create image and colorbar (avoid creating multiple colorbars)
    if "im" not in axes:
        # First frame: create image and colorbar
        im = ax.imshow(
            Z,
            extent=[-3, 3, -3, 3],
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        axes["im"] = im
        axes["cbar"] = plt.colorbar(im, ax=ax, label="Intensity")
    else:
        # Subsequent frames: just update image data
        axes["im"].set_data(Z)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Gaussian Wave (Frame {frame_idx + 1}/100)")


def example_heatmap_evolution():
    """Example 4: Evolving 2D heatmap."""
    print("\n=== Example 4: Heatmap evolution ===")

    # Prepare data
    n_frames = 100
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    frame_data = {"X": X, "Y": Y, "x": x, "y": y}

    # Create video
    output_dir = Path(__file__).parent / "example_videos"
    output_dir.mkdir(exist_ok=True)

    create_parallel_video(
        n_frames=n_frames,
        output_path=str(output_dir / "heatmap.mp4"),
        render_frame_func=render_frame_heatmap,
        setup_figure_func=setup_figure_heatmap,
        frame_data=frame_data,
        config=VideoConfig(fps=30.0, dpi=100, max_workers=4),
    )


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all examples."""
    print("=" * 60)
    print("Parallel Video Writer Examples")
    print("=" * 60)

    example_simple_sine_wave()
    example_multiple_subplots()
    example_particle_animation()
    example_heatmap_evolution()

    # Get actual output directory
    output_dir = Path(__file__).parent / "example_videos"

    print("\n" + "=" * 60)
    print("All examples complete!")
    print(f"Videos saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

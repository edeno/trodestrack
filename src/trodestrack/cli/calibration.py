"""Interactive homography calibration tool."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from trodestrack.geom.homography import compute_homography_from_corners


class HomographyCalibrator:
    """Interactive tool for homography calibration from arena corners."""

    def __init__(self, image_path: Path, arena_width_cm: float, arena_height_cm: float):
        """
        Initialize calibrator.

        Args:
            image_path: Path to video frame image
            arena_width_cm: Arena width in cm
            arena_height_cm: Arena height in cm
        """
        self.image_path = image_path
        self.arena_width_cm = arena_width_cm
        self.arena_height_cm = arena_height_cm

        # Pixel coordinates of clicked corners (top-left, top-right, bottom-right, bottom-left)
        self.pixel_corners: List[Tuple[float, float]] = []

        # Corresponding real-world coordinates
        self.real_corners = np.array(
            [
                [0, 0],  # top-left
                [arena_width_cm, 0],  # top-right
                [arena_width_cm, arena_height_cm],  # bottom-right
                [0, arena_height_cm],  # bottom-left
            ]
        )

        # UI state
        self.fig: Optional[Any] = None
        self.ax: Optional[Any] = None
        self.image: Optional[Any] = None
        self.points: List[Tuple[float, float]] = []
        self.completed = False

    def load_image(self) -> bool:
        """
        Load and display the calibration image.

        Returns:
            True if successful, False otherwise
        """
        try:
            if PIL_AVAILABLE:
                self.image = Image.open(self.image_path)
                if self.image.mode != "RGB":
                    self.image = self.image.convert("RGB")
                self.image = np.array(self.image)
            else:
                # Fallback to matplotlib's image loading
                self.image = plt.imread(self.image_path)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax or len(self.pixel_corners) >= 4:
            return

        # Add clicked point
        x, y = event.xdata, event.ydata
        self.pixel_corners.append((x, y))

        # Plot the point
        point = self.ax.plot(x, y, "ro", markersize=8)[0]
        self.points.append(point)

        # Add label
        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.ax.annotate(
            corner_names[len(self.pixel_corners) - 1],
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            color="red",
            weight="bold",
        )

        # Update display
        self.fig.canvas.draw()

        print(f"Corner {len(self.pixel_corners)}: ({x:.1f}, {y:.1f})")

        if len(self.pixel_corners) == 4:
            print("\nAll corners selected! Click 'Compute Homography' to calculate.")
            self.button_compute.set_active(True)

    def on_compute(self, event):
        """Compute homography from selected corners."""
        if len(self.pixel_corners) != 4:
            print("Error: Need exactly 4 corners!")
            return

        try:
            pixel_corners = np.array(self.pixel_corners)

            # Compute homography matrix
            H = compute_homography_from_corners(pixel_corners, self.real_corners)

            print("\nComputed homography matrix:")
            print(H)

            # Store result
            self.homography_matrix = H
            self.completed = True

            # Enable save button
            self.button_save.set_active(True)

            print("\nHomography computed successfully! Click 'Save & Exit' to save.")

        except Exception as e:
            print(f"Error computing homography: {e}")

    def on_reset(self, event):
        """Reset all selected corners."""
        self.pixel_corners.clear()

        # Remove plotted points
        for point in self.points:
            point.remove()
        self.points.clear()

        # Clear annotations
        for child in self.ax.get_children():
            if hasattr(child, "get_text"):
                child.remove()

        # Reset buttons
        self.button_compute.set_active(False)
        self.button_save.set_active(False)

        self.fig.canvas.draw()
        print(
            "Reset completed. Click 4 corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left"
        )

    def on_save(self, event):
        """Save homography and exit."""
        if not self.completed:
            print("Error: Homography not computed yet!")
            return

        # This will be handled by the main calibration function
        plt.close(self.fig)

    def run_calibration(self) -> Optional[np.ndarray]:
        """
        Run the interactive calibration.

        Returns:
            Homography matrix if successful, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Error: matplotlib is required for interactive calibration")
            print("Install with: uv add matplotlib")
            return None

        if not self.load_image():
            return None

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title(
            f"Homography Calibration - {self.image_path.name}\\n"
            f"Click 4 arena corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left"
        )
        self.ax.set_xlabel("Pixel X")
        self.ax.set_ylabel("Pixel Y")

        # Add instruction text
        instruction_text = (
            f"Arena size: {self.arena_width_cm} x {self.arena_height_cm} cm\\n"
            "1. Click top-left corner\\n"
            "2. Click top-right corner\\n"
            "3. Click bottom-right corner\\n"
            "4. Click bottom-left corner"
        )

        self.ax.text(
            0.02,
            0.98,
            instruction_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Add buttons
        ax_reset = plt.axes([0.15, 0.02, 0.1, 0.04])  # type: ignore
        ax_compute = plt.axes([0.3, 0.02, 0.15, 0.04])  # type: ignore
        ax_save = plt.axes([0.5, 0.02, 0.15, 0.04])  # type: ignore

        self.button_reset = Button(ax_reset, "Reset")
        self.button_compute = Button(ax_compute, "Compute Homography")
        self.button_save = Button(ax_save, "Save & Exit")

        # Disable buttons initially
        self.button_compute.set_active(False)
        self.button_save.set_active(False)

        # Connect event handlers
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.button_reset.on_clicked(self.on_reset)
        self.button_compute.on_clicked(self.on_compute)
        self.button_save.on_clicked(self.on_save)

        print(f"\\nStarting homography calibration for: {self.image_path}")
        print(f"Arena size: {self.arena_width_cm} x {self.arena_height_cm} cm")
        print("Click 4 corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")

        # Show the plot
        plt.show()

        return self.homography_matrix if self.completed else None


def run_interactive_calibration(
    image_path: Path,
    output_path: Path,
    arena_width_cm: float = 200.0,
    arena_height_cm: float = 150.0,
) -> bool:
    """
    Run interactive homography calibration and save results.

    Args:
        image_path: Path to video frame image
        output_path: Path to save homography YAML file
        arena_width_cm: Arena width in cm (default: 200)
        arena_height_cm: Arena height in cm (default: 150)

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for interactive calibration")
        print("Install with: uv add matplotlib")
        return False

    calibrator = HomographyCalibrator(image_path, arena_width_cm, arena_height_cm)

    try:
        homography_matrix = calibrator.run_calibration()

        if homography_matrix is not None:
            # Save to YAML file
            homography_config = {
                "mapping": {
                    "type": "homography",
                    "homography_matrix": homography_matrix.tolist(),
                    "arena_bounds": [0.0, 0.0, arena_width_cm, arena_height_cm],
                },
                "calibration_info": {
                    "image_file": str(image_path),
                    "arena_width_cm": arena_width_cm,
                    "arena_height_cm": arena_height_cm,
                    "corners_pixel": calibrator.pixel_corners,
                    "corners_real_cm": calibrator.real_corners.tolist(),
                },
            }

            with open(output_path, "w") as f:
                yaml.dump(homography_config, f, default_flow_style=False, indent=2)

            print(f"\\nHomography configuration saved to: {output_path}")
            print("\\nYou can now use this file in your session configuration:")
            print(
                f"mapping:\\n  type: homography\\n  homography_matrix: !include {output_path.name}"
            )

            return True
        else:
            print("\\nCalibration cancelled or failed.")
            return False

    except Exception as e:
        print(f"Error during calibration: {e}")
        return False


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if required dependencies are available.

    Returns:
        Tuple of (all_available, missing_packages)
    """
    missing = []

    if not MATPLOTLIB_AVAILABLE:
        missing.append("matplotlib")

    if not PIL_AVAILABLE:
        missing.append("pillow")

    return len(missing) == 0, missing

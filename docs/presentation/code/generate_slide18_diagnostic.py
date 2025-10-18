"""
Generate visual for Slide 18: 9-Panel Diagnostic Screenshot

Shows a single frame from the diagnostic video with all panels visible.
Demonstrates the comprehensive visualization tools available in TrodesTrack.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.viz.video import create_diagnostic_video

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
OUTPUT_DIR = Path(__file__).parent.parent / "visuals"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette
BLUE = "#2E86AB"
ORANGE = "#F77F00"
GREEN = "#06A77D"
RED = "#D62828"
GRAY = "#6C757D"


def generate_slide18():
    """9-panel diagnostic video screenshot"""

    # Generate simulation with dropout
    np.random.seed(42)
    config = RatIMUSimConfig(
        duration_s=10.0,  # Short duration
        fs_imu=104.0,
        fs_cam=30.0,
        cam_dropout_prob=0.15,  # Some dropout to make diagnostics interesting
        use_second_led=True,
    )
    sim = simulate_rat_imu(config)

    # Run EKF
    ekf_config = EKFConfig()
    result = extended_kalman_filter(
        ekf_config,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        sim["mask_cam"],
    )

    # Create diagnostic video but capture the animation object
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "temp.mp4"

        print("  Creating diagnostic video animation...")
        _, anim, fig = create_diagnostic_video(
            sim,
            tmp_video,
            filter_results=result,
            fps=30,
            speedup=1.0,
            time_window_s=2.0,
            trail_length_s=1.5,
            dpi=150,
            return_animation=True,
        )

        # Extract a frame from the middle of the simulation
        # (when things are interesting - after some dropout has occurred)
        frame_idx = len(sim["t_cam_exp"]) // 2

        print(f"  Extracting frame {frame_idx} of {len(sim['t_cam_exp'])}...")

        # Update the animation to this frame
        # The animation update function is stored in anim
        # We need to call it with the frame index
        anim._draw_frame(frame_idx)

        # Save the current figure
        output_path = OUTPUT_DIR / "slide18_diagnostic_panel.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"✓ Saved: {output_path}")

        # Close to free memory
        plt.close(fig)

    print(f"  Frame extracted at t={sim['t_cam_exp'][frame_idx]:.2f}s")


if __name__ == "__main__":
    print("Generating Slide 18: 9-Panel Diagnostic Screenshot...")
    print()
    generate_slide18()
    print()
    print("✅ Slide 18 visual generated!")
    print(f"   Output: {OUTPUT_DIR / 'slide18_diagnostic_panel.png'}")

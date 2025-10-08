"""Generate diagnostic video from simulation.

This script demonstrates the video visualization API for debugging
simulations and (future) filter predictions.

Usage:
    uv run python examples/03_video_diagnostic.py

Output:
    Saves MP4 video to current directory showing animated diagnostic view.
"""

from __future__ import annotations

import numpy as np

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.viz.video import create_diagnostic_video


def main() -> None:
    """Generate diagnostic video from simulation."""
    print("=" * 70)
    print("Diagnostic Video Generation Example")
    print("=" * 70)

    # Generate simulation with vision robustness features
    print("\nRunning simulation (15s with LED swaps and occlusions)...")
    config = RatIMUSimConfig(
        duration_s=15.0,
        fs_imu=200.0,
        fs_cam=30.0,
        use_second_led=True,
        led1_offset_body=np.array([0.025, 0.0]),
        led2_offset_body=np.array([-0.025, 0.0]),
        led_swap_prob=0.1,  # 10% swap rate
        cam_dropout_prob=0.20,  # 20% dropout
        cam_dropout_correlation=0.75,  # Correlated dropouts
        use_confidence=True,
        confidence_base=0.90,
    )

    sim = simulate_rat_imu(config, seed=42)

    print(f"  Generated {len(sim['t_imu'])} IMU samples")
    print(f"  Camera: {len(sim['t_cam_exp'])} frames")
    print(
        f"  Valid: {sim['mask_cam'].sum()}/{len(sim['mask_cam'])} "
        f"({100*sim['mask_cam'].mean():.1f}%)"
    )

    # Generate video at realtime speed
    print("\nGenerating diagnostic video...")
    output_path = create_diagnostic_video(
        sim,
        output_path="diagnostic_sim.mp4",
        fps=30,
        speedup=1.0,  # Realtime
        time_window_s=2.0,  # 2s IMU window
        trail_length_s=1.5,  # 1.5s trail
    )

    print("\n" + "=" * 70)
    print(f"✓ Video saved: {output_path}")
    print(f"  Duration: {config.duration_s:.1f}s")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()

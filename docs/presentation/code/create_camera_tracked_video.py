"""Create video with camera-tracked LED positions overlaid.

This script overlays the raw camera tracking positions (from position parquet file)
on the trimmed video, showing just the LED markers without any filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd


def get_video_info(video_path: str) -> dict:
    """Get video file metadata.

    Parameters
    ----------
    video_path : str
        Path to video file

    Returns
    -------
    dict
        Dictionary with keys: 'width', 'height', 'fps', 'frame_count', 'duration_s'
        Returns empty dict if file not found
    """
    cap = cv2.VideoCapture(video_path)
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        return info
    finally:
        cap.release()


def create_camera_tracked_video(
    video_path: Path,
    position_file: Path,
    output_path: Path,
    meters_per_pixel: float = 0.0022,
    led_radius: int = 8,
    trajectory_length: int = 90,
) -> None:
    """Create video with camera-tracked positions overlaid.

    Parameters
    ----------
    video_path : Path
        Path to input video
    position_file : Path
        Path to position parquet file
    output_path : Path
        Path for output video
    meters_per_pixel : float
        Camera scale factor (default: 0.0022)
    led_radius : int
        Radius of LED circles in pixels
    trajectory_length : int
        Number of frames to show in trajectory trail
    """
    print("=" * 80)
    print("Creating Camera-Tracked Video")
    print("=" * 80)

    # Load position data
    print(f"\nLoading position data from {position_file.name}...")
    pos_df = pd.read_parquet(position_file)
    print(f"  Loaded {len(pos_df):,} frames")

    # Convert LED positions to pixels
    led1_pixels = pos_df[["xloc", "yloc"]].values
    led2_pixels = pos_df[["xloc2", "yloc2"]].values

    print(
        f"  LED1 range: x=[{led1_pixels[:, 0].min():.1f}, {led1_pixels[:, 0].max():.1f}], "
        f"y=[{led1_pixels[:, 1].min():.1f}, {led1_pixels[:, 1].max():.1f}]"
    )
    print(
        f"  LED2 range: x=[{led2_pixels[:, 0].min():.1f}, {led2_pixels[:, 0].max():.1f}], "
        f"y=[{led2_pixels[:, 1].min():.1f}, {led2_pixels[:, 1].max():.1f}]"
    )

    # Get video info
    print(f"\nOpening video {video_path.name}...")
    video_info = get_video_info(str(video_path))
    if not video_info:
        raise RuntimeError("Could not read video info")

    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Frames: {video_info['frame_count']:,}")
    print(f"  Duration: {video_info['duration_s']:.1f}s")

    # Align video frames with position data
    video_frame_inds = pos_df["video_frame_ind"].values
    n_frames = min(len(video_frame_inds), video_info["frame_count"])
    print(f"\n  Processing {n_frames:,} frames...")

    # Open input and output videos
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        video_info["fps"],
        (video_info["width"], video_info["height"]),
    )

    try:
        for i in range(n_frames):
            # Read video frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_frame_inds[i]))
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {i}")
                continue

            # Draw trajectory (last N frames)
            start_idx = max(0, i - trajectory_length)
            for j in range(start_idx, i):
                # LED1 trajectory (red)
                pt1 = (int(led1_pixels[j, 0]), int(led1_pixels[j, 1]))
                pt2 = (int(led1_pixels[j + 1, 0]), int(led1_pixels[j + 1, 1]))
                alpha = (j - start_idx) / trajectory_length  # Fade older points
                color = (0, 0, int(255 * (0.3 + 0.7 * alpha)))  # Red in BGR
                cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

                # LED2 trajectory (yellow)
                pt1 = (int(led2_pixels[j, 0]), int(led2_pixels[j, 1]))
                pt2 = (int(led2_pixels[j + 1, 0]), int(led2_pixels[j + 1, 1]))
                color = (
                    0,
                    int(255 * (0.3 + 0.7 * alpha)),
                    int(255 * (0.3 + 0.7 * alpha)),
                )  # Yellow in BGR
                cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

            # Draw current LED positions
            led1_center = (int(led1_pixels[i, 0]), int(led1_pixels[i, 1]))
            led2_center = (int(led2_pixels[i, 0]), int(led2_pixels[i, 1]))

            # LED1 (red circle)
            cv2.circle(frame, led1_center, led_radius, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, led1_center, led_radius + 2, (255, 255, 255), 2, cv2.LINE_AA)

            # LED2 (yellow circle)
            cv2.circle(frame, led2_center, led_radius, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, led2_center, led_radius + 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Add timestamp
            timestamp = pos_df.index[i]
            time_s = (timestamp - pos_df.index[0]) / 1e9  # Convert ns to seconds
            cv2.putText(
                frame,
                f"t = {time_s:.2f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Write frame
            out.write(frame)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1:,} / {n_frames:,} frames ({100*(i+1)/n_frames:.1f}%)")

    finally:
        cap.release()
        out.release()

    print(f"\n✓ Video created: {output_path}")
    print(f"  Output: {output_path.stat().st_size / 1024**2:.1f} MB")
    print("=" * 80)


def main():
    """Create camera-tracked video for trimmed session."""
    # Paths - docs/presentation/code is the script location
    script_dir = Path(__file__).parent  # docs/presentation/code
    video_dir = script_dir.parent / "videos"  # docs/presentation/videos
    data_dir = script_dir.parent.parent.parent / "data"  # trodestrack/data

    video_path = video_dir / "20220324_arthur_02_r1_trimmed.mp4"
    position_file = data_dir / "arthur20220324_position_info.parquet"
    output_path = video_dir / "20220324_arthur_02_r1_trimmed_tracked.mp4"

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    if not position_file.exists():
        print(f"Error: Position file not found: {position_file}")
        return 1

    create_camera_tracked_video(
        video_path=video_path,
        position_file=position_file,
        output_path=output_path,
        meters_per_pixel=0.0022,
        led_radius=8,
        trajectory_length=90,  # ~3 seconds at 30fps
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

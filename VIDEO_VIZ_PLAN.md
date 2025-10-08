# Diagnostic Video Visualization Plan

## Overview

Create animated video diagnostics showing:

1. **Current:** Ground truth + simulation components (LED detections, IMU, dropouts)
2. **Future:** Filter predictions overlaid with ground truth (when filter implemented)

**Purpose:** Real-time visual debugging of simulation/filter behavior for development and tuning.

---

## Video Layout Design

### Option A: Multi-Panel Grid (Recommended for Diagnostics)

```
┌─────────────────────────────────────────────────────────┐
│  Title: "Example 5: Vision Robustness | t=12.45s"      │
├──────────────────────┬──────────────────────────────────┤
│                      │  IMU Measurements (recent 2s)    │
│   Arena View         │  ┌─ Gyro (rad/s)                 │
│   (2D top-down)      │  ├─ Accel X (m/s²)               │
│                      │  └─ Accel Y (m/s²)               │
│  • Rat (orientation) │                                  │
│  • LED1 (blue)       ├──────────────────────────────────┤
│  • LED2 (orange)     │  Camera Status & Confidence      │
│  • Trail (fading)    │  ┌─ LED1: [████░░] 0.85         │
│  • Dropouts (red X)  │  └─ LED2: [██████] 0.92         │
│                      │                                  │
│  [Future: filter     │  Swap indicator: ⚠ SWAPPED      │
│   prediction overlay]│                                  │
├──────────────────────┴──────────────────────────────────┤
│  State Variables (time series, trailing 5s window)      │
│  ┌─ Position (x, y)                                     │
│  ├─ Velocity (vx, vy, |v|)                              │
│  ├─ Heading (θ)                                         │
│  └─ [Future: filter estimate ± uncertainty]             │
└──────────────────────────────────────────────────────────┘
```

**Panels:**

1. **Arena View (main):** 2D bird's-eye view with rat, LEDs, trail
2. **IMU Panel:** Recent accelerometer/gyro traces
3. **Camera Panel:** Detection status, confidence bars, swap warnings
4. **Time Series Panel:** State evolution with scrolling window

### Option B: Single-Panel Focus (Simpler, Faster Rendering)

```
┌─────────────────────────────────────────────────────────┐
│         Arena View with Overlay Info                    │
│                                                          │
│    [Rat with orientation arrow]                         │
│    [LED positions with confidence halos]                │
│    [Fading trajectory trail]                            │
│                                                          │
│  Overlay HUD:                                           │
│  ┌─ t=12.45s │ v=0.42 m/s │ θ=127°                     │
│  ├─ LED1: ✓ (0.85) │ LED2: ✗ (dropout)                │
│  └─ IMU: ω=0.12 rad/s │ a=[0.3, -0.1] m/s²            │
│                                                          │
│  [Progress bar at bottom]                               │
└──────────────────────────────────────────────────────────┘
```

**Recommendation:** Start with **Option B** for simplicity, extend to **Option A** when filter is implemented.

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**Goal:** Basic video generation from simulation data

#### 1.1 Create Module Structure

```
src/trodestrack/viz/
├── __init__.py
├── video.py          # Main video generation API
├── components.py     # Reusable plot components (rat, LEDs, trails)
├── styles.py         # Color schemes, fonts (reuse from examples)
└── utils.py          # Frame interpolation, time sync
```

#### 1.2 Core API Design

```python
from trodestrack.viz.video import create_diagnostic_video

# Simple usage
create_diagnostic_video(
    sim_data,
    output_path="debug_sim.mp4",
    fps=30,
    speedup=1.0,  # Realtime
    layout="single",  # or "multi"
    show_components=["rat", "leds", "trail", "hud"],
)

# Advanced usage with filter data (future)
create_diagnostic_video(
    sim_data,
    filter_results=filter_data,  # Optional
    output_path="debug_filter.mp4",
    fps=30,
    speedup=2.0,  # 2x speedup
    layout="multi",
    show_components=["rat", "leds", "trail", "imu", "camera", "timeseries", "filter"],
    highlight_events=["swaps", "long_dropouts"],  # Auto-detect and mark
)
```

#### 1.3 Dependencies

- **matplotlib** (already used) - plotting backend
- **matplotlib.animation** - FuncAnimation for video generation
- **ffmpeg** (system dependency) - video encoding
  - Fallback to pillow for GIF if ffmpeg unavailable

#### 1.4 Key Functions

```python
# video.py
def create_diagnostic_video(
    sim_data: SimOut,
    output_path: str | Path,
    filter_results: FilterOut | None = None,
    fps: int = 30,
    speedup: float = 1.0,
    layout: Literal["single", "multi"] = "single",
    show_components: list[str] = None,
    time_window_s: float = 5.0,  # For scrolling plots
    trail_length_s: float = 2.0,  # Fading trail
    dpi: int = 100,
    codec: str = "h264",
) -> Path:
    """Generate diagnostic video from simulation/filter data."""
    ...

def _render_frame(
    frame_idx: int,
    t: float,
    sim_data: SimOut,
    filter_results: FilterOut | None,
    artists: dict[str, Any],
) -> list[Artist]:
    """Update artists for a single frame."""
    ...
```

---

### Phase 2: Arena View Components (Week 1)

**Goal:** Visualize rat, LEDs, trail in 2D arena

#### 2.1 Rat Representation

```python
# components.py
class RatArtist:
    """Visualize rat position, orientation, and scale."""

    def __init__(self, ax: Axes, config: RatIMUSimConfig):
        # Rat body: circle + orientation wedge
        self.body = Circle((0, 0), radius=0.03, color='gray', alpha=0.7)
        self.nose = Wedge((0, 0), r=0.03, theta1=-30, theta2=30, color='black')
        self.ax = ax

    def update(self, x: float, y: float, theta: float):
        """Update rat position and orientation."""
        self.body.center = (x, y)
        self.nose.center = (x, y)
        self.nose.theta1 = np.rad2deg(theta) - 30
        self.nose.theta2 = np.rad2deg(theta) + 30
```

#### 2.2 LED Representation

```python
class LEDArtist:
    """Visualize LED detections with confidence."""

    def __init__(self, ax: Axes, led_id: int, color: str):
        # LED marker + confidence halo
        self.marker = ax.plot([], [], 'o', color=color, markersize=8,
                              label=f'LED{led_id}', zorder=10)[0]
        self.halo = Circle((0, 0), radius=0.01, color=color,
                          alpha=0.3, zorder=9)
        self.dropout_marker = ax.plot([], [], 'x', color='red',
                                      markersize=12, linewidth=2)[0]

    def update(self, x: float, y: float, visible: bool, confidence: float):
        """Update LED position and visibility."""
        if visible:
            self.marker.set_data([x], [y])
            self.halo.center = (x, y)
            self.halo.set_alpha(confidence * 0.5)  # Scale by confidence
            self.dropout_marker.set_data([], [])
        else:
            self.marker.set_data([], [])
            self.halo.set_alpha(0)
            # Show last known position with red X
            if hasattr(self, 'last_x'):
                self.dropout_marker.set_data([self.last_x], [self.last_y])

        if visible:
            self.last_x, self.last_y = x, y
```

#### 2.3 Trajectory Trail

```python
class TrailArtist:
    """Fading trajectory trail."""

    def __init__(self, ax: Axes, trail_length_s: float, fps: int):
        self.trail_frames = int(trail_length_s * fps)
        self.positions = deque(maxlen=self.trail_frames)
        # Use LineCollection for efficient fading
        self.lines = LineCollection([], linewidths=2,
                                    colors='blue', alpha=0.5)
        ax.add_collection(self.lines)

    def update(self, x: float, y: float):
        """Add new position and update fading trail."""
        self.positions.append([x, y])
        if len(self.positions) > 1:
            segments = [self.positions[i:i+2]
                       for i in range(len(self.positions)-1)]
            # Fade alpha from 0 (old) to 0.5 (new)
            alphas = np.linspace(0, 0.5, len(segments))
            self.lines.set_segments(segments)
            self.lines.set_alpha(alphas)
```

---

### Phase 3: HUD and Overlays (Week 1-2)

#### 3.1 Time and State HUD

```python
class HUDArtist:
    """Heads-up display for current state."""

    def __init__(self, ax: Axes):
        # Text overlays (top-left corner)
        self.time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                va='top', fontsize=10, family='monospace',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        self.state_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                                 va='top', fontsize=8, family='monospace')

    def update(self, t: float, state: dict):
        """Update HUD text."""
        time_str = f"t = {t:.2f}s"
        state_str = (
            f"v = {state['speed']:.2f} m/s\n"
            f"θ = {np.rad2deg(state['theta']):.1f}°\n"
            f"LED1: {'✓' if state['led1_visible'] else '✗'} ({state['conf1']:.2f})\n"
            f"LED2: {'✓' if state['led2_visible'] else '✗'} ({state['conf2']:.2f})"
        )
        self.time_text.set_text(time_str)
        self.state_text.set_text(state_str)
```

#### 3.2 Event Markers

```python
class EventMarkerArtist:
    """Highlight special events (swaps, long dropouts)."""

    def __init__(self, ax: Axes):
        self.swap_banner = ax.text(0.5, 0.95, '', transform=ax.transAxes,
                                  ha='center', va='top', fontsize=12,
                                  color='red', fontweight='bold',
                                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        self.swap_timer = 0  # Frames to show banner

    def update(self, events: dict):
        """Show temporary event indicators."""
        if events.get('led_swap', False):
            self.swap_banner.set_text('⚠ LED SWAP DETECTED')
            self.swap_timer = 60  # Show for 2 seconds at 30fps

        if self.swap_timer > 0:
            self.swap_banner.set_alpha(0.8 * (self.swap_timer / 60))
            self.swap_timer -= 1
        else:
            self.swap_banner.set_text('')
```

---

### Phase 4: Multi-Panel Layout (Week 2)

#### 4.1 IMU Time Series

```python
class IMUPanelArtist:
    """Show recent IMU measurements."""

    def __init__(self, fig: Figure, gs: GridSpec, window_s: float, fps: int):
        self.ax_gyro = fig.add_subplot(gs[0, 1])
        self.ax_accel_x = fig.add_subplot(gs[1, 1], sharex=self.ax_gyro)
        self.ax_accel_y = fig.add_subplot(gs[2, 1], sharex=self.ax_gyro)

        self.window_frames = int(window_s * fps)
        self.time_buffer = deque(maxlen=self.window_frames)
        self.gyro_buffer = deque(maxlen=self.window_frames)
        self.accel_x_buffer = deque(maxlen=self.window_frames)
        self.accel_y_buffer = deque(maxlen=self.window_frames)

        # Initialize lines
        self.gyro_line, = self.ax_gyro.plot([], [], 'b-', linewidth=1)
        self.accel_x_line, = self.ax_accel_x.plot([], [], 'r-', linewidth=1)
        self.accel_y_line, = self.ax_accel_y.plot([], [], 'g-', linewidth=1)

    def update(self, t: float, imu_data: dict):
        """Add new IMU sample and scroll window."""
        self.time_buffer.append(t)
        self.gyro_buffer.append(imu_data['gyro'])
        self.accel_x_buffer.append(imu_data['accel_x'])
        self.accel_y_buffer.append(imu_data['accel_y'])

        # Update lines
        self.gyro_line.set_data(self.time_buffer, self.gyro_buffer)
        self.accel_x_line.set_data(self.time_buffer, self.accel_x_buffer)
        self.accel_y_line.set_data(self.time_buffer, self.accel_y_buffer)

        # Auto-scale x-axis to show window
        if len(self.time_buffer) > 0:
            self.ax_gyro.set_xlim(self.time_buffer[0], self.time_buffer[-1])
```

#### 4.2 Camera Status Panel

```python
class CameraPanelArtist:
    """Show camera detection status and confidence."""

    def __init__(self, fig: Figure, gs: GridSpec):
        self.ax = fig.add_subplot(gs[3, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 2)
        self.ax.set_yticks([0.5, 1.5])
        self.ax.set_yticklabels(['LED1', 'LED2'])

        # Confidence bars
        self.led1_bar = self.ax.barh(0.5, 0, height=0.4, color='blue', alpha=0.6)
        self.led2_bar = self.ax.barh(1.5, 0, height=0.4, color='orange', alpha=0.6)

    def update(self, led1_visible: bool, conf1: float,
               led2_visible: bool, conf2: float):
        """Update confidence bars."""
        self.led1_bar[0].set_width(conf1 if led1_visible else 0)
        self.led2_bar[0].set_width(conf2 if led2_visible else 0)
```

---

### Phase 5: Filter Overlay (Future - After Filter Implementation)

#### 5.1 Filter Prediction Overlay

```python
class FilterArtist:
    """Overlay filter predictions on arena view."""

    def __init__(self, ax: Axes):
        # Predicted position with uncertainty ellipse
        self.pred_marker, = ax.plot([], [], 'o', color='green',
                                   markersize=6, fillstyle='none',
                                   linewidth=2, label='Filter')
        self.uncertainty_ellipse = Ellipse((0, 0), 0, 0,
                                          color='green', alpha=0.2)
        ax.add_patch(self.uncertainty_ellipse)

    def update(self, x_pred: float, y_pred: float,
               P: np.ndarray, chi2_95: float = 5.991):
        """Update filter prediction and uncertainty."""
        self.pred_marker.set_data([x_pred], [y_pred])

        # Covariance ellipse (95% confidence)
        eigenvalues, eigenvectors = np.linalg.eig(P[:2, :2])
        angle = np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * np.sqrt(chi2_95 * eigenvalues[0])
        height = 2 * np.sqrt(chi2_95 * eigenvalues[1])

        self.uncertainty_ellipse.center = (x_pred, y_pred)
        self.uncertainty_ellipse.width = width
        self.uncertainty_ellipse.height = height
        self.uncertainty_ellipse.angle = angle
```

#### 5.2 Innovation Residuals

```python
class ResidualPanelArtist:
    """Show measurement innovations (residuals)."""

    def __init__(self, fig: Figure, gs: GridSpec, window_s: float, fps: int):
        self.ax = fig.add_subplot(gs[4, :])
        self.window_frames = int(window_s * fps)

        self.time_buffer = deque(maxlen=self.window_frames)
        self.residual_x_buffer = deque(maxlen=self.window_frames)
        self.residual_y_buffer = deque(maxlen=self.window_frames)

        self.line_x, = self.ax.plot([], [], 'b-', label='x residual')
        self.line_y, = self.ax.plot([], [], 'r-', label='y residual')

    def update(self, t: float, residuals: dict):
        """Update residual time series."""
        self.time_buffer.append(t)
        self.residual_x_buffer.append(residuals['x'])
        self.residual_y_buffer.append(residuals['y'])

        self.line_x.set_data(self.time_buffer, self.residual_x_buffer)
        self.line_y.set_data(self.time_buffer, self.residual_y_buffer)
```

---

## Technical Details

### Frame Interpolation

**Problem:** Camera at 30 Hz, IMU at 200+ Hz, video at 30 fps
**Solution:** Interpolate all data to common video frame times

```python
def prepare_video_data(
    sim_data: SimOut,
    fps: int,
    speedup: float = 1.0
) -> dict:
    """Interpolate all data to video frame times."""
    # Determine video timeline
    t_start = 0
    t_end = sim_data['t_imu'][-1]
    duration_video = (t_end - t_start) / speedup
    n_frames = int(duration_video * fps)
    t_video = np.linspace(t_start, t_end, n_frames)

    # Interpolate IMU (linear for gyro/accel)
    imu_interp = np.column_stack([
        np.interp(t_video, sim_data['t_imu'], sim_data['U_imu'][:, i])
        for i in range(3)
    ])

    # Interpolate state (wrap angles)
    from trodestrack.sim.utils import interp_angle
    X_interp = np.column_stack([
        np.interp(t_video, sim_data['t_imu'], sim_data['X_truth'][:, i])
        for i in range(4)
    ] + [
        interp_angle(t_video, sim_data['t_imu'], sim_data['X_truth'][:, 4])
    ])

    # Camera: nearest-neighbor for discrete events (dropouts, swaps)
    cam_idx = np.searchsorted(sim_data['t_cam_exp'], t_video, side='right') - 1
    cam_idx = np.clip(cam_idx, 0, len(sim_data['t_cam_exp']) - 1)

    return {
        't_video': t_video,
        'X_truth': X_interp,
        'U_imu': imu_interp,
        'cam_idx': cam_idx,  # Index into camera arrays
        'fps': fps,
        'n_frames': n_frames,
    }
```

### Video Encoding

```python
def encode_video(
    fig: Figure,
    update_func: Callable,
    n_frames: int,
    fps: int,
    output_path: Path,
    codec: str = 'h264',
    dpi: int = 100,
) -> Path:
    """Encode matplotlib animation to video file."""
    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(
        fig,
        update_func,
        frames=n_frames,
        interval=1000/fps,  # ms per frame
        blit=True,  # Faster rendering
        repeat=False,
    )

    # Save with ffmpeg writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, codec=codec, bitrate=2000)

    anim.save(str(output_path), writer=writer, dpi=dpi)

    return output_path
```

### Performance Optimization

**Rendering Speed:**

- Use `blit=True` in FuncAnimation (only redraw changed artists)
- Pre-allocate all artists, update data not recreate
- Limit trail/buffer lengths to avoid memory bloat
- Use LineCollection instead of individual plot() calls
- Consider saving frames as PNG sequence then ffmpeg encode (parallel)

**Target Performance:**

- 30s simulation at 30 fps = 900 frames
- Goal: <5 minutes rendering time on CPU
- Speedup options: 2x, 5x, 10x for long sessions

---

## Usage Examples

### Example 1: Simple Diagnostic Video

```python
from trodestrack.sim.rat_imu import simulate_rat_imu, RatIMUSimConfig
from trodestrack.viz.video import create_diagnostic_video

# Run simulation
config = RatIMUSimConfig(duration_s=10.0, led_swap_prob=0.15)
sim = simulate_rat_imu(config, seed=42)

# Generate video
create_diagnostic_video(
    sim,
    output_path="diagnostics/sim_basic.mp4",
    fps=30,
    speedup=1.0,
    layout="single",
)
```

### Example 2: Multi-Panel with Filter (Future)

```python
from trodestrack.runtime.offline import smooth_session

# Run simulation and filter
sim = simulate_rat_imu(config, seed=42)
filter_result = smooth_session(sim)  # Future filter API

# Generate comprehensive diagnostic video
create_diagnostic_video(
    sim,
    filter_results=filter_result,
    output_path="diagnostics/sim_filter_comparison.mp4",
    fps=30,
    speedup=2.0,  # 2x speed
    layout="multi",
    show_components=["rat", "leds", "trail", "filter", "residuals"],
    highlight_events=["swaps", "long_dropouts"],
)
```

### Example 3: Batch Video Generation

```python
from pathlib import Path

# Generate videos for all examples
examples = [
    ("basic", RatIMUSimConfig(duration_s=10)),
    ("swaps", RatIMUSimConfig(duration_s=20, led_swap_prob=0.2)),
    ("occlusions", RatIMUSimConfig(duration_s=30, cam_dropout_prob=0.3,
                                   cam_dropout_correlation=0.9)),
]

for name, config in examples:
    sim = simulate_rat_imu(config, seed=42)
    create_diagnostic_video(
        sim,
        output_path=f"diagnostics/{name}.mp4",
        fps=30,
    )
```

---

## Testing Strategy

### Unit Tests

```python
# tests/viz/test_video_components.py
def test_rat_artist_updates():
    """Test rat artist position/orientation updates."""
    ...

def test_led_artist_dropout_handling():
    """Test LED visibility toggling."""
    ...

def test_trail_artist_fading():
    """Test trail length and alpha fading."""
    ...
```

### Integration Tests

```python
# tests/viz/test_video_integration.py
def test_video_generation_single_layout():
    """Test full video generation with single panel layout."""
    sim = simulate_rat_imu(RatIMUSimConfig(duration_s=2.0), seed=42)
    output = create_diagnostic_video(sim, "test_output.mp4", fps=30)
    assert output.exists()
    assert output.stat().st_size > 0

def test_video_frame_count():
    """Test correct number of frames generated."""
    sim = simulate_rat_imu(RatIMUSimConfig(duration_s=5.0), seed=42)
    output = create_diagnostic_video(sim, "test_5s.mp4", fps=30, speedup=1.0)
    # Use ffprobe to verify frame count
    assert get_frame_count(output) == 150  # 5s * 30fps
```

### Visual Regression Tests

- Generate reference videos with known-good version
- Compare frame-by-frame pixel differences (allow small tolerance)
- Use perceptual hash for fuzzy matching

---

## Future Enhancements

### Phase 6: Interactive Video (Optional)

- Use `ipywidgets` + `matplotlib` for Jupyter notebook playback
- Slider to scrub through time
- Click to pause and inspect state
- Toggle visibility of components

### Phase 7: Real-Time Streaming (Optional)

- Stream video during long offline processing
- Update video as filter runs (show progress)
- Useful for 30+ minute sessions

### Phase 8: 3D Visualization (Future)

- When 3D tracking implemented (roll/pitch/yaw)
- Use `vispy` or `plotly` for 3D arena view
- Show full orientation quaternion

### Phase 9: Export Formats

- MP4 (default, h264 codec)
- GIF (for quick previews, smaller file size)
- WebM (for web embedding)
- PNG sequence (for frame-by-frame analysis)
- Interactive HTML (plotly animations)

---

## Acceptance Criteria

### Phase 1-3 (Simulation Only)

- ✅ Single-panel video shows rat, LEDs, trail, HUD
- ✅ Dropout events visible (red X markers)
- ✅ LED swaps highlighted with banner
- ✅ Video duration matches simulation * speedup
- ✅ Renders at ≥10 fps for 30s simulation
- ✅ File size reasonable (<50 MB for 30s @ 30fps)

### Phase 4 (Multi-Panel)

- ✅ All panels render correctly
- ✅ Time series scroll smoothly
- ✅ IMU/camera data synced to arena view
- ✅ Layout scales to different figure sizes

### Phase 5 (Filter Overlay)

- ✅ Filter prediction overlays on ground truth
- ✅ Uncertainty ellipse scales with covariance
- ✅ Innovation residuals visible in time series
- ✅ Color-coded for easy truth vs prediction distinction

---

## Development Timeline

| Phase | Component | Effort | Dependencies |
|-------|-----------|--------|--------------|
| 1 | Core infrastructure | 1 day | None |
| 2 | Arena view components | 1 day | Phase 1 |
| 3 | HUD and overlays | 1 day | Phase 2 |
| 4 | Multi-panel layout | 1 day | Phase 3 |
| 5 | Filter overlay | 2 days | Filter implementation |

**Total: 6 days** (Phases 1-4 can be done now, Phase 5 waits for filter)

---

## Open Questions

1. **Video codec preference?**
   - h264 (widely supported, good compression)
   - hevc (better compression, less support)
   - vp9/webm (web-friendly)
   - **Recommendation:** h264 default, make configurable

2. **Color scheme for filter vs truth?**
   - Truth: blue/orange (current LED colors)
   - Filter: green (prediction), red (rejected measurements)
   - **Recommendation:** Maintain consistency with static plots

3. **Event detection sensitivity?**
   - LED swaps: detect via spacing deviation > threshold
   - Long dropouts: consecutive frames > N
   - **Recommendation:** Make thresholds configurable

4. **Frame interpolation method?**
   - Linear (fast, simple)
   - Spline (smoother, more accurate)
   - **Recommendation:** Linear for position/velocity, wrap-aware for angles

5. **Rendering backend?**
   - matplotlib (consistent with current plots)
   - opencv (faster, lower-level)
   - **Recommendation:** matplotlib for consistency, opencv if performance issues

---

## Summary

This plan provides:

- ✅ Incremental implementation path (5 phases)
- ✅ Clear API design for current (simulation) and future (filter) use cases
- ✅ Reusable artist components for easy extension
- ✅ Performance optimizations (blit, pre-allocation)
- ✅ Comprehensive testing strategy
- ✅ Realistic timeline (6 days total, 4 days for simulation-only)

**Next Step:** Review plan and decide on layout preference (single vs multi-panel for Phase 1).

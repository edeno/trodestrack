Continue building the TrodesTrack presentation for neuroscientists.

**CONTEXT:**
This is a 30-minute, 43-slide presentation explaining TrodesTrack (sensor-fusion 2D rat tracking package) to neuroscientists. Previous work is in docs/presentation/.

**READ THESE FILES FIRST:**

1. `.claude/docs/PRD.md` - Package requirements and technical specs (CRITICAL: accuracy targets, state vector, IMU specs)
2. `.claude/docs/CLAUDE.md` - Development commands and workflow
3. `docs/presentation/PRESENTATION_OUTLINE.md` - Full 43-slide structure with pedagogical design
4. `docs/presentation/PROGRESS_REPORT.md` - Status update (what's done, what's remaining)
5. `docs/presentation/TASKS.md` - Step-by-step checklist with time estimates
6. `docs/presentation/code/generate_slide05abc_imu_physics.py` - Working code template (458 lines)

**ALREADY COMPLETE:**

- ✅ 43-slide outline with pedagogical structure (concrete→abstract learning path)
- ✅ 3 IMU physics visuals (slides 5A, 5B, 5C) saved to docs/presentation/visuals/
- ✅ Working code template using TrodesTrack simulator
- ✅ Folder structure in docs/presentation/

**YOUR TASK:**
Generate remaining visuals following TASKS.md Phase 1 checklist. Start with easiest:

1. **Slide 3**: Trajectory comparison (ground truth vs noisy vision-only with gaps) - 30 min
2. **Slide 21**: NEES histogram vs χ² theoretical distribution - 30 min
3. **Slide 18**: 9-panel diagnostic video screenshot - 30 min
4. **Slide 14**: Uncertainty evolution (covariance ellipses growing/shrinking) - 1 hour
5. **Slide 16**: Smoother comparison (vision-only vs EKF vs RTS on dropout) - 1-2 hours
6. **Slide 2**: Failure modes grid (occlusion/reflection/blur/dim) - 1-2 hours
7. **Slide 12**: IMU integration visualization between camera frames - 1-2 hours
8. **Slide 8**: Before/after video (10s split-screen: vision-only vs fusion) - 2-3 hours

**CRITICAL TECHNICAL DETAILS** (from PRD):

- **Accuracy targets**: Position RMSE ≤2cm, Velocity RMSE ≤10cm/s, Heading ≤7°, Drift @5s ≤3.5m
- **State vector**: [x, y, vₓ, vᵧ, θ, b_gz, b_ax, b_ay] (8D for "2d_full" layout)
- **IMU specs**: SpikeGadgets 104 Hz, gyro noise 0.01°/s/√Hz, accel noise 0.2mg/√Hz
- **SimOut dict keys**: `t_cam_exp` (not t_cam), `X_truth` (capital X, not x_truth)
- **EKF signature**: `extended_kalman_filter(config, t_imu, U_imu, t_cam_exp, Z_cam_led1, Z_cam_led2, mask_cam)`
- **Bias params**: `gyro_bias_rw_density`, `accel_bias_rw_density` (not `*_std`)

**CODE PATTERN** (from working example):

```python
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
import numpy as np
import matplotlib.pyplot as plt

# Color palette
BLUE, ORANGE, GREEN, RED, GRAY = "#2E86AB", "#F77F00", "#06A77D", "#D62828", "#6C757D"

# 1. Simulate
config = RatIMUSimConfig(duration_s=30.0, cam_dropout_prob=0.1, use_second_led=True)
sim = simulate_rat_imu(config)

# 2. Run filter
result = extended_kalman_filter(
    EKFConfig(), sim["t_imu"], sim["U_imu"], sim["t_cam_exp"],
    sim["Z_cam_led1"], sim["Z_cam_led2"], sim["mask_cam"]
)

# 3. Extract results
layout = get_layout("2d_full")
pos_est = result.filtered_means[:, layout.pos_idx]

# 4. Downsample truth to camera rate
cam_idx = np.searchsorted(sim["t_imu"], sim["t_cam_exp"])
cam_idx = np.clip(cam_idx, 0, len(sim["t_imu"])-1)
pos_truth = sim["X_truth"][cam_idx, :2]

# 5. Plot
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(pos_truth[:, 0], pos_truth[:, 1], "--", color=BLUE, label="Truth")
ax.plot(pos_est[:, 0], pos_est[:, 1], color=GREEN, label="EKF")
ax.set_aspect("equal")
plt.savefig("docs/presentation/visuals/slideXX_name.png", dpi=150, bbox_inches="tight")
```

**DESIGN SPECS** (from OUTLINE):

- Color palette: Blue=#2E86AB (trust), Orange=#F77F00 (energy), Green=#06A77D (success), Red=#D62828 (error)
- Resolution: 150+ DPI
- Aspect ratio: 16:9
- Fonts: 24+ pt body, 32+ pt headlines
- Style: Visual-first, minimal text, high contrast, colorblind-safe

**WORKFLOW:**

1. Read TASKS.md to pick next visual
2. Check PRESENTATION_OUTLINE.md for that slide's detailed spec
3. Look at generate_slide05abc_imu_physics.py for code patterns
4. Generate the visual, test with: `uv run python docs/presentation/code/generate_slideXX.py`
5. Update TASKS.md (check off completed item)
6. Repeat for next visual

**REFERENCE PRD METRICS** (cite these in visuals):

- Position RMSE target: ≤0.02 m (slide 21, 22)
- Velocity RMSE target: ≤0.10 m/s (slide 21, 22)
- Heading error target: ≤7° (slide 21, 22)
- Dropout drift target: ≤3.5 m @ 5s (slide 8, 16)
- NEES target: ≈8 for 8D state (slide 14, 21)
- Throughput: ≥10× realtime CPU (slide 24)

**START HERE:**
Generate Slide 3 (trajectory comparison). This is the easiest - just plot ground truth vs noisy camera observations with gaps showing dropout periods. Save to `docs/presentation/visuals/slide03_trajectory_comparison.png`.

Let me know when ready or if you have questions!

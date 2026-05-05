# TrodesTrack Presentation - Claude Code Handoff Prompt

**Purpose**: This document provides everything a fresh Claude Code session needs to pick up the TrodesTrack presentation project and continue from where we left off.

---

## Copy-Paste Prompt for New Claude Code Session

```
I need you to continue building a presentation about TrodesTrack (a sensor-fusion 2D rat tracking package).

CONTEXT:
- The presentation is for neuroscientists (behavioral/systems/computational)
- It's a 30-minute talk (43 slides) explaining the package purpose, features, and how to use it
- Previous work is located in: docs/presentation/
- Read these files first:
  1. docs/presentation/PRESENTATION_OUTLINE.md (full slide structure)
  2. docs/presentation/PROGRESS_REPORT.md (what's done, what's remaining)
  3. docs/presentation/TASKS.md (step-by-step checklist)
  4. .claude/docs/PRD.md (package requirements and specs)
  5. .claude/docs/CLAUDE.md (development commands and workflow)

WHAT'S ALREADY DONE:
- ✅ Complete 43-slide outline with pedagogical design
- ✅ 3 IMU physics visuals generated (slides 5A, 5B, 5C)
- ✅ Working code template: docs/presentation/code/generate_slide05abc_imu_physics.py
- ✅ Folder structure created in docs/presentation/

YOUR TASK:
Continue generating visuals following TASKS.md Phase 1 checklist. Start with the easiest:

1. Slide 3: Trajectory comparison (ground truth vs noisy vision-only)
2. Slide 21: NEES histogram
3. Slide 18: 9-panel diagnostic screenshot
4. Slide 14: Uncertainty evolution (covariance ellipses)
5. Slide 16: Smoother comparison (3 algorithms on dropout)
6. Slide 2: Failure modes grid (occlusion, reflection, blur, dim)
7. Slide 12: IMU integration visualization
8. Slide 8: Before/after video (10s, split-screen)

IMPORTANT TECHNICAL DETAILS:
- Use the simulator: from trodestrack.sim.rat_imu import simulate_rat_imu, RatIMUSimConfig
- SimOut dict keys: t_cam_exp (not t_cam), X_truth (not x_truth)
- EKF function signature: extended_kalman_filter(config, t_imu, U_imu, t_cam_exp, Z_cam_led1, Z_cam_led2, mask_cam)
- Bias parameters: gyro_bias_rw_density, accel_bias_rw_density (not gyro_bias_std)
- Color palette: BLUE=#2E86AB, ORANGE=#F77F00, GREEN=#06A77D, RED=#D62828, GRAY=#6C757D
- Save visuals to: docs/presentation/visuals/
- Save code to: docs/presentation/code/

WORKFLOW:
1. Read the TASKS.md file
2. Pick the next visual from Phase 1
3. Look at generate_slide05abc_imu_physics.py as a template
4. Generate the visual following the spec in PRESENTATION_OUTLINE.md
5. Test the code: uv run python docs/presentation/code/generate_slideXX.py
6. Update progress in TASKS.md (check off completed items)
7. Repeat for next visual

CONSTRAINTS:
- Follow presentation best practices (visual-first, minimal text, high contrast)
- Use TrodesTrack simulator for all demonstrations (no synthetic/fake data)
- Keep code modular and well-commented
- DPI: 150 minimum for images
- Aspect ratio: 16:9 for all visuals
- Font size: 24+ pt for body text, 32+ pt for headlines

START HERE:
Please read the 5 files listed above, then generate the code for Slide 3 (trajectory comparison).
This should be the easiest - just plot ground truth vs noisy camera observations with gaps.

Let me know when you're ready to start, or if you have questions!
```

---

## Alternative Quick-Start Prompts

### If Claude needs to generate ALL remaining visuals:

```
Continue building the TrodesTrack presentation. Read:
- docs/presentation/PRESENTATION_OUTLINE.md
- docs/presentation/TASKS.md
- docs/presentation/PROGRESS_REPORT.md

Then work through TASKS.md Phase 1 (core visuals) sequentially. Use the code template at docs/presentation/code/generate_slide05abc_imu_physics.py as a reference. Save all outputs to docs/presentation/visuals/. Start with Slide 3 (easiest).
```

### If Claude needs to assemble the PowerPoint:

```
Build a PowerPoint presentation from the TrodesTrack outline. Read:
- docs/presentation/PRESENTATION_OUTLINE.md (43-slide structure)
- docs/presentation/PROGRESS_REPORT.md (current status)

Use python-pptx to programmatically build trodestrack_presentation.pptx. Import visuals from docs/presentation/visuals/. Follow the design specs: Blue/Orange/Green color scheme, 24+ pt fonts, 16:9 aspect ratio, assertion-evidence slide structure.
```

### If Claude needs to write speaker notes:

```
Write speaker notes for the TrodesTrack presentation. Read:
- docs/presentation/PRESENTATION_OUTLINE.md (full slide details)
- .claude/docs/PRD.md (technical specs for accuracy)

Create docs/presentation/speaker_notes.md with:
- 1-2 paragraphs per slide explaining what to say
- Key points to emphasize
- Transitions between slides
- Estimated talk time per slide (target: 25-28 minutes total)

Use pedagogical best practices: concrete examples before abstract concepts, analogies for complex ideas, interactive questions to maintain engagement.
```

---

## File Paths Reference

```
docs/presentation/
├── PRESENTATION_OUTLINE.md          # 43-slide detailed outline
├── PROGRESS_REPORT.md               # Status update (see file for current numbers)
├── TASKS.md                         # Step-by-step checklist
├── HANDOFF_PROMPT.md                # This file
│
├── visuals/                         # Generated images (8 PNGs shipped today)
│   ├── slide03_trajectory_comparison.png   ✅
│   ├── slide05a_accelerometer_physics.png  ✅
│   ├── slide05b_gyroscope_physics.png      ✅
│   ├── slide05c_bias_correction.png        ✅
│   ├── slide12_imu_integration.png         ✅
│   ├── slide14_uncertainty.png             ✅
│   ├── slide16_smoother_comparison.png     ✅
│   └── slide21_nees_histogram.png          ✅
│   (slide02_failure_modes.png and slide18_diagnostic_panel.png deferred;
│    those slides currently render as bullet content in build_presentation.py.)
│
├── videos/                          # 1 MP4 shipped
│   └── slide08_beforeafter.mp4      ✅
│
└── code/                            # 7 generator scripts shipped
    ├── generate_slide03_trajectory_comparison.py ✅
    ├── generate_slide05abc_imu_physics.py        ✅
    ├── generate_slide08_beforeafter_video.py     ✅
    ├── generate_slide12_imu_integration.py       ✅
    ├── generate_slide14_uncertainty.py           ✅
    ├── generate_slide16_smoother.py              ✅
    └── generate_slide21_nees.py                  ✅
    (generate_slide02_failure_modes.py and generate_slide18_diagnostic.py
     deferred — see corresponding slides in build_presentation.py.)
```

---

## Key Technical Patterns (Copy for Reference)

### Standard Simulation + EKF Workflow

```python
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
import numpy as np

# 1. Configure simulation
config = RatIMUSimConfig(
    duration_s=30.0,
    fs_imu=104.0,
    fs_cam=30.0,
    cam_dropout_prob=0.1,  # 10% dropout
    use_second_led=True,
)

# 2. Generate synthetic data
sim = simulate_rat_imu(config)

# 3. Run EKF
result = extended_kalman_filter(
    EKFConfig(),
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],      # Note: t_cam_exp, not t_cam
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)

# 4. Extract results
layout = get_layout("2d_full")
pos_est = result.filtered_means[:, layout.pos_idx]      # [N, 2]
vel_est = result.filtered_means[:, layout.vel_idx]      # [N, 2]
heading_est = result.filtered_means[:, layout.heading_idx]  # [N,]

# 5. Get ground truth (downsampled to camera rate)
cam_indices = np.searchsorted(sim["t_imu"], sim["t_cam_exp"])
cam_indices = np.clip(cam_indices, 0, len(sim["t_imu"]) - 1)
X_truth_cam = sim["X_truth"][cam_indices]  # [N_cam, 5]
pos_truth = X_truth_cam[:, :2]
vel_truth = X_truth_cam[:, 2:4]
heading_truth = X_truth_cam[:, 4]
```

### Plotting Template

```python
import matplotlib.pyplot as plt

# Color palette
BLUE = "#2E86AB"
ORANGE = "#F77F00"
GREEN = "#06A77D"
RED = "#D62828"
GRAY = "#6C757D"

fig, ax = plt.subplots(figsize=(12, 7))

# Plot ground truth
ax.plot(pos_truth[:, 0], pos_truth[:, 1],
       linewidth=2, color=BLUE, linestyle="--", label="Ground truth")

# Plot estimate
ax.plot(pos_est[:, 0], pos_est[:, 1],
       linewidth=2, color=GREEN, alpha=0.8, label="EKF estimate")

ax.set_xlabel("X (meters)", fontsize=14, weight="bold")
ax.set_ylabel("Y (meters)", fontsize=14, weight="bold")
ax.set_title("Position Tracking", fontsize=18, weight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("docs/presentation/visuals/slideXX_name.png",
           dpi=150, bbox_inches="tight", facecolor="white")
print(f"✓ Saved: slideXX_name.png")
plt.close()
```

### Video Generation Template

```python
from trodestrack.viz import create_diagnostic_video

video_path = create_diagnostic_video(
    sim,                                                 # SimOut (positional)
    "docs/presentation/videos/slideXX_name.mp4",         # output_path (positional)
    filter_results=result,                               # plural; takes EKFResult/EKF3DResult
    fps=30,
    speedup=1.0,        # Realtime
    time_window_s=2.0,  # Show 2s of history
    trail_length_s=1.5,
    dpi=100,
    codec="h264",
    bitrate=2000,
)
print(f"✓ Saved: {video_path}")
```

---

## Common Issues & Solutions

### Issue: KeyError for 't_cam' or 'x_truth'
**Solution**: SimOut dict uses:
- `t_cam_exp` (not `t_cam`)
- `X_truth` (capital X, not lowercase)
- `U_imu` (capital U)

### Issue: TypeError on extended_kalman_filter()
**Solution**: Pass individual arrays, not the sim dict:
```python
result = extended_kalman_filter(
    config,
    sim["t_imu"], sim["U_imu"], sim["t_cam_exp"],
    sim["Z_cam_led1"], sim["Z_cam_led2"], sim["mask_cam"]
)
```

### Issue: Bias parameter error
**Solution**: Use random walk densities, not std:
```python
gyro_bias_rw_density=np.deg2rad(0.03)  # Not gyro_bias_std
accel_bias_rw_density=0.05
```

### Issue: Misaligned truth/estimate arrays
**Solution**: Downsample IMU-rate truth to camera-rate:
```python
cam_indices = np.searchsorted(sim["t_imu"], sim["t_cam_exp"])
cam_indices = np.clip(cam_indices, 0, len(sim["t_imu"]) - 1)
truth_cam = sim["X_truth"][cam_indices]
```

---

## Testing Generated Code

```bash
# Run visual generation script
uv run python docs/presentation/code/generate_slideXX.py

# View output
open docs/presentation/visuals/slideXX_name.png

# Check file size (should be <500KB for PNGs)
ls -lh docs/presentation/visuals/slideXX_name.png
```

---

## Progress Tracking

After generating each visual:

1. Open `docs/presentation/TASKS.md`
2. Check off the completed item: `- [x]` instead of `- [ ]`
3. Update the progress tracker table at the bottom
4. Commit changes:
```bash
git add docs/presentation/
git commit -m "Generate slide XX visual: [description]"
```

---

## Quality Checklist for Each Visual

Before marking a visual complete, verify:

- [ ] Follows color palette (Blue, Orange, Green, Red, Gray)
- [ ] High resolution (150+ DPI)
- [ ] Readable fonts (24+ pt body, 32+ pt headlines)
- [ ] 16:9 aspect ratio
- [ ] Proper labeling (axes, legend, title)
- [ ] Matches spec in PRESENTATION_OUTLINE.md
- [ ] Code is documented and reusable
- [ ] File saved to correct location
- [ ] File size reasonable (<1MB for PNGs, <50MB for videos)

---

## Estimated Time Remaining

| Task | Time Estimate |
|------|---------------|
| **Phase 1: Core visuals (5 remaining)** | 6-10 hours |
| Slide 3: Trajectory comparison | 30 min |
| Slide 21: NEES histogram | 30 min |
| Slide 18: Diagnostic panel | 30 min |
| Slide 14: Uncertainty evolution | 1 hour |
| Slide 16: Smoother comparison | 1-2 hours |
| Slide 2: Failure modes | 1-2 hours |
| Slide 12: IMU integration | 1-2 hours |
| Slide 8: Before/after video | 2-3 hours |
| **Phase 2: PowerPoint assembly** | 4-8 hours |
| **Phase 3: Speaker notes** | 4-6 hours |
| **Phase 4: Optional visuals (13)** | 10-15 hours |
| **Phase 5: Review & polish** | 2-4 hours |
| **TOTAL REMAINING** | **26-43 hours** |

---

## Success Criteria

The presentation is complete when:

1. ✅ All 43 slides have content (text + visuals where applicable)
2. ✅ All 8 Priority 1 visuals are generated and look professional
3. ✅ PowerPoint/PDF file is assembled and renders correctly
4. ✅ Speaker notes document is complete
5. ✅ Handout PDF is generated
6. ✅ All files are organized in docs/presentation/
7. ✅ A neuroscientist can understand the package purpose and start using it after viewing

---

## Contact & Resources

**Original Context**: See docs/presentation/PROGRESS_REPORT.md for full history

**Package Docs**:
- PRD: .claude/docs/PRD.md
- Development guide: .claude/docs/CLAUDE.md
- README: README.md

**Example Code**:
- Working example: docs/presentation/code/generate_slide05abc_imu_physics.py
- Package examples: examples/ (10 numbered files)

**If Stuck**:
1. Check PRESENTATION_OUTLINE.md for slide specifications
2. Look at generate_slide05abc_imu_physics.py for patterns
3. Review TASKS.md for step-by-step breakdown
4. Check PROGRESS_REPORT.md Section "Key Insights from IMU Physics Development"

---

## Quick Commands Reference

```bash
# Generate a visual
uv run python docs/presentation/code/generate_slideXX.py

# View generated visuals
open docs/presentation/visuals/

# Run existing examples (for reference)
uv run python examples/03_ekf_basic_scenarios.py
uv run python examples/07_smoother_demonstration.py

# Check package structure
tree -L 2 src/trodestrack/

# Test simulator
uv run python -c "from trodestrack.sim.rat_imu import simulate_rat_imu; print('OK')"

# Test EKF
uv run python -c "from trodestrack.models.ekf import extended_kalman_filter; print('OK')"
```

---

**Last Updated**: see git log for the most recent presentation-tree change.
**Status**: 8 PNGs shipped + 1 MP4 + 7 generator scripts; slides 2 (failure
modes) and 18 (9-panel diagnostic) currently render as bullet content.
**Next Task**: write generator scripts for slides 2 and 18 to upgrade them
from bullet content to full-image slides.

---

## Ready-to-Use Starter Prompt (Minimal)

```
Continue the TrodesTrack presentation. Read these 3 files:
1. docs/presentation/PRESENTATION_OUTLINE.md
2. docs/presentation/TASKS.md
3. docs/presentation/code/generate_slide05abc_imu_physics.py

Then generate Slide 3 visual (trajectory comparison: ground truth vs noisy observations).
Save to docs/presentation/visuals/slide03_trajectory_comparison.png.
Use the simulator and follow the patterns in the example code.
```

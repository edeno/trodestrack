# TrodesTrack Presentation - Task Checklist

## Phase 1: Core Visual Generation (Priority 1)

### Slide 2: Failure Modes
- [ ] Create `code/generate_slide02_failure_modes.py`
- [ ] Generate 2×2 grid showing:
  - [ ] Top-left: LED occlusion (rat near wall)
  - [ ] Top-right: LED reflection (shiny surface)
  - [ ] Bottom-left: Motion blur (fast movement)
  - [ ] Bottom-right: Dim lighting (low confidence)
- [ ] Use `simulate_rat_imu()` with high dropout rate
- [ ] Extract 4 representative frames from video
- [ ] Save as `visuals/slide02_failure_modes.png`
- [ ] **Status**: deferred — slide currently renders as bullet content
  in `build_presentation.py` (slide 2)

### Slide 3: Trajectory Comparison
- [x] Create `code/generate_slide03_trajectory_comparison.py`
- [x] Generate 30s simulation with 30% dropout
- [x] Create side-by-side subplot:
  - [x] Left: Ground truth (smooth blue line)
  - [x] Right: Vision-only observations (red scatter, gaps)
- [x] Add annotations for dropout periods
- [x] Save as `visuals/slide03_trajectory_comparison.png`
- [x] **Completed**: 30 minutes

### Slide 8: Before/After Video
- [x] Create `code/generate_slide08_beforeafter_video.py`
- [x] Generate 10s simulation with 5s dropout in middle
- [x] Create split-screen video:
  - [x] Left panel: Vision-only extrapolation (jumps, huge uncertainty)
  - [x] Right panel: Sensor fusion (smooth, bounded uncertainty)
- [x] Use custom rendering with matplotlib animation
- [x] Add synchronized time counter
- [x] Highlight dropout period with red text in title
- [x] Save as `videos/slide08_beforeafter.mp4`
- [x] **Completed**: 2 hours

### Slide 12: IMU Integration Visualization
- [x] Create `code/generate_slide12_imu_integration.py`
- [x] Simulate circular motion (easy to visualize integration)
- [x] Create timeline diagram showing:
  - [x] Camera frame k at t=0.000s
  - [x] IMU samples at 0.01, 0.02, 0.03s (with measurements)
  - [x] Camera frame k+1 at t=0.033s
- [x] Show integration: θ += ω*dt, v += a*dt, x += v*dt
- [x] Three-panel layout (timeline, measurements, spatial)
- [x] Save as `visuals/slide12_imu_integration.png`
- [x] **Completed**: 1 hour

### Slide 14: Uncertainty Evolution
- [x] Create `code/generate_slide14_uncertainty.py`
- [x] Generate 20s simulation with 5s dropout
- [x] Plot trajectory with covariance ellipses:
  - [x] Small ellipses when camera sees rat
  - [x] Growing ellipses during dropout
  - [x] Shrinking ellipses when camera returns
- [x] Implemented custom `plot_covariance_ellipse()` function
- [x] Color-code ellipses: green (camera active), red (dropout)
- [x] Save as `visuals/slide14_uncertainty.png`
- [x] **Completed**: 1 hour

### Slide 16: Smoother Comparison
- [x] Create `code/generate_slide16_smoother.py`
- [x] Generate 30s simulation with 5s dropout
- [x] Run three algorithms:
  - [x] Vision-only (naive extrapolation)
  - [x] EKF forward-only
  - [x] RTS smoother (forward + backward)
- [x] Create 3-panel figure showing trajectories during dropout
- [x] Calculate drift for each: vision (0.031m), EKF (0.008m), RTS (0.009m)
- [x] Annotate drift values on each panel
- [x] Save as `visuals/slide16_smoother_comparison.png`
- [x] **Completed**: 1 hour

### Slide 18: 9-Panel Diagnostic Screenshot
- [ ] Create `code/generate_slide18_diagnostic.py`
- [ ] Generate 10s simulation
- [ ] Run EKF with `create_diagnostic_video()`
- [ ] Extract single frame showing all 9 panels:
  1. Arena view with trajectory
  2. Gyro Z time series
  3. Accel X/Y time series
  4. Camera status/confidence
  5. Position error
  6. Velocity error
  7. Heading error
  8. NEES diagnostic
  9. Bias estimates
- [ ] Ensure frame shows interesting moment (middle of simulation)
- [ ] Save as `visuals/slide18_diagnostic_panel.png`
- [ ] **Status**: deferred — slide currently renders as bullet content
  in `build_presentation.py` (slide 18)

### Slide 21: NEES Histogram
- [x] Create `code/generate_slide21_nees.py`
- [x] Generate 60s simulation with realistic dropout
- [x] Run EKF and compute NEES
- [x] Plot histogram of NEES values
- [x] Overlay χ² theoretical distribution (5 DOF for measurable states)
- [x] Add 95% confidence interval bounds
- [x] Annotate mean NEES (~5 is ideal for 5D)
- [x] Color-code: green if in 95% CI, else red
- [x] Save as `visuals/slide21_nees_histogram.png`
- [x] **Completed**: 30 minutes

---

## Phase 2: Presentation Assembly

### Option A: Markdown → PDF (Fastest)
- [ ] Create `slides.md` in Markdown format
- [ ] Use pandoc/beamer syntax for slides
- [ ] Insert generated images with `![](visuals/...)`
- [ ] Add slide titles, bullet points, captions
- [ ] Convert to PDF:
  ```bash
  pandoc slides.md -t beamer -o trodestrack_presentation.pdf
  ```
- [ ] Review PDF output
- [ ] **Estimated time**: 4-5 hours

### Option B: Python-PPTX (Most Flexible)
- [ ] Install `python-pptx`: `uv add python-pptx`
- [ ] Create `code/build_presentation.py`
- [ ] Define slide template/theme
- [ ] For each slide:
  - [ ] Create slide with layout (title, content, image)
  - [ ] Add title text
  - [ ] Add body text (bullet points or paragraphs)
  - [ ] Insert image (if applicable)
  - [ ] Set fonts, colors, alignment
- [ ] Save as `trodestrack_presentation.pptx`
- [ ] Open in PowerPoint/LibreOffice to verify
- [ ] **Estimated time**: 6-8 hours

### Option C: Manual Assembly (Most Control)
- [ ] Open PowerPoint/Keynote
- [ ] Create 43 blank slides
- [ ] For each slide:
  - [ ] Add title
  - [ ] Add body text
  - [ ] Insert image (drag from `visuals/` folder)
  - [ ] Format text (fonts, sizes, colors)
  - [ ] Align elements
  - [ ] Add animations (optional)
- [ ] Apply consistent theme
- [ ] Save as `trodestrack_presentation.pptx`
- [ ] **Estimated time**: 10-12 hours

**Recommendation**: Start with Option A for MVP, upgrade to Option B for final version

---

## Phase 3: Supporting Materials

### Speaker Notes
- [ ] Create `speaker_notes.md`
- [ ] For each slide, write:
  - [ ] What to say (1-2 paragraphs)
  - [ ] Key points to emphasize
  - [ ] Transition to next slide
  - [ ] Estimated talk time (in seconds)
- [ ] Total script should be ~25-28 minutes (leave 2-5 min for Q&A)
- [ ] **Estimated time**: 4-6 hours

### Handout PDF
- [ ] Open `trodestrack_presentation.pptx` in PowerPoint
- [ ] Go to File → Print → Print Layout → Handouts (2 slides per page)
- [ ] Print to PDF
- [ ] Save as `handout.pdf`
- [ ] **Estimated time**: 5 minutes

---

## Phase 4: Optional/Nice-to-Have Visuals

### Slide 10: Predict-Update Cycle Diagram
- [ ] Create `code/generate_slide10_cycle.py`
- [ ] Draw flowchart with boxes and arrows
- [ ] Box 1: "Predict (IMU)" → Box 2: "Update (Camera)" → loop back
- [ ] Annotate each box with equations (simplified)
- [ ] Use matplotlib or diagrams library
- [ ] Save as `visuals/slide10_cycle_diagram.png`
- [ ] **Estimated time**: 1 hour

### Slide 11: State Vector Rat Diagram
- [ ] Create `code/generate_slide11_state_vector.py`
- [ ] Draw top-down rat schematic
- [ ] Add labeled arrows for each state variable:
  - Position (x, y)
  - Velocity (vx, vy)
  - Heading (θ)
  - Biases (b_gz, b_ax, b_ay)
- [ ] Save as `visuals/slide11_state_vector.png`
- [ ] **Estimated time**: 1 hour

### Slide 15: EKF vs UKF Comparison
- [ ] Create `code/generate_slide15_ekf_vs_ukf.py`
- [ ] Run both filters on same challenging scenario
- [ ] Plot position RMSE over time
- [ ] Create metrics comparison table (9 metrics)
- [ ] Save as `visuals/slide15_ekf_ukf_comparison.png`
- [ ] **Estimated time**: 1 hour

### Slide 20: Simulator Code Demo
- [ ] Create `code/generate_slide20_simulator.py`
- [ ] Format code snippet nicely (syntax highlighting)
- [ ] Show example `simulate_rat_imu()` call
- [ ] Show output plot (trajectory + IMU data)
- [ ] Save as `visuals/slide20_simulator_demo.png`
- [ ] **Estimated time**: 30 minutes

### Slide 22: QA Report Montage
- [ ] Generate full QA report PDF using existing tools
- [ ] Extract pages 1-5 as images
- [ ] Create montage/grid of thumbnails
- [ ] Save as `visuals/slide22_qa_report.png`
- [ ] **Estimated time**: 30 minutes

### Slide 24: Performance Bar Chart
- [ ] Create `code/generate_slide24_performance.py`
- [ ] Create horizontal bar chart:
  - CPU (5-min session): 316× realtime (0.95s)
  - GPU (estimated): 1000×+ realtime (~0.3s)
  - Realtime baseline: 1× (300s)
- [ ] Color-code: green for fast, red for slow
- [ ] Save as `visuals/slide24_performance.png`
- [ ] **Estimated time**: 30 minutes

### Slide 27: Installation Terminal Screenshot
- [ ] Run installation commands in terminal
- [ ] Capture clean screenshot showing:
  ```bash
  git clone https://github.com/edeno/trodestrack.git
  cd trodestrack
  uv sync
  uv run python examples/03_ekf_basic_scenarios.py
  ✓ [output showing success]
  ```
- [ ] Save as `visuals/slide27_installation.png`
- [ ] **Estimated time**: 15 minutes

### Slide 28: Learning Path Flowchart
- [ ] Create `code/generate_slide28_learning_path.py`
- [ ] Draw flowchart of 10 examples with arrows
- [ ] Annotate each node with example number and topic
- [ ] Use different colors for difficulty levels
- [ ] Save as `visuals/slide28_learning_path.png`
- [ ] **Estimated time**: 1 hour

### Slide 29: Filter Decision Tree
- [ ] Create `code/generate_slide29_decision_tree.py`
- [ ] Draw decision tree flowchart:
  - "Need real-time?" → Yes: EKF, No: continue
  - "Strong nonlinearities?" → Yes: UKF, No: EKF
  - "Accuracy critical?" → Yes: IEKS, No: RTS
- [ ] Save as `visuals/slide29_decision_tree.png`
- [ ] **Estimated time**: 1 hour

### Slide 34: JAX Speedup Comparison
- [ ] Create `code/generate_slide34_jax_speedup.py`
- [ ] Benchmark EKF with/without JIT compilation
- [ ] Create bar chart: Python loop (1×), JAX no-JIT (10×), JAX JIT (300×)
- [ ] Save as `visuals/slide34_jax_speedup.png`
- [ ] **Estimated time**: 1 hour

### Slide 35: 2D → 3D Extension Diagram
- [ ] Create `code/generate_slide35_3d_extension.py`
- [ ] Show side-by-side comparison:
  - Left: 2D rat (top-down view, 3 DOF)
  - Right: 3D rat (perspective view, 6 DOF with roll/pitch/yaw)
- [ ] Annotate state dimensions: 8D → 15D/16D
- [ ] Save as `visuals/slide35_3d_extension.png`
- [ ] **Estimated time**: 1 hour

### Slide 36: Plugin Architecture Diagram
- [ ] Create `code/generate_slide36_plugins.py`
- [ ] Draw architecture diagram:
  - Core filter (center)
  - Measurement model plugins (around edges): camera, heading, ZUPT, compass
  - Protocol interface (arrows)
- [ ] Save as `visuals/slide36_plugin_architecture.png`
- [ ] **Estimated time**: 1 hour

---

## Phase 5: Review & Polish

### Content Review
- [ ] Review all slides for:
  - [ ] Consistent terminology
  - [ ] No typos/grammatical errors
  - [ ] Accurate technical content
  - [ ] Proper citations
- [ ] Check that all learning objectives are addressed
- [ ] Verify PRD metrics are cited correctly

### Visual Review
- [ ] Check all images for:
  - [ ] Consistent color scheme (Blue, Orange, Green, Red, Gray)
  - [ ] Readable fonts (24+ pt minimum)
  - [ ] High resolution (150 DPI minimum)
  - [ ] Proper aspect ratio (16:9)
  - [ ] Colorblind-safe palettes
- [ ] Ensure videos play correctly
- [ ] Test presentation on projector (if possible)

### Accessibility Check
- [ ] High contrast text on backgrounds
- [ ] Alt text for all images (in speaker notes)
- [ ] No flashing animations (>3 Hz)
- [ ] Readable from back of room (30+ pt for key text)

### Final Export
- [ ] Export PowerPoint as PDF for backup
- [ ] Export handout PDF (2 slides/page)
- [ ] Package all files:
  ```
  trodestrack_presentation.pptx
  trodestrack_presentation.pdf
  handout.pdf
  speaker_notes.md
  visuals/ (folder with all images)
  videos/ (folder with all videos)
  ```
- [ ] Zip and archive

---

## Completion Checklist

### Minimal Viable Presentation (MVP)
- [ ] Phase 1: Core visuals (slides 2, 3, 8, 12, 14, 16, 18, 21) → **8 of 10 PNGs shipped + 1 MP4; slides 2 and 18 deferred (bullet content for now)**
- [ ] Phase 2: Presentation assembly (Option A or B)
- [ ] Phase 5: Basic review

**MVP Estimated Time**: 4-8 hours remaining (presentation assembly + review)

### Full Production Version
- [ ] Phase 1: All core visuals
- [ ] Phase 2: PowerPoint assembly (Option B)
- [ ] Phase 3: Speaker notes + handout
- [ ] Phase 4: All optional visuals
- [ ] Phase 5: Comprehensive review

**Full Version Estimated Time**: 30-40 hours total

---

## Progress Tracker

**Current Status**: Phase 1 partially shipped — 8 of 10 PNGs + 1 MP4. Slide 2 (failure modes) and slide 18 (9-panel diagnostic) deferred and rendered as bullet content in `build_presentation.py`.

| Phase | Tasks Complete | Tasks Remaining | % Done |
|-------|----------------|-----------------|--------|
| **Phase 1** | 8 | 2 (slides 2, 18) | 80% |
| **Phase 2** | 0 | 1 | 0% |
| **Phase 3** | 0 | 2 | 0% |
| **Phase 4** | 0 | 13 | 0% |
| **Phase 5** | 0 | 4 | 0% |
| **TOTAL** | **8** | **22** | **27%** |

---

## Quick Reference

### Shipped ✅
- Slide 5A: Accelerometer physics
- Slide 5B: Gyroscope physics
- Slide 5C: Bias correction
- Slide 3: Trajectory comparison
- Slide 8: Before/after video
- Slide 12: IMU integration
- Slide 14: Uncertainty evolution
- Slide 16: Smoother comparison
- Slide 21: NEES histogram

### Deferred ⏳ (bullet content in `build_presentation.py`)
- Slide 2: Failure modes grid — generator script not yet written
- Slide 18: 9-panel diagnostic — generator script not yet written

### Next Up (Phase 2) ⏭️
1. **Presentation Assembly** (4-8 hours) ← **NEXT**
   - Option A: Markdown → PDF (fastest, 4-5 hours)
   - Option B: Python-PPTX (most flexible, 6-8 hours)
   - Option C: Manual (most control, 10-12 hours)

---

**Last Updated**: see git log for the most recent presentation-tree change.
Phase 1: 8 of 10 visuals shipped + 1 MP4; slides 2 and 18 deferred (bullet
content in `build_presentation.py`).
**Author**: Claude Code

# TrodesTrack Presentation Outline

**Target Audience:** Neuroscientists (Behavioral, Systems, Computational)
**Duration:** 30 minutes
**Slides:** 43 slides (40 original + 3 IMU physics details)

---

## Learning Objectives

1. Understand the **tracking problem** in behavioral neuroscience
2. Grasp **why sensor fusion matters** (camera + IMU > camera alone)
3. Learn **how TrodesTrack solves** common tracking challenges
4. Understand **IMU sensor physics** (what accelerometers/gyroscopes measure, needed corrections)
5. Know **when to use** EKF vs UKF vs smoothing
6. Be able to **interpret quality metrics** (RMSE, NEES)
7. Feel **confident starting** with TrodesTrack on their own data

---

## Section Breakdown

### SECTION 1: THE PROBLEM (Slides 1-11)
- Slide 1: Title
- Slide 2: The Behavioral Tracking Challenge
- Slide 3: Real-World Consequences
- Slide 4: Why Vision-Only Tracking Fails
- Slide 5: Enter the IMU
- **Slide 5A: What Does an Accelerometer REALLY Measure?** ⭐ NEW
- **Slide 5B: What Does a Gyroscope REALLY Measure?** ⭐ NEW
- **Slide 5C: The Correction Challenge** ⭐ NEW
- Slide 6: The Sensor Fusion Idea
- Slide 7: What is TrodesTrack?
- Slide 8: Quick Preview: Before & After

### SECTION 2: HOW IT WORKS (Slides 12-21)
- Slide 9: Section Divider
- Slide 10: The Core Algorithm: Kalman Filtering
- Slide 11: What We Track (State Vector)
- Slide 12: The Predict Step (IMU Integration)
- Slide 13: The Update Step (Camera Correction)
- Slide 14: Handling Uncertainty
- Slide 15: EKF vs UKF
- Slide 16: Offline Smoothing
- Slide 17: Robustness Features
- Slide 18: The 9-Panel Diagnostic Video

### SECTION 3: FEATURES & CAPABILITIES (Slides 22-28)
- Slide 19: Section Divider
- Slide 20: Synthetic Data Simulator
- Slide 21: Quality Assurance Metrics
- Slide 22: Automated QA Reports
- Slide 23: Flexible State Tracking Modes
- Slide 24: Performance & Scalability
- Slide 25: Real Data Support

### SECTION 4: GETTING STARTED (Slides 29-35)
- Slide 26: Section Divider
- Slide 27: Installation
- Slide 28: Learning Path
- Slide 29: Decision Tree: Which Filter?
- Slide 30: When to Use TrodesTrack
- Slide 31: Troubleshooting Common Issues
- Slide 32: Resources & Support

### SECTION 5: ADVANCED TOPICS (Slides 36-39)
- Slide 33: Section Divider
- Slide 34: Under the Hood: JAX Implementation
- Slide 35: Extending to 3D
- Slide 36: Custom Measurement Models

### SECTION 6: CONCLUSION (Slides 40-43)
- Slide 37: Key Takeaways
- Slide 38: Comparison to Alternatives
- Slide 39: Future Directions
- Slide 40: Thank You + Contact

---

## New IMU Physics Slides Details

### Slide 5A: What Does an Accelerometer REALLY Measure?
**Purpose:** Clarify accelerometer physics (specific force, not acceleration)

**Content:**
- Accelerometers measure **specific force**: f = a - g
- At rest: reads +1g (not zero!)
- Free fall: reads 0g (counterintuitive)
- Challenge: Gravity contaminates motion when IMU is tilted

**Visual:** 3-panel diagram showing accelerometer readings

**Code to generate:**
```python
# Demonstrate accelerometer physics
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Stationary
# Panel 2: Free fall
# Panel 3: Horizontal acceleration

# Show spring-mass diagram with readings
```

### Slide 5B: What Does a Gyroscope REALLY Measure?
**Purpose:** Explain gyroscope physics and drift problem

**Content:**
- Measures angular velocity (°/s) in body frame
- Integration gives heading: θ = θ₀ + ∫ω dt
- Drift problem: Bias accumulates unbounded
- SpikeGadgets specs: 0.01 °/s/√Hz noise, 1-5 °/s bias

**Visual:** Rotating rat with gyro axis, integration error growth

**Code to generate:**
```python
# Show gyro integration drift
from trodestrack.sim.simple import simulate_circular
import numpy as np

# Add increasing bias to gyro, show heading error growth
```

### Slide 5C: The Correction Challenge
**Purpose:** Show 4 corrections needed and how TrodesTrack handles them

**Content:**
- Gravity removal (assume 2D or estimate tilt)
- Gyro bias estimation (in state vector)
- Accel bias estimation (in state vector)
- Frame alignment (rotation matrix)

**Visual:** Before/after comparison (raw integration vs corrected)

**Code to generate:**
```python
# Compare raw IMU integration vs bias-corrected
from trodestrack.sim.rat_imu import simulate_rat_imu
from trodestrack.models.ekf import extended_kalman_filter

# Show drift with/without bias correction
```

---

## Visual Generation Plan

### Priority 1: Core Concepts (Must-Have)
1. Slide 2: Failure modes (dropout, reflection, blur, dim)
2. Slide 3: Ground truth vs noisy trajectory
3. Slide 5A: Accelerometer physics diagram
4. Slide 5B: Gyroscope drift growth
5. Slide 5C: Bias correction comparison
6. Slide 8: Before/after video (10s)
7. Slide 12: IMU integration animation
8. Slide 14: Uncertainty evolution
9. Slide 16: Smoother 3-panel comparison
10. Slide 18: 9-panel diagnostic screenshot
11. Slide 21: NEES histogram

### Priority 2: Supporting Visuals (Nice-to-Have)
12. Slide 10: Predict-update cycle diagram
13. Slide 11: State vector rat diagram
14. Slide 13: Kalman gain illustration
15. Slide 15: EKF vs UKF accuracy plot
16. Slide 20: Simulator code + output
17. Slide 22: QA report PDF montage
18. Slide 24: Performance bar chart
19. Slide 27: Terminal screenshot
20. Slide 28: Learning path flowchart

### Priority 3: Diagrams (Can be simple)
21. Slide 6: Sensor fusion Venn diagram
22. Slide 29: Filter decision tree
23. Slide 34: JAX speedup comparison
24. Slide 35: 2D→3D comparison
25. Slide 36: Plugin architecture

---

## File Organization

```
docs/presentation/
├── PRESENTATION_OUTLINE.md          # This file
├── trodestrack_presentation.pptx    # Final PowerPoint
├── speaker_notes.md                 # Full script
├── handout.pdf                      # 2-slides-per-page PDF
├── visuals/                         # Generated images
│   ├── slide02_failure_modes.png
│   ├── slide03_trajectory_comparison.png
│   ├── slide05a_accelerometer_physics.png
│   ├── slide05b_gyro_drift.png
│   ├── slide05c_bias_correction.png
│   ├── slide08_beforeafter.mp4
│   ├── slide12_imu_integration.png
│   ├── slide14_uncertainty.png
│   ├── slide16_smoother_comparison.png
│   ├── slide18_diagnostic_panel.png
│   ├── slide21_nees_histogram.png
│   └── ...
├── videos/                          # Video files
│   ├── slide08_beforeafter.mp4
│   └── slide16_smoother_demo.mp4
└── code/                            # Scripts to regenerate visuals
    ├── generate_slide02.py
    ├── generate_slide03.py
    ├── generate_slide05abc.py
    ├── generate_slide08.py
    └── ...
```

---

## Next Steps

1. ✅ Create folder structure
2. ⏳ Generate Priority 1 visuals (IMU physics slides first)
3. ⏳ Build PowerPoint template
4. ⏳ Populate slides section-by-section
5. ⏳ Add speaker notes
6. ⏳ Review and polish
7. ⏳ Export handout PDF

---

## Estimated Timeline

- **IMU physics visuals (3 new slides):** 2-3 hours
- **Priority 1 visuals:** 6-8 hours
- **PowerPoint assembly:** 8-10 hours
- **Speaker notes:** 2-3 hours
- **Polish:** 2-3 hours
- **Total:** 20-27 hours

---

## Success Criteria

A neuroscientist who views this presentation should be able to:
1. ✓ Explain why sensor fusion is better than vision-only
2. ✓ **Describe what accelerometers and gyroscopes physically measure** ⭐ NEW
3. ✓ **List the 4 corrections needed for IMU data** ⭐ NEW
4. ✓ Describe the predict-update cycle of Kalman filtering
5. ✓ Interpret NEES values (overconfident vs underconfident)
6. ✓ Choose between EKF, UKF, and smoothing for their use case
7. ✓ Run the first example successfully
8. ✓ Feel confident TrodesTrack can solve their tracking problem

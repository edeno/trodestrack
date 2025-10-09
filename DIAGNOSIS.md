# Dropout Drift & Bias Diagnosis — Step-by-Step Playbook

## Key file anchors (line numbers from your current snapshot)

- **ekf.py**:
  - `update_heading(...)` at **L1002**
  - Heading innovation `wrap_angle(heading_obs - m[4])` around **L1085**
  - Yaw-rate minus bias (ω − b) used in predict around **L649**
  - `R_heading` usage lines: L117, L1069, L1078, L1093, L1104
- **rat_imu.py**:
  - `compute_gravity_in_tilted_frame(...)` at **L45**
  - Gravity/tilt & specific-force comments near: L14, L41, L45, L46, L49, L52, L56, L58, L64, L65, L66, L69
- **test_prd_acceptance.py**:
  - `test_prd_dropout_drift_5s` at **L282**
  - `mask_cam` references near: L79, L113, L323, L328

---

## P0 — Make the blackout real (test patch)

**Problem:** The test toggles only `mask_cam`, which might still let the EKF treat per-LED pixels as valid (if finiteness is checked). That undermines the intended 5 s no-vision interval.

**Fix:** Force both LEDs to NaN and per-LED masks to False during the blackout window, in addition to flipping `mask_cam`.

### Patch (apply inside `test_prd_dropout_drift_5s`)

```python
# After computing dropout_start_idx / dropout_end_idx and mask_with_dropout
sim_data_dropout = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in sim_data.items()}
sim_data_dropout["mask_cam"] = mask_with_dropout

# Ensure no usable pixels during blackout:
for key in ("Z_cam_led1", "Z_cam_led2"):
    if key in sim_data_dropout:
        arr = sim_data_dropout[key].copy()
        arr[dropout_start_idx:dropout_end_idx] = float("nan")
        sim_data_dropout[key] = arr

# Force per-LED masks off as well:
for key in ("mask_led1", "mask_led2"):
    if key in sim_data_dropout:
        arr = sim_data_dropout[key].copy()
        arr[dropout_start_idx:dropout_end_idx] = False
        sim_data_dropout[key] = arr
```

Now the EKF will **definitely** see a measurement-free gap.

---

## P0 — Eliminate the biggest physics mismatch first: IMU tilt

**Problem:** The simulator projects gravity into the IMU x/y via **roll/pitch tilt** and rotates it by yaw each step to create **specific force**. If the EKF assumes a flat IMU (no tilt terms in the model), gravity leakage looks like acceleration and integrates to drift during blackout.

**Quick diagnostic:** Set IMU tilts to zero for this acceptance test.

### Patch (simulation config in the test)

```python
config = config.replace(  # or construct with kwargs if not frozen
    imu_tilt_roll_deg=0.0,
    imu_tilt_pitch_deg=0.0,
)
```

If drift collapses → main culprit is unmodeled tilt. Later, either (a) keep tilt=0 for PRD tests, or (b) extend the filter to estimate tilt.

---

## P0 — Verify heading is truly informing the bias **before** the blackout

Add two tiny probes in `ekf.update_heading(...)`:

1) **Innovation direction** (should be meas−pred):

```python
print("innov_theta(rad) =", float(wrap_angle(heading_obs - m[4])))
```

2) **Bias gain element**:

```python
print("K_b_gz =", float(K[5]))
```

- `K[5]` should be **negative on average** if your state index 5 is the gyro bias and the model uses `ω − b` in predict.
- Count how many frames have an **effective** heading update (e.g., `R_heading < 1e5` if you gate by inflating noise). Too few updates → not enough bias learning time.

Tip: turn this into a debug counter returned by the EKF in a `debug` dict.

---

## P1 — Align damping & disable speed clipping (optional sanity)

The sim uses velocity drag and a tanh speed clip; the EKF may use a different damping and no clip. During IMU-only intervals, even small mismatches integrate into drift.

- Set EKF `damping_coeff` to match sim’s `vel_drag` for this test.
- Temporarily disable sim speed clipping (set very high limit) to remove extra nonlinearity.

If drift reduces, prefer aligning these params in acceptance scenarios.

---

## P1 — Ensure gates and spacing tolerance don’t silently kill heading updates

If you rely on LED spacing validity and confidence to enable heading, log:

- both LEDs finite,
- spacing within `led_distance_tolerance`,
- resulting `R_heading`.

If many frames just before the blackout have invalid spacing/confidence, reduce `measurement_noise_heading` or relax the tolerance so the bias can learn in the ~5 s pre-gap window.

---

## P1 — Tuning sweep focused on the PRD window

- Increase `process_noise_gyro_bias` ×3–×10.
- Decrease `measurement_noise_heading` ×3–×10 (or disable adaptive scaling temporarily).

Run the test; if it passes, dial back toward conservative values while staying within the bound.

---

## P2 — Optional experiments to bound effects

- **Freeze accel bias during blackout** (set its Q to 0 only over the gap) to see how much drift is from bias RW vs. tilt/mismatch.
- **Deterministic blackout helper**: create a helper that returns a mask with an exact blackout window so tests don’t rely on stochastic dropouts.

---

## EKF code sanity (signs look correct)

- Predict uses yaw-rate **minus bias** (ω − b).
  Anchor near: L649
- Heading innovation is **meas − pred** with wrap.
  Anchor near: L1085
- Heading update noise `R_heading` is inflated to disable update when invalid.
  Anchors: L117, L1069, L1078, L1093, L1104

These confirm the two classic “bias flips” are not in EKF; the remaining drivers are test masking, sim tilt, and pre-gap learn-time/tuning.

---

## Quick acceptance checklist

- [ ] Blackout window blanks **mask_cam**, **mask_led{1,2}**, and sets **Z_cam_led{1,2}** to NaN for 5 s.
- [ ] IMU tilts are **zero** in this PRD test (or the EKF estimates tilt).
- [ ] EKF `damping_coeff` ≈ sim `vel_drag` for this scenario; speed clipping disabled for acceptance runs.
- [ ] Heading pseudo-measurement is **enabled** and effective (N updates before gap ≥ threshold).
- [ ] Bias tuning allows convergence in first ~5 s (`Q_b` not too small; `R_heading` reasonable).
- [ ] Test passes deterministically with fixed seed.

---

## Pytest quick commands

```bash
# Focus on the PRD test
uv run pytest -q tests/filters/test_prd_acceptance.py::test_prd_dropout_drift_5s -k "not slow"

# Run with debug prints visible
uv run pytest -s tests/filters/test_prd_acceptance.py::test_prd_dropout_drift_5s
```

# 🧩 trodestrack — PR Fix Plan (Based on Full Code Review)

This file summarizes all recommended fixes, grouped by priority.
Each section includes context, rationale, and **code suggestions** inline for implementation.

---

## 🔴 P0 — PRD Compliance (Must Fix Before Release)

### ✅ EKF/UKF Robustness & Correctness

**Files:** `src/trodestrack/models/ekf.py`, `src/trodestrack/models/ukf.py`, `tests/filters/`

**Goals:**

* Fix angle wrapping, confidence-scaled noise, and χ² gating.
* Replace "big-R" masking with shape-specific updates (2D / 4D).
* Implement heading measurement from LED pair.

**Tasks:**

* [ ] Wrap heading `θ` in predict *and* update:

  ```python
  theta_next = wrap_angle(theta + omega_z_unbiased * dt)
  ...
  m_upd = m_in + K @ innov
  m_upd = m_upd.at[4].set(wrap_angle(m_upd[4]))
  ```

* [ ] Remove `1e10` masking. Split update paths:

  ```python
  if both_leds_valid:
      H, R = H_4d, R_4d
  else:
      H, R = H_2d, R_2d
  ```

* [ ] Add confidence-scaled measurement noise:

  ```python
  R_scaled = R_base / np.clip(conf, 1e-2, 1.0)
  ```

* [ ] Add χ² gating (p=0.997):

  ```python
  nis = innov.T @ psd_solve(S, innov)
  if nis > chi2.ppf(0.997, df=H.shape[0]):
      return m_in, P_in  # skip update
  ```

* [ ] Add heading pseudo-measurement:

  ```python
  if both_leds_valid:
      z_heading = np.arctan2(dy, dx)
      H_heading = np.zeros((1, 8)); H_heading[0, 4] = 1
  ```

* [ ] Update tests:

  * Outlier rejection (gating)
  * Low-confidence scaling
  * Heading RMSE ≤ 7°

---

### ✅ RTS Smoother

**Files:** `src/trodestrack/runtime/offline.py`, `src/trodestrack/models/ekf.py`

**Goals:** Implement offline RTS smoother per PRD §12.

**Code Suggestion:**

```python
def rts_smoother(xs, Ps, Fs, Qs):
    n = len(xs)
    x_smooth, P_smooth = xs.copy(), Ps.copy()
    for k in range(n-2, -1, -1):
        Ck = Ps[k] @ Fs[k].T @ np.linalg.inv(Fs[k] @ Ps[k] @ Fs[k].T + Qs[k])
        x_smooth[k] += Ck @ (x_smooth[k+1] - Fs[k] @ xs[k])
        P_smooth[k] += Ck @ (P_smooth[k+1] - Ps[k+1]) @ Ck.T
    return x_smooth, P_smooth
```

**Tests:**

* Assert `P_smooth ≤ P_filter`.
* Assert dropout endpoint drift decreases after smoothing.

---

### ✅ PRD Acceptance Tests (Real Thresholds)

**Files:** `tests/filters/test_prd_bounds.py`, `tests/filters/test_nees_coverage.py`

**Goals:** Replace “truth vs truth” placeholders with actual EKF/UKF validation.

**Code Suggestions:**

```python
# Thresholds in meters/radians
POS_RMSE_MAX = 0.02
VEL_RMSE_MAX = 0.10
HEAD_RMSE_MAX = np.deg2rad(7)

pos_rmse = compute_position_rmse(true_pos, est_pos)
vel_rmse = compute_velocity_rmse(true_vel, est_vel)
head_rmse = compute_heading_rmse(true_heading, est_heading)

assert pos_rmse < POS_RMSE_MAX
assert vel_rmse < VEL_RMSE_MAX
assert head_rmse < HEAD_RMSE_MAX
```

**Fix blackout window:**

```python
in_dropout = (t_cam >= 300.0) & (t_cam < 305.0)
mask_led1[in_dropout] = mask_led2[in_dropout] = False
```

**NEES coverage:**

```python
low, high = chi2.ppf(0.025, 2), chi2.ppf(0.975, 2)
in95 = np.mean((nees >= low) & (nees <= high))
assert in95 >= 0.9
```

---

## 🟠 P1 — Quality and Robustness

### ✅ Metrics Enhancements

**Files:** `src/trodestrack/metrics/metrics.py`

**Goals:** Robust handling of masks, NaNs, and dropout drift.

**Code Suggestions:**

```python
def compute_position_rmse(true, est, mask=None):
    valid = np.isfinite(true).all(axis=1) & np.isfinite(est).all(axis=1)
    if mask is not None: valid &= mask
    err = true[valid] - est[valid]
    return np.sqrt(np.mean(np.sum(err**2, axis=1)))

def chi2_ci95(df: int) -> tuple[float, float]:
    from scipy.stats import chi2
    return float(chi2.ppf(0.025, df)), float(chi2.ppf(0.975, df))

def compute_dropout_drift(pos_cm, valid_mask, t, min_s=5.0):
    bad_blocks = ~valid_mask
    # find first contiguous dropout > min_s
    ...
```

---

### ✅ Simulator Robustness

**Files:** `src/trodestrack/sim/rat_imu.py`, `src/trodestrack/sim/simple.py`

**Goals:** Deterministic, realistic simulation and confidence behavior.

**Code Suggestions:**

```python
# Clamp exposure times
t_exp_clip = np.clip(t_cam_exp, t_imu[0], t_imu[-1])

# Vectorized confidence decay
neighbor_drop = np.convolve(~mask.astype(int), [0.5,1.0,0.5], 'same')
confidence *= np.clip(neighbor_drop, 0, 1)

# Include seed in metadata
return SimOut(..., seed=seed)

# Parametric LED spacing
def simulate_circular(..., led_distance=0.04):
    ...
```

---

### ✅ Visualization Stability & Logging

**Files:** `src/trodestrack/viz/components.py`, `src/trodestrack/viz/video.py`

**Goals:** Stable plots, correct NEES band rendering, structured logging.

**Code Suggestions:**

```python
# NEES band
self.ax.axhspan(self.chi2_lower, self.chi2_upper, color="green", alpha=0.05)

# Confidence clip
conf = np.clip(conf, 0.0, 1.0)

# Ellipse stability
vals, vecs = np.linalg.eigh(P_pos)
vals = np.clip(vals, 0.0, None)

# Replace print with logging
import logging
log = logging.getLogger(__name__)
log.info("Rendering animation...")
```

---

## 🟡 P2 — Cleanup & Refactor

### ✅ DRY & Performance

**Files:** `src/trodestrack/models/*`, `src/trodestrack/metrics/*`

**Goals:** Remove duplication and clarify dtype handling.

**Code Suggestions:**

```python
def build_G_matrix(theta: float, dt: float, lam: float = 0.0):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [0.5 * c * dt**2, -0.5 * s * dt**2],
        [0.5 * s * dt**2,  0.5 * c * dt**2],
        [c * dt, -s * dt],
        [s * dt,  c * dt],
    ])
```

---

# ✅ Review Template (Claude Code)

### CRITICAL CHECKS

1. **PRD Compliance:** Meets relevant PRD section (accuracy, latency, NEES drift).
2. **Test Coverage:** Has failing tests first, edge cases included.
3. **Type Safety:** All signatures have full type hints.

### QUALITY CHECKS

1. Naming clarity
2. Function complexity (<20 lines)
3. Complete NumPy-style docstrings
4. DRY compliance (no duplication)
5. Efficiency (no repeated allocs)

---

### Critical Issues (Must Fix)

* [ ] Missing χ² gating → add NIS-based rejection (`ekf.py`)
* [ ] NEES fill bug → use `axhspan` (`viz/components.py`)
* [ ] PRD acceptance still truth-vs-truth (`test_prd_bounds.py`)

### Quality Issues (Should Fix)

* [ ] Vectorize confidence decay (`sim/rat_imu.py`)
* [ ] Mask support in metrics functions
* [ ] Logging instead of print in video output

### Suggestions (Consider)

* [ ] Dataclass return types for NEES/NIS results
* [ ] Context-managed plotting style (`with apply_tufte_style()`)

### Approved Aspects

* Excellent JAX-style filtering architecture
* Modular simulator with OU motion and dropout realism
* Clear docstrings and reproducible configs
* Strong separation of viz layers and diagnostics

### Final Rating

**REQUEST_CHANGES** — Merge after all critical issues fixed and tests pass.

---

# 🧠 TDD Reminder

1. Write failing test first.
2. Implement fix.
3. Run full checks:

   ```bash
   uv run pytest -q
   uv run mypy src/trodestrack
   uv run ruff check src/ tests/ --fix
   uv run black --check src/ tests/
   ```

4. Update docstrings and `TASKS.md` when done.

**End of File**

# Three Mysteries Solved: Sensor Fusion Analysis

## Mystery #1: Why doesn't Vision-only degrade during dropout?

### ❌ What We Thought
Vision-only filter should perform terribly during dropout (no observations for 5s).

### ✅ What Actually Happens
**Vision-only COMPLETELY FAILS during dropout - but the RMSE metric hides it!**

#### Evidence:
```
Covariance timeline:
  Before dropout (t=24s): 0.000025 m²
  Dropout start (t=25s):  0.527337 m²
  Dropout middle (t=27.5s): 62,381 m² (250m radius!)
  Dropout end (t=30s):    182,527 m² (427m radius!)
  After dropout (t=31s):  NaN (numerical divergence)
```

#### Root Cause:
1. **Vision-only has massive process noise** (vel Q = 2000 vs 2.0 normal)
   - This is intentional to disable IMU influence
   - But during dropout, this causes covariance explosion

2. **Filter diverges to NaN** after dropout
   - Covariance grows from 0.025 m² → 182,000 m² during 5s dropout
   - Recovery update causes numerical instability → NaN

3. **RMSE metric excludes NaN frames** ([metrics.py:59](../src/trodestrack/qa/metrics.py#L59))
   ```python
   valid = np.isfinite(positions_true).all(axis=1) & np.isfinite(positions_est).all(axis=1)
   ```
   - Only 900/1800 frames (50%) are valid for vision-only
   - Fusion: 1800/1800 frames (100%) valid

#### The Misleading Result:
```
Vision-only RMSE: 4.12 cm  ← ONLY computed over 900 valid frames
Fusion RMSE:      8.50 cm  ← Computed over ALL 1800 frames
```

**This is NOT a fair comparison!**

---

## Mystery #2: Why does Vision-only perform better than Fusion?

### ❌ What We Thought
Fusion should always outperform single modalities.

### ✅ What Actually Happens
**Vision-only appears better ONLY because NaN frames are excluded from RMSE!**

#### Fair Comparison (excluding dropout frames):
```
Configuration             | RMSE (all frames) | RMSE (no dropout) | Valid Frames
--------------------------|-------------------|-------------------|-------------
Fusion (Vision + IMU)     | 8.50 cm          | 0.51 cm           | 1800/1800 (100%)
Vision-only (degraded IMU)| 4.12 cm          | 0.51 cm           | 900/1800 (50%)
```

#### Key Insights:
1. **Without dropout, performance is IDENTICAL** (0.51 cm vs 0.51 cm)
   - Both rely primarily on camera observations
   - Vision-only's inflated IMU noise doesn't hurt baseline performance

2. **During dropout:**
   - Fusion: Degrades gracefully to ~20 cm spike, recovers fully
   - Vision-only: Diverges to NaN, never recovers

3. **After dropout:**
   - Fusion: Returns to 0.5 cm error immediately
   - Vision-only: Remains NaN forever (numerical instability)

#### Why Does Fusion Have Higher Overall RMSE?
Because **fusion is averaging over 100% of frames (including dropout)**, while **vision-only only averages over 50% (excludes diverged frames)**.

If we were to impute vision-only's dropout errors as "infinite", the real comparison would be:
```
Fusion:      8.50 cm over 1800 frames
Vision-only: FAILED (NaN on 900 frames, 4.12 cm on remaining 900)
```

---

## Mystery #3: What's wrong with the UKF Smoother?

### ❌ What We See
```python
UKF Smoother: WARNING - Contains NaN values, skipping metrics
```

### ✅ Root Cause (Hypothesis - Needs Investigation)

The UKF smoother likely encounters numerical issues during the backward pass due to:

1. **Sigma-point covariance reconstruction**
   - Backward pass requires inverting predicted covariances
   - If covariance becomes near-singular during dropout, inversion fails
   - Results in NaN propagation through backward recursion

2. **Adaptive Q during dropout**
   - Dropout multiplies Q by 10× → very large covariances
   - Large covariances → potential numerical instability in matrix operations
   - UKF uses different linearization than EKF (sigma points vs Jacobians)

3. **EKF RTS smoother works because:**
   - Uses analytical Jacobians (more stable)
   - Joseph form covariance updates (numerically robust)
   - Explicit symmetrization after each update

#### Recommended Fixes (Not Implemented):
1. Add covariance regularization before inversion
2. Use more robust pseudo-inverse (e.g., SVD-based)
3. Add explicit symmetrization in backward pass
4. Check for numerical issues (determinant, condition number)

---

## Corrected Visualizations Needed

The current visualizations are misleading because they don't show:

1. **Vision-only NaN regions** (should be marked as "FAILED" not just absent)
2. **Valid frame percentages** in legends
3. **Warning labels** when comparing metrics over different frame counts

### Suggested Fixes:

1. **Update RMSE reporting to show:**
   ```
   Fusion RMSE:      8.50 cm (1800/1800 frames)
   Vision-only RMSE: 4.12 cm (900/1800 frames) ⚠️ 50% frames invalid
   ```

2. **Plot NaN regions explicitly:**
   - Show red "X" markers where filter diverged
   - Add text annotations: "Filter Diverged (NaN)"

3. **Add "Valid Frames" subplot** showing:
   - Fusion: 100% green bar
   - Vision-only: 50% green, 50% red

---

## Production Recommendations

### ✅ Use Fusion (not Vision-only)
Even though vision-only shows "4.12 cm", this is misleading:
- Only valid for 50% of frames
- Complete failure during dropout
- No recovery after dropout

Fusion provides:
- 100% valid frames
- Graceful degradation during dropout (~20 cm spike)
- Immediate recovery after dropout
- Better velocity estimates (3× improvement)

### ✅ Use EKF with RTS Smoother
- EKF smoother: 6.11 cm (38% improvement over filter)
- UKF smoother: NaN (numerical issues)

### ❌ Don't Trust Single-Modality Configurations
IMU-only and Vision-only are useful for:
- Ablation studies
- Understanding fusion benefits
- Debugging sensor failures

But they should NEVER be used in production.

---

## Summary Table

| Metric                  | Fusion      | Vision-only | IMU-only  |
|------------------------|-------------|-------------|-----------|
| **RMSE (all)**         | 8.50 cm     | 4.12 cm ⚠️  | 1812 cm   |
| **RMSE (no dropout)**  | 0.51 cm     | 0.51 cm     | N/A       |
| **Valid Frames**       | 100%        | 50% ⚠️      | 100%      |
| **Dropout behavior**   | Degrades ✓  | Diverges ❌ | Diverges ❌|
| **Recovery**           | Immediate ✓ | Never ❌    | Never ❌   |
| **Velocity RMSE**      | 10.21 cm/s  | 30.70 cm/s  | 75.6 cm/s |
| **Production Ready?**  | ✅ YES      | ❌ NO       | ❌ NO     |

⚠️ = Metric is misleading due to excluded NaN frames

---

## Action Items

1. **Update visualization scripts** to show valid frame counts
2. **Add warnings** to RMSE comparisons when frame counts differ
3. **Debug UKF smoother** numerical issues
4. **Document** that vision-only is for ablation only, not production

---

Generated: 2025-10-10
Trodestrack Debug Analysis

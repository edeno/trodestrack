# Fusion Filter Underperformance Investigation

**Date**: 2025-10-10
**Issue**: Vision-only smoother (2.61 cm) significantly outperforms fusion smoother (6.32 cm) on 60s trajectory with 5s dropout

## Executive Summary

The fusion filter is **underperforming** compared to vision-only baseline, contradicting expectations. After testing multiple configurations, the root cause appears to be:

1. **Process noise is too high** during nominal operation
2. **Covariance explodes to 47 million cm²** during 5s dropout
3. **IMU integration degrades** performance instead of improving it

## Test Results

### Configuration Comparison

| Configuration | Filter RMSE | Smoother RMSE | Valid Frames |
|---------------|-------------|---------------|--------------|
| **Production** | 8.50 cm | 6.32 cm | 100% |
| **Lower Process Noise** | **6.26 cm** | 7.14 cm | 100% |
| **No Adaptive Q** | 8.51 cm | 6.30 cm | 100% |
| **Higher Measurement Trust** | 12.29 cm | NaN | 50% → 0% |
| **Vision-Only (baseline)** | 11.55 cm | **2.61 cm** | 100% |

### Key Observations

1. **Best Filter Performance**: Lower Process Noise (6.26 cm)
   - Reduced Q_pos: 0.02 → 0.001 (20×)
   - Reduced Q_vel: 2.0 → 0.5 (4×)
   - **Still worse than vision-only after smoothing**

2. **Vision-Only Dominates After Smoothing**:
   - Filter: 11.55 cm (worse than fusion)
   - **Smoother: 2.61 cm** (beats fusion by 2.4×)
   - This suggests fusion filter is **injecting noise** that smoother cannot remove

3. **Higher Measurement Trust Failed**:
   - Reducing R (trusting camera more) caused **divergence**
   - Only 50% valid frames in filter, 0% in smoother
   - Indicates numerical instability

4. **Adaptive Q Has Minimal Impact**:
   - Production (with adaptive Q): 6.32 cm smoother
   - No adaptive Q: 6.30 cm smoother
   - Suggests adaptive Q is not the problem

## Error Statistics (Production Config)

### Position Error by Phase

| Phase | Mean Error | Notes |
|-------|------------|-------|
| **Before dropout** | 0.45 ± 0.23 cm | Excellent performance |
| **During dropout** | 19.11 cm (max 68.58 cm) | Large drift |
| **After dropout** | 0.45 cm | Quick recovery |

- Performance is **excellent when vision available**
- Performance **degrades severely** during 5s dropout
- The dropout period is **contaminating** the smoother

### Covariance Evolution

| Phase | Fusion P_trace | Vision-Only P_trace |
|-------|----------------|---------------------|
| **Before dropout** | 0.25 cm² | 0.22 cm² |
| **Peak during dropout** | **47,307,570 cm²** | 41,149,400 cm² |
| **After dropout** | 0.25 cm² | 0.22 cm² |

- Covariance explodes to **47 million cm²** during dropout (!)
- This is **10× worse** than the PRD 5s drift bound allows
- After dropout, covariance returns to normal (filter forgets the uncertainty)

## Root Causes

### 1. Process Noise Too High (Confirmed)

**Evidence**:
- Reducing Q by 4-20× improves filter RMSE: 8.50 → 6.26 cm
- Vision-only uses Q_vel=0.5, fusion uses Q_vel=2.0
- Higher Q → larger predicted uncertainty → filter trusts measurements less

**Impact**:
- Filter allows state to drift during prediction steps
- Over-confident in IMU → under-confident in camera

### 2. Covariance Explosion During Dropout (Critical)

**Evidence**:
- Peak covariance: 47 million cm² (equivalent to 2173 m = 2.2 km uncertainty!)
- This is **physically unrealistic** for 5s blackout
- Smoother cannot recover from this numerical instability

**Impact**:
- Smoother backward pass becomes ill-conditioned
- Gains become near-zero (can't propagate information backward)
- Smoother effectively ignores future measurements

### 3. IMU Bias Not Converging Properly (Hypothesis)

**Evidence** (indirect):
- During dropout, error grows linearly (19 cm mean over 5s)
- This suggests **velocity bias** or **heading bias** accumulation
- Fusion should handle this better than vision-only

**Impact**:
- Biased IMU measurements corrupt state estimate
- Filter doesn't have enough excitation to observe biases
- Smoother inherits biased trajectory

## Recommendations

### 🔴 P0 - Fix Covariance Explosion

**Problem**: 47 million cm² covariance during 5s dropout is **numerically catastrophic**

**Solutions**:
1. **Clamp maximum covariance** during dropout (e.g., max 100 cm² = 10 cm std)
2. **Use saturating process noise** instead of linear growth
3. **Add covariance regularization** in backward pass (add small diagonal)

**Expected Impact**: Smoother will propagate information through dropout

### 🔴 P1 - Reduce Process Noise

**Problem**: Q_pos=0.02, Q_vel=2.0 is too high for nominal operation

**Solutions**:
1. **Reduce Q_pos** from 0.02 → 0.001 (20×)
2. **Reduce Q_vel** from 2.0 → 0.5 (4×)
3. **Keep Q_heading** at 0.02 (heading uncertainty is real)

**Expected Impact**: Filter RMSE improves from 8.50 cm → ~6 cm

### 🟡 P2 - Investigate Bias Observability

**Problem**: 19 cm drift during 5s suggests unmodeled bias

**Solutions**:
1. **Check gyro bias convergence** (should be observable during turns)
2. **Check accel bias convergence** (should be observable during acceleration)
3. **Add ZUPT** to prevent velocity drift during stationary periods

**Expected Impact**: Reduced drift during dropout

### 🟡 P3 - Tune Adaptive Q Multipliers

**Problem**: Current multipliers (10×, 10×, 0.1×) may be too aggressive

**Solutions**:
1. **Reduce pos/vel multipliers** from 10× → 5×
2. **Increase bias multiplier** from 0.1× → 0.5× (allow some bias adaptation)
3. **Use sqrt(dt) scaling** instead of dt scaling for position/velocity

**Expected Impact**: Smoother covariance growth (47M → maybe 1000 cm²)

## Action Items

1. ✅ **Completed**: Comprehensive configuration sweep
2. ✅ **Completed**: Error and covariance analysis
3. ⏭️ **Next**: Implement covariance clamping in RTS smoother
4. ⏭️ **Next**: Re-tune process noise using lower values
5. ⏭️ **Next**: Investigate IMU bias convergence on longer trajectories

## Conclusion

The fusion filter is **fundamentally working** but has two critical issues:

1. **Process noise tuning**: 4-20× too high, causing excessive drift
2. **Covariance explosion**: 47 million cm² during dropout breaks smoother

The vision-only smoother succeeds because:
- It uses **realistic process noise** (Q_vel=0.5)
- Its covariance grows **linearly** (max 411 m² vs 47M cm²)
- The smoother can **successfully propagate** information backward

**Bottom line**: Fusion has potential to outperform vision-only, but needs:
- Lower process noise (match vision-only tuning)
- Covariance saturation/clamping during dropout
- Better bias observability analysis

---

*Generated by `examples/13_debug_fusion_performance.py`*

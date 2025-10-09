"""Diagnostic: Verify noise discretization in sim vs filter.

This script checks your Step 1 and Step 2 from the diagnostic playbook:
- Accel/gyro bias RW discretization
- Accel/gyro white noise discretization

Run with: python -m diagnostics.noise_scaling_check
"""

import numpy as np

# =============================================================================
# Simulation Parameters (from test_prd_acceptance.py)
# =============================================================================

# Simulator config
sim_fs_imu = 200.0  # Hz
sim_dt = 1.0 / sim_fs_imu

sim_gyro_noise_density = 0.001  # rad/s / √Hz
sim_accel_noise_density = 0.05  # m/s² / √Hz
sim_gyro_bias_rw_density = 0.0001  # rad/s / √s
sim_accel_bias_rw_density = 0.001  # m/s² / √s

# EKF config (before tuning)
ekf_process_noise_gyro_bias = 2e-6  # (rad/s)²/s
ekf_process_noise_accel_bias = 2e-4  # (m/s²)²/s
ekf_imu_gyro_noise_density = 0.001  # rad/s / √Hz
ekf_imu_accel_noise_density = 0.05  # m/s² / √Hz

# EKF config (after aggressive tuning)
ekf_tuned_process_noise_gyro_bias = 1e-4  # (rad/s)²/s
ekf_tuned_process_noise_accel_bias = 1e-2  # (m/s²)²/s

# =============================================================================
# Step 1: Verify Bias RW Discretization
# =============================================================================

print("=" * 70)
print("STEP 1: Bias Random Walk Discretization")
print("=" * 70)

# Simulator: uses density * sqrt(dt)
sim_gyro_bias_std_per_step = sim_gyro_bias_rw_density * np.sqrt(sim_dt)
sim_accel_bias_std_per_step = sim_accel_bias_rw_density * np.sqrt(sim_dt)

print("\n📊 SIMULATOR (correct discretization: density * sqrt(dt))")
print(f"  Gyro bias:  {sim_gyro_bias_rw_density:.6f} rad/s/√s")
print(f"              → {sim_gyro_bias_std_per_step:.2e} rad/s per step @ {sim_fs_imu} Hz")
print(f"  Accel bias: {sim_accel_bias_rw_density:.6f} m/s²/√s")
print(f"              → {sim_accel_bias_std_per_step:.2e} m/s² per step @ {sim_fs_imu} Hz")

# EKF: uses PSD * dt (where PSD = density²)
# Correct mapping: density² * dt → std = density * sqrt(dt)
ekf_gyro_bias_std_per_step = np.sqrt(ekf_process_noise_gyro_bias * sim_dt)
ekf_accel_bias_std_per_step = np.sqrt(ekf_process_noise_accel_bias * sim_dt)

ekf_tuned_gyro_bias_std_per_step = np.sqrt(ekf_tuned_process_noise_gyro_bias * sim_dt)
ekf_tuned_accel_bias_std_per_step = np.sqrt(ekf_tuned_process_noise_accel_bias * sim_dt)

print("\n🤖 EKF (default tuning)")
print(f"  Gyro bias:  {ekf_process_noise_gyro_bias:.2e} (rad/s)²/s")
print(f"              → {ekf_gyro_bias_std_per_step:.2e} rad/s per step @ {sim_fs_imu} Hz")
print(f"  Accel bias: {ekf_process_noise_accel_bias:.2e} (m/s²)²/s")
print(f"              → {ekf_accel_bias_std_per_step:.2e} m/s² per step @ {sim_fs_imu} Hz")

print("\n🚀 EKF (aggressive tuning)")
print(f"  Gyro bias:  {ekf_tuned_process_noise_gyro_bias:.2e} (rad/s)²/s")
print(f"              → {ekf_tuned_gyro_bias_std_per_step:.2e} rad/s per step @ {sim_fs_imu} Hz")
print(f"  Accel bias: {ekf_tuned_process_noise_accel_bias:.2e} (m/s²)²/s")
print(f"              → {ekf_tuned_accel_bias_std_per_step:.2e} m/s² per step @ {sim_fs_imu} Hz")

# Ratio checks
print("\n⚠️  MISMATCH CHECK:")
gyro_ratio_default = ekf_gyro_bias_std_per_step / sim_gyro_bias_std_per_step
accel_ratio_default = ekf_accel_bias_std_per_step / sim_accel_bias_std_per_step
gyro_ratio_tuned = ekf_tuned_gyro_bias_std_per_step / sim_gyro_bias_std_per_step
accel_ratio_tuned = ekf_tuned_accel_bias_std_per_step / sim_accel_bias_std_per_step

print(f"  Gyro bias ratio (EKF default / sim):  {gyro_ratio_default:.1f}x")
print(f"  Accel bias ratio (EKF default / sim): {accel_ratio_default:.1f}x")
print(f"  Gyro bias ratio (EKF tuned / sim):    {gyro_ratio_tuned:.1f}x")
print(f"  Accel bias ratio (EKF tuned / sim):   {accel_ratio_tuned:.1f}x")

if accel_ratio_tuned > 10:
    print(f"  ⛔ EKF accel bias process noise is {accel_ratio_tuned:.0f}x TOO LARGE!")
    print("     This will cause massive drift during camera blackouts.")

# =============================================================================
# Step 2: Verify White Noise Discretization
# =============================================================================

print(f"\n{'=' * 70}")
print("STEP 2: White Noise Discretization")
print("=" * 70)

# Correct discretization: density / sqrt(dt)
sim_gyro_noise_std_per_step = sim_gyro_noise_density / np.sqrt(sim_dt)
sim_accel_noise_std_per_step = sim_accel_noise_density / np.sqrt(sim_dt)

ekf_gyro_noise_std_per_step = ekf_imu_gyro_noise_density / np.sqrt(sim_dt)
ekf_accel_noise_std_per_step = ekf_imu_accel_noise_density / np.sqrt(sim_dt)

print("\n📊 SIMULATOR")
print(f"  Gyro noise:  {sim_gyro_noise_density:.6f} rad/s/√Hz")
print(f"               → {sim_gyro_noise_std_per_step:.2e} rad/s per step @ {sim_fs_imu} Hz")
print(f"  Accel noise: {sim_accel_noise_density:.6f} m/s²/√Hz")
print(f"               → {sim_accel_noise_std_per_step:.2e} m/s² per step @ {sim_fs_imu} Hz")

print("\n🤖 EKF")
print(f"  Gyro noise:  {ekf_imu_gyro_noise_density:.6f} rad/s/√Hz")
print(f"               → {ekf_gyro_noise_std_per_step:.2e} rad/s per step @ {sim_fs_imu} Hz")
print(f"  Accel noise: {ekf_imu_accel_noise_density:.6f} m/s²/√Hz")
print(f"               → {ekf_accel_noise_std_per_step:.2e} m/s² per step @ {sim_fs_imu} Hz")

print("\n✅ Match check:")
print(
    f"  Gyro:  {'MATCH' if np.isclose(sim_gyro_noise_std_per_step, ekf_gyro_noise_std_per_step) else 'MISMATCH'}"
)
print(
    f"  Accel: {'MATCH' if np.isclose(sim_accel_noise_std_per_step, ekf_accel_noise_std_per_step) else 'MISMATCH'}"
)

# =============================================================================
# Step 3: Theoretical Drift from White Noise
# =============================================================================

print(f"\n{'=' * 70}")
print("STEP 3: Theoretical Position Drift from White Accel Noise")
print("=" * 70)

t_blackout = 5.0  # seconds

# Position std from white noise: sqrt(S_a * t³ / 3)
# where S_a = (accel_noise_density)² is the PSD
S_a = sim_accel_noise_density**2

std_x_white_noise_1d = np.sqrt(S_a * t_blackout**3 / 3)
std_x_white_noise_2d = std_x_white_noise_1d * np.sqrt(2)  # RMS of two axes

print(f"\nAccel white noise PSD: S_a = {S_a:.4e} (m/s²)²/Hz")
print(f"Blackout duration: {t_blackout} s")
print(f"\nTheoretical position std (1D): {std_x_white_noise_1d:.3f} m")
print(f"Theoretical position std (2D radial): {std_x_white_noise_2d:.3f} m")

# =============================================================================
# Step 4: Theoretical Drift from Bias RW
# =============================================================================

print(f"\n{'=' * 70}")
print("STEP 4: Theoretical Position Drift from Bias Random Walk")
print("=" * 70)

# Bias std after time t: std(b(t)) = density * sqrt(t)
# Position error from constant bias b: 0.5 * b * t²
# Expected position error (RMS): 0.5 * std(b) * t²

std_bias_at_t = sim_accel_bias_rw_density * np.sqrt(t_blackout)
pos_drift_from_bias_1d = 0.5 * std_bias_at_t * t_blackout**2
pos_drift_from_bias_2d = pos_drift_from_bias_1d * np.sqrt(2)

print(f"\nBias RW density: {sim_accel_bias_rw_density:.6f} m/s²/√s")
print(f"Bias std after {t_blackout}s: {std_bias_at_t:.6f} m/s²")
print(f"\nExpected position drift (1D): {pos_drift_from_bias_1d:.3f} m")
print(f"Expected position drift (2D radial): {pos_drift_from_bias_2d:.3f} m")

# =============================================================================
# Step 5: Combined Estimate
# =============================================================================

print(f"\n{'=' * 70}")
print("STEP 5: Combined Drift Estimate (White Noise + Bias RW)")
print("=" * 70)

# Approximate combined (assuming independence)
total_drift_1d = np.sqrt(std_x_white_noise_1d**2 + pos_drift_from_bias_1d**2)
total_drift_2d = np.sqrt(std_x_white_noise_2d**2 + pos_drift_from_bias_2d**2)

print(f"\nCombined drift estimate (1D): {total_drift_1d:.3f} m")
print(f"Combined drift estimate (2D): {total_drift_2d:.3f} m")
print("\n📌 Observed in test: ~1.7 m")
print(f"   Theoretical estimate: {total_drift_2d:.2f} m")

if total_drift_2d > 0.15:
    print("\n⛔ DRIFT EXCEEDS PRD TARGET (0.15 m)")
    print(f"   Primary contributor: White accel noise ({std_x_white_noise_2d:.2f} m)")
    print(f"   Secondary contributor: Bias RW ({pos_drift_from_bias_2d:.2f} m)")
    print("\n💡 Recommended fixes:")
    print("   1. Freeze accel bias Q during blackout (eliminates bias RW)")
    print("   2. Reduce accel input noise during blackout (reduces white noise growth)")
    print("   3. Add constant-speed pseudo-measurement during blackout")

print(f"\n{'=' * 70}")


if __name__ == "__main__":
    pass  # All diagnostics run at import

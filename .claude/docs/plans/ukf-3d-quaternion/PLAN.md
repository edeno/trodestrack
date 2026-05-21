# UKF 3D Quaternion Support — Implementation Plan

**Status:** Not started.

**Goal:** Add 3D quaternion-orientation support to `unscented_kalman_filter` and the sigma-point smoother, achieving feature parity with `extended_kalman_filter_3d`. The 5 rejection-contract tests in `tests/filters/test_ukf_3d_analytic.py` become real analytic tests.

**Architecture:** **Markley eigenvector method** keeps the existing 16D `LAYOUT_3D_CAM_6DOF_IMU` state layout unchanged. Quaternion-aware sigma-point mean reconstruction = principal eigenvector of `M = Σ wᵢ qᵢ qᵢᵀ` (a 4×4 weighted-outer-product matrix), with sign disambiguation against the previous mean to suppress ±q double-cover ambiguity. Quaternion-block covariance uses 4D deviations of sign-aligned sigma points plus adaptive jitter to handle the rank-3 nature of the quaternion tangent space embedded in R⁴. This mirrors the existing 2D heading circular-mean special-case at [src/trodestrack/models/ukf.py:450-456](../../../../src/trodestrack/models/ukf.py#L450-L456) — same pattern, different math.

**Tech stack:** JAX (jnp.linalg.eigh for the 4×4 eigendecomposition, vmap for sigma-point propagation), existing `normalize_quaternion` helper, existing `LAYOUT_3D_CAM_6DOF_IMU`. No new dependencies.

**Out of scope:**
- USQUE / error-state representation (15D state with separate reference quaternion). Markley is sufficient for this codebase's accuracy and stability needs; revisit only if Markley produces numerical issues on real recordings.
- 3D heading_pseudo measurement (the heading pseudo-measurement is a 2D-only construct; 3D orientation is observed via gravity_orientation_update on the accelerometer).
- New 3D layouts beyond `LAYOUT_3D_CAM_6DOF_IMU` (e.g. 3D Euler).
- UKF performance optimization beyond the natural Markley cost (~10μs/step for the 4×4 eigendecomposition).

## Inputs to read first

- [src/trodestrack/models/ukf.py:1039-1045](../../../../src/trodestrack/models/ukf.py#L1039-L1045) — the `NotImplementedError` to remove.
- [src/trodestrack/models/ukf.py:425-486](../../../../src/trodestrack/models/ukf.py#L425-L486) — `predict_step`. Lines 450-456 are the 2D heading circular-mean pattern that the quaternion path mirrors structurally.
- [src/trodestrack/models/ukf.py:494-610](../../../../src/trodestrack/models/ukf.py#L494-L610) — `update_step` (camera). The same arithmetic-mean code path needs quaternion-awareness when the layout is quaternion-orientation.
- [src/trodestrack/models/ukf.py:619-700](../../../../src/trodestrack/models/ukf.py#L619-L700) — `update_heading`. Verify this is unreachable for quaternion layouts (heading_pseudo is 2D-only); if so, gate it.
- [src/trodestrack/models/state_layout.py:90-118](../../../../src/trodestrack/models/state_layout.py#L90-L118) — `has_quaternion_orientation` property and `get_heading_index` which currently raises `NotImplementedError` for quaternion layouts. Several call sites in `ukf.py` and `runtime/offline.py` need to branch on `has_quaternion_orientation` before calling `get_heading_index`.
- [src/trodestrack/models/filter_common.py:1190-1210](../../../../src/trodestrack/models/filter_common.py#L1190-L1210) — existing `normalize_state_orientation(state, quat_idx)` helper. Use to ensure propagated sigma points stay on S³.
- [src/trodestrack/models/filter_common.py:20](../../../../src/trodestrack/models/filter_common.py#L20) — `normalize_quaternion` import.
- [src/trodestrack/runtime/offline.py:579-700](../../../../src/trodestrack/runtime/offline.py#L579-L700) — `_sigma_point_smoother_impl`. Uses `get_heading_index(layout)` at line ~637; will raise on quaternion layouts. Same Markley reconstruction is required.
- [src/trodestrack/models/ekf.py:1450+](../../../../src/trodestrack/models/ekf.py) — `extended_kalman_filter_3d` for parity reference. Read its full implementation; the UKF must produce numerically comparable outputs (~20% looser bounds documented in the test docstrings).
- [tests/filters/test_ekf_3d_analytic.py](../../../../tests/filters/test_ekf_3d_analytic.py) — the EKF analytic tests; the UKF tests will mirror this structure with looser bounds.
- [tests/filters/test_ukf_3d_analytic.py](../../../../tests/filters/test_ukf_3d_analytic.py) — currently a single rejection-contract test plus a module docstring listing the future bounds. The module docstring is the source of truth for the analytic tolerances.
- [tests/filters/conftest.py](../../../../tests/filters/conftest.py) — `simulate_3d_session` fixture used by both EKF and UKF 3D tests.

## Tasks

### Task 1 — Quaternion-aware sigma helpers (new module `src/trodestrack/models/quaternion_sigma.py`)

Three pure functions, all JAX-compatible (tracable, no Python control flow on tracers):

```python
# src/trodestrack/models/quaternion_sigma.py
"""Quaternion-aware sigma-point mean and covariance helpers.

Implements the Markley eigenvector method for computing the weighted
mean of unit quaternions, used by the UKF when the state layout has
quaternion orientation (4D heading_idx).

References
----------
Markley, F. L., Cheng, Y., Crassidis, J. L., & Oshman, Y. (2007).
"Averaging Quaternions." Journal of Guidance, Control, and Dynamics,
30(4), 1193-1196. https://doi.org/10.2514/1.28949
"""

from __future__ import annotations

import jax.numpy as jnp
from trodestrack.models.filter_common import normalize_quaternion


def quaternion_principal_eigvec_mean(
    sigma_quats: jnp.ndarray,
    w_mean: jnp.ndarray,
    q_ref: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted mean of unit quaternions via Markley's eigenvector method.

    Computes the principal eigenvector of M = Σ wᵢ qᵢ qᵢᵀ (a 4×4
    matrix), normalises it, and sign-aligns to ``q_ref`` to suppress
    the ±q double-cover ambiguity inherent to unit quaternions.

    Parameters
    ----------
    sigma_quats : jnp.ndarray
        Sigma-point quaternion components ``(2n+1, 4)`` after dynamics
        propagation. **Must be unit-norm** (caller normalises).
    w_mean : jnp.ndarray
        Sigma-point mean weights ``(2n+1,)``. May contain negative
        values; the eigenvector method handles this correctly because
        ``qᵢ qᵢᵀ`` is rank-1 PSD per sigma and the weighted sum
        remains symmetric.
    q_ref : jnp.ndarray
        Previous mean quaternion ``(4,)``, used for sign alignment.

    Returns
    -------
    jnp.ndarray, shape (4,)
        Unit-norm mean quaternion. Sign is chosen so that
        ``q_mean · q_ref >= 0``.
    """
    # Outer-product accumulation: M[i, j] = Σₖ wₖ qₖᵢ qₖⱼ
    M = jnp.einsum("k,ki,kj->ij", w_mean, sigma_quats, sigma_quats)
    # M is symmetric by construction; eigh returns ascending eigenvalues
    eigvals, eigvecs = jnp.linalg.eigh(M)
    # Principal eigenvector is the last column (largest eigenvalue)
    q_mean_unsigned = eigvecs[:, -1]
    # Sign-align to q_ref. dot < 0 ⇒ flip sign.
    sign = jnp.sign(jnp.dot(q_mean_unsigned, q_ref))
    # jnp.sign returns 0 for exact zero — fall back to +1 to keep the result deterministic
    sign = jnp.where(sign == 0.0, 1.0, sign)
    q_mean = q_mean_unsigned * sign
    # Numerical insurance — eigvec is unit-norm to machine precision but normalise to be safe
    return normalize_quaternion(q_mean)


def align_quaternion_sign(q: jnp.ndarray, q_ref: jnp.ndarray) -> jnp.ndarray:
    """Return ``q`` or ``-q`` to maximize ``q · q_ref``.

    Used per-sigma-point before computing deviations for the covariance
    reconstruction, so the deviation 4-vectors are small (closer to the
    tangent space at ``q_ref``) rather than ~±2 due to sign flips.
    """
    dot = jnp.dot(q, q_ref)
    sign = jnp.where(dot >= 0.0, 1.0, -1.0)
    return q * sign


def quaternion_sigma_deviations(
    sigma_quats: jnp.ndarray, q_mean: jnp.ndarray
) -> jnp.ndarray:
    """Per-sigma quaternion deviations with sign disambiguation.

    Each sigma quaternion is replaced with ``±qᵢ`` (whichever sign
    maximizes ``qᵢ · q_mean``) before subtracting ``q_mean``. Without
    this step, a sigma point on the antipodal side of the unit sphere
    contributes a ~±2 deviation that inflates the covariance.

    Parameters
    ----------
    sigma_quats : jnp.ndarray, shape (2n+1, 4)
        Unit-norm sigma quaternions.
    q_mean : jnp.ndarray, shape (4,)
        Mean quaternion (Markley eigvec output).

    Returns
    -------
    jnp.ndarray, shape (2n+1, 4)
        Deviations ``±qᵢ - q_mean``. The 4D deviations sit in a
        rank-3 subspace of R⁴; downstream covariance reconstruction
        must add jitter to handle the rank deficiency.
    """
    # vmap-friendly sign alignment: for each row, compute the sign
    aligned = jnp.einsum(
        "k,ki->ki",
        jnp.sign(jnp.einsum("ki,i->k", sigma_quats, q_mean) + 1e-30),
        sigma_quats,
    )
    return aligned - q_mean
```

**Subtasks for Task 1:**
1a. Create the file with the three functions above.
1b. Add `tests/models/test_quaternion_sigma.py` (unit tests, not slow):
   - `test_quaternion_principal_eigvec_mean_recovers_identity` — 5 sigmas all at identity; mean = identity.
   - `test_quaternion_principal_eigvec_mean_handles_sign_flip` — feed `[q, -q, q]` with positive weights; mean = q (not zero).
   - `test_quaternion_principal_eigvec_mean_recovers_small_rotation` — 5 sigmas symmetric around a rotation-by-30°; mean ≈ 30° rotation within 1e-6.
   - `test_align_quaternion_sign_flips_antipodal` — antipodal input returns negated.
   - `test_quaternion_sigma_deviations_zero_at_mean` — sigmas all equal to mean ⇒ deviations all zero.
   - `test_quaternion_sigma_deviations_small_for_nearby_points` — small rotations ⇒ deviations of order the rotation magnitude.

### Task 2 — Wire forward UKF predict_step

Modify [src/trodestrack/models/ukf.py:425-486](../../../../src/trodestrack/models/ukf.py#L425-L486):

After `sigmas_prop = vmap(f)(sigmas)` (line 446) — the propagated sigma points may have quaternion components that drifted off the unit sphere due to first-order dynamics integration. Apply `normalize_state_orientation` per sigma if the layout has quaternion orientation:

```python
if layout.has_quaternion_orientation:
    sigmas_prop = vmap(
        lambda s: normalize_state_orientation(s, layout.heading_idx)
    )(sigmas_prop)
```

Replace the heading circular-mean block (lines 450-456) with a generic dispatch:

```python
m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)

if layout.has_quaternion_orientation:
    # Quaternion mean via Markley eigenvector method
    quat_idx = jnp.asarray(layout.heading_idx)  # (4,) tuple → array
    sigma_quats = sigmas_prop[:, list(layout.heading_idx)]  # (2n+1, 4)
    q_ref = m[list(layout.heading_idx)]  # use previous mean as reference for sign
    q_mean = quaternion_principal_eigvec_mean(sigma_quats, w_mean, q_ref)
    for i, idx in enumerate(layout.heading_idx):
        m_pred = m_pred.at[idx].set(q_mean[i])
elif isinstance(layout.heading_idx, int):
    # Existing 2D heading circular mean
    h_idx = layout.heading_idx
    sigmas_heading_prop = sigmas_prop[:, h_idx]
    sin_weighted = w_mean @ jnp.sin(sigmas_heading_prop)
    cos_weighted = w_mean @ jnp.cos(sigmas_heading_prop)
    m_pred = m_pred.at[h_idx].set(jnp.arctan2(sin_weighted, cos_weighted))
# (Else: 3D Euler — not supported in UKF either; out of scope.)
```

Replace the heading-wrap covariance block (lines 460-465):

```python
deviations = sigmas_prop - m_pred

if layout.has_quaternion_orientation:
    # Quaternion deviations with sign alignment
    quat_sigma = sigmas_prop[:, list(layout.heading_idx)]
    q_mean_value = m_pred[list(layout.heading_idx)]
    quat_devs = quaternion_sigma_deviations(quat_sigma, q_mean_value)
    for i, idx in enumerate(layout.heading_idx):
        deviations = deviations.at[:, idx].set(quat_devs[:, i])
elif isinstance(layout.heading_idx, int):
    # Existing 2D heading wrap
    h_idx = layout.heading_idx
    deviations = deviations.at[:, h_idx].set(wrap_angle(deviations[:, h_idx]))
```

After computing `P_pred`, add quaternion-block jitter (because the 4D quaternion covariance has rank 3 — one direction is ~normal-to-S³ and has zero variance):

```python
if layout.has_quaternion_orientation:
    # Rank-3 quaternion block needs jitter for Cholesky stability downstream
    quat_jitter = 1e-9
    for idx in layout.heading_idx:
        P_pred = P_pred.at[idx, idx].add(quat_jitter)
```

### Task 3 — Wire UKF update steps

For each update function in `ukf.py`, audit whether it touches the quaternion block (directly via `get_heading_index` or indirectly via arithmetic-mean reconstruction over sigma points).

3a. **`update_step` (camera) at lines 494-610**: Camera 3D measurement returns 6 values per frame (3D LED1 + 3D LED2). The update reconstructs predicted measurement mean via `sigmas_z = vmap(camera_model.predict)(sigmas)` then `z_pred = sum(w_i * sigmas_z[i])`. This is fine — measurement space is 6D Euclidean, no quaternion handling needed. **BUT** the cross-covariance `P_xz = sum(w_i * (sigmas[i] - m) * (sigmas_z[i] - z_pred).T)` uses state-space deviations which must use the quaternion-aware deviation when `has_quaternion_orientation`. Apply the same Task-2 pattern to compute deviations.

3b. **`update_heading` at lines 619-700**: heading_pseudo is 2D-only. For quaternion layouts, the UKF caller (`_unscented_kalman_filter_impl`) should skip this update. Add an early return or assertion when `layout.has_quaternion_orientation`:
```python
if layout.has_quaternion_orientation:
    # heading_pseudo is a 2D-only measurement; no-op for quaternion layouts.
    return state, jnp.asarray(0.0, dtype=state.mean.dtype)
```

3c. **`update_zupt`**: doesn't touch quaternion components (ZUPT measurement is velocity = 0). Verify by reading. Should work as-is.

3d. **`update_event_location`**: 2D position update; doesn't touch quaternion components. Verify.

3e. **`update_gravity_orientation`** (if it exists for UKF; check by grep): DOES update the quaternion block via the accelerometer-vs-gravity residual. Must apply Task-2 mean/deviation patterns. If the UKF version doesn't exist yet (only the EKF has it), this becomes a sub-task to port the EKF implementation to UKF using sigma points.

3f. **`compute_log_likelihood` and Mahalanobis gating**: layout-agnostic (operates on innovations in measurement space). Should work.

### Task 4 — Remove the NotImplementedError gate

[src/trodestrack/models/ukf.py:1039-1045](../../../../src/trodestrack/models/ukf.py#L1039-L1045): delete the `if layout.has_quaternion_orientation: raise NotImplementedError(...)` block.

[src/trodestrack/models/ukf.py:1002-1006](../../../../src/trodestrack/models/ukf.py#L1002-L1006): update the docstring to remove the "UKF does not accept quaternion-orientation layouts" disclaimer. Document the new K=6 IMU shape for quaternion layouts.

[src/trodestrack/models/filter_common.py::validate_imu_input_shape](../../../../src/trodestrack/models/filter_common.py): verify it already accepts K=6 for quaternion layouts (the EKF uses this path); if it gates on `func_name == "extended_kalman_filter_3d"`, generalize so `unscented_kalman_filter` also accepts K=6.

### Task 5 — Wire sigma-point smoother

[src/trodestrack/runtime/offline.py:579-700+](../../../../src/trodestrack/runtime/offline.py#L579-L700): the smoother's `propagate_one_imu` and `smoother_step` use:
- Line ~637: `h_idx = get_heading_index(layout)` — currently raises on quaternion layouts.
- Line ~663: `m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)` — arithmetic mean across all state components.

Mirror the Task-2 changes:
- Replace `h_idx = get_heading_index(layout)` with a branch on `layout.has_quaternion_orientation`. Skip heading-circular-mean code, run quaternion-aware mean reconstruction instead.
- Apply per-sigma quaternion normalization, Markley eigvec, sign-aligned deviations.

The cross-covariance computation in the smoother (the `S_cross = P(x_k, x_{k+1})` term that links filtered[k] to predicted[k+1]) also needs quaternion-aware state-space deviations. Same pattern.

### Task 6 — Convert rejection-contract test to 5 analytic tests

[tests/filters/test_ukf_3d_analytic.py](../../../../tests/filters/test_ukf_3d_analytic.py) currently has one rejection-contract test plus a module docstring listing 5 future analytic bounds:
- stationary pitch=10° roll=5°: pitch/roll within ~2.4° of truth (~20% looser than EKF's 2°), quaternion norm 1 ± ~1.2e-6
- yaw-only 30°/s with gyro_z bias 0.05 rad/s: bias RMSE within ~6e-3 rad/s
- 5s camera dropout: drift ≤ ~0.18 m
- 4D NEES consistency: mean NEES in [1.0, 9.6]
- perfect-input idempotence: posterior matches truth with rtol≈1.2e-5

Rewrite the file with 5 `@pytest.mark.slow` analytic tests mirroring `tests/filters/test_ekf_3d_analytic.py` 1:1, using the documented bounds. Delete the `test_ukf_3d_quaternion_modes_raise_not_implemented` test — the rejection contract is gone.

Reuse `simulate_3d` fixture from `tests/filters/conftest.py`.

Add a `tests/runtime/test_smoother_layout_smoke.py` test for the 3D UKF smoother — pattern after the existing 3D EKF smoother smoke test in the same file. If that test already exists for the smoother, simply add a UKF variant.

### Task 7 — Documentation + CHANGELOG

7a. Update `unscented_kalman_filter` docstring at [src/trodestrack/models/ukf.py:961+](../../../../src/trodestrack/models/ukf.py#L961) to document quaternion-layout support.

7b. Update `LAYOUT_3D_CAM_6DOF_IMU` docstring at [src/trodestrack/models/state_layout.py:283-291](../../../../src/trodestrack/models/state_layout.py#L283-L291) to note UKF support (the previous version mentioned only the EKF entry point).

7c. CHANGELOG entry — add under a new `## [0.3.0] - unreleased` section (this is a feature addition, not a bug fix; minor version bump):

```
### Added
- `unscented_kalman_filter` now supports 3D quaternion-orientation state modes
  (`3d_cam_6dof_imu`). The quaternion sigma-point mean is reconstructed via
  Markley's eigenvector method (principal eigenvector of Σ wᵢ qᵢ qᵢᵀ with
  sign disambiguation against the previous mean) and the 4D quaternion-block
  covariance is reconstructed from sign-aligned sigma-point deviations with
  adaptive jitter for the rank-3 quaternion tangent space.
- `sigma_point_smoother` (RTS-style backward pass for the UKF) now supports
  the same quaternion-orientation layouts.

### Internal
- New helper module `models/quaternion_sigma.py` (3 pure functions:
  `quaternion_principal_eigvec_mean`, `align_quaternion_sign`,
  `quaternion_sigma_deviations`). Reused by both the forward UKF and the
  sigma-point smoother.
```

## Deliberately not in this plan

- **USQUE / 3-vector error-state representation**: would require restructuring the state layout (16D → 15D with separate reference quaternion). Revisit only if Markley produces numerical issues on real recordings — its rank-3 covariance handling via adaptive jitter is sufficient for the test bounds documented in `tests/filters/test_ukf_3d_analytic.py`.
- **3D Euler orientation in the UKF**: out of scope. Only quaternion 3D orientation (`LAYOUT_3D_CAM_6DOF_IMU`) is supported.
- **New 3D layouts** beyond `LAYOUT_3D_CAM_6DOF_IMU`.
- **Performance optimization** beyond Markley's natural cost. The 4×4 eigendecomposition is O(1) per step regardless of state dimension. If profiling reveals it becomes a bottleneck, consider the closed-form 4×4 eigenvector formula (Cardano-style) as a follow-up.
- **2D-orientation→quaternion migration**: keep 2D layouts using scalar heading (heading_idx: int). Mixing the two would be a separate cross-cutting refactor.
- **The `2d_cam_6dof_imu_orientation` 14D layout**: this is also flagged for quaternion-orientation rejection in the current UKF, but it predates `3d_cam_6dof_imu` and may or may not be the intended target. Verify with the user before extending to it — if yes, add as a follow-up phase; if no, document its deprecation.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_quaternion_principal_eigvec_mean_recovers_identity` | 5 sigmas all at identity quaternion → mean = identity, norm 1 ± 1e-12 |
| `test_quaternion_principal_eigvec_mean_handles_sign_flip` | Mix of `q` and `-q` sigmas → mean = q (not zero), norm 1 ± 1e-12 |
| `test_quaternion_principal_eigvec_mean_recovers_small_rotation` | Symmetric sigmas around a 30° z-axis rotation → mean within 1e-6 of truth |
| `test_align_quaternion_sign_flips_antipodal` | `align(-q, q) == q`; `align(q, q) == q` |
| `test_quaternion_sigma_deviations_zero_at_mean` | All sigmas = mean ⇒ all deviations zero |
| `test_quaternion_sigma_deviations_small_for_nearby_points` | Small perturbations ⇒ deviations of order the perturbation |
| `test_ukf_3d_stationary_pitch_roll_recovers_gravity_orientation` (`@pytest.mark.slow`) | Stationary pitch=10° roll=5° → pitch/roll within 2.4° of truth, quaternion norm 1 ± 1.2e-6 |
| `test_ukf_3d_yaw_only_motion_converges_gyro_bias_z` (`@pytest.mark.slow`) | Yaw-only 30°/s with true gyro_z bias 0.05 rad/s → estimated bias within 6e-3 rad/s of truth at t=10s |
| `test_ukf_3d_5s_dropout_drift_under_acceptance_target` (`@pytest.mark.slow`) | Position drift during 5s blackout ≤ 0.18 m |
| `test_ukf_3d_nees_consistency_on_4d_state` (`@pytest.mark.slow`) | Mean NEES on (x, y, z, yaw) in [1.0, 9.6] |
| `test_ukf_3d_perfect_input_no_drift` (`@pytest.mark.slow`) | Perfect-input posterior matches truth with rtol=1.2e-5 |
| `test_sigma_point_smoother_3d_quaternion_smoke` (`@pytest.mark.slow`) | Sigma-point smoother runs end-to-end on 3D quaternion data; outputs are finite, quaternions unit-norm, smoothed RMSE ≤ filtered RMSE on a known-stationary signal |
| `test_ukf_3d_smoother_layout_matches_ekf_within_factor_2` (`@pytest.mark.slow`) | UKF 3D smoother RMSE on `simulate_3d_session(motion="yaw_only")` is within 2× of the EKF 3D smoother on the same session |

## Fixtures

- **Existing**: `simulate_3d` fixture in `tests/filters/conftest.py` (added in Phase 2) — reused unchanged for the 5 analytic tests + the smoother smoke test.
- **New (in `tests/models/test_quaternion_sigma.py`)**: small in-test fixtures (5 sigma points, weights) constructed inline. No file-based fixtures needed.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| **Sigma points drift off S³ after dynamics** — the existing dynamics function may not renormalize per-sigma, especially since it was designed for first-order EKF updates. | Task 2 applies `normalize_state_orientation` per sigma point after `vmap(f)(sigmas)`. Verify by inspecting `dynamics_function` for an explicit `normalize_quaternion` call; if absent, the per-sigma normalize in Task 2 is the load-bearing step. |
| **Rank-3 quaternion covariance breaks Cholesky in `compute_sigma_points`** — the next predict step's sigma generation factors `P` via Cholesky; a rank-deficient quaternion block can fail. | Task 2 adds adaptive jitter (`1e-9` on diagonals of the quaternion block). The existing `adaptive_diagonal_boost` in `compute_sigma_points` provides a second line of defense if `1e-9` is insufficient. |
| **`q_ref` (previous mean) is zero on the very first step** — when `initial_state` is `None` and auto-initialized with all-zeros for the quaternion block, the sign alignment becomes undefined. | The auto-init code path for quaternion layouts must initialize the quaternion block to `[1, 0, 0, 0]` (identity), not zeros. Verify the EKF's `initialize_state` for `3d_cam_6dof_imu` and reuse the same convention. |
| **Markley's eigenvector method is numerically unstable when the principal eigenvalue is near-degenerate with the second** — could happen for sigma points spread across the unit sphere (large initial covariance). | This is unlikely in the normal operating regime (UKF sigma points sit near the mean by construction); document as a known caveat. If it surfaces in tests, escape hatch is to fall back to USQUE (out of scope for this plan). |
| **Sign-disambiguation breaks under exact 90° rotations** — `q · q_ref = 0` exactly. | `quaternion_principal_eigvec_mean` already handles `jnp.sign(0) == 0` by falling back to `+1`. Verify via the parametrized test cases (rotation angles near 90° should pass). |
| **Tests are slow** — adding 5 slow tests to the UKF 3D suite + 2 slow smoother tests = ~120-150s additional CI time. | All new tests `@pytest.mark.slow`; default `-m "not slow"` excludes them. CI's slow-test job picks them up. |
| **`update_gravity_orientation` may not exist for the UKF yet** — only the EKF has it today. | Audit during Task 3e. If missing, port the EKF version using sigma points + the new quaternion-aware mean/deviation helpers. If too large for this PR, defer to a follow-up phase and document the gap. |

## Rollout Strategy

Single PR, single `0.3.0` minor version bump. The change is purely additive: no existing UKF behavior changes, no APIs renamed, no deprecations. Users with 2D UKF code see no diff. Users with 3D code currently using the EKF gain the option to use the UKF.

## Review

Before opening the PR, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task is implemented as specified; nothing in "Deliberately not in this plan" was added.
- The `NotImplementedError` gate at `ukf.py:1039` and any sibling gate in the smoother are both removed.
- Quaternion mean (Markley eigvec) and deviation (sign-aligned) helpers are JAX-traceable — no Python control flow on tracers.
- All 5 UKF 3D analytic tests pass with the documented bounds.
- All EKF analytic tests still pass (no regression — the EKF code path is untouched).
- 2D UKF tests still pass (the new branches only fire when `layout.has_quaternion_orientation`).
- Quaternion norm stays at 1 ± 1.2e-6 throughout the 30s test sessions (the test's primary regression-of-numerical-discipline guard).
- The smoother smoke test confirms smoothed RMSE ≤ filtered RMSE (otherwise the smoother is mis-wired).
- Tests aren't trivial — they exercise the new behavior, not just function-runs-without-error.
- Docstrings, test names, and module names don't reference this plan.
- CHANGELOG `[0.3.0]` entry is well-formed and the compare-link footer is updated.

## Estimated Effort

| Section | LOC (rough) |
| --- | --- |
| `models/quaternion_sigma.py` (new) | +120 |
| `tests/models/test_quaternion_sigma.py` (new) | +150 |
| `models/ukf.py` (predict_step + update_step + remove NIE + docstring) | +80 / -10 |
| `runtime/offline.py` (sigma_point_smoother quaternion branches) | +60 |
| `models/filter_common.py` (validate_imu_input_shape K=6 generalization, if needed) | +5 |
| `tests/filters/test_ukf_3d_analytic.py` (rewrite from rejection-contract to analytic) | +400 / -70 |
| `tests/runtime/test_smoother_layout_smoke.py` (add 3D UKF smoother smoke) | +60 |
| `CHANGELOG.md` | +15 |

Total: roughly +890 / -80 LOC across ~6 src files and ~3 test files. Substantial but coherent for a single feature PR.

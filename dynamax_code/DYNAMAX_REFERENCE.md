# Dynamax Nonlinear Gaussian SSM Reference

This document summarizes the dynamax implementations of Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for reference when implementing `trodestrack` filters.

## Overview

The dynamax library implements nonlinear Gaussian state-space models (NLGSSM) with the following structure:

```
p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)  # Dynamics
p(y_t | z_t, u_t)     = N(y_t | h(z_t, u_t), R_t)      # Emissions
p(z_1)                = N(z_1 | m, S)                    # Initial
```

**Key Files:**

- `inference_ekf.py` - Extended Kalman Filter and RTS Smoother
- `inference_ukf.py` - Unscented Kalman Filter and Smoother
- `models.py` - Parameter structure (`ParamsNLGSSM`)
- `sarkka_lib.py` - Reference implementations from Särkkä (2013)
- `inference_test_utils.py` - Testing utilities

---

## Data Structures

### ParamsNLGSSM (models.py)

```python
class ParamsNLGSSM(NamedTuple):
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_function: Callable  # f(state, input?) -> state
    dynamics_covariance: Float[Array, "state_dim state_dim"]  # Q
    emission_function: Callable  # h(state, input?) -> observation
    emission_covariance: Float[Array, "emission_dim emission_dim"]  # R
```

**Key Points:**

- Functions can be `f(state)` or `f(state, input)` - dynamax auto-detects
- Covariances can be time-varying: `Q.shape = (T, D, D)` or static: `Q.shape = (D, D)`
- Uses `jax.jacfwd` for Jacobians (forward-mode autodiff)

### Posterior Objects

```python
PosteriorGSSMFiltered(
    marginal_loglik: float
    filtered_means: Array[T, state_dim]
    filtered_covariances: Array[T, state_dim, state_dim]
    predicted_means: Array[T, state_dim]
    predicted_covariances: Array[T, state_dim, state_dim]
)

PosteriorGSSMSmoothed(
    marginal_loglik: float
    filtered_means: Array[T, state_dim]
    filtered_covariances: Array[T, state_dim, state_dim]
    smoothed_means: Array[T, state_dim]
    smoothed_covariances: Array[T, state_dim, state_dim]
)
```

---

## Extended Kalman Filter (inference_ekf.py)

### Core Functions

#### 1. Prediction Step

```python
def _predict(m, P, f, F, Q, u):
    """Predict next mean and covariance using first-order additive EKF

    Args:
        m: prior mean (D_hid,)
        P: prior covariance (D_hid, D_hid)
        f: dynamics function
        F: Jacobian of dynamics (computed via jacfwd)
        Q: dynamics covariance
        u: inputs

    Returns:
        mu_pred: predicted mean
        Sigma_pred: predicted covariance
    """
    F_x = F(m, u)  # Linearize around current mean
    mu_pred = f(m, u)
    Sigma_pred = F_x @ P @ F_x.T + Q
    return mu_pred, Sigma_pred
```

**Key Design:**

- Linearization around **prior mean** (not posterior)
- First-order Taylor expansion: `f(z) ≈ f(m) + F(m)·(z - m)`
- Covariance propagation: `P_pred = F·P·F^T + Q`

#### 2. Update Step (Iterated EKF)

```python
def _condition_on(m, P, h, H, R, u, y, num_iter):
    """Condition on observation with optional re-linearization

    Args:
        m: predicted mean
        P: predicted covariance
        h: emission function
        H: Jacobian of emission
        R: emission covariance
        u: inputs
        y: observation
        num_iter: number of re-linearizations (1 = standard EKF)

    Returns:
        mu_cond: filtered mean
        Sigma_cond: filtered covariance
    """
    def _step(carry, _):
        prior_mean, prior_cov = carry
        H_x = H(prior_mean, u)  # Linearize around current estimate
        S = R + H_x @ prior_cov @ H_x.T
        K = psd_solve(S, H_x @ prior_cov).T  # Kalman gain
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(prior_mean, u))
        return (posterior_mean, posterior_cov), None

    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, symmetrize(Sigma_cond)
```

**Key Design:**

- **Iterated EKF**: Re-linearize around posterior (num_iter > 1)
- Uses `psd_solve` for numerical stability (avoids explicit inverse)
- `symmetrize` ensures covariance stays symmetric despite numerics

#### 3. Filter

```python
def extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: List[str] = ["filtered_means", "filtered_covariances", ...]
) -> PosteriorGSSMFiltered:
    """Run EKF with lax.scan for efficiency"""

    # Get functions and Jacobians
    f, h = params.dynamics_function, params.emission_function
    F, H = jacfwd(f), jacfwd(h)

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Get time-specific parameters
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u, y = inputs[t], emissions[t]

        # Update log-likelihood
        H_x = H(pred_mean, u)
        ll += MVN(h(pred_mean, u), H_x @ pred_cov @ H_x.T + R).log_prob(y)

        # Filter: condition on observation
        filtered_mean, filtered_cov = _condition_on(
            pred_mean, pred_cov, h, H, R, u, y, num_iter
        )

        # Predict: propagate to next timestep
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, f, F, Q, u)

        carry = (ll, pred_mean, pred_cov)
        outputs = {"filtered_means": filtered_mean, ...}
        return carry, outputs

    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=ll, **outputs)
```

**Key Design:**

- Uses `jax.lax.scan` for efficient compilation and memory
- Computes marginal log-likelihood incrementally
- Supports time-varying Q, R via `_get_params`
- Optional `output_fields` to reduce memory (e.g., drop predicted means/covs)

#### 4. RTS Smoother

```python
def extended_kalman_smoother(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMSmoothed:
    """Run RTS smoother (backward pass)"""

    # Run filter if not provided
    if filtered_posterior is None:
        filtered_posterior = extended_kalman_filter(params, emissions, inputs=inputs)

    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    f, F = params.dynamics_function, jacfwd(f)

    def _step(carry, args):
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]
        F_x = F(filtered_mean, u)

        # Predict from filtered
        m_pred = f(filtered_mean, u)
        S_pred = Q + F_x @ filtered_cov @ F_x.T

        # Smoother gain
        G = psd_solve(S_pred, F_x @ filtered_cov).T

        # Smooth
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run backward pass
    _, (smoothed_means, smoothed_covs) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1]),  # Initialize with last filtered
        (jnp.arange(num_timesteps - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True  # Backward pass
    )

    # Concatenate last timestep
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs, filtered_covs[-1][None, ...]))

    return PosteriorGSSMSmoothed(...)
```

**Key Design:**

- RTS = Rauch-Tung-Striebel backward smoother
- Requires forward pass first (filtered estimates)
- Uses `reverse=True` in `lax.scan` for backward pass
- Smoother gain: `G = P_filt · F^T · (F·P_filt·F^T + Q)^{-1}`

---

## Unscented Kalman Filter (inference_ukf.py)

### UKF Hyperparameters

```python
class UKFHyperParams(NamedTuple):
    alpha: float = jnp.sqrt(3)  # Spread of sigma points
    beta: int = 2               # Prior knowledge (2 for Gaussian)
    kappa: int = 1              # Secondary scaling
```

**Key formulas:**

- `lambda = alpha^2 * (n + kappa) - n`
- Determines spread of sigma points around mean

### Core Functions

#### 1. Sigma Points

```python
def _compute_sigmas(m, P, n, lamb):
    """Compute (2n+1) sigma points for unscented transform

    Returns:
        sigmas: [2n+1, n] array of sigma points
            - sigmas[0] = m (mean)
            - sigmas[1:n+1] = m + sqrt((n+lamb)·P) columns
            - sigmas[n+1:] = m - sqrt((n+lamb)·P) columns
    """
    distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
    sigma_plus = jnp.array([m + distances[:, i] for i in range(n)])
    sigma_minus = jnp.array([m - distances[:, i] for i in range(n)])
    return jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))
```

#### 2. Weights

```python
def _compute_weights(n, alpha, beta, lamb):
    """Compute weights for mean and covariance reconstruction

    Returns:
        w_mean: [2n+1] weights for computing mean
        w_cov: [2n+1] weights for computing covariance
    """
    factor = 1 / (2 * (n + lamb))
    w_mean = jnp.concatenate((
        jnp.array([lamb / (n + lamb)]),
        jnp.ones(2 * n) * factor
    ))
    w_cov = jnp.concatenate((
        jnp.array([lamb / (n + lamb) + (1 - alpha**2 + beta)]),
        jnp.ones(2 * n) * factor
    ))
    return w_mean, w_cov
```

**Key Point:** Different weights for mean vs. covariance (beta incorporates prior)

#### 3. Prediction

```python
def _predict(m, P, f, Q, lamb, w_mean, w_cov, u):
    """Predict using unscented transform"""
    n = len(m)

    # Form and propagate sigma points
    sigmas = _compute_sigmas(m, P, n, lamb)
    u_s = jnp.array([u] * len(sigmas))
    sigmas_prop = vmap(f, (0, 0), 0)(sigmas, u_s)

    # Reconstruct mean and covariance
    m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)
    P_pred = jnp.tensordot(
        w_cov,
        _outer(sigmas_prop - m_pred, sigmas_prop - m_pred),
        axes=1
    ) + Q
    P_cross = jnp.tensordot(
        w_cov,
        _outer(sigmas - m, sigmas_prop - m_pred),
        axes=1
    )
    return m_pred, P_pred, P_cross
```

**Key Design:**

- Propagate sigma points through **nonlinear** function (no Jacobian!)
- `P_cross` needed for smoother (cross-covariance between pre/post)
- Uses `vmap` for vectorized function application

#### 4. Update

```python
def _condition_on(m, P, h, R, lamb, w_mean, w_cov, u, y):
    """Update using unscented transform"""
    n = len(m)

    # Form and propagate sigma points
    sigmas = _compute_sigmas(m, P, n, lamb)
    u_s = jnp.array([u] * len(sigmas))
    sigmas_prop = vmap(h, (0, 0), 0)(sigmas, u_s)

    # Predict observation
    pred_mean = jnp.tensordot(w_mean, sigmas_prop, axes=1)
    pred_cov = jnp.tensordot(
        w_cov,
        _outer(sigmas_prop - pred_mean, sigmas_prop - pred_mean),
        axes=1
    ) + R
    pred_cross = jnp.tensordot(
        w_cov,
        _outer(sigmas - m, sigmas_prop - pred_mean),
        axes=1
    )

    # Kalman update
    ll = MVN(pred_mean, pred_cov).log_prob(y)
    K = psd_solve(pred_cov, pred_cross.T).T
    m_cond = m + K @ (y - pred_mean)
    P_cond = P - K @ pred_cov @ K.T

    return ll, m_cond, P_cond
```

#### 5. Smoother

```python
def unscented_kalman_smoother(...):
    """UKF + backward pass (similar to RTS)"""

    # Run forward UKF
    ukf_posterior = unscented_kalman_filter(params, emissions, hyperparams, inputs)

    def _step(carry, args):
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Predict from filtered using unscented transform
        m_pred, S_pred, S_cross = _predict(
            filtered_mean, filtered_cov, f, Q, lamb, w_mean, w_cov, u
        )

        # Smoother gain (uses cross-covariance!)
        G = psd_solve(S_pred, S_cross.T).T

        # Smooth
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Backward pass
    _, (smoothed_means, smoothed_covs) = lax.scan(_step, ..., reverse=True)
    return PosteriorGSSMSmoothed(...)
```

**Key Difference from EKF:**

- UKF smoother uses cross-covariance `S_cross` from unscented transform
- EKF smoother uses Jacobian: `F·P_filt`

---

## Testing Patterns (inference_test_utils.py)

### Creating Test Data

```python
def random_nlgssm_args(key=0, num_timesteps=15, state_dim=4, emission_dim=2):
    """Generate random NLGSSM with nonlinear dynamics/emissions"""
    params = make_nlgssm_params(state_dim, emission_dim, key=key)
    model = NonlinearGaussianSSM(state_dim, emission_dim)
    states, emissions = model.sample(params, key, num_timesteps)
    return params, states, emissions

def make_nlgssm_params(state_dim, emission_dim, key):
    """Create nonlinear test functions"""
    dynamics_weights = jr.normal(key, (state_dim, state_dim * 2))
    f = lambda z: jnp.sin(dynamics_weights @ to_poly(z, degree=1))

    emission_weights = jr.normal(key, (emission_dim, state_dim * 2))
    h = lambda z: jnp.cos(emission_weights @ to_poly(z, degree=1))

    return ParamsNLGSSM(
        initial_mean=0.2 * jnp.ones(state_dim),
        initial_covariance=jnp.eye(state_dim),
        dynamics_function=f,
        dynamics_covariance=jnp.eye(state_dim),
        emission_function=h,
        emission_covariance=jnp.eye(emission_dim)
    )
```

### Validation Against Linear Case

```python
def test_extended_kalman_filter_linear():
    """EKF should match KF on linear problems"""
    args, _, emissions = random_lgssm_args(key=0, num_timesteps=15)

    # Standard Kalman filter
    kf_post = lgssm_filter(args, emissions)

    # Extended Kalman filter (convert linear to nonlinear)
    ekf_post = extended_kalman_filter(lgssm_to_nlgssm(args), emissions)

    # Should be identical
    assert allclose(kf_post.filtered_means, ekf_post.filtered_means)
    assert allclose(kf_post.filtered_covariances, ekf_post.filtered_covariances)
```

### Cross-Validation with Sarkka Library

```python
def test_extended_kalman_filter_nonlinear():
    """Compare dynamax EKF to reference implementation"""
    args, _, emissions = random_nlgssm_args(key=42, num_timesteps=15)

    # Reference EKF from sarkka-jax
    means_ext, covs_ext = ekf(*args, emissions)

    # Dynamax EKF
    ekf_post = extended_kalman_filter(args, emissions)

    assert allclose(means_ext, ekf_post.filtered_means)
    assert allclose(covs_ext, ekf_post.filtered_covariances)
```

---

## Key Implementation Details

### 1. Helper Functions

```python
# Get time-varying or static parameters
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x

# Process functions to handle optional inputs
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f

# Default inputs to zeros if not provided
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x
```

### 2. Numerical Stability

- **`psd_solve(A, b)`**: Solves `A·x = b` for PSD matrix A (more stable than `inv(A) @ b`)
- **`symmetrize(P)`**: Ensures covariance stays symmetric: `(P + P.T) / 2`
- **Cholesky decomposition**: Used in UKF for sigma points (numerically stable)

### 3. JAX Patterns

- **`lax.scan`**: Efficient sequential computation (compiles to single loop)
- **`vmap`**: Vectorize function over batch dimension
- **`jacfwd`**: Forward-mode Jacobian (efficient for wide Jacobians: out_dim << in_dim)
- **`jnp.tensordot`**: Weighted sum over sigma points

---

## Recommended Adaptations for Trodestrack

### 1. State Structure

Adapt ParamsNLGSSM for 2D tracking with IMU biases:

```python
# State: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
state_dim = 8

# Observations: dual LED positions
emission_dim = 4  # [x_led1, y_led1, x_led2, y_led2]
```

### 2. Dynamics Function

IMU pre-integration between camera frames:

```python
def dynamics_function(state, imu_preint):
    """
    Args:
        state: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
        imu_preint: Pre-integrated IMU (Δθ, Δv_x, Δv_y)

    Returns:
        next_state: [x', y', vx', vy', θ', b_gz', b_ax', b_ay']
    """
    x, y, vx, vy, theta, b_gz, b_ax, b_ay = state
    delta_theta, delta_vx, delta_vy = imu_preint

    # Heading update
    theta_new = wrap_angle(theta + delta_theta)

    # Velocity update (with damping)
    vx_new = vx + delta_vx
    vy_new = vy + delta_vy

    # Position update
    x_new = x + vx * dt + 0.5 * delta_vx * dt
    y_new = y + vy * dt + 0.5 * delta_vy * dt

    # Biases (random walk)
    b_gz_new = b_gz  # Will have process noise
    b_ax_new = b_ax
    b_ay_new = b_ay

    return jnp.array([x_new, y_new, vx_new, vy_new, theta_new,
                      b_gz_new, b_ax_new, b_ay_new])
```

### 3. Emission Function

```python
def emission_function(state, led_config):
    """Dual LED observations

    Args:
        state: [x, y, vx, vy, θ, ...]
        led_config: LED spacing

    Returns:
        [x_led1, y_led1, x_led2, y_led2]
    """
    x, y, _, _, theta = state[:5]

    # LED positions in body frame
    r_led = led_config['front_back_distance'] / 2

    # Transform to world frame
    c, s = jnp.cos(theta), jnp.sin(theta)
    x_led1 = x + r_led * c
    y_led1 = y + r_led * s
    x_led2 = x - r_led * c
    y_led2 = y - r_led * s

    return jnp.array([x_led1, y_led1, x_led2, y_led2])
```

### 4. Adaptive Noise Covariance

Time-varying R based on DLC confidence:

```python
def make_emission_covariance(confidence_led1, confidence_led2, base_cov):
    """Scale noise by confidence scores"""
    scale1 = 1.0 / jnp.maximum(confidence_led1, 0.1)  # Avoid division by zero
    scale2 = 1.0 / jnp.maximum(confidence_led2, 0.1)

    R = jnp.diag([base_cov * scale1**2, base_cov * scale1**2,
                  base_cov * scale2**2, base_cov * scale2**2])
    return R

# Create time-varying R
R_t = vmap(make_emission_covariance)(confidences_led1, confidences_led2, base_cov)
# R_t.shape = (T, 4, 4)
```

### 5. Gating and Masking

Handle dropouts:

```python
def filter_with_gating(params, emissions, masks, confidence, threshold=9.21):
    """Filter with Mahalanobis gating and dropout handling

    Args:
        masks: [T] boolean array (True = valid)
        threshold: chi-squared threshold (9.21 = 99% for 4-DOF)
    """
    def _step_with_gate(carry, t):
        ll, pred_mean, pred_cov = carry

        if masks[t]:  # Valid observation
            # Compute innovation
            y_pred = h(pred_mean)
            innovation = emissions[t] - y_pred

            # Mahalanobis distance
            S = H @ pred_cov @ H.T + R[t]
            mahal = innovation @ jnp.linalg.inv(S) @ innovation

            # Gate: only update if within threshold
            filtered_mean, filtered_cov = lax.cond(
                mahal < threshold,
                lambda: _condition_on(...),  # Accept
                lambda: (pred_mean, pred_cov)  # Reject
            )
        else:  # Dropout: skip update
            filtered_mean, filtered_cov = pred_mean, pred_cov

        # Always predict
        pred_mean, pred_cov = _predict(...)
        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run filter
    ...
```

### 6. LED Swap Resolution

Mixture update for swap ambiguity:

```python
def condition_on_with_swap(m, P, h, R, y_led1, y_led2):
    """Handle LED swap ambiguity via mixture"""

    # Hypothesis 1: No swap
    y_no_swap = jnp.concatenate([y_led1, y_led2])
    ll1, m1, P1 = _condition_on(m, P, h, R, y_no_swap)

    # Hypothesis 2: Swapped
    y_swap = jnp.concatenate([y_led2, y_led1])  # Reversed!
    ll2, m2, P2 = _condition_on(m, P, h, R, y_swap)

    # Mixture weights (posterior probability of each hypothesis)
    w1 = jnp.exp(ll1) / (jnp.exp(ll1) + jnp.exp(ll2))
    w2 = 1 - w1

    # Mixture update
    m_mix = w1 * m1 + w2 * m2
    P_mix = w1 * (P1 + jnp.outer(m1 - m_mix, m1 - m_mix)) + \
            w2 * (P2 + jnp.outer(m2 - m_mix, m2 - m_mix))

    return m_mix, P_mix
```

---

## Summary of Best Practices

1. **Use `lax.scan` for sequential operations** - compiles efficiently
2. **Use `vmap` for batch operations** - sigma points, time-varying params
3. **Use `psd_solve` instead of explicit inverse** - numerical stability
4. **Symmetrize covariances** after updates - combat numerical drift
5. **Support time-varying parameters** via `_get_params` helper
6. **Optional inputs** handled gracefully via `_process_fn`
7. **Separate filter and smoother** - reuse filtered posterior
8. **Iterated EKF** via `num_iter > 1` - better for highly nonlinear
9. **Output field selection** - reduce memory for large state spaces
10. **Test against linear case** - EKF/UKF should match KF exactly

---

## File Organization for Trodestrack

Suggested structure based on dynamax patterns:

```
trodestrack/
  models/
    states.py         # State definitions, ParamsNLGSSM-like
    dynamics.py       # IMU pre-integration, dynamics functions
    measurements.py   # LED emission functions, R scaling

  filters/
    ekf.py           # Extended Kalman Filter (adapt inference_ekf.py)
    ukf.py           # Unscented Kalman Filter (adapt inference_ukf.py)
    utils.py         # psd_solve, symmetrize, helpers

  runtime/
    online.py        # OnlineTracker (incremental filter)
    offline.py       # smooth_session (batch filter + smoother)

  tests/
    filters/
      test_ekf.py          # Test against analytic solutions
      test_ukf.py          # Compare EKF vs UKF
      test_bias_obs.py     # Bias observability tests
```

---

## References

- Särkkä, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press.
- Dynamax: <https://github.com/probml/dynamax>
- Sarkka-JAX: <https://github.com/petergchang/sarkka-jax>

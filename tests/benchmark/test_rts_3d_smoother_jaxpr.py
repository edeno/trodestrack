"""Benchmarks and trace checks for the 3D quaternion RTS smoother."""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jax_core

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter_3d
from trodestrack.models.filter_common import compute_imu_index_arrays
from trodestrack.models.quaternion import (
    quaternion_from_rotation_vector,
    rotate_vector_body_to_world,
    rotate_vector_world_to_body,
)
from trodestrack.models.state_layout import StateLayout, get_layout
from trodestrack.runtime.offline import (
    _rts_smoother_impl,
    _rts_smoother_jit,
)


class RTS3DBenchmarkCase(NamedTuple):
    config: EKFConfig
    layout: StateLayout
    t_imu: np.ndarray
    U_imu: np.ndarray
    t_cam: np.ndarray
    mask_cam: np.ndarray
    filtered_means: jnp.ndarray
    filtered_covariances: jnp.ndarray
    t_imu_jax: jnp.ndarray
    U_imu_jax: jnp.ndarray
    mask_cam_jax: jnp.ndarray
    imu_index_arrays: jnp.ndarray
    dt_imu_mean: jnp.ndarray


def make_rts_3d_benchmark_case(
    *,
    duration_s: float = 1.0,
    fs_imu: float = 200.0,
    fs_cam: float = 30.0,
) -> RTS3DBenchmarkCase:
    """Create deterministic 3D EKF output for RTS timing and trace checks."""

    layout = get_layout("3d_cam_6dof_imu")
    config = EKFConfig(
        state_mode="3d_cam_6dof_imu",
        measurement_noise_pos=1e-4,
        enable_experimental_accel_translation=True,
        enable_zupt=False,
        use_mahalanobis_gating=False,
        use_gravity_orientation_update=False,
    )
    t_imu = np.arange(0.0, duration_s + 0.5 / fs_imu, 1.0 / fs_imu).astype(np.float32)
    t_cam = np.arange(0.0, duration_s + 0.5 / fs_cam, 1.0 / fs_cam).astype(np.float32)
    led_offsets = jnp.array(
        [
            [-0.03, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.0, 0.025, 0.02],
        ],
        dtype=jnp.float32,
    )
    true_position = jnp.array([0.2, -0.1, 0.25], dtype=jnp.float32)
    true_quat = quaternion_from_rotation_vector(
        jnp.array([0.04, -0.03, 0.09], dtype=jnp.float32)
    )
    z_leds_one = true_position[None, :] + rotate_vector_body_to_world(
        true_quat,
        led_offsets,
    )
    z_leds = np.repeat(np.asarray(z_leds_one)[None, :, :], t_cam.shape[0], axis=0)
    U_imu = np.zeros((t_imu.shape[0], 6), dtype=np.float32)
    gravity_body = rotate_vector_world_to_body(
        true_quat,
        jnp.array([0.0, 0.0, 9.81], dtype=jnp.float32),
    )
    U_imu[:, 3:6] = np.asarray(gravity_body)
    mask_cam = np.ones(t_cam.shape[0], dtype=bool)

    filter_result = extended_kalman_filter_3d(
        config,
        t_imu,
        U_imu,
        t_cam,
        z_leds,
        np.asarray(led_offsets),
    )
    t_imu_jax = jnp.asarray(t_imu)
    return RTS3DBenchmarkCase(
        config=config,
        layout=layout,
        t_imu=t_imu,
        U_imu=U_imu,
        t_cam=t_cam,
        mask_cam=mask_cam,
        filtered_means=filter_result.filtered_means,
        filtered_covariances=filter_result.filtered_covariances,
        t_imu_jax=t_imu_jax,
        U_imu_jax=jnp.asarray(U_imu),
        mask_cam_jax=jnp.asarray(mask_cam),
        imu_index_arrays=jnp.asarray(compute_imu_index_arrays(t_imu, t_cam)),
        dt_imu_mean=jnp.mean(jnp.diff(t_imu_jax)),
    )


def _block_until_ready(result):
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result


def _time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _block_until_ready(result)
    return result, time.perf_counter() - start


def _nested_primitive_names(jaxpr: jax_core.Jaxpr) -> list[str]:
    primitive_names: list[str] = []
    for eqn in jaxpr.eqns:
        primitive_names.append(eqn.primitive.name)
        for value in eqn.params.values():
            if isinstance(value, jax_core.ClosedJaxpr):
                primitive_names.extend(_nested_primitive_names(value.jaxpr))
            elif isinstance(value, jax_core.Jaxpr):
                primitive_names.extend(_nested_primitive_names(value))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    if isinstance(item, jax_core.ClosedJaxpr):
                        primitive_names.extend(_nested_primitive_names(item.jaxpr))
                    elif isinstance(item, jax_core.Jaxpr):
                        primitive_names.extend(_nested_primitive_names(item))
    return primitive_names


def _run_rts_impl(
    case: RTS3DBenchmarkCase,
    filtered_means: jnp.ndarray,
    filtered_covariances: jnp.ndarray,
    *,
    num_iter: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _rts_smoother_impl(
        filtered_means.copy(),
        filtered_means,
        filtered_covariances.copy(),
        case.t_imu_jax,
        case.U_imu_jax,
        case.mask_cam_jax,
        True,
        case.imu_index_arrays,
        case.dt_imu_mean,
        num_iter=num_iter,
        ekf_config=case.config,
        layout=case.layout,
    )


def _run_rts_jit(
    case: RTS3DBenchmarkCase,
    filtered_means: jnp.ndarray,
    filtered_covariances: jnp.ndarray,
    *,
    num_iter: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _rts_smoother_jit(
        filtered_means.copy(),
        filtered_means,
        filtered_covariances.copy(),
        case.t_imu_jax,
        case.U_imu_jax,
        case.mask_cam_jax,
        True,
        case.imu_index_arrays,
        case.dt_imu_mean,
        num_iter=num_iter,
        ekf_config=case.config,
        layout=case.layout,
    )


@pytest.mark.benchmark
def test_rts_3d_smoother_jit_parity_and_timing() -> None:
    """Measure 3D RTS eager/JIT behavior before optimizing dynamics calls."""

    case = make_rts_3d_benchmark_case()
    eager_result, eager_s = _time_call(
        _run_rts_impl,
        case,
        case.filtered_means,
        case.filtered_covariances,
        num_iter=1,
    )
    compile_result, compile_s = _time_call(
        _run_rts_jit,
        case,
        case.filtered_means,
        case.filtered_covariances,
        num_iter=1,
    )
    warmed_result, warmed_s = _time_call(
        _run_rts_jit,
        case,
        case.filtered_means,
        case.filtered_covariances,
        num_iter=1,
    )
    iter2_compile_result, iter2_compile_s = _time_call(
        _run_rts_jit,
        case,
        case.filtered_means,
        case.filtered_covariances,
        num_iter=2,
    )
    iter2_warmed_result, iter2_warmed_s = _time_call(
        _run_rts_jit,
        case,
        case.filtered_means,
        case.filtered_covariances,
        num_iter=2,
    )

    np.testing.assert_allclose(
        np.asarray(warmed_result[0]),
        np.asarray(eager_result[0]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(warmed_result[1]),
        np.asarray(eager_result[1]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(compile_result[0]),
        np.asarray(warmed_result[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(iter2_compile_result[0]),
        np.asarray(iter2_warmed_result[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isfinite(np.asarray(iter2_warmed_result[0])).all()

    print("\n=== 3D RTS Smoother JIT Benchmark ===")
    print(f"Camera frames: {case.t_cam.shape[0]}")
    print(f"IMU samples: {case.t_imu.shape[0]}")
    print(f"Eager smoother num_iter=1: {eager_s:.4f} s")
    print(f"First jitted smoother num_iter=1: {compile_s:.4f} s")
    print(f"Warmed jitted smoother num_iter=1: {warmed_s:.4f} s")
    print(f"First jitted smoother num_iter=2: {iter2_compile_s:.4f} s")
    print(f"Warmed jitted smoother num_iter=2: {iter2_warmed_s:.4f} s")
    print(f"Warmed speedup vs eager: {eager_s / warmed_s:.1f}x")

    # Wall-clock comparison is informational only — see comment in
    # tests/benchmark/test_ekf_3d_core_jit.py for the same rationale.


def test_rts_3d_smoother_jaxpr_shape_contract() -> None:
    """Inspect the 3D RTS trace before attempting dynamics-call reduction."""

    case = make_rts_3d_benchmark_case(duration_s=0.2)

    def run_smoother(
        filtered_means: jnp.ndarray,
        filtered_covariances: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _run_rts_jit(
            case,
            filtered_means,
            filtered_covariances,
            num_iter=1,
        )

    jaxpr = jax.make_jaxpr(run_smoother)(
        case.filtered_means,
        case.filtered_covariances,
    )
    primitive_names = _nested_primitive_names(jaxpr.jaxpr)

    assert primitive_names.count("scan") >= 2
    assert "nonzero" not in primitive_names

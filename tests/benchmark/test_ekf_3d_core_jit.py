"""Benchmarks and parity checks for the traceable 3D EKF core."""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jax_core

from trodestrack.models.ekf import (
    EKFConfig,
    _extended_kalman_filter_3d_core,
    _extended_kalman_filter_3d_jit,
)
from trodestrack.models.filter_common import FilterState, compute_imu_index_arrays
from trodestrack.models.quaternion import (
    quaternion_from_rotation_vector,
    rotate_vector_body_to_world,
    rotate_vector_world_to_body,
)
from trodestrack.models.state_layout import StateLayout, get_layout


class EKF3DBenchmarkCase(NamedTuple):
    config: EKFConfig
    layout: StateLayout
    t_imu: jnp.ndarray
    U_imu: jnp.ndarray
    t_cam: jnp.ndarray
    z_leds: jnp.ndarray
    led_offsets: jnp.ndarray
    initial_state: FilterState
    imu_index_arrays: jnp.ndarray


def make_ekf_3d_benchmark_case(
    *,
    duration_s: float = 2.0,
    fs_imu: float = 200.0,
    fs_cam: float = 30.0,
) -> EKF3DBenchmarkCase:
    """Create a deterministic 3D EKF case for JIT timing and parity."""

    layout = get_layout("3d_cam_6dof_imu")
    config = EKFConfig(
        state_mode="3d_cam_6dof_imu",
        measurement_noise_pos=1e-4,
        enable_experimental_accel_translation=True,
        enable_zupt=False,
        use_mahalanobis_gating=True,
        use_gravity_orientation_update=False,
    )
    t_imu_np = np.arange(0.0, duration_s + 0.5 / fs_imu, 1.0 / fs_imu).astype(
        np.float32
    )
    t_cam_np = np.arange(0.0, duration_s + 0.5 / fs_cam, 1.0 / fs_cam).astype(
        np.float32
    )

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
        jnp.array([0.05, -0.04, 0.08], dtype=jnp.float32)
    )
    z_leds_one = true_position[None, :] + rotate_vector_body_to_world(
        true_quat,
        led_offsets,
    )
    z_leds = jnp.repeat(z_leds_one[None, :, :], t_cam_np.shape[0], axis=0)

    gravity_body = rotate_vector_world_to_body(
        true_quat,
        jnp.array([0.0, 0.0, 9.81], dtype=jnp.float32),
    )
    U_imu = jnp.zeros((t_imu_np.shape[0], 6), dtype=jnp.float32)
    U_imu = U_imu.at[:, 3:6].set(gravity_body)

    mean = jnp.zeros(layout.n, dtype=jnp.float32)
    mean = mean.at[jnp.array(layout.heading_idx)].set(true_quat)
    mean = mean.at[jnp.array(layout.pos_idx)].set(true_position)
    initial_state = FilterState(
        mean=mean,
        cov=jnp.eye(layout.n, dtype=jnp.float32) * jnp.asarray(0.1, dtype=jnp.float32),
    )

    return EKF3DBenchmarkCase(
        config=config,
        layout=layout,
        t_imu=jnp.asarray(t_imu_np),
        U_imu=U_imu,
        t_cam=jnp.asarray(t_cam_np),
        z_leds=z_leds,
        led_offsets=led_offsets,
        initial_state=initial_state,
        imu_index_arrays=compute_imu_index_arrays(t_imu_np, t_cam_np),
    )


def _run_core(case: EKF3DBenchmarkCase, U_imu: jnp.ndarray, z_leds: jnp.ndarray):
    return _extended_kalman_filter_3d_core(
        case.config,
        jnp.asarray(True),
        case.t_imu,
        U_imu,
        case.t_cam,
        z_leds,
        case.led_offsets,
        None,
        None,
        case.initial_state,
        case.imu_index_arrays,
        layout=case.layout,
    )


def _run_core_jit(case: EKF3DBenchmarkCase, U_imu: jnp.ndarray, z_leds: jnp.ndarray):
    return _extended_kalman_filter_3d_jit(
        case.config,
        jnp.asarray(True),
        case.t_imu,
        U_imu,
        case.t_cam,
        z_leds,
        case.led_offsets,
        None,
        None,
        case.initial_state,
        case.imu_index_arrays,
        layout=case.layout,
    )


def _block_until_ready(result):
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result


def _time_call(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
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


@pytest.mark.benchmark
def test_ekf_3d_core_jit_parity_and_timing() -> None:
    """Compare eager and explicitly jitted 3D EKF core execution."""

    case = make_ekf_3d_benchmark_case()

    def run_core(U_imu_arg: jnp.ndarray, z_leds_arg: jnp.ndarray):
        return _run_core(case, U_imu_arg, z_leds_arg)

    def run_core_jit(U_imu_arg: jnp.ndarray, z_leds_arg: jnp.ndarray):
        return _run_core_jit(case, U_imu_arg, z_leds_arg)

    eager_result, eager_s = _time_call(run_core, case.U_imu, case.z_leds)
    compile_result, compile_s = _time_call(run_core_jit, case.U_imu, case.z_leds)
    warmed_result, warmed_s = _time_call(run_core_jit, case.U_imu, case.z_leds)

    np.testing.assert_allclose(
        np.asarray(warmed_result.filtered_means),
        np.asarray(eager_result.filtered_means),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(warmed_result.filtered_covariances),
        np.asarray(eager_result.filtered_covariances),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        float(warmed_result.marginal_loglik),
        float(eager_result.marginal_loglik),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(compile_result.filtered_means),
        np.asarray(warmed_result.filtered_means),
        rtol=1e-6,
        atol=1e-6,
    )

    print("\n=== 3D EKF Core JIT Benchmark ===")
    print(f"Camera frames: {case.t_cam.shape[0]}")
    print(f"IMU samples: {case.t_imu.shape[0]}")
    print(f"Eager core: {eager_s:.4f} s")
    print(f"First jitted call: {compile_s:.4f} s")
    print(f"Warmed jitted call: {warmed_s:.4f} s")
    print(f"Warmed speedup vs eager: {eager_s / warmed_s:.1f}x")

    assert np.isfinite(float(warmed_result.marginal_loglik))
    assert warmed_s < eager_s


def test_ekf_3d_core_jaxpr_shape_contract() -> None:
    """Guard the 3D EKF core trace against dynamic-shape regressions."""

    case = make_ekf_3d_benchmark_case(duration_s=0.2)

    def run_loglik(U_imu_arg: jnp.ndarray, z_leds_arg: jnp.ndarray) -> jnp.ndarray:
        return _run_core(case, U_imu_arg, z_leds_arg).marginal_loglik

    jaxpr = jax.make_jaxpr(run_loglik)(case.U_imu, case.z_leds)
    primitive_names = _nested_primitive_names(jaxpr.jaxpr)

    assert primitive_names.count("scan") >= 2
    assert "nonzero" not in primitive_names

"""Benchmarks and trace checks for the 3D camera measurement model."""

from __future__ import annotations

import time
from collections import Counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.extend import core as jax_core

from trodestrack.models.quaternion import (
    quaternion_from_rotation_vector,
    rotate_vector_body_to_world,
)
from trodestrack.models.sensors.camera_position_3d import Camera3DPositionModel
from trodestrack.models.state_layout import StateLayout, get_layout


class Camera3DBenchmarkCase(NamedTuple):
    layout: StateLayout
    model: Camera3DPositionModel
    state_mean: jnp.ndarray
    n_frames: int


def make_camera_3d_benchmark_case(
    *,
    n_frames: int = 61,
) -> Camera3DBenchmarkCase:
    """Create deterministic 3D camera model inputs for trace inspection."""

    layout = get_layout("3d_cam_6dof_imu")
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
    z_leds = jnp.repeat(z_leds_one[None, :, :], n_frames, axis=0)
    mask_leds = jnp.ones((n_frames, led_offsets.shape[0]), dtype=bool)
    # Exercise masked coordinate handling without changing static shapes.
    mask_leds = mask_leds.at[n_frames // 3, 1].set(False)
    conf_leds = jnp.ones((n_frames, led_offsets.shape[0]), dtype=jnp.float32)
    conf_leds = conf_leds.at[n_frames // 2, 2].set(0.25)

    model = Camera3DPositionModel(
        led_offsets_body=led_offsets,
        measurement_noise_base=1e-4,
        layout=layout,
        z_leds_all=z_leds,
        mask_leds_all=mask_leds,
        conf_all=conf_leds,
    )
    state_mean = jnp.zeros(layout.n, dtype=jnp.float32)
    state_mean = state_mean.at[jnp.array(layout.pos_idx)].set(true_position)
    state_mean = state_mean.at[jnp.array(layout.heading_idx)].set(true_quat)
    return Camera3DBenchmarkCase(
        layout=layout,
        model=model,
        state_mean=state_mean,
        n_frames=n_frames,
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


def _run_camera_model_scan(case: Camera3DBenchmarkCase, state_mean: jnp.ndarray):
    model = case.model

    def scan_one(_, frame_idx):
        meas_pred = model.predict(state_mean)
        innovation = model.innovation(frame_idx, meas_pred)
        H = model.jacobian(state_mean, frame_idx)
        R = model.meas_cov(frame_idx)
        valid = model.valid_coordinates(frame_idx)
        return None, (
            meas_pred,
            innovation,
            H,
            jnp.diag(R),
            valid.astype(jnp.int32),
        )

    _, outputs = lax.scan(
        scan_one,
        None,
        jnp.arange(case.n_frames, dtype=jnp.int32),
    )
    return outputs


@pytest.mark.benchmark
def test_camera_3d_model_jit_parity_and_timing() -> None:
    """Measure camera-model scan behavior before index/mask refactors."""

    case = make_camera_3d_benchmark_case()

    def run_model(state_mean: jnp.ndarray):
        return _run_camera_model_scan(case, state_mean)

    run_model_jit = jax.jit(run_model)
    eager_result, eager_s = _time_call(run_model, case.state_mean)
    compile_result, compile_s = _time_call(run_model_jit, case.state_mean)
    warmed_result, warmed_s = _time_call(run_model_jit, case.state_mean)

    for eager_leaf, warmed_leaf in zip(eager_result, warmed_result, strict=True):
        np.testing.assert_allclose(
            np.asarray(warmed_leaf),
            np.asarray(eager_leaf),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_allclose(
        np.asarray(compile_result[0]),
        np.asarray(warmed_result[0]),
        rtol=1e-6,
        atol=1e-6,
    )

    print("\n=== Camera3DPositionModel JIT Benchmark ===")
    print(f"Camera frames: {case.n_frames}")
    print(f"Measurement dimension: {case.model.meas_dim}")
    print(f"Eager model scan: {eager_s:.4f} s")
    print(f"First jitted model scan: {compile_s:.4f} s")
    print(f"Warmed jitted model scan: {warmed_s:.4f} s")
    print(f"Warmed speedup vs eager: {eager_s / warmed_s:.1f}x")

    # Wall-clock comparison is informational only — see comment in
    # tests/benchmark/test_ekf_3d_core_jit.py for the same rationale.


def test_camera_3d_model_jaxpr_shape_contract() -> None:
    """Inspect model trace for dynamic-shape and repeated-index work."""

    case = make_camera_3d_benchmark_case(n_frames=7)

    def run_model(state_mean: jnp.ndarray):
        return _run_camera_model_scan(case, state_mean)

    jaxpr = jax.make_jaxpr(run_model)(case.state_mean)
    primitive_counts = Counter(_nested_primitive_names(jaxpr.jaxpr))

    assert primitive_counts["scan"] >= 1
    assert primitive_counts["nonzero"] == 0
    assert primitive_counts["gather"] > 0

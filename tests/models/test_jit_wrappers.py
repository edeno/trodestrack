"""Ensure hot-path filters expose compiled JIT entry points with metadata.

These tests enforce the performance refactor requirements in Milestone M6:
the EKF forward pass and RTS smoother must provide `jax.jit`-compiled
implementations that donate large buffers and treat the state layout as a
static argument to avoid recompilation churn.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "jit_attr", "static_attr", "donate_attr"),
    [
        (
            "trodestrack.models.ekf",
            "_extended_kalman_filter_jit",
            "EXTENDED_KALMAN_FILTER_STATIC_ARGNAMES",
            "EXTENDED_KALMAN_FILTER_DONATE_ARGNUMS",
        ),
        (
            "trodestrack.runtime.offline",
            "_rts_smoother_jit",
            "RTS_SMOOTHER_STATIC_ARGNAMES",
            "RTS_SMOOTHER_DONATE_ARGNUMS",
        ),
    ],
)
def test_hot_path_modules_expose_jit_metadata(
    module_name, jit_attr, static_attr, donate_attr
):
    """Modules must expose compiled JIT entry points and metadata."""
    module = importlib.import_module(module_name)

    assert hasattr(
        module, jit_attr
    ), f"{module_name} should define compiled JIT helper `{jit_attr}` for hot path execution"
    assert callable(
        getattr(module, jit_attr)
    ), f"{module_name}.{jit_attr} should be callable"

    assert hasattr(
        module, static_attr
    ), f"{module_name} should declare static argnames via `{static_attr}`"
    static_argnames = getattr(module, static_attr)
    assert (
        "layout" in static_argnames
    ), f"{module_name} must treat `layout` as static arg; found {static_argnames!r}"
    if module_name == "trodestrack.models.ekf":
        assert (
            "config_for_filter" in static_argnames
        ), f"{module_name} must treat config as static; got {static_argnames!r}"
    if module_name == "trodestrack.runtime.offline":
        assert (
            "ekf_config" in static_argnames
        ), f"{module_name} must treat ekf_config as static; got {static_argnames!r}"
        assert (
            "num_iter" in static_argnames
        ), f"{module_name} must treat num_iter as static; got {static_argnames!r}"

    assert hasattr(
        module, donate_attr
    ), f"{module_name} should declare donated argnums via `{donate_attr}`"
    donate_argnums = getattr(module, donate_attr)
    assert isinstance(
        donate_argnums, tuple
    ), f"{module_name}.{donate_attr} must be a tuple of donated argument indices"

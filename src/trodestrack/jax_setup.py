"""JAX configuration setup for numerical stability."""

from jax import config as _jax_config

# Enforce 64-bit globally for numerical stability.
_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

from trodestrack.models.filter_common import confidence_to_R_diagonal


def test_confidence_none_defaults_to_base_noise():
    base = 0.01
    diag = confidence_to_R_diagonal(None, base=base, size=4)
    assert jnp.allclose(diag, jnp.full(4, base))


def test_low_confidence_increases_noise():
    base = 0.01
    conf = jnp.array([1.0, 0.5, 0.1, 0.01])
    diag = confidence_to_R_diagonal(conf, base=base, size=4)
    # R = base / conf (clipped), so lower conf => larger noise
    expected = jnp.array([base / 1.0, base / 0.5, base / 0.1, base / 0.01])
    assert jnp.allclose(diag, expected)


def test_confidence_clipped_minimum():
    base = 0.02
    conf = jnp.array([0.0, 1e-6, 1e-3, 1.0])
    diag = confidence_to_R_diagonal(conf, base=base, size=4, clip_min=1e-2)
    expected = jnp.array([base / 1e-2, base / 1e-2, base / 1e-2, base / 1.0])
    assert jnp.allclose(diag, expected)

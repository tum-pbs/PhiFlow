"""
*Experimental* Jax integration.

Importing this module registers the Jax backend with `phi.math`.
Without this, Jax tensors cannot be handled by `phi.math` functions.

To make Jax the default backend, import `phi.jax.flow`.
"""
from phi import math as _math

try:
    from ._jax_backend import JAX_BACKEND

    JAX_BACKEND = JAX_BACKEND  # to show up in pdoc
    """Backend for Jax operations."""

    _math.backend.BACKENDS.append(JAX_BACKEND)
except ImportError:
    JAX_BACKEND = None

__all__ = [key for key in globals().keys() if not key.startswith('_')]

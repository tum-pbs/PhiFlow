"""
Jax integration.

Importing this module registers the Jax backend with `phi.math`.
Without this, Jax tensors cannot be handled by `phi.math` functions.

To make Jax the default backend, import `phi.jax.flow`.
"""
from phi import math as _math

from ._jax_backend import JaxBackend as _JaxBackend

JAX = _JaxBackend()
"""Backend for Jax operations."""

_math.backend.BACKENDS.append(JAX)

__all__ = [key for key in globals().keys() if not key.startswith('_')]

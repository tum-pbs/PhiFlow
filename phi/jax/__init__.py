"""
Jax integration.

Importing this module registers the Jax backend with `phiml.math`.
Without this, Jax tensors cannot be handled by `phiml.math` functions.

To make Jax the default backend, import `phi.jax.flow`.
"""
from phiml.backend.jax import JAX

__all__ = [key for key in globals().keys() if not key.startswith('_')]

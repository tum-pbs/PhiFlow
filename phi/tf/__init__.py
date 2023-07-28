"""
TensorFlow integration.

Importing this module registers the TensorFlow backend with `phiml.math`.
Without this, TensorFlow tensors cannot be handled by `phiml.math` functions.

To make TensorFlow the default backend, import `phi.tf.flow`.
"""
from phiml.backend.tensorflow import TENSORFLOW

__all__ = [key for key in globals().keys() if not key.startswith('_')]

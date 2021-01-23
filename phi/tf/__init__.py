"""
TensorFlow integration.

Importing this module registers the TensorFlow backend with `phi.math`.
Without this, TensorFlow tensors cannot be handled by `phi.math` functions.

To make TensorFlow the default backend, import `phi.tf.flow`.
"""
from phi import math
from ._tf_backend import TF_BACKEND
from ._util import GradientTape, gradients, variable, constant

TF_BACKEND = TF_BACKEND  # to show up in pdoc
"""Backend for TensorFlow operations."""

math.backend.BACKENDS.append(TF_BACKEND)

__all__ = [key for key in globals().keys() if not key.startswith('_')]

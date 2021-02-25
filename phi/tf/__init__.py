"""
TensorFlow integration.

Importing this module registers the TensorFlow backend with `phi.math`.
Without this, TensorFlow tensors cannot be handled by `phi.math` functions.

To make TensorFlow the default backend, import `phi.tf.flow`.
"""
from phi import math as _math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors

from ._tf_backend import TF_BACKEND

TF_BACKEND = TF_BACKEND  # to show up in pdoc
"""Backend for TensorFlow operations."""

_math.backend.BACKENDS.append(TF_BACKEND)

__all__ = [key for key in globals().keys() if not key.startswith('_')]

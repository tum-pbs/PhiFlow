"""
TensorFlow integration.

Importing this module registers the TensorFlow backend with `phi.math`.
Without this, TensorFlow tensors cannot be handled by `phi.math` functions.

To make TensorFlow the default backend, import `phi.tf.flow`.
"""
from phi import math
from .tf_backend import TF_BACKEND

math.backend.BACKENDS.append(TF_BACKEND)

"""
TensorFlow integration.

Importing this module registers the TensorFlow backend with `phi.math`.
Without this, TensorFlow tensors cannot be handled by `phi.math` functions.

To make TensorFlow the default backend, import `phi.tf.flow`.
"""
import platform as _platform
import warnings as _warnings

from phi import math as _math
import os
import tensorflow as _tf

from ..math.backend import PHI_LOGGER as _LOGGER

if _tf.__version__.startswith('1.'):
    raise ImportError(f"PhiFlow 2.x and newer requires TensorFlow 2 but found TensorFlow {_tf.__version__}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors
if _platform.system().lower() == 'windows':  # prevent Blas GEMM launch failed on Windows
    for i, device in enumerate(_tf.config.list_physical_devices('GPU')):
        _tf.config.experimental.set_memory_growth(device, True)
        _LOGGER.info(f"phi.tf: Setting memory_growth on GPU {i} to True to prevent Blas errors")

from ._compile_cuda import compile_cuda_ops

from ._tf_backend import TFBackend as _TFBackend

TENSORFLOW = _TFBackend()
"""Backend for TensorFlow operations."""

_math.backend.BACKENDS.append(TENSORFLOW)

__all__ = [key for key in globals().keys() if not key.startswith('_')]

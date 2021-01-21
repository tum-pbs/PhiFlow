"""
TensorFlow integration.
"""
from phi import math
from .tf_backend import TF_BACKEND

math.backend.BACKENDS.append(TF_BACKEND)

"""
PyTorch integration.
"""
from phi import math
from .torch_backend import TorchBackend

TORCH_BACKEND = TorchBackend()
math.backend.BACKENDS.append(TORCH_BACKEND)

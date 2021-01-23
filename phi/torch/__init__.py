"""
PyTorch integration.

Importing this module registers the PyTorch backend with `phi.math`.
Without this, PyTorch tensors cannot be handled by `phi.math` functions.

To make PyTorch the default backend, import `phi.torch.flow`.
"""
from phi import math
from .torch_backend import TorchBackend

TORCH_BACKEND = TorchBackend()
math.backend.BACKENDS.append(TORCH_BACKEND)

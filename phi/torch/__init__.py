"""
PyTorch integration.

Importing this module registers the PyTorch backend with `phi.math`.
Without this, PyTorch tensors cannot be handled by `phi.math` functions.

To make PyTorch the default backend, import `phi.torch.flow`.
"""
from phi import math
from ._torch_backend import TORCH_BACKEND
from ._torch_util import variable

TORCH_BACKEND = TORCH_BACKEND  # to show up in pdoc
"""Backend for PyTorch operations."""

math.backend.BACKENDS.append(TORCH_BACKEND)

__all__ = [key for key in globals().keys() if not key.startswith('_')]

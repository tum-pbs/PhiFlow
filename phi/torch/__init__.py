"""
PyTorch integration.

Importing this module registers the PyTorch backend with `phiml.math`.
Without this, PyTorch tensors cannot be handled by `phiml.math` functions.

To make PyTorch the default backend, import `phi.torch.flow`.
"""
from phiml.backend.torch import TORCH

__all__ = [key for key in globals().keys() if not key.startswith('_')]

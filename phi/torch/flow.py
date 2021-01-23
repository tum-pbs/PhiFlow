# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for PyTorch mode.

Importing this module registers the PyTorch backend as the default backend.
New tensors created via `phi.math` functions will be backed by PyTorch tensors.
"""

import torch
from phi.flow import *
from .torch_util import *
from .torch_app import *
from .torch_backend import TorchBackend

from phi.math import backend
backend.set_global_default_backend(TORCH_BACKEND)

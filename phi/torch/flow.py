# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for PyTorch mode.

Importing this module registers the PyTorch backend as the default backend.
New tensors created via `phi.math` functions will be backed by PyTorch tensors.
"""

import torch
from phi.flow import *
from ._torch_util import variable
from ._torch_backend import TORCH_BACKEND

from phi.math import backend
backend.set_global_default_backend(TORCH_BACKEND)

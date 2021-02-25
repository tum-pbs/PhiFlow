# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for PyTorch mode.

Extends the import `from phi.flow import *` by PyTorch-related functions and modules.

The following TensorFlow modules are included: `torch`, `torchf` (short for `torch.nn.functional`).

Importing this module registers the PyTorch backend as the default backend.
New tensors created via `phi.math` functions will be backed by PyTorch tensors.

See `phi.flow`, `phi.tf.flow`.
"""

from phi.flow import *
from ._torch_util import variable
from ._torch_backend import TORCH_BACKEND
import torch
import torch.nn.functional as torchf

backend.set_global_default_backend(TORCH_BACKEND)

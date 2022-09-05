# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for PyTorch mode.

Extends the import `from phi.flow import *` by PyTorch-related functions and modules.

The following PyTorch modules are included: `torch`, *torch.nn.functional* as `torchf`, `optim`.

Importing this module registers the PyTorch backend as the default backend unless called within a backend context.
New tensors created via `phi.math` functions will be backed by PyTorch tensors.

See `phi.flow`, `phi.tf.flow`, `phi.jax.flow`.
"""

from phi.flow import *
from . import TORCH
from .nets import parameter_count, get_parameters, save_state, load_state, dense_net, u_net, update_weights, adam, conv_net, res_net, sgd, sgd as SGD, rmsprop, adagrad, conv_classifier

import torch
import torch.nn.functional as torchf
import torch.optim as optim


if not backend.context_backend():
    backend.set_global_default_backend(TORCH)
else:
    from ..math.backend import PHI_LOGGER as _LOGGER
    _LOGGER.warning(f"Importing '{__name__}' within a backend context will not set the default backend.")

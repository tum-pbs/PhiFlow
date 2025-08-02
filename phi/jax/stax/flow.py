# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for Jax + Stax mode.

Extends the import `from phi.flow import *` by Jax-related functions and modules.

The following Jax modules are included: `jax`, `jax.numpy` as `jnp`, `jax.scipy` as `jsp`.

Importing this module registers the Jax backend as the default backend unless called within a backend context.
New tensors created via `phiml.math` functions will be backed by Jax tensors.

See `phi.flow`, `phi.torch.flow`, `phi.tf.flow`.
"""
from ..flow import *

from . import nets
from .nets import (
    parameter_count, get_parameters, save_state, load_state,
    dense_net, u_net, conv_net, res_net, conv_classifier, invertible_net, coupling_layer,
    update_weights,
    adam, sgd, sgd as SGD, rmsprop, adagrad, # set_learning_rate not compatible with paradigm
)
from phiml.nn import train

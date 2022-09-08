# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for Jax + Stax mode.

Extends the import `from phi.flow import *` by Jax-related functions and modules.

The following Jax modules are included: `jax`, `jax.numpy` as `jnp`, `jax.scipy` as `jsp`.

Importing this module registers the Jax backend as the default backend unless called within a backend context.
New tensors created via `phi.math` functions will be backed by Jax tensors.

See `phi.flow`, `phi.torch.flow`, `phi.tf.flow`.
"""
from ..flow import *
from .nets import parameter_count, get_parameters, save_state, load_state, dense_net, u_net, update_weights, adam, conv_net, res_net, adagrad, rmsprop, sgd, sgd as SGD, conv_classifier
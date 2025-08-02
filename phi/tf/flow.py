# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for TensorFlow mode.

Extends the import `from phi.flow import *` by TensorFlow-related functions and modules.

The following TensorFlow modules are included: `tensorflow` / `tf`, `keras`, `layers`.

Importing this module registers the TensorFlow backend as the default backend unless called within a backend context.
New tensors created via `phiml.math` functions will be backed by TensorFlow tensors.

See `phi.flow`, `phi.torch.flow`, `phi.jax.flow`.
"""

from phi.flow import *
from . import TENSORFLOW

from . import nets
from .nets import (
    parameter_count, get_parameters, save_state, load_state,
    dense_net, u_net, conv_net, res_net, conv_classifier, invertible_net,
    update_weights,
    adam, sgd, sgd as SGD, rmsprop, adagrad, set_learning_rate, get_learning_rate,
)
from phiml.nn import train

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from phiml.backend import ML_LOGGER as _LOGGER

tf = tensorflow

if not backend.context_backend():
    backend.set_global_default_backend(TENSORFLOW)
else:
    _LOGGER.warn(f"Importing '{__name__}' within a backend context will not set the default backend.")
# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for TensorFlow mode.

Extends the import `from phi.flow import *` by TensorFlow-related functions and modules.

The following TensorFlow modules are included: `tensorflow` / `tf`, `keras`, `layers`.

Importing this module registers the TensorFlow backend as the default backend.
New tensors created via `phi.math` functions will be backed by TensorFlow tensors.

See `phi.flow`, `phi.torch.flow`.
"""

from phi.flow import *
from ._util import GradientTape, gradients, variable, constant
from ._tf_backend import TF_BACKEND
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

tf = tensorflow
backend.set_global_default_backend(TF_BACKEND)

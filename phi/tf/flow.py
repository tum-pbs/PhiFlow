# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for TensorFlow mode.

Importing this module registers the TensorFlow backend as the default backend.
New tensors created via `phi.math` functions will be backed by TensorFlow tensors.
"""

from phi.flow import *
from ._util import *
from ._tf_backend import TF_BACKEND
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

tf = tensorflow
math.backend.set_global_default_backend(TF_BACKEND)

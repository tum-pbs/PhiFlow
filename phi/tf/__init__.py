import warnings
from phi import math

import tensorflow

if int(tensorflow.__version__[0]) > 1:
    warnings.warn('TensorFlow 2 is not fully supported by PhiFlow.')
    tensorflow = tensorflow.compat.v1
    tensorflow.disable_eager_execution()

tf = tensorflow

from .tf_backend import TFBackend

TF_BACKEND = TFBackend()
math.DYNAMIC_BACKEND.add_backend(TF_BACKEND)

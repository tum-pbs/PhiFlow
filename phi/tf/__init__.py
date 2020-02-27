import tensorflow as tf

from phi import math

from .tf_backend import TFBackend


TF_BACKEND = TFBackend()
math.DYNAMIC_BACKEND.add_backend(TF_BACKEND)


if int(tf.__version__[0]) > 1:
    import warnings
    warnings.warn('TensorFlow 2 is not fully supported by PhiFlow.')

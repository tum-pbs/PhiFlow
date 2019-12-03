import tensorflow as tf
from phi import math


if int(tf.__version__[0]) > 1:
    import warnings
    warnings.warn('TensorFlow 2 is not fully supported by PhiFlow.')


def _load_tensorflow():
    """
    Internal function to register the TensorFlow backend.
    This function is called automatically once a TFSimulation is instantiated.

    :return: True if TensorFlow could be imported, else False
    """
    try:
        from phi.math.tensorflow_backend import TFBackend
        for b in math.backend.backends:
            if isinstance(b, TFBackend):
                return True
        math.backend.backends.append(TFBackend())
        return True
    except BaseException as e:
        import logging
        logging.fatal("Failed to load TensorFlow backend. Error: %s" % e)
        print("Failed to load TensorFlow backend. Error: %s" % e)
        return False


_load_tensorflow()

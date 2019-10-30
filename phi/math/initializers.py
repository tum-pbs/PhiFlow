import numpy as np
from numbers import Number

from phi import struct
from .nd import fftfreq
from .backend import backend as math


def _is_python_shape(obj):
    if not isinstance(obj, (tuple, list, np.ndarray)):
        return False
    for element in obj:
        if not isinstance(element, Number) and element is not None:
            return False
    return True


def _none_to_one(shape):
    result = list(map(lambda val: 1 if val is None else val, shape))
    return result


def zeros(shape, dtype=np.float32):
    f = lambda s: np.zeros(_none_to_one(s), dtype=dtype)
    return struct.map(f, shape, leaf_condition=_is_python_shape)


def zeros_like(object):
    f = lambda tensor: math.zeros_like(tensor)
    return struct.map(f, object, leaf_condition=_is_python_shape)


def ones(shape, dtype=np.float32):
    f = lambda s: np.ones(_none_to_one(s), dtype)
    return struct.map(f, shape, leaf_condition=_is_python_shape)


def randn(shape, dtype=np.float32):
    f = lambda s: np.random.randn(*_none_to_one(s)).astype(dtype)
    return struct.map(f, shape, leaf_condition=_is_python_shape)


def randfreq(shape, dtype=np.float32, power=8):
    def genarray(shape):
        fft = randn(shape, dtype) + 1j * randn(shape, dtype)
        k = fftfreq(shape[1:-1], mode='absolute')
        shape_fac = math.sqrt(math.mean(shape[1:-1]))  # 16: 4, 64: 8, 256: 24,
        print('Shape fac: %f' % shape_fac)
        fft *= (1 / (k + 1)) ** power * power * shape_fac
        array = math.ifft(fft)
        array = array.astype(dtype)
        return array
    return struct.map(genarray, shape, leaf_condition=_is_python_shape)

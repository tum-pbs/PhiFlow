from phi import struct
from .nd import upsample2x
from .base import backend as math
import numpy as np
from numbers import Number


def _is_python_shape(obj):
    if not isinstance(obj, (tuple, list)): return False
    for element in obj:
        if not isinstance(element, Number) and element is not None: return False
    return True


def _none_to_one(shape):
    result = list(map(lambda val: 1 if val is None else val, shape))
    return result


def zeros(shape, dtype=np.float32):
    f = lambda s: np.zeros(_none_to_one(s), dtype)
    return struct.map(f, shape, leaf_condition=_is_python_shape)


def zeros_like(object):
    f = lambda tensor: math.zeros_like(tensor)
    return struct.map(f, object, leaf_condition=_is_python_shape)


def ones(shape, dtype=np.float32):
    f = lambda s: np.ones(_none_to_one(s), dtype)
    return struct.map(f, shape, leaf_condition=_is_python_shape)


def randn(mean=0, sigma=1, levels=(1.0,)):  # TODO pass mean, sigma, doesn't correctly scale staggered grids
    def randn_impl(shape, dtype=np.float32):
        return struct.map(lambda s: _random_tensor(s, levels, dtype) * sigma + mean, shape, leaf_condition=_is_python_shape)
    return randn_impl


def _random_tensor(shape, levels, dtype):
    shape = _none_to_one(shape)
    result = 0
    for i in range(len(levels)): # high-res first
        lowres_shape = np.array(shape)
        lowres_shape[1:-1] //= 2 ** i
        rnd = np.random.randn(*lowres_shape) * levels[i]
        for j in range(i):
            rnd = upsample2x(rnd)
        result = result + rnd
    return result.astype(dtype)
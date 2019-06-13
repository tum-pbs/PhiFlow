from .struct import *
from .nd import upsample2x
from .base import backend as math
import numpy as np


def zeros(shape, dtype=np.float32):
    return Struct.flatmap(lambda s: np.zeros(s, dtype), shape)


def zeros_like(object):
    return Struct.flatmap(lambda tensor: math.zeros_like(tensor), object)


def ones(shape, dtype=np.float32):
    return Struct.flatmap(lambda s: np.ones(s, dtype), shape)


def empty(shape, dtype=np.float32):
    return Struct.flatmap(lambda s: np.empty(s, dtype), shape)


def empty_like(object):
    return Struct.flatmap(lambda tensor: np.empty_like(tensor), object)


def randn(shape, levels=(1.0,)):
    return Struct.flatmap(lambda s: _random_tensor(s, levels), shape)


def _random_tensor(shape, levels):
    result = 0
    for i in range(len(levels)): # high-res first
        lowres_shape = np.array(shape)
        lowres_shape[1:-1] //= 2 ** i
        rnd = np.random.randn(*lowres_shape) * levels[i]
        for j in range(i):
            rnd = upsample2x(rnd)
        result = result + rnd
    return result
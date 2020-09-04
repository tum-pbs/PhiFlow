import warnings
from numbers import Number
import numpy as np

from phi import struct
from phi.backend.dynamic_backend import DYNAMIC_BACKEND as math
from phi.backend.dynamic_backend import NoBackendFound
from phi.struct.functions import mappable

from .nd import fftfreq


@mappable(item_condition=struct.ALL_ITEMS, content_type=type)
def types(x):
    warnings.warn("math.types is deprecated. Use struct.dtype isntead.", DeprecationWarning)
    try:
        return math.dtype(x)
    except NoBackendFound:
        return type(x)


def is_static_shape(obj):
    if not isinstance(obj, (tuple, list, np.ndarray)):
        return False
    for element in obj:
        if not isinstance(element, Number) and element is not None:
            return False
    return True


def _none_to_one(shape):
    result = list(map(lambda val: 1 if val is None else val, shape))
    return result


@mappable(leaf_condition=is_static_shape)
def zeros(shape, dtype=None):
    if dtype is not None:
        return np.zeros(_none_to_one(shape), dtype=dtype)
    else:
        return math.to_float(np.zeros(_none_to_one(shape), np.int8))


@mappable(leaf_condition=is_static_shape)
def ones(shape, dtype=None):
    if dtype is not None:
        return np.ones(_none_to_one(shape), dtype)
    else:
        return math.to_float(np.ones(_none_to_one(shape), np.int8))


@mappable(leaf_condition=is_static_shape)
def randn(shape, dtype=None):
    array = np.random.randn(*_none_to_one(shape))
    if dtype is not None:
        return array.astype(dtype)
    else:
        return math.to_float(array)


def randfreq(shape, dtype=None, power=8):
    warnings.warn('randfreq() is deprecated. Use Noise() instead.')

    def genarray(shape):
        fft = randn(shape, dtype) + 1j * randn(shape, dtype)
        k = fftfreq(shape[1:-1], mode='absolute')
        shape_fac = math.sqrt(math.mean(shape[1:-1]))  # 16: 4, 64: 8, 256: 24,
        fft *= (1 / (k + 1)) ** power * power * shape_fac
        array = math.real(math.ifft(fft))
        if dtype is not None:
            return array.astype(dtype)
        else:
            return math.to_float(array)
    return struct.map(genarray, shape, leaf_condition=is_static_shape)


def interpolate(alpha, x, y, clip_alpha=True):
    """
    Linear interpolation between x and y according to the value of `alpha`

    :param alpha: blending between `x` (`alpha=0`) and `y` (`alpha=1`)
    :param x: value for `alpha=0`, must be compatible with y and alpha
    :param y: value for `alpha=1`, must be compatible with x and alpha
    :param clip_alpha:
    :return:
    """
    if clip_alpha:
        alpha = math.clip(alpha, 0, 1)
    return x * (1 - alpha) + y * alpha

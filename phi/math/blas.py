# coding=utf-8
import warnings

from ..backend.dynamic_backend import DYNAMIC_BACKEND as math
from .optim import conjugate_gradient as new_cg


def conjugate_gradient(k, apply_A, initial_x=None, accuracy=1e-5, max_iterations=1024, back_prop=False):
    warnings.warn("conjugate_gradient from phi.math.blas is deprecated. Use phi.math.optim.conjugate_gradient instead.", DeprecationWarning)
    if initial_x is None:
        initial_x = math.zeros_like(k)
    result = new_cg(function=apply_A, y=k, x0=initial_x, accuracy=accuracy, max_iterations=max_iterations, back_prop=back_prop)
    return result.x, result.iterations

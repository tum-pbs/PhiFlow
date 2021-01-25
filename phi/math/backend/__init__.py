"""
Low-level library wrappers for delegating vector operations.
"""

from contextlib import contextmanager

from ._dtype import DType, from_numpy_dtype, to_numpy_dtype
from ._backend import Backend
from ._scipy_backend import SCIPY_BACKEND, SciPyBackend
from ._optim import Solve, LinearSolve

BACKENDS = [SCIPY_BACKEND]
_DEFAULT = [SCIPY_BACKEND]  # [0] = global default, [1:] from 'with' blocks
_PRECISION = [32]  # [0] = global precision in bits, [1:] from 'with' blocks


def choose_backend(*values, prefer_default=False, raise_error=True) -> Backend:
    """
    Selects a suitable backend to handle the given values.

    This function is used by most math functions operating on `Tensor` objects to delegate the actual computations.

    Args:
        *values:
        prefer_default: if True, selects the default backend assuming it can handle handle the values, see `default_backend()`.
        raise_error: Determines the behavior of this function if no backend can handle the given values.
            If True, raises a `NoBackendFound` error, else returns `None`.

    Returns:
        the selected backend
    """
    # --- Default Backend has priority ---
    if _is_specific(_DEFAULT[-1], values):
        return _DEFAULT[-1]
    if prefer_default and _is_applicable(_DEFAULT[-1], values):
        return _DEFAULT[-1]
    # --- Filter out non-applicable ---
    backends = [backend for backend in BACKENDS if _is_applicable(backend, values)]
    if len(backends) == 0:
        if raise_error:
            raise NoBackendFound('No backend found for values %s; registered backends are %s' % (values, BACKENDS))
        else:
            return None
    # --- Native tensors? ---
    for backend in backends:
        if _is_specific(backend, values):
            return backend
    else:
        return backends[0]


class NoBackendFound(Exception):
    """
    Thrown by `choose_backend` if no backend can handle the given values.
    """

    def __init__(self, msg):
        Exception.__init__(self, msg)


def default_backend():
    """
    The default backend is preferred by `choose_backend()`.

    The default backend can be set globally using `set_global_default_backend()` and locally using `with backend:`.

    Returns:
        current default backend
    """
    return _DEFAULT[-1]


def set_global_default_backend(backend: Backend):
    """
    Sets the given backend as default.
    This setting can be overridden using `with backend:`.

    See `default_backend()`, `choose_backend()`.

    Args:
        backend: backend to set as default
    """
    assert isinstance(backend, Backend)
    _DEFAULT[0] = backend


def set_global_precision(floating_point_bits):
    """
    Sets the floating point precision of DYNAMIC_BACKEND which affects all registered backends.
    
    If `floating_point_bits` is an integer, all floating point tensors created henceforth will be of the corresponding data type, float16, float32 or float64.
    Operations may also convert floating point values to this precision, even if the input had a different precision.
    
    If `floating_point_bits` is None, new tensors will default to float32 unless specified otherwise.
    The output of math operations has the same precision as its inputs.

    Args:
      floating_point_bits: one of (16, 32, 64, None)
    """
    _PRECISION[0] = floating_point_bits


def get_precision() -> int:
    """
    Gets the current target floating point precision in bits.
    The precision can be set globally using `set_global_precision()` or locally using `with precision(p):`.

    Any Backend method may convert floating point values to this precision, even if the input had a different precision.

    Returns:
        16 for half, 32 for single, 64 for double
    """
    return _PRECISION[-1]


@contextmanager
def precision(floating_point_bits):
    """
    Sets the floating point precision for the local context.

    Usage: `with precision(p):`

    This overrides the global setting, see `set_global_precision()`.

    Args:
        floating_point_bits: 16 for half, 32 for single, 64 for double
    """
    _PRECISION.append(floating_point_bits)
    try:
        yield None
    finally:
        _PRECISION.pop(-1)


# Backend choice utility functions

def _is_applicable(backend, values):
    for value in values:
        if not backend.is_tensor(value, only_native=False):
            return False
    return True


def _is_specific(backend, values):
    for value in values:
        if backend.is_tensor(value, only_native=True):
            return True
    return False

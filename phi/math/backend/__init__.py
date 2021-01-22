from contextlib import contextmanager

from ._dtype import DType, from_numpy_dtype, to_numpy_dtype
from ._backend import Backend
from ._scipy_backend import SCIPY_BACKEND, SciPyBackend
from ._optim import Solve, LinearSolve

BACKENDS = [SCIPY_BACKEND]
_DEFAULT = [SCIPY_BACKEND]  # [0] = global default, [1:] from 'with' blocks
_PRECISION = [32]  # [0] = global precision in bits, [1:] from 'with' blocks


def choose_backend(*values, prefer_default=False, raise_error=True) -> Backend:
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

    def __init__(self, msg):
        Exception.__init__(self, msg)


def default_backend():
    return _DEFAULT[-1]


def set_global_default_backend(backend: Backend):
    assert isinstance(backend, Backend)
    _DEFAULT[0] = backend


def set_global_precision(floating_point_bits):
    """
    Sets the floating point precision of DYNAMIC_BACKEND which affects all registered backends.

    If `floating_point_bits` is an integer, all floating point tensors created henceforth will be of the corresponding data type, float16, float32 or float64.
    Operations may also convert floating point values to this precision, even if the input had a different precision.

    If `floating_point_bits` is None, new tensors will default to float32 unless specified otherwise.
    The output of math operations has the same precision as its inputs.

    :param floating_point_bits: one of (16, 32, 64, None)
    """
    _PRECISION[0] = floating_point_bits


def get_precision() -> int:
    """
    Any Backend method may convert floating point values to this precision, even if the input had a different precision.
    """
    return _PRECISION[-1]


@contextmanager
def precision(floating_point_bits):
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

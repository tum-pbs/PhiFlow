"""
Low-level library wrappers for delegating vector operations.
"""

from ._dtype import DType, from_numpy_dtype, to_numpy_dtype
from ._backend import (
    Backend, choose_backend, NoBackendFound,
    ComputeDevice,
    default_backend, set_global_default_backend, BACKENDS, _DEFAULT,
    get_precision, precision, set_global_precision,
)
from ._scipy_backend import SCIPY_BACKEND
from ._optim import Solve, LinearSolve
from ._profile import Profile, get_current_profile, profile, profile_function


BACKENDS.append(SCIPY_BACKEND)
_DEFAULT.append(SCIPY_BACKEND)

BACKENDS = BACKENDS
""" Global list of all registered backends. Register a `Backend` by adding it to the list. """


__all__ = [key for key in globals().keys() if not key.startswith('_')]

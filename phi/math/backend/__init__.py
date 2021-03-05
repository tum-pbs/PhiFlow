"""
Low-level library wrappers for delegating vector operations.
"""

from ._dtype import DType, from_numpy_dtype, to_numpy_dtype
from ._backend import (
    Backend, choose_backend, NoBackendFound,
    ComputeDevice,
    default_backend, set_global_default_backend, BACKENDS, context_backend, _DEFAULT,
    get_precision, precision, set_global_precision,
)
from ._numpy_backend import NUMPY_BACKEND
from ._optim import Solve, LinearSolve
from ._profile import Profile, get_current_profile, profile, profile_function


BACKENDS.append(NUMPY_BACKEND)
_DEFAULT.append(NUMPY_BACKEND)

BACKENDS = BACKENDS  # to show up in pdoc
""" Global list of all registered backends. Register a `Backend` by adding it to the list. """

SCIPY_BACKEND = NUMPY_BACKEND  # to show up in pdoc
""" Alias for `NUMPY_BACKEND` """
NUMPY_BACKEND = NUMPY_BACKEND  # to show up in pdoc
"""Default backend for NumPy arrays and SciPy objects."""


__all__ = [key for key in globals().keys() if not key.startswith('_')]

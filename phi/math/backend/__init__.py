"""
Low-level library wrappers for delegating vector operations.
"""
from ._backend import (
    Backend, choose_backend, NoBackendFound,
    ComputeDevice,
    default_backend, set_global_default_backend, BACKENDS, context_backend, _DEFAULT,
    get_precision, precision, set_global_precision,
    convert,
    PHI_LOGGER,
)
from ._numpy_backend import NumPyBackend as _NumPyBackend
from ._profile import Profile, get_current_profile, profile, profile_function


NUMPY = _NumPyBackend()
"""Default backend for NumPy arrays and SciPy objects."""
BACKENDS.append(NUMPY)
_DEFAULT.append(NUMPY)


__all__ = [key for key in globals().keys() if not key.startswith('_')]

__pdoc__ = {
    'ComputeDevice.__init__': False,
    'NoBackendFound.__init__': False,
    'Profile.__init__': False,
}

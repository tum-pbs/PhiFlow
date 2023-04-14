"""
Low-level library wrappers for delegating vector operations.
"""
import warnings

from ._backend import (
    Backend, choose_backend, NoBackendFound,
    ComputeDevice,
    default_backend, set_global_default_backend, BACKENDS, context_backend, _DEFAULT,
    get_precision, precision, set_global_precision,
    convert,
    PHI_LOGGER,
)
from ._profile import Profile, get_current_profile, profile, profile_function

try:
    import numpy as _np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False
    warnings.warn("NumPy is not installed.", ImportWarning)

if _NP_AVAILABLE:
    from ._numpy_backend import NumPyBackend as _NumPyBackend
    NUMPY = _NumPyBackend()
    """Default backend for NumPy arrays and SciPy objects."""
    BACKENDS.append(NUMPY)
    _DEFAULT.append(NUMPY)

from ._object import ObjectBackend as _ObjectBackend
OBJECTS = _ObjectBackend()
BACKENDS.append(OBJECTS)

__all__ = [key for key in globals().keys() if not key.startswith('_')]

__pdoc__ = {
    'ComputeDevice.__init__': False,
    'NoBackendFound.__init__': False,
    'Profile.__init__': False,
}

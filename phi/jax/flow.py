# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import
"""
Standard import for Jax mode without neural network support.

Extends the import `from phi.flow import *` by Jax-related functions and modules.

The following Jax modules are included: `jax`, `jax.numpy` as `jnp`, `jax.scipy` as `jsp`.

Importing this module registers the Jax backend as the default backend unless called within a backend context.
New tensors created via `phi.math` functions will be backed by Jax tensors.

See `phi.flow`, `phi.torch.flow`, `phi.tf.flow`.
"""

from phi.flow import *
from . import JAX
import jax
import jax.numpy as jnp
import jax.scipy as jsp


if not backend.context_backend():
    backend.set_global_default_backend(JAX)
else:
    from ..math.backend import PHI_LOGGER as _LOGGER
    _LOGGER.warning(f"Importing '{__name__}' within a backend context will not set the default backend.")

"""
Open-source simulation toolkit built for optimization and machine learning applications.

Use one of the following imports:

* `from phi.flow import *`  for NumPy mode
* `from phi.tf.flow import *`  for TensorFlow mode
* `from phi.torch.flow import *` for PyTorch mode
* `from phi.jax.flow import *` for Jax mode

Project homepage: https://github.com/tum-pbs/PhiFlow

Documentation overview: https://tum-pbs.github.io/PhiFlow

PyPI: https://pypi.org/project/phiflow/
"""

import os as _os


with open(_os.path.join(_os.path.dirname(__file__), 'VERSION'), 'r') as version_file:
    __version__ = version_file.read()


def verify():
    """
    Checks your configuration for potential problems and prints a summary.

    To run verify without importing `phi`, run the script `tests/verify.py` included in the source distribution.
    """
    import sys
    from ._troubleshoot import assert_minimal_config, troubleshoot
    try:
        assert_minimal_config()
    except AssertionError as fail_err:
        print("\n".join(fail_err.args), file=sys.stderr)
        return
    print(troubleshoot())


def detect_backends() -> tuple:
    """
    Registers all available backends and returns them.
    This includes only backends for which the minimal requirements are fulfilled.

    Returns:
        `tuple` of `phi.math.backend.Backend`
    """
    try:
        from .torch import TORCH
    except ImportError:
        pass
    try:
        from .tf import TENSORFLOW
    except ImportError:
        pass
    try:
        from .jax import JAX
    except ImportError:
        pass
    from .math.backend import BACKENDS
    return tuple(BACKENDS)


def set_logging_level(level='debug'):
    """
    Sets the logging level for PhiFlow functions.

    Args:
        level: Logging level, one of `'critical', 'fatal', 'error', 'warning', 'info', 'debug'`
    """
    from phi.math.backend import PHI_LOGGER
    PHI_LOGGER.setLevel(level.upper())

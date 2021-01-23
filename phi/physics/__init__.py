"""
Contains built-in physics functions, e.g. for fluids.

Main class: `Domain`

See the `phi.physics` module documentation at https://github.com/tum-pbs/PhiFlow/blob/develop/phi/physics
"""
from ._boundaries import Domain, Material, OPEN, CLOSED, PERIODIC

__all__ = [key for key in globals().keys() if not key.startswith('_')]

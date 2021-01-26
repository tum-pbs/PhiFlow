"""
Contains built-in physics functions, e.g. for fluids.

Main class: `Domain`

See the `phi.physics` module documentation at https://tum-pbs.github.io/PhiFlow/Physics.html
"""
from ._boundaries import Domain, Material, OPEN, CLOSED, PERIODIC, Obstacle

__all__ = [key for key in globals().keys() if not key.startswith('_')]

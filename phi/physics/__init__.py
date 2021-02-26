"""
Contains built-in physics functions, e.g. for fluids.
The actual physics functions are located in the sub-modules of `phi.physics`.
A common trait of many physics functions is the time increment (`dt`) argument.

Main class: `Domain`

See the `phi.physics` module documentation at https://tum-pbs.github.io/PhiFlow/Physics.html
"""
from ._boundaries import Domain, OPEN, SLIPPERY, STICKY, PERIODIC, Obstacle

OPEN: dict = OPEN  # to show up in pdoc
""" Open boundary conditions take the value 0 outside the valid region. See https://tum-pbs.github.io/PhiFlow/Physics.html#boundary-conditions """
CLOSED: dict = SLIPPERY  # to show up in pdoc
""" Closed boundary conditions extrapolate the closest valid value. See https://tum-pbs.github.io/PhiFlow/Physics.html#boundary-conditions """
PERIODIC: dict = PERIODIC  # to show up in pdoc
""" Periodic `Domain` boundary conditions. See https://tum-pbs.github.io/PhiFlow/Physics.html#boundary-conditions """

__all__ = [key for key in globals().keys() if not key.startswith('_')]

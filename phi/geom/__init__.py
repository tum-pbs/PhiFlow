"""
Differentiable geometry.

See the `phi.geom` module documentation at https://github.com/tum-pbs/PhiFlow/blob/develop/phi/geom
"""

from ._geom import Geometry, assert_same_rank
from ._union import union  # Union is private
from ._box import Box, GridCell, AbstractBox
from ._sphere import Sphere

__all__ = [key for key in globals().keys() if not key.startswith('_')]

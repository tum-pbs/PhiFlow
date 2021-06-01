"""
Differentiable geometry package.

Classes:

* `Geometry` (abstract)
* `Box`
* `Sphere`

See the `phi.geom` module documentation at https://tum-pbs.github.io/PhiFlow/Geometry.html
"""

from ._geom import Geometry, Point, assert_same_rank
from ._union import union
from ._box import Box, GridCell, AbstractBox
from ._sphere import Sphere
from ._stack import stack
from ._geom_math import concat, invert

__all__ = [key for key in globals().keys() if not key.startswith('_')]

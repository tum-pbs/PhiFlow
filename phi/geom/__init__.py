"""
Differentiable geometry package.

Classes:

* `Geometry` (base type)
* `Box`
* `Sphere`

See the `phi.geom` module documentation at https://tum-pbs.github.io/PhiFlow/Geometry.html
"""
from ..math import stack, concat, pack_dims  # for compatibility
from ._functions import normal_from_slope
from ._geom import Geometry, GeometryException, Point, assert_same_rank, invert, rotate
from ._box import Box, BaseBox, Cuboid
from ._sphere import Sphere
from ._grid import UniformGrid
from ._mesh import Mesh, load_su2, mesh_from_numpy
from ._transform import embed, infinite_cylinder
from ._heightmap import Heightmap
from ._geom_ops import union

__all__ = [key for key in globals().keys() if not key.startswith('_')]

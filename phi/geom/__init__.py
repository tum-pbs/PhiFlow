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
from ._geom import Geometry, GeometryException, Point, assert_same_rank, invert, rotate, sample_function
from ._box import Box, BaseBox, Cuboid
from ._sphere import Sphere
from ._grid import UniformGrid, enclosing_grid
from ._graph import Graph, graph
from ._mesh import Mesh, mesh, load_su2, load_gmsh, mesh_from_numpy, build_mesh
from ._transform import embed, infinite_cylinder
from ._heightmap import Heightmap
from ._sdf_grid import SDFGrid, sample_sdf
from ._sdf import SDF, numpy_sdf
from ._geom_ops import union, intersection
from ._convert import surface_mesh, as_sdf
from ._geom_functions import line_trace

__all__ = [key for key in globals().keys() if not key.startswith('_')]

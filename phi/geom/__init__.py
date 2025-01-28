"""
Differentiable geometry package.

Classes:

* `Geometry` (base type)
* `Box`
* `Sphere`

See the `phi.geom` module documentation at https://tum-pbs.github.io/PhiFlow/Geometry.html
"""
from ..math import stack, concat, pack_dims  # for compatibility

# --- Low-level functions ---
from ._geom import Geometry, GeometryException, Point, assert_same_rank, invert, sample_function
from ._functions import normal_from_slope, clip_length, cross, rotation_matrix, rotation_angles, rotation_matrix_from_axis_and_angle, rotation_matrix_from_directions
from ._transform import scale, rotate

# --- Geometry types ---
from ._box import Box, BaseBox, Cuboid, bounding_box
from ._sphere import Sphere
from ._cylinder import Cylinder, cylinder
from ._grid import UniformGrid, enclosing_grid
from ._graph import Graph, graph
from ._mesh import Mesh, mesh, load_su2, load_gmsh, load_stl, mesh_from_numpy, build_mesh
from ._heightmap import Heightmap
from ._sdf_grid import SDFGrid, sample_sdf
from ._sdf import SDF, numpy_sdf
from ._embed import embed, infinite_cylinder

# --- Top-level functions ---
from ._geom_ops import union, intersection
from ._convert import surface_mesh, as_sdf
from ._geom_functions import line_trace, length, squared_length, normalize

__all__ = [key for key in globals().keys() if not key.startswith('_')]

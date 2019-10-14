from .obstacle import *
from .material import *
from phi import math
import numpy as np
from phi.geom import Box



class Grid(struct.Struct):
    __struct__ = struct.Def((), ('_resolution', '_box'))

    def __init__(self, resolution, box=None):
        self._resolution = np.array(resolution)
        if box is not None:
            self._box = box
        else:
            self._box = Box([0 for d in resolution], self._resolution)

    @property
    def resolution(self):
        return self._resolution

    @property
    def box(self):
        return self._box

    @property
    def rank(self):
        return len(self._resolution)

    def cell_index(self, global_position):
        local_position = self._box.global_to_local(global_position) * self._resolution
        position = math.to_int(local_position - 0.5)
        position = math.maximum(0, position)
        position = math.minimum(position, self._resolution-1)
        return position

    def center_points(self):
        idx_zyx = np.meshgrid(*[np.arange(0.5,dim+0.5,1) for dim in self._resolution], indexing="ij")
        return math.expand_dims(math.stack(idx_zyx, axis=-1), 0)

    def staggered_points(self, dimension):
        idx_zyx = np.meshgrid(*[np.arange(0.5,dim+1.5,1) if dim != dimension else np.arange(0,dim+1,1) for dim in self._resolution], indexing="ij")
        return math.expand_dims(math.stack(idx_zyx, axis=-1), 0)


    def indices(self):
        """
    Constructs a grid containing the index-location as components.
    Each index denotes the location within the tensor starting from zero.
    Indices are encoded as vectors in the index tensor.
        :param dtype: a numpy data type (default float32)
        :return: an index tensor of shape (1, spatial dimensions..., spatial rank)
        """
        idx_zyx = np.meshgrid(*[range(dim) for dim in self._resolution], indexing="ij")
        return math.expand_dims(np.stack(idx_zyx, axis=-1))

    @staticmethod
    def equal(grid1, grid2):
        assert isinstance(grid1, Grid), 'Not a grid: %s' % type(grid1)
        assert isinstance(grid2, Grid), 'Not a grid: %s' % type(grid2)
        return grid1._resolution == grid2._resolution and grid1._box == grid2._box





class Domain(Grid):
    __struct__ = Grid.__struct__.extend([], ['_boundaries'])

    def __init__(self, dimensions, boundaries=OPEN, box=None):
        """
Simulation domain that specifies size and boundary conditions.

If all boundary surfaces should have the same behaviour, pass a single Material instance.

To specify the boundary properties per dimension or surface, pass a tuple or list with as many elements as there are spatial dimensions (highest dimension first).
Each element can either be a Material, specifying the faces perpendicular to that axis, or a pair
of Material holding (lower_face_material, upper_face_material).

Examples:

Domain(grid, OPEN) - all surfaces are open

DomainBoundary(grid, boundaries=[(SLIPPY, OPEN), SLIPPY]) - creates a 2D domain with an open top and otherwise solid boundaries

        :param grid: Grid object or 1D tensor specifying the grid dimensions
        :param boundaries: Material or list of Material/Pair of Material
        """
        Grid.__init__(self, dimensions, box=box)
        assert isinstance(boundaries, (Material, list, tuple))
        if isinstance(boundaries, (tuple, list)):
            assert len(boundaries) == self.rank
        self._boundaries = _collapse_equals(boundaries, leaf_type=Material)

    def default_physics(self):
        return STATIC

    @property
    def boundaries(self):
        return self._boundaries

    def _get_paddings(self, material_condition, margin=1):
        true_paddings = [[0, 0] for i in range(self.rank)]
        false_paddings = [[0, 0] for i in range(self.rank)]
        for dim in range(self.rank):
            for upper in (False, True):
                if material_condition(self.surface_material(dim, upper)):
                    true_paddings[dim][upper] = margin
                else:
                    false_paddings[dim][upper] = margin
        return [[0, 0]] + true_paddings + [[0, 0]], [[0, 0]] + false_paddings + [[0, 0]]

    def surface_material(self, dimension=0, upper_boundary=False):
        if isinstance(self._boundaries, Material):
            return self._boundaries
        else:
            dim_boundaries = self._boundaries[dimension]
            if isinstance(dim_boundaries, Material):
                return dim_boundaries
            else:
                return dim_boundaries[upper_boundary]



    # def _friction_mask(self, dt=1): TODO
    #     material.friction_multiplier(dt=dt)


"""
        :param active_mask: (Optional) Scalar channel encoding active cells as ones and inactive (open/obstacle) as zero.
        :param fluid_mask: (Optional) Scalar channel encoding fluid cells as ones and obstacles as zero.
         Has the same dimensions as the divergence channel. If no obstacles are present, None may be passed.
        :param boundaries: DomainBoundary object defining open and closed boundaries
"""


def _collapse_equals(obj, leaf_type):
    if isinstance(obj, leaf_type):
        return obj
    else:
        list = tuple([_collapse_equals(element, leaf_type) for element in obj])
        first = list[0]
        for element in list[1:]:
            if element != first:
                return list
        return first


def _friction_mask(masks_and_multipliers):
    for mask, multiplier in masks_and_multipliers:
        return mask


def geometry_mask(geometries, grid):
    if len(geometries) == 0:
        return math.zeros(grid.shape())
    location = grid.center_points()
    return math.max([geometry.value_at(location) for geometry in geometries], axis=0)
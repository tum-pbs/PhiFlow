from .obstacle import *
from .material import *
from phi import math

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
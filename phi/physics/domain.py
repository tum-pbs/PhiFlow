from phi.math import *
from .geom import *
from .material import *
from .world import *
from .objects import *


class Domain(object):

    def __init__(self, grid, boundaries=OPEN):
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
        self._grid = grid if isinstance(grid, Grid) else Grid(grid)
        assert isinstance(boundaries, (Material, list, tuple))
        assert isinstance(world, World)
        if isinstance(boundaries, (tuple, list)):
            assert len(boundaries) == self._grid.rank
        self._boundaries = _collapse_equals(boundaries, leaf_type=Material)

    @property
    def grid(self):
        return self._grid

    @property
    def rank(self):
        return self.grid.rank

    @property
    def boundaries(self):
        return self._boundaries

    def serialize_to_dict(self):
        return {
            "dimensions": [int(d) for d in self.grid.dimensions],
            # "boundaries": TODO
        }

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


class DomainState(object):

    def __init__(self, domain, worldstate, active=None, accessible=None):
        self._domain = domain
        self._worldstate = worldstate
        self._active = active if active is not None else self._domain._grid.ones()
        self._accessible = accessible if accessible is not None else self._domain._grid.ones()

    @property
    def domain(self):
        return self._domain

    @property
    def grid(self):
        return self.domain._grid

    @property
    def rank(self):
        return self.grid.rank

    def with_hard_boundary_conditions(self, velocity):
        masked = velocity * _frictionless_velocity_mask(self.accessible(extend=1))
        return masked  # TODO add surface velocity

    def active(self, extend=0):
        """
Scalar channel encoding active cells as ones and inactive (open/obstacle) as zero.
Active cells are those for which physical properties such as pressure or velocity are calculated.
        :param extend: Extend the grid in all directions beyond the grid size specified by the domain
        """
        if extend is None or extend == 0:
            return self._active
        else:
            return pad(self._active, [[0, 0]] + [[1, 1]] * self.rank + [[0, 0]], "constant")

    def accessible(self, extend=0):
        """
Scalar channel encoding cells that are accessible, i.e. not solid, as ones and obstacles as zero.
        :param extend: Extend the grid in all directions beyond the grid size specified by the domain
        """
        if extend is None or extend == 0:
            return self._accessible
        else:
            solid_paddings, open_paddings = self.domain._get_paddings(lambda material: material.solid, margin=extend)
            mask = self._accessible
            mask = pad(mask, open_paddings, "constant", 1)
            mask = pad(mask, solid_paddings, "constant", 0)
            return mask



            


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


def _frictionless_velocity_mask(accessible_mask):
    dims = range(spatial_rank(accessible_mask))
    bcs = []
    for d in dims:
        upper_slices = tuple([(slice(1, None) if i == d else slice(1, None)) for i in dims])
        lower_slices = tuple([(slice(0, -1) if i == d else slice(1, None)) for i in dims])
        bc_d = minimum(accessible_mask[(slice(None),) + upper_slices + (slice(None),)],
                       accessible_mask[(slice(None),) + lower_slices + (slice(None),)])
        bcs.append(bc_d)
    return nd.StaggeredGrid(concat(bcs, axis=-1))


def _friction_mask(masks_and_multipliers):
    for mask, multiplier in masks_and_multipliers:
        return mask


Open3D = Domain(Grid([0] * 3))
Open2D = Domain(Grid([0] * 2))


def inflow_mask(world, grid):
    inflows = world.state.get_by_tag('inflow')
    if len(inflows) == 0:
        return zeros(grid.shape())
    location = grid.center_points()
    return add([inflow.geometry.value_at(location) * inflow.rate for inflow in inflows])


def geometry_mask(world, grid, tag):
    geometries = geometries_with_tag(world.state, tag)
    if len(geometries) == 0:
        return zeros(grid.shape())
    location = grid.center_points()
    return max([geometry.value_at(location) for geometry in geometries], axis=0)
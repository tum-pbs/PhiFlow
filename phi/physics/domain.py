import warnings
from functools import partialmethod

from phi import math, struct
from phi.math import spatial_shape
from phi.geom import Box, GridCell
from phi.field import CenteredGrid, StaggeredGrid
from .material import OPEN, Material
from .physics import State


class Domain:

    def __init__(self, resolution: math.Shape or tuple or list, boundaries: Material or tuple or list = OPEN, box=None):
        """
        Simulation domain that specifies size and boundary conditions.

        If all boundary surfaces should have the same behaviour, pass a single Material instance.

        To specify the boundary constants_dict per dimension or surface, pass a tuple or list with as many elements as there are spatial dimensions (highest dimension first).
        Each element can either be a Material, specifying the faces perpendicular to that axis, or a pair
        of Material holding (lower_face_material, upper_face_material).

        Examples:

        Domain(grid, OPEN) - all surfaces are open

        DomainBoundary(grid, boundaries=[(SLIPPY, OPEN), SLIPPY]) - creates a 2D domain with an open top and otherwise solid boundaries

        :param resolution: 1D tensor specifying the grid dimensions
        :param boundaries: Material or list of Material/Pair of Material
        :param box: physical size of the domain, box-like
        """
        self.resolution = spatial_shape(resolution)
        self.boundaries = Material.as_material(boundaries)
        self.box = Box.to_box(box, resolution_hint=self.resolution)

    def __repr__(self):
        return '(%s, size=%s)' % (self.resolution, self.box.size)

    @property
    def rank(self):
        return self.resolution.rank

    @property
    def dx(self):
        return self.box.size / self.resolution

    @property
    def cells(self):
        return GridCell(self.resolution, self.box)

    def center_points(self):
        return self.cells.center

    def grid(self, value, type: type = CenteredGrid, extrapolation: math.Extrapolation = None):
        """
        Creates a grid matching the domain by sampling the given value.

        This method uses Material.extrapolation_mode of the domain's boundaries.

        :param value: Field or tensor or tensor function
        :param type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
        :param extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries.vector_extrapolation
        :return: Grid of specified type
        """
        extrapolation = extrapolation or self.boundaries.grid_extrapolation
        if type is CenteredGrid:
            return CenteredGrid.sample(value, self.resolution, self.box, extrapolation)
        elif type is StaggeredGrid:
            return StaggeredGrid.sample(value, self.resolution, self.box, extrapolation)
        else:
            raise ValueError('Unknown grid type: %s' % type)

    def vector_grid(self, value, type: type = CenteredGrid, extrapolation: math.Extrapolation = None):
        """
        Creates a vector grid matching the domain by sampling the given value.

        This method uses Material.vector_extrapolation_mode of the domain's boundaries.

        :param value: Field or tensor or tensor function
        :param type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
        :param extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries.grid_extrapolation
        :return: Grid of specified type
        """
        extrapolation = extrapolation or self.boundaries.vector_extrapolation
        if type is CenteredGrid:
            grid = CenteredGrid.sample(value, self.resolution, self.box, extrapolation)
            if grid.shape.channel.rank == 0:
                grid = grid.with_data(math.expand_channel(grid.data, self.rank, 0))
            else:
                assert grid.shape.channel.sizes[0] == self.rank
            return grid
        elif type is StaggeredGrid:
            return StaggeredGrid.sample(value, self.resolution, self.box, extrapolation)
        else:
            raise ValueError('Unknown grid type: %s' % type)

    staggered_grid = partialmethod(vector_grid, type=StaggeredGrid)

    vgrid = vector_grid
    sgrid = staggered_grid


@struct.definition()
class DomainState(State):

    @struct.constant()
    def domain(self, domain: Domain) -> Domain:
        assert isinstance(domain, Domain)
        return domain

    @property
    def resolution(self):
        return self.domain.resolution

    @property
    def rank(self):
        return self.domain.rank

    def centered_grid(self, name, value, components=1, dtype=None):
        warnings.warn("DomainState.centered_grid() is deprecated. The arguments 'name, components, dtype' were ignored.", DeprecationWarning)
        return self.domain.grid(value, CenteredGrid)

    def staggered_grid(self, name, value, dtype=None):
        warnings.warn("DomainState.staggered_grid() is deprecated. The arguments 'name, components, dtype' were ignored.", DeprecationWarning)
        return self.domain.vec_grid(value, StaggeredGrid)

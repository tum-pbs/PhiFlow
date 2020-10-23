from __future__ import annotations

from functools import partialmethod

from phi import math, struct
from phi.field import CenteredGrid, StaggeredGrid, GeometryMask, Grid
from phi.geom import Box, GridCell
from phi.geom import Geometry
from phi.math import extrapolation
from phi.math import spatial_shape
from ._effect import FieldEffect
from ._physics import Physics
from ._physics import State


class Material:
    """
    Defines the extrapolation modes / boundary conditions for a surface.
    The surface can be an obstacle or the domain boundary.
    """
    def __init__(self, name, grid_extrapolation, vector_extrapolation, active_extrapolation, accessible_extrapolation):
        self.name = name
        self.grid_extrapolation = grid_extrapolation
        self.vector_extrapolation = vector_extrapolation
        self.active_extrapolation = active_extrapolation
        self.accessible_extrapolation = accessible_extrapolation

    def __repr__(self):
        return self.name

    @staticmethod
    def as_material(obj: Material or tuple or list or dict) -> Material:
        if isinstance(obj, Material):
            return obj
        if isinstance(obj, (tuple, list)):
            axes = [math.GLOBAL_AXIS_ORDER.axis_name(i, len(obj)) for i in range(len(obj))]
            obj = {ax: mat for ax, mat in zip(axes, obj)}
        if isinstance(obj, dict):
            grid_extrapolation = {ax: mat.grid_extrapolation for ax, mat in obj.items()}
            vector_extrapolation = {ax: mat.vector_extrapolation for ax, mat in obj.items()}
            active_extrapolation = {ax: mat.active_extrapolation for ax, mat in obj.items()}
            accessible_extrapolation = {ax: mat.accessible_extrapolation for ax, mat in obj.items()}
            return Material('mixed', grid_extrapolation, vector_extrapolation, active_extrapolation, accessible_extrapolation)
        raise NotImplementedError()


OPEN = Material('open', extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ONE)
CLOSED = NO_STICK = SLIPPERY = Material('slippery', extrapolation.BOUNDARY, extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ZERO)
NO_SLIP = STICKY = Material('sticky', extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO)
PERIODIC = Material('periodic', extrapolation.PERIODIC, extrapolation.PERIODIC, extrapolation.ONE, extrapolation.ONE)


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

    def vector_grid(self, value, type: type = CenteredGrid, extrapolation: math.Extrapolation = None) -> Grid:
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
                grid = grid.with_values(math.expand_channel(grid.values, self.rank, 0))
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
class Obstacle(State):
    """
    An obstacle defines boundary conditions inside a geometry.
    It can also have a linear and angular velocity.
    """

    def __init__(self, geometry, material=CLOSED, velocity=0, tags=('obstacle',), **kwargs):
        State.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def geometry(self, geometry):
        assert isinstance(geometry, Geometry)
        return geometry

    @struct.constant(default=CLOSED)
    def material(self, material):
        assert isinstance(material, Material)
        return material

    @struct.constant(default=0)
    def velocity(self, velocity):
        return velocity

    @struct.constant(default=0)
    def angular_velocity(self, av):
        return av

    @struct.derived()
    def is_stationary(self):
        return self.velocity is 0 and self.angular_velocity is 0


class GeometryMovement(Physics):

    def __init__(self, geometry_function):
        Physics.__init__(self)
        self.geometry_at = geometry_function

    def step(self, obj, dt=1.0, **dependent_states):
        next_geometry = self.geometry_at(obj.age + dt)
        h = 1e-2 * dt if dt > 0 else 1e-2
        perturbed_geometry = self.geometry_at(obj.age + dt + h)
        velocity = (perturbed_geometry.center - next_geometry.center) / h
        if isinstance(obj, Obstacle):
            return obj.copied_with(geometry=next_geometry, velocity=velocity, age=obj.age + dt)
        if isinstance(obj, FieldEffect):
            with struct.ALL_ITEMS:
                next_field = struct.map(lambda x: x.copied_with(geometries=next_geometry) if isinstance(x, GeometryMask) else x, obj.field, leaf_condition=lambda x: isinstance(x, GeometryMask))
            return obj.copied_with(field=next_field, age=obj.age + dt)

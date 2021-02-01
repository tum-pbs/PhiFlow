from __future__ import annotations

from functools import partialmethod

from phi import math, struct
from phi.field import CenteredGrid, StaggeredGrid, GeometryMask, Grid
from phi.geom import Box, GridCell
from phi.geom import Geometry
from phi.math import extrapolation, Tensor
from phi.math import spatial_shape
from ._effect import FieldEffect
from ._physics import Physics
from ._physics import State
from ..math.extrapolation import combine_sides


class Material:
    """
    Defines the extrapolation modes / boundary conditions for a surface.
    The surface can be an Obstacle or a Domain boundary.
    
    Use Material.as_material() to mix different materials for different sides.
    """

    def __init__(self, name: str, grid_extrapolation, vector_extrapolation, near_vector_extrapolation, active_extrapolation, accessible_extrapolation):
        """
        Create a Material for a Domain or Obstacle.

        Args:
          name: material name
          grid_extrapolation: extrapolation mode of grids created via Domain.grid()
          vector_extrapolation: extrapolation mode of grids created via Domain.vector_grid() or Domain.staggered_grid()
          near_vector_extrapolation: Used in pressure solve.
          active_extrapolation: Whether cells outside the domain bounds also belong to the domain. Used in pressure solve.
          accessible_extrapolation: Whether quantities can move in and out of the domain. Used in pressure solve.
        """
        self.name = name
        """ Material name """
        self.grid_extrapolation = grid_extrapolation
        """ Extrapolation mode of grids created via Domain.grid() """
        self.vector_extrapolation = vector_extrapolation
        """ Extrapolation mode of grids created via Domain.vector_grid() or Domain.staggered_grid() """
        self.near_vector_extrapolation = near_vector_extrapolation
        """ Used in pressure solve. """
        self.active_extrapolation = active_extrapolation
        """ Whether cells outside the domain bounds also belong to the domain. Used in pressure solve. """
        self.accessible_extrapolation = accessible_extrapolation
        """ Whether quantities can move in and out of the domain. Used in pressure solve. """

    def __repr__(self):
        return self.name

    @staticmethod
    def as_material(obj: Material or tuple or list or dict) -> Material:
        """
        Construct a mixed material from from a sequence of materials.

        Args:
          obj: sequence of materials
          obj: Material or tuple or list or dict: 

        Returns:
          : Single mixed Material

        """
        if isinstance(obj, Material):
            return obj
        if isinstance(obj, (tuple, list)):
            dims = [math.GLOBAL_AXIS_ORDER.axis_name(i, len(obj)) for i in range(len(obj))]
            obj = {dim: mat for dim, mat in zip(dims, obj)}
        if isinstance(obj, dict):
            grid_extrapolation = _mix(obj, 'grid_extrapolation')
            near_vector_extrapolation = _mix(obj, 'near_vector_extrapolation')
            vector_extrapolation = _mix(obj, 'vector_extrapolation')
            active_extrapolation = _mix(obj, 'active_extrapolation')
            accessible_extrapolation = _mix(obj, 'accessible_extrapolation')
            return Material('mixed', grid_extrapolation, near_vector_extrapolation, vector_extrapolation, active_extrapolation, accessible_extrapolation)
        raise NotImplementedError()


def _mix(material_dict, ext_property: str):
    extrapolations = {}
    for dim, material in material_dict.items():
        if isinstance(material, Material):
            extrapolations[dim] = getattr(material, ext_property)
        else:
            extrapolations[dim] = (getattr(material[0], ext_property), getattr(material[1], ext_property))
    return combine_sides(extrapolations)


OPEN = Material('open', extrapolation.ZERO, extrapolation.ZERO, extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ONE)
CLOSED = NO_STICK = SLIPPERY = Material('slippery', extrapolation.BOUNDARY, extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO)
NO_SLIP = STICKY = Material('sticky', extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO)
PERIODIC = Material('periodic', extrapolation.PERIODIC, extrapolation.PERIODIC, extrapolation.PERIODIC, extrapolation.ONE, extrapolation.ONE)


class Domain:

    def __init__(self, resolution: math.Shape = math.EMPTY_SHAPE, boundaries: Material or tuple or list = OPEN, bounds: Box = None, **resolution_):
        """
        The Domain specifies the grid resolution, physical size and boundary conditions of a simulation.

        It provides convenience methods for creating Grids fitting the domain, e.g. `grid()`, `vector_grid()` and `staggered_grid()`.

        Also see the `phi.physics` module documentation at https://tum-pbs.github.io/PhiFlow/Physics.html

        Args:
          resolution: grid dimensions as Shape or sequence of integers. Alternatively, dimensions can be specified directly as kwargs.
          boundaries: specifies the extrapolation modes of grids created from this Domain as a Material instance.
            Default materials include OPEN, CLOSED, PERIODIC.
            To specify boundary conditions per face of the domain, pass a sequence of Materials or Material pairs (lower, upper)., e.g. [CLOSED, (CLOSED, OPEN)].
          bounds: physical size of the domain. If not provided, the size is equal to the resolution (unit cubes).
        """
        self.resolution = spatial_shape(resolution) & spatial_shape(resolution_)
        """ Grid dimensions as `Shape` object """
        self.boundaries = Material.as_material(boundaries)
        """ Outer boundary condition as `Material` object """
        self.bounds = Box(0, math.tensor(self.resolution, names='vector')) if bounds is None else bounds
        """ Physical dimensions of the domain as `Box` object """

    def __repr__(self):
        return '(%s, size=%s)' % (self.resolution, self.bounds.size)

    @property
    def rank(self):
        """Number of spatial dimensions of the simulation; spatial rank. 1 = 1D, 2 = 2D, 3 = 3D, etc."""
        return self.resolution.rank

    @property
    def dx(self):
        """Size of a single grid cell (physical size divided by resolution) as `Tensor`"""
        return self.bounds.size / self.resolution

    @property
    def cells(self):
        """
        Returns the geometry of all cells as a `Box` object.
        The box will have spatial dimensions matching the resolution of the Domain, i.e. `domain.cells.shape == domain.resolution`.

        Args:

        Returns:

        """
        return GridCell(self.resolution, self.bounds)

    def center_points(self):
        """
        Returns a Tensor enumerating the physical center locations of all cells within the Domain.
        This is equivalent to calling `domain.cells.center`.
        
        The shape of the returned Tensor extends the domain resolution by one vector dimension.

        Args:

        Returns:

        """
        return self.cells.center

    def grid(self, value: Tensor or float or int or complex or callable or Geometry = 0,
             type: type = CenteredGrid,
             extrapolation: math.Extrapolation = None) -> StaggeredGrid or CenteredGrid:
        """
        Creates a grid matching the resolution and bounds of the domain.
        The grid is created from the given `value` which must be one of the following:
        
        * Number (int, float, complex or zero-dimensional tensor): all grid values will be equal to `value`. This has a near-zero memory footprint.
        * Field: the given value is resampled to the grid cells of this Domain.
        * Tensor with spatial dimensions matching the domain resolution: grid values will equal `value`.
        * Geometry: grid values are determined from the volume overlap between grid cells and geometry. Non-overlapping = 0, fully enclosed grid cell = 1.
        * function(location: Tensor) returning one of the above.

        Args:
          value: constant, Field, Tensor or function specifying the grid values
          type: type of Grid to create, must be either CenteredGrid or StaggeredGrid
          extrapolation: optional) grid extrapolation, defaults to Domain.boundaries.grid_extrapolation
          value: Tensor or float or int or complex or callable or Geometry:  (Default value = 0)
          type: type:  (Default value = CenteredGrid)
          extrapolation: math.Extrapolation:  (Default value = None)

        Returns:
          Grid of specified type

        """
        extrapolation = extrapolation or self.boundaries.grid_extrapolation
        if type is CenteredGrid:
            return CenteredGrid.sample(value, self.resolution, self.bounds, extrapolation)
        elif type is StaggeredGrid:
            return StaggeredGrid.sample(value, self.resolution, self.bounds, extrapolation)
        else:
            raise ValueError('Unknown grid type: %s' % type)

    def vector_grid(self, value: Tensor or float or int or complex or callable or Geometry = 0,
                    type: type = CenteredGrid,
                    extrapolation: math.Extrapolation = None) -> StaggeredGrid or CenteredGrid:
        """
        Creates a vector grid matching the resolution and bounds of the domain.
        The grid is created from the given `value` which must be one of the following:
        
        * Number (int, float, complex or zero-dimensional tensor): all grid values will be equal to `value`. This has a near-zero memory footprint.
        * Field: the given value is resampled to the grid cells of this Domain.
        * Tensor with spatial dimensions matcing the domain resolution: grid values will equal `value`.
        * Geometry: grid values are determined from the volume overlap between grid cells and geometry. Non-overlapping = 0, fully enclosed grid cell = 1.
        * function(location: Tensor) returning one of the above.
        
        The returned grid will have a vector dimension with size equal to the rank of the domain.
        
        Aliases: `vector_grid`, `vgrid`

        Args:
          value: constant, Field, Tensor or function specifying the grid values
          type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
          extrapolation: optional) grid extrapolation, defaults to Domain.boundaries.grid_extrapolation
          value: Tensor or float or int or complex or callable or Geometry:  (Default value = 0)
          type: type:  (Default value = CenteredGrid)
          extrapolation: math.Extrapolation:  (Default value = None)

        Returns:
          Grid of specified type

        """
        extrapolation = extrapolation or self.boundaries.vector_extrapolation
        if type is CenteredGrid:
            grid = CenteredGrid.sample(value, self.resolution, self.bounds, extrapolation)
            if grid.shape.channel.rank == 0:
                grid = grid.with_(values=math.expand_channel(grid.values, 'vector', dim_size=self.rank))
            else:
                assert grid.shape.channel.sizes[0] == self.rank
            return grid
        elif type is StaggeredGrid:
            return StaggeredGrid.sample(value, self.resolution, self.bounds, extrapolation)
        else:
            raise ValueError('Unknown grid type: %s' % type)

    vgrid = vector_grid

    def staggered_grid(self, value: Tensor or float or int or complex or callable or Geometry = 0,
                       extrapolation: math.Extrapolation = None) -> StaggeredGrid:
        """
        Creates a staggered grid matching the resolution and bounds of the domain.
        This is equal to calling `vector_grid()` with `type=StaggeredGrid`.
        
        The grid is created from the given `value` which must be one of the following:
        
        * Number (int, float, complex or zero-dimensional tensor): all grid values will be equal to `value`. This has a near-zero memory footprint.
        * Field: the given value is resampled to the grid cells of this Domain.
        * Tensor with spatial dimensions matcing the domain resolution: grid values will equal `value`.
        * Geometry: grid values are determined from the volume overlap between grid cells and geometry. Non-overlapping = 0, fully enclosed grid cell = 1.
        * function(location: Tensor) returning one of the above.
        
        The returned grid will have a vector dimension with size equal to the rank of the domain.
        
        Aliases: `staggered_grid`, `sgrid`

        Args:
          value: constant, Field, Tensor or function specifying the grid values
          type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
          extrapolation: optional) grid extrapolation, defaults to Domain.boundaries.grid_extrapolation
          value: Tensor or float or int or complex or callable or Geometry:  (Default value = 0)
          extrapolation: math.Extrapolation:  (Default value = None)

        Returns:
          Grid of specified type

        """
        return self.vector_grid(value, type=StaggeredGrid, extrapolation=extrapolation)

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
        """ Physical shape and size of the obstacle. """
        assert isinstance(geometry, Geometry)
        return geometry

    @struct.constant(default=CLOSED)
    def material(self, material):
        """ Boundary conditions to apply inside and on the surface of the obstacle. """
        assert isinstance(material, Material)
        return material

    @struct.constant(default=0)
    def velocity(self, velocity):
        """ Linear velocity vector of the obstacle. """
        return velocity

    @struct.constant(default=0)
    def angular_velocity(self, av):
        """ Rotation speed of the obstacle. Scalar value in 2D, vector in 3D. """
        return av

    @struct.derived()
    def is_stationary(self):
        """ Test whether the obstacle is completely still. """
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

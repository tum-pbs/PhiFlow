import warnings
from numbers import Number

from phi import math, struct, field
from phi.field import CenteredGrid, StaggeredGrid, GeometryMask, PointCloud, Field, HardGeometryMask, Grid
from phi.geom import Box, GridCell, Sphere, union, assert_same_rank
from phi.geom import Geometry
from phi.math import extrapolation, Tensor
from phi.math import spatial_shape
from ._effect import FieldEffect
from ._physics import Physics
from ._physics import State
from ..math.extrapolation import combine_sides


def _create_boundary_conditions(obj: dict or tuple or list, spatial_dims: tuple) -> dict:
    """
    Construct mixed boundary conditions from from a sequence of boundary conditions.

    Args:
      obj: single boundary condition or sequence of boundary conditions

    Returns:
      Mixed boundary conditions as `dict`.

    """
    if isinstance(obj, dict) and all(dim in obj for dim in spatial_dims):
        spatial_dims = obj.keys()
        obj = tuple(obj.values())
    elif isinstance(obj, dict):
        return obj
    if isinstance(obj, (tuple, list)):
        keys = obj[0].keys() if isinstance(obj[0], dict) else obj[0][0].keys()
        result = {}
        for key in keys:
            dim_to_extrap = {dim: (extrap[0][key], extrap[1][key]) if isinstance(extrap, (tuple, list)) else extrap[key]
                             for dim, extrap in zip(spatial_dims, obj)}
            result[key] = combine_sides(dim_to_extrap)
        return result
    else:
        raise ValueError(obj)


OPEN = {
    'scalar_extrapolation': extrapolation.ZERO,
    'vector_extrapolation': extrapolation.ZERO,
    'near_vector_extrapolation': extrapolation.BOUNDARY,
    'active_extrapolation': extrapolation.ZERO,
    'accessible_extrapolation': extrapolation.ONE,
}

SLIPPERY = {
    'scalar_extrapolation': extrapolation.BOUNDARY,
    'vector_extrapolation': extrapolation.BOUNDARY,
    'near_vector_extrapolation': extrapolation.ZERO,
    'active_extrapolation': extrapolation.ZERO,
    'accessible_extrapolation': extrapolation.ZERO,
}

STICKY = {
    'scalar_extrapolation': extrapolation.BOUNDARY,
    'vector_extrapolation': extrapolation.ZERO,
    'near_vector_extrapolation': extrapolation.ZERO,
    'active_extrapolation': extrapolation.ZERO,
    'accessible_extrapolation': extrapolation.ZERO,
}

PERIODIC = {
    'scalar_extrapolation': extrapolation.PERIODIC,
    'vector_extrapolation': extrapolation.PERIODIC,
    'near_vector_extrapolation': extrapolation.PERIODIC,
    'active_extrapolation': extrapolation.ONE,
    'accessible_extrapolation': extrapolation.ONE,
}


class Domain:

    def __init__(self, resolution: math.Shape = math.EMPTY_SHAPE, boundaries: dict or tuple or list = OPEN, bounds: Box = None, **resolution_):
        """
        The Domain specifies the grid resolution, physical size and boundary conditions of a simulation.

        It provides convenience methods for creating Grids fitting the domain, e.g. `grid()`, `vector_grid()` and `staggered_grid()`.

        Also see the `phi.physics` module documentation at https://tum-pbs.github.io/PhiFlow/Physics.html

        Args:
          resolution: grid dimensions as Shape or sequence of integers. Alternatively, dimensions can be specified directly as kwargs.
          boundaries: specifies the extrapolation modes of grids created from this Domain.
            Default materials include OPEN, CLOSED, PERIODIC.
            To specify boundary conditions per face of the domain, pass a sequence of boundaries or boundary pairs (lower, upper)., e.g. [CLOSED, (CLOSED, OPEN)].
            See https://tum-pbs.github.io/PhiFlow/Physics.html#boundary-conditions .
          bounds: physical size of the domain. If not provided, the size is equal to the resolution (unit cubes).
        """
        self.resolution: math.Shape = spatial_shape(resolution) & spatial_shape(resolution_)
        """ Grid dimensions as `Shape` object containing spatial dimensions only. """
        self.boundaries: dict = _create_boundary_conditions(boundaries, self.resolution.names)
        """ Outer boundary conditions. """
        self.bounds: Box = Box(0, math.wrap(self.resolution, names='vector')) if bounds is None else bounds
        """ Physical dimensions of the domain. """

    def __repr__(self):
        return '(%s, size=%s)' % (self.resolution, self.bounds.size)

    @property
    def shape(self) -> math.Shape:
        """ Alias for `Domain.resolution` """
        return self.resolution

    @property
    def rank(self) -> int:
        """Number of spatial dimensions of the simulation; spatial rank. 1 = 1D, 2 = 2D, 3 = 3D, etc."""
        return self.resolution.rank

    @property
    def dx(self) -> math.Tensor:
        """Size of a single grid cell (physical size divided by resolution) as `Tensor`"""
        return self.bounds.size / self.resolution

    @property
    def cells(self) -> GridCell:
        """
        Returns the geometry of all cells as a `Box` object.
        The box will have spatial dimensions matching the resolution of the Domain, i.e. `domain.cells.shape == domain.resolution`.
        """
        return GridCell(self.resolution, self.bounds)

    def center_points(self) -> math.Tensor:
        """
        Returns a Tensor enumerating the physical center locations of all cells within the Domain.
        This is equivalent to calling `domain.cells.center`.
        
        The shape of the returned Tensor extends the domain resolution by one vector dimension.
        """
        return self.cells.center

    def grid(self,
             value: Field or Tensor or Number or Geometry or callable = 0.,
             type: type = CenteredGrid,
             extrapolation: math.Extrapolation = None) -> CenteredGrid or StaggeredGrid:
        """
        *Deprecated* due to inconsistent extrapolation selection. Use `scalar_grid()` or `vector_grid()` instead.

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
          extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['scalar_extrapolation']

        Returns:
          Grid of specified type
        """
        warnings.warn("Domain.grid is deprecated. Use scalar_grid or vector_grid instead.", DeprecationWarning)
        extrapolation = extrapolation or self.boundaries['scalar_extrapolation']
        if type is CenteredGrid:
            return CenteredGrid.sample(value, self.resolution, self.bounds, extrapolation)
        elif type is StaggeredGrid:
            return StaggeredGrid.sample(value, self.resolution, self.bounds, extrapolation)
        else:
            raise ValueError('Unknown grid type: %s' % type)

    def scalar_grid(self,
                    value: Field or Tensor or Number or Geometry or callable = 0.,
                    extrapolation: math.Extrapolation = None) -> CenteredGrid:
        """
        Creates a scalar grid matching the resolution and bounds of the domain.
        The grid is created from the given `value` which must be one of the following:

        * Number (int, float, complex or zero-dimensional tensor): all grid values will be equal to `value`. This has a near-zero memory footprint.
        * Scalar `Field`: the given value is resampled to the grid cells of this Domain.
        * Tensor with spatial dimensions matching the domain resolution: grid values will equal `value`.
        * Geometry: grid values are determined from the volume overlap between grid cells and geometry. Non-overlapping = 0, fully enclosed grid cell = 1.
        * function(location: Tensor) returning one of the above.
        * Native tensor: the number and order of axes are matched with the resolution of the domain.

        Args:
          value: constant, Field, Tensor or function specifying the grid values
          extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['scalar_extrapolation']

        Returns:
          `CenteredGrid` with no channel dimensions
        """
        extrapolation = extrapolation or self.boundaries['scalar_extrapolation']
        if isinstance(value, Field):
            assert_same_rank(value.spatial_rank, self.rank, f"Cannot resample {value.spatial_rank}D field to {self.rank}D domain.")
        elif isinstance(value, Tensor):
            assert value.shape.channel.rank == 0
        elif isinstance(value, (Number, Geometry)):
            pass
        elif callable(value):
            pass
        else:
            try:
                value = math.wrap(value, names=self.resolution.names)
            except AssertionError:
                pass
            value = math.wrap(value)
        result = CenteredGrid.sample(value, self.resolution, self.bounds, extrapolation)
        assert result.shape.channel_rank == 0
        return result

    def vector_grid(self,
                    value: Field or Tensor or Number or Geometry or callable = 0.,
                    type: type = CenteredGrid,
                    extrapolation: math.Extrapolation = None) -> CenteredGrid or StaggeredGrid:
        """
        Creates a vector grid matching the resolution and bounds of the domain.
        The grid is created from the given `value` which must be one of the following:
        
        * Number (int, float, complex or zero-dimensional tensor): all grid values will be equal to `value`. This has a near-zero memory footprint.
        * Field: the given value is resampled to the grid cells of this Domain.
        * Tensor with spatial dimensions matcing the domain resolution: grid values will equal `value`.
        * Geometry: grid values are determined from the volume overlap between grid cells and geometry. Non-overlapping = 0, fully enclosed grid cell = 1.
        * function(location: Tensor) returning one of the above.
        
        The returned grid will have a vector dimension with size equal to the rank of the domain.

        Args:
          value: constant, Field, Tensor or function specifying the grid values
          type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
          extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['vector_extrapolation']

        Returns:
          Grid of specified type

        """
        extrapolation = extrapolation or self.boundaries['vector_extrapolation']
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

    def staggered_grid(self,
                       value: Field or Tensor or Number or Geometry or callable = 0.,
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

        Args:
          value: constant, Field, Tensor or function specifying the grid values
          type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
          extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['vector_extrapolation']

        Returns:
          Grid of specified type
        """
        return self.vector_grid(value, type=StaggeredGrid, extrapolation=extrapolation)

    def accessible_mask(self, not_accessible: tuple or list, type: type = CenteredGrid) -> CenteredGrid or StaggeredGrid:
        """
        Unifies domain and Obstacle or Geometry objects into a binary StaggeredGrid mask which can be used
        to enforce boundary conditions.

        Args:
            not_accessible: blocked region(s) of space specified by geometries

        Returns:
            Binary mask indicating valid fields w.r.t. the boundary conditions.
            The result is of type `type` and uses the extrapolation `Domain.boundaries['accessible_extrapolation']`.
        """
        accessible_mask = self.scalar_grid(HardGeometryMask(~union(not_accessible)), extrapolation=self.boundaries['accessible_extrapolation'])
        if type is CenteredGrid:
            return accessible_mask
        elif type is StaggeredGrid:
            return field.stagger(accessible_mask, math.minimum, self.boundaries['accessible_extrapolation'])
        else:
            raise ValueError('Unknown grid type: %s' % type)

    def points(self,
               points: Tensor or Number or tuple or list,
               radius: Tensor or float or int or None = None,
               extrapolation: math.Extrapolation = None,
               color: str or Tensor or tuple or list or None = None) -> PointCloud:
        """
        Create a `phi.field.PointCloud` from the given `points`.
        The created field has no channel dimensions and all points carry the value `1`.

        Args:
            points: point locations in physical units
            radius: (optional) size of the particles
            extrapolation: (optional) extrapolation to use, defaults to extrapolation.ZERO
            color: (optional) color used when plotting the points

        Returns:
            `phi.field.PointCloud` object
        """
        extrapolation = extrapolation or math.extrapolation.ZERO
        if radius is None:
            radius = math.mean(self.bounds.size) * 0.005
        # --- Parse points: tuple / list ---
        if isinstance(points, (tuple, list)):
            if len(points) == 0:  # no points
                points = math.zeros(points=0, vector=1)
            elif isinstance(points[0], Number):  # single point
                points = math.wrap([points], 'points, vector')
            else:
                points = math.wrap(points, 'points, vector')
        elements = Sphere(points, radius)
        return PointCloud(elements, math.ones(), extrapolation, add_overlapping=False, bounds=self.bounds, color=color)

    def distribute_points(self,
                          geometries: tuple or list,
                          points_per_cell: int = 8,
                          color: str = None,
                          center: bool = False) -> PointCloud:
        """
        Transforms `Geometry` objects into a PointCloud.

        Args:
            geometries: Geometry objects marking the cells which should contain points
            points_per_cell: Number of points for each cell of `geometries`
            color (Optional): Color of PointCloud
            center: Set all points to the center of the grid cells.

        Returns:
             PointCloud representation of `geometries`.
        """
        geometries = HardGeometryMask(union(geometries)) >> self.grid()
        initial_points = _distribute_points(geometries.values, points_per_cell, center=center)
        return self.points(initial_points, color=color)


@struct.definition()
class Obstacle(State):
    """
    An obstacle defines boundary conditions inside a geometry.
    It can also have a linear and angular velocity.
    """

    def __init__(self, geometry, material=SLIPPERY, velocity=0, tags=('obstacle',), **kwargs):
        State.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def geometry(self, geometry):
        """ Physical shape and size of the obstacle. """
        assert isinstance(geometry, Geometry)
        return geometry

    @struct.constant(default=SLIPPERY)
    def material(self, material):
        """ Boundary conditions to apply inside and on the surface of the obstacle. """
        assert isinstance(material, dict)
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


def _distribute_points(mask: math.Tensor, points_per_cell: int = 1, center: bool = False) -> math.Tensor:
    """
    Generates points (either uniformly distributed or at the cell centers) according to the given tensor mask.

    Args:
        mask: Tensor with nonzero values at the indices where particles should get generated.
        points_per_cell: Number of particles to generate at each marked index
        center: Set points to cell centers. If False, points will be distributed using a uniform
            distribution within each cell.

    Returns:
        A tensor containing the positions of the generated points.
    """
    indices = math.to_float(math.nonzero(mask, list_dim='points'))
    temp = []
    for _ in range(points_per_cell):
        if center:
            temp.append(indices + 0.5)
        else:
            temp.append(indices + (math.random_uniform(indices.shape)))
    return math.concat(temp, dim='points')

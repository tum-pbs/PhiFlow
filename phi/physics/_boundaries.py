import warnings
from numbers import Number

from phi import math, field
from phi.field import CenteredGrid, StaggeredGrid, PointCloud, Field, HardGeometryMask
from phi.geom import Box, GridCell, Sphere, union, assert_same_rank
from phi.geom import Geometry
from phi.math import Tensor, channel, instance
from phi.math.extrapolation import ZERO, ONE, PERIODIC, BOUNDARY
from phi.math import spatial
from ..math.extrapolation import combine_sides
from .fluid import Obstacle  # for compatibility


warnings.warn("""Domain (phi.physics._boundaries) is deprecated and will be removed in a future release.
Please create grids directly, replacing the domain with a dict, e.g.
    domain = dict(x=64, y=128, bounds=Box(x=1, y=1))
    grid = CenteredGrid(0, **domain)""", FutureWarning, stacklevel=2)


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
            result[key] = combine_sides(**dim_to_extrap)
        return result
    else:
        raise ValueError(obj)


OPEN = {
    'scalar': ZERO,
    'vector': BOUNDARY,
    'active': ZERO,
    'accessible': ONE,
}

STICKY = {
    'scalar': BOUNDARY,
    'vector': ZERO,
    'active': ZERO,
    'accessible': ZERO,
}

PERIODIC = {
    'scalar': PERIODIC,
    'vector': PERIODIC,
    'active': PERIODIC,
    'accessible': PERIODIC,
}


class Domain:

    def __init__(self, resolution: math.Shape or tuple or list = math.EMPTY_SHAPE, boundaries: dict or tuple or list = OPEN, bounds: Box = None, **resolution_):
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
        warnings.warn("Domain is deprecated and will be removed in a future release. Use a dict instead, e.g. CenteredGrid(values, extrapolation, **domain_dict)", DeprecationWarning, stacklevel=2)
        warnings.warn("Domain is deprecated and will be removed in a future release. Use a dict instead, e.g. CenteredGrid(values, extrapolation, **domain_dict)", FutureWarning, stacklevel=2)
        self.resolution: math.Shape = spatial(resolution) & spatial(**resolution_)
        assert self.resolution.rank > 0, "Cannot create Domain because no dimensions were specified."
        """ Grid dimensions as `Shape` object containing spatial dimensions only. """
        self.boundaries: dict = _create_boundary_conditions(boundaries, self.resolution.names)
        """ Outer boundary conditions. """
        self.bounds: Box = Box(math.const_vec(0, self.resolution), math.wrap(self.resolution, channel('vector'))) if bounds is None else bounds
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
             extrapolation: math.Extrapolation = 'scalar') -> CenteredGrid or StaggeredGrid:
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
          extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['scalar']

        Returns:
            Grid of specified type
        """
        extrapolation = extrapolation if isinstance(extrapolation, math.Extrapolation) else self.boundaries[extrapolation]
        return type(value, resolution=self.resolution, bounds=self.bounds, extrapolation=extrapolation)

    def scalar_grid(self,
                    value: Field or Tensor or Number or Geometry or callable = 0.,
                    extrapolation: str or math.Extrapolation = 'scalar') -> CenteredGrid:
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
          extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['scalar']

        Returns:
          `CenteredGrid` with no channel dimensions
        """
        extrapolation = extrapolation if isinstance(extrapolation, math.Extrapolation) else self.boundaries[extrapolation]
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
                value = math.wrap(value, self.resolution)
            except math.IncompatibleShapes:
                pass
            value = math.wrap(value)
        result = CenteredGrid(value, resolution=self.resolution, bounds=self.bounds, extrapolation=extrapolation)
        assert result.shape.channel_rank == 0
        return result

    def vector_grid(self,
                    value: Field or Tensor or Number or Geometry or callable = 0.,
                    type: type = CenteredGrid,
                    extrapolation: math.Extrapolation or str = 'vector') -> CenteredGrid or StaggeredGrid:
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
          extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['vector']

        Returns:
          Grid of specified type
        """
        extrapolation = extrapolation if isinstance(extrapolation, math.Extrapolation) else self.boundaries[extrapolation]
        result = type(value, resolution=self.resolution, bounds=self.bounds, extrapolation=extrapolation)
        if result.shape.channel_rank == 0:
            result = result.with_values(math.expand(result.values, channel(vector=self.rank)))
        else:
            assert result.shape.get_size('vector') == self.rank
        return result

    def staggered_grid(self,
                       value: Field or Tensor or Number or Geometry or callable = 0.,
                       extrapolation: math.Extrapolation or str = 'vector') -> StaggeredGrid:
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
          extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['vector']

        Returns:
          Grid of specified type
        """
        return self.vector_grid(value, type=StaggeredGrid, extrapolation=extrapolation)

    def vector_potential(self,
                         value: Field or Tensor or Number or Geometry or callable = 0.,
                         extrapolation: str or math.Extrapolation = 'scalar',
                         curl_type=CenteredGrid):
        if self.rank == 2 and curl_type == StaggeredGrid:
            pot_bounds = Box(self.bounds.lower - 0.5 * self.dx, self.bounds.upper + 0.5 * self.dx)
            alt_domain = Domain(self.boundaries, self.resolution + 1, bounds=pot_bounds)
            return alt_domain.scalar_grid(value, extrapolation=extrapolation)
        raise NotImplementedError()

    def accessible_mask(self, not_accessible: tuple or list, type: type = CenteredGrid, extrapolation='accessible') -> CenteredGrid or StaggeredGrid:
        """
        Unifies domain and Obstacle or Geometry objects into a binary StaggeredGrid mask which can be used
        to enforce boundary conditions.

        Args:
            not_accessible: blocked region(s) of space specified by geometries
            type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
            extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['accessible']

        Returns:
            Binary mask indicating valid fields w.r.t. the boundary conditions.
        """
        extrapolation = extrapolation if isinstance(extrapolation, math.Extrapolation) else self.boundaries[extrapolation]
        accessible_mask = self.scalar_grid(HardGeometryMask(~union(not_accessible)), extrapolation=extrapolation)
        if type is CenteredGrid:
            return accessible_mask
        elif type is StaggeredGrid:
            return field.stagger(accessible_mask, math.minimum, extrapolation)
        else:
            raise ValueError('Unknown grid type: %s' % type)

    def points(self,
               points: Tensor or Number or tuple or list,
               values: Tensor or Number = None,
               radius: Tensor or float or int or None = None,
               extrapolation: math.Extrapolation = math.extrapolation.ZERO,
               color: str or Tensor or tuple or list or None = None) -> PointCloud:
        """
        Create a `phi.field.PointCloud` from the given `points`.
        The created field has no channel dimensions and all points carry the value `1`.

        Args:
            points: point locations in physical units
            values: (optional) values of the particles, defaults to 1.
            radius: (optional) size of the particles
            extrapolation: (optional) extrapolation to use, defaults to extrapolation.ZERO
            color: (optional) color used when plotting the points

        Returns:
            `phi.field.PointCloud` object
        """
        extrapolation = extrapolation if isinstance(extrapolation, math.Extrapolation) else self.boundaries[extrapolation]
        if radius is None:
            radius = math.mean(self.bounds.size) * 0.005
        # --- Parse points: tuple / list ---
        if isinstance(points, (tuple, list)):
            if len(points) == 0:  # no points
                points = math.zeros(instance(points=0), channel(vector=1))
            elif isinstance(points[0], Number):  # single point
                points = math.tensor([points], instance('points'), channel('vector'))
            else:
                points = math.tensor(points, instance('points'), channel('vector'))
        elements = Sphere(points, radius)
        if values is None:
            values = math.tensor(1.)
        return PointCloud(elements, values, extrapolation, add_overlapping=False, bounds=self.bounds, color=color)

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
        geometries = HardGeometryMask(union(geometries)) @ self.grid()
        initial_points = _distribute_points(geometries.values, points_per_cell, center=center)
        return self.points(initial_points, color=color)


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
    indices = math.to_float(math.nonzero(mask, list_dim=instance('points')))
    temp = []
    for _ in range(points_per_cell):
        if center:
            temp.append(indices + 0.5)
        else:
            temp.append(indices + (math.random_uniform(indices.shape)))
    return math.concat(temp, dim=instance('points'))

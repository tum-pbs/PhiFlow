import numpy as np

from phi import math
from phi.geom import Box, Geometry, assert_same_rank, GridCell, AbstractBox
from ._field import Field
from ._field import SampledField
from ._mask import SoftGeometryMask, HardGeometryMask
from ..geom._stack import GeometryStack
from ..math import tensor, Shape
from ..math._tensors import TensorStack, Tensor


class Grid(SampledField):
    """
    Base class for CenteredGrid, StaggeredGrid.
    
    Grids are defined by
    
    * data: Tensor, defines resolution
    * bounds: physical size of the grid, defines dx
    * extrapolation: values of virtual grid points lying outside the data bounds

    Args:

    Returns:

    """

    def __init__(self, values: Tensor, resolution: Shape, bounds: Box, extrapolation=math.extrapolation.ZERO):
        SampledField.__init__(self, GridCell(resolution, bounds), values, extrapolation)
        self._bounds = bounds
        assert_same_rank(self.values.shape, bounds, 'data dimensions %s do not match box %s' % (self.values.shape, bounds))

    def sample_at(self, points: Tensor, reduce_channels=()) -> Tensor:
        raise NotImplementedError(self)

    def closest_values(self, points: Tensor, reduce_channels=()):
        """
        Sample the closest grid point values of this field at the world-space locations (in physical units) given by `points`.
        Points must have a single channel dimension named `vector`.
        It may additionally contain any number of batch and spatial dimensions, all treated as batch dimensions.

        Args:
          points: world-space locations
          reduce_channels: (optional) See `Field.sample_at()` for a description.

        Returns:
          Closest grid point values as a `Tensor`.
          For each dimension, the grid points immediately left and right of the sample points are evaluated.
          For each point in `points`, a *2^d* cube of points is determined where *d* is the number of spatial dimensions of this field.
          These values are stacked along the new dimensions `'closest_<dim>'` where `<dim>` refers to the name of a spatial dimension.
        """
        raise NotImplementedError(self)

    @property
    def bounds(self) -> Box:
        return self._bounds

    @property
    def box(self) -> Box:
        return self._bounds

    @property
    def resolution(self) -> Shape:
        return self.shape.spatial

    @property
    def dx(self) -> Tensor:
        return self.box.size / self.resolution

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.shape}, size={self.box.size}, extrapolation={self._extrapolation}]"


class CenteredGrid(Grid):
    """
    N-dimensional grid with values sampled at the cell centers.
    A centered grid is defined through its data tensor, its bounds describing the physical size and extrapolation.
    
    Centered grids support arbitrary batch, spatial and channel dimensions.
    
    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html

    Args:

    Returns:

    """

    def __init__(self, values, bounds: Box, extrapolation=math.extrapolation.ZERO):
        Grid.__init__(self, values, values.shape.spatial, bounds, extrapolation)

    @staticmethod
    def sample(value: Geometry or Field or int or float or callable,
               resolution: Shape,
               box: Box,
               extrapolation=math.extrapolation.ZERO):
        if isinstance(value, Geometry):
            value = SoftGeometryMask(value)
        if isinstance(value, Field):
            elements = GridCell(resolution, box)
            data = value.sample_in(elements)
        else:
            if callable(value):
                x = GridCell(resolution, box).center
                value = value(x)
            value = tensor(value)
            data = math.zeros(resolution) + value
        return CenteredGrid(data, box, extrapolation)

    def sample_in(self, geometry: Geometry, reduce_channels=()) -> Tensor:
        if reduce_channels:
            assert len(reduce_channels) == 1
            geometries = geometry.unstack(reduce_channels[0])
            components = self.vector.unstack(len(geometries))
            sampled = [c.sample_in(g) for c, g in zip(components, geometries)]
            return math.channel_stack(sampled, 'vector')
        if isinstance(geometry, GeometryStack):
            sampled = [self.sample_in(g) for g in geometry.geometries]
            return math.batch_stack(sampled, geometry.stack_dim_name)
        if isinstance(geometry, GridCell):
            if self.elements == geometry:
                return self.values
            elif math.close(self.dx, geometry.size):
                fast_resampled = self._shift_resample(geometry.resolution, geometry.bounds)
                if fast_resampled is not NotImplemented:
                    return fast_resampled
        return self.sample_at(geometry.center, reduce_channels)

    def sample_at(self, points, reduce_channels=()) -> Tensor:
        local_points = self.box.global_to_local(points) * self.resolution - 0.5
        if len(reduce_channels) == 0:
            return math.grid_sample(self.values, local_points, self.extrapolation)
        else:
            assert self.shape.channel.sizes == points.shape.get_size(reduce_channels)
            if len(reduce_channels) > 1:
                raise NotImplementedError(f"{len(reduce_channels)} > 1. Only 1 reduced channel allowed.")
            channels = []
            for i, channel in enumerate(self.values.vector.unstack()):
                channels.append(math.grid_sample(channel, local_points[{reduce_channels[0]: i}], self.extrapolation))
            return math.channel_stack(channels, 'vector')

    def _shift_resample(self, resolution, box, threshold=1e-5, max_padding=20):
        lower = math.to_int(math.ceil(math.maximum(0, self.box.lower - box.lower) / self.dx - threshold))
        upper = math.to_int(math.ceil(math.maximum(0, box.upper - self.box.upper) / self.dx - threshold))
        total_padding = math.sum(lower) + math.sum(upper)
        if total_padding == 0:
            origin_in_local = self.box.global_to_local(box.lower) * self.resolution
            data = math.sample_subgrid(self.values, origin_in_local, resolution)
            return data
        elif total_padding < max_padding:
            from phi.field import pad
            padded = pad(self, {dim: (int(lower[i]), int(upper[i])) for i, dim in enumerate(self.shape.spatial.names)})
            return padded._shift_resample(resolution, box)

    def closest_values(self, points: Tensor, reduce_channels=()):
        local_points = self.box.global_to_local(points) * self.resolution - 0.5
        return math.closest_grid_values(self.values, local_points, self.extrapolation)


class StaggeredGrid(Grid):
    """
    N-dimensional grid whose vector components are sampled at the respective face centers.
    A staggered grid is defined through its values tensor, its bounds describing the physical size and extrapolation.
    
    Staggered grids support arbitrary batch and spatial dimensions but only one channel dimension for the staggered vector components.
    
    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self, values: TensorStack, bounds=None, extrapolation=math.extrapolation.ZERO):
        values = _validate_staggered_values(values)
        x = values.vector['x']
        resolution = x.shape.spatial.with_size('x', x.shape.get_size('x') - 1)
        Grid.__init__(self, values, resolution, bounds, extrapolation)

    @staticmethod
    def sample(value: Field or Geometry or callable or Tensor or float or int,
               resolution: Shape,
               bounds: Box,
               extrapolation=math.extrapolation.ZERO) -> 'StaggeredGrid':
        """
        Creates a StaggeredGrid from `value`.
        `value` has to be one of the following:
        
        * Geometry: sets inside values to 1, outside to 0
        * Field: resamples the Field to the staggered sample points
        * float, int: uses the value for all sample points
        * tuple, list: interprets the sequence as vector, used for all sample points
        * Tensor compatible with grid dims: uses tensor values as grid values

        Args:
          value: values to use for the grid
          resolution: grid resolution
          bounds: physical grid bounds
          extrapolation: return: Sampled values in staggered grid form matching domain resolution (Default value = math.extrapolation.ZERO)
          value: Field or Geometry or callable or Tensor or float or int: 
          resolution: Shape: 
          bounds: Box: 

        Returns:
          Sampled values in staggered grid form matching domain resolution

        """
        if isinstance(value, Geometry):
            value = HardGeometryMask(value)
        if isinstance(value, Field):
            assert_same_rank(value.spatial_rank, bounds.spatial_rank, 'rank of value (%s) does not match domain (%s)' % (value.spatial_rank, bounds.spatial_rank))
            if isinstance(value, StaggeredGrid) and value.bounds == bounds and np.all(value.resolution == resolution):
                return value
            else:
                components = value.vector.unstack(bounds.spatial_rank)
                tensors = []
                for dim, comp in zip(resolution.spatial.names, components):
                    comp_cells = GridCell(resolution, bounds).extend_symmetric(dim, 1)
                    comp_grid = CenteredGrid.sample(comp, comp_cells.resolution, comp_cells.bounds, extrapolation)
                    tensors.append(comp_grid.values)
                return StaggeredGrid(math.channel_stack(tensors, 'vector'), bounds, extrapolation)
        else:  # value is function or constant
            if callable(value):
                points = GridCell(resolution, bounds).face_centers()
                value = value(points)
            value = tensor(value)
            components = (value.staggered if 'staggered' in value.shape else value.vector).unstack(resolution.spatial_rank)
            tensors = []
            for dim, component in zip(resolution.spatial.names, components):
                comp_cells = GridCell(resolution, bounds).extend_symmetric(dim, 1)
                tensors.append(math.zeros(comp_cells.resolution) + component)
            return StaggeredGrid(math.channel_stack(tensors, 'vector'), bounds, extrapolation)

    def with_(self,
              elements: Geometry or None = None,
              values: Tensor = None,
              extrapolation: math.Extrapolation = None) -> 'StaggeredGrid':
        assert elements is None
        values = _validate_staggered_values(values) if values is not None else None
        return Grid.with_(self, values=values, extrapolation=extrapolation)

    @property
    def cells(self):
        return GridCell(self.resolution, self.bounds)

    def sample_in(self, geometry: Geometry, reduce_channels=()) -> Tensor:
        if geometry == self.elements and reduce_channels:
            return self.values
        if not reduce_channels:
            channels = [component.sample_in(geometry) for component in self.unstack()]
        else:
            assert len(reduce_channels) == 1
            geometries = geometry.unstack(reduce_channels[0])
            channels = [component.sample_in(g) for g, component in zip(geometries, self.unstack())]
        return math.channel_stack(channels, 'vector')

    def sample_at(self, points: Tensor, reduce_channels=()) -> Tensor:
        if not reduce_channels:
            channels = [component.sample_at(points) for component in self.unstack()]
        else:
            assert len(reduce_channels) == 1
            points = points.unstack(reduce_channels[0])
            channels = [component.sample_at(p) for p, component in zip(points, self.unstack())]
        return math.channel_stack(channels, 'vector')

    def closest_values(self, points: Tensor, reduce_channels=()):
        if not reduce_channels:
            channels = [component.sample_at(points) for component in self.unstack()]
        else:
            assert len(reduce_channels) == 1
            points = points.unstack(reduce_channels[0])
            channels = [component.closest_values(p) for p, component in zip(points, self.unstack())]
        return math.channel_stack(channels, 'vector')

    def at_centers(self) -> CenteredGrid:
        return CenteredGrid(self.sample_in(self.cells), self.bounds, self.extrapolation)

    def unstack(self, dimension='vector'):
        if dimension == 'vector':
            result = []
            for dim, data in zip(self.resolution.spatial.names, self.values.vector.unstack()):
                comp_cells = GridCell(self.resolution, self._bounds).extend_symmetric(dim, 1)
                result.append(CenteredGrid(data, comp_cells.bounds, self.extrapolation))
            return tuple(result)
        else:
            values = self.values.unstack(dimension)
            return tuple(self.with_(values=v) for v in values)

    @property
    def x(self) -> CenteredGrid:
        """ x component """
        return self.unstack()[self.resolution.index('x')]

    @property
    def y(self) -> CenteredGrid:
        """ y component """
        return self.unstack()[self.resolution.index('y')]

    @property
    def z(self) -> CenteredGrid:
        """ z component """
        return self.unstack()[self.resolution.index('z')]

    @property
    def elements(self):
        grids = [grid.elements for grid in self.unstack()]
        return GeometryStack(grids, 'staggered')

    def staggered_tensor(self):
        return stack_staggered_components(self.values)

    def _op2(self, other, operator):
        if isinstance(other, StaggeredGrid) and self.bounds == other.bounds and self.shape.spatial == other.shape.spatial:
            values = operator(self._values, other.values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return self.with_(values=values, extrapolation=extrapolation_)
        else:
            return SampledField._op2(self, other, operator)

    # def downsample2x(self):
    #     values = []
    #     for axis in range(self.rank):
    #         grid = self.unstack()[axis].values
    #         grid = grid[tuple([slice(None, None, 2) if d - 1 == axis else slice(None) for d in range(self.rank + 2)])]  # Discard odd indices along axis
    #         grid = math.downsample2x(grid, dims=tuple(filter(lambda ax2: ax2 != axis, range(self.rank))))  # Interpolate values along other dims
    #         values.append(grid)
    #     return self.with_(values=values)


def unstack_staggered_tensor(data: Tensor) -> TensorStack:
    sliced = []
    for dim, component in zip(data.shape.spatial.names, data.unstack('vector')):
        sliced.append(component[{d: slice(None, -1) for d in data.shape.spatial.without(dim).names}])
    return math.channel_stack(sliced, 'vector')


def stack_staggered_components(data: Tensor) -> Tensor:
    padded = []
    for dim, component in zip(data.shape.spatial.names, data.unstack('vector')):
        padded.append(math.pad(component, {d: (0, 1) for d in data.shape.spatial.without(dim).names}, mode=math.extrapolation.ZERO))
    return math.channel_stack(padded, 'vector')


def _validate_staggered_values(values: TensorStack):
    if 'vector' in values.shape:
        return values
    else:
        if 'staggered' in values.shape:
            return values.staggered.as_channel('vector')
        else:
            raise ValueError("values needs to have 'vector' or 'staggered' dimension")

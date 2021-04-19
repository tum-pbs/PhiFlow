from typing import TypeVar

import numpy as np

from phi import math
from phi.geom import Box, Geometry, assert_same_rank, GridCell, AbstractBox
from ._field import Field
from ._field import SampledField
from ._mask import SoftGeometryMask, HardGeometryMask
from ..geom._stack import GeometryStack
from ..math import wrap, Shape
from ..math._shape import CHANNEL_DIM
from ..math._tensors import TensorStack, Tensor


class Grid(SampledField):
    """
    Base class for CenteredGrid, StaggeredGrid.
    
    Grids are defined by
    
    * data: Tensor, defines resolution
    * bounds: physical size of the grid, defines dx
    * extrapolation: values of virtual grid points lying outside the data bounds
    """

    def __init__(self, elements: Geometry, values: Tensor, extrapolation: math.Extrapolation, resolution: Shape, bounds: Box):
        SampledField.__init__(self, elements, values, extrapolation)
        self._bounds = bounds
        self._resolution = resolution
        assert values.shape.spatial_rank == bounds.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"

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

    def with_(self, elements: Geometry or None = None, values: Tensor = None, extrapolation: math.Extrapolation = None, **other_attributes) -> 'Grid':
        raise NotImplementedError(self)

    def __getitem__(self, item: dict) -> 'Grid':
        raise NotImplementedError(self)

    @property
    def bounds(self) -> Box:
        return self._bounds

    @property
    def box(self) -> Box:
        return self._bounds

    @property
    def resolution(self) -> Shape:
        return self._resolution

    @property
    def dx(self) -> Tensor:
        return self.box.size / self.resolution

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.shape.non_spatial & self.resolution}, size={self.box.size}, extrapolation={self._extrapolation}]"


GridType = TypeVar('GridType', bound=Grid)


class CenteredGrid(Grid):
    """
    N-dimensional grid with values sampled at the cell centers.
    A centered grid is defined through its data tensor, its bounds describing the physical size and extrapolation.
    
    Centered grids support arbitrary batch, spatial and channel dimensions.
    
    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html

    Args:

    Returns:

    """

    def __init__(self, values, bounds: Box, extrapolation: math.Extrapolation):
        Grid.__init__(self, GridCell(values.shape.spatial, bounds), values, extrapolation, values.shape.spatial, bounds)

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
            value = wrap(value)
            data = math.zeros(resolution) + value
        return CenteredGrid(data, box, extrapolation)

    def with_(self,
              elements: Geometry or None = None,
              values: Tensor = None,
              extrapolation: math.Extrapolation = None,
              **other_attributes) -> 'CenteredGrid':
        assert elements is None
        assert not other_attributes, f"Invalid attributes for type {type(self)}: {other_attributes}"
        values = values if values is not None else self.values
        extrapolation = extrapolation if extrapolation is not None else self._extrapolation
        return CenteredGrid(values, self.bounds, extrapolation)

    def __getitem__(self, item: dict):
        values = self._values[{dim: slice(sel, sel + 1) if isinstance(sel, int) and dim in self.shape.spatial else sel for dim, sel in item.items()}]
        extrapolation = self._extrapolation[item]
        bounds = self.elements[item].bounds
        return CenteredGrid(values, bounds, extrapolation)

    def sample_in(self, geometry: Geometry) -> Tensor:
        if 'vector' in geometry.shape:
            geometries = geometry.unstack('vector')
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
        return self.sample_at(geometry.center)

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
        if total_padding > max_padding:
            return NotImplemented
        elif total_padding > 0:
            from phi.field import pad
            padded = pad(self, {dim: (int(lower[i]), int(upper[i])) for i, dim in enumerate(self.shape.spatial.names)})
            grid_box, grid_resolution, grid_values = padded.box, padded.resolution, padded.values
        else:
            grid_box, grid_resolution, grid_values = self.box, self.resolution, self.values
        origin_in_local = grid_box.global_to_local(box.lower) * grid_resolution
        data = math.sample_subgrid(grid_values, origin_in_local, resolution)
        return data

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

    def __init__(self, values: TensorStack, bounds: Box, extrapolation: math.Extrapolation):
        values = _validate_staggered_values(values)
        any_dim = values.shape.spatial.names[0]
        x = values.vector[any_dim]
        resolution = x.shape.spatial.with_size(any_dim, x.shape.get_size(any_dim) - 1)
        grids = [GridCell(resolution, bounds).extend_symmetric(dim, 1) for dim in values.shape.spatial.names]
        elements = GeometryStack(grids, 'vector', CHANNEL_DIM)
        Grid.__init__(self, elements, values, extrapolation, resolution, bounds)

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
            value = wrap(value)
            components = (value.staggered if 'staggered' in value.shape else value.vector).unstack(resolution.spatial_rank)
            tensors = []
            for dim, component in zip(resolution.spatial.names, components):
                comp_cells = GridCell(resolution, bounds).extend_symmetric(dim, 1)
                tensors.append(math.zeros(comp_cells.resolution) + component)
            return StaggeredGrid(math.channel_stack(tensors, 'vector'), bounds, extrapolation)

    def with_(self,
              elements: Geometry or None = None,
              values: TensorStack = None,
              extrapolation: math.Extrapolation = None,
              **other_attributes) -> 'StaggeredGrid':
        assert elements is None
        assert not other_attributes, f"Invalid attributes for type {type(self)}: {other_attributes}"
        values = _validate_staggered_values(values) if values is not None else self.values
        return StaggeredGrid(values, self.bounds, extrapolation if extrapolation is not None else self._extrapolation)

    @property
    def cells(self):
        return GridCell(self.resolution, self.bounds)

    def sample_in(self, geometry: Geometry) -> Tensor:
        if self.elements.shallow_equals(geometry):
            return self.values
        if 'vector' in geometry.shape:
            geometries = geometry.unstack('vector')
            channels = [component.sample_in(g) for g, component in zip(geometries, self.vector.unstack())]
        else:
            channels = [component.sample_in(geometry) for component in self.vector.unstack()]
        return math.channel_stack(channels, 'vector')

    def sample_at(self, points: Tensor, reduce_channels=()) -> Tensor:
        if not reduce_channels:
            channels = [component.sample_at(points) for component in self.vector.unstack()]
        else:
            assert len(reduce_channels) == 1
            points = points.unstack(reduce_channels[0])
            channels = [component.sample_at(p) for p, component in zip(points, self.vector.unstack())]
        return math.channel_stack(channels, 'vector')

    def closest_values(self, points: Tensor, reduce_channels=()):
        if not reduce_channels:
            channels = [component.sample_at(points) for component in self.vector.unstack()]
        else:
            assert len(reduce_channels) == 1
            points = points.unstack(reduce_channels[0])
            channels = [component.closest_values(p) for p, component in zip(points, self.vector.unstack())]
        return math.channel_stack(channels, 'vector')

    def at_centers(self) -> CenteredGrid:
        return CenteredGrid(self.sample_in(self.cells), self.bounds, self.extrapolation)

    # @property
    # def x(self) -> CenteredGrid:
    #     """ X component as `CenteredGrid`. Equal to `grid.vector['x']` """
    #     return self.vector['x']
    #
    # @property
    # def y(self) -> CenteredGrid:
    #     """ Y component as `CenteredGrid`. Equal to `grid.vector['y']` """
    #     return self.vector['y']
    #
    # @property
    # def z(self) -> CenteredGrid:
    #     """ Z component as `CenteredGrid`. Equal to `grid.vector['z']` """
    #     return self.vector['z']

    def __getitem__(self, item: dict):
        values = self._values[{dim: sel for dim, sel in item.items() if dim not in self.shape.spatial}]
        for dim, sel in item.items():
            if dim in self.shape.spatial:
                sel = slice(sel, sel + 1) if isinstance(sel, int) else sel
                values = []
                for vdim, val in zip(self.shape.spatial.names, self.values.unstack('vector')):
                    if vdim == dim:
                        values.append(val[{dim: slice(sel.start, sel.stop + 1)}])
                    else:
                        values.append(val[{dim: sel}])
                values = math.channel_stack(values, 'vector')
        extrapolation = self._extrapolation[item]
        bounds = GridCell(self._resolution, self._bounds)[item].bounds
        if 'vector' in item:
            if isinstance(item['vector'], int):
                dim = self.shape.spatial.names[item['vector']]
                comp_cells = GridCell(self.resolution, bounds).extend_symmetric(dim, 1)
                return CenteredGrid(values, comp_cells.bounds, extrapolation)
            else:
                assert isinstance(item['vector'], slice) and not item['vector'].start and not item['vector'].stop
        return StaggeredGrid(values, bounds, extrapolation)

    def staggered_tensor(self):
        return stack_staggered_components(self.values)

    def _op2(self, other, operator):
        if isinstance(other, StaggeredGrid) and self.bounds == other.bounds and self.shape.spatial == other.shape.spatial:
            values = operator(self._values, other.values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return self.with_(values=values, extrapolation=extrapolation_)
        else:
            return SampledField._op2(self, other, operator)


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

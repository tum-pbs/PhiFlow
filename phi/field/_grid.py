import numpy as np

from phi import math
from phi.geom import Box, Geometry, assert_same_rank, GridCell, AbstractBox
from ._field import Field, IncompatibleFieldTypes
from ._field import SampledField
from ._mask import GeometryMask
from ..geom._stack import GeometryStack
from ..math import tensor, Shape
from ..math._shape import CHANNEL_DIM
from ..math._tensors import TensorStack, Tensor


class Grid(SampledField):
    """
    Base class for CenteredGrid, StaggeredGrid.

    Grids are defined by

    * data: Tensor, defines resolution
    * box: physical size of the grid, defines dx
    * extrapolation: values of virtual grid points lying outside the data bounds
    """

    def __init__(self, values: Tensor, resolution: Shape, bounds: Box, extrapolation=math.extrapolation.ZERO):
        SampledField.__init__(self, GridCell(resolution, bounds), values, extrapolation)
        self._bounds = bounds
        assert_same_rank(self.values.shape, bounds, 'data dimensions %s do not match box %s' % (self.values.shape, bounds))

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
        return '%s[%s, size=%s, extrapolation=%s]' % (self.__class__.__name__, self.shape, self.box.size, self._extrapolation)


class CenteredGrid(Grid):
    """
    N-dimensional grid with values sampled at the cell centers.
    A centered grid is defined through its data tensor, its box describing the physical size and extrapolation.

    Centered grids support arbitrary batch, spatial and channel dimensions.
    """

    def __init__(self, values, bounds: Box, extrapolation=math.extrapolation.ZERO):
        Grid.__init__(self, values, values.shape.spatial, bounds, extrapolation)

    @staticmethod
    def sample(value: Geometry or Field or int or float or callable, resolution, box, extrapolation=math.extrapolation.ZERO):
        if isinstance(value, Geometry):
            value = GeometryMask(value)
        if isinstance(value, Field):
            elements = GridCell(resolution, box)
            data = value.volume_sample(elements)
        else:
            if callable(value):
                x = GridCell(resolution, box).center
                value = value(x)
            value = tensor(value, infer_dimension_types=False)
            data = math.zeros(resolution) + value
        return CenteredGrid(data, box, extrapolation)

    def volume_sample(self, geometry: Geometry, reduce_channels=()) -> Tensor:
        if isinstance(geometry, GridCell):
            if geometry.bounds == self.box and geometry.resolution == self.resolution:
                return self.values
            elif math.close(self.dx, geometry.size):
                fast_resampled = self._shift_resample(geometry.resolution, geometry.bounds)
                if fast_resampled is not NotImplemented:
                    return fast_resampled
        return self.sample_at(geometry.center, reduce_channels)

    def sample_at(self, points, reduce_channels=()) -> Tensor:
        local_points = self.box.global_to_local(points)
        local_points = local_points * self.resolution - 0.5
        if len(reduce_channels) == 0:
            return math.grid_sample(self.values, local_points, self.extrapolation)
        else:
            assert self.shape.channel.sizes == points.shape.get_size(reduce_channels)
            if len(reduce_channels) > 1:
                raise NotImplementedError()
            channels = []
            for i, channel in enumerate(self.values.vector.unstack()):
                channels.append(math.grid_sample(channel, local_points[{reduce_channels[0]: i}], self.extrapolation))
            return math.channel_stack(channels, 'vector')

    def _shift_resample(self, resolution, box):
        paddings = _required_paddings_transposed(self.box, self.dx, box)
        if math.sum(paddings) == 0:
            origin_in_local = self.box.global_to_local(box.lower) * self.resolution
            data = math.interpolate_linear(self.values, origin_in_local, resolution.sizes)
            return data
        elif math.sum(paddings) < 16:
            padded = self.padded(np.transpose(paddings).tolist())
            return padded.at(representation)

    def closest_values(self, points):
        local_points = self.box.global_to_local(points)
        indices = local_points * math.to_float(self.resolution) - 0.5
        return math.closest_grid_values(self.values, indices, self.extrapolation)

    def compatible(self, other):
        return isinstance(other, CenteredGrid) and other.box == self.box and other.resolution == self.resolution


def _required_paddings_transposed(box, dx, target, threshold=1e-5):
    lower = math.to_int(math.ceil(math.maximum(0, box.lower - target.lower) / dx - threshold))
    upper = math.to_int(math.ceil(math.maximum(0, target.upper - box.upper) / dx - threshold))
    return [lower, upper]


class StaggeredGrid(Grid):
    """
    N-dimensional grid whose vector components are sampled at the respective face centers.
    A staggered grid is defined through its values tensor, its box describing the physical size and extrapolation.

    Centered grids support arbitrary batch and spatial dimensions but only one channel dimension for the staggered vector components.
    """

    def __init__(self, values: TensorStack, bounds=None, extrapolation=math.extrapolation.ZERO):
        values = _validate_staggered_values(values)
        x = values.vector[0 if math.GLOBAL_AXIS_ORDER.is_x_first else -1]
        resolution = x.shape.spatial.with_size('x', x.shape.get_size('x') - 1)
        Grid.__init__(self, values, resolution, bounds, extrapolation)

    @staticmethod
    def sample(value, resolution, box, extrapolation=math.extrapolation.ZERO):
        """
        Sampmles the value to a staggered grid.

        :param value: Either constant, staggered tensor, or Field
        :return: Sampled values in staggered grid form matching domain resolution
        :rtype: StaggeredGrid
        """
        if isinstance(value, Geometry):
            value = GeometryMask(value)
        if isinstance(value, Field):
            assert_same_rank(value.rank, box.rank, 'rank of value (%s) does not match domain (%s)' % (value.rank, box.rank))
            if isinstance(value, StaggeredGrid) and value.box == box and np.all(value.resolution == resolution):
                return value
            else:
                components = value.unstack('vector') if 'vector' in value.shape else [value] * box.rank
                tensors = []
                for dim, comp in zip(resolution.spatial.names, components):
                    comp_res, comp_box = extend_symmetric(resolution, box, dim)
                    comp_grid = CenteredGrid.sample(comp, comp_res, comp_box, extrapolation)
                    tensors.append(comp_grid.values)
                return StaggeredGrid(math.channel_stack(tensors, 'vector'), box, extrapolation)
        elif callable(value):
            raise NotImplementedError()
            x = CenteredGrid.getpoints(domain.bounds, domain.resolution).copied_with(extrapolation=Material.extrapolation_mode(domain.boundaries), name=name)
            value = value(x)
            return value
        else:  # value is constant
            tensors = []
            for dim in resolution.spatial.names:
                comp_res, comp_box = extend_symmetric(resolution, box, dim)
                tensors.append(math.zeros(comp_res) + value)
            return StaggeredGrid(math.channel_stack(tensors, 'vector'), box, extrapolation)

    def _with(self, values: Tensor = None, extrapolation: math.Extrapolation = None):
        values = _validate_staggered_values(values) if values is not None else None
        return Grid._with(self, values, extrapolation)

    def sample_at(self, points, reduce_channels=()):
        if not reduce_channels:
            if isinstance(points, StaggeredGrid) and points.resolution == self.resolution and points.bounds == self.bounds:
                return self
            channels = [component.sample_at(points) for component in self.unstack()]
        else:
            assert len(reduce_channels) == 1
            points = points.unstack(reduce_channels[0])
            channels = [component.sample_at(p) for p, component in zip(points, self.unstack())]
        return math.channel_stack(channels, 'vector')

    def at_centers(self):
        centered = []
        for grid in self.unstack():
            centered.append(CenteredGrid.sample(grid, self.resolution, self.bounds, self._extrapolation).values)
        tensor = math.channel_stack(centered, 'vector')
        return CenteredGrid(tensor, self.bounds, self._extrapolation)

    def unstack(self, dimension='vector'):
        if dimension == 'vector':
            result = []
            for dim, data in zip(self.resolution.spatial.names, self.values.vector.unstack()):
                result.append(CenteredGrid(data, extend_symmetric(self.resolution, self.box, dim)[1], self.extrapolation))
            return tuple(result)
        else:
            raise NotImplementedError()

    @property
    def x(self):
        return self.unstack()[self.resolution.index('x')]

    @property
    def y(self):
        return self.unstack()[self.resolution.index('y')]

    @property
    def z(self):
        return self.unstack()[self.resolution.index('z')]

    @property
    def elements(self):
        grids = [grid.elements for grid in self.unstack()]
        return GeometryStack(grids, 'staggered')

    def __repr__(self):
        return 'StaggeredGrid[%s, size=%s]' % (self.shape, self.box.size.numpy())

    def staggered_tensor(self):
        return stack_staggered_components(self.values)

    def _op2(self, other, operator):
        if isinstance(other, StaggeredGrid) and self.bounds == other.bounds and self.shape.spatial == other.shape.spatial:
            values = operator(self._values, other.values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return self._with(values, extrapolation_)
        else:
            return SampledField._op2(self, other, operator)
    #
    # def padded(self, widths):

    #
    # def downsample2x(self):
    #     values = []
    #     for axis in range(self.rank):
    #         grid = self.unstack()[axis].values
    #         grid = grid[tuple([slice(None, None, 2) if d - 1 == axis else slice(None) for d in range(self.rank + 2)])]  # Discard odd indices along axis
    #         grid = math.downsample2x(grid, axes=tuple(filter(lambda ax2: ax2 != axis, range(self.rank))))  # Interpolate values along other axes
    #         values.append(grid)
    #     return self._with(values)


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


def extend_symmetric(resolution: Shape, box: AbstractBox, axis, cells=1):
    axis_mask = np.array(resolution.mask(axis)) * cells
    unit = box.size / resolution * axis_mask
    delta_size = unit / 2
    box = Box(box.lower - delta_size, box.upper + delta_size)
    ext_res = resolution.sizes + axis_mask
    return resolution.with_sizes(ext_res), box


def _validate_staggered_values(values: TensorStack):
    if 'vector' in values.shape:
        return values
    else:
        if 'staggered' in values.shape:
            return values.staggered.as_channel('vector')
        else:
            raise ValueError("values needs to have 'vector' or 'staggered' dimension")

import numpy as np

from phi import struct, math
from ._geom import Geometry, assert_same_rank, _fill_spatial_with_singleton
from ._transform import rotate
from ..math import tensor, combined_shape, spatial_shape
from ..math._shape import CHANNEL_DIM
from ..math._tensors import NativeTensor, TensorStack, Tensor


class AbstractBox(Geometry):

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def center(self):
        raise NotImplementedError()

    def shifted(self, delta):
        raise NotImplementedError()

    @property
    def size(self) -> Tensor:
        raise NotImplementedError(self)

    @property
    def half_size(self) -> Tensor:
        raise NotImplementedError(self)

    @property
    def lower(self) -> Tensor:
        raise NotImplementedError(self)

    @property
    def upper(self) -> Tensor:
        raise NotImplementedError(self)

    def bounding_radius(self):
        return math.max(self.size, 'vector') * 1.414214

    def bounding_half_extent(self):
        return self.size * 0.5

    def global_to_local(self, global_position):
        return (global_position - self.lower) / self.size

    def local_to_global(self, local_position):
        return local_position * self.size + self.lower

    def lies_inside(self, location):
        bool_inside = (location >= self.lower) & (location <= self.upper)
        return math.all(bool_inside, 'vector')

    def approximate_signed_distance(self, location):
        """
Computes the signed L-infinity norm (manhattan distance) from the location to the nearest side of the box.
For an outside location `l` with the closest surface point `s`, the distance is `max(abs(l - s))`.
For inside locations it is `-max(abs(l - s))`.
        :param location: float tensor of shape (batch_size, ..., rank)
        :return: float tensor of shape (*location.shape[:-1], 1).
        """
        center = 0.5 * (self.lower + self.upper)
        extent = self.upper - self.lower
        distance = math.abs(location - center) - extent * 0.5
        return math.max(distance, 'vector')

    def get_lower(self, axis):
        return self._get(self.lower, axis)

    def get_upper(self, axis):
        return self._get(self.upper, axis)

    @staticmethod
    def _get(vector, axis):
        if vector.shape[-1] == 1:
            return vector[...,0]
        else:
            return vector[...,axis]

    def corner_representation(self):
        return Box(self.lower, self.upper)

    def center_representation(self):
        return Cuboid(self.center, self.half_size)

    def contains(self, other):
        if isinstance(other, AbstractBox):
            return np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper)
        else:
            raise NotImplementedError()

    def rotated(self, angle):
        return rotate(self, angle)


class BoxType(type):
    """
    Convenience function for creating N-dimensional boxes / cuboids.

    Examples to create a box from (0, 0) to (10, 20):

    * box[0:10, 0:20]
    * box((0, 0), (10, 20))
    * box((5, 10), size=(10, 20))
    * box(center=(5, 10), size=(10, 20))
    * box((10, 20))
    """
    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = [item]
        lower = []
        upper = []
        for dim in item:
            assert isinstance(dim, slice)
            assert dim.step is None or dim.step == 1, "Box: step must be 1 but is %s" % dim.step
            lower.append(dim.start if dim.start is not None else -np.inf)
            upper.append(dim.stop if dim.stop is not None else np.inf)
        return Box(lower, upper)


class Box(AbstractBox, metaclass=BoxType):
    """
    Axis-aligned box, defined by lower and upper corner.
    Boxes can be created using the shorthand notation box[slices], (e.g. box[:,0:1] to create an inifinite-height box from x=0 to x=1).
    """

    def __init__(self, lower, upper):
        self._lower = tensor(lower, names='..., vector', channel_dims=1, spatial_dims=0)
        self._upper = tensor(upper, names='..., vector', channel_dims=1, spatial_dims=0)
        self._shape = _fill_spatial_with_singleton(combined_shape(self._lower, self._upper))

    @property
    def shape(self):
        return self._shape

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def size(self):
        return self.upper - self.lower

    @struct.derived()
    def center(self):
        return 0.5 * (self.lower + self.upper)

    @struct.derived()
    def half_size(self):
        return self.size * 0.5

    def without_axis(self, axis):
        lower = []
        upper = []
        for ax in range(self.rank):
            if ax != axis:
                lower.append(self.get_lower(ax))
                upper.append(self.get_upper(ax))
        return Box(lower, upper)

    def shifted(self, delta):
        return Box(self.lower + delta, self.upper + delta)

    def __repr__(self):
        if self.shape.non_channel.volume == 1:
            return 'box[%s at %s]' % ('x'.join([str(x) for x in self.size.numpy().flatten()]), ','.join([str(x) for x in self.lower.numpy().flatten()]))
        else:
            return 'box[shape=%s]' % self._shape

    @staticmethod
    def to_box(value, resolution_hint=None):
        if value is None:
            assert resolution_hint is not None
            result = Box([0] * len(resolution_hint), tensor(resolution_hint))
        elif isinstance(value, Box):
            result = value
        elif isinstance(value, int):
            if resolution_hint is None:
                result = Box(0, value)
            else:
                size = [value] * (1 if math.ndims(resolution_hint) == 0 else len(resolution_hint))
                result = Box(0, size)
        else:
            raise ValueError("Box extent not understood: '%s'" % value)
        if resolution_hint is not None:
            assert_same_rank(len(resolution_hint), result, 'Box rank does not match resolution.')
        return result


class Cuboid(AbstractBox):

    def __init__(self, center, half_size):
        self._center = tensor(center, names='..., vector', channel_dims=1, spatial_dims=0)
        self._half_size = tensor(half_size, names='..., vector', channel_dims=1, spatial_dims=0)
        self._shape = _fill_spatial_with_singleton(combined_shape(self._center, self._half_size)).without('vector')

    @property
    def center(self):
        return self._center

    @property
    def half_size(self):
        return self._half_size

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return 2 * self.half_size

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    def shifted(self, delta):
        return Cuboid(self._center + delta, self._half_size)


# def box(*args, **kwargs):
#     if len(args) == 2 and len(kwargs) == 0:
#         lower, upper = args
#         return Box(lower, upper)
#     elif len(args) == 1 and 'size' in kwargs:
#         center, = args
#         size = kwargs['size']
#         return Cuboid(center, 0.5 * tensor(size))
#     elif 'size' in kwargs and 'center' in kwargs:
#         center, size = kwargs['center'], kwargs['size']
#         return Cuboid(center, 0.5 * tensor(size))
#     elif len(args) == 1 and len(kwargs) == 0:
#         return Box(0, args[0])
#     else:
#         raise ValueError('Cannot create box from args=%s, kwargs=%s' % (args, kwargs))


def bounding_box(geometry):
    center = geometry.center
    extent = geometry.bounding_half_extent()
    return Box(lower=center - extent, upper=center + extent)


class GridCell(Cuboid):

    def __init__(self, resolution: math.Shape, bounds: AbstractBox):
        assert resolution.spatial_rank == resolution.rank, 'resolution must be purely spatial but got %s' % (resolution,)
        local_coords = math.meshgrid(*[np.linspace(0.5 / size, 1 - 0.5 / size, size) for size in resolution.sizes])
        points = bounds.local_to_global(local_coords)
        Cuboid.__init__(self, points, half_size=bounds.size / resolution.sizes / 2)
        self._resolution = resolution
        self._bounds = bounds

    @property
    def resolution(self):
        return self._resolution

    @property
    def bounds(self):
        return self._bounds

    @property
    def grid_size(self):
        return self._bounds.size

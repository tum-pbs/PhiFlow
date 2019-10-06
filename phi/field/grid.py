from .field import *
from phi import math, struct
from phi.math.initializers import _is_python_shape
import numpy as np


def initialize_field(value, shape, dtype=np.float32):
    if isinstance(value, (int, float)):
        return math.zeros(shape, dtype=dtype) + value
    elif callable(value):
        return value(shape, dtype=dtype)
    if isinstance(shape, struct.Struct):
        if type(shape) == type(value):
            zipped = struct.zip([value, shape], leaf_condition=_is_python_shape)
            return struct.map(lambda val, sh: initialize_field(val, sh), zipped)
        else:
            return type(shape)(value)
    else:
        return value


def _crop_for_interpolation(data, offset_float, window_resolution):
    offset = math.to_int(offset_float)
    slices = [slice(o, o+res+1) for o, res in zip(offset, window_resolution)]
    data = data[[slice(None)] + slices + [slice(None)]]
    return data


class CenteredGrid(Field):

    __struct__ = Field.__struct__.extend([], ['_interpolation'])

    def __init__(self, name, box, data, flags=(), batch_size=None):
        Field.__init__(self, name=name, bounds=box, data=data, flags=flags, batch_size=batch_size)
        self._sample_points = None
        self._extrapolation = None  # TODO
        self._interpolation = 'linear'
        self._boundary = 'replicate'  # TODO this is a temporary replacement for extrapolation

    @property
    def resolution(self):
        return math.as_tensor(math.staticshape(self._data)[1:-1])

    @property
    def dx(self):
        return self.bounds.size / self.resolution

    @property
    def domain(self):
        return self.bounds

    @property
    def rank(self):
        return math.spatial_rank(self.data)

    def sample_at(self, points):
        local_points = self.domain.global_to_local(points) * self.resolution - 0.5
        return math.resample(self.data, local_points, boundary=self._boundary, interpolation=self._interpolation)

    def resample(self, other_field, force_optimization=False):
        if self.compatible(other_field):
            return self

        if isinstance(other_field, CenteredGrid) and np.all(self.dx == other_field.dx) and self.bounds.contains(other_field.bounds):
            origin_in_local = self.bounds.global_to_local(other_field.bounds.origin) * self.resolution
            data = _crop_for_interpolation(self.data, origin_in_local, other_field.resolution)
            dimensions = self.resolution != other_field.resolution
            dimensions = [d for d in math.spatial_dimensions(data) if dimensions[d-1]]
            data = math.interpolate_linear(data, origin_in_local % 1.0, dimensions)
            return CenteredGrid(self.name, other_field.bounds, data, batch_size=self._batch_size)

        return Field.resample(self, other_field, force_optimization=force_optimization)

    def component_count(self):
        return self._data.shape[-1]

    def unstack(self):
        return [CenteredGrid('%s[...,%d]' % (self.name, i), self.bounds, c) for i,c in enumerate(math.unstack(self._data, -1))]

    @property
    def points(self):
        if 'points' in self.flags:
            return self
        if self._sample_points is None:
            idx_zyx = np.meshgrid(*[np.linspace(0.5 / dim, 1 - 0.5 / dim, dim) for dim in self.resolution], indexing="ij")
            local_coords = math.expand_dims(math.stack(idx_zyx, axis=-1), 0)
            points = self.bounds.local_to_global(local_coords)
            self._sample_points = CenteredGrid('%s.points', self.domain, points, flags=['points'])
        return self._sample_points

    def compatible(self, other_field):
        if isinstance(other_field, CenteredGrid):
            return self.bounds == other_field.bounds and np.all(self.resolution == other_field.resolution)
        else:
            return False

    def __repr__(self):
        return 'Grid[%s, size=%s]' % ('x'.join([str(r) for r in self.resolution]), self.bounds.size)
from .field import *
from .constant import *
from .flag import SAMPLE_POINTS
from phi import math
import numpy as np
from phi.geom import Box


def _crop_for_interpolation(data, offset_float, window_resolution):
    offset = math.to_int(offset_float)
    slices = [slice(o, o+res+1) for o, res in zip(offset, window_resolution)]
    data = data[[slice(None)] + slices + [slice(None)]]
    return data


class CenteredGrid(Field):

    __struct__ = Field.__struct__.extend([], ['_box'])

    def __init__(self, name, box, data, flags=(), batch_size=None):
        Field.__init__(self, name=name, bounds=box, data=data, flags=flags, batch_size=batch_size)
        assert isinstance(box, Box) or box is None
        self._box = box
        self._sample_points = None
        self._extrapolation = None  # TODO
        self._interpolation = 'linear'
        self._boundary = 'replicate'  # TODO this is a temporary replacement for extrapolation

    @property
    def resolution(self):
        return math.as_tensor(math.staticshape(self._data)[1:-1])

    @property
    def box(self):
        return self._box

    @property
    def dx(self):
        return self.box.size / self.resolution

    @property
    def rank(self):
        return math.spatial_rank(self.data)

    def sample_at(self, points, collapse_dimensions=True):
        local_points = self.box.global_to_local(points)
        local_points = local_points * math.to_float(self.resolution) - 0.5
        return math.resample(self.data, local_points, boundary=self._boundary, interpolation=self._interpolation)

    def at(self, other_field, collapse_dimensions=True, force_optimization=False):
        if self.compatible(other_field):
            return self

        if isinstance(other_field, CenteredGrid) and np.allclose(self.dx, other_field.dx):
            paddings = _required_paddings_transposed(self.box, self.dx, other_field.box)
            if math.sum(paddings) == 0:
                origin_in_local = self.box.global_to_local(other_field.box.origin) * self.resolution
                data = _crop_for_interpolation(self.data, origin_in_local, other_field.resolution)
                dimensions = self.resolution != other_field.resolution
                dimensions = [d for d in math.spatial_dimensions(data) if dimensions[d-1]]
                data = math.interpolate_linear(data, origin_in_local % 1.0, dimensions)
                return CenteredGrid(self.name, other_field.box, data, batch_size=self._batch_size)
            elif math.sum(paddings) < 16:
                padded = self.padded(np.transpose(paddings).tolist())
                return padded.at(other_field, collapse_dimensions, force_optimization)

        return Field.at(self, other_field, force_optimization=force_optimization)

    @property
    def component_count(self):
        return self._data.shape[-1]

    def unstack(self):
        flags = propagate_flags_children(self.flags, self.rank, 1)
        return [CenteredGrid('%s[...,%d]' % (self.name, i), self.box, c, flags=flags, batch_size=self._batch_size) for i,c in enumerate(math.unstack(self._data, -1))]

    @property
    def points(self):
        if SAMPLE_POINTS in self.flags:
            return self
        if self._sample_points is None:
            self._sample_points = CenteredGrid.getpoints(self.box, self.resolution)
        return self._sample_points

    def compatible(self, other_field):
        if not other_field.has_points: return True
        if isinstance(other_field, CenteredGrid):
            if self.box != other_field.box: return False
            if self.rank != other_field.rank: return False
            for r1, r2 in zip(self.resolution, other_field.resolution):
                if r1 != r2 and r2 != 1 and r1 != 1:
                    return False
            return True
        else:
            return False

    def __repr__(self):
        return 'Grid[%s(%d), size=%s]' % ('x'.join([str(r) for r in self.resolution]), self.component_count, self.box.size)

    def padded(self, widths):
        data = math.pad(self.data, [[0,0]]+widths+[[0,0]], 'symmetric' if self._boundary == 'replicate' else 'constant')
        w_lower, w_upper = np.transpose(widths)
        box = Box(self.box.origin - w_lower * self.dx, self.box.size + (w_lower+w_upper) * self.dx)
        return CenteredGrid(self.name, box, data, batch_size=self._batch_size)

    @staticmethod
    def getpoints(box, resolution):
        idx_zyx = np.meshgrid(*[np.linspace(0.5 / dim, 1 - 0.5 / dim, dim) for dim in resolution], indexing="ij")
        local_coords = math.expand_dims(math.stack(idx_zyx, axis=-1), 0).astype(np.float32)
        points = box.local_to_global(local_coords)
        return CenteredGrid('%s.points', box, points, flags=[SAMPLE_POINTS])


def _required_paddings_transposed(box, dx, target):
    lower = math.to_int(math.ceil(math.maximum(0, box.origin - target.origin)))
    upper = math.to_int(math.ceil(math.maximum(0, target.upper - box.upper)))
    return [lower, upper]
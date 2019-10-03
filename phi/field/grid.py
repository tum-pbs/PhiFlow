from .field import *
from phi import math
import numpy as np


class CenteredGrid(Field):

    __struct__ = Field.__struct__.extend([], [])

    def __init__(self, domain, data, flags=()):
        Field.__init__(self, bounds=domain, flags=flags)
        self._data = data
        self._sample_points = None

    @property
    def resolution(self):
        return math.staticshape(self._data)[1:-1]

    def resample(self, location):
        if self.compatible(location):
            return self
        # If compatible: return self
        pass

    def component_count(self):
        return self._data.shape[-1]

    def unstack(self):
        return [CenteredGrid(self.bounds, c) for c in math.unstack(self._data, -1)]

    def sample_points(self):
        if self._sample_points is None:
            idx_zyx = np.meshgrid(*[np.linspace(0.5 / dim, 1 - 0.5 / dim, dim) for dim in self.resolution], indexing="ij")
            local_coords = math.expand_dims(math.stack(idx_zyx, axis=-1), 0)
            self._sample_points = self.bounds.local_to_global(local_coords)
        return self._sample_points

    def compatible(self, other_field):
        if isinstance(other_field, CenteredGrid):
            return self.bounds == other_field.bounds and self.resolution == other_field.resolution
        else:
            return False

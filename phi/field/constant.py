from .field import *
from phi import math


class ConstantField(Field):

    __struct__ = Field.__struct__

    def __init__(self, name, value=1.0, flags=(), batch_size=None):
        Field.__init__(self, name, None, _convert_constant_to_data(value), flags=flags, batch_size=batch_size)

    def sample_at(self, points, collapse_dimensions=True):
        collapsed = math.expand_dims(self.data, 1, math.spatial_rank(points))
        if collapse_dimensions: return collapsed
        else:
            return math.tile

    @property
    def rank(self):
        return None

    @property
    def component_count(self):
        return self.data.shape[-1]

    def unstack(self):
        flags = propagate_flags_children(self.flags, self.rank, 1)
        return [ConstantField('%s[%d]' % (self.name, i), c, flags, self._batch_size) for i,c in enumerate(math.unstack(self.data, -1))]

    @property
    def points(self):
        return None

    def compatible(self, other_field):
        return True

    def __repr__(self):
        return repr(self.data)


def _convert_constant_to_data(value):
    if isinstance(value, math.Number):
        value = math.expand_dims(value)
    if len(math.staticshape(value)) < 2:
        value = math.expand_dims(value)
    return value
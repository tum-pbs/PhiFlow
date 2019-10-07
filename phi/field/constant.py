from .field import *
from phi import math


class ConstantField(Field):

    __struct__ = State.__struct__.extend([], ['_data', '_bounds', '_name', '_flags'])

    def __init__(self, name, bounds=None, value=1.0, flags=(), batch_size=None):
        if isinstance(value, math.Number):
            value = math.expand_dims(value)
        Field.__init__(self, name, bounds, value, flags=flags, batch_size=batch_size)

    def sample_at(self, points):
        if self._bounds is None:
            return math.expand_dims(self.data, 1, math.spatial_rank(points))
        else:
            return self.bounds.value_at(points) * self.data

    @property
    def rank(self):
        if self.bounds is None:
            return 0
        else:
            return self.bounds.rank

    @property
    def component_count(self):
        return self.data.shape[-1]

    def unstack(self):
        flags = propagate_flags_children(self.flags, self.rank, 1)
        return [ConstantField('%s[%d]' % (self.name, i), self.bounds, c, flags=flags, batch_size=self._batch_size) for i,c in enumerate(math.unstack(self.data, -1))]

    @property
    def points(self):
        return math.zeros([])

    def compatible(self, other_field):
        return isinstance(other_field, ConstantField)

    def __repr__(self):
        return repr(self.data)


class ComplexConstantField(ConstantField):
    """
    This class is required because complex numbers are not JSON serializable, see https://github.com/bmabey/pyLDAvis/issues/69
    """

    __struct__ = State.__struct__.extend([], ['_real', '_imag', '_bounds', '_name', '_flags'])

    def __init__(self, name, bounds=None, value=1.0, flags=(), batch_size=None):
        if isinstance(value, math.Number):
            value = math.expand_dims(value)
        Field.__init__(self, name, bounds, value, flags=flags, batch_size=batch_size)

    @property
    def real(self):
        return math.real(self.data)

    @property
    def imag(self):
        return math.imag(self.data)

    def sample_at(self, location):
        return math.to_complex(ConstantField.sample_at(self, location))
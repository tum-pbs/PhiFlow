from .field import *
from .constant import _convert_constant_to_data


class GeometryMask(Field):

    __struct__ = Field.__struct__.extend([], ['_geometries'])

    def __init__(self, name, geometries, value=1.0, batch_size=None):
        Field.__init__(self, name, None, _convert_constant_to_data(value), batch_size=batch_size)
        self._geometries = tuple(geometries)

    @property
    def geometries(self):
        return self._geometries

    def sample_at(self, points):
        if len(self._geometries) == 0:
            return math.expand_dims(0, 0, len(math.staticshape(points)))
        result = self._geometries[0].value_at(points)
        for geometry in self._geometries[1:]:
            result += geometry.value_at(points)
        if len(self._geometries) > 1:
            result = math.minimum(result, 1)
        return result * self.data

    @property
    def rank(self):
        return self._geometries[0].rank

    @property
    def component_count(self):
        return 1

    def unstack(self):
        return [self]

    @property
    def points(self):
        return None

    def compatible(self, other_field):
        return True

    def __repr__(self):
        return 'Union{%s}' % ', '.join([repr(g) for g in self._geometries])


def mask(geometry):
    assert isinstance(geometry, Geometry)
    return GeometryMask('mask', [geometry])
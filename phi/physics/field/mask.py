from phi import struct, math
from phi.geom import Geometry

from .field import propagate_flags_children
from .analytic import AnalyticField
from .constant import ConstantField, _convert_constant_to_data, _expand_axes


@struct.definition()
class GeometryMask(AnalyticField):

    def __init__(self, geometries, value=1.0, name=None, flags=(), **kwargs):
        data = _convert_constant_to_data(value)
        AnalyticField.__init__(self, geometries[0].rank, **struct.kwargs(locals(), ignore='value'))

    @struct.constant()
    def geometries(self, geometries):
        return tuple(geometries)

    def sample_at(self, points):
        if len(self.geometries) == 0:
            return _expand_axes(math.zeros([1,1]), points)
        if len(self.geometries) == 1:
            result = self.geometries[0].value_at(points)
        else:
            result = math.max([geometry.value_at(points) for geometry in self.geometries], axis=0)
        return math.mul(result, self.data)

    @property
    def component_count(self):
        return self.data.shape[-1]

    def unstack(self):
        flags = propagate_flags_children(self.flags, self.rank, 1)
        return [GeometryMask(self.geometries, c, '%s[%d]' % (self.name, i), flags, batch_size=self._batch_size) for i, c in enumerate(math.unstack(self.data, -1))]

    def __repr__(self):
        return 'Union{%s}' % ', '.join([repr(g) for g in self.geometries])


def mask(geometry):
    assert isinstance(geometry, Geometry)
    return GeometryMask([geometry], name='mask')


def union_mask(geometries):
    if len(geometries) == 0:
        return ConstantField(0.0)
    for geom in geometries:
        assert isinstance(geom, Geometry)
    return GeometryMask(geometries, name='union')

import numpy as np

from phi import struct
from phi import math


class Geometry(struct.Struct):

    def value_at(self, location):
        raise NotImplementedError(self.__class__)

    @property
    def rank(self):
        raise NotImplementedError()


class AABox(Geometry):

    def __init__(self, lower, upper, **kwargs):
        Geometry.__init__(**struct.kwargs(locals()))

    @struct.prop()
    def lower(self, lower):
        return math.to_float(math.as_tensor(lower))

    @struct.prop()
    def upper(self, upper):
        return math.to_float(math.as_tensor(upper))

    @property
    def size(self):
        return self.upper - self.lower

    @property
    def rank(self):
        if len(self.size.shape) > 0:
            return self.size.shape[-1]
        else:
            return 0

    def global_to_local(self, global_position):
        return (global_position - self.lower) / self.size

    def local_to_global(self, local_position):
        return local_position * self.size + self.lower

    def value_at(self, global_position):
        bool_inside = (global_position >= self.lower) & (global_position <= self.upper)
        bool_inside = math.all(bool_inside, axis=-1, keepdims=True)
        return math.to_float(bool_inside)

    def contains(self, other):
        if isinstance(other, AABox):
            return np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return '%s at (%s)' % ('x'.join([str(x) for x in self.size]), ','.join([str(x) for x in self.lower]))


class AABoxGenerator(object):

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
        return AABox(lower, upper)


box = AABoxGenerator()


class Sphere(Geometry):

    def __init__(self, center, radius, **kwargs):
        Geometry.__init__(**struct.kwargs(locals()))

    @struct.prop()
    def radius(self, radius):
        return math.as_tensor(radius)

    @struct.prop()
    def center(self, center):
        return math.as_tensor(center)

    def value_at(self, location):
        bool_inside = math.expand_dims(math.sum((location - self.center)**2, axis=-1) <= self.radius**2, -1)
        return math.to_float(bool_inside)

    @property
    def rank(self):
        return len(self.center)


class _Union(Geometry):

    def __init__(self, geometries, **kwargs):
        Geometry.__init__(**struct.kwargs(locals()))

    def __validate_geometries__(self):
        assert len(self.geometries) > 0
        rank = self.geometries[0].rank
        for g in self.geometries[1:]:
            assert g.rank == rank or g.rank is None or rank is None
        self.geometries = tuple(self.geometries)

    def value_at(self, points, collapse_dimensions=True):
        if len(self.geometries) == 1:
            result = self.geometries[0].value_at(points)
        else:
            result = math.max([geometry.value_at(points) for geometry in self.geometries], axis=0)
        return result

    @property
    def rank(self):
        if len(self.geometries) == 0:
            return None
        else:
            return self.geometries[0].rank

    @struct.prop()
    def geometries(self, geometries):
        return tuple(geometries)


def union(geometries):
    if len(geometries) == 0:
        return NO_GEOMETRY
    else:
        return _Union(geometries)


class _NoGeometry(Geometry):

    def rank(self):
        return None

    def value_at(self, location):
        return 0


NO_GEOMETRY = _NoGeometry()

from phi import math, struct
import numpy as np


class Geometry(struct.Struct):

    def value_at(self, location):
        raise NotImplementedError(self.__class__)

    @property
    def rank(self):
        raise NotImplementedError()


class Box(Geometry):
    __struct__ = struct.Def([], ['_origin', '_size'])

    def __init__(self, origin, size):
        self._origin = math.to_float(math.as_tensor(origin))
        self._size = math.to_float(math.as_tensor(size))
        self._upper = self.origin + self.size

    @property
    def origin(self):
        return self._origin

    @property
    def size(self):
        return self._size

    @property
    def upper(self):
        return self._upper

    @property
    def rank(self):
        return len(self.size)

    def global_to_local(self, global_position):
        return (global_position - self.origin) / self.size

    def local_to_global(self, local_position):
        return local_position * self.size + self.origin

    def value_at(self, global_position):
        # local = self.global_to_local(global_position)
        # bool_inside = (local >= 0) & (local <= 1)
        bool_inside = (global_position >= self.origin) & (global_position <= (self.upper))
        bool_inside = math.all(bool_inside, axis=-1, keepdims=True)
        return math.to_float(bool_inside)

    def contains(self, other):
        if isinstance(other, Box):
            return np.all(other.origin >= self.origin) and np.all(other.upper <= self.upper)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return '%s at (%s)' % ('x'.join([str(x) for x in self.size]), ','.join([str(x) for x in self.origin]))


class BoxGenerator(object):

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = [item]
        origin = []
        size = []
        for dim in item:
            if isinstance(dim, (int, float)):
                origin.append(dim)
                size.append(1)
            elif isinstance(dim, slice):
                assert dim.step is None or dim.step == 1, "Box: step must be 1 but is %s" % dim.step
                origin.append(dim.start)
                size.append(dim.stop - dim.start)
        return Box(origin, size)


box = BoxGenerator()


class Sphere(Geometry):
    __struct__ = struct.Def((), ('_center', '_radius'))

    def __init__(self, center, radius):
        self._center = math.as_tensor(center)
        self._radius = math.as_tensor(radius)

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    def value_at(self, location):
        bool_inside = math.expand_dims(math.sum((location - self._center)**2, axis=-1) <= self._radius ** 2, -1)
        return math.to_float(bool_inside)

    @property
    def rank(self):
        return len(self._center)


class _Union(Geometry):
    __struct__ = Geometry.__struct__.extend([], ['_geometries'])

    def __init__(self, geometries):
        self._geometries = geometries

    def __validate_geometries__(self):
        assert len(self._geometries) > 0
        rank = self._geometries[0].rank
        for g in self._geometries[1:]:
            assert g.rank == rank or g.rank is None or rank is None
        self._geometries = tuple(self._geometries)

    def value_at(self, points, collapse_dimensions=True):
        if len(self._geometries) == 1:
            result = self._geometries[0].value_at(points)
        else:
            result = math.max([geometry.value_at(points) for geometry in self._geometries], axis=0)
        return result

    @property
    def rank(self):
        if len(self._geometries) == 0: return None
        else:
            return self._geometries[0].rank

    @property
    def geometries(self):
        return self._geometries


def union(geometries):
    if len(geometries) == 0:
        return NO_GEOMETRY
    else:
        return _Union(geometries)


class _NoGeometry(Geometry):

    def rank(self): return None

    def value_at(self, location): return 0


NO_GEOMETRY = _NoGeometry()
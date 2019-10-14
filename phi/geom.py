from phi import math, struct
import numpy as np


class Geometry(struct.Struct):

    def value_at(self, location):
        raise NotImplementedError(self.__class__)

    def at(self, grid):
        return self.value_at(grid.center_points())


class Box(Geometry):
    __struct__ = struct.Def((), ('origin', 'size'))

    def __init__(self, origin, size):
        self._origin = math.as_tensor(origin)
        self._size = math.as_tensor(size)
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
    def spatial_rank(self):
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


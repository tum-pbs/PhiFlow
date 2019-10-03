from .field import *
from phi import math


def _res(tensor, dim):
    res = list(math.staticshape(tensor)[1:-1])
    res[dim] -= 1
    return tuple(res)


class StaggeredGrid(Field):

    def __init__(self, domain, components, flags=()):
        Field.__init__(self, domain, flags)
        self._components = tuple(components)
        self._resolution = _res(components[0], 0)
        for i, c in enumerate(components):
            assert _res(c, i) == self._resolution

    @property
    def cell_resolution(self):
        return self._resolution

    def resample(self, location):
        pass

    def component_count(self):
        return len(self._components)

    def unstack(self):
        return self._components

    def sample_points(self):
        raise StaggeredSamplePoints(self)

    def compatible(self, other_field):
        if isinstance(other_field, StaggeredGrid):
            return self.bounds == other_field.bounds and self.cell_resolution == other_field.cell_resolution
        else:
            return False
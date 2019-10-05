from .field import *
from .grid import *
from phi import math


_SUBSCRIPTS = ['x', 'y', 'z', 'w']


def _subname(name, i):
    if i < 4:
        return '%s.%s' % (name, _SUBSCRIPTS[i])
    else:
        return '%s.%d' % (name, i)


def _res(tensor, dim):
    res = list(math.staticshape(tensor)[1:-1])
    res[dim] -= 1
    return tuple(res)


def _create_fields(name, staggered_box, data):
    components = []
    for i, component in enumerate(data):
        box = None  # TODO
        raise NotImplementedError()
        components.append(CenteredGrid(_subname(name, i), box, component))


def unstack_staggered_tensor(tensor):
    tensors = math.unstack(tensor, -1)
    for i, dim in enumerate(math.spatial_dimensions(tensor)):
        slices = [slice(None, -1) if d != dim else slice(None) for d in math.spatial_dimensions(tensor)]
        tensors[i] = math.expand_dims(tensors[i][[slice(None)]+slices], -1)
    return tensors


class StaggeredGrid(Field):

    def __init__(self, name, box, data, flags=(), batch_size=None):
        components = _create_fields(name, box, data)
        components = [c.copied_with(bounds=_bounds(box, i)) for i, c in enumerate(components)]
        Field.__init__(self, name, box, tuple(components), flags=flags, batch_size=batch_size)
        self._resolution = _res(components[0], 0)
        for i, c in enumerate(components):
            assert _res(c, i) == self._resolution

    @property
    def cell_resolution(self):
        return self._resolution

    def sample_at(self, points):
        return math.concat([component.sample_at(points) for component in self.data], axis=-1)

    def component_count(self):
        return len(self.data)

    def unstack(self):
        return self.data

    @property
    def points(self):
        raise StaggeredSamplePoints(self)

    def compatible(self, other_field):
        if isinstance(other_field, StaggeredGrid):
            return self.bounds == other_field.bounds and self.cell_resolution == other_field.cell_resolution
        else:
            return False
from phi.math.geom import Box
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


def _create_components(name, staggered_box, data, staggered_flags, batch_size):
    """
    Sets up the bounds of the component fields.
    :return: tuple of Field
    """
    result = []
    if not isinstance(data, (list, tuple)):
        data = unstack_staggered_tensor(data)
    components = []
    for c in data:
        if isinstance(c, CenteredGrid):
            components.append(c.data)
        else:
            components.append(c)

    for i, component in enumerate(components):
        resolution_i = math.staticshape(component)[i+1] - 1
        unit = np.array([1 if i==d else 0 for d in range(len(components))])
        unit = unit * staggered_box.size / resolution_i
        box = Box(staggered_box.origin - unit/2, size=staggered_box.size + unit)
        flags = propagate_flags_children(staggered_flags, math.spatial_rank(component), 1)
        result.append(CenteredGrid(_subname(name, i), box, component, flags=flags, batch_size=batch_size))
    return tuple(result)


def unstack_staggered_tensor(tensor):
    tensors = math.unstack(tensor, -1)
    for i, dim in enumerate(math.spatial_dimensions(tensor)):
        slices = [slice(None, -1) if d != dim else slice(None) for d in math.spatial_dimensions(tensor)]
        tensors[i] = math.expand_dims(tensors[i][[slice(None)]+slices], -1)
    return tensors


def stack_staggered_components(tensors):
    for i, tensor in enumerate(tensors):
        paddings = [[0, 1] if d != i else [0, 0] for d in range(len(tensors))]
        tensors[i] = math.pad(tensor, [[0, 0]] + paddings + [[0, 0]])
    return math.concat(tensors, -1)


class StaggeredGrid(Field):

    def __init__(self, name, box, data, flags=(), batch_size=None):
        components = _create_components(name, box, data, flags, batch_size)
        Field.__init__(self, name, box, components, flags=flags, batch_size=batch_size)
        self._resolution = _res(components[0].data, 0)
        for i, c in enumerate(components):
            assert _res(c.data, i) == self._resolution

    @property
    def rank(self):
        return self.component_count

    @property
    def cell_resolution(self):
        return self._resolution

    def sample_at(self, points):
        return math.concat([component.sample_at(points) for component in self.data], axis=-1)

    @property
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

    def staggered_tensor(self):
        tensors = [c.data for c in self.data]
        return stack_staggered_components(tensors)
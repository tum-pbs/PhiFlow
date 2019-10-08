from phi.geom import Box
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
        return math.as_tensor(self._resolution)

    @property
    def dx(self):
        return self.bounds.size / self.cell_resolution

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
            return self.bounds == other_field.bounds and np.all(self.cell_resolution == other_field.cell_resolution)
        else:
            return False

    def staggered_tensor(self):
        tensors = [c.data for c in self.data]
        return stack_staggered_components(tensors)

    def divergence(self):
        components = []
        for dim, field in enumerate(self.data):
            grad = math.axis_gradient(field.data, dim) / self.dx[dim]
            components.append(grad)
        data = math.add(components)
        return CenteredGrid(u'∇·%s' % self.name, self.bounds, data, batch_size=self._batch_size)

    @staticmethod
    def gradient(scalar_grid, padding='symmetric'):
        data = scalar_grid.data
        if data.shape[-1] != 1: raise ValueError('input must be a scalar field')
        components = []
        for dim in math.spatial_dimensions(data):
            upper = math.pad(data, [[0,1] if d == dim else [0,0] for d in math.all_dimensions(data)], padding)
            lower = math.pad(data, [[1,0] if d == dim else [0,0] for d in math.all_dimensions(data)], padding)
            components.append((upper - lower) / scalar_grid.dx[dim-1])
        return StaggeredGrid(u'∇%s' % scalar_grid.name, scalar_grid.bounds, components, batch_size=scalar_grid._batch_size)

    @staticmethod
    def from_scalar(scalar_field, axis_forces, padding_mode='constant'):
        assert scalar_field.shape[-1] == 1, 'channel must be scalar but has %d components' % scalar_field.shape[-1]
        rank = spatial_rank(scalar_field)
        dims = range(rank)
        df_dq = []
        for dimension in dims:
            padded_field = math.pad(scalar_field,
                                    [[0, 0]] + [[1, 1] if i == dimension else [0, 1] for i in dims] + [[0, 0]],
                                    padding_mode)
            upper_slices = [(slice(1, None) if i == dimension else slice(None)) for i in dims]
            lower_slices = [(slice(-1) if i == dimension else slice(None)) for i in dims]
            neighbour_sum = padded_field[(slice(None),) + tuple(upper_slices) + (slice(None),)] + \
                            padded_field[(slice(None),) + tuple(lower_slices) + (slice(None),)]
            df_dq.append(neighbour_sum * 0.5 / rank)
        df_dq = math.concat(df_dq, axis=-1)
        return StaggeredGrid(df_dq * axis_forces)
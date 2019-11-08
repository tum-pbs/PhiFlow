# coding=utf-8
from numbers import Number
import numpy as np

from phi import math, struct
from phi.geom import AABox
from .field import Field, propagate_flags_children, IncompatibleFieldTypes, broadcast_at, StaggeredSamplePoints, \
    propagate_flags_resample, propagate_flags_operation
from .grid import CenteredGrid


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


def complete_staggered_properties(components, staggeredgrid):
    data = []
    for i, component in enumerate(components):
        name = component.name if component.name is not None else _subname(staggeredgrid.name, i)
        box = component.box
        if box is None:
            resolution_i = staggeredgrid.resolution[i]
            unit = np.array([1 if i == d else 0 for d in range(len(components))])
            unit = unit * staggeredgrid.box.size / resolution_i
            box = AABox(staggeredgrid.box.lower - unit / 2, staggeredgrid.box.upper + unit / 2)
        flags = component.flags
        if flags is None:
            flags = propagate_flags_children(staggeredgrid.flags, math.spatial_rank(component), 1)
        batch_size = component._batch_size if component._batch_size is not None else staggeredgrid._batch_size
        data.append(CenteredGrid(name, box, component.data, flags=flags, batch_size=batch_size))
    return data


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

    def __init__(self, name, box, data, resolution, flags=(), **kwargs):
        Field.__init__(**struct.kwargs(locals()))

    @staticmethod
    def from_tensors(name, box, tensors, flags=(), batch_size=None):
        resolution = _res(tensors[0], 0)
        for i, tensor in enumerate(tensors):
            assert _res(tensor, i) == resolution
        if box is None:
            box = AABox(0, resolution)
        with struct.anytype():
            staggeredgrid = StaggeredGrid(name, box, None, resolution, flags=flags, batch_size=batch_size)
            components = [CenteredGrid(None, None, tensor, None) for tensor in tensors]
        components = complete_staggered_properties(components, staggeredgrid)
        return staggeredgrid.copied_with(data=components, flags=flags)

    @struct.attr()
    def data(self, data):
        assert data is not None
        assert len(self.data) == len(self.resolution) == self.box.rank
        for field in data:
            assert isinstance(field, CenteredGrid), field
            assert field.component_count == 1
            assert field.rank == self.rank
        return data
        # Field.__validate_data__(self)

    @property
    def rank(self):
        return len(self.resolution)

    @struct.prop()
    def resolution(self, resolution):
        return math.as_tensor(resolution)

    @struct.prop()
    def box(self, box):
        assert isinstance(box, AABox)
        return box

    @property
    def dx(self):
        return self.box.size / self.resolution

    def sample_at(self, points, collapse_dimensions=True):
        return math.concat([component.sample_at(points) for component in self.data], axis=-1)

    def at(self, other_field, collapse_dimensions=True, force_optimization=False, return_self_if_compatible=False):
        if isinstance(other_field, StaggeredGrid) and other_field.box == self.box:
            return self
        try:
            points = other_field.points
            resampled = [centeredgrid.at(other_field) for centeredgrid in self.data]
            data = math.concat([field.data for field in resampled], -1)
            return other_field.copied_with(data=data, flags=propagate_flags_resample(self, other_field.flags, other_field.rank))
        except IncompatibleFieldTypes:
            return broadcast_at(self, other_field)
        except StaggeredSamplePoints:
            return broadcast_at(self, other_field)

    @property
    def component_count(self):
        return len(self.data)

    def unstack(self):
        return self.data

    @property
    def points(self):
        raise StaggeredSamplePoints(self)

    @property
    def center_points(self):
        return CenteredGrid.getpoints(self.box, self.resolution)

    def __repr__(self):
        return 'StaggeredGrid[%s, size=%s]' % ('x'.join([str(r) for r in self.resolution]), self.box.size)

    def compatible(self, other_field):
        if not other_field.has_points:
            return True
        if isinstance(other_field, StaggeredGrid):
            return self.box == other_field.box and np.all(self.resolution == other_field.resolution)
        else:
            return False

    def __dataop__(self, other, linear_if_scalar, data_operator):
        if isinstance(other, StaggeredGrid):
            assert self.compatible(other), 'Fields are not compatible: %s and %s' % (self, other)
            data = [data_operator(c1, c2) for c1, c2 in zip(self.data, other.data)]
            flags = propagate_flags_operation(self.flags+other.flags, False, self.rank, self.component_count)
        else:
            flags = propagate_flags_operation(self.flags, linear_if_scalar, self.rank, self.component_count)
            data = [data_operator(c1, other) for c1 in self.data]
        return self.copied_with(data=np.array(data, dtype=np.object), flags=flags)

    def staggered_tensor(self):
        tensors = [c.data for c in self.data]
        return stack_staggered_components(tensors)

    def divergence(self, physical_units=True):
        components = []
        for dim, field in enumerate(self.data):
            grad = math.axis_gradient(field.data, dim)
            if physical_units:
                grad /= self.dx[dim]
            components.append(grad)
        data = math.add(components)
        return CenteredGrid(u'∇·%s' % self.name, self.box, data, batch_size=self._batch_size)

    @property
    def dtype(self):
        return self.data[0].dtype

    @staticmethod
    def gradient(scalar_field, padding_mode='symmetric'):
        assert isinstance(scalar_field, CenteredGrid)
        data = scalar_field.data
        if data.shape[-1] != 1:
            raise ValueError('input must be a scalar field')
        with struct.anytype():
            staggeredgrid = StaggeredGrid(u'∇%s' % scalar_field.name, scalar_field.box, None, scalar_field.resolution, batch_size=scalar_field._batch_size)
            tensors = []
            for dim in math.spatial_dimensions(data):
                upper = math.pad(data, [[0,1] if d == dim else [0,0] for d in math.all_dimensions(data)], padding_mode)
                lower = math.pad(data, [[1,0] if d == dim else [0,0] for d in math.all_dimensions(data)], padding_mode)
                tensors.append((upper - lower) / scalar_field.dx[dim - 1])
            components = [CenteredGrid(None, None, t, flags=None) for t in tensors]
            data = complete_staggered_properties(components, staggeredgrid)
        return staggeredgrid.copied_with(data=data)

    @staticmethod
    def from_scalar(scalar_field, axis_forces, padding_mode='constant', name=None):
        assert isinstance(scalar_field, CenteredGrid)
        assert scalar_field.component_count == 1, 'channel must be scalar but has %d components' % scalar_field.component_count
        with struct.anytype():
            staggeredgrid = StaggeredGrid(name, scalar_field.box, None, scalar_field.resolution, batch_size=scalar_field._batch_size)
        tensors = []
        for i in range(scalar_field.rank):
            force = axis_forces[i] if isinstance(axis_forces, (list, tuple)) else axis_forces[...,i]
            if isinstance(force, Number) and force == 0:
                dims = list(math.staticshape(scalar_field.data))
                dims[i+1] += 1
                tensors.append(math.zeros(dims, math.dtype(scalar_field.data)))
            else:
                upper = math.pad(scalar_field.data, [[0,1] if d == i+1 else [0,0] for d in math.all_dimensions(scalar_field.data)], padding_mode)
                lower = math.pad(scalar_field.data, [[1,0] if d == i+1 else [0,0] for d in math.all_dimensions(scalar_field.data)], padding_mode)
                tensors.append((upper + lower) / 2 * force)
        with struct.anytype():
            components = [CenteredGrid(None, None, t, flags=None) for t in tensors]
            data = complete_staggered_properties(components, staggeredgrid)
        return staggeredgrid.copied_with(data=data)

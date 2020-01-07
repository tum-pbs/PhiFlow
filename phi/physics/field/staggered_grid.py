# coding=utf-8
from numbers import Number
import numpy as np
import six

from phi import math, struct
from phi.geom import AABox
from phi.geom.geometry import assert_same_rank
from phi.struct.tensorop import collapse
from .field import Field, propagate_flags_children, IncompatibleFieldTypes, broadcast_at, StaggeredSamplePoints, \
    propagate_flags_resample, propagate_flags_operation
from .grid import CenteredGrid


_SUBSCRIPTS = ['x', 'y', 'z', 'w']


def _subname(name, i):
    if i < 4:
        return '%s.%s' % (name, _SUBSCRIPTS[i])
    else:
        return '%s.%d' % (name, i)


def _res(tensor, axis):
    if isinstance(tensor, CenteredGrid):
        tensor = tensor.data
    res = list(math.staticshape(tensor)[1:-1])
    res[axis] -= 1
    return tuple(res)


def unstack_staggered_tensor(tensor):
    tensors = math.unstack(tensor, -1)
    for i, dim in enumerate(math.spatial_dimensions(tensor)):
        slices = [slice(None, -1) if d != dim else slice(None) for d in math.spatial_dimensions(tensor)]
        tensors[i] = math.expand_dims(tensors[i][tuple([slice(None)]+slices)], -1)
    return tensors


def stack_staggered_components(tensors):
    for i, tensor in enumerate(tensors):
        paddings = [[0, 1] if d != i else [0, 0] for d in range(len(tensors))]
        tensors[i] = math.pad(tensor, [[0, 0]] + paddings + [[0, 0]])
    return math.concat(tensors, -1)


def staggered_component_box(resolution, axis, box_like=None):
    staggered_box = AABox(0, resolution) if box_like is None else AABox.to_box(box_like, resolution_hint=resolution)
    unit = np.array([(staggered_box.size[axis] / resolution[axis]) if d == axis else 0 for d in range(len(resolution))])
    box = AABox(staggered_box.lower - unit / 2, staggered_box.upper + unit / 2)
    return box


@struct.definition()
class StaggeredGrid(Field):

    def __init__(self, data, box=None, name=None, **kwargs):
        Field.__init__(self, **struct.kwargs(locals()))

    @struct.variable(dependencies=[Field.name, Field.flags])
    def data(self, data):
        assert data is not None
        if math.is_tensor(data) is True:
            components = unstack_staggered_tensor(data)
        else:
            components = data
        data = []
        for cmp_idx, grid in enumerate(components):
            data.append(self._component_grid(grid, cmp_idx))
        return tuple(data)

    def _component_grid(self, grid, axis):
        resolution = list(grid.resolution if isinstance(grid, CenteredGrid) else math.staticshape(grid)[1:-1])
        resolution[axis] -= 1
        box = staggered_component_box(resolution, axis, self.box)
        if isinstance(grid, CenteredGrid):
            assert grid.component_count == 1
            assert grid.rank == self.rank
            assert grid.box == box
            assert grid.extrapolation == self.extrapolation
        else:
            grid = CenteredGrid(data=grid, box=box, extrapolation=self.extrapolation, name=_subname(self.name, axis),
                                batch_size=self._batch_size, flags=propagate_flags_children(self.flags, box.rank, 1))
        return grid

    @property
    def rank(self):
        return len(self.resolution)

    @struct.derived()
    def resolution(self):
        return _res(self.data[0], 0)

    @struct.constant(dependencies=Field.data)
    def box(self, box):
        box = AABox.to_box(box, resolution_hint=self.resolution)
        assert_same_rank(len(self.data), self.box, 'StaggeredGrid.data does not match box.')
        return box

    @property
    def dx(self):
        return self.box.size / self.resolution

    @struct.constant(default='boundary')
    def extrapolation(self, extrapolation):
        if extrapolation is None:
            return 'boundary'
        assert extrapolation in ('periodic', 'constant', 'boundary') or isinstance(extrapolation, (tuple, list)), extrapolation
        return collapse(extrapolation)

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

    def at_centers(self):
        return self.at(self.center_points)

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
        data = math.sum(components, 0)
        return CenteredGrid(data, self.box, name='div(%s)' % self.name, batch_size=self._batch_size)

    @property
    def dtype(self):
        return self.data[0].dtype

    @staticmethod
    def gradient(scalar_field, padding_mode='replicate'):
        assert isinstance(scalar_field, CenteredGrid)
        data = scalar_field.data
        if data.shape[-1] != 1:
            raise ValueError('input must be a scalar field')
        tensors = []
        for dim in math.spatial_dimensions(data):
            upper = math.pad(data, [[0,1] if d == dim else [0,0] for d in math.all_dimensions(data)], padding_mode)
            lower = math.pad(data, [[1,0] if d == dim else [0,0] for d in math.all_dimensions(data)], padding_mode)
            tensors.append((upper - lower) / scalar_field.dx[dim - 1])
        return StaggeredGrid(tensors, scalar_field.box, name='grad(%s)' % scalar_field.name,
                             batch_size=scalar_field._batch_size)

    @staticmethod
    def from_scalar(scalar_field, axis_forces, name=None):
        assert isinstance(scalar_field, CenteredGrid)
        assert scalar_field.component_count == 1, 'channel must be scalar but has %d components' % scalar_field.component_count
        tensors = []
        for axis in range(scalar_field.rank):
            force = axis_forces[axis] if isinstance(axis_forces, (list, tuple)) else axis_forces[...,axis]
            if isinstance(force, Number) and force == 0:
                dims = list(math.staticshape(scalar_field.data))
                dims[axis+1] += 1
                tensors.append(math.zeros(dims, math.dtype(scalar_field.data)))
            else:
                upper = scalar_field.axis_padded(axis, 0, 1).data
                lower = scalar_field.axis_padded(axis, 1, 0).data
                tensors.append(math.mul((upper + lower) / 2, force))
        return StaggeredGrid(tensors, scalar_field.box, name=name, batch_size=scalar_field._batch_size)

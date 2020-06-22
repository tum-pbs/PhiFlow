import numpy as np
import six

from phi import math, struct
from phi.backend.backend_helper import general_grid_sample_nd
from phi.geom import AABox, box
from phi.geom.geometry import assert_same_rank
from phi.math.helper import map_for_axes
from phi.physics.domain import Domain
from phi.physics.material import Material
from phi.struct.functions import mappable
from phi.struct.tensorop import collapse

from .field import Field, propagate_flags_children
from .flag import SAMPLE_POINTS


def _crop_for_interpolation(data, offset_float, window_resolution):
    offset = math.to_int(offset_float)
    slices = [slice(o, o + res + 1) for o, res in zip(offset, window_resolution)]
    data = data[tuple([slice(None)] + slices + [slice(None)])]
    return data


@struct.definition()
class CenteredGrid(Field):

    def __init__(self, data, box=None, extrapolation='boundary', name=None, **kwargs):
        """Create new CenteredGrid from array like data

        :param data: numerical values to be set as values of CenteredGrid (immutable)
        :type data: array-like
        :param box: numerical values describing the surrounding area of the CenteredGrid, defaults to None
        :type box: domain.box, optional
        :param extrapolation: set conditions for boundaries, defaults to 'boundary'
        :type extrapolation: str, optional
        :param name: give CenteredGrid a custom name (immutable), defaults to None
        :type name: string, optional
        """
        Field.__init__(self, **struct.kwargs(locals()))

    @staticmethod
    def sample(value, domain, batch_size=None, name=None):
        assert isinstance(domain, Domain)
        if isinstance(value, Field):
            assert_same_rank(value.rank, domain.rank, 'rank of value (%s) does not match domain (%s)' % (value.rank, domain.rank))
            if isinstance(value, CenteredGrid) and value.box == domain.box and np.all(value.resolution == domain.resolution):
                data = value.data
            else:
                point_field = CenteredGrid.getpoints(domain.box, domain.resolution)
                point_field._batch_size = batch_size
                data = value.at(point_field).data
        else:  # value is constant
            if callable(value):
                x = CenteredGrid.getpoints(domain.box, domain.resolution).copied_with(extrapolation=Material.extrapolation_mode(domain.boundaries), name=name)
                value = value(x)
                return value
            components = math.staticshape(value)[-1] if math.ndims(value) > 0 else 1
            data = math.add(math.zeros((batch_size,) + tuple(domain.resolution) + (components,)), value)
        return CenteredGrid(data, box=domain.box, extrapolation=Material.extrapolation_mode(domain.boundaries), name=name)

    @struct.variable()
    def data(self, data):
        if data is None:
            return None
        if isinstance(data, (tuple, list)):
            data = np.array(data)  # numbers or objects
        if self.content_type in (struct.shape, struct.staticshape):
            assert math.ndims(data) == 1
        else:
            if math.ndims(data) < 2:
                data = math.expand_dims(data, 0, number=2 - math.ndims(data))
        return data
    data.override(struct.staticshape, lambda self, data: (self._batch_size,) + math.staticshape(data)[1:])

    @property
    def resolution(self):
        if self.content_type in (struct.VALID, struct.INVALID):
            return math.as_tensor(math.staticshape(self.data)[1:-1])
        elif self.content_type in (struct.shape, struct.staticshape):
            return self.data[1:-1]
        else:
            raise AssertionError('Cannot compute resolution of invalid CenteredGrid (content type = %s)' % self.content_type)

    @struct.constant(dependencies=Field.data)
    def box(self, box):
        return AABox.to_box(box, resolution_hint=self.resolution)

    @struct.derived()
    def dx(self):
        return self.box.size / self.resolution

    @property
    def rank(self):
        return len(self.resolution)

    @struct.constant(default='boundary')
    def extrapolation(self, extrapolation):
        if extrapolation is None:
            return 'boundary'
        assert extrapolation in ('periodic', 'constant', 'boundary') or isinstance(extrapolation, (tuple, list)), extrapolation
        return collapse(extrapolation)

    @struct.constant(default=0.0)
    def extrapolation_value(self, value):
        return collapse(value)

    @struct.constant(default='linear')
    def interpolation(self, interpolation):
        assert interpolation == 'linear'
        return interpolation

    def sample_at(self, points):
        local_points = self.box.global_to_local(points)
        local_points = math.mul(local_points, math.to_float(self.resolution)) - 0.5
        resampled = math.resample(self.data, local_points, boundary=_pad_mode(self.extrapolation), interpolation=self.interpolation, constant_values=_pad_value(self.extrapolation_value))
        return resampled

    def general_sample_at(self, points, reduce):
        local_points = self.box.global_to_local(points)
        local_points = math.mul(local_points, math.to_float(self.resolution)) - 0.5
        result = general_grid_sample_nd(self.data, local_points, boundary=_pad_mode(self.extrapolation), constant_values=_pad_value(self.extrapolation_value), math=math.choose_backend([self.data, points]), reduce=reduce)
        return result

    def at(self, other_field):
        if self.compatible(other_field):
            return self
        if isinstance(other_field, CenteredGrid) and np.allclose(self.dx, other_field.dx):
            paddings = _required_paddings_transposed(self.box, self.dx, other_field.box)
            if math.sum(paddings) == 0:
                origin_in_local = self.box.global_to_local(other_field.box.lower) * self.resolution
                data = _crop_for_interpolation(self.data, origin_in_local, other_field.resolution)
                dimensions = self.resolution != other_field.resolution
                dimensions = [d for d in math.spatial_dimensions(data) if dimensions[d - 1]]
                data = math.interpolate_linear(data, origin_in_local % 1.0, dimensions)
                return CenteredGrid(data, other_field.box, name=self.name, batch_size=self._batch_size)
            elif math.sum(paddings) < 16:
                padded = self.padded(np.transpose(paddings).tolist())
                return padded.at(other_field)
        return Field.at(self, other_field)

    @property
    def component_count(self):
        if self.content_type in (struct.shape, struct.staticshape):
            return self.data[-1]
        else:
            return self.data.shape[-1]

    def unstack(self):
        flags = propagate_flags_children(self.flags, self.rank, 1)
        components = math.unstack(self.data, axis=-1, keepdims=True)
        return [CenteredGrid(component, box=self.box, flags=flags, batch_size=self._batch_size) for i, component in enumerate(components)]

    @property
    def points(self):
        if self.is_valid and SAMPLE_POINTS in self.flags:
            return self
        return CenteredGrid.getpoints(self.box, self.resolution)

    @property
    def elements(self):
        return box(center=self.points.data, size=self.dx)

    def compatible(self, other_field):
        if isinstance(other_field, (Domain, CenteredGrid)):
            if self.box != other_field.box:
                return False
            if self.rank != other_field.rank:
                return False
            for r1, r2 in zip(self.resolution, other_field.resolution):
                if r1 != r2 and r2 != 1 and r1 != 1:
                    return False
            return True
        if not other_field.has_points:
            return other_field.compatible(self)
        else:
            return False

    def __can_validate__(self):
        return self.content_type in (struct.INVALID, struct.shape, struct.staticshape)

    def __repr__(self):
        if self.is_valid:
            return 'Grid[%s(%d), size=%s, %s]' % ('x'.join([str(r) for r in self.resolution]), self.component_count, self.box.size, self.dtype.data)
        else:
            return struct.Struct.__repr__(self)

    def padded(self, widths):
        if isinstance(widths, int):
            widths = [[widths, widths]] * self.rank
        data = math.pad(self.data, [[0, 0]] + widths + [[0, 0]], _pad_mode(self.extrapolation), constant_values=_pad_value(self.extrapolation_value))
        w_lower, w_upper = np.transpose(widths)
        box = AABox(self.box.lower - w_lower * self.dx, self.box.upper + w_upper * self.dx)
        return self.copied_with(data=data, box=box)

    def axis_padded(self, axis, lower, upper):
        widths = [[lower, upper] if ax == axis else [0,0] for ax in range(self.rank)]
        return self.padded(widths)

    @staticmethod
    def getpoints(box, resolution):
        idx_zyx = np.meshgrid(*[np.linspace(0.5 / dim, 1 - 0.5 / dim, dim) for dim in resolution], indexing="ij")
        local_coords = math.to_float(math.expand_dims(math.stack(idx_zyx, axis=-1), 0))
        points = box.local_to_global(local_coords)
        return CenteredGrid(points, box, name='grid_centers(%s, %s)' % (box, resolution), flags=[SAMPLE_POINTS])

    def laplace(self, physical_units=True, axes=None):
        if not physical_units:
            data = math.laplace(self.data, padding=_pad_mode(self.extrapolation), axes=axes)
        else:
            if not self.has_cubic_cells:
                raise NotImplementedError('Only cubic cells supported.')
            laplace = math.laplace(self.data, padding=_pad_mode(self.extrapolation), axes=axes)
            data = laplace / self.dx[0] ** 2
        extrapolation = map_for_axes(_gradient_extrapolation, self.extrapolation, axes, self.rank)
        return self.copied_with(data=data, extrapolation=extrapolation, flags=())

    def gradient(self, physical_units=True, difference='central'):
        if not physical_units or self.has_cubic_cells:
            data = math.gradient(self.data, dx=np.mean(self.dx), difference=difference, padding=_pad_mode(self.extrapolation))
            return self.copied_with(data=data, extrapolation=_gradient_extrapolation(self.extrapolation), flags=())
        else:
            raise NotImplementedError('Only cubic cells supported.')

    @property
    def has_cubic_cells(self):
        return np.allclose(self.dx, np.mean(self.dx))

    def normalized(self, total, epsilon=1e-5):
        if isinstance(total, CenteredGrid):
            total = total.data
        normalize_data = math.normalize_to(self.data, total, epsilon)
        return self.with_data(normalize_data)

    @struct.derived()
    def frequencies(self):
        return self.with_data(math.fftfreq(self.resolution, mode='vector') / self.dx)

    @struct.derived()
    def squared_frequencies(self):
        return self.with_data(math.sum(self.frequencies.data ** 2, axis=-1, keepdims=True))

    def fft(self):
        return self.with_data(math.fft(self.data))

    @struct.derived()
    def abs(self):
        return self.with_data(math.abs(self.data))


def _required_paddings_transposed(box, dx, target, threshold=1e-5):
    lower = math.to_int(math.ceil(math.maximum(0, box.lower - target.lower) / dx - threshold))
    upper = math.to_int(math.ceil(math.maximum(0, target.upper - box.upper) / dx - threshold))
    return [lower, upper]


def _pad_mode(extrapolation):
    """ Inserts 'constant' padding for batch dimension and channel dimension. """
    if isinstance(extrapolation, six.string_types):
        return _pad_mode_str(extrapolation)
    else:
        return _pad_mode_str(['constant'] + list(extrapolation) + ['constant'])


def _pad_value(value):
    if math.is_tensor(value):
        return value
    else:
        return [0] + list(value) + [0]


@mappable()
def _pad_mode_str(extrapolation):
    """
Converts an extrapolation string (or struct of strings) to a string that can be passed to math functions like math.pad or math.resample.
    :param extrapolation: field extrapolation
    :return: padding mode, same type as extrapolation
    """
    return {'periodic': 'circular',
            'boundary': 'replicate',
            'constant': 'constant'}[extrapolation]


@mappable()
def _gradient_extrapolation(extrapolation):
    """
Given the extrapolation of a field, returns the extrapolation mode of the corresponding gradient field.
    :param extrapolation: string or struct of strings
    :return: same type as extrapolation
    """
    return {'periodic': 'periodic',
            'boundary': 'constant',
            'constant': 'constant'}[extrapolation]

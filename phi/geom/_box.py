import numpy as np

from phi import struct, math
from ._geom_util import assert_same_rank
from ._geom import Geometry
from ._transform import rotate


@struct.definition()
class AbstractBox(Geometry):

    @property
    def size(self):
        raise NotImplementedError(self)

    @property
    def half_size(self):
        raise NotImplementedError(self)

    @property
    def lower(self):
        raise NotImplementedError(self)

    @property
    def upper(self):
        raise NotImplementedError(self)

    def bounding_radius(self):
        return math.max(self.size, axis=-1, keepdims=True) * 1.414214

    def bounding_half_extent(self):
        return self.size * 0.5

    def global_to_local(self, global_position):
        size, lower = math.batch_align([self.size, self.lower], 1, global_position)
        return (global_position - lower) / size

    def local_to_global(self, local_position):
        size, lower = math.batch_align([self.size, self.lower], 1, local_position)
        return local_position * size + lower

    def lies_inside(self, location):
        lower, upper = math.batch_align([self.lower, self.upper], 1, location)
        bool_inside = (location >= lower) & (location <= upper)
        return math.all(bool_inside, axis=-1, keepdims=True)

    def approximate_signed_distance(self, location):
        """
Computes the signed L-infinity norm (manhattan distance) from the location to the nearest side of the box.
For an outside location `l` with the closest surface point `s`, the distance is `max(abs(l - s))`.
For inside locations it is `-max(abs(l - s))`.
        :param location: float tensor of shape (batch_size, ..., rank)
        :return: float tensor of shape (*location.shape[:-1], 1).
        """
        lower, upper = math.batch_align([self.lower, self.upper], 1, location)
        center = 0.5 * (lower + upper)
        extent = upper - lower
        distance = math.abs(location - center) - extent * 0.5
        return math.max(distance, axis=-1, keepdims=True)

    def get_lower(self, axis):
        return self._get(self.lower, axis)

    def get_upper(self, axis):
        return self._get(self.upper, axis)

    @staticmethod
    def _get(vector, axis):
        if vector.shape[-1] == 1:
            return vector[...,0]
        else:
            return vector[...,axis]

    def corner_representation(self):
        return AABox(self.lower, self.upper)

    def center_representation(self):
        return Cuboid(self.center, self.half_size)

    def contains(self, other):
        if isinstance(other, AbstractBox):
            return np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper)
        else:
            raise NotImplementedError()

    def rotated(self, angle):
        return rotate(self, angle)


@struct.definition(traits=[math.BATCHED])
class AABox(AbstractBox):
    """
    Axis-aligned box, defined by lower and upper corner.
    AABoxes can be created using the shorthand notation box[slices], (e.g. box[:,0:1] to create an inifinite-height box from x=0 to x=1).
    """

    def __init__(self, lower, upper, **kwargs):
        AbstractBox.__init__(self, **struct.kwargs(locals()))

    @struct.constant(min_rank=1)
    def lower(self, lower):
        return math.to_float(lower)

    @struct.constant(min_rank=1)
    def upper(self, upper):
        return math.to_float(upper)

    @property
    def rank(self):
        if math.ndims(self.size) > 0:
            return math.staticshape(self.size)[-1]
        else:
            return None

    @struct.derived()
    def size(self):
        return self.upper - self.lower

    @struct.derived()
    def center(self):
        return 0.5 * (self.lower + self.upper)

    @struct.derived()
    def half_size(self):
        return self.size * 0.5

    def without_axis(self, axis):
        lower = []
        upper = []
        for ax in range(self.rank):
            if ax != axis:
                lower.append(self.get_lower(ax))
                upper.append(self.get_upper(ax))
        return self.copied_with(lower=lower, upper=upper)

    def shifted(self, delta):
        return self.copied_with(lower=self.lower + delta, upper=self.upper + delta)

    def __repr__(self):
        if self.is_valid:
            return '%s at (%s)' % ('x'.join([str(x) for x in self.size]), ','.join([str(x) for x in self.lower]))
        else:
            return struct.Struct.__repr__(self)

    @staticmethod
    def to_box(value, resolution_hint=None):
        if value is None:
            assert resolution_hint is not None
            result = AABox(0, resolution_hint)
        elif isinstance(value, AABox):
            result = value
        elif isinstance(value, int):
            if resolution_hint is None:
                result = AABox(0, value)
            else:
                size = [value] * (1 if math.ndims(resolution_hint) == 0 else len(resolution_hint))
                result = AABox(0, size)
        elif isinstance(value, (tuple, list)):
            result = AABox(0, box)
        else:
            raise ValueError("Box extent not understood: '%s'" % value)
        if resolution_hint is not None:
            assert_same_rank(len(resolution_hint), result, 'AABox rank does not match resolution.')
        return result


@struct.definition(traits=[math.BATCHED])
class Cuboid(AbstractBox):

    def __init__(self, center, half_size, **kwargs):
        AbstractBox.__init__(self, **struct.kwargs(locals()))

    @struct.constant(min_rank=1)
    def center(self, center):
        return math.to_float(center)

    @struct.constant(min_rank=1)
    def half_size(self, half_size):
        return math.to_float(half_size)

    @property
    def rank(self):
        if math.ndims(self.upper) > 0:
            return math.staticshape(self.upper)[-1]
        else:
            return None

    @struct.derived()
    def size(self):
        return 2 * self.half_size

    @struct.derived()
    def lower(self):
        return self.center - self.half_size

    @struct.derived()
    def upper(self):
        return self.center + self.half_size

    def shifted(self, delta):
        return self.copied_with(center=self.center + delta)


class BoxGenerator(object):

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

    def __call__(self, *args, **kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            lower, upper = args
            return AABox(lower, upper)
        elif len(args) == 1 and 'size' in kwargs:
            center, = args
            size = kwargs['size']
            return Cuboid(center, 0.5 * math.to_float(size))
        elif 'size' in kwargs and 'center' in kwargs:
            center, size = kwargs['center'], kwargs['size']
            return Cuboid(center, 0.5 * math.to_float(size))
        elif len(args) == 1 and len(kwargs) == 0:
            return AABox(0, args[0])
        else:
            raise ValueError('Cannot create box from args=%s, kwargs=%s' % (args, kwargs))


def bounding_box(geometry):
    center = geometry.center
    extent = geometry.bounding_half_extent()
    return AABox(lower=center - extent, upper=center + extent)

import numpy as np

from phi import struct
from phi import math


@struct.definition()
class Geometry(struct.Struct):

    def value_at(self, location):
        """
Samples the geometry at the given locations and returns a binary mask, labelling the points as inside=1, outside=0.
        :param location: tensor of the shape (batch_size, ..., rank)
        :return: float tensor of same shape as location but with shape[-1]=1
        """
        raise NotImplementedError(self.__class__)

    @property
    def rank(self):
        raise NotImplementedError()


@struct.definition()
class AABox(Geometry):

    def __init__(self, lower, upper, **kwargs):
        Geometry.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def lower(self, lower):
        return math.to_float(math.as_tensor(lower))

    @struct.constant()
    def upper(self, upper):
        return math.to_float(math.as_tensor(upper))

    def get_lower(self, axis):
        return self._get(self.lower, axis)

    def get_upper(self, axis):
        return self._get(self.upper, axis)

    @staticmethod
    def _get(vector, axis):
        if math.ndims(vector) == 0:
            return vector
        else:
            return vector[...,axis]

    @struct.derived()
    def size(self):
        return self.upper - self.lower

    @property
    def rank(self):
        if math.ndims(self.size) > 0:
            return self.size.shape[-1]
        else:
            return None

    def global_to_local(self, global_position):
        size, lower = math.batch_align([self.size, self.lower], 1, global_position)
        return (global_position - lower) / size

    def local_to_global(self, local_position):
        size, lower = math.batch_align([self.size, self.lower], 1, local_position)
        return local_position * size + lower

    def value_at(self, global_position):
        lower, upper = math.batch_align([self.lower, self.upper], 1, global_position)
        bool_inside = (global_position >= lower) & (global_position <= upper)
        bool_inside = math.all(bool_inside, axis=-1, keepdims=True)
        return math.to_float(bool_inside)

    def contains(self, other):
        if isinstance(other, AABox):
            return np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper)
        else:
            raise NotImplementedError()

    def without_axis(self, axis):
        lower = []
        upper = []
        for ax in range(self.rank):
            if ax != axis:
                lower.append(self.get_lower(ax))
                upper.append(self.get_upper(ax))
        return self.copied_with(lower=lower, upper=upper)

    def __repr__(self):
        try:
            return '%s at (%s)' % ('x'.join([str(x) for x in self.size]), ','.join([str(x) for x in self.lower]))
        except TypeError:
            return '%s at %s' % (self.size, self.lower)

    @staticmethod
    def to_box(value, resolution_hint=None):
        if value is None:
            result = AABox(0, resolution_hint)
        elif isinstance(value, AABox):
            result = value
        elif isinstance(value, int):
            if resolution_hint is None:
                result = AABox(0, value)
            else:
                size = [value] * (1 if math.ndims(resolution_hint) == 0 else len(resolution_hint))
                result = AABox(0, size)
        else:
            result = AABox(box)
        if resolution_hint is not None:
            assert_same_rank(len(resolution_hint), result, 'AABox rank does not match resolution.')
        return result


class AABoxGenerator(object):

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


box = AABoxGenerator()


@struct.definition()
class Sphere(Geometry):

    def __init__(self, center, radius, **kwargs):
        Geometry.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def radius(self, radius):
        return math.as_tensor(radius)

    @struct.constant()
    def center(self, center):
        return math.as_tensor(center)

    def value_at(self, location):
        center = math.batch_align(self.center, 1, location)
        radius = math.batch_align(self.radius, 0, location)
        distance_squared = math.sum((location - center)**2, axis=-1, keepdims=True)
        bool_inside = distance_squared <= radius**2
        return math.to_float(bool_inside)

    @property
    def rank(self):
        return len(self.center)


@struct.definition()
class _Union(Geometry):

    def __init__(self, geometries, **kwargs):
        Geometry.__init__(self, **struct.kwargs(locals()))

    def value_at(self, points, collapse_dimensions=True):
        if len(self.geometries) == 1:
            result = self.geometries[0].value_at(points)
        else:
            result = math.max([geometry.value_at(points) for geometry in self.geometries], axis=0)
        return result

    @property
    def rank(self):
        if len(self.geometries) == 0:
            return None
        else:
            return self.geometries[0].rank

    @struct.constant()
    def geometries(self, geometries):
        assert len(geometries) > 0
        rank = geometries[0].rank
        for g in geometries[1:]:
            assert g.rank == rank or g.rank is None or rank is None
        return tuple(geometries)


def union(geometries):
    if len(geometries) == 0:
        return NO_GEOMETRY
    else:
        return _Union(geometries)


@struct.definition()
class _NoGeometry(Geometry):

    def rank(self):
        return None

    def value_at(self, location):
        return 0


NO_GEOMETRY = _NoGeometry()


def assert_same_rank(rank1, rank2, error_message):
    rank1_, rank2_ = _rank(rank1), _rank(rank2)
    if rank1_ is not None and rank2_ is not None:
        assert rank1_ == rank2_, 'Ranks do not match: %s and %s. %s' % (rank1_, rank2_, error_message)


def _rank(rank):
    if rank is None:
        return None
    elif isinstance(rank, int):
        pass
    elif isinstance(rank, Geometry):
        rank = rank.rank
    else:
        rank = math.spatial_rank(rank)
    return None if rank == 0 else rank

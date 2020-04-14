import warnings

import numpy as np

from phi import struct
from phi import math


@struct.definition()
class Geometry(struct.Struct):

    def value_at(self, location):
        """
Samples the geometry at the given locations and returns a binary mask, labelling the points as inside=1, outside=0.
        :param location: tensor of shape (batch_size, ..., rank)
        :return: float tensor of shape (*location.shape[:-1], 1).
        """
        warnings.warn("Geometry.value_at() is deprecated. Use lies_inside or approximate_fraction_inside instead.", DeprecationWarning)
        return math.to_float(self.lies_inside(location))

    def lies_inside(self, location):
        """
Tests whether the given location lies inside or outside of the geometry. Locations on the surface count as inside.
        :param location: float tensor of shape (batch_size, ..., rank)
        :return: bool tensor of shape (*location.shape[:-1], 1).
        """
        raise NotImplementedError(self.__class__)

    def approximate_signed_distance(self, location):
        """
Computes the approximate distance from location to the surface of the geometry.
Locations outside return positive values, inside negative values and zero exactly at the boundary.

The exact distance metric used depends on the geometry.
The approximation holds close to the surface and the distance grows to infinity as the location is moved infinitely far from the geometry.
The distance metric is differentiable and its gradients are bounded at every point in space.
        :param location: float tensor of shape (batch_size, ..., rank)
        :return: float tensor of shape (*location.shape[:-1], 1).
        """
        raise NotImplementedError(self.__class__)

    def approximate_fraction_inside(self, location, cell_size):
        """
Computes the approximate overlap between the geometry and small cells.
Cells that lie completely inside the geometry return 1.0, those that lie completely outside return 0.0.
Close to the geometry surface, the fraction filled is differentiable w.r.t. the cell location and size.

No specific cell shape is assumed. Cells may be approximated as spheres or axis-aligned cubes.

Cell sizes should rather be overestimated than underestimated to avoid zero gradients.
        :param location: float tensor of shape (batch_size, ..., rank)
        :param cell_size: length or diameter of each cell. Scalar or tensor of shape compatible with location.
        :return: fraction of cell volume lying inside the geometry. float tensor of shape (*location.shape[:-1], 1).
        """
        radius = 0.707 * cell_size
        distance = self.approximate_signed_distance(location)
        inside_fraction = 0.5 - distance / radius
        inside_fraction = math.clip(inside_fraction, 0, 1)
        return inside_fraction

    @property
    def rank(self):
        raise NotImplementedError(self.__class__)

    def __add__(self, other):
        if isinstance(other, _Union):
            return other.__add__(self)
        return union(self, other)


@struct.definition(traits=[math.BATCHED])
class AABox(Geometry):
    """
    Axis-aligned box, defined by lower and upper corner.
    AABoxes can be created using the shorthand notation box[slices], (e.g. box[:,0:1] to create an inifinite-height box from x=0 to x=1).
    """

    def __init__(self, lower, upper, **kwargs):
        Geometry.__init__(self, **struct.kwargs(locals()))

    @struct.constant(min_rank=1)
    def lower(self, lower):
        return math.to_float(lower)

    @struct.constant(min_rank=1)
    def upper(self, upper):
        return math.to_float(upper)

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
        if self.is_valid:
            return '%s at (%s)' % ('x'.join([str(x) for x in self.size]), ','.join([str(x) for x in self.lower]))
        else:
            return struct.Struct.__repr__(self)

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


box = AABoxGenerator()  # Instantiate an AABox using the syntax box[slices]


@struct.definition(traits=[math.BATCHED])
class Sphere(Geometry):

    def __init__(self, center, radius, **kwargs):
        Geometry.__init__(self, **struct.kwargs(locals()))

    @struct.constant(min_rank=0)
    def radius(self, radius):
        return radius

    @struct.constant(min_rank=1)
    def center(self, center):
        return center

    @property
    def rank(self):
        return len(self.center)

    def lies_inside(self, location):
        center = math.batch_align(self.center, 1, location)
        radius = math.batch_align(self.radius, 0, location)
        distance_squared = math.sum((location - center) ** 2, axis=-1, keepdims=True)
        return distance_squared <= radius ** 2

    def approximate_signed_distance(self, location):
        """
Computes the exact distance from location to the closest point on the sphere.
Very close to the sphere center, the distance takes a constant value.
        :param location: float tensor of shape (batch_size, ..., rank)
        :return: float tensor of shape (*location.shape[:-1], 1).
        """
        center = math.batch_align(self.center, 1, location)
        radius = math.batch_align(self.radius, 0, location)
        distance_squared = math.sum((location - center)**2, axis=-1, keepdims=True)
        distance_squared = math.maximum(distance_squared, radius * 1e-2)  # Prevent infinite gradient at sphere center
        distance = math.sqrt(distance_squared)
        return distance - radius


@struct.definition()
class _Union(Geometry):

    def __init__(self, geometries, **kwargs):
        Geometry.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def geometries(self, geometries):
        assert len(geometries) > 0
        rank = geometries[0].rank
        for g in geometries[1:]:
            assert g.rank == rank or g.rank is None or rank is None
        return tuple(geometries)

    @property
    def rank(self):
        return self.geometries[0].rank

    def lies_inside(self, location):
        return math.any([geometry.lies_inside(location) for geometry in self.geometries], axis=0)

    def approximate_signed_distance(self, location):
        return math.min([geometry.approximate_signed_distance(location) for geometry in self.geometries], axis=0)

    def __add__(self, other):
        other_geometries = other.geometries if not isinstance(other, _Union) else (other,)
        return _Union(self.geometries + other_geometries)


def union(*geometries):
    if len(geometries) == 1 and isinstance(geometries[0], (tuple, list)):
        geometries = geometries[0]
    if len(geometries) == 0:
        return NO_GEOMETRY
    else:
        return _Union(geometries)


@struct.definition()
class _NoGeometry(Geometry):

    def rank(self):
        return None

    def approximate_signed_distance(self, location):
        return math.tile(np.inf, list(math.shape(location)[:-1]) + [1])

    def lies_inside(self, location):
        return math.tile(False, list(math.shape(location)[:-1]) + [1])

    def approximate_fraction_inside(self, location, cell_size):
        return math.tile(math.to_float(0), list(math.shape(location)[:-1]) + [1])


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

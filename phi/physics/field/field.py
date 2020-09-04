import numpy as np

from phi import math, struct
from phi.geom import Geometry
from phi.physics import State
from phi.physics.field.flag import _PROPAGATOR


def _to_valid_data(data):
    if data is None:
        return None
    if isinstance(data, (tuple, list)):
        return np.array(data)  # numbers or objects
    else:
        return data


@struct.definition()
class Field(State):

    def __init__(self, data, name=None, **kwargs):
        if 'tags' not in kwargs:
            tags = [name, 'field'] if name is not None else ['field']
        State.__init__(self, **struct.kwargs(locals()))

    @property
    def points(self):
        """
        Returns a Field containing all sample points of this field.
        The returned Field is compatible with this one.
        If the components of this field are sampled at different locations, this method raises StaggeredSamplePoints.
        If this field has no sample points, points is None.
        :return: vector Field
        """
        raise NotImplementedError(self)

    @property
    def elements(self):
        """
        Returns the geometry of all cells/particles/elements at which this Field is sampled.
        If the components of this field are sampled at different locations, this method raises StaggeredSamplePoints.
        If this field has no sample points, this method returns None.
        :rtype: Geometry
        :return: all sample elements
        """
        if self.points is None:  # may raise StaggeredSamplePoints
            return None
        raise NotImplementedError(self)

    @struct.variable()
    def data(self, data):
        """
        Data holds the values of this field according to the order specified by points.
        For composite fields, data holds a tuple of component fields.
        :return: n-dimensional tensor
        """
        return _to_valid_data(data)

    @struct.constant()
    def flags(self, flags):
        """
        Flags describe constants_dict of a Field such as divergence-freeness.
        :return: tuple of flags
        """
        if flags is None:
            return ()
        else:
            flags = tuple(set(flags))  # remove duplicates
            for flag in flags:
                if not flag.is_applicable(self.rank, self.component_count):
                    raise ValueError('Flag "%s" is not applicable to field %s' % (flag, self))
            return flags

    def sample_at(self, points):
        """
        Samples this field at the given points.
        :param points: tensor or rank >= 2 containing world-space vectors
        :return: tensor of shape (*location.shape[:-1], field.component_count)
        """
        raise NotImplementedError(self)

    def approximate_mean_value_in(self, geometry):
        """
        Computes the (approximate) mean field value inside the region specified by `geometry`.
        The geometry is assumed to be small compared to the structure of this field and may be approximated as a simpler shape.

        Let V be the volume of `geometry`. This method approximates `1/V * integral_V field.sample_at(x) dx`.
        :param geometry: (batched) Geometry
        :type geometry: Geometry
        :return: float tensor of shape (*geometry.batch_dimensions, field.component_count)
        """
        assert isinstance(geometry, Geometry)
        return self.sample_at(geometry.center)

    def at(self, other_field):
        """
        Resample this field at the same points as other_field.
        The returned Field is compatible with other_field.
        :param other_field: Field
        :return: a new Field which samples all components of this field at the points of other_field
        """
        try:
            values = self.approximate_mean_value_in(other_field.elements)
            result = other_field.copied_with(data=values, flags=propagate_flags_resample(self, other_field.flags, other_field.rank))
            return result
        except StaggeredSamplePoints:  # other_field is staggered
            return broadcast_at(self, other_field)

    @property
    def rank(self):
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        Note that this does not indicate the shape of the data array.
        If the field is independent of the dimensionality, the rank property is None.
        :return: int
        """
        raise NotImplementedError(self)

    @property
    def component_count(self):
        """
        Number of components of this Field.
        The components can be sampled at the same points or at different points (like with StaggeredGrids).
        :return: int
        :rtype: int
        """
        raise NotImplementedError(self)

    def unstack(self):
        """
        Split the Field by components.
        If the field only has one component, returns a list containing itself.
        :return: tuple of Fields
        :rtype: tuple
        """
        raise NotImplementedError(self)

    @property
    def has_points(self):
        try:
            return self.points is not None
        except StaggeredSamplePoints:
            return True

    def compatible(self, other_field):
        """
        Checks if two Fields have the same sample points and values are stored in the same order.
        For performance reasons, this method does not actually check every single point.
        Even if this method returns False, the sample points may still be the same.
        :type other_field: Field
        :return: True if both Fields have the same sample points.
        :rtype: bool
        """
        raise NotImplementedError(self)

    def with_data(self, data):
        return self.copied_with(data=data, flags=())

    def __mul__(self, other):
        return self.__dataop__(other, True, lambda d1, d2: math.mul(d1, d2))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.__dataop__(other, True, lambda d1, d2: math.div(d1, d2))

    def __rtruediv__(self, other):
        return self.__dataop__(other, False, lambda d1, d2: math.div(d2, d1))

    def __sub__(self, other):
        return self.__dataop__(other, False, lambda d1, d2: math.sub(d1, d2))

    def __rsub__(self, other):
        return self.__dataop__(other, False, lambda d1, d2: math.sub(d2, d1))

    def __add__(self, other):
        return self.__dataop__(other, False, lambda d1, d2: math.add(d1, d2))

    __radd__ = __add__

    def __pow__(self, power, modulo=None):
        return self.__dataop__(power, False, lambda f, p: math.pow(f, p))

    def __neg__(self):
        return self * -1

    def __gt__(self, other):
        return self.__dataop__(other, False, lambda x, y: x > y)

    def __ge__(self, other):
        return self.__dataop__(other, False, lambda x, y: x >= y)

    def __lt__(self, other):
        return self.__dataop__(other, False, lambda x, y: x < y)

    def __le__(self, other):
        return self.__dataop__(other, False, lambda x, y: x <= y)

    def __and__(self, other):
        return self.__dataop__(other, False, lambda x, y: x & y)

    def __or__(self, other):
        return self.__dataop__(other, False, lambda x, y: x | y)

    def __dataop__(self, other, linear_if_scalar, data_operator):
        if isinstance(other, Field):
            assert self.compatible(other), 'Fields are not compatible: %s and %s' % (self, other)
            flags = propagate_flags_operation(self.flags + other.flags, False, self.rank, self.component_count)
            self_data = self.data if self.has_points else self.at(other).data
            other_data = other.data if other.has_points else other.at(self).data
            data = data_operator(self_data, other_data)
        else:
            flags = propagate_flags_operation(self.flags, linear_if_scalar, self.rank, self.component_count)
            data = data_operator(self.data, other)
        return self.copied_with(data=data, flags=flags)

    def default_physics(self):
        from .effect import FieldPhysics
        return FieldPhysics(self.name)


class StaggeredSamplePoints(Exception):

    def __init__(self, *args):
        Exception.__init__(self, *args)


class IncompatibleFieldTypes(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


def propagate_flags_resample(data_field, structure_flags, resulting_rank):
    flags = []
    for flag in data_field.flags:
        if flag.is_data_bound and \
                flag.propagates(_PROPAGATOR.RESAMPLE) and \
                flag.is_applicable(resulting_rank, data_field.component_count):
            flags.append(flag)
    for flag in structure_flags:
        if flag.is_structure_bound and \
                flag.propagates(_PROPAGATOR.RESAMPLE) and \
                flag.is_applicable(resulting_rank, data_field.component_count):
            flags.append(flag)
    return tuple(flags)


def propagate_flags_children(flags, child_rank, child_component_count):
    result = []
    for flag in flags:
        if flag.propagates(_PROPAGATOR.CHILDREN) and flag.is_applicable(child_rank, child_component_count):
            result.append(flag)
    return tuple(result)


def propagate_flags_operation(flags, is_linear, result_rank, result_components):
    result = []
    propagator = _PROPAGATOR.LINEAR_OPERATIONS if is_linear else _PROPAGATOR.ALL_OPERATIONS
    for flag in flags:
        if flag.is_data_bound and\
                flag.propagates(propagator) and\
                flag.is_applicable(result_rank, result_components):
            result.append(flag)
    return tuple(result)


def broadcast_at(field1, field2):
    if field1.component_count != field2.component_count and field1.component_count != 1:
        raise IncompatibleFieldTypes('Can only resample to staggered fields with same number of components.\n%s\n%s' % (field1, field2))
    if field1.component_count == 1:
        new_components = [field1.at(f2) for f2 in field2.unstack()]
    else:
        new_components = [f1.at(f2) for f1, f2 in zip(field1.unstack(), field2.unstack())]
    return field2.copied_with(data=tuple(new_components), flags=propagate_flags_resample(field1, field2.flags, field2.rank))

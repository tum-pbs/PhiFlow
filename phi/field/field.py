from phi.physics.physics import State
from phi import math
from .flag import Flag, _PROPAGATOR


DIVERGENCE_FREE = 'divergence-free'


class Field(State):
    __struct__ = State.__struct__.extend(['_data'], ['_bounds', '_name', '_flags'])

    def __init__(self, name, bounds, data, flags=(), batch_size=None):
        State.__init__(self, tags=[name, 'field'], batch_size=batch_size)
        self._data = data
        self._name = name
        self._bounds = bounds
        self._flags = tuple(set(flags))  # remove duplicates
        for flag in flags:
            if not flag.is_applicable(self.rank, self.component_count):
                raise ValueError('Flag "%s" is not applicable to field %s' % (flag, self))

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        """
Data holds the values of this field according to the order specified by points.
For composite fields, data holds a tuple of component fields.
        :return: n-dimensional tensor
        """
        return self._data

    @property
    def bounds(self):
        """
The bounds describe the spatial region inside which this field is defined.
Depending on the boundary conditions, the field may be extrapolated beyond the bounds.
        :return:
        """
        return self._bounds

    @property
    def flags(self):
        """
Flags describe properties of a Field such as divergence-freeness.
        :return: tuple of flags
        """
        return self._flags

    def sample_at(self, points):
        """
Resample this field at the given points.
        :param points: tensor or rank >= 2 containing world-space vectors
        :return: tensor of shape location.shape[:-1]+[components]
        """
        raise NotImplementedError(self)

    def resample(self, other_field, force_optimization=False):
        """
Resample this field at the same points as other_field.
The returned Field is compatible with other_field.
        :param location: Field
        :param force_optimization: If true, this algorithm either uses an optimized implementation
        :return: a new Field which samples all components of this field at the points of other_field
        """
        if force_optimization:
            raise ValueError('No optimized resample algorithm found for fields %s, %s' % (self, other_field))
        try:
            resampled = self.sample_at(other_field.points.data)
            resampled = other_field.copied_with(data=resampled, flags=propagate_flags(self, other_field))
            return resampled
        except StaggeredSamplePoints:
            assert self.component_count == other_field.component_count, 'Can only resample to staggered fields with same number of components.\n%s\n%s' % (self, other_field)
            new_components = [f1.resample(f2) for f1, f2 in zip(self.unstack(), other_field.unstack())]
            return other_field.copied_with(data=tuple(new_components), flags=propagate_flags(self, other_field))

    @property
    def component_count(self):
        """
Number of components of this Field.
The components can be sampled at the same points or at different points (like with StaggeredGrids).
        :return: int
        """
        raise NotImplementedError(self)

    @property
    def rank(self):
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        Note that this does not indicate the shape of the data array.
        :return: int
        """
        raise NotImplementedError(self)

    def unstack(self):
        """
Split the Field by components.
If the field only has one component, returns a list containing itself.
        :return: tuple of Fields
        """
        raise NotImplementedError(self)

    @property
    def points(self):
        """
Returns a Field containing all sample points of this field.
The returned Field is compatible with this one.
If the components of this field are sampled at different locations, this method raises StaggeredSamplePoints.
        :return: VectorField
        """
        raise NotImplementedError(self)

    def compatible(self, other_field):
        """
Checks if two Fields have the same sample points and values are stored in the same order.
For performance reasons, this method does not actually check every single point.
Even if this method returns False, the sample points may still be the same.
        :param other_field:
        :return: True if both Fields have the same sample points.
        """
        return other_field.points == self.points


class StaggeredSamplePoints(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(*args, **kwargs)


def propagate_flags(data_field, structure_field):
    flags = []
    for flag in data_field.flags:
        if flag.is_data_bound and \
                flag.propagates(_PROPAGATOR.RESAMPLE) and \
                flag.is_applicable(structure_field.rank, data_field.component_count):
            flags.append(flag)
    for flag in structure_field.flags:
        if flag.is_structure_bound and \
                flag.propagates(_PROPAGATOR.RESAMPLE) and \
                flag.is_applicable(structure_field.rank, data_field.component_count):
            flags.append(flag)
    return flags
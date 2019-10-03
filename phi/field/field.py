from phi import struct


class Field(struct.Struct):
    __struct__ = struct.Def((), ('_bounds',))

    def __init__(self, bounds=None, flags=()):
        struct.Struct.__init__(self)
        self._bounds = bounds
        self._flags = flags

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

    def resample(self, location):
        """
Resample this field at the given location(s).
Resampled values outside the bounds may be extrapolated.
        :param location: VectorField or n-dimensional tensor containing world-space vectors
        :return: a new Field which samples all components of this field at the given locations
        """
        raise NotImplementedError(self)

    @property
    def component_count(self):
        raise NotImplementedError(self)

    def unstack(self):
        """
Split the Field by components.
If the field only has one component, returns a list containing itself.
        :return: tuple of Fields
        """
        raise NotImplementedError(self)

    @property
    def sample_points(self):
        """
Returns an n-dimensional tensor containing all sample points of this field.
If the components of this field are sampled at different locations, this method raises StaggeredSamplePoints.
        :return: n-dimensional tensor
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
        return other_field.sample_points is self.sample_points


class StaggeredSamplePoints(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(*args, **kwargs)

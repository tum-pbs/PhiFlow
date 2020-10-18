from phi import math, struct

from ._geom import Geometry
from ._geom_util import _fill_spatial_with_singleton
from ..math import tensor, combined_shape


class Sphere(Geometry):
    """
    N-dimensional sphere.
    Defined through center position and radius.
    """

    def __init__(self, center, radius):
        self._center = tensor(center, names='..., vector', channel_dims=1, spatial_dims=0)
        self._radius = tensor(radius, names='..., vector', channel_dims=0, spatial_dims=0)
        self._shape = _fill_spatial_with_singleton(combined_shape(self._center, self._radius))

    @property
    def shape(self):
        return self._shape

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    def lies_inside(self, location):
        distance_squared = math.sum((location - self.center) ** 2, axis=0)
        return distance_squared <= self.radius ** 2

    def approximate_signed_distance(self, location):
        """
Computes the exact distance from location to the closest point on the sphere.
Very close to the sphere center, the distance takes a constant value.
        :param location: float tensor of shape (batch_size, ..., rank)
        :return: float tensor of shape (*location.shape[:-1], 1).
        """
        distance_squared = math.vec_squared(location - self.center)
        distance_squared = math.maximum(distance_squared, self.radius * 1e-2)  # Prevent infinite gradient at sphere center
        distance = math.sqrt(distance_squared)
        return distance - self.radius

    def bounding_radius(self):
        return self.radius

    def bounding_half_extent(self):
        return self.radius

    def shifted(self, delta):
        return Sphere(self._center + delta, self._radius)

    def rotated(self, angle):
        return self

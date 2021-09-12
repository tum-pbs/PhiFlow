from typing import Dict

from phi import math, struct

from ._geom import Geometry, _fill_spatial_with_singleton
from ..math import wrap


class Sphere(Geometry):
    """
    N-dimensional sphere.
    Defined through center position and radius.

    Args:

    Returns:

    """

    def __init__(self, center, radius):
        self._center = wrap(center)
        assert 'vector' in self._center.shape, f"Sphere.center must have a 'vector' dimension. Try ({center},) * rank."
        self._radius = wrap(radius)

    @property
    def shape(self):
        return _fill_spatial_with_singleton(self._center.shape & self._radius.shape).without('vector')

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    @property
    def volume(self) -> math.Tensor:
        return 4 / 3 * math.PI * self._radius ** 3

    def lies_inside(self, location):
        distance_squared = math.sum((location - self.center) ** 2, dim='vector')
        return math.any(distance_squared <= self.radius ** 2, self.shape.instance)  # union for instance dimensions

    def approximate_signed_distance(self, location):
        """
        Computes the exact distance from location to the closest point on the sphere.
        Very close to the sphere center, the distance takes a constant value.

        Args:
          location: float tensor of shape (batch_size, ..., rank)

        Returns:
          float tensor of shape (*location.shape[:-1], 1).

        """
        distance_squared = math.vec_squared(location - self.center)
        distance_squared = math.maximum(distance_squared, self.radius * 1e-2)  # Prevent infinite spatial_gradient at sphere center
        distance = math.sqrt(distance_squared)
        return math.min(distance - self.radius, self.shape.instance)  # union for instance dimensions

    def bounding_radius(self):
        return self.radius

    def bounding_half_extent(self):
        return self.radius

    def shifted(self, delta):
        return Sphere(self._center + delta, self._radius)

    def rotated(self, angle):
        return self

    def __variable_attrs__(self):
        return '_radius', '_center'

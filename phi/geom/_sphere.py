from typing import Dict

from phi import math, struct

from ._geom import Geometry, _fill_spatial_with_singleton
from ..math import tensor


class Sphere(Geometry):
    """
    N-dimensional sphere.
    Defined through center position and radius.

    Args:

    Returns:

    """

    def __init__(self, center, radius):
        self._center = tensor(center)
        self._radius = tensor(radius)
        self._shape = _fill_spatial_with_singleton(self._center.shape & self._radius.shape)

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
        distance_squared = math.sum((location - self.center) ** 2, dim='vector')
        return distance_squared <= self.radius ** 2

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

    def __eq__(self, other):
        return isinstance(other, Sphere) \
               and self._shape == other.shape \
               and math.all(self._radius == other.radius) \
               and math.all(self._center == other.center)

    def __characteristics__(self) -> Dict[str, math.Tensor]:
        return {
            'radius': self._radius,
            'center': self._center,
        }

    def __with__(self, center=None, radius=None):
        return Sphere(center if center is not None else self._center,
                      radius if radius is not None else self._radius)

from phi import math, struct

from ._geom import Geometry


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

    def bounding_radius(self):
        return self.radius

    def bounding_half_extent(self):
        return self.radius

    def shifted(self, delta):
        return self.copied_with(center=self.center + delta)

    def rotated(self, angle):
        return self

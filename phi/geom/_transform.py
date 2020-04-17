
from phi import struct, math
from ._geom import Geometry
from ._sphere import Sphere


@struct.definition()
class RotatedGeometry(Geometry):

    def __init__(self, geometry, rotation, **kwargs):
        Geometry.__init__(self, struct.kwargs(locals()))

    @struct.constant()
    def geometry(self, geometry):
        return geometry

    @struct.constant()
    def rotation(self, rotation):
        return rotation

    @property
    def center(self):
        return self.geometry.center

    def global_to_child(self, location):
        delta = location - self.center
        rotated = delta  # ToDo apply rotation
        final = rotated + self.center
        raise NotImplementedError(self)
        return final

    def lies_inside(self, location):
        return self.geometry.lies_inside(self.global_to_child(location))

    def approximate_signed_distance(self, location):
        return self.geometry.approximate_signed_distance(self.global_to_child(location))

    def bounding_radius(self):
        return self.geometry.bounding_radius()

    def bounding_half_extent(self):
        bounding_sphere = Sphere(self.center, self.bounding_radius())
        return bounding_sphere.bounding_half_extent()

    @property
    def rank(self):
        return self.geometry.rank


def rotate(geometry, rotation):
    assert isinstance(geometry, Geometry)
    if isinstance(geometry, RotatedGeometry):
        total_rotation = geometry.rotation + rotation  # ToDo concatenate rotations
        return RotatedGeometry(geometry.geometry, total_rotation)
    else:
        return RotatedGeometry(geometry, rotation)

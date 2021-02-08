import warnings

from phi import math
from phi.math import GLOBAL_AXIS_ORDER, Tensor
from ._geom import Geometry
from ._sphere import Sphere


class RotatedGeometry(Geometry):

    def __init__(self, geometry: Geometry, angle: float or math.Tensor):
        if isinstance(geometry, RotatedGeometry):
            warnings.warn('Using RotatedGeometry of RotatedGeometry. Consider simplifying your setup.')
        self._geometry = geometry
        self._angle = math.tensor(angle)

    @property
    def shape(self):
        return self._geometry.shape

    @property
    def geometry(self):
        return self._geometry

    @property
    def angle(self):
        return self._angle

    @property
    def center(self):
        return self.geometry.center

    def _rotate(self, location):
        sin = math.sin(self.angle)
        cos = math.cos(self.angle)
        y, x = location.vector.unstack()
        if GLOBAL_AXIS_ORDER.is_x_first:
            x, y = y, x
        rot_x = cos * x - sin * y
        rot_y = sin * x + cos * y
        return math.channel_stack([rot_y, rot_x], 'vector')

    def global_to_child(self, location):
        """
        Inverse transform

        Args:
          location: 

        Returns:

        """
        delta = location - self.center
        if location.shape.vector == 2:
            rotated = self._rotate(delta)
        elif location.shape.vector == 3:
            raise NotImplementedError('not yet implemented')  # ToDo apply angle
        else:
            raise NotImplementedError('Rotation only supported in 2D and 3D')
        final = rotated + self.center
        return final

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0):
        rotated = self.global_to_child(positions)
        shifted_positions = self.geometry.push(rotated, outward=outward, shift_amount=shift_amount)
        return positions + self._rotate(shifted_positions - rotated)

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
        return self.geometry.spatial_rank

    def shifted(self, delta):
        return RotatedGeometry(self._geometry.shifted(delta), self._angle)

    def rotated(self, angle):
        return RotatedGeometry(self._geometry, self._angle + angle)


def rotate(geometry, angle):
    """
    package-internal rotation function. Users should use Geometry.rotated() instead.

    Args:
      geometry: 
      angle: 

    Returns:

    """
    assert isinstance(geometry, Geometry)
    if isinstance(geometry, RotatedGeometry):
        total_rotation = geometry.angle + angle  # ToDo concatenate rotations
        return RotatedGeometry(geometry.geometry, total_rotation)
    else:
        return RotatedGeometry(geometry, angle)

import warnings
from numbers import Number

from phi import math
from phi.math import GLOBAL_AXIS_ORDER, Tensor, channel
from ._geom import Geometry
from ._sphere import Sphere


class RotatedGeometry(Geometry):

    def __init__(self, geometry: Geometry, angle: float or math.Tensor):
        assert not isinstance(geometry, RotatedGeometry)
        self._geometry = geometry
        self._angle = math.wrap(angle)

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

    @property
    def volume(self) -> Tensor:
        return self._geometry.volume

    @property
    def shape_type(self) -> Tensor:
        return math.map(lambda s: f"rot{s}", self._geometry.shape_type)

    def global_to_child(self, location):
        """ Inverse transform. """
        delta = location - self.center
        if location.shape.get_size('vector') == 2:
            rotated = math.rotate_vector(delta, self._angle)
        elif location.shape.get_size('vector') == 3:
            raise NotImplementedError('not yet implemented')  # ToDo apply angle
        else:
            raise NotImplementedError('Rotation only supported in 2D and 3D')
        final = rotated + self.center
        return final

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0):
        rotated = self.global_to_child(positions)
        shifted_positions = self.geometry.push(rotated, outward=outward, shift_amount=shift_amount)
        return positions + math.rotate_vector(shifted_positions - rotated, self._angle)

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

    def shifted(self, delta) -> Geometry:
        return RotatedGeometry(self._geometry.shifted(delta), self._angle)

    def rotated(self, angle) -> Geometry:
        return RotatedGeometry(self._geometry, self._angle + angle)

    def scaled(self, factor: float or Tensor) -> 'Geometry':
        return RotatedGeometry(self._geometry.scaled(factor), self._angle)

    def unstack(self, dimension: str) -> tuple:
        return tuple([RotatedGeometry(g, self._angle) for g in self._geometry.unstack(dimension)])

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        loc = self._geometry.sample_uniform(*shape)
        return math.rotate_vector(loc, self._angle)

    def __hash__(self):
        return hash(self._angle) + hash(self._geometry)


def rotate(geometry: Geometry, angle: Number or Tensor) -> Geometry:
    """ Package-internal rotation function. Users should use Geometry.rotated() instead. """
    assert isinstance(geometry, Geometry)
    if isinstance(geometry, RotatedGeometry):
        total_rotation = geometry.angle + angle  # ToDo concatenate rotations
        return RotatedGeometry(geometry.geometry, total_rotation)
    else:
        return RotatedGeometry(geometry, angle)

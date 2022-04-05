import warnings
from typing import Dict

from phi import math

from ._geom import Geometry
from ..math import wrap, Tensor
from ..math.backend import PHI_LOGGER


class Sphere(Geometry):
    """
    N-dimensional sphere.
    Defined through center position and radius.
    """

    def __init__(self,
                 center: Tensor = None,
                 radius: float or Tensor = None,
                 **center_: float or Tensor):
        """
        Args:
            center: Sphere center as `Tensor` with `vector` dimension.
                The spatial dimension order should be specified in the `vector` dimension via item names.
            radius: Sphere radius as `float` or `Tensor`
            **center_: Specifies center when the `center` argument is not given. Center position by dimension, e.g. `x=0.5, y=0.2`.
        """
        if center is not None:
            if not isinstance(center, Tensor):
                warnings.warn(f"constructor Sphere({type(center)}) is deprecated. Use Sphere(Tensor) or Sphere(**center_) instead.", DeprecationWarning)
                self._center = wrap(center)
            else:
                self._center = center
            assert 'vector' in self._center.shape, f"Sphere.center must have a 'vector' dimension. Use the syntax x=0.5."
        else:
            self._center = math.wrap(tuple(center_.values()), math.channel(vector=tuple(center_.keys())))
            # self._center = wrap(math.spatial(**center_), math.channel('vector'))
        assert radius is not None, "radius must be specified."
        self._radius = wrap(radius)

    @property
    def shape(self):
        if self._center is None or self._radius is None:
            return None
        return (self._center.shape & self._radius.shape).without('vector')

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    @property
    def volume(self) -> math.Tensor:
        if self.spatial_rank == 1:
            return 2 * self._radius
        elif self.spatial_rank == 2:
            return math.PI * self._radius ** 2
        elif self.spatial_rank == 3:
            return 4 / 3 * math.PI * self._radius ** 3
        else:
            raise NotImplementedError()
            # n = self.spatial_rank
            # return math.pi ** (n // 2) / math.faculty(math.ceil(n / 2)) * self._radius ** n

    @property
    def shape_type(self) -> Tensor:
        return math.tensor('S')

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

    def sample_uniform(self, *shape: math.Shape):
        raise NotImplementedError('Not yet implemented')  # ToDo

    def bounding_radius(self):
        return self.radius

    def bounding_half_extent(self):
        return self.radius

    def shifted(self, delta):
        return Sphere(self._center + delta, self._radius)

    def rotated(self, angle):
        return self

    def scaled(self, factor: float or Tensor) -> 'Geometry':
        return Sphere(self.center, self.radius * factor)

    def __variable_attrs__(self):
        return '_radius', '_center'

    def unstack(self, dimension: str) -> tuple:
        center = self.center.dimension(dimension).unstack(self.shape.get_size(dimension))
        radius = self.radius.dimension(dimension).unstack(self.shape.get_size(dimension))
        return tuple([Sphere(c, r) for c, r in zip(center, radius)])

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError()

    def __hash__(self):
        return hash(self._center) + hash(self._radius)

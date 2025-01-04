from typing import Union, Dict, Tuple

from phi import math
from phiml.math import Shape, dual, PI, non_channel, instance
from ._geom import Geometry, _keep_vector
from ..math import wrap, Tensor, expand
from ..math.magic import slicing_dict


class Sphere(Geometry):
    """
    N-dimensional sphere.
    Defined through center position and radius.
    """

    def __init__(self,
                 center: Tensor = None,
                 radius: Union[float, Tensor] = None,
                 volume: Union[float, Tensor] = None,
                 radius_variable=True,
                 **center_: Union[float, Tensor]):
        """
        Args:
            center: Sphere center as `Tensor` with `vector` dimension.
                The spatial dimension order should be specified in the `vector` dimension via item names.
                Can be left empty to specify dimensions via kwargs.
            radius: Sphere radius as `float` or `Tensor`
            **center_: Specifies center when the `center` argument is not given. Center position by dimension, e.g. `x=0.5, y=0.2`.
        """
        if center is not None:
            assert isinstance(center, Tensor), f"center must be a Tensor but got {type(center).__name__}"
            assert 'vector' in center.shape, f"Sphere center must have a 'vector' dimension."
            assert center.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Sphere(x=x, y=y) to assign names."
            self._center = center
        else:
            self._center = wrap(tuple(center_.values()), math.channel(vector=tuple(center_.keys())))
        if radius is None:
            assert volume is not None, f"Either radius or volume must be specified but got neither."
            self._radius = Sphere.radius_from_volume(wrap(volume), self._center.vector.size)
        else:
            self._radius = wrap(radius)
        self._radius_variable = radius_variable
        assert 'vector' not in self._radius.shape, f"Sphere radius must not vary along vector but got {radius}"

    def __all_attrs__(self) -> tuple:
        return ('_center', '_radius')

    def __variable_attrs__(self) -> tuple:
        return ('_center', '_radius') if self._radius_variable else ('_center',)

    def __value_attrs__(self) -> tuple:
        return ()

    @property
    def shape(self):
        if self._center is None or self._radius is None:
            return None
        return self._center.shape & self._radius.shape

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    @property
    def volume(self) -> math.Tensor:
        return Sphere.volume_from_radius(self._radius, self.spatial_rank)

    @staticmethod
    def volume_from_radius(radius: Union[float, Tensor], spatial_rank: int):
        if spatial_rank == 1:
            return 2 * radius
        elif spatial_rank == 2:
            return PI * radius ** 2
        elif spatial_rank == 3:
            return 4/3 * PI * radius ** 3
        else:
            raise NotImplementedError(f"spatial_rank>3 not supported, got {spatial_rank}")
            # n = self.spatial_rank
            # return math.pi ** (n // 2) / math.faculty(math.ceil(n / 2)) * self._radius ** n

    @staticmethod
    def radius_from_volume(volume: Union[float, Tensor], spatial_rank: int):
        if spatial_rank == 1:
            return volume / 2
        elif spatial_rank == 2:
            return math.sqrt(volume / PI)
        elif spatial_rank == 3:
            return (.75 / PI * volume) ** (1/3)
        else:
            raise NotImplementedError(f"spatial_rank>3 not supported, got {spatial_rank}")

    @staticmethod
    def area_from_radius(radius: Union[float, Tensor], spatial_rank: int):
        if spatial_rank == 1:
            return 0
        elif spatial_rank == 2:
            return 2*PI * radius
        elif spatial_rank == 3:
            return 4*PI * radius**2
        else:
            raise NotImplementedError(f"spatial_rank>3 not supported, got {spatial_rank}")

    def lies_inside(self, location):
        distance_squared = math.sum((location - self.center) ** 2, dim='vector')
        return math.any(distance_squared <= self.radius ** 2, self.shape.instance)  # union for instance dimensions

    def approximate_signed_distance(self, location: Union[Tensor, tuple]):
        """
        Computes the exact distance from location to the closest point on the sphere.
        Very close to the sphere center, the distance takes a constant value.

        Args:
          location: float tensor of shape (batch_size, ..., rank)

        Returns:
          float tensor of shape (*location.shape[:-1], 1).

        """
        distance = math.vec_length(location - self._center, eps=1e-3)
        return math.min(distance - self.radius, self.shape.instance)  # union for instance dimensions

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self_center = self.center
        self_radius = self.radius
        center_delta = location - self_center
        center_dist = math.vec_length(center_delta)
        sgn_dist = center_dist - self_radius
        if instance(self):
            self_center, self_radius, sgn_dist, center_delta, center_dist = math.at_min((self.center, self.radius, sgn_dist, center_delta, center_dist), key=abs(sgn_dist), dim=instance(self))
        normal = math.safe_div(center_delta, center_dist)
        default_normal = wrap([1] + [0] * (self.spatial_rank-1), self.shape['vector'])
        normal = math.where(center_dist == 0, default_normal, normal)
        surface_pos = self_center + self_radius * normal
        delta = surface_pos - location
        face_index = expand(0, non_channel(location))
        offset = normal.vector @ surface_pos.vector
        return sgn_dist, delta, normal, offset, face_index

    def sample_uniform(self, *shape: math.Shape):
        # --- Choose a distance from the center of the sphere, equally weighted by mass ---
        uniform = math.random_uniform(self.shape.non_singleton.without('vector'), *shape)
        if self.spatial_rank == 1:
            r = self.radius * uniform
        else:
            r = self.radius * (uniform ** (1 / self.spatial_rank))
        # --- Uniformly sample a unit vector for direction over the surface of the sphere (Muller 1959, Marsaglia 1972) ---
        unit_vector = math.random_normal(self.shape.non_singleton.without('vector'), *shape, self.shape['vector'])
        unit_vector /= math.vec_length(unit_vector, vec_dim='vector')
        return self.center + r * unit_vector

    def bounding_radius(self):
        return self.radius

    def bounding_half_extent(self):
        return expand(self.radius, self._center.shape.only('vector'))

    def at(self, center: Tensor) -> 'Geometry':
        return Sphere(center, self._radius, radius_variable=self._radius_variable)

    def rotated(self, angle):
        return self

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return Sphere(self.center, self.radius * factor, radius_variable=self._radius_variable)

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        return Sphere(self._center[_keep_vector(item)], self._radius[item], radius_variable=self._radius_variable)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError(f"Sphere.faces not implemented.")

    @property
    def face_centers(self) -> Tensor:
        return math.zeros(self.shape & dual(shell=0))

    @property
    def face_areas(self) -> Tensor:
        return expand(Sphere.area_from_radius(self._radius, self.spatial_rank), instance(self) + dual(shell=1))

    @property
    def face_normals(self) -> Tensor:
        return math.zeros(self.shape & dual(shell=0))

    @property
    def boundary_elements(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {}

    @property
    def face_shape(self) -> Shape:
        return self.shape.without('vector') & dual(shell=1)

    @property
    def corners(self) -> Tensor:
        return math.zeros(self.shape & dual(corners=0))

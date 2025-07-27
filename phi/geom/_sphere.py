from dataclasses import dataclass
from functools import cached_property
from typing import Union, Dict, Tuple

from phi import math
from phiml.dataclasses import sliceable, replace
from phiml.math import Shape, dual, PI, non_channel, instance
from ._functions import vec_length
from ._geom import Geometry
from ..math import wrap, Tensor, expand


class SphereType(type):
    
    def __call__(cls, center: Tensor = None,
                 radius: Union[float, Tensor] = None,
                 volume: Union[float, Tensor] = None,
                 variable_attrs=('pos', 'radius'),
                 radius_variable=None,
                 pos: Tensor = None,
                 **center_: Union[float, Tensor]):
        assert radius_variable is None, f"radius_variable has been replaced by variable_attrs"
        if pos is not None:
            center = pos
        if center is not None:
            assert isinstance(center, Tensor), f"center must be a Tensor but got {type(center).__name__}"
            assert 'vector' in center.shape, f"Sphere center must have a 'vector' dimension."
            assert center.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Sphere(x=x, y=y) to assign names."
        else:
            center = wrap(tuple(center_.values()), math.channel(vector=tuple(center_.keys())))
        if radius is None:
            assert volume is not None, f"Either radius or volume must be specified but got neither."
            radius = Sphere.radius_from_volume(wrap(volume), center.vector.size)
        else:
            radius = wrap(radius)
        return type.__call__(cls, pos=center, radius=radius, variable_attrs=variable_attrs)


@sliceable(keepdims='vector')
@dataclass(frozen=True, eq=False)
class Sphere(Geometry, metaclass=SphereType):
    """
    N-dimensional sphere.
    Defined through center position and radius.
    """
    
    pos: Tensor
    radius: Tensor
    
    variable_attrs: Tuple[str, ...] = ('pos', 'radius')
    
    def __post_init__(self):
        assert 'vector' in self.pos.shape
        assert 'vector' not in self.radius.shape, f"Sphere radius must not vary along vector but got {self.radius}"

    @cached_property
    def shape(self):
        return self.pos.shape & self.radius.shape

    @property
    def center(self):
        return self.pos

    @cached_property
    def volume(self) -> math.Tensor:
        return Sphere.volume_from_radius(self.radius, self.spatial_rank)

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
            # return math.pi ** (n // 2) / math.faculty(math.ceil(n / 2)) * self.radius ** n

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
        distance = vec_length(location - self.pos, eps=1e-3)
        return math.min(distance - self.radius, self.shape.instance)  # union for instance dimensions

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self_center = self.center
        self_radius = self.radius
        center_delta = location - self_center
        center_dist = vec_length(center_delta)
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
        unit_vector /= vec_length(unit_vector)
        return self.center + r * unit_vector

    def bounding_radius(self):
        return self.radius

    def bounding_half_extent(self):
        return expand(self.radius, self.pos.shape.only('vector'))

    def at(self, center: Tensor) -> 'Geometry':
        return replace(self, pos=center)

    def rotated(self, angle):
        return self

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return replace(self, radius=self.radius * factor)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError(f"Sphere.faces not implemented.")

    @property
    def face_centers(self) -> Tensor:
        return math.zeros(self.shape & dual(shell=0))

    @property
    def face_areas(self) -> Tensor:
        return expand(Sphere.area_from_radius(self.radius, self.spatial_rank), instance(self) + dual(shell=1))

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

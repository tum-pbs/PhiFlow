from typing import Callable, Union
from numbers import Number

from phi import math

from ._field import FieldInitializer, slice_off_constant, get_sample_points
from ..geom import Geometry, UniformGrid
from ..math import Shape, spatial, instance, Tensor, wrap, Extrapolation


class AngularVelocity(FieldInitializer):
    """
    Model of a single vortex or set of vortices.
    The falloff of the velocity magnitude can be controlled.

    Without a specified falloff, the velocity increases linearly with the distance from the vortex center.
    This is the case with rotating rigid bodies, for example.
    """

    def __init__(self,
                 location: Union[Tensor, tuple, list, Number],
                 strength: Union[Tensor, Number] = 1.0,
                 falloff: Callable = None,
                 component: str = None):
        location = wrap(location)
        strength = wrap(strength)
        assert location.shape.channel.names == ('vector',), "location must have a single channel dimension called 'vector'"
        assert location.shape.spatial.is_empty, "location tensor cannot have any spatial dimensions"
        assert not instance(location), "AngularVelocity does not support instance dimensions"
        self.location = location
        self.strength = strength
        self.falloff = falloff
        spatial_names = location.vector.item_names
        assert spatial_names is not None, "location.vector must list spatial dimensions as item names"
        self._shape = location.shape & spatial(**{dim: 1 for dim in spatial_names})

    def _sample(self, geometry: Geometry, at: str, boundaries: Extrapolation, **kwargs) -> math.Tensor:
        points = get_sample_points(geometry, at, boundaries)
        distances = points - self.location
        strength = self.strength if self.falloff is None else self.strength * self.falloff(distances)
        velocity = math.cross_product(strength, distances)
        velocity = math.sum(velocity, self.location.shape.batch.without(points.shape))
        return velocity

    @property
    def shape(self) -> Shape:
        return self._shape

    def __getitem__(self, item: dict):
        assert all(dim == 'vector' for dim in item), f"Cannot slice AngularVelocity with {item}"
        if 'vector' in item:
            component = self.shape.spatial.names[item['vector']]
            return AngularVelocity(self.location, self.strength, self.falloff, component)
        else:
            return self

from collections import Callable
from numbers import Number

from phi import math

from ._field import Field
from ..geom import Geometry
from ..math import Shape, GLOBAL_AXIS_ORDER, spatial


class AngularVelocity(Field):
    """
    Model of a single vortex or set of vortices.
    The falloff of the velocity magnitude can be controlled.

    Without a specified falloff, the velocity increases linearly with the distance from the vortex center.
    This is the case with rotating rigid bodies, for example.
    """

    def __init__(self,
                 location: math.Tensor or tuple or list or Number,
                 strength: math.Tensor or Number = 1.0,
                 falloff: Callable = None,
                 component: str = None):
        location = math.wrap(location)
        strength = math.wrap(strength)
        assert location.shape.channel.names == ('vector',), "location must have a single channel dimension called 'vector'"
        assert location.shape.spatial.is_empty, "location tensor cannot have any spatial dimensions"
        self.location = location
        self.strength = strength
        self.falloff = falloff
        self.component = component
        spatial_names = [GLOBAL_AXIS_ORDER.axis_name(i, location.vector.size) for i in range(location.vector.size)]
        self._shape = location.shape & spatial(**{dim: 1 for dim in spatial_names})

    def _sample(self, geometry: Geometry) -> math.Tensor:
        points = geometry.center
        distances = points - self.location
        strength = self.strength if self.falloff is None else self.strength * self.falloff(distances)
        velocity = math.cross_product(strength, distances)
        velocity = math.sum(velocity, self.location.shape.batch.without(points.shape))
        if self.component:
            velocity = velocity.vector[self.component]
        return velocity

    @property
    def shape(self) -> Shape:
        return self._shape

    def __getitem__(self, item: dict):
        assert all(dim == 'vector' for dim in item), f"Cannot slice AngularVelocity with {item}"
        if 'vector' in item:
            assert item['vector'] == 0 or self.component is None
            component = self.shape.spatial.names[item['vector']]
            return AngularVelocity(self.location, self.strength, self.falloff, component)
        else:
            return self

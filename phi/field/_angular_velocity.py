from phi import math

from ._analytic import AnalyticField
from ..math import Shape, GLOBAL_AXIS_ORDER


class AngularVelocity(AnalyticField):

    def __init__(self, location, strength=1.0, falloff: callable = None):
        location = math.tensor(location)
        assert location.shape.channel.names == ('vector',), "location must have a single channel dimension called 'vector'"
        assert location.shape.spatial.is_empty, "location tensor cannot have any spatial dimensions"
        self.location = location
        self.strength = strength
        self.falloff = falloff
        spatial_names = [GLOBAL_AXIS_ORDER.axis_name(i, location.vector.size) for i in range(location.vector.size)]
        self._shape = location.shape.combined(math.spatial_shape([1] * location.vector.size, spatial_names))

    def sample_at(self, points, reduce_channels=()) -> math.Tensor:
        distances = points - self.location
        strength = self.strength if self.falloff is None else self.strength * self.falloff(distances)
        if reduce_channels:
            assert len(reduce_channels) == 1
            velocities = [math.cross_product(strength, dist).vector[i] for i, dist in enumerate(distances.unstack(reduce_channels[0]))]  # TODO this is inefficient, computes components that are discarded
            velocity = math.channel_stack(velocities, 'vector')
        else:
            velocity = math.cross_product(strength, distances)
        velocity = math.sum(velocity, self.location.shape.batch.without(points.shape))
        return velocity

    @property
    def shape(self) -> Shape:
        return self._shape

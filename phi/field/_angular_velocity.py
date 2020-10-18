from phi import math
from phi.geom import Geometry

from ._analytic import AnalyticField
from ..math import Shape


class AngularVelocity(AnalyticField):

    def __init__(self, location: math.Tensor, strength=1.0, falloff: callable = None):
        assert location.shape.channel.names == ('vector',), "location must have a single channel dimension called 'vector'"
        assert location.shape.spatial.is_empty, "location tensor cannot have any spatial dimensions"
        self.location = location
        self.strength = strength
        self.falloff = falloff
        self._shape = location.shape.combined(math.spatial_shape([1] * location.vector.size))

    def sample_at(self, points, reduce_channels=()) -> math.Tensor:
        if isinstance(points, Geometry):
            points = points.center  # TODO correct for cells very close to location
        distances = points - self.location
        strength = self.strength if self.falloff is None else self.strength * self.falloff(distances)
        if points.vector.size == 2:  # Curl in 2D
            dist_0, dist_1 = distances.vector.unstack()
            if reduce_channels:
                assert len(reduce_channels) == 1
                dist_0 = dist_0[{reduce_channels[0]: 0}]
                dist_1 = dist_1[{reduce_channels[0]: 1}]
            if math.GLOBAL_AXIS_ORDER.is_x_first:
                velocity = strength * math.channel_stack([-dist_1, dist_0], 'vector')
            else:
                velocity = strength * math.channel_stack([dist_1, -dist_0], 'vector')
        elif points.vector.size == 3:  # Curl in 3D
            raise NotImplementedError('not yet implemented')
        else:
            raise AssertionError('Vector product not available in > 3 dimensions')
        # velocity = math.vec_prod(strength, distances)
        velocity = math.sum(velocity, self.location.shape.batch.without(points.shape))
        return velocity

    @property
    def shape(self) -> Shape:
        return self._shape

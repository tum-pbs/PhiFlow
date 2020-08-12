from phi import math, struct
from phi.geom import GLOBAL_AXIS_ORDER

from .analytic import AnalyticField


@struct.definition()
class AngularVelocity(AnalyticField):

    def __init__(self, location, strength=1.0, **kwargs):
        AnalyticField.__init__(self, rank=None, **struct.kwargs(locals()))

    def sample_at(self, points):
        points_rank = math.spatial_rank(points)
        src_rank = math.spatial_rank(self.location)
        # --- Expand shapes to format (batch_size, points_dims..., src_dims..., channels) ---
        points = math.expand_dims(points, axis=-2, number=src_rank)
        src_location = math.expand_dims(self.location, axis=-3, number=points_rank)
        src_strength = math.expand_dims(self.strength, axis=-1)
        if math.ndims(src_strength) == 1:
            src_strength = math.expand_dims(src_strength, axis=-2)
        src_strength = math.expand_dims(src_strength, axis=-3, number=points_rank)
        src_axes = tuple(range(-2, -2 - src_rank, -1))
        # --- Compute distances and falloff ---
        distances = points - src_location
        if self.falloff is not None:
            falloff_value = self.falloff(distances)
            strength = src_strength * falloff_value
        else:
            strength = src_strength
        # --- Compute velocities ---
        if math.staticshape(points)[-1] == 2:  # Curl in 2D
            dist_1, dist_2 = math.unstack(distances, axis=-1)
            if GLOBAL_AXIS_ORDER.is_x_first:
                velocity = strength * math.stack([-dist_2, dist_1], axis=-1)
            else:
                velocity = strength * math.stack([dist_2, -dist_1], axis=-1)
        elif math.staticshape(points)[-1] == 3:  # Curl in 3D
            raise NotImplementedError('not yet implemented')
        else:
            raise AssertionError('Vector product not available in > 3 dimensions')
        velocity = math.sum(velocity, axis=src_axes)
        return velocity

    @property
    def component_count(self):
        return self.rank

    @struct.variable()
    def location(self, loc):
        loc = math.to_float(loc)
        assert math.staticshape(loc)[-1] in (2, 3)
        if math.ndims(loc) < 2:
            loc = math.expand_dims(loc, axis=0, number=2 - math.ndims(loc))
        return loc

    @struct.variable()
    def strength(self, strength):
        return math.to_float(strength)

    @struct.constant()
    def falloff(self, falloff):
        assert callable(falloff) or falloff is None
        return falloff

    @property
    def rank(self):
        return math.staticshape(self.location)[-1]

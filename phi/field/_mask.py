from phi import math
from phi.geom import Geometry
from ._analytic import AnalyticField


class GeometryMask(AnalyticField):
    """
    Field that takes the value 1 inside a Geometry object and 0 outside.
    When sampled at single points, the result is binary.
    When sampled given another geometry, the approximate overlap between the geometries is computed, allowing for fractional values between 0 and 1.

    This field supports batched geometries.
    """

    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    @property
    def shape(self):
        return self.geometry.shape.non_channel

    def sample_at(self, points, reduce_channels=()):
        if isinstance(points, Geometry):
            return self.geometry.approximate_fraction_inside(points)
        else:
            return math.to_float(self.geometry.lies_inside(points))

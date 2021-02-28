from phi import math
from phi.geom import Geometry
from ._field import Field
from ..math import Tensor


class HardGeometryMask(Field):
    """
    Field that takes the value 1 inside a Geometry object and 0 outside.
    For volume sampling, performs sampling at the center points.
    """

    def __init__(self, geometry: Geometry):
        assert isinstance(geometry, Geometry)
        self.geometry = geometry

    @property
    def shape(self):
        return self.geometry.shape.non_channel

    def sample_at(self, points: Tensor, reduce_channels=()) -> Tensor:
        inside = math.to_float(self.geometry.lies_inside(points))
        if reduce_channels:
            assert len(reduce_channels) == 1
            inside = inside.dimension(reduce_channels[0]).as_channel('vector')
        return inside

    def unstack(self, dimension: str) -> tuple:
        geometries = self.geometry.unstack(dimension)
        return tuple(HardGeometryMask(g) for g in geometries)


class SoftGeometryMask(HardGeometryMask):
    """
    When sampled given another geometry, the approximate overlap between the geometries is computed, allowing for fractional values between 0 and 1.
    """

    def sample_at(self, points: Tensor, reduce_channels=()) -> Tensor:
        raise NotImplementedError("Use HardGeometryMask to sample at points")

    def sample_in(self, geometry: Geometry, reduce_channels=()) -> Tensor:
        inside = self.geometry.approximate_fraction_inside(geometry)
        if reduce_channels:
            assert len(reduce_channels) == 1
            inside = inside.dimension(reduce_channels[0]).as_channel('vector')
        return inside

    def unstack(self, dimension: str) -> tuple:
        geometries = self.geometry.unstack(dimension)
        return tuple(SoftGeometryMask(g) for g in geometries)

from phi import math
from phi.geom import Geometry
from ._field import Field
from .numerical import Scheme
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

    def _sample(self, geometry: Geometry, scheme: Scheme) -> Tensor:
        return math.to_float(self.geometry.lies_inside(geometry.center))

    def __getitem__(self, item: dict):
        return HardGeometryMask(self.geometry[item])


class SoftGeometryMask(HardGeometryMask):
    """
    When sampled given another geometry, the approximate overlap between the geometries is computed, allowing for fractional values between 0 and 1.
    """
    def __init__(self, geometry: Geometry, balance: Tensor or float = 0.5):
        super().__init__(geometry)
        self.balance = balance

    def _sample(self, geometry: Geometry, scheme: Scheme) -> Tensor:
        return self.geometry.approximate_fraction_inside(geometry, self.balance)

    def __getitem__(self, item: dict):
        return SoftGeometryMask(self.geometry[item], self.balance)

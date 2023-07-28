import warnings
from typing import Union

from phi import math
from phi.geom import Geometry
from ._field import Field
from phiml.math import Tensor


class HardGeometryMask(Field):
    """
    Deprecated since version 1.3. Use `phi.field.mask()` or `phi.field.resample()` instead.
    """

    def __init__(self, geometry: Geometry):
        warnings.warn("HardGeometryMask and SoftGeometryMask are deprecated. Use field.mask or field.resample instead.", DeprecationWarning, stacklevel=2)
        assert isinstance(geometry, Geometry)
        self.geometry = geometry

    @property
    def shape(self):
        return self.geometry.shape.non_channel

    def _sample(self, geometry: Geometry, **kwargs) -> Tensor:
        return math.to_float(self.geometry.lies_inside(geometry.center))

    def __getitem__(self, item: dict):
        return HardGeometryMask(self.geometry[item])


class SoftGeometryMask(HardGeometryMask):
    """
    Deprecated since version 1.3. Use `phi.field.mask()` or `phi.field.resample()` instead.
    """
    def __init__(self, geometry: Geometry, balance: Union[Tensor, float] = 0.5):
        warnings.warn("HardGeometryMask and SoftGeometryMask are deprecated. Use field.mask or field.resample instead.", DeprecationWarning, stacklevel=2)
        super().__init__(geometry)
        self.balance = balance

    def _sample(self, geometry: Geometry, **kwargs) -> Tensor:
        return self.geometry.approximate_fraction_inside(geometry, self.balance)

    def __getitem__(self, item: dict):
        return SoftGeometryMask(self.geometry[item], self.balance)

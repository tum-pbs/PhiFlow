import warnings

from phi import math
from phi.math import Shape
from ._field import Field
from ..geom import Geometry


class ConstantField(Field):
    """
    Deprecated.
    """

    def __init__(self, value=1.0):
        warnings.warn("ConstantField is deprecated. Use numbers or tuples instead.", DeprecationWarning)
        self.value = math.wrap(value)

    @property
    def shape(self) -> Shape:
        return self.value.shape

    def _op1(self, operator) -> Field:
        return ConstantField(operator(self.value))

    def _op2(self, other, operator) -> Field:
        return ConstantField(operator(self.value, other))

    def _sample(self, geometry: Geometry) -> math.Tensor:
        return self.value

    def __getitem__(self, item):
        return ConstantField(self.value[item])

    def unstack(self, dimension: str):
        warnings.warn("ConstantField.unstack() is deprecated. Use field.unstack(ConstantField) instead.", DeprecationWarning)
        return tuple(ConstantField(v) for v in self.value.unstack(dimension))

    def __repr__(self):
        return repr(self.value)

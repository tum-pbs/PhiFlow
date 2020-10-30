from phi import math
from phi.math import Shape
from ._field import Field
from ..geom import Geometry


class ConstantField(Field):

    def __init__(self, value=1.0):
        self.value = math.tensor(value)

    @property
    def shape(self) -> Shape:
        return self.value.shape

    def _op1(self, operator) -> Field:
        return ConstantField(operator(self.value))

    def _op2(self, other, operator) -> Field:
        return ConstantField(operator(self.value, other))

    def sample_at(self, points, reduce_channels=()) -> math.Tensor:
        return self.value

    def unstack(self, dimension: str):
        return tuple(ConstantField(v) for v in self.value.unstack(dimension))

    def __repr__(self):
        return repr(self.value)

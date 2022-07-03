from phi.geom import GridCell, Box
from phi.math import Tensor, spatial, Extrapolation, Shape, stack
from ._field import Field, sample
from ..math.extrapolation import Undefined, ConstantExtrapolation


class FieldEmbedding(Extrapolation):

    def __init__(self, field: Field):
        super().__init__(pad_rank=1)
        self.field = field

    def to_dict(self) -> dict:
        raise NotImplementedError("FieldEmbedding cannot be converted to dict")

    def __value_attrs__(self):
        return 'field',

    def __getitem__(self, item):
        return FieldEmbedding(self.field[item])

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'ConstantExtrapolation':
        if all(isinstance(v, FieldEmbedding) for v in values):
            return ConstantExtrapolation(stack([v.field for v in values], dim, **kwargs))
        else:
            return NotImplemented

    def spatial_gradient(self) -> 'Extrapolation':
        return NotImplemented
        from ._field_math import spatial_gradient
        return FieldEmbedding(spatial_gradient(self.field))  # this is not supported for all fields

    def valid_outer_faces(self, dim) -> tuple:
        return False, False

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, bounds: Box = None, already_padded: dict = None, **kwargs) -> Tensor:
        assert bounds is not None, f"{type(self)}.pad() requires 'bounds' argument"
        if already_padded:
            padded_res = spatial(**{dim: lo + up for dim, (lo, up) in already_padded.items()})
            resolution = spatial(value) - padded_res
            value_grid = GridCell(resolution, bounds).padded(already_padded)
        else:
            value_grid = GridCell(spatial(value), bounds)
        if upper_edge:
            pad_grid = value_grid.padded({dim: (0, width)})[{dim: slice(-width, None)}]
        else:
            pad_grid = value_grid.padded({dim: (width, 0)})[{dim: slice(0, width)}]
        result = sample(self.field, pad_grid)
        return result
    
    @property
    def is_flexible(self) -> bool:
        return False

    def __repr__(self):
        return repr(self.field)

    def __abs__(self):
        return Undefined(self)

    def __neg__(self):
        return Undefined(self)

    def _op(self, other, op):
        if isinstance(other, ConstantExtrapolation):  # some operations can be handled by ConstantExtrapolation, e.g. * 0
            op = getattr(other, op.__name__)
            result = op(self)
            return Undefined(self) if result is NotImplemented else result
        elif isinstance(other, FieldEmbedding):
            if other.field is self.field:
                return self
            return Undefined(self)
        else:
            return Undefined(self)

    def __add__(self, other):
        return self._op(other, ConstantExtrapolation.__add__)

    def __radd__(self, other):
        return self._op(other, ConstantExtrapolation.__add__)

    def __mul__(self, other):
        return self._op(other, ConstantExtrapolation.__mul__)

    def __rmul__(self, other):
        return self._op(other, ConstantExtrapolation.__mul__)

    def __sub__(self, other):
        return self._op(other, ConstantExtrapolation.__rsub__)

    def __rsub__(self, other):
        return self._op(other, ConstantExtrapolation.__sub__)

    def __truediv__(self, other):
        return self._op(other, ConstantExtrapolation.__rtruediv__)

    def __rtruediv__(self, other):
        return self._op(other, ConstantExtrapolation.__truediv__)

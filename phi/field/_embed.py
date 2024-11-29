from typing import Union, Tuple

from phi.geom import UniformGrid, Box
from phi.math import Tensor, spatial, Extrapolation, Shape, stack
from phi.math.extrapolation import Undefined, ConstantExtrapolation, ZERO
from phiml import math
from phiml.math import unstack, rename_dims, instance, dual
from ._field import Field
from ._resample import sample


class FieldEmbedding(Extrapolation):

    def __init__(self, field: Field, gradient=False):
        if gradient:
            raise NotImplementedError("gradient of FieldEmbedding not yet supported")
        super().__init__(pad_rank=1)
        self.field = field

    @property
    def shape(self):
        return self.field.shape

    def to_dict(self) -> dict:
        raise NotImplementedError("FieldEmbedding cannot be converted to dict")

    def __value_attrs__(self):
        return 'field',

    def __variable_attrs__(self):
        return 'field',

    def __getitem__(self, item):
        return FieldEmbedding(self.field[item])

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'FieldEmbedding':
        if all(isinstance(v, FieldEmbedding) for v in values):
            stacked_fields = stack([v.field for v in values], dim, **kwargs)
            return FieldEmbedding(stacked_fields)
        else:
            return NotImplemented

    def spatial_gradient(self) -> 'Extrapolation':
        return ZERO
        # from ._field_math import spatial_gradient
        # return FieldEmbedding(spatial_gradient(self.field))  # this is not supported for all fields

    def determines_boundary_values(self, boundary_key: Union[Tuple[str, bool], str]) -> bool:
        return False

    def is_face_valid(self, key) -> bool:
        return False

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, bounds: Box = None, already_padded: dict = None, **kwargs) -> Tensor:
        assert bounds is not None or (already_padded and not value.shape.is_non_uniform), f"{type(self)}.pad() requires 'bounds' argument"
        if value.shape.is_non_uniform:
            unstacked = unstack(value, value.shape.non_uniform_shape)
            indices = value.shape.non_uniform_shape.meshgrid(names=True)
            padded = [self[i].pad_values(u, width, dim, upper_edge, bounds=bounds, already_padded=already_padded, **kwargs) for u, i in zip(unstacked, indices)]
            return stack(padded, value.shape.non_uniform_shape)
        if already_padded:
            resolution = spatial(value).after_pad({dim: (-lo, -up) for dim, (lo, up) in already_padded.items()})
            value_grid = UniformGrid(resolution, bounds).padded(already_padded)
        else:
            value_grid = UniformGrid(spatial(value), bounds)
        if upper_edge:
            pad_grid = value_grid.padded({dim: (0, width)})[{dim: slice(-width, None)}]
        else:
            pad_grid = value_grid.padded({dim: (width, 0)})[{dim: slice(0, width)}]
        result = sample(self.field, pad_grid)
        return result
    
    @property
    def is_flexible(self) -> bool:
        return False

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        assert 'mesh' in kwargs, f"sparse padding with Field as boundary only supported for meshes"
        from ..geom import Mesh
        mesh: Mesh = kwargs['mesh']
        boundary_slice = mesh.boundary_faces[dim]
        face_pos = mesh.face_centers[boundary_slice]
        face_pos = math.stored_values(face_pos)  # does this always preserve the order?
        sampled = sample(self.field, face_pos)
        return rename_dims(sampled, instance, dual(value))

    def __eq__(self, other):
        if not isinstance(other, FieldEmbedding):
            return False
        if other.field is self.field:
            return True
        if self.field.values is None or other.field.values is None:
            return self.field == other.field
        return False

    def __hash__(self):
        return hash(self.field)

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

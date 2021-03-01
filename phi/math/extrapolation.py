"""
Defines standard extrapolations.

Extrapolations are used for padding tensors and sampling coordinates lying outside the tensor bounds.
"""
from typing import Union

from . import _functions as math
from .backend import choose_backend
from ._track import SparseLinearOperation, ShiftLinOp
from ._shape import Shape
from ._tensors import Tensor, NativeTensor, CollapsedTensor, TensorStack, wrap


class Extrapolation:

    def __init__(self, pad_rank):
        """
        Extrapolations are used to determine values of grids or other structures outside the sampled bounds.

        They play a vital role in padding and sampling.

        Args:
          pad_rank: low-ranking extrapolations are handled first during mixed-extrapolation padding.
        The typical order is periodic=1, boundary=2, symmetric=3, reflect=4, constant=5.

        Returns:

        """
        self.pad_rank = pad_rank

    def to_dict(self) -> dict:
        """
        Serialize this extrapolation to a dictionary that is serializable (JSON-writable).
        
        Use `from_dict()` to restore the Extrapolation object.
        """
        raise NotImplementedError()

    def spatial_gradient(self) -> 'Extrapolation':
        """Returns the extrapolation for the spatial spatial_gradient of a tensor/field with this extrapolation."""
        raise NotImplementedError()

    def pad(self, value: Tensor, widths: dict) -> Tensor:
        """
        Pads a tensor using values from self.pad_values()

        Args:
          value: tensor to be padded
          widths: name: str -> (lower: int, upper: int)}
          value: Tensor: 
          widths: dict: 

        Returns:

        """
        for dim in widths:
            values = []
            if widths[dim][False] > 0:
                values.append(self.pad_values(value, widths[dim][False], dim, False))
            values.append(value)
            if widths[dim][True] > 0:
                values.append(self.pad_values(value, widths[dim][True], dim, True))
            value = math.concat(values, dim)
        return value

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        """
        Determines the values with which the given tensor would be padded at the specified using this extrapolation.

        Args:
          value: tensor to be padded
          width: number of cells to pad perpendicular to the face. Must be larger than zero.
          dimension: axis in which to pad
          upper_edge: True for upper edge, False for lower edge
          value: Tensor: 
          width: int: 
          dimension: str: 
          upper_edge: bool: 

        Returns:
          tensor that can be concatenated to value for padding

        """
        raise NotImplementedError()

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        """
        If is_copy_pad, transforms outsider coordinates to point to the index from which the value should be copied.
        
        Otherwise, the grid tensor is assumed to hold the correct boundary values for this extrapolation at the edge.
        Coordinates are then snapped to the valid index range.
        This is the default implementation.

        Args:
          coordinates: integer coordinates in index space
          shape: tensor shape
          coordinates: Tensor: 
          shape: Shape: 

        Returns:
          transformed coordinates

        """
        return math.clip(coordinates, 0, math.wrap(shape.spatial - 1, 'vector'))

    @property
    def is_copy_pad(self):
        """:return: True if all pad values are copies of existing values in the tensor to be padded"""
        return False

    @property
    def native_grid_sample_mode(self) -> Union[str, None]:
        return None

    def __getitem__(self, item):
        return self


class ConstantExtrapolation(Extrapolation):
    """
    Extrapolate with a constant value.
    """

    def __init__(self, value: Tensor):
        Extrapolation.__init__(self, 5)
        self.value = wrap(value)
        """ Extrapolation value """

    def __repr__(self):
        return repr(self.value)

    def to_dict(self) -> dict:
        return {'type': 'constant', 'value': self.value.numpy()}

    def spatial_gradient(self):
        return ZERO

    def pad(self, value: Tensor, widths: dict):
        """
        Pads a tensor using CONSTANT values

        Args:
          value: tensor to be padded
          widths: name: str -> (lower: int, upper: int)}
          value: Tensor: 
          widths: dict: 

        Returns:

        """
        value = value.__simplify__()
        if isinstance(value, NativeTensor):
            native = value.native()
            ordered_pad_widths = value.shape.order(widths, default=(0, 0))
            backend = choose_backend(native)
            result_tensor = backend.pad(native, ordered_pad_widths, 'constant', self.value.native())
            new_shape = value.shape.with_sizes(backend.staticshape(result_tensor))
            return NativeTensor(result_tensor, new_shape)
        elif isinstance(value, CollapsedTensor):
            if value._inner.shape.volume > 1 or not math.all_available(self.value, value) or not math.close(self.value, value._inner):  # .inner should be safe after __simplify__
                return self.pad(value._expand(), widths)
            else:  # Stays constant value, only extend shape
                new_sizes = []
                for size, dim, dim_type in value.shape.dimensions:
                    if dim not in widths:
                        new_sizes.append(size)
                    else:
                        delta = sum(widths[dim]) if isinstance(widths[dim], (tuple, list)) else 2 * widths[dim]
                        new_sizes.append(size + int(delta))
                new_shape = value.shape.with_sizes(new_sizes)
                return CollapsedTensor(value._inner, new_shape)
        # elif isinstance(value, SparseLinearOperation):
        #     return pad_operator(value, pad_width, mode)
        elif isinstance(value, TensorStack):
            if not value.requires_broadcast:
                return self.pad(value._cache(), widths)
            inner_widths = {dim: w for dim, w in widths.items() if dim != value.stack_dim_name}
            tensors = [self.pad(t, inner_widths) for t in value.tensors]
            return TensorStack(tensors, value.stack_dim_name, value.stack_dim_type)
        elif isinstance(value, SparseLinearOperation):
            (row, col), data = choose_backend(value.dependency_matrix).coordinates(value.dependency_matrix, unstack_coordinates=True)
            assert len(value.shape) == 2  # TODO nd
            y = row // value.shape[1]
            dy0, dy1 = widths[value.shape.names[0]]
            dx0, dx1 = widths[value.shape.names[1]]
            padded_row = row + dy0 * (value.shape[1] + dx0 + dx1) + dx0 * (y + 1) + dx1 * y
            new_sizes = list(value.shape.sizes)
            for i, dim in enumerate(value.shape.names):
                new_sizes[i] += sum(widths[dim])
            new_shape = value.shape.with_sizes(new_sizes)
            padded_matrix = choose_backend(padded_row, col, data).sparse_tensor((padded_row, col), data, shape=(new_shape.volume, value.dependency_matrix.shape[1]))
            return SparseLinearOperation(value.source, padded_matrix, new_shape)
        elif isinstance(value, ShiftLinOp):
            assert self.is_zero()
            lower = {dim: -lo for dim, (lo, _) in widths.items()}
            return value.shift(lower, lambda v: self.pad(v, widths), value.shape.after_pad(widths))
        else:
            raise NotImplementedError()

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        raise NotImplementedError()
        return math.zeros()

    def __eq__(self, other):
        return isinstance(other, ConstantExtrapolation) and math.close(self.value, other.value)

    def __hash__(self):
        return hash(self.__class__)

    def is_zero(self):
        return self == ZERO

    def is_one(self):
        return self == ONE

    @property
    def native_grid_sample_mode(self) -> Union[str, None]:
        return 'zeros' if self.is_zero() else None

    def __add__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value + other.value)
        elif self.is_zero():
            return other
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value - other.value)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(other.value - self.value)
        elif self.is_zero():
            return other
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value * other.value)
        elif self.is_one():
            return other
        elif self.is_zero():
            return self
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value / other.value)
        elif self.is_zero():
            return self
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(other.value / self.value)
        elif self.is_one():
            return other
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value < other.value)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value > other.value)
        else:
            return NotImplemented

    def __abs__(self):
        return ConstantExtrapolation(abs(self.value))

    def _op1(self, operator):
        return ConstantExtrapolation(self.value._op1(operator))


class _CopyExtrapolation(Extrapolation):

    @property
    def is_copy_pad(self):
        return True

    def to_dict(self) -> dict:
        return {'type': repr(self)}

    def pad(self, value: Tensor, widths: dict) -> Tensor:
        value = value.__simplify__()
        if isinstance(value, NativeTensor):
            native = value.native()
            ordered_pad_widths = value.shape.order(widths, default=(0, 0))
            result_tensor = choose_backend(native).pad(native, ordered_pad_widths, repr(self))
            if result_tensor is NotImplemented:
                return Extrapolation.pad(self, value, widths)
            new_shape = value.shape.with_sizes(result_tensor.shape)
            return NativeTensor(result_tensor, new_shape)
        elif isinstance(value, CollapsedTensor):
            inner = value._inner  # should be fine after __simplify__
            inner_widths = {dim: w for dim, w in widths.items() if dim in inner.shape}
            if len(inner_widths) > 0:
                inner = self.pad(inner, widths)
            new_sizes = []
            for size, dim, dim_type in value.shape.dimensions:
                if dim not in widths:
                    new_sizes.append(size)
                else:
                    delta = sum(widths[dim]) if isinstance(widths[dim], (tuple, list)) else 2 * widths[dim]
                    new_sizes.append(size + int(delta))
            new_shape = value.shape.with_sizes(new_sizes)
            return CollapsedTensor(inner, new_shape)
        # elif isinstance(value, SparseLinearOperation):
        #     return pad_operator(value, widths, mode)
        elif isinstance(value, TensorStack):
            if not value.requires_broadcast:
                return self.pad(value._cache(), widths)
            inner_widths = {dim: w for dim, w in widths.items() if dim != value.stack_dim_name}
            tensors = [self.pad(t, inner_widths) for t in value.tensors]
            return TensorStack(tensors, value.stack_dim_name, value.stack_dim_type)
        elif isinstance(value, ShiftLinOp):
            return self._pad_linear_operation(value, widths)
        else:
            raise NotImplementedError(f'{type(value)} not supported')

    def _pad_linear_operation(self, value: ShiftLinOp, widths: dict) -> ShiftLinOp:
        raise NotImplementedError()

    @property
    def native_grid_sample_mode(self) -> Union[str, None]:
        return str(self)

    def __eq__(self, other):
        return type(other) == type(self)

    def __hash__(self):
        return hash(self.__class__)

    def _op(self, other, op):
        if type(other) == type(self):
            return self
        elif isinstance(other, Extrapolation) and not isinstance(other, _CopyExtrapolation):
            op = getattr(other, op.__name__)
            return op(self)
        else:
            return NotImplemented

    def __add__(self, other):
        return self._op(other, ConstantExtrapolation.__add__)

    def __mul__(self, other):
        return self._op(other, ConstantExtrapolation.__mul__)

    def __sub__(self, other):
        return self._op(other, ConstantExtrapolation.__rsub__)

    def __truediv__(self, other):
        return self._op(other, ConstantExtrapolation.__rtruediv__)

    def __lt__(self, other):
        return self._op(other, ConstantExtrapolation.__gt__)

    def __gt__(self, other):
        return self._op(other, ConstantExtrapolation.__lt__)

    def __neg__(self):
        return self  # assume also applied to values

    def __abs__(self):
        return self  # assume also applied to values

    def _op1(self, operator):
        return self  # assume also applied to values


class _BoundaryExtrapolation(_CopyExtrapolation):
    """Uses the closest defined value for points lying outside the defined region."""

    _CACHED_LOWER_MASKS = {}
    _CACHED_UPPER_MASKS = {}

    def __repr__(self):
        return 'boundary'

    def spatial_gradient(self):
        return ZERO

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        if upper_edge:
            edge = value[{dimension: slice(-1, None)}]
        else:
            edge = value[{dimension: slice(1)}]
        return math.concat([edge] * width, dimension)

    def _pad_linear_operation(self, value: ShiftLinOp, widths: dict) -> ShiftLinOp:
        """
        *Warning*:
        This implementation discards corners, i.e. values that lie outside the original tensor in more than one dimension.
        These are typically sliced off in differential operators. Corners are instead assigned the value 0.
        To take corners into account, call pad() for each axis individually. This is inefficient with ShiftLinOp.

        Args:
          value: ShiftLinOp: 
          widths: dict: 

        Returns:

        """
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        result = value.shift(lower, lambda v: ZERO.pad(v, widths), value.shape.after_pad(widths))  # inner values  ~half the computation time
        for bound_dim, (bound_lo, bound_hi) in widths.items():
            for i in range(bound_lo):  # i=0 means outer
                # this sets corners to 0
                lower = {dim: -i if dim == bound_dim else -lo for dim, (lo, _) in widths.items()}
                mask = self._lower_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, lambda v: self.pad(v, widths) * mask, result.shape)
                result += boundary
            for i in range(bound_hi):
                lower = {dim: i - lo - hi if dim == bound_dim else -lo for dim, (lo, hi) in widths.items()}
                mask = self._upper_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, lambda v: self.pad(v, widths) * mask, result.shape)  # ~ half the computation time
                result += boundary  # this does basically nothing if value is the identity
        return result

    def _lower_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        key = (shape, tuple(widths.keys()), tuple(widths.values()), bound_dim, bound_lo, bound_hi, i)
        if key in _BoundaryExtrapolation._CACHED_LOWER_MASKS:
            result = math.tensor(_BoundaryExtrapolation._CACHED_LOWER_MASKS[key])
            _BoundaryExtrapolation._CACHED_LOWER_MASKS[key] = result
            return result
        else:
            mask = ZERO.pad(math.zeros(shape), {bound_dim: (bound_lo - i - 1, 0)})
            mask = ONE.pad(mask, {bound_dim: (1, 0)})
            mask = ZERO.pad(mask, {dim: (i, bound_hi) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
            _BoundaryExtrapolation._CACHED_LOWER_MASKS[key] = mask
            return mask

    def _upper_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        key = (shape, tuple(widths.keys()), tuple(widths.values()), bound_dim, bound_lo, bound_hi, i)
        if key in _BoundaryExtrapolation._CACHED_UPPER_MASKS:
            result = math.tensor(_BoundaryExtrapolation._CACHED_UPPER_MASKS[key])
            _BoundaryExtrapolation._CACHED_UPPER_MASKS[key] = result
            return result
        else:
            mask = ZERO.pad(math.zeros(shape), {bound_dim: (0, bound_hi - i - 1)})
            mask = ONE.pad(mask, {bound_dim: (0, 1)})
            mask = ZERO.pad(mask, {dim: (bound_lo, i) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
            _BoundaryExtrapolation._CACHED_UPPER_MASKS[key] = mask
            return mask


class _PeriodicExtrapolation(_CopyExtrapolation):
    def __repr__(self):
        return 'periodic'

    def spatial_gradient(self):
        return self

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        return coordinates % shape.spatial

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        if upper_edge:
            return value[{dimension: slice(width)}]
        else:
            return value[{dimension: slice(-width, None)}]

    def _pad_linear_operation(self, value: ShiftLinOp, widths: dict) -> ShiftLinOp:
        if value.shape.get_size(tuple(widths.keys())) != value.source.shape.get_size(tuple(widths.keys())):
            raise NotImplementedError("Periodicity does not match input: %s but input has %s. This can happen when padding an already padded or sliced tensor." % (value.shape.only(tuple(widths.keys())), value.source.shape.only(tuple(widths.keys()))))
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        return value.shift(lower, lambda v: self.pad(v, widths), value.shape.after_pad(widths))


class _SymmetricExtrapolation(_CopyExtrapolation):
    """Mirror with the boundary value occurring twice."""

    def __repr__(self):
        return 'symmetric'

    def spatial_gradient(self):
        return -self

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        coordinates = coordinates % (2 * shape)
        return ((2 * shape - 1) - abs((2 * shape - 1) - 2 * coordinates)) // 2

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        raise NotImplementedError()
        raise NotImplementedError()  # only used by PyTorch which does not support ::-1 axis flips
        dims = range(math.ndims(value))
        for dim in dims:
            pad_lower, pad_upper = pad_width[dim]
            if pad_lower == 0 and pad_upper == 0:
                continue  # Nothing to pad
            top_rows = value[
                tuple([slice(value.shape[dim] - pad_upper, None) if d == dim else slice(None) for d in dims])]
            bottom_rows = value[tuple([slice(None, pad_lower) if d == dim else slice(None) for d in dims])]
            top_rows = math.flip_axis(top_rows, dim)
            bottom_rows = math.flip_axis(bottom_rows, dim)
            value = math.concat([bottom_rows, value, top_rows], axis=dim)
        return value


class _ReflectExtrapolation(_CopyExtrapolation):
    """Mirror of inner elements. The boundary value is not duplicated."""

    def __repr__(self):
        return 'reflect'

    def spatial_gradient(self):
        return -self

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        if upper_edge:
            return value[{dimension: slice(-1-width, -1)}].flip(dimension)
        else:
            return value[{dimension: slice(1, width+1)}].flip(dimension)

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        coordinates = coordinates % (2 * shape - 2)
        return (shape - 1) - math.abs((shape - 1) - coordinates)


ZERO = ConstantExtrapolation(0)
""" Extrapolates with the constant value 0 """
ONE = ConstantExtrapolation(1)
""" Extrapolates with the constant value 1 """
PERIODIC = _PeriodicExtrapolation(1)
""" Extends a grid by tiling it """
BOUNDARY = _BoundaryExtrapolation(2)
""" Extends a grid with its edge values. The value of a point lying outside the grid is determined by the closest grid value(s). """
SYMMETRIC = _SymmetricExtrapolation(3)
""" Extends a grid by tiling it. Every other copy of the grid is flipped. Edge values occur twice per seam. """
REFLECT = _ReflectExtrapolation(4)
""" Like SYMMETRIC but the edge values are not copied and only occur once per seam. """


def combine_sides(extrapolations: dict) -> Extrapolation:
    """
    Create a single Extrapolation object that uses different extrapolations for different sides of a box.

    Args:
      extrapolations: dict mapping dim: str -> extrapolation or (lower, upper)
      extrapolations: dict: 

    Returns:
      single extrapolation

    """
    values = set()
    for ext in extrapolations.values():
        if isinstance(ext, Extrapolation):
            values.add(ext)
        else:
            values.add(ext[0])
            values.add(ext[1])
    if len(values) == 1:
        return next(iter(values))
    else:
        return _MixedExtrapolation(extrapolations)


class _MixedExtrapolation(Extrapolation):

    def __init__(self, extrapolations: dict):
        """
        A mixed extrapolation uses different extrapolations for different sides.

        Args:
          extrapolations: axis: str -> (lower: Extrapolation, upper: Extrapolation) or Extrapolation
        """
        Extrapolation.__init__(self, None)
        self.ext = {ax: (e, e) if isinstance(e, Extrapolation) else tuple(e) for ax, e in extrapolations.items()}

    def to_dict(self) -> dict:
        return {
            'type': 'mixed',
            'dims': {ax: (es[0].to_dict(), es[1].to_dict()) for ax, es in self.ext.items()}
        }

    def __eq__(self, other):
        if isinstance(other, _MixedExtrapolation):
            return self.ext == other.ext
        else:
            simplified = combine_sides(self.ext)
            if not isinstance(simplified, _MixedExtrapolation):
                return simplified == other
            else:
                return False

    def __hash__(self):
        simplified = combine_sides(self.ext)
        if not isinstance(simplified, _MixedExtrapolation):
            return hash(simplified)
        else:
            return hash(frozenset(self.ext.items()))

    def __repr__(self):
        return repr(self.ext)

    def spatial_gradient(self) -> Extrapolation:
        return combine_sides({ax: (es[0].spatial_gradient(), es[1].spatial_gradient())
                              for ax, es in self.ext.items()})

    def pad(self, value: Tensor, widths: dict) -> Tensor:
        """
        Pads a tensor using mixed values

        Args:
          value: tensor to be padded
          widths: name: str -> (lower: int, upper: int)}
          value: Tensor: 
          widths: dict: 

        Returns:

        """
        extrapolations = set(sum(self.ext.values(), ()))
        extrapolations = tuple(sorted(extrapolations, key=lambda e: e.pad_rank))
        for ext in extrapolations:
            ext_widths = {ax: (l if self.ext[ax][0] == ext else 0, u if self.ext[ax][1] == ext else 0)
                          for ax, (l, u) in widths.items()}
            value = ext.pad(value, ext_widths)
        return value

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        extrap: Extrapolation = self.ext[dimension][upper_edge]
        return extrap.pad_values(value, width, dimension, upper_edge)

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        coordinates = coordinates.vector.unstack()
        assert len(self.ext) == len(shape.spatial) == len(coordinates)
        result = []
        for dim, dim_coords in zip(shape.spatial.unstack(), coordinates):
            dim_extrapolations = self.ext[dim.name]
            if dim_extrapolations[0] == dim_extrapolations[1]:
                result.append(dim_extrapolations[0].transform_coordinates(dim_coords, dim))
            else:  # separate boundary for lower and upper face
                lower = dim_extrapolations[0].transform_coordinates(dim_coords, dim)
                upper = dim_extrapolations[1].transform_coordinates(dim_coords, dim)
                result.append(math.where(dim_coords <= 0, lower, upper))
        if 'vector' in result[0].shape:
            return math.concat(result, 'vector')
        else:
            return math.channel_stack(result, 'vector')

    def __getitem__(self, item):
        dim, face = item
        return self.ext[dim][face]

    def __add__(self, other):
        return self._op2(other, lambda e1, e2: e1 + e2)

    def __radd__(self, other):
        return self._op2(other, lambda e1, e2: e2 + e1)

    def __sub__(self, other):
        return self._op2(other, lambda e1, e2: e1 - e2)

    def __rsub__(self, other):
        return self._op2(other, lambda e1, e2: e2 - e1)

    def __mul__(self, other):
        return self._op2(other, lambda e1, e2: e1 * e2)

    def __rmul__(self, other):
        return self._op2(other, lambda e1, e2: e2 * e1)

    def _op2(self, other, operator):
        if isinstance(other, _MixedExtrapolation):
            assert self.ext.keys() == other.ext.keys()
            return combine_sides({ax: (operator(lo, other.ext[ax][False]), operator(hi, other.ext[ax][True])) for ax, (lo, hi) in self.ext.items()})
        else:
            return combine_sides({ax: (operator(lo, other), operator(hi, other)) for ax, (lo, hi) in self.ext.items()})


def from_dict(dictionary: dict) -> Extrapolation:
    """
    Loads an `Extrapolation` object from a dictionary that was created using `Extrapolation.to_dict()`.

    Args:
        dictionary: serializable dictionary holding all extrapolation properties

    Returns:
        Loaded extrapolation
    """
    etype = dictionary['type']
    if etype == 'constant':
        return ConstantExtrapolation(dictionary['value'])
    elif etype == 'periodic':
        return PERIODIC
    elif etype == 'boundary':
        return BOUNDARY
    elif etype == 'symmetric':
        return SYMMETRIC
    elif etype == 'reflect':
        return REFLECT
    else:
        raise ValueError(dictionary)

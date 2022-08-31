"""
Extrapolations are used for padding tensors and sampling coordinates lying outside the tensor bounds.
Standard extrapolations are listed as global variables in this module.

Extrapolations are an important part of sampled fields such as grids.
See the documentation at https://tum-pbs.github.io/PhiFlow/Fields.html#extrapolations .
"""
import warnings
from typing import Union, Dict, Callable, Tuple

from phi.math.backend._backend import get_spatial_derivative_order
from .backend import choose_backend
from ._shape import Shape, channel, spatial
from ._magic_ops import concat, stack
from ._tensors import Tensor, NativeTensor, CollapsedTensor, TensorStack, wrap
from . import _ops as math  # TODO this executes _ops.py, can we avoid this?


class Extrapolation:
    """
    Extrapolations are used to determine values of grids or other structures outside the sampled bounds.
    They play a vital role in padding and sampling.
    """

    def __init__(self, pad_rank):
        """
        Args:
            pad_rank: low-ranking extrapolations are handled first during mixed-extrapolation padding.
                The typical order is periodic=1, boundary=2, symmetric=3, reflect=4, constant=5.
        """
        self.pad_rank = pad_rank

    def to_dict(self) -> dict:
        """
        Serialize this extrapolation to a dictionary that is serializable (JSON-writable).
        
        Use `from_dict()` to restore the Extrapolation object.
        """
        raise NotImplementedError()

    def spatial_gradient(self) -> 'Extrapolation':
        """
        Returns the extrapolation for the spatial gradient of a tensor/field with this extrapolation.

        Returns:
            `Extrapolation` or `NotImplemented`
        """
        raise NotImplementedError()

    def valid_outer_faces(self, dim) -> tuple:
        """ `(lower: bool, upper: bool)` indicating whether the values sampled at the outer-most faces of a staggered grid with this extrapolation are valid, i.e. need to be stored and are not redundant. """
        raise NotImplementedError()

    @property
    def is_flexible(self) -> bool:
        """
        Whether the outside values are affected by the inside values.
        Only `True` if there are actual outside values, i.e. PERIODIC is not flexible.

        This property is important for pressure solves to determine whether the total divergence is fixed or can be adjusted during the solve.
        """
        raise NotImplementedError()

    def pad(self, value: Tensor, widths: dict, **kwargs) -> Tensor:
        """
        Pads a tensor using values from `self.pad_values()`.

        If `value` is a linear tracer, assume pad_values() to produce constant values, independent of `value`.
        To change this behavior, override this method.

        Args:
            value: `Tensor` to be padded
            widths: `dict` mapping `dim: str -> (lower: int, upper: int)`
            kwargs: Additional keyword arguments for padding, passed on to `pad_values()`.

        Returns:
            Padded `Tensor`
        """
        from phi.math._functional import ShiftLinTracer
        if isinstance(value, ShiftLinTracer):
            lower = {dim: -lo for dim, (lo, _) in widths.items()}
            return value.shift(lower, new_shape=value.shape.after_pad(widths), val_fun=lambda v: ZERO.pad(v, widths, **kwargs), bias_fun=lambda b: self.pad(b, widths, **kwargs))
        already_padded = {}
        for dim, width in widths.items():
            assert (w > 0 for w in width), "Negative widths not allowed in Extrapolation.pad(). Use math.pad() instead."
            values = []
            if width[False] > 0:
                values.append(self.pad_values(value, width[False], dim, False, already_padded=already_padded, **kwargs))
            values.append(value)
            if width[True] > 0:
                values.append(self.pad_values(value, width[True], dim, True, already_padded=already_padded, **kwargs))
            value = concat(values, dim)
            already_padded[dim] = width
        return value

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        """
        Determines the values with which the given tensor would be padded at the specified using this extrapolation.

        Args:
            value: `Tensor` to be padded.
            width: `int > 0`: Number of cells to pad along `dimension`.
            dim: Dimension name as `str`.
            upper_edge: `True` for upper edge, `False` for lower edge.

        Returns:
            `Tensor` that can be concatenated to `value` along `dimension`
        """
        raise NotImplementedError()

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        """
        If `self.is_copy_pad`, transforms outside coordinates to the index from which the value is copied.
        
        Otherwise, the grid tensor is assumed to hold the correct boundary values for this extrapolation at the edge.
        Coordinates are then snapped to the valid index range.
        This is the default implementation.

        Args:
            coordinates: integer coordinates in index space
            shape: tensor shape

        Returns:
            Transformed coordinates
        """
        res = shape.spatial[coordinates.shape.get_item_names('vector')] if 'vector' in coordinates.shape and coordinates.shape.get_item_names('vector') else shape.spatial
        return math.clip(coordinates, 0, math.wrap(res - 1, channel('vector')))

    def is_copy_pad(self, dim: str, upper_edge: bool):
        """:return: True if all pad values are copies of existing values in the tensor to be padded"""
        return False

    @property
    def native_grid_sample_mode(self) -> Union[str, None]:
        return None

    def shortest_distance(self, start: Tensor, end: Tensor, domain_size: Tensor):
        """
        Computes the shortest distance between two points.
        Both points are assumed to lie within the domain

        Args:
            start: Start position.
            end: End position.
            domain_size: Domain side lengths as vector.

        Returns:
            Shortest distance from `start` to `end`.
        """
        return end - start

    def __getitem__(self, item):
        return self

    def __abs__(self):
        raise NotImplementedError(self.__class__)

    def __neg__(self):
        raise NotImplementedError(self.__class__)

    def __add__(self, other):
        raise NotImplementedError(self.__class__)

    def __radd__(self, other):
        raise NotImplementedError(self.__class__)

    def __sub__(self, other):
        raise NotImplementedError(self.__class__)

    def __rsub__(self, other):
        raise NotImplementedError(self.__class__)

    def __mul__(self, other):
        raise NotImplementedError(self.__class__)

    def __rmul__(self, other):
        raise NotImplementedError(self.__class__)

    def __truediv__(self, other):
        raise NotImplementedError(self.__class__)

    def __rtruediv__(self, other):
        raise NotImplementedError(self.__class__)


class ConstantExtrapolation(Extrapolation):
    """
    Extrapolate with a constant value.
    """

    def __init__(self, value: Tensor or float):
        Extrapolation.__init__(self, 5)
        self.value = wrap(value)
        """ Extrapolation value """

    def __repr__(self):
        return repr(self.value)

    def to_dict(self) -> dict:
        return {'type': 'constant', 'value': self.value.numpy()}

    def __value_attrs__(self):
        return 'value',

    def __getitem__(self, item):
        return ConstantExtrapolation(self.value[item])

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'ConstantExtrapolation':
        if all(isinstance(v, ConstantExtrapolation) for v in values):
            return ConstantExtrapolation(stack([v.value for v in values], dim, **kwargs))
        else:
            return NotImplemented

    def spatial_gradient(self):
        return ZERO

    def valid_outer_faces(self, dim) -> tuple:
        return False, False

    @property
    def is_flexible(self) -> bool:
        return False

    def pad(self, value: Tensor, widths: dict, **kwargs):
        """
        Pads a tensor using CONSTANT values

        Args:
          value: tensor to be padded
          widths: name: str -> (lower: int, upper: int)}
          value: Tensor: 
          widths: dict: 

        Returns:

        """
        derivative = get_spatial_derivative_order()
        pad_value = self.value if derivative == 0 else math.zeros()
        value = value._simplify()
        if isinstance(value, NativeTensor):
            native = value._native
            ordered_pad_widths = order_by_shape(value.shape, widths, default=(0, 0))
            backend = choose_backend(native)
            result_tensor = backend.pad(native, ordered_pad_widths, 'constant', pad_value.native())
            new_shape = value.shape.with_sizes(backend.staticshape(result_tensor))
            return NativeTensor(result_tensor, new_shape)
        elif isinstance(value, CollapsedTensor):
            if value._inner.shape.volume > 1 or not math.all_available(pad_value, value) or not math.close(pad_value, value._inner):  # .inner should be safe after _simplify
                return self.pad(value._cache(), widths)
            else:  # Stays constant value, only extend shape
                new_sizes = []
                for size, dim, *_ in value.shape._dimensions:
                    if dim not in widths:
                        new_sizes.append(size)
                    else:
                        delta = sum(widths[dim]) if isinstance(widths[dim], (tuple, list)) else 2 * widths[dim]
                        new_sizes.append(size + int(delta))
                new_shape = value.shape.with_sizes(new_sizes)
                return CollapsedTensor(value._inner, new_shape)
        elif isinstance(value, TensorStack):
            if not value.requires_broadcast:
                return self.pad(value._cache(), widths)
            inner_widths = {dim: w for dim, w in widths.items() if dim != value.stack_dim_name}
            tensors = [self.pad(t, inner_widths) for t in value.dimension(value.stack_dim.name)]
            return TensorStack(tensors, value.stack_dim)
        else:
            return Extrapolation.pad(self, value, widths, **kwargs)

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        shape = value.shape.after_gather({dim: slice(0, width)})
        return math.expand(self.value, shape)

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

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value - other.value)
        elif self.is_zero():
            return -other
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

    __rmul__ = __mul__

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

    def __neg__(self):
        return ConstantExtrapolation(-self.value)


class _CopyExtrapolation(Extrapolation):

    def is_copy_pad(self, dim: str, upper_edge: bool):
        return True

    def to_dict(self) -> dict:
        return {'type': repr(self)}

    def __value_attrs__(self):
        return ()

    def valid_outer_faces(self, dim):
        return True, True

    def pad(self, value: Tensor, widths: dict, **kwargs) -> Tensor:
        value = value._simplify()
        from phi.math._functional import ShiftLinTracer
        if isinstance(value, NativeTensor):
            native = value._native
            ordered_pad_widths = order_by_shape(value.shape, widths, default=(0, 0))
            result_tensor = choose_backend(native).pad(native, ordered_pad_widths, repr(self))
            if result_tensor is NotImplemented:
                return Extrapolation.pad(self, value, widths)
            new_shape = value.shape.with_sizes(result_tensor.shape)
            return NativeTensor(result_tensor, new_shape)
        elif isinstance(value, CollapsedTensor):
            inner = value._inner  # should be fine after _simplify
            inner_widths = {dim: w for dim, w in widths.items() if dim in inner.shape}
            if len(inner_widths) > 0:
                inner = self.pad(inner, widths)
            new_sizes = []
            for size, dim, *_ in value.shape._dimensions:
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
            tensors = [self.pad(t, inner_widths) for t in value.dimension(value.stack_dim.name)]
            return TensorStack(tensors, value.stack_dim)
        elif isinstance(value, ShiftLinTracer):
            return self._pad_linear_tracer(value, widths)
        else:
            raise NotImplementedError(f'{type(value)} not supported')

    def _pad_linear_tracer(self, value, widths: dict):
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
        if isinstance(other, ConstantExtrapolation):  # some operations can be handled by ConstantExtrapolation, e.g. * 0
            op = getattr(other, op.__name__)
            return op(self)
        else:
            return NotImplemented

    def __abs__(self):
        return self  # assume also applied to values

    def __neg__(self):
        return self  # assume also applied to values

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

    def __lt__(self, other):
        return self._op(other, ConstantExtrapolation.__gt__)

    def __gt__(self, other):
        return self._op(other, ConstantExtrapolation.__lt__)


class _BoundaryExtrapolation(_CopyExtrapolation):
    """Uses the closest defined value for points lying outside the defined region."""

    _CACHED_LOWER_MASKS = {}
    _CACHED_UPPER_MASKS = {}

    def __repr__(self):
        return 'boundary'

    def spatial_gradient(self):
        return ZERO

    @property
    def is_flexible(self) -> bool:
        return True

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        if upper_edge:
            edge = value[{dim: slice(-1, None)}]
        else:
            edge = value[{dim: slice(1)}]
        return concat([edge] * width, value.shape[dim])

    def _pad_linear_tracer(self, value: 'ShiftLinTracer', widths: dict) -> 'ShiftLinTracer':
        """
        *Warning*:
        This implementation discards corners, i.e. values that lie outside the original tensor in more than one dimension.
        These are typically sliced off in differential operators. Corners are instead assigned the value 0.
        To take corners into account, call pad() for each axis individually. This is inefficient with ShiftLinTracer.

        Args:
          value: ShiftLinTracer:
          widths: dict: 

        Returns:

        """
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        result = value.shift(lower, new_shape=value.shape.after_pad(widths), val_fun=lambda v: ZERO.pad(v, widths), bias_fun=lambda b: ZERO.pad(b, widths))  # inner values  ~half the computation time
        for bound_dim, (bound_lo, bound_hi) in widths.items():
            for i in range(bound_lo):  # i=0 means outer
                # this sets corners to 0
                lower = {dim: -i if dim == bound_dim else -lo for dim, (lo, _) in widths.items()}
                mask = self._lower_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))
                result += boundary
            for i in range(bound_hi):
                lower = {dim: i - lo - hi if dim == bound_dim else -lo for dim, (lo, hi) in widths.items()}
                mask = self._upper_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))  # ~ half the computation time
                result += boundary  # this does basically nothing if value is the identity
        return result

    def _lower_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        # key = (shape, tuple(widths.keys()), tuple(widths.values()), bound_dim, bound_lo, bound_hi, i)
        # if key in _BoundaryExtrapolation._CACHED_LOWER_MASKS:
        #     result = math.tensor(_BoundaryExtrapolation._CACHED_LOWER_MASKS[key])
        #     _BoundaryExtrapolation._CACHED_LOWER_MASKS[key] = result
        #     return result
        # else:
            mask = ZERO.pad(math.zeros(shape), {bound_dim: (bound_lo - i - 1, 0)})
            mask = ONE.pad(mask, {bound_dim: (1, 0)})
            mask = ZERO.pad(mask, {dim: (i, bound_hi) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
            # _BoundaryExtrapolation._CACHED_LOWER_MASKS[key] = mask
            return mask

    def _upper_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        # key = (shape, tuple(widths.keys()), tuple(widths.values()), bound_dim, bound_lo, bound_hi, i)
        # if key in _BoundaryExtrapolation._CACHED_UPPER_MASKS:
        #     result = math.tensor(_BoundaryExtrapolation._CACHED_UPPER_MASKS[key])
        #     _BoundaryExtrapolation._CACHED_UPPER_MASKS[key] = result
        #     return result
        # else:
            mask = ZERO.pad(math.zeros(shape), {bound_dim: (0, bound_hi - i - 1)})
            mask = ONE.pad(mask, {bound_dim: (0, 1)})
            mask = ZERO.pad(mask, {dim: (bound_lo, i) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
            # _BoundaryExtrapolation._CACHED_UPPER_MASKS[key] = mask
            return mask


class _PeriodicExtrapolation(_CopyExtrapolation):
    def __repr__(self):
        return 'periodic'

    def spatial_gradient(self):
        return self

    def valid_outer_faces(self, dim):
        return True, False

    @property
    def is_flexible(self) -> bool:
        return False

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        return coordinates % shape.spatial

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        if upper_edge:
            return value[{dim: slice(width)}]
        else:
            return value[{dim: slice(-width, None)}]

    def _pad_linear_tracer(self, value: 'ShiftLinTracer', widths: dict) -> 'ShiftLinTracer':
        if value.shape.get_sizes(tuple(widths.keys())) != value.source.shape.get_sizes(tuple(widths.keys())):
            raise NotImplementedError("Periodicity does not match input: %s but input has %s. This can happen when padding an already padded or sliced tensor." % (value.shape.only(tuple(widths.keys())), value.source.shape.only(tuple(widths.keys()))))
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        return value.shift(lower, new_shape=value.shape.after_pad(widths), val_fun=lambda v: self.pad(v, widths), bias_fun=lambda b: ZERO.pad(b, widths))

    def shortest_distance(self, start: Tensor, end: Tensor, domain_size: Tensor):
        dx = end - start
        return (dx + domain_size / 2) % domain_size - domain_size / 2


class _SymmetricExtrapolation(_CopyExtrapolation):
    """Mirror with the boundary value occurring twice."""

    def __repr__(self):
        return 'symmetric'

    def spatial_gradient(self):
        return -self

    @property
    def is_flexible(self) -> bool:
        return True

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        coordinates = coordinates % (2 * shape)
        return ((2 * shape - 1) - abs((2 * shape - 1) - 2 * coordinates)) // 2

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        if upper_edge:
            return value[{dim: slice(-width, None)}].flip(dim)
        else:
            return value[{dim: slice(0, width)}].flip(dim)


class _ReflectExtrapolation(_CopyExtrapolation):
    """Mirror of inner elements. The boundary value is not duplicated."""

    def __repr__(self):
        return 'reflect'

    def spatial_gradient(self):
        return -self

    @property
    def is_flexible(self) -> bool:
        return True

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        if upper_edge:
            return value[{dim: slice(-1-width, -1)}].flip(dim)
        else:
            return value[{dim: slice(1, width+1)}].flip(dim)

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        coordinates = coordinates % (2 * shape - 2)
        return (shape - 1) - math.abs_((shape - 1) - coordinates)


class _NoExtrapolation(Extrapolation):  # singleton
    def to_dict(self) -> dict:
        return {'type': 'none'}

    def pad(self, value: Tensor, widths: dict, **kwargs) -> Tensor:
        return value

    def spatial_gradient(self) -> 'Extrapolation':
        return self

    def valid_outer_faces(self, dim):
        return True, True

    def __value_attrs__(self):
        return ()

    @property
    def is_flexible(self) -> bool:
        raise AssertionError(f"is_flexible not defined by {self.__class__}")

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        return math.zeros(value.shape._replace_single_size(dim, 0))

    def __repr__(self):
        return "none"

    def __abs__(self):
        return self

    def __neg__(self):
        return self

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self


class Undefined(Extrapolation):
    """
    The extrapolation is unknown and must be replaced before usage.
    Any access to outside values will raise an AssertionError.
    """

    def __init__(self, derived_from: Extrapolation):
        super().__init__(-1)
        self.derived_from = derived_from

    def to_dict(self) -> dict:
        return {'type': 'undefined', 'derived_from': self.derived_from.to_dict()}

    def pad(self, value: Tensor, widths: dict, **kwargs) -> Tensor:
        for (lo, up) in widths.items():
            assert lo == 0 and up == 0, "Undefined extrapolation"

    def spatial_gradient(self) -> 'Extrapolation':
        return self

    def valid_outer_faces(self, dim):
        return self.derived_from.valid_outer_faces(dim)

    @property
    def is_flexible(self) -> bool:
        raise AssertionError("Undefined extrapolation")

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        raise AssertionError("Undefined extrapolation")

    def __repr__(self):
        return "undefined"

    def __abs__(self):
        return self

    def __neg__(self):
        return self

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self


ZERO = ConstantExtrapolation(0)
""" Extrapolates with the constant value 0 (Dirichlet boundary condition). """
ONE = ConstantExtrapolation(1)
""" Extrapolates with the constant value 1 (Dirichlet boundary condition). """
PERIODIC = _PeriodicExtrapolation(1)
""" Extends a grid by tiling it (Periodic boundary condition). """
BOUNDARY = _BoundaryExtrapolation(2)
""" Extends a grid with its edge values (Neumann boundary condition). The value of a point lying outside the grid is determined by the closest grid value(s). """
SYMMETRIC = _SymmetricExtrapolation(3)
""" Extends a grid by tiling it. Every other copy of the grid is flipped. Edge values occur twice per seam. """
REFLECT = _ReflectExtrapolation(4)
""" Like SYMMETRIC but the edge values are not copied and only occur once per seam. """
NONE = _NoExtrapolation(-1)
""" Raises AssertionError when used to determine outside values. Padding operations will have no effect with this extrapolation. """


def combine_sides(**extrapolations: Extrapolation or tuple) -> Extrapolation:
    """
    Specify extrapolations for each side / face of a box.

    Args:
        **extrapolations: map from dim: str -> `Extrapolation` or `tuple` (lower, upper)

    Returns:
        `Extrapolation`
    """
    values = set()
    proper_dict = {}
    for dim, ext in extrapolations.items():
        if isinstance(ext, Extrapolation):
            values.add(ext)
            proper_dict[dim] = (ext, ext)
        elif isinstance(ext, tuple):
            assert len(ext) == 2, "Tuple must contain exactly two elements, (lower, upper)"
            lower = ext[0] if isinstance(ext[0], Extrapolation) else ConstantExtrapolation(ext[0])
            upper = ext[1] if isinstance(ext[1], Extrapolation) else ConstantExtrapolation(ext[1])
            values.add(lower)
            values.add(upper)
            proper_dict[dim] = (lower, upper)
        else:
            proper_ext = ext if isinstance(ext, Extrapolation) else ConstantExtrapolation(ext)
            values.add(proper_ext)
            proper_dict[dim] = (proper_ext, proper_ext)
    if len(values) == 1:  # All equal -> return any
        return next(iter(values))
    else:
        return _MixedExtrapolation(proper_dict)


class _MixedExtrapolation(Extrapolation):

    def __init__(self, extrapolations: Dict[str, Tuple[Extrapolation, Extrapolation]]):
        """
        A mixed extrapolation uses different extrapolations for different sides.

        Args:
          extrapolations: axis: str -> (lower: Extrapolation, upper: Extrapolation) or Extrapolation
        """
        super().__init__(pad_rank=None)
        self.ext = extrapolations

    def to_dict(self) -> dict:
        return {
            'type': 'mixed',
            'dims': {ax: (es[0].to_dict(), es[1].to_dict()) for ax, es in self.ext.items()}
        }

    def __value_attrs__(self):
        return 'ext',

    def __eq__(self, other):
        if isinstance(other, _MixedExtrapolation):
            return self.ext == other.ext
        else:
            simplified = combine_sides(**self.ext)
            if not isinstance(simplified, _MixedExtrapolation):
                return simplified == other
            else:
                return False

    def __hash__(self):
        simplified = combine_sides(**self.ext)
        if not isinstance(simplified, _MixedExtrapolation):
            return hash(simplified)
        else:
            return hash(frozenset(self.ext.items()))

    def __repr__(self):
        return repr(self.ext)

    def spatial_gradient(self) -> Extrapolation:
        return combine_sides(**{ax: (es[0].spatial_gradient(), es[1].spatial_gradient()) for ax, es in self.ext.items()})

    def valid_outer_faces(self, dim):
        e_lower, e_upper = self.ext[dim]
        return e_lower.valid_outer_faces(dim)[0], e_upper.valid_outer_faces(dim)[1]

    def is_copy_pad(self, dim: str, upper_edge: bool):
        return self.ext[dim][upper_edge].is_copy_pad(dim, upper_edge)

    @property
    def is_flexible(self) -> bool:
        result_by_dim = [lo.is_flexible or up.is_flexible for lo, up in self.ext.values()]
        return any(result_by_dim)

    def pad(self, value: Tensor, widths: dict, **kwargs) -> Tensor:
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
            value = ext.pad(value, ext_widths, **kwargs)
        return value

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        extrap: Extrapolation = self.ext[dim][upper_edge]
        return extrap.pad_values(value, width, dim, upper_edge, **kwargs)

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        coordinates = coordinates.vector.unstack()
        assert len(self.ext) == len(shape.spatial) == len(coordinates)
        result = []
        for dim, dim_coords in zip(shape.spatial.unstack(), coordinates):
            dim_extrapolations = self.ext[dim.name]
            if dim_extrapolations[0] == dim_extrapolations[1]:
                result.append(dim_extrapolations[0].transform_coordinates(dim_coords, dim, **kwargs))
            else:  # separate boundary for lower and upper face
                lower = dim_extrapolations[0].transform_coordinates(dim_coords, dim, **kwargs)
                upper = dim_extrapolations[1].transform_coordinates(dim_coords, dim, **kwargs)
                result.append(math.where(dim_coords <= 0, lower, upper))
        if 'vector' in result[0].shape:
            return concat(result, channel('vector'))
        else:
            return stack(result, channel('vector'))

    def __getitem__(self, item):
        if isinstance(item, dict):
            return combine_sides(**{dim: (e1[item], e2[item]) for dim, (e1, e2) in self.ext.items()})
        else:
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

    def __truediv__(self, other):
        return self._op2(other, lambda e1, e2: e1 / e2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda e1, e2: e2 / e1)

    def _op2(self, other, operator):
        if isinstance(other, _MixedExtrapolation):
            assert self.ext.keys() == other.ext.keys()
            return combine_sides(**{ax: (operator(lo, other.ext[ax][False]), operator(hi, other.ext[ax][True])) for ax, (lo, hi) in self.ext.items()})
        else:
            return combine_sides(**{ax: (operator(lo, other), operator(hi, other)) for ax, (lo, hi) in self.ext.items()})

    def __abs__(self):
        return combine_sides(**{ax: (abs(lo), abs(up)) for ax, (lo, up) in self.ext.items()})

    def __neg__(self):
        return combine_sides(**{ax: (-lo, -up) for ax, (lo, up) in self.ext.items()})


class _NormalTangentialExtrapolation(Extrapolation):

    def __init__(self, normal: Extrapolation, tangential: Extrapolation):
        super().__init__(pad_rank=min(normal.pad_rank, tangential.pad_rank))
        self.normal = normal
        self.tangential = tangential

    def to_dict(self) -> dict:
        return {
            'type': 'normal-tangential',
            'normal': self.normal.to_dict(),
            'tangential': self.tangential.to_dict(),
        }

    def __value_attrs__(self):
        return 'normal', 'tangential'

    def __repr__(self):
        return f"normal={self.normal}, tangential={self.tangential}"

    def spatial_gradient(self) -> 'Extrapolation':
        return combine_by_direction(self.normal.spatial_gradient(), self.tangential.spatial_gradient())

    def valid_outer_faces(self, dim) -> tuple:
        return self.normal.valid_outer_faces(dim)

    def is_copy_pad(self, dim: str, upper_edge: bool):
        return False  # normal and tangential might copy from different places, so no.

    @property
    def is_flexible(self) -> bool:
        return self.normal.is_flexible

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, **kwargs) -> Tensor:
        if 'vector' not in value.shape:
            warnings.warn(f'{self} adding a vector dimension to tensor {value.shape}')
            from phi.math import expand
            value = expand(value, channel(vector=spatial(value).names))
        assert value.vector.item_names is not None, "item_names must be present when padding with normal-tangential"
        result = []
        for component_name, component in zip(value.vector.item_names, value.vector):
            ext = self.normal if component_name == dim else self.tangential
            result.append(ext.pad_values(component, width, dim, upper_edge, **kwargs))
        from ._magic_ops import stack
        result = stack(result, value.shape.only('vector'))
        return result

    def __eq__(self, other):
        return isinstance(other, _NormalTangentialExtrapolation) and self.normal == other.normal and self.tangential == other.tangential

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

    def __truediv__(self, other):
        return self._op2(other, lambda e1, e2: e1 / e2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda e1, e2: e2 / e1)

    def _op2(self, other, operator):
        if isinstance(other, _NormalTangentialExtrapolation):
            return combine_by_direction(normal=operator(self.normal, other.normal), tangential=operator(self.tangential, other.tangential))
        else:
            return combine_by_direction(normal=operator(self.normal, other), tangential=operator(self.tangential, other))

    def __abs__(self):
        return combine_by_direction(normal=abs(self.normal), tangential=abs(self.tangential))

    def __neg__(self):
        return combine_by_direction(normal=-self.normal, tangential=-self.tangential)


def combine_by_direction(normal: Extrapolation or float or Tensor, tangential: Extrapolation or float or Tensor) -> Extrapolation:
    """
    Use a different extrapolation for the normal component of vector-valued tensors.

    Args:
        normal: Extrapolation for the component that is orthogonal to the boundary.
        tangential: Extrapolation for the component that is tangential to the boundary.

    Returns:
        `Extrapolation`
    """
    normal = normal if isinstance(normal, Extrapolation) else ConstantExtrapolation(normal)
    tangential = tangential if isinstance(tangential, Extrapolation) else ConstantExtrapolation(tangential)
    return normal if normal == tangential else _NormalTangentialExtrapolation(normal, tangential)


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
    elif etype == 'mixed':
        dims: Dict[str, tuple] = dictionary['dims']
        extrapolations = {dim: (from_dict(lo_up[0]), from_dict(lo_up[1])) for dim, lo_up in dims.items()}
        return _MixedExtrapolation(extrapolations)
    elif etype == 'normal-tangential':
        normal = from_dict(dictionary['normal'])
        tangential = from_dict(dictionary['tangential'])
        return _NormalTangentialExtrapolation(normal, tangential)
    elif etype == 'none':
        return NONE
    elif etype == 'undefined':
        derived_from = from_dict(dictionary['derived_from'])
        return Undefined(derived_from)
    else:
        raise ValueError(dictionary)


def order_by_shape(shape: Shape, sequence, default=None) -> tuple or list:
    """
    If sequence is a dict with dimension names as keys, orders its values according to this shape.

    Otherwise, the sequence is returned unchanged.

    Args:
      sequence: Sequence or dict to be ordered
      default: default value used for dimensions not contained in sequence

    Returns:
      ordered sequence of values
    """
    if isinstance(sequence, dict):
        result = [sequence.get(name, default) for name in shape.names]
        return result
    elif isinstance(sequence, (tuple, list)):
        assert len(sequence) == shape.rank
        return sequence
    else:  # just a constant
        return sequence


def map(f: Callable[[Extrapolation], Extrapolation], extrapolation):
    """
    Applies a function to all leaf extrapolations in `extrapolation`.
    Non-leaves are those created by `combine_sides()` and `combine_by_direction()`.

    The tree will be collapsed if possible.

    Args:
        f: Function mapping a leaf `Extrapolation` to another `Extrapolation`.
        extrapolation: Input tree for `f`.

    Returns:
        `Extrapolation`
    """
    if isinstance(extrapolation, _MixedExtrapolation):
        return combine_sides(**{dim: (map(f, lo), map(f, up)) for dim, (lo, up) in extrapolation.ext.items()})
    elif isinstance(extrapolation, _NormalTangentialExtrapolation):
        return combine_by_direction(map(f, extrapolation.normal), map(f, extrapolation.tangential))
    else:
        return f(extrapolation)

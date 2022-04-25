import copy
import numbers
import traceback
import warnings
from typing import Tuple, Callable, List, TypeVar

import numpy as np

from phi.math._shape import TYPE_ABBR, IncompatibleShapes, INSTANCE_DIM, _construct_shape, instance
from ._config import GLOBAL_AXIS_ORDER, should_use_color
from ._shape import (Shape,
                     CHANNEL_DIM, BATCH_DIM, SPATIAL_DIM, EMPTY_SHAPE,
                     parse_dim_order, shape_stack, merge_shapes, channel, concat_shapes)
from .backend import NoBackendFound, choose_backend, BACKENDS, get_precision, default_backend, convert as convert_, \
    Backend
from .backend._dtype import DType


class Sliceable:

    @property
    def shape(self) -> Shape:
        """
        Returns the shape of this object.

        Returns:
            `Shape`
        """
        raise NotImplementedError(self.__class__)

    def __getitem__(self, item: dict) -> 'Sliceable':
        raise NotImplementedError(self.__class__)

    def __getattr__(self, name: str) -> 'BoundDim':
        if name.startswith('_'):
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        if hasattr(self.__class__, name):
            raise RuntimeError(f"Failed to get attribute '{name}' of {self.__class__}")
        return BoundDim(self, name)


class BoundDim:
    """
    Represents a dimension of a sliceable object.
    Any instance of `BoundDim` is bound to the sliceable object and is immutable.
    All operations upon the dim affect return a copy of the sliceable object.
    """

    def __init__(self, obj: Sliceable, name: str):
        self.obj = obj
        self.name = name

    @property
    def exists(self):
        """ Whether the dimension is listed in the `Shape` of the object. """
        return self.name in self.obj.shape

    def __repr__(self):
        if self.name not in self.obj.shape:
            return f"{type(self.obj).__name__}.{self.name} (non-existent)"
        items = self.item_names
        if items is not None:
            if len(items) <= 4:
                size_repr = ",".join(items)
            else:
                size_repr = f"{self.size}:{items[0]}..{items[-1]}"
        else:
            size_repr = self.size
        return f"{type(self.obj).__name__}.{self.name}{TYPE_ABBR.get(self.dim_type, '?')}={size_repr}"

    @property
    def size(self):
        """ Length of this dimension as listed in the `Shape` of the bound object. """
        return self.obj.shape.get_size(self.name) if self.exists else None

    @property
    def dim_type(self):
        return self.obj.shape.get_type(self.name)

    @property
    def _dim_type(self):
        return self.obj.shape.get_type(self.name)

    @property
    def is_spatial(self):
        """ Whether the type of this dimension as listed in the `Shape` is *spatial*. Only defined for existing dimensions. """
        return self._dim_type == SPATIAL_DIM

    @property
    def is_batch(self):
        """ Whether the type of this dimension as listed in the `Shape` is *batch*. Only defined for existing dimensions. """
        return self._dim_type == BATCH_DIM

    @property
    def is_channel(self):
        """ Whether the type of this dimension as listed in the `Shape` is *channel*. Only defined for existing dimensions. """
        return self._dim_type == CHANNEL_DIM

    @property
    def item_names(self):
        return self.obj.shape.get_item_names(self.name)

    @property
    def index(self):
        """ The index of this dimension in the `Shape` of the `Tensor`. """
        return self.obj.shape.index(self.name)

    def __int__(self):
        return self.index

    def __getitem__(self, item):
        return self.obj[{self.name: item}]

    def unstack(self, size: int or None = None) -> tuple:
        """
        Lists the slices along this dimension as a `tuple`.

        Args:
            size: (optional) If given as `int`, this dimension can be unstacked even if it is not present on the object.
                In that case, `size` copies of the object are returned.

        Returns:
            `tuple` of `Sliceable`
        """
        from ._ops import unstack
        if size is None:
            return unstack(self.obj, self.name)
        else:
            if self.exists:
                unstacked = unstack(self.obj, self.name)
                assert len(unstacked) == size, f"Size of dimension {self.name} does not match {size}."
                return unstacked
            else:
                return (self.obj,) * size

    def __iter__(self):
        """ Iterate over slices along this dim """
        if self.exists:
            return iter(self.unstack())
        else:
            return iter([self.obj])

    def __call__(self, *args, **kwargs):
        raise TypeError(f"Method {type(self.obj).__name__}.{self.name}() does not exist.")


class Tensor(Sliceable):
    """
    Abstract base class to represent structured data of one data type.
    This class replaces the native tensor classes `numpy.ndarray`, `torch.Tensor`, `tensorflow.Tensor` or `jax.numpy.ndarray` as the main data container in Φ<sub>Flow</sub>.

    `Tensor` instances are different from native tensors in two important ways:

    * The dimensions of Tensors have *names* and *types*.
    * Tensors can have non-uniform shapes, meaning that the size of dimensions can vary along other dimensions.

    To check whether a value is a tensor, use `isinstance(value, Tensor)`.

    To construct a Tensor, use `phi.math.tensor()`, `phi.math.wrap()` or one of the basic tensor creation functions,
    see https://tum-pbs.github.io/PhiFlow/Math.html#tensor-creation .

    Tensors are not editable.
    When backed by an editable native tensor, e.g. a `numpy.ndarray`, do not edit the underlying data structure.
    """

    def native(self, order: str or tuple or list or Shape = None):
        """
        Returns a native tensor object with the dimensions ordered according to `order`.
        
        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        Args:
            order: (Optional) list of dimension names. If not given, the current dimension order is kept.

        Returns:
            Native tensor representation

        Raises:
            ValueError if the tensor cannot be transposed to match target_shape
        """
        raise NotImplementedError()

    def numpy(self, order: str or tuple or list = None) -> np.ndarray:
        """
        Converts this tensor to a `numpy.ndarray` with dimensions ordered according to `order`.
        
        *Note*: Using this function breaks the autograd chain. The returned tensor is not differentiable.
        To get a differentiable tensor, use `Tensor.native()` instead.
        
        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        If this `Tensor` is backed by a NumPy array, a reference to this array may be returned.

        See Also:
            `phi.math.numpy()`

        Args:
            order: (Optional) list of dimension names. If not given, the current dimension order is kept.

        Returns:
            NumPy representation

        Raises:
            ValueError if the tensor cannot be transposed to match target_shape
        """
        native = self.native(order=order)
        return choose_backend(native).numpy(native)

    @property
    def dtype(self) -> DType:
        """ Data type of the elements of this `Tensor`. """
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        """ The `Shape` lists the dimensions with their sizes, names and types. """
        raise NotImplementedError()

    @property
    def default_backend(self):
        from ._ops import choose_backend_t
        return choose_backend_t(self)

    def _with_shape_replaced(self, new_shape: Shape):
        raise NotImplementedError()

    def _with_natives_replaced(self, natives: list):
        """ Replaces all n _natives() of this Tensor with the first n elements of the list and removes them from the list. """
        raise NotImplementedError()

    @property
    def rank(self) -> int:
        """
        Number of explicit dimensions of this `Tensor`. Equal to `tensor.shape.rank`.
        This replaces [`numpy.ndarray.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html) /
        [`torch.Tensor.dim`](https://pytorch.org/docs/master/generated/torch.Tensor.dim.html) /
        [`tf.rank()`](https://www.tensorflow.org/api_docs/python/tf/rank) /
        [`jax.numpy.ndim()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndim.html).
        """
        return self.shape.rank

    @property
    def _is_tracer(self) -> bool:
        """
        Tracers store additional internal information.
        They should not be converted to `native()` in intermediate operations.
        
        TensorStack prevents performing the actual stack operation if one of its component tensors is special.
        """
        raise NotImplementedError()

    def _to_dict(self):
        return cached(self)._to_dict()

    def __len__(self):
        return self.shape.volume if self.rank == 1 else NotImplemented

    def __bool__(self):
        assert self.rank == 0, f"Cannot convert tensor with non-empty shape {self.shape} to bool. Use tensor.any or tensor.all instead."
        from ._ops import all_
        if not self.default_backend.supports(Backend.jit_compile):  # NumPy
            return bool(self.native()) if self.rank == 0 else bool(all_(self).native())
        else:
            # __bool__ does not work with TensorFlow tracing.
            # TensorFlow needs to see a tf.Tensor in loop conditions but won't allow bool() invocations.
            # However, this function must always return a Python bool.
            raise AssertionError("To evaluate the boolean value of a Tensor, use 'Tensor.all'.")

    @property
    def all(self):
        """ Whether all values of this `Tensor` are `True` as a native bool. """
        from ._ops import all_, cast
        if self.rank == 0:
            return cast(self, DType(bool)).native()
        else:
            return all_(self, dim=self.shape).native()

    @property
    def any(self):
        """ Whether this `Tensor` contains a `True` value as a native bool. """
        from ._ops import any_, cast
        if self.rank == 0:
            return cast(self, DType(bool)).native()
        else:
            return any_(self, dim=self.shape).native()

    @property
    def mean(self):
        """ Mean value of this `Tensor` as a native scalar. """
        from ._ops import mean
        return mean(self, dim=self.shape).native()

    @property
    def std(self):
        """ Standard deviation of this `Tensor` as a native scalar. """
        from ._ops import std
        return std(self, dim=self.shape).native()

    @property
    def sum(self):
        """ Sum of all values of this `Tensor` as a native scalar. """
        from ._ops import sum_
        return sum_(self, dim=self.shape).native()

    @property
    def min(self):
        """ Minimum value of this `Tensor` as a native scalar. """
        from ._ops import min_
        return min_(self, dim=self.shape).native()

    @property
    def max(self):
        """ Maximum value of this `Tensor` as a native scalar. """
        from ._ops import max_
        return max_(self, dim=self.shape).native()

    @property
    def real(self):
        from ._ops import real
        return real(self)

    @property
    def imag(self):
        from ._ops import imag
        return imag(self)

    def __int__(self):
        return int(self.native()) if self.shape.volume == 1 else NotImplemented

    def __float__(self):
        return float(self.native()) if self.shape.volume == 1 else NotImplemented

    def __complex__(self):
        return complex(self.native()) if self.shape.volume == 1 else NotImplemented

    def __index__(self):
        assert self.shape.volume == 1, f"Only scalar tensors can be converted to index but has shape {self.shape}"
        assert self.dtype.kind == int, f"Only int tensors can be converted to index but dtype is {self.dtype}"
        return int(self.native())

    def _summary_str(self) -> str:
        if should_use_color():
            v = '\033[94m'  # value
            s = '\033[92m'  # shape
            e = '\033[0m'   # end
            d = '\033[93m'  # dtype
            g = '\033[37m'  # grey (additional)
            # BOLD = '\033[1m'
            # UNDERLINE = '\033[4m'
        else:
            v, s, d, e, g = '', '', '', '', ''

        try:
            from ._ops import all_available
            if all_available(self):
                if self.rank == 0:
                    return f"{v}{str(self.numpy())}{e}"
                elif self.shape.volume is not None and self.shape.volume <= 6:
                    content = list(np.reshape(self.numpy(self.shape.names), [-1]))
                    if self.shape.rank == 1 and self.shape.get_item_names(0) is not None:
                        content = ", ".join([f"{item}={v}{number}{e}" for number, item in zip(content, self.shape.get_item_names(0))])
                    else:
                        content = ', '.join([f"{v}{number}{e}" for number in content])
                    if self.shape.rank == 1 and (self.dtype.kind in (bool, int) or self.dtype.precision == get_precision()):
                        if self.shape.name == 'vector' and self.shape.type == CHANNEL_DIM:
                            return f"({content})"
                        return f"({content}) along {s}{self.shape.name}{TYPE_ABBR[self.shape.type]}{e}"
                    return f"{s}{self.shape}{e} {d}{self.dtype}{e}  {content}"
                else:
                    if self.dtype.kind in (float, int):
                        min_val, max_val, mean, std = [float(f) for f in [self.min, self.max, self.mean, self.std]]
                        if std == 0:
                            return f"{s}{self.shape}{e} {d}{self.dtype}{e} const {v}{mean}{e}"
                        if any([abs(val) < 0.001 or abs(val) > 1000 for val in [mean, std]]):
                            return f"{s}{self.shape}{e} {d}{self.dtype}{e}  {v}{mean:.2e} ± {std:.1e}{e} {g}({min_val:.0e}...{max_val:.0e}){e}"
                        else:
                            return f"{s}{self.shape}{e} {d}{self.dtype}{e}  {v}{mean:.3f} ± {std:.3f}{e} {g}({min_val:.0e}...{max_val:.0e}){e}"
                    elif self.dtype.kind == complex:
                        max_val = abs(self).max
                        return f"{s}{self.shape}{e} {d}{self.dtype}{e} {v}|...| < {max_val}{e}"
                    elif self.dtype.kind == bool:
                        return f"{s}{self.shape}{e} {v}{self.sum} / {self.shape.volume} True{e}"
                    else:
                        return f"{s}{self.shape}{e} {d}{self.dtype}{e}"
            else:
                if self.rank == 0:
                    return f"{self.default_backend} scalar {d}{self.dtype}{e}"
                else:
                    return f"{self.default_backend} {s}{self.shape}{e} {d}{self.dtype}{e}"
        except BaseException as err:
            return f"{self.shape}, failed to fetch values: {err}"

    def __repr__(self):
        return self._summary_str()

    def __format__(self, format_spec):
        from ._ops import all_available
        if not all_available(self):
            return self._summary_str()
        if self.shape.volume > 1:
            return self._summary_str()
        val = self.numpy()
        return format(val, format_spec)

    def __getitem__(self, item):
        if isinstance(item, Tensor):
            from ._ops import gather
            return gather(self, item)
        if isinstance(item, (int, slice)):
            assert self.rank == 1
            item = {self.shape.names[0]: item}
        if isinstance(item, (tuple, list)):
            if item[0] == Ellipsis:
                assert len(item) - 1 == self.shape.channel.rank
                item = {name: selection for name, selection in zip(self.shape.channel.names, item[1:])}
            elif len(item) == self.shape.channel.rank:
                item = {name: selection for name, selection in zip(self.shape.channel.names, item)}
            elif len(item) == self.shape.rank:  # legacy indexing
                warnings.warn("Slicing a Tensor with a tuple or list should only be used for channel dimensions. Use a dict or the special slicing syntax tensor.dim[slice] instead", SyntaxWarning, stacklevel=2)
                item = {name: selection for name, selection in zip(self.shape.names, item)}
        assert isinstance(item, dict)  # dict mapping name -> slice/int
        selections = {}
        sliced = self
        for dim, selection in item.items():
            if dim not in self.shape:
                continue
            if isinstance(selection, Shape):
                selection = selection.name if selection.rank == 1 else selection.names
            if isinstance(selection, str) and ',' in selection:
                selection = parse_dim_order(selection)
            if isinstance(selection, str):  # single item name
                item_names = self.shape.get_item_names(dim, fallback_spatial=True)
                assert item_names is not None, f"No item names defined for dim '{dim}' in tensor {self.shape} and dimension size does not match spatial rank."
                assert selection in item_names, f"Accessing tensor.{dim}['{selection}'] failed. Item names are {item_names}."
                selection = item_names.index(selection)
            # Either handle slicing directly or add it to the dict
            if isinstance(selection, (tuple, list)):
                selection_int = list(selection)
                if any([isinstance(s, str) for s in selection]):
                    item_names = self.shape.get_item_names(dim, fallback_spatial=True)
                    for i, s in enumerate(selection):
                        if isinstance(s, str):
                            assert s in item_names, f"Accessing tensor.{dim}['{s}'] failed. Item names are {item_names}."
                            selection_int[i] = item_names.index(s)
                from ._ops import stack
                result = [sliced[{dim: i}] for i in selection_int]
                item_names = [str(n) for n in selection]
                stack_dim = self.shape[dim] if dim in self.shape else channel(dim)
                sliced = stack({n: r for n, r in zip(item_names, result)}, stack_dim)
            elif isinstance(selection, Tensor) and selection.dtype.kind == bool:
                from ._ops import boolean_mask
                sliced = boolean_mask(sliced, dim, selection)
            elif isinstance(selection, Tensor) and selection.dtype.kind == int:
                from ._ops import gather
                sliced = gather(sliced, selection, dims=dim)
            else:
                selections[dim] = selection
        return sliced._getitem(selections) if selections else sliced


    def _getitem(self, selection: dict) -> 'Tensor':
        """
        Slice the tensor along specified dimensions.

        Args:
          selection: dim_name: str -> int or slice
          selection: dict: 

        Returns:

        """
        raise NotImplementedError()

    def flip(self, *dims: str) -> 'Tensor':
        """
        Reverses the order of elements along one or multiple dimensions.

        Args:
            *dims: dimensions to flip

        Returns:
            `Tensor` of the same `Shape`
        """
        raise NotImplementedError()

    # def __setitem__(self, key, value):
    #     """
    #     All tensors are editable.
    #
    #     :param key: list/tuple of slices / indices
    #     :param value:
    #     :return:
    #     """
    #     raise NotImplementedError()

    def unstack(self, dimension: str):
        """
        Splits this tensor along the specified dimension.
        The returned tensors have the same dimensions as this tensor save the unstacked dimension.

        Raises an error if the dimension is not part of the `Shape` of this `Tensor`.

        See Also:
            `TensorDim.unstack()`

        Args:
          dimension(str or int or TensorDim): name of dimension or Dimension or None for component dimension

        Returns:
          tuple of tensors

        """
        raise NotImplementedError()

    def dimension(self, name: str or Shape) -> 'TensorDim':
        """
        Returns a reference to a specific dimension of this tensor.
        This is equivalent to the syntax `tensor.<name>`.

        The dimension need not be part of the `Tensor.shape` in which case its size is 1.

        Args:
            name: dimension name

        Returns:
            `TensorDim` corresponding to a dimension of this tensor
        """
        if isinstance(name, str):
            return TensorDim(self, name)
        elif isinstance(name, Shape):
            return TensorDim(self, name.name)
        else:
            raise ValueError(name)

    def pack(self, dims, packed_dim):
        """ See `pack_dims()` """
        from ._ops import pack_dims
        return pack_dims(self, dims, packed_dim)

    def unpack(self, dim, unpacked_dims):
        """ See `unpack_dims()` """
        from ._ops import unpack_dims
        return unpack_dims(self, dim, unpacked_dims)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        if name == 'is_tensor_like':  # TensorFlow replaces abs() while tracing and checks for this attribute
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        assert name not in ('shape', '_shape', 'tensor'), name
        return TensorDim(self, name)

    def __add__(self, other):
        return self._op2(other, lambda x, y: x + y, lambda x, y: choose_backend(x, y).add(x, y), 'add', '+')

    def __radd__(self, other):
        return self._op2(other, lambda x, y: y + x, lambda x, y: choose_backend(x, y).add(y, x), 'radd', '+')

    def __sub__(self, other):
        return self._op2(other, lambda x, y: x - y, lambda x, y: choose_backend(x, y).sub(x, y), 'sub', '-')

    def __rsub__(self, other):
        return self._op2(other, lambda x, y: y - x, lambda x, y: choose_backend(x, y).sub(y, x), 'rsub', '-')

    def __and__(self, other):
        return self._op2(other, lambda x, y: x & y, lambda x, y: choose_backend(x, y).and_(x, y), 'and', '&')

    def __rand__(self, other):
        return self._op2(other, lambda x, y: y & x, lambda x, y: choose_backend(x, y).and_(y, x), 'rand', '&')

    def __or__(self, other):
        return self._op2(other, lambda x, y: x | y, lambda x, y: choose_backend(x, y).or_(x, y), 'or', '|')

    def __ror__(self, other):
        return self._op2(other, lambda x, y: y | x, lambda x, y: choose_backend(x, y).or_(y, x), 'ror', '|')

    def __xor__(self, other):
        return self._op2(other, lambda x, y: x ^ y, lambda x, y: choose_backend(x, y).xor(x, y), 'xor', '^')

    def __rxor__(self, other):
        return self._op2(other, lambda x, y: y ^ x, lambda x, y: choose_backend(x, y).xor(y, x), 'rxor', '^')

    def __mul__(self, other):
        return self._op2(other, lambda x, y: x * y, lambda x, y: choose_backend(x, y).mul(x, y), 'mul', '*')

    def __rmul__(self, other):
        return self._op2(other, lambda x, y: y * x, lambda x, y: choose_backend(x, y).mul(y, x), 'rmul', '*')

    def __truediv__(self, other):
        return self._op2(other, lambda x, y: x / y, lambda x, y: choose_backend(x, y).div(x, y), 'truediv', '/')

    def __rtruediv__(self, other):
        return self._op2(other, lambda x, y: y / x, lambda x, y: choose_backend(x, y).div(y, x), 'rtruediv', '/')

    def __divmod__(self, other):
        return self._op2(other, lambda x, y: divmod(x, y), lambda x, y: divmod(x, y), 'divmod', 'divmod')

    def __rdivmod__(self, other):
        return self._op2(other, lambda x, y: divmod(y, x), lambda x, y: divmod(y, x), 'rdivmod', 'divmod')

    def __floordiv__(self, other):
        return self._op2(other, lambda x, y: x // y, lambda x, y: choose_backend(x, y).floordiv(x, y), 'floordiv', '//')

    def __rfloordiv__(self, other):
        return self._op2(other, lambda x, y: y // x, lambda x, y: choose_backend(x, y).floordiv(y, x), 'rfloordiv', '//')

    def __pow__(self, power, modulo=None):
        assert modulo is None
        return self._op2(power, lambda x, y: x ** y, lambda x, y: choose_backend(x, y).pow(x, y), 'pow', '**')

    def __rpow__(self, other):
        return self._op2(other, lambda x, y: y ** x, lambda x, y: choose_backend(x, y).pow(y, x), 'rpow', '**')

    def __mod__(self, other):
        return self._op2(other, lambda x, y: x % y, lambda x, y: choose_backend(x, y).mod(x, y), 'mod', '%')

    def __rmod__(self, other):
        return self._op2(other, lambda x, y: y % x, lambda x, y: choose_backend(x, y).mod(y, x), 'rmod', '%')

    def __eq__(self, other):
        return self._op2(other, lambda x, y: x == y, lambda x, y: choose_backend(x, y).equal(x, y), 'eq', '==')

    def __ne__(self, other):
        return self._op2(other, lambda x, y: x != y, lambda x, y: choose_backend(x, y).not_equal(x, y), 'ne', '!=')

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y, lambda x, y: choose_backend(x, y).greater_than(y, x), 'lt', '<')

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y, lambda x, y: choose_backend(x, y).greater_or_equal(y, x), 'le', '<=')

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y, lambda x, y: choose_backend(x, y).greater_than(x, y), 'gt', '>')

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y, lambda x, y: choose_backend(x, y).greater_or_equal(x, y), 'ge', '>=')

    def __abs__(self):
        return self._op1(lambda t: choose_backend(t).abs(t))

    def __round__(self, n=None):
        return self._op1(lambda t: choose_backend(t).round(t))

    def __copy__(self):
        return self._op1(lambda t: choose_backend(t).copy(t, only_mutable=True))

    def __deepcopy__(self, memodict={}):
        return self._op1(lambda t: choose_backend(t).copy(t, only_mutable=False))

    def __neg__(self):
        return self._op1(lambda t: -t)

    def __invert__(self):
        return self._op1(lambda t: ~t)

    def __reversed__(self):
        assert self.shape.channel.rank == 1
        return self[::-1]

    def __iter__(self):
        if self.rank == 1:
            return iter(self.native())
        elif self.rank == 0:
            return iter([self.native()])
        else:
            from ._ops import flatten
            return iter(flatten(self))

    def _tensor(self, other):
        if isinstance(other, Tensor):
            return other
        return compatible_tensor(other, compat_shape=self.shape, compat_natives=self._natives(), convert=False)

    def _op1(self, native_function):
        """
        Transform the values of this tensor given a function that can be applied to any native tensor.

        Args:
          native_function:

        Returns:

        """
        raise NotImplementedError(self.__class__)

    def _op2(self, other: 'Tensor', operator: Callable, native_function: Callable, op_name: str = 'unknown', op_symbol: str = '?') -> 'Tensor':
        """
        Apply a broadcast operation on two tensors.

        Args:
            other: second argument
            operator: function (Tensor, Tensor) -> Tensor, used to propagate the operation to children tensors to have Python choose the callee
            native_function: function (native tensor, native tensor) -> native tensor
            op_name: Name of the python function without leading and trailing `__`.
                Examples: 'add', 'radd', 'sub', 'mul', 'and', 'eq', 'ge'.
            op_symbol: Operation symbol, such as '+', '-', '&', '%', '>='

        Returns:
            `Tensor`
        """
        raise NotImplementedError()

    def _natives(self) -> tuple:
        raise NotImplementedError(self.__class__)

    def _expand(self):
        """ Expands all compressed tensors to their defined size as if they were being used in `Tensor.native()`. """
        warnings.warn("Tensor._expand() is deprecated, use cached(Tensor) instead.", DeprecationWarning)
        raise NotImplementedError(self.__class__)

    def _tensor_reduce(self,
                       dims: Tuple[str],
                       native_function: Callable,
                       collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                       unaffected_function: Callable = lambda value: value):
        raise NotImplementedError(self.__class__)

    def _simplify(self):
        return self


def shape(arg: Tensor or Shape) -> Shape:
    """
    If `arg` is a `Tensor` or has a compatible `shape` property, returns its shape.

    If `arg` is a `Shape`, returns `arg`.
    This function can be passed as a `dim` argument to an operation to specify that it should act upon all dimensions.

    Args:
        arg: `Tensor` or `Shape`

    Returns:
        `Shape`
    """
    if hasattr(arg, 'shape') and isinstance(arg.shape, Shape):
        return arg.shape
    elif isinstance(arg, Shape):
        return arg
    else:
        raise ValueError(f'shape() requires Tensor of Shape argument but got {arg}')


class TensorDim(BoundDim):
    """
    Reference to a specific dimension of a `Tensor`.

    To obtain a `TensorDim`, use `Tensor.dimension()` or the syntax `tensor.<dim>`.

    Indexing a `TensorDim` as `tdim[start:stop:step]` returns a sliced `Tensor`.

    See the documentation at https://tum-pbs.github.io/PhiFlow/Math.html#indexing-slicing-unstacking .
    """

    def __init__(self, tensor: Tensor, name: str):
        super().__init__(tensor, name)
        self.tensor = tensor

    def optional_unstack(self):
        """
        Unstacks the `Tensor` along this dimension if the dimension is listed in the `Shape`.
        Otherwise returns the original `Tensor`.

        Returns:
            `tuple` of sliced tensors or original `Tensor`
        """
        warnings.warn("optional_unstack() is deprecated.", DeprecationWarning)
        if self.exists:
            return self.unstack()
        else:
            return self.tensor

    def unstack_spatial(self, components: str or tuple or list or Shape) -> tuple:
        """
        Deprecated.
        Use item names instead.

        Slices the tensor along this dimension, returning only the selected components in the specified order.

        Args:
            components: Spatial dimension names as comma-separated `str` or sequence of `str`.

        Returns:
            selected components
        """
        # warnings.warn(f"unstack_spatial() is deprecated. Use tensor.dim[order].dim")
        if isinstance(components, (Shape, str)):
            components = parse_dim_order(components)
        if self.exists:
            spatial = self.tensor.shape.spatial
            result = []
            if spatial.is_empty:
                spatial = [GLOBAL_AXIS_ORDER.axis_name(i, len(components)) for i in range(len(components))]
            for dim in components:
                component_index = spatial.index(dim)
                result.append(self.tensor[{self.name: component_index}])
        else:
            result = [self.tensor] * len(components)
        return tuple(result)

    def __len__(self):
        warnings.warn("Use Tensor.dim.size instead of len(Tensor.dim). len() only supports with integer sizes.", DeprecationWarning)
        return self.size

    def as_batch(self, name: str = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *batch*. """
        return self._as(BATCH_DIM, name)

    def as_spatial(self, name: str = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *spatial*. """
        return self._as(SPATIAL_DIM, name)

    def as_channel(self, name: str = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *channel*. """
        return self._as(CHANNEL_DIM, name)

    def as_instance(self, name: str = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *instance*. """
        return self._as(INSTANCE_DIM, name)

    def rename(self, name: str):
        """ Returns a shallow copy of the `Tensor` where this dimension has the specified name. """
        if not self.exists:
            return self.tensor
        return self._as(self._dim_type, name)

    def as_type(self, dim_type: Callable or str):
        return self._as(dim_type('d').type if callable(dim_type) else dim_type, None)

    def _as(self, dim_type: str, name: str or None):
        if not self.exists:
            return self.tensor
        shape = self.tensor.shape
        new_types = list(shape.types)
        new_types[self.index] = dim_type
        new_names = shape.names
        if name is not None:
            new_names = list(new_names)
            new_names[self.index] = name
        new_shape = Shape(shape.sizes, tuple(new_names), tuple(new_types), shape.item_names)
        return self.tensor._with_shape_replaced(new_shape)

    def flip(self):
        """ Flips the element order along this dimension and returns the result as a `Tensor`. """
        warnings.warn("dim.flip() is deprecated. Use dim[::-1] instead", DeprecationWarning)
        return self.tensor.flip(self.name)

    def split(self, split_dimensions: Shape):
        """ See `phi.math.unpack_dims()` """
        warnings.warn("dim.split() is deprecated. Use math.split_dims() instead.")
        from ._ops import unpack_dims
        return unpack_dims(self.tensor, self.name, split_dimensions)

    def __mul__(self, other):
        if isinstance(other, TensorDim):
            from ._ops import dot
            return dot(self.tensor, (self.name,), other.tensor, (other.name,))
        else:
            return NotImplemented

    def sum(self):
        from ._ops import sum_
        return sum_(self.tensor, self.name)

    def prod(self):
        from ._ops import prod
        return prod(self.tensor, self.name)


class Layout(Tensor):

    def __init__(self, obj, shape: Shape):
        self._obj = obj
        self._shape = shape

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dtype(self) -> DType:
        return DType(object)

    @property
    def default_backend(self):
        return None

    def native(self, order: str or tuple or list or Shape = None):
        order = parse_dim_order(order)
        assert order is None or order == self._shape.names, "Layout.native() does not allow for changing the dimension order"
        return self._obj

    def _getitem(self, selection: dict) -> 'Tensor':
        selection_list = [selection.get(dim, None) for dim in self._shape.names]
        native = self._getitem_recursive(self._obj, tuple(selection_list))
        new_shape = self._shape.after_gather(selection)
        return Layout(native, new_shape)

    def __repr__(self):
        return repr(self._obj)

    def unstack(self, dimension: str):
        if dimension == self._shape.names[0]:
            native = tuple(self._obj.values()) if isinstance(self._obj, dict) else self._obj
            inner_shape = self._shape[1:]
            return tuple([Layout(n, inner_shape) for n in native])
        else:
            raise NotImplementedError()

    @staticmethod
    def _getitem_recursive(native, selection: tuple):
        if not selection:
            return native
        native = tuple(native.values()) if isinstance(native, dict) else native
        if len(selection) == 1:
            return native if selection[0] is None else native[selection[0]]
        else:
            if selection[0] is None:
                return type(native)([Layout._getitem_recursive(n, selection[1:]) for n in native])
            if isinstance(selection[0], int):
                return Layout._getitem_recursive(native[selection[0]], selection[1:])
            elif isinstance(selection[0], slice):
                subset = native[selection[0]]
                return type(subset)([Layout._getitem_recursive(n, selection[1:]) for n in subset])
            else:
                raise ValueError(f"Illegal selection: {selection}")

    def _as_list(self):
        return self._as_list_recursive(self._obj, self._shape.rank, [])

    @staticmethod
    def _as_list_recursive(native, dims: int, result: list):
        if dims == 0:
            result.append(native)
        else:
            native = tuple(native.values()) if isinstance(native, dict) else native
            for n in native:
                Layout._as_list_recursive(n, dims - 1, result)
        return result


class NativeTensor(Tensor):

    def __init__(self, native_tensor, shape: Shape):
        assert isinstance(shape, Shape), f"Expected Shape but got '{type(shape)}'"
        backend = choose_backend(native_tensor)
        # if backend.is_available(native_tensor):
        assert backend.staticshape(native_tensor) == shape.sizes, f"Shape {shape} does not match native tensor with shape {backend.staticshape(native_tensor)}"
        self._native = native_tensor
        self._shape = shape

    def native(self, order: str or tuple or list or Shape = None):
        order = parse_dim_order(order, check_rank=self.rank)
        if order is None or tuple(order) == self.shape.names:
            return self._native
        # --- Insert missing dims ---
        native = self._native
        backend = choose_backend(native)
        shape = self.shape
        for name in order:
            if name not in self.shape:
                native = backend.expand_dims(native, axis=-1)
                shape = concat_shapes(shape, _construct_shape('tmp_perm', **{name: 1}))
        # --- Transpose ---
        perm = shape._perm(order)
        native = backend.transpose(native, perm)
        return native

    @property
    def dtype(self):
        return choose_backend(self._native).dtype(self._native)

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        if new_shape.rank != self._shape.rank:
            raise IncompatibleShapes(f"Tensor {self} is not compatible with shape {new_shape}", self._shape, new_shape)
        new_shape = Shape(self._shape.sizes, new_shape.names, new_shape.types, new_shape.item_names)
        return NativeTensor(self._native, new_shape)

    def _with_natives_replaced(self, natives: list):
        native = natives.pop(0)
        new_shape = self._shape.with_sizes(choose_backend(native).shape(native))
        return NativeTensor(native, new_shape)

    @property
    def _is_tracer(self) -> bool:
        return False

    def _to_dict(self):
        result = self.shape._to_dict(include_sizes=False)
        if self.rank == 0:
            result['data'] = self.numpy().item()
        else:
            result['data'] = self.numpy(self._shape).tolist()  # works for all 1+ dimensional arrays
        return result

    def _getitem(self, selection: dict):
        if len(selection) == 0:
            return self
        new_shape = self.shape
        selections = [slice(None)] * self.rank
        for name, sel in selection.items():
            if name in self.shape:
                selections[self.shape.index(name)] = sel
                if isinstance(sel, int):
                    new_shape = new_shape.without(name)
            else:
                assert isinstance(sel, int), f"Attempting slice missing dimension {name} with {selection}"
        if len(selections) == 0:
            return self
        gathered = self.default_backend.multi_slice(self._native, tuple(selections))
        new_shape = new_shape.with_sizes(choose_backend(gathered).staticshape(gathered))
        return NativeTensor(gathered, new_shape)

    def flip(self, *dims: str) -> 'Tensor':
        dims = [dim for dim in dims if dim in self._shape]
        native = choose_backend(self._native).flip(self._native, self._shape.indices(dims))
        return NativeTensor(native, self._shape.flipped(dims))

    def unstack(self, dimension):
        dim_index = self.shape.index(dimension)
        new_shape = self.shape.without(dimension)
        tensors = choose_backend(self._native).unstack(self._native, axis=dim_index)
        return tuple([NativeTensor(t, new_shape) for t in tensors])

    def _op1(self, native_function):
        native = native_function(self._native)
        return NativeTensor(native, self.shape) if native is not None else self

    def _op2(self, other, operator, native_function, op_name: str = 'unknown', op_symbol: str = '?'):
        try:
            other = self._tensor(other)
        except NoBackendFound:
            return NotImplemented
        if isinstance(other, NativeTensor):
            return op2_native(self, other, native_function)
        else:
            return NotImplemented

    def _natives(self) -> tuple:
        return self._native,

    def _expand(self):
        pass

    def _tensor_reduce(self,
                       dims: Tuple[str],
                       native_function: Callable,
                       collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                       unaffected_function: Callable = lambda value: value):
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        dims = [dim for dim in dims if dim in self.shape]
        backend = choose_backend(self._native)
        result = native_function(backend, self._native, dim=self._shape.indices(dims))
        return NativeTensor(result, self._shape.without(dims))


class CollapsedTensor(Tensor):  # package-private
    """
    Tensor that is constant along some dimensions.
    Non-constant dimensions are represented by `_inner` while `_shape` lists all dimensions.

    When cached via `_cache()`, `_inner` is replaced by `_cached` which is a NativeTensor.
    From this point on, all operations must use `_cached`, otherwise gradients will be incorrect.
    The method `Tensor._expand()` causes a full Tensor structure to cache collapsed dimensions and must be called before gradients are recorded.
    """

    def __init__(self, tensor: Tensor, shape: Shape):
        for name in tensor.shape.names:
            assert name in shape
        for size, name, dim_type, *_ in tensor.shape._dimensions:
            assert wrap(shape.get_size(name) == size).all, f"Shape mismatch while trying to set {name}={shape.get_size(name)} but has size {size}"
            assert shape.get_type(name) == dim_type, f"Dimension type mismatch for dimension '{name}': {shape.get_type(name)}, {dim_type}"
        if isinstance(tensor, CollapsedTensor):
            if tensor.is_cached:
                self._inner = tensor._cached
            else:
                self._inner = tensor._inner
            assert self._inner is not None
        else:
            self._inner = tensor  # this will be set to None once cached. Otherwise gradients will be incorrect.
        self._shape = shape
        self._cached = None  # NativeTensor. Once cached, use only _cached

    def _cache(self):
        if self._cached is None:
            if self._inner._is_tracer:
                return None
            if self.shape.is_uniform:
                native = self._inner.native(order=self.shape.names)
                multiples = [1 if name in self._inner.shape else size for size, name, *_ in self.shape._dimensions]
                tiled = choose_backend(native).tile(native, multiples)
                self._cached = NativeTensor(tiled, self.shape)
                self._inner = None
            else:
                raise NotImplementedError()
        return self._cached

    @property
    def is_cached(self):
        return self._cached is not None

    def _simplify(self):
        if self.is_cached:
            return self._cached
        else:
            return self

    def native(self, order: str or tuple or list or Shape = None):
        if self.is_cached:
            return self._cached.native(order)
        order = parse_dim_order(order, check_rank=self.rank)
        if order is None or tuple(order) == self.shape.names:
            return self._cache().native(order)
        else:
            native = self._inner.native(order=order)
            multiples = [1 if name in self._inner.shape else (self.shape.get_size(name) if name in self.shape else 1) for name in order]
            tiled = choose_backend(native).tile(native, multiples)
            return tiled

    @property
    def dtype(self):
        if self.is_cached:
            return self._cached.dtype
        else:
            return self._inner.dtype

    @property
    def shape(self):
        return self._shape

    def unstack(self, dimension):
        if self.is_cached:
            return self._cached.unstack(dimension)
        unstacked_shape = self.shape.without(dimension)
        if dimension in self._inner.shape:
            unstacked = self._inner.unstack(dimension)
            return tuple(CollapsedTensor(t, unstacked_shape) for t in unstacked)
        else:
            return (CollapsedTensor(self._inner, unstacked_shape),) * self.shape.get_size(dimension)

    def _with_shape_replaced(self, new_shape: Shape):
        if self.is_cached:
            return self._cached._with_shape_replaced(new_shape)
        else:
            inner_indices = [self.shape.index(d) for d in self._inner.shape.names]
            new_inner_shape = new_shape[inner_indices]
            result = CollapsedTensor(self._inner._with_shape_replaced(new_inner_shape), new_shape)
            return result

    @property
    def _is_tracer(self) -> bool:
        if self.is_cached:
            return self._cached._is_tracer
        else:
            return self._inner._is_tracer

    def _getitem(self, selection: dict):
        if self.is_cached:
            return self._cached._getitem(selection)
        else:
            inner_dict = {name: selection for name, selection in selection.items() if name in self._inner.shape}
            inner = self._inner._getitem(inner_dict)
            new_shape = self.shape.after_gather(selection)
            merge_shapes(inner.shape, new_shape)  # check that sizes match
            return CollapsedTensor(inner, new_shape)

    def flip(self, *dims: str) -> 'Tensor':
        if self.is_cached:
            return self._cached.flip(*dims)
        else:
            return CollapsedTensor(self._inner.flip(*dims), self._shape.flipped(dims))

    def _op1(self, native_function):
        if self.is_cached:
            return self._cached._op1(native_function)
        else:
            return CollapsedTensor(self._inner._op1(native_function), self._shape)

    def _op2(self, other, operator, native_function, op_name: str = 'unknown', op_symbol: str = '?'):
        try:
            other_t = self._tensor(other)
        except NoBackendFound:
            return NotImplemented
        if isinstance(other_t, CollapsedTensor) and other_t.is_cached:
            other_t = other_t._cached
        if isinstance(other_t, NativeTensor):
            if all([dim in other_t.shape for dim in self._shape.names]):  # other is dense and has all dimensions
                return op2_native(self, other_t, native_function)
            else:
                other_t = CollapsedTensor(other_t, other_t.shape)
        if isinstance(other_t, CollapsedTensor):
            other_inner = other_t._inner  # case that other is cached handled above
            self_inner = self._cached if self.is_cached else self._inner
            inner = operator(self_inner, other_inner)
            if all(dim in inner.shape for dim in self.shape.names + other_t.shape.names):  # shape already complete
                result = inner._with_shape_replaced(inner.shape._with_types(self._shape & other_t._shape))
                return result
            else:
                combined_shape = (self._shape & other_t._shape).with_sizes(inner.shape)
                return CollapsedTensor(inner, combined_shape)
        else:
            return NotImplemented

    def _natives(self) -> tuple:
        if self.is_cached:
            return self._cached._natives()
        else:
            return self._inner._natives()

    def _with_natives_replaced(self, natives: list):
        assert self.is_cached, "Cannot replace natives in uncached state. Expand tensor beforehand."
        return self._cached._with_natives_replaced(natives)

    def _expand(self):
        self._cache()
        # from phi.math import all_available
        # if not all_available(self._cached):
        #     raise AssertionError("Cannot cache a Tensor while it is being traced.")

    def _tensor_reduce(self,
                       dims: Tuple[str],
                       native_function: Callable,
                       collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                       unaffected_function: Callable = lambda value: value):
        if self.is_cached:
            return self._cached._tensor_reduce(dims, native_function, collapsed_function, unaffected_function)
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        inner_dims = [dim for dim in dims if dim in self._inner.shape]
        inner_reduce = self._inner._tensor_reduce(tuple(inner_dims), native_function, collapsed_function, unaffected_function)
        collapsed_dims = self._shape.without(self._inner.shape)
        final_shape = self._shape.without(dims)
        total_reduce = collapsed_function(inner_reduce, collapsed_dims.only(dims))
        return CollapsedTensor(total_reduce, final_shape)


class TensorStack(Tensor):
    """
    Implicit stack of multiple tensors.
    List of tensors, does not store stacked tensor in memory.

    Args:

    Returns:

    """

    def __init__(self, components: tuple or list, stack_dim: Shape):
        assert isinstance(stack_dim, Shape) and stack_dim.rank == 1, f"stack_dim must be a single-dimension Shape object but got {type(stack_dim)}"
        for t in components:
            assert isinstance(t, Tensor)
            assert t.dtype == components[0].dtype, f"Stacked tensors must have the same data type but got {[t.dtype for t in components]}"
            assert stack_dim.name not in t.shape, f"Cannot stack along '{stack_dim.name}' because the dimension already exists."
        self.tensors = tuple(components)
        self.stack_dim = stack_dim.with_sizes([len(components)])
        self._varying_shapes = any([v.shape != components[0].shape for v in components[1:]])
        self._shape = shape_stack(self.stack_dim, *[t.shape for t in self.tensors])
        self._cached = None

    @property
    def _is_tracer(self) -> bool:
        return any([t._is_tracer for t in self.tensors])

    @property
    def requires_broadcast(self):
        return self._varying_shapes or not self._shape.well_defined or self._is_tracer

    def _cache(self):
        if self._cached is None:
            if self.requires_broadcast:
                return None
            elif all([t.shape.is_uniform for t in self.tensors]):
                natives = [t.native(order=self._shape.names) for t in self.tensors]
                native = choose_backend(*natives).concat(natives, axis=self.shape.index(self.stack_dim.name))
                self._cached = NativeTensor(native, self._shape)
            else:  # cache stack_dim on inner tensors
                non_uniform_dim = self.tensors[0].shape.shape.without('dims')
                unstacked = [t.unstack(non_uniform_dim.name) for t in self.tensors]
                stacked = []
                for to_stack in zip(*unstacked):
                    tensor = TensorStack(to_stack, self.stack_dim)._cache()
                    stacked.append(tensor)
                self._cached = TensorStack(stacked, non_uniform_dim)
        return self._cached

    @property
    def dtype(self):
        return self.tensors[0].dtype

    @property
    def shape(self):
        return self._shape

    def native(self, order: str or tuple or list or Shape = None):
        if self._cached is not None:
            return self._cached.native(order=order)
        else:
            order = parse_dim_order(order, check_rank=self.rank)
            # Is only the stack dimension shifted?
            if order is not None and self._shape.without(self.stack_dim).names == tuple(filter(lambda name: name != self.stack_dim.name, order)):
                inner_order = [dim for dim in order if dim != self.stack_dim.name]
                natives = [t.native(inner_order) for t in self.tensors]
                assert self.stack_dim.name in order, f"Dimension {self.stack_dim} missing from 'order'. Got {order} but tensor has shape {self.shape}."
                native = choose_backend(*natives).stack(natives, axis=order.index(self.stack_dim.name))
                return native
            assert not self.shape.is_non_uniform, f"Cannot convert non-uniform tensor with shape {self.shape} to native tensor."
            return self._cache().native(order=order)

    def _with_shape_replaced(self, new_shape: Shape):
        if self._cached is not None:
            return self._cached._with_shape_replaced(new_shape)
        else:
            new_stack_dim = new_shape[self._shape.index(self.stack_dim.name)]
            inner_shape = new_shape.without(new_stack_dim)
            tensors = [t._with_shape_replaced(inner_shape) for t in self.tensors]
            return TensorStack(tensors, new_stack_dim)

    def _getitem(self, selection: dict):
        if self._cached is not None:
            return self._cached._getitem(selection)
        if (self.stack_dim.name not in selection or len(selection) != 1) and not self.requires_broadcast:
            return self._cache()._getitem(selection)
        # --- Inner dims ---
        inner_dict = {dim: sel for dim, sel in selection.items() if dim != self.stack_dim.name}
        tensors = self.tensors
        if len(inner_dict) > 0:
            tensors = [t[inner_dict] for t in tensors]
        # --- stack dimension ---
        if self.stack_dim.name in selection:
            selection = selection[self.stack_dim.name]
            if isinstance(selection, int):
                return self.tensors[selection]
            elif isinstance(selection, slice):
                return TensorStack(tensors[selection], self.stack_dim)
            else:
                raise NotImplementedError(f"{type(selection)} not supported. Only (int, slice) allwoed")
        else:
            return TensorStack(tensors, self.stack_dim)

    def flip(self, *dims: str) -> 'Tensor':
        if self._cached is not None:
            return self._cached.flip(*dims)
        else:
            tensors = [t.flip(*dims) for t in self.tensors]
            if self.stack_dim.name in dims:
                tensors = tensors[::-1]
            return TensorStack(tensors, self.stack_dim)

    def unstack(self, dimension):
        if self._cached is not None:
            return self._cached.unstack(dimension)
        if dimension == self.stack_dim.name:
            return self.tensors
        else:
            if self.requires_broadcast:
                unstacked = [t.unstack(dimension) for t in self.tensors]
                result = [TensorStack(items, self.stack_dim) for items in zip(*unstacked)]
                return result
            else:
                return self._cache().unstack(dimension=dimension)

    def _op1(self, native_function):
        if self.requires_broadcast:
            tensors = [t._op1(native_function) for t in self.tensors]
            return TensorStack(tensors, self.stack_dim)
        else:
            return self._cache()._op1(native_function)

    def _op2(self, other, operator, native_function, op_name: str = 'unknown', op_symbol: str = '?'):
        other = self._tensor(other)
        if self.requires_broadcast:
            if self.stack_dim.name in other.shape:
                other = other.unstack(self.stack_dim.name)
                tensors = [operator(t1, t2) for t1, t2 in zip(self.tensors, other)]
            else:
                tensors = [operator(t, other) for t in self.tensors]
            return TensorStack(tensors, self.stack_dim)
        elif isinstance(other, (CollapsedTensor, NativeTensor)):
            return op2_native(self, other, native_function)
        elif isinstance(other, TensorStack) and not other.requires_broadcast:
            return op2_native(self, other, native_function)
        else:
            return NotImplemented

    def _natives(self) -> tuple:
        if self._cached is not None:
            return self._cached._natives()
        else:
            return sum([t._natives() for t in self.tensors], ())

    def _with_natives_replaced(self, natives: list):
        if self._cached is not None:
            return self._cached._with_natives_replaced(natives)
        else:
            tensors = [t._with_natives_replaced(natives) for t in self.tensors]
            return TensorStack(tensors, self.stack_dim)

    def _expand(self):
        if self.requires_broadcast:
            for t in self.tensors:
                t._expand()
        self._cache()
        if self._cached is not None:
            from phi.math import all_available
            assert all_available(self._cached), "Cannot cache a Tensor while it is being traced."

    def _simplify(self):
        if self._cached is not None:
            return self._cached
        else:
            return self

    def _tensor_reduce(self,
                       dims: Tuple[str],
                       native_function: Callable,
                       collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                       unaffected_function: Callable = lambda value: value):
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        if self._cached is not None:
            return self._cached._tensor_reduce(dims, native_function, collapsed_function, unaffected_function)
        # --- inner reduce ---
        inner_axes = [dim for dim in dims if dim != self.stack_dim.name]
        red_inners = [t._tensor_reduce(inner_axes, native_function, collapsed_function, unaffected_function) for t in self.tensors]
        # --- outer reduce ---
        if self.stack_dim.name in dims:
            if any([t._is_tracer for t in red_inners]):
                return sum(red_inners[1:], red_inners[0])  # TODO this may not always be the sum
            else:
                inner_order = red_inners[0].shape.names
                natives = [t.native(inner_order) for t in red_inners]
                backend = choose_backend(*natives)
                result = native_function(backend, backend.stack(natives), dim=0)  # TODO not necessary if tensors are CollapsedTensors
                return NativeTensor(result, red_inners[0].shape)
        else:
            return TensorStack(red_inners, self.stack_dim)


def tensor(data: Tensor or Shape or tuple or list or numbers.Number,
           *shape: Shape,
           convert: bool = True,
           default_list_dim=channel('vector')) -> Tensor:  # TODO assume convert_unsupported, add convert_external=False for constants
    """
    Create a Tensor from the specified `data`.
    If `convert=True`, converts `data` to the preferred format of the default backend.

    `data` must be one of the following:
    
    * Number: returns a dimensionless Tensor.
    * Native tensor such as NumPy array, TensorFlow tensor or PyTorch tensor.
    * `tuple` or `list` of numbers: backs the Tensor with native tensor.
    * `tuple` or `list` of non-numbers: creates tensors for the items and stacks them.
    * Tensor: renames dimensions and dimension types if `names` is specified. Converts all internal native values of the tensor if `convert=True`.
    * Shape: creates a 1D tensor listing the dimension sizes.
    
    While specifying `names` is optional in some cases, it is recommended to always specify them.
    
    Dimension types are always inferred from the dimension names if specified.

    Implementations:

    * NumPy: [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
    * PyTorch: [`torch.tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html), [`torch.from_numpy`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html)
    * TensorFlow: [`tf.convert_to_tensor`](https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor)
    * Jax: [`jax.numpy.array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)

    See Also:
        `phi.math.wrap()` which uses `convert=False`, `layout()`.

    Args:
      data: native tensor, scalar, sequence, Shape or Tensor
      shape: Ordered dimensions and types. If sizes are defined, they will be checked against `data`.`
      convert: If True, converts the data to the native format of the current default backend.
        If False, wraps the data in a `Tensor` but keeps the given data reference if possible.

    Raises:
      AssertionError: if dimension names are not provided and cannot automatically be inferred
      ValueError: if `data` is not tensor-like

    Returns:
      Tensor containing same values as data
    """
    assert all(isinstance(s, Shape) for s in shape), f"Cannot create tensor because shape needs to be one or multiple Shape instances but got {shape}"
    shape = None if len(shape) == 0 else concat_shapes(*shape)
    if isinstance(data, Tensor):
        if convert:
            backend = data.default_backend
            if backend != default_backend():
                data = data._op1(lambda n: convert_(n, use_dlpack=False))
        if shape is None:
            return data
        else:
            if None in shape.sizes:
                shape = shape.with_sizes(data.shape.sizes)
            return data._with_shape_replaced(shape)
    elif isinstance(data, Shape):
        if shape is None:
            shape = channel('dims')
        else:
            assert shape.rank == 1, "Can only convert 1D shapes to Tensors"
        shape = shape._with_item_names((data.names,))
        data = data.sizes
    elif isinstance(data, (numbers.Number, bool, str)):
        assert not shape, f"Trying to create a zero-dimensional Tensor from value '{data}' but shape={shape}"
        if convert:
            data = default_backend().as_tensor(data, convert_external=True)
        return NativeTensor(data, EMPTY_SHAPE)
    if isinstance(data, (tuple, list)):
        if all([isinstance(d, (bool, int, float, complex, str)) for d in data]):
            array = np.array(data)
            assert array.dtype != object
            data = array
        else:
            inner_shape = [] if shape is None else [shape[1:]]
            tensors = [d if isinstance(d, Tensor) else tensor(d, *inner_shape, convert=convert) for d in data]
            common_shape = merge_shapes(*[e.shape for e in tensors])
            stack_dim = default_list_dim if shape is None else shape[0].with_sizes([len(tensors)])
            assert all(stack_dim not in t.shape for t in tensors), f"Cannot stack tensors with dimension '{stack_dim}' because a tensor already has that dimension."
            elements = [CollapsedTensor(e, common_shape) if e.shape.rank < common_shape.rank else e for e in tensors]
            from ._ops import cast_same
            elements = cast_same(*elements)
            return TensorStack(elements, stack_dim)
    try:
        backend = choose_backend(data)
        if shape is None:
            assert backend.ndims(data) <= 1, "Specify dimension names for tensors with more than 1 dimension"
            shape = default_list_dim if backend.ndims(data) == 1 else EMPTY_SHAPE
            shape = shape.with_sizes(backend.staticshape(data))
        else:
            # fill in sizes or check them
            sizes = backend.staticshape(data)
            if len(sizes) != len(shape):
                raise IncompatibleShapes(f"Rank of given shape {shape} does not match data with sizes {sizes}")
            for size, s in zip(sizes, shape.sizes):
                if s is not None:
                    assert s == size, f"Given shape {shape} does not match data with sizes {sizes}. Consider leaving the sizes undefined."
            shape = shape.with_sizes(sizes)
        if convert:
            data = convert_(data, use_dlpack=False)
        return NativeTensor(data, shape)
    except NoBackendFound:
        raise ValueError(f"{type(data)} is not supported. Only (Tensor, tuple, list, np.ndarray, native tensors) are allowed.\nCurrent backends: {BACKENDS}")


def wrap(data: Tensor or Shape or tuple or list or numbers.Number,
         *shape: Shape) -> Tensor:
    """ Short for `phi.math.tensor()` with `convert=False`. """
    return tensor(data, *shape, convert=False)  # TODO inline, simplify


def layout(objects, *shape: Shape) -> Tensor:
    """
    Wraps a Python tree in a `Tensor`, allowing elements to be accessed via dimensions.
    A python tree is a structure of nested `tuple`, `list`, `dict` and *leaf* objects where leaves can be any Python object.

    All keys of `dict` containers must be of type `str`.
    The keys are automatically assigned as item names along that dimension unless conflicting with other elements.

    Strings may also be used as containers.

    Example:
    ```python
    t = layout({'a': 'text', 'b': [0, 1]}, channel('dict,inner'))
    t.inner[1].dict['a'].native()  # returns 'e'
    ```

    See Also:
        `tensor()`, `wrap()`.

    Args:
        objects: PyTree of `list` or `tuple`.
        *shape: Tensor dimensions

    Returns:
        `Tensor`.
        Calling `Tensor.native()` on the returned tensor will return `objects`.
    """
    assert all(isinstance(s, Shape) for s in shape), f"shape needs to be one or multiple Shape instances but got {shape}"
    shape = EMPTY_SHAPE if len(shape) == 0 else concat_shapes(*shape)
    if isinstance(objects, Layout):
        assert objects.shape == shape
        return objects

    if not shape.well_defined:

        def recursive_determine_shape(native, shape: Shape):
            if not shape:
                return shape
            if isinstance(native, dict):
                assert all([isinstance(k, str) for k in native.keys()]), f"All dict keys in PyTrees must be str but got {tuple(native.keys())}"
                shape = shape._with_item_name(shape.names[0], tuple(native.keys()))
            if shape.rank == 1:
                return shape.with_sizes((len(native),))
            inner_shape = shape[1:]
            inner_shapes = [recursive_determine_shape(n, inner_shape) for n in native]
            return shape_stack(shape[0], *inner_shapes)

        shape = recursive_determine_shape(objects, shape)

    return Layout(objects, shape)
    # if shape.volume == 1:
    #     objects = np.asarray(objects, dtype=np.object)
    #
    # if isinstance(objects, (tuple, list)):
    #     objects = np.asarray(objects, dtype=np.object)
    # if isinstance(objects, np.ndarray) and objects.dtype == np.object:
    #     return Layout(objects, shape)
    # else:
    #     assert shape.volume == 1, f"Cannot layout object of type {objects} along {shape}, a tuple, list or object array is required."


def compatible_tensor(data, compat_shape: Shape = None, compat_natives=(), convert=False):
    if isinstance(data, Tensor):
        return data
    elif isinstance(data, Shape):
        assert compat_shape.channel.rank == 1, "Only single-channel tensors support implicit casting from Shape to tensor"
        assert data.rank == compat_shape.channel.volume
        return wrap(data.spatial.sizes, *compat_shape.channel._with_item_names((data.names,)))
    else:
        backend = choose_backend(*compat_natives, data)
        try:
            other_tensor = backend.as_tensor(data, convert_external=convert)
            shape = backend.staticshape(other_tensor)
        except ValueError as e:
            raise ValueError(e)
        if len(shape) == 0:
            return NativeTensor(other_tensor, EMPTY_SHAPE)
        elif len(shape) == compat_shape.rank:
            return NativeTensor(other_tensor, compat_shape.with_sizes(shape))  # TODO this can lead to errors, remove?
        elif len(shape) == compat_shape.channel.rank:
            other_tensor = wrap(data, compat_shape.channel)
            return other_tensor
        elif len(shape) == 1:
            return NativeTensor(other_tensor, Shape(shape, ('vector',), (CHANNEL_DIM,), (None,)))
        else:
            raise ValueError("Cannot broadcast object of rank %d to tensor with shape %s" % (backend.ndims(data), compat_shape))


def broadcastable_native_tensors(*tensors):
    """
    Expands and transposes the dimensions of the given tensors so that they all have the same dimension order.

    Args:
      tensors: sequence of Tensors
      *tensors: 

    Returns:
      shape, native tensors)

    """
    broadcast_shape = merge_shapes(*[t.shape for t in tensors])
    natives = [t.native(order=broadcast_shape.names) if t.rank > 0 else t.native() for t in tensors]
    return broadcast_shape, natives


def op2_native(x: Tensor, y: Tensor, native_function: Callable):
    new_shape, (native1, native2) = broadcastable_native_tensors(x, y)
    result_tensor = native_function(native1, native2)
    return NativeTensor(result_tensor, new_shape)


def custom_op2(x: Tensor or float, y: Tensor or float, l_operator, l_native_function, r_operator=None, r_native_function=None, op_name: str = 'unknown') -> Tensor:
    """
    Perform a custom operator on two tensors.
    This method first tries calling _op2() on the first tensor and if that fails, tries it on the second tensor.

    Args:
      x: Tensor or float: 
      y: Tensor or float: 
      l_operator: 
      l_native_function: 
      r_operator:  (Default value = None)
      r_native_function:  (Default value = None)
      op_name: Name of the operator function for debugging purposes. Leading 'r' will be added for the operand-reversed version.

    Returns:
        `Tensor`
    """
    x = wrap(x)
    y = wrap(y)
    result = x._op2(y, l_operator, l_native_function, op_name, op_name)
    if result is NotImplemented:
        result = y._op2(x, r_operator or l_operator, r_native_function or l_native_function, f'r{op_name}', op_name)
        if result is NotImplemented:
            raise NotImplementedError(f"Operation not supported between {type(x)} and {type(y)}")
    return result


def disassemble_tensors(obj: Tensor or tuple or list, expand=True) -> tuple:
    assert isinstance(obj, (Tensor, tuple, list)), f"jit-compiled function returned {type(obj)} but must return either a 'phi.math.Tensor' or tuple/list of tensors."
    if isinstance(obj, Tensor):
        if expand:
            obj._expand()
        return obj._natives(), obj.shape
    else:
        if expand:
            for t in obj:
                t._expand()
        return sum([t._natives() for t in obj], ()), tuple(t.shape for t in obj)


def assemble_tensors(natives: tuple, shapes: Shape or Tuple[Shape]):
    natives = list(natives)
    if isinstance(shapes, Shape):
        return _assemble_pop(natives, shapes)
    else:
        return [_assemble_pop(natives, shape) for shape in shapes]


def _assemble_pop(natives: list, shape: Shape):
    if shape.is_uniform:
        native = natives.pop(0)
        ndim = choose_backend(native).ndims(native)
        if ndim != shape.rank:
            if ndim == 0 and shape.rank > 0:
                inner = NativeTensor(native, EMPTY_SHAPE)
                return CollapsedTensor(inner, shape)
            else:
                raise NotImplementedError("Cannot restore CollapsedTensor from native and shape")
        return NativeTensor(native, shape)
    else:
        s2 = shape.shape.without('dims')
        if len(s2) > 1:
            raise NotImplementedError('More than one non-uniform dimension not supported.')
        shapes = shape.unstack(s2.name)
        tensors = [NativeTensor(natives.pop(0), s) for s in shapes]
        from phi.math._ops import stack
        return TensorStack(tensors, s2)


class _TensorLikeType(type):

    def __instancecheck__(self, instance):
        if isinstance(instance, Tensor):
            return True
        if isinstance(instance, type(MISSING_TENSOR)) and instance == MISSING_TENSOR:
            return True
        if instance is None or isinstance(instance, Tensor):
            return True
        elif isinstance(instance, (tuple, list)):
            return all(isinstance(item, TensorLike) for item in instance)
        elif isinstance(instance, Dict):
            return True
        elif isinstance(instance, dict):
            return all(isinstance(name, str) for name in instance.keys()) and all(isinstance(val, TensorLike) for val in instance.values())
        else:
            return hasattr(instance, '__variable_attrs__') or hasattr(instance, '__value_attrs__')


class TensorLike(metaclass=_TensorLikeType):
    """
    Tensor-like objects can interoperate with some `phi.math` functions, depending on what methods they implement.
    Objects are considered `TensorLike` if they implement `TensorLike.__variable_attrs__()` or `TensorLike.__value_attrs__()`.
    This is reflected in `isinstance` checks.

    `TensorLike` objects may be used as keys, for example in `jit_compile()`.
    In key mode, all variable attributes are set to `None`.
    When used as keys, `TensorLike` should also implement `__eq__()` to compare any non-variable properties that can affect a function.

    Do not declare this class as a superclass.
    """

    def __value_attrs__(self) -> Tuple[str]:
        """
        Returns all `Tensor` or `TensorLike` attribute names of `self` that should be transformed by single-operand math operations,
        such as `sin()`, `exp()`.

        Returns:
            `tuple` of `str` attributes.
                Calling `getattr(self, attr)` must return a `Tensor` or `TensorLike` for all returned attributes.
        """
        raise NotImplementedError()

    def __variable_attrs__(self) -> Tuple[str]:
        """
        Returns all `Tensor` or `TensorLike` attribute names of `self` whose values are variable.
        Variables denote values that can change from one function call to the next or for which gradients can be recorded.
        If this method is not implemented, all attributes returned by `__value_attrs__()` are considered variable.

        The returned properties are used by the following functions:

        - `jit_compile()`
        - `jit_compile_linear()`
        - `stop_gradient()`
        - `functional_gradient()`
        - `custom_gradient()`

        Returns:
            `tuple` of `str` attributes.
                Calling `getattr(self, attr)` must return a `Tensor` or `TensorLike` for all returned attributes.
        """
        raise NotImplementedError()

    def __with_attrs__(self, **attrs):
        """
        Creates a copy of this object which has the `Tensor` or `TensorLike` attributes contained in `tattrs` replaced.
        If this method is not implemented, tensor attributes are replaced using `setattr()`.

        Args:
            **attrs: `dict` mapping `str` attribute names to `Tensor` or `TensorLike`.

        Returns:
            Altered copy of `self`
        """
        raise NotImplementedError()


def copy_with(obj, **tensor_attributes):
    if hasattr(obj, '__with_tattrs__'):
        return obj.__with_tattrs__(**tensor_attributes)
    else:
        cpy = copy.copy(obj)
        for attr, value in tensor_attributes.items():
            setattr(cpy, attr, value)
        return cpy


def variable_attributes(obj) -> Tuple[str]:
    if hasattr(obj, '__variable_attrs__'):
        return obj.__variable_attrs__()
    elif hasattr(obj, '__value_attrs__'):
        return obj.__value_attrs__()
    else:
        raise ValueError(f"Not TensorLike: {type(obj)}")


def value_attributes(obj):
    assert hasattr(obj, '__value_attrs__'), f"{type(obj)} must implement '__value_attrs__()' to be used with value functions."
    return obj.__value_attrs__()


def variable_values(obj):
    assert hasattr(obj, '__value_attrs__'), f"{type(obj)} must implement '__value_attrs__()' to be used with value functions."
    if hasattr(obj, '__variable_attrs__'):
        values = obj.__value_attrs__()
        variables = obj.__variable_attrs__()
        return [a for a in values if a in variables]
    else:
        return obj.__value_attrs__()



TensorLikeType = TypeVar('TensorLikeType')


MISSING_TENSOR = 'missing'


def disassemble_tree(obj: TensorLikeType) -> Tuple[TensorLikeType, List[Tensor]]:
    """
    Splits a nested structure of Tensors into the structure without the tensors and an ordered list of tensors.
    Native tensors will be wrapped in phi.math.Tensors with default dimension names and dimension types `None`.

    See Also:
        `assemble_tree()`

    Args:
        obj: Nested structure of `Tensor` objects.
            Nested structures include: `tuple`, `list`, `dict`, `TensorLike`.

    Returns:
        empty structure: Same structure as `obj` but with the tensors replaced by `None`.
        tensors: Ordered `list` of all contained `Tensor` objects.
    """
    if obj is None:
        return MISSING_TENSOR, []
    elif isinstance(obj, Tensor):
        return None, [obj]
    elif isinstance(obj, (tuple, list)):
        keys = []
        values = []
        for item in obj:
            key, value = disassemble_tree(item)
            keys.append(key)
            values.extend(value)
        return (tuple(keys) if isinstance(obj, tuple) else keys), values
    elif isinstance(obj, dict):
        keys = {}
        values = []
        for name, item in obj.items():
            key, value = disassemble_tree(item)
            keys[name] = key
            values.extend(value)
        return keys, values
    elif isinstance(obj, TensorLike):
        attributes = variable_attributes(obj)
        keys = {}
        values = []
        for attr in attributes:
            key, value = disassemble_tree(getattr(obj, attr))
            keys[attr] = key
            values.extend(value)
        return copy_with(obj, **keys), values
    else:
        backend = choose_backend(obj)
        sizes = backend.staticshape(obj)
        shape = Shape(sizes, tuple([f"dim{i}" for i in range(len(sizes))]), (None,) * len(sizes), (None,) * len(sizes))
        shape.is_native_shape = True
        # if backend.ndims(obj) != 0:
        #     warnings.warn(f"Only scalar native tensors should be used in function inputs/outputs but got tensor with shape {backend.staticshape(obj)}. Consider using phi.math.Tensor instances instead. Using shape {shape}.")
        return None, [NativeTensor(obj, shape)]


def assemble_tree(obj: TensorLikeType, values: List[Tensor]) -> TensorLikeType:
    """ Reverses `disassemble_tree()` given an empty nested structure and a list of tensors. """
    if obj == MISSING_TENSOR:
        return None
    elif obj is None:
        assert isinstance(values[0], Tensor)
        value = values.pop(0)
        if value.shape.rank > 0 and all([t is None for t in value.shape.types]):
            assert value.shape.is_native_shape  # custom attribute set in disassemble_tree
            return value.native(value.shape)
        elif hasattr(value.shape, 'is_native_shape') and value.shape.is_native_shape:
            return value.native(value.shape)
        else:
            return value
    elif isinstance(obj, list):
        return [assemble_tree(item, values) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([assemble_tree(item, values) for item in obj])
    elif isinstance(obj, dict):
        return {name: assemble_tree(val, values) for name, val in obj.items()}
    elif isinstance(obj, TensorLike):
        attributes = variable_attributes(obj)
        values = {a: assemble_tree(getattr(obj, a), values) for a in attributes}
        return copy_with(obj, **values)
    else:
        raise ValueError(f"Value must be Tensor or tensor-like but got {type(obj)}")


def cached(t: Tensor or TensorLike) -> Tensor or TensorLike:
    assert isinstance(t, (Tensor, TensorLike)), f"All arguments must be Tensors but got {type(t)}"
    if isinstance(t, NativeTensor):
        return t
    elif isinstance(t, CollapsedTensor):
        if t.is_cached:
            return t._cached
        if t._inner._is_tracer:
            return t
        if t.shape.is_uniform:
            native = t._inner.native(order=t.shape.names)
            multiples = [1 if name in t._inner.shape else size for size, name, *_ in t.shape._dimensions]
            backend = choose_backend(native)
            tiled = backend.tile(native, multiples)
            return NativeTensor(tiled, t.shape)
        else:
            raise NotImplementedError()
    elif isinstance(t, TensorStack):
        if t._cached is not None:
            return t._cached
        inners = cached(t.tensors)
        if t.requires_broadcast:
            return TensorStack(inners, t.stack_dim)
        else:
            natives = [t.native(order=t.shape.names) for t in inners]
            native = choose_backend(*natives).stack(natives, axis=t.shape.index(t.stack_dim.name))
            return NativeTensor(native, t.shape)
    elif isinstance(t, TensorLike):
        tree, tensors = disassemble_tree(t)
        tensors_ = [cached(t_) for t_ in tensors]
        return assemble_tree(tree, tensors_)
    else:
        raise AssertionError(f"Cannot cache {type(t)} {t}")


class Dict(dict):
    """
    Dictionary of `Tensor` or `TensorLike` values.
    In addition to dictionary functions, supports mathematical operators with other `Dict`s and lookup via `.key` syntax.
    `Dict` implements `TensorLike` so instances can be passed to math operations like `sin`.
    """

    def __value_attrs__(self):
        return tuple(self.keys())
    
    # --- Dict[key] ---

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)
        
    # --- operators ---
    
    def __neg__(self):
        return Dict({k: -v for k, v in self.items()})
    
    def __invert__(self):
        return Dict({k: ~v for k, v in self.items()})
    
    def __abs__(self):
        return Dict({k: abs(v) for k, v in self.items()})
    
    def __round__(self, n=None):
        return Dict({k: round(v) for k, v in self.items()})

    def __add__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val + other[key] for key, val in self.items()})
        else:
            return Dict({key: val + other for key, val in self.items()})

    def __radd__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] + val for key, val in self.items()})
        else:
            return Dict({key: other + val for key, val in self.items()})

    def __sub__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val - other[key] for key, val in self.items()})
        else:
            return Dict({key: val - other for key, val in self.items()})

    def __rsub__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] - val for key, val in self.items()})
        else:
            return Dict({key: other - val for key, val in self.items()})

    def __mul__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val * other[key] for key, val in self.items()})
        else:
            return Dict({key: val * other for key, val in self.items()})

    def __rmul__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] * val for key, val in self.items()})
        else:
            return Dict({key: other * val for key, val in self.items()})

    def __truediv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val / other[key] for key, val in self.items()})
        else:
            return Dict({key: val / other for key, val in self.items()})

    def __rtruediv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] / val for key, val in self.items()})
        else:
            return Dict({key: other / val for key, val in self.items()})

    def __floordiv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val // other[key] for key, val in self.items()})
        else:
            return Dict({key: val // other for key, val in self.items()})

    def __rfloordiv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] // val for key, val in self.items()})
        else:
            return Dict({key: other // val for key, val in self.items()})

    def __pow__(self, power, modulo=None):
        assert modulo is None
        if isinstance(power, Dict):
            return Dict({key: val ** power[key] for key, val in self.items()})
        else:
            return Dict({key: val ** power for key, val in self.items()})

    def __rpow__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] ** val for key, val in self.items()})
        else:
            return Dict({key: other ** val for key, val in self.items()})

    def __mod__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val % other[key] for key, val in self.items()})
        else:
            return Dict({key: val % other for key, val in self.items()})

    def __rmod__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] % val for key, val in self.items()})
        else:
            return Dict({key: other % val for key, val in self.items()})

    def __eq__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val == other[key] for key, val in self.items()})
        else:
            return Dict({key: val == other for key, val in self.items()})

    def __ne__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val != other[key] for key, val in self.items()})
        else:
            return Dict({key: val != other for key, val in self.items()})

    def __lt__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val < other[key] for key, val in self.items()})
        else:
            return Dict({key: val < other for key, val in self.items()})

    def __le__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val <= other[key] for key, val in self.items()})
        else:
            return Dict({key: val <= other for key, val in self.items()})

    def __gt__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val > other[key] for key, val in self.items()})
        else:
            return Dict({key: val > other for key, val in self.items()})

    def __ge__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val >= other[key] for key, val in self.items()})
        else:
            return Dict({key: val >= other for key, val in self.items()})

    # --- overridden methods ---

    def copy(self):
        return Dict(self)


def to_dict(value: Tensor or Shape):
    """
    Returns a serializable form of a `Tensor` or `Shape`.
    The result can be written to a JSON file, for example.

    See Also:
        `from_dict()`.

    Args:
        value: `Tensor` or `Shape`

    Returns:
        Serializable Python tree of primitives
    """
    if isinstance(value, Shape):
        return value._to_dict(include_sizes=True)
    elif isinstance(value, Tensor):
        return value._to_dict()
    raise ValueError(f"Cannot convert {value} to a dict")


def from_dict(dict_: dict, convert=False):
    """
    Loads a `Tensor` or `Shape` from a serialized form.

    See Also:
        `to_dict()`.

    Args:
        dict_: Serialized tensor properties.
        convert: Whether to convert the data to the current backend format or keep it as a Numpy array.

    Returns:
        `Tensor` or `Shape`.
    """
    shape = Shape._from_dict(dict_)
    if 'data' in dict_:
        return tensor(dict_['data'], shape, convert=convert)
    else:
        return shape

import numbers
import warnings
from typing import Tuple

import numpy as np

from._config import GLOBAL_AXIS_ORDER
from . import _shape, DType
from .backend import NoBackendFound, choose_backend, BACKENDS, get_precision, default_backend
from ._shape import Shape, CHANNEL_DIM, BATCH_DIM, SPATIAL_DIM, EMPTY_SHAPE


class Tensor:
    """
    Abstract base class to represent structured data of one data type.

    Unlike with `numpy.ndarray`, the dimensions of Tensors have names and types.
    Additionally, tensors can have non-uniform shapes, meaning that the size of dimensions can vary along other dimensions.

    To check whether a value is a tensor, use `isinstance(value, Tensor)`.

    To construct a Tensor, use `tensor()` or one of the basic tensor creation functions,
    see https://tum-pbs.github.io/PhiFlow/Math.html#tensor-creation .

    Tensors are not editable.
    When backed by an editable native tensor, e.g. a `numpy.ndarray`, do not edit the underlying data structure.
    """

    def native(self, order: str or tuple or list = None):
        """
        Returns a native tensor object with the dimensions ordered according to `order`.
        
        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
        
        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        Args:
          order: optional) list of dimension names. If not given, the current order is kept.
          order: str or tuple or list:  (Default value = None)

        Returns:
          native tensor object
          :raise: ValueError if the tensor cannot be transposed to match target_shape

        """
        raise NotImplementedError()

    def numpy(self, order: str or tuple or list = None) -> np.ndarray:
        """
        Returns this tensor as a NumPy ndarray object with dimensions ordered according to `order`.
        
        *Note*: Using this function breaks the autograd chain. The returned tensor is not differentiable.
        To get a differentiable tensor, use :func:`Tensor.native` instead.
        
        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
        
        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        Args:
          order: optional) list of dimension names. If not given, the current order is kept.
          order: str or tuple or list:  (Default value = None)

        Returns:
          NumPy representation
          :raise: ValueError if the tensor cannot be transposed to match target_shape

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

    def _with_shape_replaced(self, new_shape):
        raise NotImplementedError()

    @property
    def rank(self) -> int:
        """ Equal to `tensor.shape.rank`. """
        return self.shape.rank

    @property
    def _is_special(self) -> bool:
        """
        Special tensors store additional internal information.
        They should not be converted to native() in intermediate operations.
        
        Tracking tensors are special tensors.
        
        TensorStack prevents performing the actual stack operation if one of its component tensors is special.

        Args:

        Returns:

        """
        raise NotImplementedError()

    def __len__(self):
        return self.shape.volume if self.rank == 1 else NotImplemented

    def __bool__(self):
        if self.rank == 0:
            return bool(self.native())
        else:
            from phi.math._functions import all_
            return bool(all_(self))

    def __int__(self):
        return int(self.native()) if self.rank == 0 else NotImplemented

    def __float__(self):
        return float(self.native()) if self.rank == 0 else NotImplemented

    def __complex__(self):
        return complex(self.native()) if self.rank == 0 else NotImplemented

    def __index__(self):
        return int(self.native()) if self.rank == 0 and np.issubdtype(self.dtype, int) else NotImplemented

    def _summary_str(self) -> str:
        try:
            from ._functions import all_available, min_, max_
            if all_available(self):
                if self.rank == 0:
                    return str(self.numpy())
                elif self.shape.volume is not None and self.shape.volume <= 6:
                    content = list(np.reshape(self.numpy(), [-1]))
                    content = ', '.join([repr(number) for number in content])
                    if self.shape.rank == 1 and (self.dtype.kind in (bool, int) or self.dtype.precision == get_precision()):
                        return f"({content}) along {self.shape.name}"
                    return f"{self.shape} {self.dtype}  {content}"
                else:
                    min_val, max_val = min_(self), max_(self)
                    return f"{self.shape} {self.dtype}  {min_val} < ... < {max_val}"
            else:
                if self.rank == 0:
                    return f"scalar {self.dtype}"
                else:
                    return f"{self.shape} {self.dtype}"
        except BaseException as err:
            return f"{self.shape}, failed to fetch values with error {err}"

    def __repr__(self):
        return self._summary_str()

    def __getitem__(self, item):
        if isinstance(item, Tensor):
            from ._functions import gather
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
                warnings.warn("Slicing with sequence should only be used for channel dimensions.")
                item = {name: selection for name, selection in zip(self.shape.names, item)}
        assert isinstance(item, dict)  # dict mapping name -> slice/int
        return self._getitem(item)

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

    def dimension(self, name) -> 'TensorDim':
        """
        Returns a reference to a specific dimension of this tensor.
        This is equivalent to the syntax `tensor.<name>`.

        The dimension need not be part of the `Tensor.shape` in which case its size is 1.

        Args:
            name: dimension name

        Returns:
            `TensorDim` corresponding to a dimension of this tensor
        """
        return TensorDim(self, name)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        assert name not in ('shape', '_shape', 'tensor'), name
        return TensorDim(self, name)

    def __add__(self, other):
        return self._op2(other, lambda x, y: x + y, lambda x, y: choose_backend(x, y).add(x, y))

    def __radd__(self, other):
        return self._op2(other, lambda x, y: y + x, lambda x, y: choose_backend(x, y).add(y, x))

    def __sub__(self, other):
        return self._op2(other, lambda x, y: x - y, lambda x, y: choose_backend(x, y).sub(x, y))

    def __rsub__(self, other):
        return self._op2(other, lambda x, y: y - x, lambda x, y: choose_backend(x, y).sub(y, x))

    def __and__(self, other):
        return self._op2(other, lambda x, y: x & y, lambda x, y: x & y)

    def __or__(self, other):
        return self._op2(other, lambda x, y: x | y, lambda x, y: x | y)

    def __xor__(self, other):
        return self._op2(other, lambda x, y: x ^ y, lambda x, y: x ^ y)

    def __mul__(self, other):
        return self._op2(other, lambda x, y: x * y, lambda x, y: choose_backend(x, y).mul(x, y))

    def __rmul__(self, other):
        return self._op2(other, lambda x, y: y * x, lambda x, y: choose_backend(x, y).mul(y, x))

    def __truediv__(self, other):
        return self._op2(other, lambda x, y: x / y, lambda x, y: choose_backend(x, y).div(x, y))

    def __rtruediv__(self, other):
        return self._op2(other, lambda x, y: y / x, lambda x, y: choose_backend(x, y).div(y, x))

    def __divmod__(self, other):
        return self._op2(other, lambda x, y: divmod(x, y), lambda x, y: divmod(x, y))

    def __rdivmod__(self, other):
        return self._op2(other, lambda x, y: divmod(y, x), lambda x, y: divmod(y, x))

    def __floordiv__(self, other):
        return self._op2(other, lambda x, y: x // y, lambda x, y: x // y)

    def __rfloordiv__(self, other):
        return self._op2(other, lambda x, y: y // x, lambda x, y: y // x)

    def __pow__(self, power, modulo=None):
        assert modulo is None
        return self._op2(power, lambda x, y: x ** y, lambda x, y: choose_backend(x, y).pow(x, y))

    def __rpow__(self, other):
        return self._op2(other, lambda x, y: y ** x, lambda x, y: choose_backend(x, y).pow(y, x))

    def __mod__(self, other):
        return self._op2(other, lambda x, y: x % y, lambda x, y: choose_backend(x, y).mod(x, y))

    def __rmod__(self, other):
        return self._op2(other, lambda x, y: y % x, lambda x, y: choose_backend(x, y).mod(y, x))

    def __eq__(self, other):
        return self._op2(other, lambda x, y: x == y, lambda x, y: choose_backend(x, y).equal(x, y))

    def __ne__(self, other):
        return self._op2(other, lambda x, y: x != y, lambda x, y: choose_backend(x, y).not_equal(x, y))

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y, lambda x, y: choose_backend(x, y).greater_than(y, x))

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y, lambda x, y: choose_backend(x, y).greater_or_equal(y, x))

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y, lambda x, y: choose_backend(x, y).greater_than(x, y))

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y, lambda x, y: choose_backend(x, y).greater_or_equal(x, y))

    def __abs__(self):
        return self._op1(lambda t: choose_backend(t).abs(t))

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
        assert self.rank == 1, f"Can only iterate over 1D tensors but got {self.shape}"
        return iter(self.native())

    def _tensor(self, other):
        if isinstance(other, Tensor):
            return other
        elif isinstance(other, Shape):
            assert self.shape.channel.rank == 1, "Only single-channel tensors support implicit casting from Shape to tensor"
            assert other.rank == self.shape.channel.volume
            return tensor(other.spatial.sizes, names=self.shape.channel.names)
        else:
            backend = choose_backend(other)
            try:
                other_tensor = backend.as_tensor(other, convert_external=True)
                shape = backend.staticshape(other_tensor)
            except ValueError as e:
                raise ValueError(e)
            if len(shape) == 0:
                return NativeTensor(other_tensor, EMPTY_SHAPE)
            elif len(shape) == self.rank:
                return NativeTensor(other_tensor, self.shape.with_sizes(shape))
            elif len(shape) == self.shape.channel.rank:
                other_tensor = tensor(other, names=self.shape.channel.names)
                return other_tensor
            elif len(shape) == 1 and self.shape.channel.rank == 0:
                return NativeTensor(other_tensor, Shape(shape, ['vector'], [CHANNEL_DIM]))
            else:
                raise ValueError("Cannot broadcast object of rank %d to tensor with shape %s" % (backend.ndims(other), self.shape))

    def _op1(self, native_function):
        """
        Transform the values of this tensor given a function that can be applied to any native tensor.

        Args:
          native_function:

        Returns:

        """
        raise NotImplementedError(self.__class__)

    def _op2(self, other: 'Tensor', operator: callable, native_function: callable) -> 'Tensor':
        """
        Apply a broadcast operation on two tensors.

        Args:
          other: second argument
          operator: function (Tensor, Tensor) -> Tensor, used to propagate the operation to children tensors to have Python choose the callee
          native_function: function (native tensor, native tensor) -> native tensor
          other: 'Tensor': 
          operator: callable: 
          native_function: callable: 

        Returns:

        """
        raise NotImplementedError()

    def _natives(self) -> tuple:
        raise NotImplementedError(self.__class__)

    def _expand(self):
        """ Expands all compressed tensors to their defined size as if they were being used in `Tensor.native()`. """
        raise NotImplementedError(self.__class__)

    def __tensor_reduce__(self,
                dims: Tuple[str],
                native_function: callable,
                collapsed_function: callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                unaffected_function: callable = lambda value: value):
        raise NotImplementedError(self.__class__)

    def __simplify__(self):
        return self


class TensorDim:
    """
    Reference to a specific dimension of a `Tensor`.

    To obtain a `TensorDim`, use `Tensor.dimension()` or the syntax `tensor.<dim>`.

    Indexing a `TensorDim` as `tdim[start:stop:step]` returns a sliced `Tensor`.

    See the documentation at https://tum-pbs.github.io/PhiFlow/Math.html#indexing-slicing-unstacking .
    """

    def __init__(self, tensor: Tensor, name: str):
        self.tensor = tensor
        self.name = name

    @property
    def exists(self):
        """ Whether the dimension is listed in the `Shape` of the `Tensor`. """
        return self.name in self.tensor.shape

    def __str__(self):
        """ Dimension name. """
        return self.name

    def unstack(self, size: int or None = None, to_numpy=False, to_python=False) -> tuple:
        """
        See `unstack_spatial()`.

        Args:
            size: (optional)
                None: unstack along this dimension, error if dimension does not exist
                int: repeating unstack if dimension does not exist
            to_numpy: Whether to convert the selected data to `numpy.ndarray` objects.
            to_python: Whether to convert the selected data to Python types, i.e. `int, float, complex, bool, tuple, list`.

        Returns:
            sliced tensors
        """
        if size is None:
            result = self.tensor.unstack(self.name)
        else:
            if self.exists:
                unstacked = self.tensor.unstack(self.name)
                assert len(unstacked) == size, f"Size of dimension {self.name} does not match {size}."
                result = unstacked
            else:
                result = (self.tensor,) * size
        if to_numpy or to_python:
            result = tuple(component.numpy() for component in result)
            if to_python:
                result = tuple(component.tolist() for component in result)
        return result

    def optional_unstack(self, to_numpy=False, to_python=False):
        """
        Unstacks the `Tensor` along this dimension if the dimension is listed in the `Shape`.
        Otherwise returns the original `Tensor`.

        Args:
            to_numpy: Whether to convert the selected data to `numpy.ndarray` objects.
            to_python: Whether to convert the selected data to Python types, i.e. `int, float, complex, bool, tuple, list`.

        Returns:
            `tuple` of sliced tensors or original `Tensor`
        """
        if self.exists:
            return self.unstack(to_numpy=to_numpy, to_python=to_python)
        else:
            if to_numpy or to_python:
                result = self.tensor.numpy()
                if to_python:
                    return result.tolist()
                return result
            return self.tensor

    def unstack_spatial(self, components: str or tuple or list, to_numpy=False, to_python=False) -> tuple:
        """
        Slices the tensor along this dimension, returning only the selected components in the specified order.

        Args:
            components:
            to_numpy: Whether to convert the selected data to `numpy.ndarray` objects.
            to_python: Whether to convert the selected data to Python types, i.e. `int, float, complex, bool, tuple, list`.

        Returns:
            selected components
        """
        if isinstance(components, str):
            components = _shape.parse_dim_order(components)
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
        if to_numpy or to_python:
            result = tuple(component.numpy() for component in result)
            if to_python:
                result = tuple(component.tolist() for component in result)
        return tuple(result)

    @property
    def index(self):
        """ The index of this dimension in the `Shape` of the `Tensor`. """
        return self.tensor.shape.index(self.name)

    def __int__(self):
        return self.index

    def __len__(self):
        return self.tensor.shape.get_size(self.name)

    @property
    def size(self):
        """ Length of this tensor dimension as listed in the `Shape`, otherwise `1`. """
        if self.exists:
            return self.tensor.shape.get_size(self.name)
        else:
            return 1

    def as_batch(self, name: str or None = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *batch*. """
        return self._as(BATCH_DIM, name)

    def as_spatial(self, name: str or None = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *spatial*. """
        return self._as(SPATIAL_DIM, name)

    def as_channel(self, name: str or None = None):
        """ Returns a shallow copy of the `Tensor` where the type of this dimension is *channel*. """
        return self._as(CHANNEL_DIM, name)

    def _as(self, dim_type: int, name: str or None):
        shape = self.tensor.shape
        new_types = list(shape.types)
        new_types[self.index] = dim_type
        new_names = shape.names
        if name is not None:
            new_names = list(new_names)
            new_names[self.index] = name
        new_shape = Shape(shape.sizes, new_names, new_types)
        return self.tensor._with_shape_replaced(new_shape)

    @property
    def _dim_type(self):
        return self.tensor.shape.get_type(self.name)

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

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.tensor.shape.spatial.index(item)
        return self.tensor[{self.name: item}]

    def flip(self):
        """ Flips the element order along this dimension and returns the result as a `Tensor`. """
        return self.tensor.flip(self.name)

    def split(self, split_dimensions: Shape):
        """ See `phi.math.split_dimension()` """
        from ._functions import split_dimension
        return split_dimension(self, split_dimensions)


class NativeTensor(Tensor):

    def __init__(self, native_tensor, shape):
        assert isinstance(shape, Shape), f"Expected Shape but got '{type(shape)}'"
        backend = choose_backend(native_tensor)
        assert backend.staticshape(native_tensor) == shape.sizes, f"Shape {shape} does not match native tensor with shape {backend.staticshape(native_tensor)}"
        self._native = native_tensor
        self._shape = shape

    def native(self, order: str or tuple or list = None):
        order = _shape.parse_dim_order(order)
        if order is None or tuple(order) == self.shape.names:
            return self._native
        # --- Insert missing dims ---
        native = self._native
        backend = choose_backend(native)
        shape = self.shape
        for name in order:
            if name not in self.shape:
                native = backend.expand_dims(native, axis=-1)
                shape = shape.expand(1, name, CHANNEL_DIM, pos=-1)
        # --- Transpose ---
        perm = shape.perm(order)
        native = backend.transpose(native, perm)
        return native

    @property
    def dtype(self):
        return choose_backend(self._native).dtype(self._native)

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        new_shape = Shape(self._shape.sizes, new_shape.names, new_shape.types)
        return NativeTensor(self._native, new_shape)

    @property
    def _is_special(self) -> bool:
        return False

    def _getitem(self, selection: dict):
        new_shape = self.shape
        selections = [slice(None)] * self.rank
        for name, selection in selection.items():
            if name in self.shape:
                selections[self.shape.index(name)] = selection
                if isinstance(selection, int):
                    new_shape = new_shape.without(name)
            else:
                assert isinstance(selection, int), f"Attempting slice missing dimension {name} with {selection}"
        gathered = self._native[tuple(selections)]
        new_shape = new_shape.with_sizes(choose_backend(gathered).staticshape(gathered))
        return NativeTensor(gathered, new_shape)

    def flip(self, *dims: str) -> 'Tensor':
        dims = [dim for dim in dims if dim in self._shape]
        native = choose_backend(self._native).flip(self._native, self._shape.index(dims))
        return NativeTensor(native, self._shape)

    def unstack(self, dimension):
        dim_index = self.shape.index(dimension)
        new_shape = self.shape.without(dimension)
        tensors = choose_backend(self._native).unstack(self._native, axis=dim_index)
        return tuple([NativeTensor(t, new_shape) for t in tensors])

    def _op1(self, native_function):
        native = native_function(self.native())
        return NativeTensor(native, self.shape) if native is not None else self

    def _op2(self, other, operator, native_function):
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

    def __tensor_reduce__(self,
                dims: Tuple[str],
                native_function: callable,
                collapsed_function: callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                unaffected_function: callable = lambda value: value):
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        backend = choose_backend(self._native)
        result = native_function(backend, self._native, dim=self._shape.index(dims))
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
        for size, name, dim_type in tensor.shape.dimensions:
            assert shape.get_size(name) == size
            assert shape.get_type(name) == dim_type
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
            if self._inner._is_special:
                return None
            native = self._inner.native(order=self.shape.names)
            multiples = [1 if name in self._inner.shape else size for size, name, _ in self.shape.dimensions]
            tiled = choose_backend(native).tile(native, multiples)
            self._cached = NativeTensor(tiled, self.shape)
            self._inner = None
        return self._cached

    @property
    def is_cached(self):
        return self._cached is not None

    def __simplify__(self):
        if self.is_cached:
            return self._cached
        else:
            return self

    def native(self, order: str or tuple or list = None):
        if self.is_cached:
            return self._cached.native(order)
        order = _shape.parse_dim_order(order)
        if order is None or tuple(order) == self.shape.names:
            return self._cache().native()
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

    def _with_shape_replaced(self, new_shape):
        result = CollapsedTensor(self._inner, new_shape)
        result._cached = self._cached
        return result

    @property
    def _is_special(self) -> bool:
        if self.is_cached:
            return self._cached._is_special
        else:
            return self._inner._is_special

    def _getitem(self, selection: dict):
        if self.is_cached:
            return self._cached._getitem(selection)
        else:
            inner_dict = {name: selection for name, selection in selection.items() if name in self._inner.shape}
            inner = self._inner._getitem(inner_dict)
            new_shape = self.shape.after_gather(selection)
            inner.shape.combined(new_shape)  # check that sizes match
            return CollapsedTensor(inner, new_shape)

    def flip(self, *dims: str) -> 'Tensor':
        if self.is_cached:
            return self._cached.flip(*dims)
        else:
            return CollapsedTensor(self._inner.flip(*dims), self._shape)

    def _op1(self, native_function):
        if self.is_cached:
            return self._cached._op1(native_function)
        else:
            return CollapsedTensor(self._inner._op1(native_function), self._shape)

    def _op2(self, other, operator, native_function):
        other_t = self._tensor(other)
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
                result = inner._with_shape_replaced(inner.shape.with_types(self._shape & other_t._shape))
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

    def _expand(self):
        return self._cache()

    def __tensor_reduce__(self,
                dims: Tuple[str],
                native_function: callable,
                collapsed_function: callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                unaffected_function: callable = lambda value: value):
        if self.is_cached:
            return self._cached.__tensor_reduce__(dims, native_function, collapsed_function, unaffected_function)
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        inner_reduce = self._inner.__tensor_reduce__(dims, native_function, collapsed_function, unaffected_function)
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

    def __init__(self, components, dim_name, dim_type):
        for t in components:
            assert isinstance(t, Tensor)
            assert t.dtype == components[0].dtype, f"Stacked tensors must have the same data type but got {[t.dtype for t in components]}"
            assert dim_name not in t.shape, f"Cannot stack along '{dim_name}' because the dimension already exists."
        self.tensors = tuple(components)
        self.stack_dim_name = dim_name
        self.stack_dim_type = dim_type
        self._varying_shapes = any([v.shape != components[0].shape for v in components[1:]])
        self._shape = _shape.shape_stack(dim_name, dim_type, *[t.shape for t in self.tensors])
        self._cached = None

    @property
    def _is_special(self) -> bool:
        return any([t._is_special for t in self.tensors])

    @property
    def requires_broadcast(self):
        return self._varying_shapes or not self._shape.well_defined or self._is_special

    def _cache(self):
        if self._cached is None:
            natives = [t.native(order=self._shape.names) for t in self.tensors]
            native = choose_backend(*natives).concat(natives, axis=self.shape.index(self.stack_dim_name))
            self._cached = NativeTensor(native, self._shape)
        return self._cached

    @property
    def dtype(self):
        return self.tensors[0].dtype

    @property
    def shape(self):
        return self._shape

    def native(self, order: str or tuple or list = None):
        order = _shape.parse_dim_order(order)
        if self._cached is not None:
            return self._cached.native(order=order)
        # Is only the stack dimension shifted?
        if order is not None and self._shape.without(self.stack_dim_name).names == tuple(filter(lambda name: name != self.stack_dim_name, order)):
            natives = [t.native() for t in self.tensors]
            native = choose_backend(*natives).stack(natives, axis=tuple(order).index(self.stack_dim_name))
            return native
        assert not self.shape.is_non_uniform, f"Cannot convert non-uniform tensor with shape {self.shape} to native tensor."
        return self._cache().native(order=order)

    def _with_shape_replaced(self, new_shape: Shape):
        stack_dim_name = new_shape.names[self._shape.index(self.stack_dim_name)]
        inner_shape = new_shape.without(stack_dim_name)
        tensors = [t._with_shape_replaced(inner_shape) for t in self.tensors]
        return TensorStack(tensors, stack_dim_name, new_shape.get_type(stack_dim_name))

    def _getitem(self, selection: dict):
        if (self.stack_dim_name not in selection or len(selection) != 1) and not self.requires_broadcast:
            return self._cache()._getitem(selection)
        # --- Inner dims ---
        inner_dict = {dim: sel for dim, sel in selection.items() if dim != self.stack_dim_name}
        tensors = self.tensors
        if len(inner_dict) > 0:
            tensors = [t[inner_dict] for t in tensors]
        # --- stack dimension ---
        if self.stack_dim_name in selection:
            selection = selection[self.stack_dim_name]
            if isinstance(selection, int):
                return self.tensors[selection]
            elif isinstance(selection, slice):
                return TensorStack(tensors[selection], self.stack_dim_name, self.shape.get_type(self.stack_dim_name))
            else:
                raise NotImplementedError(f"{type(selection)} not supported. Only (int, slice) allwoed")
        else:
            return TensorStack(tensors, self.stack_dim_name, self.shape.get_type(self.stack_dim_name))

    def flip(self, *dims: str) -> 'Tensor':
        tensors = [t.flip(*dims) for t in self.tensors]
        if self.stack_dim_name in dims:
            tensors = tensors[::-1]
        return TensorStack(tensors, self.stack_dim_name, self.stack_dim_type)

    def unstack(self, dimension):
        if dimension == self.stack_dim_name:
            return self.tensors
        else:
            if self.requires_broadcast:
                unstacked = [t.unstack(dimension) for t in self.tensors]
                result = [TensorStack(items, self.stack_dim_name, self.stack_dim_type) for items in zip(*unstacked)]
                return result
            else:
                return self._cache().unstack(dimension=dimension)

    def _op1(self, native_function):
        if self.requires_broadcast:
            tensors = [t._op1(native_function) for t in self.tensors]
            return TensorStack(tensors, self.stack_dim_name, self.stack_dim_type)
        else:
            return self._cache()._op1(native_function)

    def _op2(self, other, operator, native_function):
        other = self._tensor(other)
        if self.requires_broadcast:
            if self.stack_dim_name in other.shape:
                other = other.unstack(self.stack_dim_name)
                tensors = [operator(t1, t2) for t1, t2 in zip(self.tensors, other)]
            else:
                tensors = [operator(t, other) for t in self.tensors]
            return TensorStack(tensors, self.stack_dim_name, self.stack_dim_type)
        elif isinstance(other, (CollapsedTensor, NativeTensor)):
            return op2_native(self, other, native_function)
        elif isinstance(other, TensorStack) and not other.requires_broadcast:
            return op2_native(self, other, native_function)
        else:
            return NotImplemented

    def _natives(self) -> tuple:
        return sum([t._natives() for t in self.tensors], ())

    def _expand(self):
        for t in self.tensors:
            t._expand()

    def __tensor_reduce__(self,
                dims: Tuple[str],
                native_function: callable,
                collapsed_function: callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                unaffected_function: callable = lambda value: value):
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        # --- inner reduce ---
        inner_axes = [dim for dim in dims if dim != self.stack_dim_name]
        red_inners = [t.__tensor_reduce__(inner_axes, native_function, collapsed_function, unaffected_function) for t in self.tensors]
        # --- outer reduce ---
        if self.stack_dim_name in dims:
            if any([t._is_special for t in red_inners]):
                return sum(red_inners[1:], red_inners[0])
            else:
                natives = [t.native() for t in red_inners]
                result = native_function(choose_backend(*natives), natives, dim=0)  # TODO not necessary if tensors are CollapsedTensors
                return NativeTensor(result, red_inners[0].shape)
        else:
            return TensorStack(red_inners, self.stack_dim_name, self.stack_dim_type)



def tensors(*objects: Tensor or Shape or tuple or list or numbers.Number,
            names: str or tuple or list = None,
            convert: bool = False):
    """
    Calls `tensor()` on multiple arguments independently.

    Example:

        scalar_tensor, vector_tensor = tensors(0, (1, 2, 3))

    Returns:
        Sequence of same length as `objects`.
    """
    return [tensor(obj, names, convert) for obj in objects]


def tensor(data: Tensor or Shape or tuple or list or numbers.Number,
           names: str or tuple or list = None,
           convert: bool = False) -> Tensor:
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

    Args:
      data: native tensor, scalar, sequence, Shape or Tensor
      names: Dimension names. Dimension types are inferred from the names.
      convert: If True, converts the data to the native format of the current default backend.
        If False, wraps the data in a `Tensor` but keeps the given data reference if possible.

    Raises:
      AssertionError if dimension names are not provided and cannot automatically be inferred

    Returns:
      Tensor containing same values as data
    """
    if isinstance(data, Tensor):
        if convert:
            backend = choose_backend(*data._natives())
            if backend != default_backend():
                data = data._op1(lambda native: default_backend().as_tensor(backend.numpy(native), convert_external=True))
        if names is None:
            return data
        else:
            names = _shape.parse_dim_names(names, data.rank)
            names = [n if n is not None else o for n, o in zip(names, data.shape.names)]
            types = [_shape._infer_dim_type_from_name(n) if n is not None else o for n, o in zip(names, data.shape.types)]
            new_shape = Shape(data.shape.sizes, names, types)
            return data._with_shape_replaced(new_shape)
    elif isinstance(data, (tuple, list)):
        array = np.array(data)
        if array.dtype != np.object:
            data = array
        else:
            elements = tensors(*data, names=None if names is None else names[1:], convert=convert)
            common_shape = _shape.combine_safe(*[e.shape for e in elements])
            rank = 1 + common_shape.rank
            stack_dim = 'vector' if names is None else _shape.parse_dim_names(names, rank)[0]
            assert all(stack_dim not in t.shape for t in elements), f"Cannot stack tensors with dimension '{stack_dim}' because a tensor already has that dimension."
            elements = [CollapsedTensor(e, common_shape) if e.shape.rank < common_shape.rank else e for e in elements]
            from ._functions import cast_same
            elements = cast_same(*elements)
            return TensorStack(elements, dim_name=stack_dim, dim_type=_shape._infer_dim_type_from_name(stack_dim))
    elif isinstance(data, (numbers.Number, bool, str)):
        assert not names, f"Trying to create a zero-dimensional Tensor from value '{data}' but names={names}"
        if convert:
            data = default_backend().as_tensor(data, convert_external=True)
        return NativeTensor(data, EMPTY_SHAPE)
    elif isinstance(data, Shape):
        assert names is not None
        return tensor(data.sizes, names, convert=convert)
    backend = choose_backend(data, raise_error=False)
    if backend:
        if names is None:
            assert data.ndim <= 1, "Specify dimension names for tensors with more than 1 dimension"
            names = ['vector'] * backend.ndims(data)  # [] or ['vector']
            types = [CHANNEL_DIM] * backend.ndims(data)
        else:
            names = _shape.parse_dim_names(names, len(data.shape))
            assert None not in names, f"All names must be specified but got {names}"
            types = [_shape._infer_dim_type_from_name(n) for n in names]
        shape = Shape(data.shape, names, types)
        if convert and backend != default_backend():
            data = backend.numpy(data)
            data = default_backend().as_tensor(data, convert_external=True)
        return NativeTensor(data, shape)
    raise ValueError(f"{type(data)} is not supported. Only (Tensor, tuple, list, np.ndarray, native tensors) are allowed.\nCurrent backends: {BACKENDS}")


def broadcastable_native_tensors(*tensors):
    """
    Expands and transposes the dimensions of the given tensors so that they all have the same dimension order.

    Args:
      tensors: sequence of Tensors
      *tensors: 

    Returns:
      shape, native tensors)

    """
    broadcast_shape = _shape.combine_safe(*[t.shape for t in tensors])
    natives = [t.native(order=broadcast_shape.names) for t in tensors]
    return broadcast_shape, natives


def op2_native(x: Tensor, y: Tensor, native_function: callable):
    new_shape, (native1, native2) = broadcastable_native_tensors(x, y)
    result_tensor = native_function(native1, native2)
    return NativeTensor(result_tensor, new_shape)


def custom_op2(x: Tensor or float, y: Tensor or float, l_operator, l_native_function, r_operator=None, r_native_function=None):
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

    Returns:

    """
    x, y = tensors(x, y)
    result = x._op2(y, l_operator, l_native_function)
    if result is NotImplemented:
        result = y._op2(x, r_operator or l_operator, r_native_function or l_native_function)
        if result is NotImplemented:
            raise NotImplementedError(f"Operation not supported between {type(x)} and {type(y)}")
    return result

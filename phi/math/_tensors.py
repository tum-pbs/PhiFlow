import numbers
import warnings

import numpy as np

from . import _shape
from .backend import math as native_math
from ._shape import Shape, CHANNEL_DIM, BATCH_DIM, SPATIAL_DIM, EMPTY_SHAPE


class Tensor:
    """
    Tensors with grouped and named dimensions.

    All tensors are editable.

    The internal data representation of a tensor can change, even without being edited.
    """

    def native(self, order: str or tuple or list = None):
        """
        Returns a native tensor object with the dimensions ordered according to `order`.

        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.

        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        :param order: (optional) list of dimension names. If not given, the current order is kept.
        :return: native tensor object
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

        :param order: (optional) list of dimension names. If not given, the current order is kept.
        :return: NumPy representation
        :raise: ValueError if the tensor cannot be transposed to match target_shape
        """
        native = self.native(order=order)
        return native_math.numpy(native)

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        raise NotImplementedError()

    def _with_shape_replaced(self, new_shape):
        raise NotImplementedError()

    @property
    def ndims(self) -> int:
        return self.shape.rank

    @property
    def rank(self) -> int:
        return self.shape.rank

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

    def __repr__(self):
        try:
            content = self.numpy()
            dtype = content.dtype
        except ValueError as e:
            try:
                dtype = self.dtype
            except AttributeError:
                return repr(self.shape)
            return "%s %s" % (self.shape, dtype)
        if self.rank == 0:
            return str(content)
        if self.shape.volume is not None and self.shape.volume <= 4:
            content = list(np.reshape(content, [-1]))
            content = ', '.join([repr(number) for number in content])
            return "%s %s  %s" % (self.shape, dtype, content)
        else:
            min_, max_ = np.min(content), np.max(content)
            return "%s %s  %s < ... < %s" % (self.shape, content.dtype, min_, max_)

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

        :param selection: dim_name: str -> int or slice
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

    def unstack(self, dimension):
        """
        Splits this tensor along the specified dimension.
        The returned tensors have the same dimensions as this tensor save the unstacked dimension.

        :param dimension: name of dimension or Dimension or None for component dimension
        :type dimension: str or int or _TensorDim
        :return: tuple of tensors
        """
        raise NotImplementedError()

    def dimension(self, name):
        return _TensorDim(self, name)

    def __getattr__(self, name):
        assert name not in ('shape', '_shape')
        return _TensorDim(self, name)

    def __add__(self, other):
        return self._op2(other, lambda x, y: x + y, lambda x, y: native_math.add(x, y))

    def __radd__(self, other):
        return self._op2(other, lambda x, y: y + x, lambda x, y: native_math.add(y, x))

    def __sub__(self, other):
        return self._op2(other, lambda x, y: x - y, lambda x, y: native_math.sub(x, y))

    def __rsub__(self, other):
        return self._op2(other, lambda x, y: y - x, lambda x, y: native_math.sub(y, x))

    def __and__(self, other):
        return self._op2(other, lambda x, y: x & y, lambda x, y: x & y)

    def __or__(self, other):
        return self._op2(other, lambda x, y: x | y, lambda x, y: x | y)

    def __xor__(self, other):
        return self._op2(other, lambda x, y: x ^ y, lambda x, y: x ^ y)

    def __mul__(self, other):
        return self._op2(other, lambda x, y: x * y, lambda x, y: native_math.mul(x, y))

    def __rmul__(self, other):
        return self._op2(other, lambda x, y: y * x, lambda x, y: native_math.mul(y, x))

    def __truediv__(self, other):
        return self._op2(other, lambda x, y: x / y, lambda x, y: native_math.div(x, y))

    def __rtruediv__(self, other):
        return self._op2(other, lambda x, y: y / x, lambda x, y: native_math.div(y, x))

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
        return self._op2(power, lambda x, y: x ** y, lambda x, y: native_math.pow(x, y))

    def __rpow__(self, other):
        return self._op2(other, lambda x, y: y ** x, lambda x, y: native_math.pow(y, x))

    def __mod__(self, other):
        return self._op2(other, lambda x, y: x % y, lambda x, y: native_math.mod(x, y))

    def __rmod__(self, other):
        return self._op2(other, lambda x, y: y % x, lambda x, y: native_math.mod(y, x))

    def __eq__(self, other):
        return self._op2(other, lambda x, y: x == y, lambda x, y: native_math.equal(x, y))

    def __ne__(self, other):
        return self._op2(other, lambda x, y: x != y, lambda x, y: x != y)

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y, lambda x, y: x < y)

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y, lambda x, y: x <= y)

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y, lambda x, y: x > y)

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y, lambda x, y: x >= y)

    def __abs__(self):
        return self._op1(lambda t: native_math.abs(t))

    def as_complex(self):
        return self._op1(lambda t: native_math.to_complex(t))

    def as_float(self):
        return self._op1(lambda t: native_math.to_float(t))

    def as_int(self, int64=False):
        return self._op1(lambda t: native_math.to_int(t, int64=int64))

    def __copy__(self):
        return self._op1(lambda t: native_math.copy(t, only_mutable=True))

    def __deepcopy__(self, memodict={}):
        return self._op1(lambda t: native_math.copy(t, only_mutable=False))

    def __neg__(self):
        return self._op1(lambda t: -t)

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
            try:
                other_tensor = native_math.as_tensor(other, convert_external=True)
                shape = native_math.staticshape(other_tensor)
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
                raise ValueError("Cannot broadcast object of rank %d to tensor with shape %s" % (native_math.ndims(other), self.shape))

    def _op2(self, other: 'Tensor', operator: callable, native_function: callable) -> 'Tensor':
        """
        Apply a broadcast operation on two tensors.

        :param other: second argument
        :param operator: function (Tensor, Tensor) -> Tensor, used to propagate the operation to children tensors to have Python choose the callee
        :param native_function: function (native tensor, native tensor) -> native tensor
        """
        raise NotImplementedError()

    def _op1(self, native_function):
        """ Transform the values of this tensor given a function that can be applied to any native tensor. """
        raise NotImplementedError(self.__class__)


class _TensorDim:

    def __init__(self, tensor, name):
        self.tensor = tensor
        self.name = name

    @property
    def exists(self):
        return self.name in self.tensor.shape.names

    def __str__(self):
        return self.name

    def unstack(self, size: int or None = None):
        if size is None:
            return self.tensor.unstack(self.name)
        else:
            if self.exists:
                unstacked = self.tensor.unstack(self.name)
                assert len(unstacked) == size, f"Size of dimension {self.name} does not match {size}."
                return unstacked
            else:
                return (self.tensor,) * size

    @property
    def index(self):
        return self.tensor.shape.index(self.name)

    def __int__(self):
        return self.index

    @property
    def size(self):
        return self.tensor.shape.sizes[self.index]

    def as_batch(self, name: str or None = None):
        return self._as(BATCH_DIM, name)

    def as_spatial(self, name: str or None = None):
        return self._as(SPATIAL_DIM, name)

    def as_channel(self, name: str or None = None):
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
    def dim_type(self):
        return self.tensor.shape.types[self.index]

    @property
    def is_spatial(self):
        return self.tensor.shape.types[self.index] == SPATIAL_DIM

    @property
    def is_batch(self):
        return self.tensor.shape.types[self.index] == BATCH_DIM

    @property
    def is_channel(self):
        return self.tensor.shape.types[self.index] == CHANNEL_DIM

    def __getitem__(self, item):
        return self.tensor[{self.name: item}]

    def __setitem__(self, key, value):
        self.tensor[{self.name: key}] = value


class NativeTensor(Tensor):

    def __init__(self, native_tensor, shape):
        assert not isinstance(native_tensor, Tensor)
        assert isinstance(shape, Shape)
        assert native_math.staticshape(native_tensor) == shape.sizes, f"Shape {shape} does not match native tensor with shape {native_math.staticshape(native_tensor)}"
        self.tensor = native_tensor
        self._shape = shape

    def native(self, order: str or tuple or list = None):
        order = _shape.parse_dim_order(order)
        if order is None or tuple(order) == self.shape.names:
            return self.tensor
        # --- Insert missing dims ---
        tensor = self.tensor
        shape = self.shape
        for name in order:
            if name not in self.shape:
                tensor = native_math.expand_dims(tensor, axis=-1)
                shape = shape.expand(1, name, CHANNEL_DIM, pos=-1)
        # --- Transpose ---
        perm = shape.perm(order)
        tensor = native_math.transpose(tensor, perm)
        return tensor

    @property
    def dtype(self):
        return native_math.dtype(self.tensor)

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        new_shape = Shape(self._shape.sizes, new_shape.names, new_shape.types)
        return NativeTensor(self.tensor, new_shape)

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
        gathered = self.tensor[tuple(selections)]
        new_shape = new_shape.with_sizes(native_math.staticshape(gathered))
        return NativeTensor(gathered, new_shape)

    def unstack(self, dimension):
        dim_index = self.shape.index(dimension)
        new_shape = self.shape.without(dimension)
        tensors = native_math.unstack(self.tensor, axis=dim_index)
        return tuple([NativeTensor(t, new_shape) for t in tensors])

    def _op1(self, native_function):
        return NativeTensor(native_function(self.native()), self.shape)

    def _op2(self, other, operator, native_function):
        other = self._tensor(other)
        if isinstance(other, NativeTensor):
            return op2_native(self, other, native_function)
        else:
            return NotImplemented


class CollapsedTensor(Tensor):
    """
    Tiled / Repeated tensor along additional dimensions.
    """

    def __init__(self, tensor: Tensor, shape: Shape):
        for name in tensor.shape.names:
            assert name in shape
        for size, name, dim_type in tensor.shape.dimensions:
            assert shape.get_size(name) == size
            assert shape.get_type(name) == dim_type
        self.tensor = tensor
        self._shape = shape
        self._cached = None

    def _cache(self):
        if self._cached is None:
            native = self.tensor.native(order=self.shape.names)
            multiples = [1 if name in self.tensor.shape else size for size, name, _ in self.shape.dimensions]
            tiled = native_math.tile(native, multiples)
            self._cached = NativeTensor(tiled, self.shape)
        return self._cached

    def expand(self):
        return self._cache()

    def native(self, order: str or tuple or list = None):
        order = _shape.parse_dim_order(order)
        if order is None or tuple(order) == self.shape.names:
            return self._cache().native()
        else:
            native = self.tensor.native(order=order)
            multiples = [1 if name in self.tensor.shape else (self.shape.get_size(name) if name in self.shape else 1) for name in order]
            tiled = native_math.tile(native, multiples)
            return tiled

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def shape(self):
        return self._shape

    def unstack(self, dimension):
        unstacked_shape = self.shape.without(dimension)
        if dimension in self.tensor.shape:
            unstacked = self.tensor.unstack(dimension)
            return tuple(CollapsedTensor(t, unstacked_shape) for t in unstacked)
        else:
            return (CollapsedTensor(self.tensor, unstacked_shape),) * self.shape.get_size(dimension)

    def _with_shape_replaced(self, new_shape):
        return CollapsedTensor(self.tensor, new_shape)

    def _getitem(self, selection: dict):
        inner_dict = {name: selection for name, selection in selection.items() if name in self.tensor.shape}
        inner = self.tensor._getitem(inner_dict)
        new_shape = self.shape.after_gather(selection)
        inner.shape.combined(new_shape)  # check that sizes match
        return CollapsedTensor(inner, new_shape)

    def _op1(self, native_function):
        return CollapsedTensor(self.tensor._op1(native_function), self._shape)

    def _op2(self, other, operator, native_function):
        other = self._tensor(other)
        if isinstance(other, CollapsedTensor):
            self_inner = self.tensor._with_shape_replaced(self.tensor.shape.to_batch())
            other_inner = other.tensor._with_shape_replaced(other.tensor.shape.to_batch())
            inner = operator(self_inner, other_inner)
            if inner.shape.rank >= self.rank and inner.shape.rank >= other.rank:
                result = inner._with_shape_replaced(inner.shape.with_types(self.shape.combined(other.shape)))
                return result
            else:
                return CollapsedTensor(inner, self.shape.combined(other.shape).with_sizes(inner.shape))
        elif isinstance(other, NativeTensor):
            return op2_native(self, other, native_function)
        else:
            return NotImplemented


class TensorStack(Tensor):
    """
    Implicit stack of multiple tensors.
    List of tensors, does not store stacked tensor in memory.
    """

    def __init__(self, tensors, dim_name, dim_type, keep_separate=False):
        for t in tensors:
            assert isinstance(t, Tensor)
            assert t.dtype == tensors[0].dtype
            assert dim_name not in t.shape, f"Cannot stack along '{dim_name}' because the dimension already exists."
            # assert tensor.shape == tensors[0].shape or keep_separate
        self.tensors = tuple(tensors)
        self.stack_dim_name = dim_name
        self.stack_dim_type = dim_type
        self.keep_separate = keep_separate
        self._shape = _shape.shape_stack(dim_name, dim_type, *[t.shape for t in self.tensors])
        self._cached = None

    def _cache(self):
        if self._cached is None:
            native = native_math.concat([t.native(order=self._shape.names) for t in self.tensors], axis=self.shape.index(self.stack_dim_name))
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
            native = native_math.stack([t.native() for t in self.tensors], axis=tuple(order).index(self.stack_dim_name))
            return native
        return self._cache().native(order=order)

    def _with_shape_replaced(self, new_shape: Shape):
        stack_dim_name = new_shape.names[self._shape.index(self.stack_dim_name)]
        inner_shape = new_shape.without(stack_dim_name)
        tensors = [t._with_shape_replaced(inner_shape) for t in self.tensors]
        return TensorStack(tensors, stack_dim_name, new_shape.get_type(stack_dim_name), keep_separate=self.keep_separate)

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
            return TensorStack(tensors, self.stack_dim_name, self.shape.get_type(self.stack_dim_name), keep_separate=self.keep_separate)

    def unstack(self, dimension):
        if dimension == self.stack_dim_name:
            return self.tensors
        else:
            if self.keep_separate:
                unstacked = [t.unstack(dimension) for t in self.tensors]
                result = [TensorStack(items, self.stack_dim_name, self.stack_dim_type, keep_separate=self.keep_separate) for items in zip(*unstacked)]
                return result
            else:
                return self._cache().unstack(dimension=dimension)

    def _op2(self, other, operator, native_function):
        other = self._tensor(other)
        if self.requires_broadcast:
            if self.stack_dim_name in other.shape:
                other = other.unstack(self.stack_dim_name)
                tensors = [operator(t1, t2) for t1, t2 in zip(self.tensors, other)]
            else:
                tensors = [operator(t, other) for t in self.tensors]
            return TensorStack(tensors, self.stack_dim_name, self.stack_dim_type, self.keep_separate)
        elif isinstance(other, (CollapsedTensor, NativeTensor)):
            return op2_native(self, other, native_function)
        elif isinstance(other, TensorStack) and not other.requires_broadcast:
            return op2_native(self, other, native_function)
        else:
            return NotImplemented

    def _op1(self, native_function):
        if self.requires_broadcast:
            tensors = [t._op1(native_function) for t in self.tensors]
            return TensorStack(tensors, self.stack_dim_name, self.stack_dim_type, self.keep_separate)
        else:
            return self._cache()._op1(native_function)

    @property
    def requires_broadcast(self):
        from phi.math._track import ShiftLinOp
        return self.keep_separate or not self._shape.well_defined or np.any([isinstance(t, ShiftLinOp) for t in self.tensors])


def tensors(*objects, names=None):
    return [tensor(obj, names) for obj in objects]


def tensor(data: Tensor or Shape or tuple or list or numbers.Number,
           names: str or tuple or list = None) -> Tensor:
    """
    Create a Tensor from the specified data.

    If dimension names are specified, dimension types are inferred from them.
    Otherwise, existing or default dimension names are used.

    :param data: native tensor, scalar, sequence, Shape or Tensor
    :param names: Dimension names. Dimension types are inferred from the names.
    :return: Tensor representing `obj`
    """
    if isinstance(data, Tensor):
        if names is None:
            return data
        else:
            names = _shape.parse_dim_names(names, data.rank)
            names = [n if n is not None else o for n, o in zip(names, data.shape.names)]
            types = [_shape._infer_dim_type_from_name(n) if n is not None else o for n, o in zip(names, data.shape.types)]
            new_shape = Shape(data.shape.sizes, names, types)
            return data._with_shape_replaced(new_shape)
    if isinstance(data, (tuple, list)):
        array = np.array(data)
        if array.dtype != np.object:
            data = array
        else:
            raise NotImplementedError(f"{array.dtype} dtype for iterable not allowed. Only np.object supported.")
            return TensorStack(tensor(data), dim_name=None, dim_type=CHANNEL_DIM)
    if isinstance(data, np.ndarray) and data.dtype != np.object:
        if names is None:
            assert data.ndim <= 1, "Specify dimension names for tensors with more than 1 dimension"
            names = ['vector'] * data.ndim
        else:
            names = _shape.parse_dim_names(names, len(data.shape))
        shape = Shape(data.shape, names, [CHANNEL_DIM] * len(data.shape))
        return NativeTensor(data, shape)
    if isinstance(data, numbers.Number):
        assert not names
        array = np.array(data)
        return NativeTensor(array, EMPTY_SHAPE)
    if isinstance(data, Shape):
        assert names is not None
        return tensor(data.sizes, names)
    raise ValueError(f"{type(data)} is not supported. Only (Tensor, tuple, list, np.ndarray, NativeTensor).")


def broadcastable_native_tensors(*tensors):
    """
    Expands and transposes the dimensions of the given tensors so that they all have the same dimension order.

    :param tensors: sequence of Tensors
    :return: (shape, native tensors)
    """
    broadcast_shape = _shape.combine_safe(*[t.shape for t in tensors])
    natives = [t.native(order=broadcast_shape.names) for t in tensors]
    return broadcast_shape, natives


def op2_native(x: Tensor, y: Tensor, native_function: callable):
    new_shape, (native1, native2) = broadcastable_native_tensors(x, y)
    result_tensor = native_function(native1, native2)
    return NativeTensor(result_tensor, new_shape)


def custom_op2(x: Tensor or float, y: Tensor or float, l_operator, l_native_function, r_operator=None, r_native_function=None):
    x, y = tensors(x, y)
    result = x._op2(y, l_operator, l_native_function)
    if result is NotImplemented:
        result = y._op2(x, r_operator or l_operator, r_native_function or l_native_function)
        if result is NotImplemented:
            raise NotImplementedError(f"Operation not supported between {type(x)} and {type(y)}")
    return result

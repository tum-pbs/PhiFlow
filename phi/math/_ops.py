import functools
import math
import warnings
from numbers import Number
from typing import Tuple, Callable, Any

import numpy as np

from . import extrapolation as e_
from ._magic_ops import expand, pack_dims, flatten, unpack_dim, cast, copy_with, value_attributes
from ._shape import (Shape, EMPTY_SHAPE,
                     spatial, batch, channel, instance, merge_shapes, parse_dim_order, concat_shapes,
                     IncompatibleShapes, DimFilter, non_batch)
from ._tensors import Tensor, wrap, tensor, broadcastable_native_tensors, NativeTensor, TensorStack, CollapsedTensor, \
    custom_op2, compatible_tensor, variable_attributes, disassemble_tree, assemble_tree, \
    cached, is_scalar, Layout
from .backend import default_backend, choose_backend, Backend, get_precision, convert as b_convert, BACKENDS, \
    NoBackendFound
from .backend._dtype import DType, combine_types
from .magic import PhiTreeNode


def choose_backend_t(*values, prefer_default=False) -> Backend:
    """
    Choose backend for given `Tensor` or native tensor values.
    Backends need to be registered to be available, e.g. via the global import `phi.<backend>` or `phi.detect_backends()`.

    Args:
        *values: Sequence of `Tensor`s, native tensors or constants.
        prefer_default: Whether to always select the default backend if it can work with `values`, see `default_backend()`.

    Returns:
        The selected `phi.math.backend.Backend`
    """
    natives = sum([v._natives() if isinstance(v, Tensor) else (v,) for v in values], ())
    return choose_backend(*natives, prefer_default=prefer_default)


def convert(x, backend: Backend = None, use_dlpack=True):
    """
    Convert the native representation of a `Tensor` or `PhiTreeNode` to the native format of `backend`.

    *Warning*: This operation breaks the automatic differentiation chain.

    See Also:
        `phi.math.backend.convert()`.

    Args:
        x: `Tensor` to convert. If `x` is a `PhiTreeNode`, its variable attributes are converted.
        backend: Target backend. If `None`, uses the current default backend, see `phi.math.backend.default_backend()`.

    Returns:
        `Tensor` with native representation belonging to `backend`.
    """
    if isinstance(x, Tensor):
        return x._op1(lambda native: b_convert(native, backend, use_dlpack=use_dlpack))
    elif isinstance(x, PhiTreeNode):
        return copy_with(x, **{a: convert(getattr(x, a), backend, use_dlpack=use_dlpack) for a in variable_attributes(x)})
    else:
        return choose_backend(x).as_tensor(x)


def all_available(*values: Tensor) -> bool:
    """
    Tests if the values of all given tensors are known and can be read at this point.
    Tracing placeholders are considered not available, even when they hold example values.

    Tensors are not available during `jit_compile()`, `jit_compile_linear()` or while using TensorFlow's legacy graph mode.
    
    Tensors are typically available when the backend operates in eager mode and is not currently tracing a function.

    This can be used instead of the native checks

    * PyTorch: `torch._C._get_tracing_state()`
    * TensorFlow: `tf.executing_eagerly()`
    * Jax: `isinstance(x, jax.core.Tracer)`

    Args:
      values: Tensors to check.

    Returns:
        `True` if no value is a placeholder or being traced, `False` otherwise.
    """
    from phi.math._functional import ShiftLinTracer
    for value in values:
        if isinstance(value, ShiftLinTracer):
            return False
        natives = value._natives()
        natives_available = [choose_backend(native).is_available(native) for native in natives]
        if not all(natives_available):
            return False
    return True


def seed(seed: int):
    """
    Sets the current seed of all backends and the built-in `random` package.

    Calling this function with a fixed value at the start of an application yields reproducible results
    as long as the same backend is used.

    Args:
        seed: Seed to use.
    """
    for backend in BACKENDS:
        backend.seed(seed)
    import random
    random.seed(0)


def native(value: Tensor or Number or tuple or list or Any):
    """
    Returns the native tensor representation of `value`.
    If `value` is a `phi.math.Tensor`, this is equal to calling `phi.math.Tensor.native()`.
    Otherwise, checks that `value` is a valid tensor object and returns it.

    Args:
        value: `Tensor` or native tensor or tensor-like.

    Returns:
        Native tensor representation

    Raises:
        ValueError if the tensor cannot be transposed to match target_shape
    """
    if isinstance(value, Tensor):
        return value.native()
    else:
        choose_backend(value)  # check that value is a native tensor
        return value


def numpy(value: Tensor or Number or tuple or list or Any):
    """
    Converts `value` to a `numpy.ndarray` where value must be a `Tensor`, backend tensor or tensor-like.
    If `value` is a `phi.math.Tensor`, this is equal to calling `phi.math.Tensor.numpy()`.

    *Note*: Using this function breaks the autograd chain. The returned tensor is not differentiable.
    To get a differentiable tensor, use `Tensor.native()` instead.

    Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
    If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

    If `value` is a NumPy array, it may be returned directly.

    Returns:
        NumPy representation of `value`

    Raises:
        ValueError if the tensor cannot be transposed to match target_shape
    """
    if isinstance(value, Tensor):
        return value.numpy()
    else:
        backend = choose_backend(value)
        return backend.numpy(value)


def reshaped_native(value: Tensor,
                    groups: tuple or list,
                    force_expand: Any = False,
                    to_numpy=False):
    """
    Returns a native representation of `value` where dimensions are laid out according to `groups`.

    See Also:
        `native()`, `pack_dims()`, `reshaped_tensor()`, `reshaped_numpy()`.

    Args:
        value: `Tensor`
        groups: Sequence of dimension names as `str` or groups of dimensions to be packed_dim as `Shape`.
        force_expand: `bool` or sequence of dimensions.
            If `True`, repeats the tensor along missing dimensions.
            If `False`, puts singleton dimensions where possible.
            If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.
        to_numpy: If True, converts the native tensor to a `numpy.ndarray`.

    Returns:
        Native tensor with dimensions matching `groups`.
    """
    assert isinstance(value, Tensor), f"value must be a Tensor but got {type(value)}"
    order = []
    for i, group in enumerate(groups):
        if isinstance(group, Shape):
            present = value.shape.only(group)
            if force_expand is True or present.volume > 1 or (force_expand is not False and group.only(force_expand).volume > 1):
                value = expand(value, group)
            value = pack_dims(value, group, batch(f"group{i}"))
            order.append(f"group{i}")
        else:
            assert isinstance(group, str), f"Groups must be either str or Shape but got {group}"
            order.append(group)
    return value.numpy(order) if to_numpy else value.native(order)


def reshaped_numpy(value: Tensor, groups: tuple or list, force_expand: Any = False):
    """
    Returns the NumPy representation of `value` where dimensions are laid out according to `groups`.

    See Also:
        `numpy()`, `reshaped_native()`, `pack_dims()`, `reshaped_tensor()`.

    Args:
        value: `Tensor`
        groups: Sequence of dimension names as `str` or groups of dimensions to be packed_dim as `Shape`.
        force_expand: `bool` or sequence of dimensions.
            If `True`, repeats the tensor along missing dimensions.
            If `False`, puts singleton dimensions where possible.
            If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.

    Returns:
        NumPy `ndarray` with dimensions matching `groups`.
    """
    return reshaped_native(value, groups, force_expand=force_expand, to_numpy=True)


def reshaped_tensor(value: Any,
                    groups: tuple or list,
                    check_sizes=False,
                    convert=True):
    """
    Creates a `Tensor` from a native tensor or tensor-like whereby the dimensions of `value` are split according to `groups`.

    See Also:
        `phi.math.tensor()`, `reshaped_native()`, `unpack_dim()`.

    Args:
        value: Native tensor or tensor-like.
        groups: Sequence of dimension groups to be packed_dim as `tuple[Shape]` or `list[Shape]`.
        check_sizes: If True, group sizes must match the sizes of `value` exactly. Otherwise, allows singleton dimensions.
        convert: If True, converts the data to the native format of the current default backend.
            If False, wraps the data in a `Tensor` but keeps the given data reference if possible.

    Returns:
        `Tensor` with all dimensions from `groups`
    """
    assert all(isinstance(g, Shape) for g in groups), "groups must be a sequence of Shapes"
    dims = [batch(f'group{i}') for i, group in enumerate(groups)]
    try:
        value = tensor(value, *dims, convert=convert)
    except IncompatibleShapes:
        raise IncompatibleShapes(f"Cannot reshape native tensor with sizes {value.shape} given groups {groups}")
    for i, group in enumerate(groups):
        if value.shape.get_size(f'group{i}') == group.volume:
            value = unpack_dim(value, f'group{i}', group)
        elif check_sizes:
            raise AssertionError(f"Group {group} does not match dimension {i} of value {value.shape}")
        else:
            value = unpack_dim(value, f'group{i}', group)
    return value


def copy(value: Tensor):
    """
    Copies the data buffer and encapsulating `Tensor` object.

    Args:
        value: `Tensor` to be copied.

    Returns:
        Copy of `value`.
    """
    if value._is_tracer:
        warnings.warn("Tracing tensors cannot be copied.", RuntimeWarning)
        return value
    return value._op1(lambda native: choose_backend(native).copy(native))


def native_call(f: Callable, *inputs: Tensor, channels_last=None, channel_dim='vector', spatial_dim=None):
    """
    Calls `f` with the native representations of the `inputs` tensors in standard layout and returns the result as a `Tensor`.

    All inputs are converted to native tensors (including precision cast) depending on `channels_last`:

    * `channels_last=True`: Dimension layout `(total_batch_size, spatial_dims..., total_channel_size)`
    * `channels_last=False`: Dimension layout `(total_batch_size, total_channel_size, spatial_dims...)`

    All batch dimensions are compressed into a single dimension with `total_batch_size = input.shape.batch.volume`.
    The same is done for all channel dimensions.

    Additionally, missing batch and spatial dimensions are added so that all `inputs` have the same batch and spatial shape.

    Args:
        f: Function to be called on native tensors of `inputs`.
            The function output must have the same dimension layout as the inputs, unless overridden by `spatial_dim`,
            and the batch size must be identical.
        *inputs: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
            If `None`, the channels are put in the default position associated with the current backend,
            see `phi.math.backend.Backend.prefers_channels_last()`.
        channel_dim: Name of the channel dimension of the result.
        spatial_dim: Name of the spatial dimension of the result.

    Returns:
        `Tensor` with batch and spatial dimensions of `inputs`, unless overridden by `spatial_dim`,
        and single channel dimension `channel_dim`.
    """
    if channels_last is None:
        try:
            backend = choose_backend(f)
        except NoBackendFound:
            backend = choose_backend_t(*inputs, prefer_default=True)
        channels_last = backend.prefers_channels_last()
    batch = merge_shapes(*[i.shape.batch for i in inputs])
    spatial = merge_shapes(*[i.shape.spatial for i in inputs])
    natives = []
    for i in inputs:
        groups = (batch, *i.shape.spatial.names, i.shape.channel) if channels_last else (batch, i.shape.channel, *i.shape.spatial.names)
        natives.append(reshaped_native(i, groups))
    output = f(*natives)
    if isinstance(channel_dim, str):
        channel_dim = channel(channel_dim)
    assert isinstance(channel_dim, Shape), "channel_dim must be a Shape or str"
    if isinstance(output, (tuple, list)):
        raise NotImplementedError()
    else:
        if spatial_dim is None:
            groups = (batch, *spatial, channel_dim) if channels_last else (batch, channel_dim, *spatial)
        else:
            if isinstance(spatial_dim, str):
                spatial_dim = spatial(spatial_dim)
            assert isinstance(spatial_dim, Shape), "spatial_dim must be a Shape or str"
            groups = (batch, *spatial_dim, channel_dim) if channels_last else (batch, channel_dim, *spatial_dim)
        result = reshaped_tensor(output, groups, convert=False)
        if result.shape.get_size(channel_dim.name) == 1:
            result = result.dimension(channel_dim.name)[0]  # remove vector dim if not required
        return result


def print_(obj: Tensor or PhiTreeNode or Number or tuple or list or None = None, name: str = ""):
    """
    Print a tensor with no more than two spatial dimensions, slicing it along all batch and channel dimensions.
    
    Unlike NumPy's array printing, the dimensions are sorted.
    Elements along the alphabetically first dimension is printed to the right, the second dimension upward.
    Typically, this means x right, y up.

    Args:
        obj: tensor-like
        name: name of the tensor

    Returns:

    """
    def variables(obj) -> dict:
        if hasattr(obj, '__variable_attrs__') or hasattr(obj, '__value_attrs__'):
            return {f".{a}": getattr(obj, a) for a in variable_attributes(obj)}
        elif isinstance(obj, (tuple, list)):
            return {f"[{i}]": item for i, item in enumerate(obj)}
        elif isinstance(obj, dict):
            return obj
        else:
            raise ValueError(f"Not PhiTreeNode: {type(obj)}")

    if name:
        print(" " * 12 + name)
    if obj is None:
        print("None")
    elif isinstance(obj, Tensor):
        print(f"{obj:full}")
    elif isinstance(obj, PhiTreeNode):
        for n, val in variables(obj).items():
            print_(val, name + n)
    else:
        print(f"{wrap(obj):full}")


def map_(function, *values, **kwargs) -> Tensor or None:
    """
    Calls `function` on all elements of `value`.

    Args:
        function: Function to be called on single elements contained in `value`. Must return a value that can be stored in tensors.
        values: Tensors to iterate over. Number of tensors must match `function` signature.
        kwargs: Keyword arguments for `function`.

    Returns:
        `Tensor` of same shape as `value`.
    """
    values = [wrap(v) for v in values]
    shape = merge_shapes(*[v.shape for v in values])
    values_reshaped = [expand(v, shape) for v in values]
    flat = [flatten(v) for v in values_reshaped]
    result = []
    for items in zip(*flat):
        result.append(function(*items, **kwargs))
    if None in result:
        assert all(r is None for r in result), f"map function returned None for some elements, {result}"
        return None
    return unpack_dim(wrap(result, channel('_c')), '_c', shape)


def _initialize(uniform_initializer, shapes: tuple) -> Tensor:
    shape = concat_shapes(*shapes)
    if shape.is_non_uniform:
        stack_dim = shape.shape.without('dims')[0:1]
        shapes = shape.unstack(stack_dim.name)
        tensors = [_initialize(uniform_initializer, s) for s in shapes]
        return stack_tensors(tensors, stack_dim)
    else:
        return uniform_initializer(shape)


def zeros(*shape: Shape, dtype=None) -> Tensor:
    """
    Define a tensor with specified shape with value `0.0` / `0` / `False` everywhere.
    
    This method may not immediately allocate the memory to store the values.

    See Also:
        `zeros_like()`, `ones()`.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: Data type as `DType` object. Defaults to `float` matching the current precision setting.

    Returns:
        `Tensor`
    """
    return _initialize(lambda shape: CollapsedTensor(NativeTensor(default_backend().zeros((), dtype=DType.as_dtype(dtype)), EMPTY_SHAPE), shape), shape)


def zeros_like(obj: Tensor or PhiTreeNode) -> Tensor or PhiTreeNode:
    """ Create a `Tensor` containing only `0.0` / `0` / `False` with the same shape and dtype as `obj`. """
    nest, values = disassemble_tree(obj)
    zeros_ = []
    for val in values:
        val = wrap(val)
        with val.default_backend:
            zeros_.append(zeros(val.shape, dtype=val.dtype))
    return assemble_tree(nest, zeros_)


def ones(*shape: Shape, dtype=None) -> Tensor:
    """
    Define a tensor with specified shape with value `1.0`/ `1` / `True` everywhere.
    
    This method may not immediately allocate the memory to store the values.

    See Also:
        `ones_like()`, `zeros()`.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: Data type as `DType` object. Defaults to `float` matching the current precision setting.

    Returns:
        `Tensor`
    """
    return _initialize(lambda shape: CollapsedTensor(NativeTensor(default_backend().ones((), dtype=DType.as_dtype(dtype)), EMPTY_SHAPE), shape), shape)


def ones_like(value: Tensor) -> Tensor:
    """ Create a `Tensor` containing only `1.0` / `1` / `True` with the same shape and dtype as `obj`. """
    return zeros_like(value) + 1


def random_normal(*shape: Shape, dtype=None) -> Tensor:
    """
    Creates a `Tensor` with the specified shape, filled with random values sampled from a normal / Gaussian distribution.

    Implementations:

    * NumPy: [`numpy.random.standard_normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html)
    * PyTorch: [`torch.randn`](https://pytorch.org/docs/stable/generated/torch.randn.html)
    * TensorFlow: [`tf.random.normal`](https://www.tensorflow.org/api_docs/python/tf/random/normal)
    * Jax: [`jax.random.normal`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html)

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: (optional) floating point `DType`. If `None`, a float tensor with the current default precision is created, see `get_precision()`.

    Returns:
        `Tensor`
    """

    def uniform_random_normal(shape):
        native = choose_backend(*shape.sizes, prefer_default=True).random_normal(shape.sizes, DType.as_dtype(dtype))
        return NativeTensor(native, shape)

    return _initialize(uniform_random_normal, shape)


def random_uniform(*shape: Shape,
                   low: Tensor or float = 0,
                   high: Tensor or float = 1,
                   dtype: DType or tuple = None) -> Tensor:
    """
    Creates a `Tensor` with the specified shape, filled with random values sampled from a uniform distribution.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: (optional) `DType` or `(kind, bits)`.
            The dtype kind must be one of `float`, `int`, `complex`.
            If not specified, a `float` tensor with the current default precision is created, see `get_precision()`.
        low: Minimum value, included.
        high: Maximum value, excluded.
    Returns:
        `Tensor`
    """
    def uniform_random_uniform(shape):
        native = choose_backend(low, high, *shape.sizes, prefer_default=True).random_uniform(shape.sizes, low, high, DType.as_dtype(dtype))
        return NativeTensor(native, shape)

    return _initialize(uniform_random_uniform, shape)


def transpose(x: Tensor, axes):
    """
    Swap the dimension order of `x`.
    This operation is superfluous since tensors will be reshaped under the hood or when getting the native/numpy representations.

    Implementations:

    * NumPy: [`numpy.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
    * PyTorch: [`x.permute`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute)
    * TensorFlow: [`tf.transpose`](https://www.tensorflow.org/api_docs/python/tf/transpose)
    * Jax: [`jax.numpy.transpose`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html)

    Args:
        x: `Tensor` or native tensor.
        axes: `tuple` or `list`

    Returns:
        `Tensor` or native tensor, depending on `x`.
    """
    if isinstance(x, Tensor):
        return CollapsedTensor(x, x.shape[axes])  # TODO avoid nesting
    else:
        return choose_backend(x).transpose(x, axes)


def cumulative_sum(x: Tensor, dim: DimFilter):
    """
    Performs a cumulative sum of `x` along `dim`.

    Implementations:

    * NumPy: [`cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
    * PyTorch: [`cumsum`](https://pytorch.org/docs/stable/generated/torch.cumsum.html)
    * TensorFlow: [`cumsum`](https://www.tensorflow.org/api_docs/python/tf/math/cumsum)
    * Jax: [`cumsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cumsum.html)

    Args:
        x: `Tensor`
        dim: Dimension along which to sum, as `str` or `Shape`.

    Returns:
        `Tensor` with the same shape as `x`.
    """
    dim = x.shape.only(dim)
    assert len(dim) == 1, f"dim must be a single dimension but got {dim}"
    native_x = x.native(x.shape)
    native_result = choose_backend(native_x).cumsum(native_x, x.shape.index(dim))
    return NativeTensor(native_result, x.shape)


def fftfreq(resolution: Shape, dx: Tensor or float = 1, dtype: DType = None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    Args:
        resolution: Grid resolution measured in cells
        dx: Distance between sampling points in real space.
        dtype: Data type of the returned tensor (Default value = None)

    Returns:
        `Tensor` holding the frequencies of the corresponding values computed by math.fft
    """
    k = meshgrid(**{dim: np.fft.fftfreq(int(n)) for dim, n in resolution.spatial._named_sizes})
    k /= dx
    return to_float(k) if dtype is None else cast(k, dtype)


def meshgrid(dim_type=spatial, stack_dim=channel('vector'), assign_item_names=True, **dimensions: int or Tensor) -> Tensor:
    """
    Generate a mesh-grid `Tensor` from keyword dimensions.

    Args:
        **dimensions: Mesh-grid dimensions, mapping names to values.
            Values may be `int`, 1D `Tensor` or 1D native tensor.
        dim_type: Dimension type of mesh-grid dimensions, one of `spatial`, `channel`, `batch`, `instance`.
        stack_dim: Vector dimension along which grids are stacked.
        assign_item_names: Whether to use the dimension names from `**dimensions` as item names for `stack_dim`.

    Returns:
        Mesh-grid `Tensor`
    """
    assert 'vector' not in dimensions
    dim_values = []
    dim_sizes = []
    for dim, spec in dimensions.items():
        if isinstance(spec, int):
            dim_values.append(tuple(range(spec)))
            dim_sizes.append(spec)
        elif isinstance(spec, Tensor):
            assert spec.rank == 1, f"Only 1D sequences allowed, got {spec} for dimension '{dim}'."
            dim_values.append(spec.native())
            dim_sizes.append(spec.shape.volume)
        else:
            backend = choose_backend(spec)
            shape = backend.staticshape(spec)
            assert len(shape) == 1, "Only 1D sequences allowed, got {spec} for dimension '{dim}'."
            dim_values.append(spec)
            dim_sizes.append(shape[0])
    backend = choose_backend(*dim_values, prefer_default=True)
    indices_list = backend.meshgrid(*dim_values)
    grid_shape = dim_type(**{dim: size for dim, size in zip(dimensions.keys(), dim_sizes)})
    channels = [NativeTensor(t, grid_shape) for t in indices_list]
    if assign_item_names:
        return stack_tensors(channels, stack_dim._with_item_names((tuple(dimensions.keys()),)))
    else:
        return stack_tensors(channels, stack_dim)


def linspace(start: int or Tensor, stop, dim: Shape) -> Tensor:
    """
    Returns `number` evenly spaced numbers between `start` and `stop`.

    See Also:
        `arange()`, `meshgrid()`.

    Args:
        start: First value, `int` or `Tensor`.
        stop: Last value, `int` or `Tensor`.
        dim: Linspace dimension of integer size.
            The size determines how many values to linearly space between `start` and `stop`.
            The values will be laid out along `dim`.

    Returns:
        `Tensor`
    """
    assert isinstance(dim, Shape) and dim.rank == 1, f"dim must be a single-dimension Shape but got {dim}"
    if is_scalar(start) and is_scalar(stop):
        native_linspace = choose_backend(start, stop, prefer_default=True).linspace(start, stop, dim.size)
        return NativeTensor(native_linspace, dim)
    else:
        return map_(linspace, start, stop, dim=dim)


def arange(dim: Shape, start_or_stop: int or None = None, stop: int or None = None, step=1):
    """
    Returns evenly spaced values between `start` and `stop`.
    If only one limit is given, `0` is used for the start.

    See Also:
        `range_tensor()`, `linspace()`, `meshgrid()`.

    Args:
        dim: Dimension name and type as `Shape` object.
            The `size` of `dim` is interpreted as `stop` unless `start_or_stop` is specified.
        start_or_stop: (Optional) `int`. Interpreted as `start` if `stop` is specified as well. Otherwise this is `stop`.
        stop: (Optional) `int`. `stop` value.
        step: Distance between values.

    Returns:
        `Tensor`
    """
    if start_or_stop is None:
        assert stop is None, "start_or_stop must be specified when stop is given."
        assert isinstance(dim.size, int), "When start_or_stop is not specified, dim.size must be an integer."
        start, stop = 0, dim.size
    elif stop is None:
        start, stop = 0, start_or_stop
    else:
        start = start_or_stop
    native = choose_backend(start, stop, prefer_default=True).range(start, stop, step, DType(int, 32))
    return NativeTensor(native, dim.with_sizes([stop - start]))


def range_tensor(shape: Shape):
    """
    Returns a `Tensor` with given `shape` containing the linear indices of each element.
    For 1D tensors, this equivalent to `arange()` with `step=1`.

    See Also:
        `arange()`, `meshgrid()`.

    Args:
        shape: Tensor shape.

    Returns:
        `Tensor`
    """
    data = arange(spatial('range'), 0, shape.volume)
    return unpack_dim(data, 'range', shape)


def stack_tensors(values: tuple or list, dim: Shape):
    values = [wrap(v) for v in values]
    values = cast_same(*values)

    def inner_stack(*values):
        return TensorStack(values, dim)

    result = broadcast_op(inner_stack, values)
    return result


def concat_tensor(values: tuple or list, dim: str) -> Tensor:
    assert len(values) > 0, "concat() got empty sequence"
    assert isinstance(dim, str), f"dim must be a single-dimension Shape but got '{dim}' of type {type(dim)}"
    broadcast_shape = merge_shapes(*[t.shape._with_item_name(dim, None).with_sizes([None] * t.shape.rank) for t in values])
    natives = [v.native(order=broadcast_shape.names) for v in values]
    backend = choose_backend(*natives)
    concatenated = backend.concat(natives, broadcast_shape.index(dim))
    if all([v.shape.get_item_names(dim) is not None for v in values]):
        broadcast_shape = broadcast_shape._with_item_name(dim, sum([v.shape.get_item_names(dim) for v in values], ()))
    return NativeTensor(concatenated, broadcast_shape.with_sizes(backend.staticshape(concatenated)))


def pad(value: Tensor, widths: dict, mode: 'e_.Extrapolation', **kwargs) -> Tensor:
    """
    Pads a tensor along the specified dimensions, determining the added values using the given extrapolation.
    Unlike `Extrapolation.pad()`, this function can handle negative widths which slice off outer values.

    Args:
        value: `Tensor` to be padded
        widths: `dict` mapping dimension name (`str`) to `(lower, upper)`
            where `lower` and `upper` are `int` that can be positive (pad), negative (slice) or zero (pass).
        mode: `Extrapolation` used to determine values added from positive `widths`.
        kwargs: Additional padding arguments.
            These are ignored by the standard extrapolations defined in `phi.math.extrapolation` but can be used to pass additional contextual information to custom extrapolations.
            Grid classes from `phi.field` will pass the argument `bounds: Box`.

    Returns:
        Padded `Tensor`
    """
    has_negative_widths = any(w[0] < 0 or w[1] < 0 for w in widths.values())
    slices = None
    if has_negative_widths:
        slices = {dim: slice(max(0, -w[0]), min(0, w[1]) or None) for dim, w in widths.items()}
        widths = {dim: (max(0, w[0]), max(0, w[1])) for dim, w in widths.items()}
    result = mode.pad(value, widths, **kwargs)
    return result[slices] if has_negative_widths else result


def closest_grid_values(grid: Tensor,
                        coordinates: Tensor,
                        extrap: 'e_.Extrapolation',
                        stack_dim_prefix='closest_',
                        **kwargs):
    """
    Finds the neighboring grid points in all spatial directions and returns their values.
    The result will have 2^d values for each vector in coordiantes in d dimensions.

    Args:
      grid: grid data. The grid is spanned by the spatial dimensions of the tensor
      coordinates: tensor with 1 channel dimension holding vectors pointing to locations in grid index space
      extrap: grid extrapolation
      stack_dim_prefix: For each spatial dimension `dim`, stacks lower and upper closest values along dimension `stack_dim_prefix+dim`.
      kwargs: Additional information for the extrapolation.

    Returns:
      Tensor of shape (batch, coord_spatial, grid_spatial=(2, 2,...), grid_channel)

    """
    return broadcast_op(functools.partial(_closest_grid_values, extrap=extrap, stack_dim_prefix=stack_dim_prefix, pad_kwargs=kwargs), [grid, coordinates])


def _closest_grid_values(grid: Tensor,
                         coordinates: Tensor,
                         extrap: 'e_.Extrapolation',
                         stack_dim_prefix: str,
                         pad_kwargs: dict):
    # alternative method: pad array for all 2^d combinations, then stack to simplify gather.
    # --- Pad tensor where transform is not possible ---
    non_copy_pad = {dim: (0 if extrap.is_copy_pad(dim, False) else 1, 0 if extrap.is_copy_pad(dim, True) else 1) for dim in grid.shape.spatial.names}
    grid = extrap.pad(grid, non_copy_pad, **pad_kwargs)
    coordinates += wrap([not extrap.is_copy_pad(dim, False) for dim in grid.shape.spatial.names], channel('vector'))
    # --- Transform coordiantes ---
    min_coords = to_int32(floor(coordinates))
    max_coords = extrap.transform_coordinates(min_coords + 1, grid.shape)
    min_coords = extrap.transform_coordinates(min_coords, grid.shape)

    def left_right(is_hi_by_axis_left, ax_idx):
        is_hi_by_axis_right = is_hi_by_axis_left | np.array([ax == ax_idx for ax in range(grid.shape.spatial_rank)])
        coords_left = where(is_hi_by_axis_left, max_coords, min_coords)
        coords_right = where(is_hi_by_axis_right, max_coords, min_coords)
        if ax_idx == grid.shape.spatial_rank - 1:
            values_left = gather(grid, coords_left)
            values_right = gather(grid, coords_right)
        else:
            values_left = left_right(is_hi_by_axis_left, ax_idx + 1)
            values_right = left_right(is_hi_by_axis_right, ax_idx + 1)
        return stack_tensors([values_left, values_right], channel(f"{stack_dim_prefix}{grid.shape.spatial.names[ax_idx]}"))

    result = left_right(np.array([False] * grid.shape.spatial_rank), 0)
    return result


def grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'e_.Extrapolation', **kwargs):
    """
    Samples values of `grid` at the locations referenced by `coordinates`.
    Values lying in between sample points are determined via linear interpolation.

    For values outside the valid bounds of `grid` (`coord < 0 or coord > grid.shape - 1`), `extrap` is used to determine the neighboring grid values.
    If the extrapolation does not support resampling, the grid is padded by one cell layer before resampling.
    In that case, values lying further outside will not be sampled according to the extrapolation.

    Args:
        grid: Grid with at least one spatial dimension and no instance dimensions.
        coordinates: Coordinates with a single channel dimension called `'vector'`.
            The size of the `vector` dimension must match the number of spatial dimensions of `grid`.
        extrap: Extrapolation used to determine the values of `grid` outside its valid bounds.
        kwargs: Additional information for the extrapolation.

    Returns:
        `Tensor` with channel dimensions of `grid`, spatial and instance dimensions of `coordinates` and combined batch dimensions.
    """
    result = broadcast_op(functools.partial(_grid_sample, extrap=extrap, pad_kwargs=kwargs), [grid, coordinates])
    return result


def _grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'e_.Extrapolation' or None, pad_kwargs: dict):
    if grid.shape.batch == coordinates.shape.batch or grid.shape.batch.volume == 1 or coordinates.shape.batch.volume == 1:
        # call backend.grid_sample()
        batch = grid.shape.batch & coordinates.shape.batch
        backend = choose_backend_t(grid, coordinates)
        result = NotImplemented
        if extrap is None:
            result = backend.grid_sample(reshaped_native(grid, [batch, *grid.shape.spatial, grid.shape.channel]),
                                         reshaped_native(coordinates, [batch, *coordinates.shape.instance, *coordinates.shape.spatial, 'vector']),
                                         'undefined')
        elif extrap.native_grid_sample_mode:
            result = backend.grid_sample(reshaped_native(grid, [batch, *grid.shape.spatial, grid.shape.channel]),
                                         reshaped_native(coordinates, [batch, *coordinates.shape.instance, *coordinates.shape.spatial, 'vector']),
                                         extrap.native_grid_sample_mode)
        if result is NotImplemented:
            # pad one layer
            grid_padded = pad(grid, {dim: (1, 1) for dim in grid.shape.spatial.names}, extrap or e_.ZERO, **pad_kwargs)
            if extrap is not None:
                from .extrapolation import _CopyExtrapolation
                if isinstance(extrap, _CopyExtrapolation):
                    inner_coordinates = extrap.transform_coordinates(coordinates, grid.shape) + 1
                else:
                    inner_coordinates = extrap.transform_coordinates(coordinates + 1, grid_padded.shape)
            else:
                inner_coordinates = coordinates + 1
            result = backend.grid_sample(reshaped_native(grid_padded, [batch, *grid_padded.shape.spatial.names, grid.shape.channel]),
                                         reshaped_native(inner_coordinates, [batch, *coordinates.shape.instance, *coordinates.shape.spatial, 'vector']),
                                         'boundary')
        if result is not NotImplemented:
            result = reshaped_tensor(result, [grid.shape.batch & coordinates.shape.batch, *coordinates.shape.instance, *coordinates.shape.spatial, grid.shape.channel])
            return result
    # fallback to slower grid sampling
    neighbors = _closest_grid_values(grid, coordinates, extrap or e_.ZERO, '_closest_', pad_kwargs)
    binary = meshgrid(**{f'_closest_{dim}': (0, 1) for dim in grid.shape.spatial.names}, dim_type=channel, assign_item_names=False)
    right_weights = coordinates % 1
    binary, right_weights = join_spaces(binary, right_weights)
    weights = prod(binary * right_weights + (1 - binary) * (1 - right_weights), 'vector')
    result = sum_(neighbors * weights, dim=[f"_closest_{dim}" for dim in grid.shape.spatial.names])
    return result


def join_spaces(*tensors):
    """
    Adds the spatial dimensions of all tensors to all other tensors.
    When spatial dimensions are present with multiple tensors, they must have the same size.

    Args:
        *tensors: Sequence of `Tensor`s.

    Returns:
        List of `Tensor`s with same values as `tensors` but additional spatial dimensions.
    """
    spatial_dims = merge_shapes(*[t.shape.spatial for t in tensors])
    return [CollapsedTensor(t, t.shape.non_spatial & spatial_dims) for t in tensors]


def broadcast_op(operation: Callable,
                 tensors: tuple or list,
                 iter_dims: set or tuple or list or Shape = None,
                 no_return=False):
    if iter_dims is None:
        iter_dims = set()
        for tensor in tensors:
            if isinstance(tensor, TensorStack) and tensor.requires_broadcast:
                iter_dims.add(tensor.stack_dim.name)
    if len(iter_dims) == 0:
        return operation(*tensors)
    else:
        if isinstance(iter_dims, Shape):
            iter_dims = iter_dims.names
        dim = next(iter(iter_dims))
        dim_type = None
        size = None
        unstacked = []
        for tensor in tensors:
            if dim in tensor.shape.names:
                unstacked_tensor = tensor.unstack(dim)
                unstacked.append(unstacked_tensor)
                if size is None:
                    size = len(unstacked_tensor)
                    dim_type = tensor.shape.get_type(dim)
                else:
                    assert size == len(unstacked_tensor)
                    assert dim_type == tensor.shape.get_type(dim)
            else:
                unstacked.append(tensor)
        result_unstacked = []
        for i in range(size):
            gathered = [t[i] if isinstance(t, tuple) else t for t in unstacked]
            result_unstacked.append(broadcast_op(operation, gathered, iter_dims=set(iter_dims) - {dim}))
        if not no_return:
            return TensorStack(result_unstacked, Shape((None,), (dim,), (dim_type,), (None,)))


def where(condition: Tensor or float or int, value_true: Tensor or float or int, value_false: Tensor or float or int):
    """
    Builds a tensor by choosing either values from `value_true` or `value_false` depending on `condition`.
    If `condition` is not of type boolean, non-zero values are interpreted as True.
    
    This function requires non-None values for `value_true` and `value_false`.
    To get the indices of True / non-zero values, use :func:`nonzero`.

    Args:
      condition: determines where to choose values from value_true or from value_false
      value_true: Values to pick where `condition != 0 / True`
      value_false: Values to pick where `condition == 0 / False`

    Returns:
        `Tensor` containing dimensions of all inputs.
    """
    condition = wrap(condition)
    value_true = wrap(value_true)
    value_false = wrap(value_false)

    def inner_where(c: Tensor, vt: Tensor, vf: Tensor):
        if vt._is_tracer or vf._is_tracer or c._is_tracer:
            return c * vt + (1 - c) * vf  # ToDo this does not take NaN into account
        shape, (c, vt, vf) = broadcastable_native_tensors(c, vt, vf)
        result = choose_backend(c, vt, vf).where(c, vt, vf)
        return NativeTensor(result, shape)

    return broadcast_op(inner_where, [condition, value_true, value_false])


def nonzero(value: Tensor, list_dim: Shape or str = instance('nonzero'), index_dim: Shape = channel('vector')):
    """
    Get spatial indices of non-zero / True values.
    
    Batch dimensions are preserved by this operation.
    If channel dimensions are present, this method returns the indices where any component is nonzero.

    Implementations:

    * NumPy: [`numpy.argwhere`](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html)
    * PyTorch: [`torch.nonzero`](https://pytorch.org/docs/stable/generated/torch.nonzero.html)
    * TensorFlow: [`tf.where(tf.not_equal(values, 0))`](https://www.tensorflow.org/api_docs/python/tf/where)
    * Jax: [`jax.numpy.nonzero`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nonzero.html)

    Args:
        value: spatial tensor to find non-zero / True values in.
        list_dim: Dimension listing non-zero values.
        index_dim: Index dimension.

    Returns:
        `Tensor` of shape (batch dims..., `list_dim`=#non-zero, `index_dim`=value.shape.spatial_rank)

    """
    if value.shape.channel_rank > 0:
        value = sum_(abs(value), value.shape.channel)

    if isinstance(list_dim, str):
        list_dim = instance(list_dim)

    def unbatched_nonzero(value: Tensor):
        native = reshaped_native(value, [*value.shape.spatial])
        backend = choose_backend(native)
        indices = backend.nonzero(native)
        indices_shape = Shape(backend.staticshape(indices), (list_dim.name, index_dim.name), (list_dim.type, index_dim.type), (None, value.shape.spatial.names))
        return NativeTensor(indices, indices_shape)

    return broadcast_op(unbatched_nonzero, [value], iter_dims=value.shape.batch.names)


def _reduce(value: Tensor or list or tuple,
            dim: DimFilter,
            dtype: type or None,
            native_function: Callable,
            collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
            unaffected_function: Callable = lambda value: value) -> Tensor:
    """
    Args:
        value:
        dim: Which dimensions should be reduced
        dtype: (Optional) Whether the reducing operation converts the data to a different type like bool.
        native_function:
        collapsed_function: handles collapsed dimensions, called as `collapsed_function(inner_reduced, collapsed_dims_to_reduce)`
        unaffected_function: returns `unaffected_function(value)` if `len(dims) > 0` but none of them are part of `value`
    """
    if dim in ((), [], EMPTY_SHAPE):
        return value
    else:
        if isinstance(value, (tuple, list)):
            values = [wrap(v) for v in value]
            value = stack_tensors(values, instance('0'))
            assert dim in ('0', None), "dim must be '0' or None when passing a sequence of tensors"
        else:
            value = wrap(value)
        dims = value.shape.only(dim)
        return value._tensor_reduce(dims.names, dtype, native_function, collapsed_function, unaffected_function)


def sum_(value: Tensor or list or tuple, dim: DimFilter = non_batch) -> Tensor:
    """
    Sums `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return _reduce(value, dim, float,
                   native_function=lambda backend, native, dim: backend.sum(native, dim),
                   collapsed_function=lambda inner, red_shape: inner * red_shape.volume)


def prod(value: Tensor or list or tuple, dim: DimFilter = non_batch) -> Tensor:
    """
    Multiplies `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return _reduce(value, dim, None,
                   native_function=lambda backend, native, dim: backend.prod(native, dim),
                   collapsed_function=lambda inner, red_shape: inner ** red_shape.volume)


def mean(value: Tensor or list or tuple, dim: DimFilter = non_batch) -> Tensor:
    """
    Computes the mean over `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return _reduce(value, dim, float, native_function=lambda backend, native, dim: backend.mean(native, dim))


def std(value: Tensor or list or tuple, dim: DimFilter = non_batch) -> Tensor:
    """
    Computes the standard deviation over `values` along the specified dimensions.

    *Warning*: The standard deviation of non-uniform tensors along the stack dimension is undefined.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return _reduce(cached(value), dim, float,
                   native_function=lambda backend, native, dim: backend.std(native, dim),
                   collapsed_function=lambda inner, red_shape: inner,
                   unaffected_function=lambda value: value * 0)


def any_(boolean_tensor: Tensor or list or tuple, dim: DimFilter = non_batch) -> Tensor:
    """
    Tests whether any entry of `boolean_tensor` is `True` along the specified dimensions.

    Args:
        boolean_tensor: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return _reduce(boolean_tensor, dim, bool, native_function=lambda backend, native, dim: backend.any(native, dim))


def all_(boolean_tensor: Tensor or list or tuple, dim: DimFilter = non_batch) -> Tensor:
    """
    Tests whether all entries of `boolean_tensor` are `True` along the specified dimensions.

    Args:
        boolean_tensor: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return _reduce(boolean_tensor, dim, bool, native_function=lambda backend, native, dim: backend.all(native, dim))


def max_(value: Tensor or list or tuple, dim: DimFilter = non_batch) -> Tensor:
    """
    Determines the maximum value of `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return _reduce(value, dim, None, native_function=lambda backend, native, dim: backend.max(native, dim))


def min_(value: Tensor or list or tuple, dim: DimFilter = non_batch) -> Tensor:
    """
    Determines the minimum value of `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return _reduce(value, dim, None, native_function=lambda backend, native, dim: backend.min(native, dim))


def finite_min(value, dim: DimFilter = non_batch, default: complex or float = float('NaN')):
    """
    Finds the minimum along `dim` ignoring all non-finite values.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    value_inf = where(is_finite(value), value, float('inf'))
    result_inf = min_(value_inf, dim)
    return where(is_finite(result_inf), result_inf, default)


def finite_max(value, dim: DimFilter = non_batch, default: complex or float = float('NaN')):
    """
    Finds the maximum along `dim` ignoring all non-finite values.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    value_inf = where(is_finite(value), value, float('-inf'))
    result_inf = max_(value_inf, dim)
    return where(is_finite(result_inf), result_inf, default)


def finite_sum(value, dim: DimFilter = non_batch, default: complex or float = float('NaN')):
    """
    Sums all finite values in `value` along `dim`.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    finite = is_finite(value)
    summed = sum_(where(finite, value, 0), dim)
    return where(any_(finite, dim), summed, default)


def finite_mean(value, dim: DimFilter = non_batch, default: complex or float = float('NaN')):
    """
    Computes the mean value of all finite values in `value` along `dim`.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    finite = is_finite(value)
    summed = sum_(where(finite, value, 0), dim)
    count = sum_(finite, dim)
    mean_nan = summed / count
    return where(is_finite(mean_nan), mean_nan, default)


def quantile(value: Tensor,
             quantiles: float or tuple or list or Tensor,
             dim: DimFilter = non_batch):
    """
    Compute the q-th quantile of `value` along `dim` for each q in `quantiles`.

    Implementations:

    * NumPy: [`quantile`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
    * PyTorch: [`quantile`](https://pytorch.org/docs/stable/generated/torch.quantile.html#torch.quantile)
    * TensorFlow: [`tfp.stats.percentile`](https://www.tensorflow.org/probability/api_docs/python/tfp/stats/percentile)
    * Jax: [`quantile`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.quantile.html)

    Args:
        value: `Tensor`
        quantiles: Single quantile or tensor of quantiles to compute.
            Must be of type `float`, `tuple`, `list` or `Tensor`.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to reduce the sequence of Tensors

    Returns:
        `Tensor` with dimensions of `quantiles` and non-reduced dimensions of `value`.
    """
    dims = value.shape.only(dim)
    native_values = reshaped_native(value, [*value.shape.without(dims), value.shape.only(dims)])
    backend = choose_backend(native_values)
    q = tensor(quantiles, default_list_dim=instance('quantiles'))
    native_quantiles = reshaped_native(q, [q.shape])
    native_result = backend.quantile(native_values, native_quantiles)
    return reshaped_tensor(native_result, [q.shape, *value.shape.without(dims)])


def median(value, dim: DimFilter = non_batch):
    """
    Reduces `dim` of `value` by picking the median value.
    For odd dimension sizes (ambigous choice), the linear average of the two median values is computed.

    Currently implemented via `quantile()`.

    Args:
        value: `Tensor`
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor`
    """
    return quantile(value, 0.5, dim)


def dot(x: Tensor,
        x_dims: DimFilter,
        y: Tensor,
        y_dims: DimFilter) -> Tensor:
    """
    Computes the dot product along the specified dimensions.
    Contracts `x_dims` with `y_dims` by first multiplying the elements and then summing them up.

    For one dimension, this is equal to matrix-matrix or matrix-vector multiplication.

    The function replaces the traditional `dot` / `tensordot` / `matmul` / `einsum` functions.

    * NumPy: [`numpy.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html), [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
    * PyTorch: [`torch.tensordot`](https://pytorch.org/docs/stable/generated/torch.tensordot.html#torch.tensordot), [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html)
    * TensorFlow: [`tf.tensordot`](https://www.tensorflow.org/api_docs/python/tf/tensordot), [`tf.einsum`](https://www.tensorflow.org/api_docs/python/tf/einsum)
    * Jax: [`jax.numpy.tensordot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tensordot.html), [`jax.numpy.einsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html)

    Args:
        x: First `Tensor`
        x_dims: Dimensions of `x` to reduce against `y`
        y: Second `Tensor`
        y_dims: Dimensions of `y` to reduce against `x`.

    Returns:
        Dot product as `Tensor`.
    """
    x_dims = x.shape.only(x_dims)
    y_dims = y.shape.only(y_dims)
    if not x_dims:
        assert y_dims.volume == 1, f"Cannot compute dot product between dimensions {x_dims} on {x.shape} and {y_dims} on {y.shape}"
        y = y[{d: 0 for d in y_dims.names}]
        return x * y
    if not y_dims:
        assert x_dims.volume == 1, f"Cannot compute dot product between dimensions {x_dims} on {x.shape} and {y_dims} on {y.shape}"
        x = x[{d: 0 for d in x_dims.names}]
        return x * y
    x_native = x.native(x.shape)
    y_native = y.native(y.shape)
    backend = choose_backend(x_native, y_native)
    remaining_shape_x = x.shape.without(x_dims)
    remaining_shape_y = y.shape.without(y_dims)
    assert x_dims.volume == y_dims.volume, f"Failed to reduce {x_dims} against {y_dims} in dot product of {x.shape} and {y.shape}. Sizes do not match."
    if remaining_shape_y.only(remaining_shape_x).is_empty:  # no shared batch dimensions -> tensordot
        result_native = backend.tensordot(x_native, x.shape.indices(x_dims), y_native, y.shape.indices(y_dims))
        result_shape = concat_shapes(remaining_shape_x, remaining_shape_y)
    else:  # shared batch dimensions -> einsum
        REDUCE_LETTERS = list('ijklmn')
        KEEP_LETTERS = list('abcdefgh')
        x_letters = [(REDUCE_LETTERS if dim in x_dims else KEEP_LETTERS).pop(0) for dim in x.shape.names]
        x_letter_map = {dim: letter for dim, letter in zip(x.shape.names, x_letters)}
        REDUCE_LETTERS = list('ijklmn')
        y_letters = []
        for dim in y.shape.names:
            if dim in y_dims:
                y_letters.append(REDUCE_LETTERS.pop(0))
            else:
                if dim in x.shape and dim not in x_dims:
                    y_letters.append(x_letter_map[dim])
                else:
                    y_letters.append(KEEP_LETTERS.pop(0))
        keep_letters = list('abcdefgh')[:-len(KEEP_LETTERS)]
        subscripts = f'{"".join(x_letters)},{"".join(y_letters)}->{"".join(keep_letters)}'
        result_native = backend.einsum(subscripts, x_native, y_native)
        result_shape = merge_shapes(x.shape.without(x_dims), y.shape.without(y_dims))  # don't check group match  ToDo the order might be incorrect here
    return NativeTensor(result_native, result_shape)


def _backend_op1(x, unbound_method) -> Tensor or PhiTreeNode:
    if isinstance(x, Tensor):
        def apply_op(native_tensor):
            return getattr(choose_backend(native_tensor), unbound_method.__name__)(native_tensor)
        apply_op.__name__ = unbound_method.__name__
        return x._op1(apply_op)
    elif isinstance(x, PhiTreeNode):
        return copy_with(x, **{a: _backend_op1(getattr(x, a), unbound_method) for a in value_attributes(x)})
    else:
        backend = choose_backend(x)
        y = getattr(backend, unbound_method.__name__)(x)
        return y


def abs_(x) -> Tensor or PhiTreeNode:
    """
    Computes *||x||<sub>1</sub>*.
    Complex `x` result in matching precision float values.

    *Note*: The gradient of this operation is undefined for *x=0*.
    TensorFlow and PyTorch return 0 while Jax returns 1.

    Args:
        x: `Tensor` or `PhiTreeNode`

    Returns:
        Absolute value of `x` of same type as `x`.
    """
    return _backend_op1(x, Backend.abs)


def sign(x) -> Tensor or PhiTreeNode:
    """
    The sign of positive numbers is 1 and -1 for negative numbers.
    The sign of 0 is undefined.

    Args:
        x: `Tensor` or `PhiTreeNode`

    Returns:
        `Tensor` or `PhiTreeNode` matching `x`.
    """
    return _backend_op1(x, Backend.sign)


def round_(x) -> Tensor or PhiTreeNode:
    """ Rounds the `Tensor` or `PhiTreeNode` `x` to the closest integer. """
    return _backend_op1(x, Backend.round)


def ceil(x) -> Tensor or PhiTreeNode:
    """ Computes *⌈x⌉* of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.ceil)


def floor(x) -> Tensor or PhiTreeNode:
    """ Computes *⌊x⌋* of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.floor)


def sqrt(x) -> Tensor or PhiTreeNode:
    """ Computes *sqrt(x)* of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sqrt)


def exp(x) -> Tensor or PhiTreeNode:
    """ Computes *exp(x)* of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.exp)


def to_float(x) -> Tensor or PhiTreeNode:
    """
    Converts the given tensor to floating point format with the currently specified precision.
    
    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.
    
    See the `phi.math` module documentation at https://tum-pbs.github.io/PhiFlow/Math.html

    See Also:
        `cast()`.

    Args:
        x: `Tensor` or `PhiTreeNode` to convert

    Returns:
        `Tensor` or `PhiTreeNode` matching `x`.
    """
    return _backend_op1(x, Backend.to_float)


def to_int32(x) -> Tensor or PhiTreeNode:
    """ Converts the `Tensor` or `PhiTreeNode` `x` to 32-bit integer. """
    return _backend_op1(x, Backend.to_int32)


def to_int64(x) -> Tensor or PhiTreeNode:
    """ Converts the `Tensor` or `PhiTreeNode` `x` to 64-bit integer. """
    return _backend_op1(x, Backend.to_int64)


def to_complex(x) -> Tensor or PhiTreeNode:
    """
    Converts the given tensor to complex floating point format with the currently specified precision.

    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.

    See the `phi.math` module documentation at https://tum-pbs.github.io/PhiFlow/Math.html

    See Also:
        `cast()`.

    Args:
        x: values to convert

    Returns:
        `Tensor` of same shape as `x`
    """
    return _backend_op1(x, Backend.to_complex)


def is_finite(x) -> Tensor or PhiTreeNode:
    """ Returns a `Tensor` or `PhiTreeNode` matching `x` with values `True` where `x` has a finite value and `False` otherwise. """
    return _backend_op1(x, Backend.isfinite)


def real(x) -> Tensor or PhiTreeNode:
    """
    See Also:
        `imag()`, `conjugate()`.

    Args:
        x: `Tensor` or `PhiTreeNode` or native tensor.

    Returns:
        Real component of `x`.
    """
    return _backend_op1(x, Backend.real)


def imag(x) -> Tensor or PhiTreeNode:
    """
    Returns the imaginary part of `x`.
    If `x` does not store complex numbers, returns a zero tensor with the same shape and dtype as this tensor.

    See Also:
        `real()`, `conjugate()`.

    Args:
        x: `Tensor` or `PhiTreeNode` or native tensor.

    Returns:
        Imaginary component of `x` if `x` is complex, zeros otherwise.
    """
    return _backend_op1(x, Backend.imag)


def conjugate(x) -> Tensor or PhiTreeNode:
    """
    See Also:
        `imag()`, `real()`.

    Args:
        x: Real or complex `Tensor` or `PhiTreeNode` or native tensor.

    Returns:
        Complex conjugate of `x` if `x` is complex, else `x`.
    """
    return _backend_op1(x, Backend.conj)


def degrees(deg):
    """ Convert degrees to radians. """
    return deg * (3.1415 / 180.)


def sin(x) -> Tensor or PhiTreeNode:
    """ Computes *sin(x)* of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sin)


def arcsin(x) -> Tensor or PhiTreeNode:
    """ Computes the inverse of *sin(x)* of the `Tensor` or `PhiTreeNode` `x`.
    For real arguments, the result lies in the range [-π/2, π/2].
    """
    return _backend_op1(x, Backend.arcsin)


def cos(x) -> Tensor or PhiTreeNode:
    """ Computes *cos(x)* of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.cos)


def arccos(x) -> Tensor or PhiTreeNode:
    """ Computes the inverse of *cos(x)* of the `Tensor` or `PhiTreeNode` `x`.
    For real arguments, the result lies in the range [0, π].
    """
    return _backend_op1(x, Backend.cos)


def tan(x) -> Tensor or PhiTreeNode:
    """ Computes *tan(x)* of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.tan)


def log(x) -> Tensor or PhiTreeNode:
    """ Computes the natural logarithm of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.log)


def log2(x) -> Tensor or PhiTreeNode:
    """ Computes *log(x)* of the `Tensor` or `PhiTreeNode` `x` with base 2. """
    return _backend_op1(x, Backend.log2)


def log10(x) -> Tensor or PhiTreeNode:
    """ Computes *log(x)* of the `Tensor` or `PhiTreeNode` `x` with base 10. """
    return _backend_op1(x, Backend.log10)


def sigmoid(x) -> Tensor or PhiTreeNode:
    """ Computes the sigmoid function of the `Tensor` or `PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sigmoid)


def cast_same(*values: Tensor) -> Tuple[Tensor]:
    """
    Casts all tensors to the same `DType`.
    If all data types are of the same kind, returns the largest occurring data type.
    Otherwise casts `bool` &rarr; `int` &rarr; `float` &rarr; `complex`.

    Args:
        *values: tensors to cast

    Returns:
        Tuple of Tensors with same data type.
    """
    assert all(isinstance(v, Tensor) for v in values), f"Only Tensor arguments allowed but got {values}"
    dtypes = [v.dtype for v in values]
    if any(dt != dtypes[0] for dt in dtypes):
        common_type = combine_types(*dtypes, fp_precision=get_precision())
        return tuple([cast(v, common_type) for v in values])
    else:
        return values


def divide_no_nan(x: float or Tensor, y: float or Tensor):
    """ Computes *x/y* with the `Tensor`s `x` and `y` but returns 0 where *y=0*. """
    return custom_op2(x, y,
                      l_operator=divide_no_nan,
                      l_native_function=lambda x_, y_: choose_backend(x_, y_).divide_no_nan(x_, y_),
                      r_operator=lambda y_, x_: divide_no_nan(x_, y_),
                      r_native_function=lambda y_, x_: choose_backend(x_, y_).divide_no_nan(x_, y_),
                      op_name='divide_no_nan')


def maximum(x: Tensor or float, y: Tensor or float):
    """ Computes the element-wise maximum of `x` and `y`. """
    return custom_op2(x, y, maximum, lambda x_, y_: choose_backend(x_, y_).maximum(x_, y_), op_name='maximum')


def minimum(x: Tensor or float, y: Tensor or float):
    """ Computes the element-wise minimum of `x` and `y`. """
    return custom_op2(x, y, minimum, lambda x_, y_: choose_backend(x_, y_).minimum(x_, y_), op_name='minimum')


def clip(x: Tensor, lower_limit: float or Tensor, upper_limit: float or Tensor):
    """ Limits the values of the `Tensor` `x` to lie between `lower_limit` and `upper_limit` (inclusive). """
    if isinstance(lower_limit, Number) and isinstance(upper_limit, Number):

        def clip_(x):
            return x._op1(lambda native: choose_backend(native).clip(native, lower_limit, upper_limit))

        return broadcast_op(clip_, [x])
    else:
        return maximum(lower_limit, minimum(x, upper_limit))


def convolve(value: Tensor,
             kernel: Tensor,
             extrapolation: 'e_.Extrapolation' = None) -> Tensor:
    """
    Computes the convolution of `value` and `kernel` along the spatial axes of `kernel`.

    The channel dimensions of `value` are reduced against the equally named dimensions of `kernel`.
    The result will have the non-reduced channel dimensions of `kernel`.

    Args:
        value: `Tensor` whose shape includes all spatial dimensions of `kernel`.
        kernel: `Tensor` used as convolutional filter.
        extrapolation: If not None, pads `value` so that the result has the same shape as `value`.

    Returns:
        `Tensor`
    """
    assert all(dim in value.shape for dim in kernel.shape.spatial.names), f"Value must have all spatial dimensions of kernel but got value {value} kernel {kernel}"
    conv_shape = kernel.shape.spatial
    in_channels = value.shape.channel
    out_channels = kernel.shape.channel.without(in_channels)
    batch = value.shape.batch & kernel.shape.batch
    if extrapolation is not None and extrapolation != e_.ZERO:
        value = pad(value, {dim: (kernel.shape.get_size(dim) // 2, (kernel.shape.get_size(dim) - 1) // 2)
                            for dim in conv_shape.name}, extrapolation)
    native_kernel = reshaped_native(kernel, (batch, out_channels, in_channels, *conv_shape.names), force_expand=in_channels)
    native_value = reshaped_native(value, (batch, in_channels, *conv_shape.names), force_expand=batch)
    backend = choose_backend(native_value, native_kernel)
    native_result = backend.conv(native_value, native_kernel, zero_padding=extrapolation == e_.ZERO)
    result = reshaped_tensor(native_result, (batch, out_channels, *conv_shape))
    return result


def boolean_mask(x: Tensor, dim: str, mask: Tensor):
    """
    Discards values `x.dim[i]` where `mask.dim[i]=False`.
    All dimensions of `mask` that are not `dim` are treated as batch dimensions.

    Alternative syntax: `x.dim[mask]`.

    Implementations:

    * NumPy: Slicing
    * PyTorch: [`masked_select`](https://pytorch.org/docs/stable/generated/torch.masked_select.html)
    * TensorFlow: [`tf.boolean_mask`](https://www.tensorflow.org/api_docs/python/tf/boolean_mask)
    * Jax: Slicing

    Args:
        x: `Tensor` of values.
        dim: Dimension of `x` to along which to discard slices.
        mask: Boolean `Tensor` marking which values to keep. Must have the dimension `dim` matching `x´.

    Returns:
        Selected values of `x` as `Tensor` with dimensions from `x` and `mask`.
    """
    assert dim in mask.shape, f"mask dimension '{dim}' must be present on the mask but got {mask.shape}"
    
    def uniform_boolean_mask(x: Tensor, mask_1d: Tensor):
        if dim in x.shape:
            x_native = x.native(x.shape.names)  # order does not matter
            mask_native = mask_1d.native()  # only has 1 dim
            backend = choose_backend(x_native, mask_native)
            result_native = backend.boolean_mask(x_native, mask_native, axis=x.shape.index(dim))
            new_shape = x.shape.with_sizes(backend.staticshape(result_native))
            return NativeTensor(result_native, new_shape)
        else:
            total = int(sum_(to_int64(mask_1d), mask_1d.shape))
            new_shape = mask_1d.shape.with_sizes([total])
            return expand(x, new_shape)

    return broadcast_op(uniform_boolean_mask, [x, mask], iter_dims=mask.shape.without(dim))


def gather(values: Tensor, indices: Tensor, dims: DimFilter or None = None):
    """
    Gathers the entries of `values` at positions described by `indices`.

    See Also:
        `scatter()`.

    Args:
        values: `Tensor` containing values to gather.
        indices: `int` `Tensor`. Multi-dimensional position references in `values`.
            Must contain a single channel dimension for the index vector matching the number of `dims`.
        dims: Dimensions indexed by `indices`.
            If `None`, will default to all spatial dimensions or all instance dimensions, depending on which ones are present (but not both).

    Returns:
        `Tensor` with combined batch dimensions, channel dimensions of `values` and spatial/instance dimensions of `indices`.
    """
    if dims is None:
        assert values.shape.instance.is_empty or values.shape.spatial.is_empty, f"Specify gather dimensions for values with both instance and spatial dimensions. Got {values.shape}"
        dims = values.shape.instance if values.shape.spatial.is_empty else values.shape.spatial
    if indices.dtype.kind == bool:
        indices = to_int32(indices)
    dims = parse_dim_order(dims)
    batch = (values.shape.batch & indices.shape.batch).without(dims)
    channel = values.shape.without(dims).without(batch)
    native_values = reshaped_native(values, [batch, *dims, channel])
    native_indices = reshaped_native(indices, [batch, *indices.shape.non_batch.non_channel, indices.shape.channel])
    backend = choose_backend(native_values, native_indices)
    native_result = backend.batched_gather_nd(native_values, native_indices)
    result = reshaped_tensor(native_result, [batch, *indices.shape.non_channel.non_batch, channel])
    return result


def scatter(base_grid: Tensor or Shape,
            indices: Tensor,
            values: Tensor,
            mode: str = 'update',
            outside_handling: str = 'discard',
            indices_gradient=False):
    """
    Scatters `values` into `base_grid` at `indices`.
    instance dimensions of `indices` and/or `values` are reduced during scattering.
    Depending on `mode`, this method has one of the following effects:

    * `mode='update'`: Replaces the values of `base_grid` at `indices` by `values`. The result is undefined if `indices` contains duplicates.
    * `mode='add'`: Adds `values` to `base_grid` at `indices`. The values corresponding to duplicate indices are accumulated.
    * `mode='mean'`: Replaces the values of `base_grid` at `indices` by the mean of all `values` with the same index.

    Implementations:

    * NumPy: Slice assignment / `numpy.add.at`
    * PyTorch: [`torch.scatter`](https://pytorch.org/docs/stable/generated/torch.scatter.html), [`torch.scatter_add`](https://pytorch.org/docs/stable/generated/torch.scatter_add.html)
    * TensorFlow: [`tf.tensor_scatter_nd_add`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_add), [`tf.tensor_scatter_nd_update`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update)
    * Jax: [`jax.lax.scatter_add`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter_add.html), [`jax.lax.scatter`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter.html)

    See Also:
        `gather()`.

    Args:
        base_grid: `Tensor` into which `values` are scattered.
        indices: `Tensor` of n-dimensional indices at which to place `values`.
            Must have a single channel dimension with size matching the number of spatial dimensions of `base_grid`.
            This dimension is optional if the spatial rank is 1.
            Must also contain all `scatter_dims`.
        values: `Tensor` of values to scatter at `indices`.
        mode: Scatter mode as `str`. One of ('add', 'mean', 'update')
        outside_handling: Defines how indices lying outside the bounds of `base_grid` are handled.

            * `'discard'`: outside indices are ignored.
            * `'clamp'`: outside indices are projected onto the closest point inside the grid.
            * `'undefined'`: All points are expected to lie inside the grid. Otherwise an error may be thrown or an undefined tensor may be returned.
        indices_gradient: Whether to allow the gradient of this operation to be backpropagated through `indices`.

    Returns:
        Copy of `base_grid` with updated values at `indices`.
    """
    assert mode in ('update', 'add', 'mean')
    assert outside_handling in ('discard', 'clamp', 'undefined')
    assert isinstance(indices_gradient, bool)
    grid_shape = base_grid if isinstance(base_grid, Shape) else base_grid.shape
    assert indices.shape.channel.names == ('vector',) or (grid_shape.spatial_rank + grid_shape.instance_rank == 1 and indices.shape.channel_rank == 0)
    batches = values.shape.non_channel.non_instance & indices.shape.non_channel.non_instance
    channels = grid_shape.channel & values.shape.channel
    # --- Set up grid ---
    if isinstance(base_grid, Shape):
        with choose_backend_t(indices, values):
            base_grid = zeros(base_grid & batches & values.shape.channel)
        if mode != 'add':
            base_grid += math.nan
    # --- Handle outside indices ---
    if outside_handling == 'clamp':
        indices = clip(indices, 0, tensor(grid_shape.spatial, channel('vector')) - 1)
    elif outside_handling == 'discard':
        indices_inside = min_((round_(indices) >= 0) & (round_(indices) < tensor(grid_shape.spatial, channel('vector'))), 'vector')
        indices = boolean_mask(indices, indices.shape.instance.name, indices_inside)
        if instance(values).rank > 0:
            values = boolean_mask(values, values.shape.instance.name, indices_inside)
        if indices.shape.is_non_uniform:
            raise NotImplementedError()
    lists = indices.shape.instance & values.shape.instance

    def scatter_forward(base_grid, indices, values):
        indices = to_int32(round_(indices))
        native_grid = reshaped_native(base_grid, [batches, *base_grid.shape.instance, *base_grid.shape.spatial, channels], force_expand=True)
        native_values = reshaped_native(values, [batches, lists, channels], force_expand=True)
        native_indices = reshaped_native(indices, [batches, lists, 'vector'], force_expand=True)
        backend = choose_backend(native_indices, native_values, native_grid)
        if mode in ('add', 'update'):
            native_result = backend.scatter(native_grid, native_indices, native_values, mode=mode)
        else:  # mean
            zero_grid = backend.zeros_like(native_grid)
            summed = backend.scatter(zero_grid, native_indices, native_values, mode='add')
            count = backend.scatter(zero_grid, native_indices, backend.ones_like(native_values), mode='add')
            native_result = summed / backend.maximum(count, 1)
            native_result = backend.where(count == 0, native_grid, native_result)
        return reshaped_tensor(native_result, [batches, *instance(base_grid), *spatial(base_grid), channels], check_sizes=True)

    def scatter_backward(shaped_base_grid_, shaped_indices_, shaped_values_, output, d_output):
        from ._nd import spatial_gradient
        values_grad = gather(d_output, shaped_indices_)
        spatial_gradient_indices = gather(spatial_gradient(d_output), shaped_indices_)
        indices_grad = mean(spatial_gradient_indices * shaped_values_, 'vector_')
        return None, indices_grad, values_grad

    scatter_function = scatter_forward
    if indices_gradient:
        from phi.math import custom_gradient
        scatter_function = custom_gradient(scatter_forward, scatter_backward)

    result = scatter_function(base_grid, indices, values)
    return result


def fft(x: Tensor, dims: DimFilter = spatial) -> Tensor:
    """
    Performs a fast Fourier transform (FFT) on all spatial dimensions of x.
    
    The inverse operation is `ifft()`.

    Implementations:

    * NumPy: [`np.fft.fft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html),
      [`numpy.fft.fft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html),
      [`numpy.fft.fftn`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html)
    * PyTorch: [`torch.fft.fft`](https://pytorch.org/docs/stable/fft.html)
    * TensorFlow: [`tf.signal.fft`](https://www.tensorflow.org/api_docs/python/tf/signal/fft),
      [`tf.signal.fft2d`](https://www.tensorflow.org/api_docs/python/tf/signal/fft2d),
      [`tf.signal.fft3d`](https://www.tensorflow.org/api_docs/python/tf/signal/fft3d)
    * Jax: [`jax.numpy.fft.fft`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fft.html),
      [`jax.numpy.fft.fft2`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fft2.html)
      [`jax.numpy.fft.fft`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fftn.html)

    Args:
        x: Uniform complex or float `Tensor` with at least one spatial dimension.
        dims: Dimensions along which to perform the FFT.
            If `None`, performs the FFT along all spatial dimensions of `x`.

    Returns:
        *Ƒ(x)* as complex `Tensor`
    """
    dims = x.shape.only(dims)
    x_native = x.native(x.shape)
    result_native = choose_backend(x_native).fft(x_native, x.shape.indices(dims))
    return NativeTensor(result_native, x.shape)


def ifft(k: Tensor, dims: DimFilter = spatial):
    """
    Inverse of `fft()`.

    Args:
        k: Complex or float `Tensor` with at least one spatial dimension.
        dims: Dimensions along which to perform the inverse FFT.
            If `None`, performs the inverse FFT along all spatial dimensions of `k`.

    Returns:
        *Ƒ<sup>-1</sup>(k)* as complex `Tensor`
    """
    dims = k.shape.only(dims)
    k_native = k.native(k.shape)
    result_native = choose_backend(k_native).ifft(k_native, k.shape.indices(dims))
    return NativeTensor(result_native, k.shape)


def dtype(x) -> DType:
    """
    Returns the data type of `x`.

    Args:
        x: `Tensor` or native tensor.

    Returns:
        `DType`
    """
    if isinstance(x, Tensor):
        return x.dtype
    else:
        return choose_backend(x).dtype(x)


def expand_tensor(value: float or Tensor, dims: Shape):
    value = wrap(value)
    shape = value.shape
    for dim in reversed(dims):
        if dim in value.shape:
            shape &= dim  # checks sizes, copies item names
        else:
            if dim.size is None:
                dim = dim.with_sizes([1])
            shape = concat_shapes(dim, shape)
    return CollapsedTensor(value, shape)


def close(*tensors, rel_tolerance=1e-5, abs_tolerance=0) -> bool:
    """
    Checks whether all tensors have equal values within the specified tolerance.
    
    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    Args:
        *tensors: `Tensor` or tensor-like (constant) each
        rel_tolerance: relative tolerance (Default value = 1e-5)
        abs_tolerance: absolute tolerance (Default value = 0)

    Returns:
        Whether all given tensors are equal to the first tensor within the specified tolerance.
    """
    tensors = [wrap(t) for t in tensors]
    for other in tensors[1:]:
        if not _close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance):
            return False
    return True


def _close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return True
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = choose_backend(native1).numpy(native1)
    np2 = choose_backend(native2).numpy(native2)
    return np.allclose(np1, np2, rel_tolerance, abs_tolerance)


def assert_close(*values,
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0,
                 msg: str = "",
                 verbose: bool = True):
    """
    Checks that all given tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.
    
    Does not check that the shapes match as long as they can be broadcast to a common shape.

    Args:
      values: Tensors or native tensors or numbers or sequences of numbers.
      rel_tolerance: Relative tolerance.
      abs_tolerance: Absolute tolerance.
      msg: Optional error message.
      verbose: Whether to print conflicting values.
    """
    if not values:
        return
    phi_tensors = [t for t in values if isinstance(t, Tensor)]
    if phi_tensors:
        values = [compatible_tensor(t, phi_tensors[0].shape)._simplify() for t in values]  # use Tensor to infer dimensions
        for other in values[1:]:
            _assert_close(values[0], other, rel_tolerance, abs_tolerance, msg, verbose)
    elif all(isinstance(v, PhiTreeNode) for v in values):
        tree0, tensors0 = disassemble_tree(values[0])
        for value in values[1:]:
            tree, tensors_ = disassemble_tree(value)
            assert tree0 == tree, f"Tree structures do not match: {tree0} and {tree}"
            for t0, t in zip(tensors0, tensors_):
                _assert_close(t0, t, rel_tolerance, abs_tolerance, msg, verbose)
    else:
        np_values = [choose_backend(t).numpy(t) for t in values]
        for other in np_values[1:]:
            np.testing.assert_allclose(np_values[0], other, rel_tolerance, abs_tolerance, err_msg=msg, verbose=verbose)


def _assert_close(tensor1: Tensor, tensor2: Tensor, rel_tolerance: float, abs_tolerance: float, msg: str, verbose: bool):
    if tensor2 is tensor1:
        return
    # if isinstance(tensor2, (int, float, bool)):
    #     np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)
    if isinstance(tensor1, Layout):
        tensor1._assert_close(tensor2, rel_tolerance, abs_tolerance, msg, verbose)
    elif isinstance(tensor2, Layout):
        tensor2._assert_close(tensor1, rel_tolerance, abs_tolerance, msg, verbose)
    else:
        def inner_assert_close(tensor1, tensor2):
            new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
            np1 = choose_backend(native1).numpy(native1)
            np2 = choose_backend(native2).numpy(native2)
            if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
                np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance, err_msg=msg, verbose=verbose)

        broadcast_op(inner_assert_close, [tensor1, tensor2], no_return=True)


def _native_wrapper(tensor_function: Callable, create_native_function: Callable, persistent_refs=False):
    INPUT_TENSORS = []
    OUTPUT_TENSORS = []

    def native_function(*natives):
        natives = list(natives)
        values = [t._op1(lambda _: natives.pop(0)) for t in INPUT_TENSORS]
        assert len(natives) == 0, "Not all arguments were converted"
        result = tensor_function(*values)
        results = [result] if not isinstance(result, (tuple, list)) else result
        OUTPUT_TENSORS.clear()
        OUTPUT_TENSORS.extend(results)
        return sum([v._natives() for v in results], ())

    backend = default_backend()
    traced = create_native_function(native_function, backend)
    if traced is NotImplemented:
        warnings.warn(f"Backend '{backend}' not supported. Returning original function.", RuntimeWarning)
        return tensor_function, None, INPUT_TENSORS, OUTPUT_TENSORS

    def wrapper(*values: Tensor):
        INPUT_TENSORS.clear()
        INPUT_TENSORS.extend(values)
        for v in values:
            v._expand()
        natives = sum([v._natives() for v in values], ())
        results_native = list(traced(*natives))
        results = [t._with_natives_replaced(results_native) for t in OUTPUT_TENSORS]
        if not persistent_refs:
            INPUT_TENSORS.clear()
            # OUTPUT_TENSORS.clear()  outputs need to be saved because native_function may be called only the first time. Will get garbage collected once the function is not referenced anymore.
        assert len(results_native) == 0
        return results[0] if len(results) == 1 else results

    return wrapper, traced, INPUT_TENSORS, OUTPUT_TENSORS


def stop_gradient(x):
    """
    Disables gradients for the given tensor.
    This may switch off the gradients for `x` itself or create a copy of `x` with disabled gradients.

    Implementations:

    * PyTorch: [`x.detach()`](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach)
    * TensorFlow: [`tf.stop_gradient`](https://www.tensorflow.org/api_docs/python/tf/stop_gradient)
    * Jax: [`jax.lax.stop_gradient`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.stop_gradient.html)

    Args:
        x: `Tensor` or `PhiTreeNode` for which gradients should be disabled.

    Returns:
        Copy of `x`.
    """
    if isinstance(x, Tensor):
        return x._op1(lambda native: choose_backend(native).stop_gradient(native))
    elif isinstance(x, PhiTreeNode):
        nest, values = disassemble_tree(x)
        new_values = [stop_gradient(v) for v in values]
        return assemble_tree(nest, new_values)
    else:
        return wrap(choose_backend(x).stop_gradient(x))

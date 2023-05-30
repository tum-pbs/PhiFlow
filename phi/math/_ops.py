import functools
import math
import warnings
from numbers import Number
from typing import Tuple, Callable, Any, Union, Optional

import numpy as np

from . import extrapolation as e_
from ._magic_ops import expand, pack_dims, unpack_dim, cast, copy_with, value_attributes, bool_to_int, tree_map, concat, stack
from ._shape import (Shape, EMPTY_SHAPE,
                     spatial, batch, channel, instance, merge_shapes, parse_dim_order, concat_shapes,
                     IncompatibleShapes, DimFilter, non_batch, dual, non_channel, shape)
from ._sparse import CompressedSparseMatrix, dot_compressed_dense, dense, SparseCoordinateTensor, dot_coordinate_dense, get_format, to_format, stored_indices, tensor_like, sparse_dims, same_sparsity_pattern, is_sparse
from ._tensors import (Tensor, wrap, tensor, broadcastable_native_tensors, NativeTensor, TensorStack,
                       custom_op2, compatible_tensor, variable_attributes, disassemble_tree, assemble_tree,
                       is_scalar, Layout, expand_tensor)
from .backend import default_backend, choose_backend, Backend, get_precision, convert as b_convert, BACKENDS, NoBackendFound, ComputeDevice, NUMPY
from .backend._dtype import DType, combine_types
from .magic import PhiTreeNode, Shapable


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
    Convert the native representation of a `Tensor` or `phi.math.magic.PhiTreeNode` to the native format of `backend`.

    *Warning*: This operation breaks the automatic differentiation chain.

    See Also:
        `phi.math.backend.convert()`.

    Args:
        x: `Tensor` to convert. If `x` is a `phi.math.magic.PhiTreeNode`, its variable attributes are converted.
        backend: Target backend. If `None`, uses the current default backend, see `phi.math.backend.default_backend()`.

    Returns:
        `Tensor` with native representation belonging to `backend`.
    """
    if isinstance(x, Tensor):
        return x._op1(lambda native: b_convert(native, backend, use_dlpack=use_dlpack))
    elif isinstance(x, PhiTreeNode):
        return copy_with(x, **{a: convert(getattr(x, a), backend, use_dlpack=use_dlpack) for a in variable_attributes(x)})
    else:
        return b_convert(x, backend, use_dlpack=use_dlpack)


def to_device(value, device: ComputeDevice or str, convert=True, use_dlpack=True):
    """
    Allocates the tensors of `value` on `device`.
    If the value already exists on that device, this function may either create a copy of `value` or return `value` directly.

    See Also:
        `to_cpu()`.

    Args:
        value: `Tensor` or `phi.math.magic.PhiTreeNode` or native tensor.
        device: Device to allocate value on.
            Either `ComputeDevice` or category `str`, such as `'CPU'` or `'GPU'`.
        convert: Whether to convert tensors that do not belong to the corresponding backend to compatible native tensors.
            If `False`, this function has no effect on numpy tensors.
        use_dlpack: Only if `convert==True`.
            Whether to use the DLPack library to convert from one GPU-enabled backend to another.

    Returns:
        Same type as `value`.
    """
    assert isinstance(device, (ComputeDevice, str)), f"device must be a ComputeDevice or str but got {type(device)}"
    return tree_map(_to_device, value, device=device, convert_to_backend=convert, use_dlpack=use_dlpack)


def _to_device(value: Tensor or Any, device: ComputeDevice or str, convert_to_backend: bool, use_dlpack: bool):
    if isinstance(value, Tensor):
        if not convert and value.default_backend == NUMPY:
            return value
        natives = [_to_device(n, device, convert_to_backend, use_dlpack) for n in value._natives()]
        return value._with_natives_replaced(natives)
    else:
        old_backend = choose_backend(value)
        if isinstance(device, str):
            device = old_backend.list_devices(device)[0]
        if old_backend != device.backend:
            if convert_to_backend:
                value = b_convert(value, device.backend, use_dlpack=use_dlpack)
            else:
                return value
        return device.backend.allocate_on_device(value, device)


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
    return all([v.available for v in values])


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


def native(value: Union[Tensor, Number, tuple, list, Any]):
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


def numpy(value: Union[Tensor, Number, tuple, list, Any]):
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
                    groups: Union[tuple, list],
                    force_expand: Any = True,
                    to_numpy=False):
    """
    Returns a native representation of `value` where dimensions are laid out according to `groups`.

    See Also:
        `native()`, `pack_dims()`, `reshaped_tensor()`, `reshaped_numpy()`.

    Args:
        value: `Tensor`
        groups: `tuple` or `list` of dimensions to be packed into one native dimension. Each entry must be one of the following:

            * `str`: the name of one dimension that is present on `value`.
            * `Shape`: Dimensions to be packed. If `force_expand`, missing dimensions are first added, otherwise they are ignored.
            * Filter function: Packs all dimensions of this type that are present on `value`.

        force_expand: `bool` or sequence of dimensions.
            If `True`, repeats the tensor along missing dimensions.
            If `False`, puts singleton dimensions where possible.
            If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.
        to_numpy: If True, converts the native tensor to a `numpy.ndarray`.

    Returns:
        Native tensor with dimensions matching `groups`.
    """
    assert isinstance(value, Tensor), f"value must be a Tensor but got {type(value)}"
    assert value.shape.is_uniform, f"Only uniform (homogenous) tensors can be converted to native but got shape {value.shape}"
    assert isinstance(groups, (tuple, list)), f"groups must be a tuple or list but got {type(value)}"
    order = []
    groups = [group(value) if callable(group) else group for group in groups]
    for i, group in enumerate(groups):
        if isinstance(group, Shape):
            present = value.shape.only(group)
            if force_expand is True or present.volume > 1 or (force_expand is not False and group.only(force_expand).volume > 1):
                value = expand(value, group)
            value = pack_dims(value, group, batch(f"group{i}"))
            order.append(f"group{i}")
        else:
            assert isinstance(group, str), f"Groups must be either single-dim str or Shape but got {group}"
            assert ',' not in group, f"When packing multiple dimensions, pass a well-defined Shape instead of a comma-separated str. Got {group}"
            order.append(group)
    return value.numpy(order) if to_numpy else value.native(order)


def reshaped_numpy(value: Tensor, groups: Union[tuple, list], force_expand: Any = True):
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
                    groups: Union[tuple, list],
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
        raise IncompatibleShapes(f"Cannot reshape native tensor {type(value)} with sizes {value.shape} given groups {groups}")
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
        natives.append(reshaped_native(i, groups, force_expand=False))
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
        if result.shape.get_size(channel_dim.name) == 1 and not channel_dim.item_names[0]:
            result = result.dimension(channel_dim.name)[0]  # remove vector dim if not required
        return result


def print_(obj: Union[Tensor, PhiTreeNode, Number, tuple, list, None] = None, name: str = ""):
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


def map_(function, *values, range=range, **kwargs) -> Union[Tensor, None]:
    """
    Calls `function` on all elements of `values`.

    Args:
        function: Function to be called on single elements contained in `value`. Must return a value that can be stored in tensors.
        *values: `Tensors` containing positional arguments for `function`.
            Number of tensors must match `function` signature.
        range: Range function. Can be used to generate tqdm output by passing `trange`.
        **kwargs: Non-`Tensor` keyword arguments for `function`.
            Their shapes are not broadcast with the positional arguments.

    Returns:
        `Tensor` of same shape as `value`.
    """
    if not values:
        return function(**kwargs)
    values = [v if isinstance(v, Shapable) else wrap(v) for v in values]
    shape = merge_shapes(*[v.shape for v in values])
    flat = [pack_dims(expand(v, shape), shape, channel(flat=shape.volume)) for v in values]
    result = []
    results = None
    for _, items in zip(range(flat[0].flat.size_or_1), zip(*flat)):
        f_output = function(*items, **kwargs)
        if isinstance(f_output, tuple):
            if results is None:
                results = [[] for _ in f_output]
            for result_i, output_i in zip(results, f_output):
                result_i.append(output_i)
        else:
            result.append(f_output)
    if results is None:
        if any(r is None for r in result):
            assert all(r is None for r in result), f"map function returned None for some elements, {result}"
            return None
        return unpack_dim(stack(result, channel('_c')) if isinstance(result, Shapable) else wrap(result, channel('_c')), '_c', shape)
    else:
        for i, result_i in enumerate(results):
            if any(r is None for r in result_i):
                assert all(r is None for r in result_i), f"map function returned None for some elements at output index {i}, {result_i}"
                results[i] = None
        return tuple([unpack_dim(stack(result_i, channel('_c')) if isinstance(result_i, Shapable) else wrap(result_i, channel('_c')), '_c', shape) for result_i in results])


def _initialize(uniform_initializer, shapes: Tuple[Shape]) -> Tensor:
    shape = concat_shapes(*shapes)
    assert shape.well_defined, f"When creating a Tensor, shape needs to have definitive sizes but got {shape}"
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
    return _initialize(lambda shape: expand_tensor(NativeTensor(default_backend().zeros((), dtype=DType.as_dtype(dtype)), EMPTY_SHAPE), shape), shape)


def zeros_like(obj: Union[Tensor, PhiTreeNode]) -> Union[Tensor, PhiTreeNode]:
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
    return _initialize(lambda shape: expand_tensor(NativeTensor(default_backend().ones((), dtype=DType.as_dtype(dtype)), EMPTY_SHAPE), shape), shape)


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
                   low: Union[Tensor, float] = 0,
                   high: Union[Tensor, float] = 1,
                   dtype: Union[DType, tuple] = None) -> Tensor:
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
        return expand(x, x.shape[axes])
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


def fftfreq(resolution: Shape, dx: Union[Tensor, float] = 1, dtype: DType = None):
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


def meshgrid(dims: Union[Callable, Shape] = spatial, stack_dim=channel('vector'), **dimensions: Union[int, Tensor]) -> Tensor:
    """
    Generate a mesh-grid `Tensor` from keyword dimensions.

    Args:
        **dimensions: Mesh-grid dimensions, mapping names to values.
            Values may be `int`, 1D `Tensor` or 1D native tensor.
        dims: Dimension type of mesh-grid dimensions, one of `spatial`, `channel`, `batch`, `instance`.
        stack_dim: Channel dim along which grids are stacked.
            This is optional for 1D mesh-grids. In that case returns a `Tensor` without a stack dim if `None` or an empty `Shape` is passed.

    Returns:
        Mesh-grid `Tensor` with the dimensions of `dims` / `dimensions` and `stack_dim`.

    Examples:
        >>> math.meshgrid(x=2, y=2)
        (xˢ=2, yˢ=2, vectorᶜ=x,y) 0.500 ± 0.500 (0e+00...1e+00)

        >>> math.meshgrid(x=2, y=(-1, 1))
        (xˢ=2, yˢ=2, vectorᶜ=x,y) 0.250 ± 0.829 (-1e+00...1e+00)

        >>> math.meshgrid(x=2, stack_dim=None)
        (0, 1) along xˢ
    """
    assert 'dim_type' not in dimensions, f"dim_type has been renamed to dims"
    assert not stack_dim or stack_dim.name not in dimensions
    if isinstance(dims, Shape):
        assert not dimensions, f"When passing a Shape to meshgrid(), no kwargs are allowed"
        dimensions = {d: s for d, s in zip(dims.names, dims.sizes)}
        grid_shape = dims
        dim_values = [tuple(range(s)) for s in dims.sizes]
    else:
        dim_type = dims
        assert callable(dim_type), f"dims must be a Shape or dimension type but got {dims}"
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
        grid_shape = dim_type(**{dim: size for dim, size in zip(dimensions.keys(), dim_sizes)})
    backend = choose_backend(*dim_values, prefer_default=True)
    indices_list = backend.meshgrid(*dim_values)
    channels = [NativeTensor(t, grid_shape) for t in indices_list]
    if not stack_dim:
        assert len(channels) == 1, f"meshgrid with multiple dimension requires a valid stack_dim but got {stack_dim}"
        return channels[0]
    if stack_dim.item_names[0] is None:
        stack_dim = stack_dim.with_size(tuple(dimensions.keys()))
    return stack_tensors(channels, stack_dim)


def linspace(start: Union[float, Tensor], stop: Union[float, Tensor], dim: Shape) -> Tensor:
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

    Examples:
        >>> math.linspace(0, 1, spatial(x=5))
        (0.000, 0.250, 0.500, 0.750, 1.000) along xˢ

        >>> math.linspace(0, (-1, 1), spatial(x=3))
        (0.000, 0.000); (-0.500, 0.500); (-1.000, 1.000) (xˢ=3, vectorᶜ=2)
    """
    assert isinstance(dim, Shape) and dim.rank == 1, f"dim must be a single-dimension Shape but got {dim}"
    if is_scalar(start) and is_scalar(stop):
        if isinstance(start, Tensor):
            start = start.native()
        if isinstance(stop, Tensor):
            stop = stop.native()
        native_linspace = choose_backend(start, stop, prefer_default=True).linspace(start, stop, dim.size)
        return NativeTensor(native_linspace, dim)
    else:
        return map_(linspace, start, stop, dim=dim)


def arange(dim: Shape, start_or_stop: Union[int, None] = None, stop: Union[int, None] = None, step=1):
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


def range_tensor(*shape: Shape):
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
    shape = concat_shapes(*shape)
    data = arange(spatial('range'), 0, shape.volume)
    return unpack_dim(data, 'range', shape)


def stack_tensors(values: Union[tuple, list], dim: Shape):
    if len(values) == 1 and not dim:
        return values[0]
    values = [wrap(v) for v in values]
    values = cast_same(*values)

    def inner_stack(*values):
        if len(values) > 1 or not isinstance(values[0], NativeTensor):
            if all(isinstance(t, SparseCoordinateTensor) for t in values):
                if all(values[0]._indices is t._indices for t in values):
                    return values[0]._with_values(stack_tensors([v._values for v in values], dim))
            return TensorStack(values, dim)
        else:
            value: NativeTensor = values[0]
            return NativeTensor(value._native, value._native_shape, value.shape & dim.with_size(1))

    result = broadcast_op(inner_stack, values)
    return result


def concat_tensor(values: Union[tuple, list], dim: str) -> Tensor:
    assert len(values) > 0, "concat() got empty sequence"
    assert isinstance(dim, str), f"dim must be a single-dimension Shape but got '{dim}' of type {type(dim)}"

    def inner_concat(*values):
        broadcast_shape: Shape = values[0].shape  # merge_shapes(*[t.shape.with_sizes([None] * t.shape.rank) for t in values])
        dim_index = broadcast_shape.index(dim)
        natives = [v.native(order=broadcast_shape.names) for v in values]
        concatenated = choose_backend(*natives).concat(natives, dim_index)
        if all([v.shape.get_item_names(dim) is not None for v in values]):
            broadcast_shape = broadcast_shape.with_dim_size(dim, sum([v.shape.get_item_names(dim) for v in values], ()))
        else:
            broadcast_shape = broadcast_shape.with_dim_size(dim, sum([v.shape.get_size(dim) for v in values]))
        return NativeTensor(concatenated, broadcast_shape)

    result = broadcast_op(inner_concat, values)
    return result


def pad(value: Tensor, widths: dict, mode: Union['e_.Extrapolation', Tensor, Number], **kwargs) -> Tensor:
    """
    Pads a tensor along the specified dimensions, determining the added values using the given extrapolation.
    Unlike `Extrapolation.pad()`, this function can handle negative widths which slice off outer values.

    Args:
        value: `Tensor` to be padded
        widths: `dict` mapping dimension name (`str`) to `(lower, upper)`
            where `lower` and `upper` are `int` that can be positive (pad), negative (slice) or zero (pass).
        mode: `Extrapolation` used to determine values added from positive `widths`.
            Assumes constant extrapolation if given a number or `Tensor` instead.
        kwargs: Additional padding arguments.
            These are ignored by the standard extrapolations defined in `phi.math.extrapolation` but can be used to pass additional contextual information to custom extrapolations.
            Grid classes from `phi.field` will pass the argument `bounds: Box`.

    Returns:
        Padded `Tensor`

    Examples:
        >>> math.pad(math.ones(spatial(x=10, y=10)), {'x': (1, 1), 'y': (2, 1)}, 0)
        (xˢ=12, yˢ=13) 0.641 ± 0.480 (0e+00...1e+00)

        >>> math.pad(math.ones(spatial(x=10, y=10)), {'x': (1, -1)}, 0)
        (xˢ=10, yˢ=10) 0.900 ± 0.300 (0e+00...1e+00)
    """
    mode = mode if isinstance(mode, e_.Extrapolation) else e_.ConstantExtrapolation(mode)
    has_negative_widths = any(w0 < 0 or w1 < 0 for w0, w1 in widths.values())
    has_positive_widths = any(w0 > 0 or w1 > 0 for w0, w1 in widths.values())
    slices = None
    if has_negative_widths:
        slices = {dim: slice(max(0, -w[0]), min(0, w[1]) or None) for dim, w in widths.items()}
        widths = {dim: (max(0, w[0]), max(0, w[1])) for dim, w in widths.items()}
    result_padded = mode.pad(value, widths, **kwargs) if has_positive_widths else value
    result_sliced = result_padded[slices] if has_negative_widths else result_padded
    return result_sliced


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


def _grid_sample(grid: Tensor, coordinates: Tensor, extrap: Union['e_.Extrapolation', None], pad_kwargs: dict):
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
    binary = meshgrid(channel, **{f'_closest_{dim}': (0, 1) for dim in grid.shape.spatial.names}, stack_dim=channel(coordinates))
    right_weights = coordinates % 1
    weights = prod(binary * right_weights + (1 - binary) * (1 - right_weights), 'vector')
    result = sum_(neighbors * weights, dim=[f"_closest_{dim}" for dim in grid.shape.spatial.names])
    return result


def broadcast_op(operation: Callable,
                 tensors: Union[tuple, list],
                 iter_dims: Union[set, tuple, list, Shape] = None,
                 no_return=False):
    if iter_dims is None:
        iter_dims = set()
        for tensor in tensors:
            iter_dims.update(tensor.shape.shape.without('dims').names)
            if isinstance(tensor, TensorStack) and tensor.requires_broadcast:
                iter_dims.add(tensor._stack_dim.name)
    if len(iter_dims) == 0:
        return operation(*tensors)
    else:
        if isinstance(iter_dims, Shape):
            iter_dims = iter_dims.names
        dim = next(iter(iter_dims))
        dim_type = None
        size = None
        item_names = None
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
                if item_names is None:
                    item_names = tensor.shape.get_item_names(dim)
            else:
                unstacked.append(tensor)
        result_unstacked = []
        for i in range(size):
            gathered = [t[i] if isinstance(t, tuple) else t for t in unstacked]
            result_unstacked.append(broadcast_op(operation, gathered, iter_dims=set(iter_dims) - {dim}))
        if not no_return:
            return TensorStack(result_unstacked, Shape((size,), (dim,), (dim_type,), (item_names,)))


def where(condition: Union[Tensor, float, int], value_true: Union[Tensor, float, int], value_false: Union[Tensor, float, int]):
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
        if is_sparse(vt) or is_sparse(vf):
            if same_sparsity_pattern(vt, vf, allow_const=True) and same_sparsity_pattern(c, vt, allow_const=True):
                c_values = c._values if is_sparse(c) else c
                vt_values = vt._values if is_sparse(vt) else vt
                vf_values = vf._values if is_sparse(vf) else vf
                result_values = where(c_values, vt_values, vf_values)
                return c._with_values(result_values)
            raise NotImplementedError
        shape, (c, vt, vf) = broadcastable_native_tensors(c, vt, vf)
        result = choose_backend(c, vt, vf).where(c, vt, vf)
        return NativeTensor(result, shape)

    return broadcast_op(inner_where, [condition, value_true, value_false])


def nonzero(value: Tensor, list_dim: Union[Shape, str] = instance('nonzero'), index_dim: Shape = channel('vector')):
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
        if isinstance(value, CompressedSparseMatrix):
            value = value.decompress()
        if isinstance(value, SparseCoordinateTensor):
            nonzero_values = nonzero(value._values)
            nonzero_indices = value._indices[nonzero_values]
            return nonzero_indices
        else:
            dims = value.shape.non_channel
            native = reshaped_native(value, [*dims])
            backend = choose_backend(native)
            indices = backend.nonzero(native)
            indices_shape = Shape(backend.staticshape(indices), (list_dim.name, index_dim.name), (list_dim.type, index_dim.type), (None, dims.names))
            return NativeTensor(indices, indices_shape)
    return broadcast_op(unbatched_nonzero, [value], iter_dims=value.shape.batch.names)


def reduce_(f, value, dims, require_all_dims_present=False, required_kind: type = None):
    if not dims:
        return value
    else:
        if isinstance(value, (tuple, list)):
            values = [wrap(v) for v in value]
            value = stack_tensors(values, instance('0'))
            dims = value.shape.only(dims)
            assert '0' in dims, "When passing a sequence of tensors to be reduced, the sequence dimension '0' must be reduced."
        elif isinstance(value, Layout):
            if not value.shape.without(dims):  # reduce all
                dims = batch('_flat_layout')
                values = value._as_list()
                if required_kind is not None:
                    values = [required_kind(v) for v in values]
                value = wrap(values, dims)
        else:
            value = wrap(value)
        dims = value.shape.only(dims)
        if require_all_dims_present and any(d not in value.shape for d in dims):
            raise ValueError(f"Cannot sum dimensions {dims} because tensor {value.shape} is missing at least one of them")
        return f(value._simplify(), dims)


def sum_(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
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
    return reduce_(_sum, bool_to_int(value), dim, require_all_dims_present=True)


def _sum(value: Tensor, dims: Shape) -> Tensor:
    if not dims:
        return value
    if isinstance(value, NativeTensor):
        result = value.default_backend.sum(value._native, value._native_shape.indices(dims)) * value.collapsed_dims.only(dims).volume
        return NativeTensor(result, value._native_shape.without(dims), value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_sum(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x + y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (CompressedSparseMatrix, SparseCoordinateTensor)):
        if value.sparse_dims in dims:  # reduce all sparse dims
            return _sum(value._values, dims.without(value.sparse_dims) & instance(value._values))
        value_only_dims = dims.only(value._values.shape).without(value.sparsity_batch)
        if value_only_dims:
            value = value._with_values(_sum(value._values, value_only_dims))
        dims = dims.without(value_only_dims)
        if not dims:
            return value
        if isinstance(value, CompressedSparseMatrix):
            if value._compressed_dims in dims and value._uncompressed_dims.isdisjoint(dims):  # We can ignore the pointers
                result_base = zeros(value.shape.without(value._compressed_dims))
                return scatter(result_base, value._indices, value._values, mode='add', outside_handling='undefined')
            elif value.sparse_dims.only(dims):  # reduce some sparse dims
                return dot(value, dims, ones(dims), dims)  # this is what SciPy does in both axes, actually.
            return value
            # first sum value dims that are not part of indices
        else:
            assert isinstance(value, SparseCoordinateTensor)
            if value._dense_shape in dims:  # sum all sparse dims
                v_dims = dims.without(value._dense_shape) & instance(value._values)
                return _sum(value._values, v_dims)
            else:
                result_base = zeros(value.shape.without(dims))
                remaining_sparse_dims = value._dense_shape.without(dims)
                indices = value._indices.vector[remaining_sparse_dims.names]
                if remaining_sparse_dims.rank == 1:  # return dense result
                    result = scatter(result_base, indices, value._values, mode='add', outside_handling='undefined')
                    return result
                else:  # return sparse result
                    raise NotImplementedError
    else:
        raise ValueError(type(value))


def prod(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
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
    return reduce_(_prod, value, dim, require_all_dims_present=True)


def _prod(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.prod(value._native, value._native_shape.indices(dims)) ** value.collapsed_dims.only(dims).volume
        return NativeTensor(result, value._native_shape.without(dims), value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_prod(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x * y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    else:
        raise ValueError(type(value))


def mean(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
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
    return reduce_(_mean, value, dim)


def _mean(value: Tensor, dims: Shape) -> Tensor:
    if not dims:
        return value
    if isinstance(value, NativeTensor):
        result = value.default_backend.mean(value._native, value._native_shape.indices(dims))
        return NativeTensor(result, value._native_shape.without(dims), value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_mean(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x + y, reduced_inners) / len(reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    else:
        raise ValueError(type(value))


def std(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
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
    if not dim:
        warnings.warn("std along empty shape returns 0", RuntimeWarning, stacklevel=2)
        return zeros_like(value)
    if not callable(dim) and set(parse_dim_order(dim)) - set(value.shape.names):
        return zeros_like(value)  # std along constant dim is 0
    return reduce_(_std, value, dim)


def _std(value: Tensor, dims: Shape) -> Tensor:
    if value.shape.is_uniform:
        result = value.default_backend.std(value.native(value.shape), value.shape.indices(dims))
        return NativeTensor(result, value.shape.without(dims))
    else:
        non_uniform_dim = value.shape.shape.without('dims')
        assert non_uniform_dim.only(dims).is_empty, f"Cannot compute std along non-uniform dims {dims}. shape={value.shape}"
        return stack([_std(t, dims) for t in value.unstack(non_uniform_dim.name)], non_uniform_dim)


def any_(boolean_tensor: Union[Tensor, list, tuple], dim: DimFilter = non_batch) -> Tensor:
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
    return reduce_(_any, boolean_tensor, dim)


def _any(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.any(value._native, value._native_shape.indices(dims))
        return NativeTensor(result, value._native_shape.without(dims), value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_any(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x | y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    else:
        raise ValueError(type(value))


def all_(boolean_tensor: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
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
    return reduce_(_all, boolean_tensor, dim)


def _all(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.all(value.native(value.shape), value.shape.indices(dims))
        return NativeTensor(result, value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_all(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x & y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (SparseCoordinateTensor, CompressedSparseMatrix)):
        if sparse_dims(value) in dims:
            values_all = _all(value._values, dims.without(sparse_dims(value)) & instance(value._values))
            return all_([values_all, value._default], '0') if value._default is not None else values_all
    raise ValueError(type(value))


def max_(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
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
    return reduce_(_max, value, dim)


def _max(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.max(value.native(value.shape), value.shape.indices(dims))
        return NativeTensor(result, value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_max(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: maximum(x, y), reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (SparseCoordinateTensor, CompressedSparseMatrix)):
        if sparse_dims(value) in dims:
            values_max = _max(value._values, dims.without(sparse_dims(value)) & instance(value._values))
            return maximum(values_max, value._default) if value._default is not None else values_max
    raise ValueError(type(value))


def min_(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
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
    return reduce_(_min, value, dim)


def _min(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.min(value.native(value.shape), value.shape.indices(dims))
        return NativeTensor(result, value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_min(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: minimum(x, y), reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (SparseCoordinateTensor, CompressedSparseMatrix)):
        if sparse_dims(value) in dims:
            values_min = _min(value._values, dims.without(sparse_dims(value)) & instance(value._values))
            return minimum(values_min, value._default) if value._default is not None else values_min
    raise ValueError(type(value))


def finite_min(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
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


def finite_max(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
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


def finite_sum(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
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


def finite_mean(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
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
             quantiles: Union[float, tuple, list, Tensor],
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
    if isinstance(x, CompressedSparseMatrix):
        if isinstance(y, (CompressedSparseMatrix, SparseCoordinateTensor)):
            if x_dims.isdisjoint(sparse_dims(x)) and y_dims.isdisjoint(sparse_dims(y)):
                return x._op2(y, lambda vx, vy: dot(vx, x_dims, vy, y_dims), None, 'dot', '@')
            if x_dims.only(sparse_dims(x)) and y_dims.only(sparse_dims(y)):
                raise NotImplementedError("sparse-sparse multiplication not yet supported")
            raise NotImplementedError
        return dot_compressed_dense(x, x_dims, y, y_dims)
    elif isinstance(y, CompressedSparseMatrix):
        if isinstance(x, (CompressedSparseMatrix, SparseCoordinateTensor)):
            raise NotImplementedError("sparse-sparse multiplication not yet supported")
        return dot_compressed_dense(y, y_dims, x, x_dims)
    if isinstance(x, SparseCoordinateTensor):
        if isinstance(y, (CompressedSparseMatrix, SparseCoordinateTensor)):
            if x_dims.isdisjoint(sparse_dims(x)) and y_dims.isdisjoint(sparse_dims(y)):
                return x._op2(y, lambda vx, vy: dot(vx, x_dims, vy, y_dims), None, 'dot', '@')
            raise NotImplementedError("sparse-sparse multiplication not yet supported")
        return dot_coordinate_dense(x, x_dims, y, y_dims)
    elif isinstance(y, SparseCoordinateTensor):
        if isinstance(x, (CompressedSparseMatrix, SparseCoordinateTensor)):
            raise NotImplementedError("sparse-sparse multiplication not yet supported")
        return dot_coordinate_dense(y, y_dims, x, x_dims)
    x_native = x.native(x.shape)
    y_native = y.native(y.shape)
    backend = choose_backend(x_native, y_native)
    remaining_shape_x = x.shape.without(x_dims)
    remaining_shape_y = y.shape.without(y_dims)
    assert x_dims.volume == y_dims.volume, f"Failed to reduce {x_dims} against {y_dims} in dot product of {x.shape} and {y.shape}. Sizes do not match."
    if remaining_shape_y.isdisjoint(remaining_shape_x):  # no shared batch dimensions -> tensordot
        result_native = backend.tensordot(x_native, x.shape.indices(x_dims), y_native, y.shape.indices(y_dims))
        result_shape = concat_shapes(remaining_shape_x, remaining_shape_y)
    else:  # shared batch dimensions -> einsum
        result_shape = merge_shapes(x.shape.without(x_dims), y.shape.without(y_dims))
        REDUCE_LETTERS = list('ijklmn')
        KEEP_LETTERS = list('abcdefgh')
        x_letters = [(REDUCE_LETTERS if dim in x_dims else KEEP_LETTERS).pop(0) for dim in x.shape.names]
        letter_map = {dim: letter for dim, letter in zip(x.shape.names, x_letters)}
        REDUCE_LETTERS = list('ijklmn')
        y_letters = []
        for dim in y.shape.names:
            if dim in y_dims:
                y_letters.append(REDUCE_LETTERS.pop(0))
            else:
                if dim in x.shape and dim not in x_dims:
                    y_letters.append(letter_map[dim])
                else:
                    next_letter = KEEP_LETTERS.pop(0)
                    letter_map[dim] = next_letter
                    y_letters.append(next_letter)
        keep_letters = [letter_map[dim] for dim in result_shape.names]
        subscripts = f'{"".join(x_letters)},{"".join(y_letters)}->{"".join(keep_letters)}'
        result_native = backend.einsum(subscripts, x_native, y_native)
    return NativeTensor(result_native, result_shape)


def _backend_op1(x, unbound_method) -> Union[Tensor, PhiTreeNode]:
    if isinstance(x, Tensor):
        def apply_op(native_tensor):
            backend = choose_backend(native_tensor)
            return getattr(backend, unbound_method.__name__)(backend.auto_cast(native_tensor)[0])
        apply_op.__name__ = unbound_method.__name__
        return x._op1(apply_op)
    elif isinstance(x, PhiTreeNode):
        return copy_with(x, **{a: _backend_op1(getattr(x, a), unbound_method) for a in value_attributes(x)})
    else:
        backend = choose_backend(x)
        y = getattr(backend, unbound_method.__name__)(backend.auto_cast(x)[0])
        return y


def abs_(x) -> Union[Tensor, PhiTreeNode]:
    """
    Computes *||x||<sub>1</sub>*.
    Complex `x` result in matching precision float values.

    *Note*: The gradient of this operation is undefined for *x=0*.
    TensorFlow and PyTorch return 0 while Jax returns 1.

    Args:
        x: `Tensor` or `phi.math.magic.PhiTreeNode`

    Returns:
        Absolute value of `x` of same type as `x`.
    """
    return _backend_op1(x, Backend.abs)


def sign(x) -> Union[Tensor, PhiTreeNode]:
    """
    The sign of positive numbers is 1 and -1 for negative numbers.
    The sign of 0 is undefined.

    Args:
        x: `Tensor` or `phi.math.magic.PhiTreeNode`

    Returns:
        `Tensor` or `phi.math.magic.PhiTreeNode` matching `x`.
    """
    return _backend_op1(x, Backend.sign)


def round_(x) -> Union[Tensor, PhiTreeNode]:
    """ Rounds the `Tensor` or `phi.math.magic.PhiTreeNode` `x` to the closest integer. """
    return _backend_op1(x, Backend.round)


def ceil(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *⌈x⌉* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.ceil)


def floor(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *⌊x⌋* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.floor)


def sqrt(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *sqrt(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sqrt)


def exp(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *exp(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.exp)


def soft_plus(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *softplus(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.softplus)


def factorial(x) -> Union[Tensor, PhiTreeNode]:
    """
    Computes *factorial(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`.
    For floating-point numbers computes the continuous factorial using the gamma function.
    For integer numbers computes the exact factorial and returns the same integer type.
    However, this results in integer overflow for inputs larger than 12 (int32) or 19 (int64).
    """
    return _backend_op1(x, Backend.factorial)


def log_gamma(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *log(gamma(x))* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.log_gamma)


def to_float(x) -> Union[Tensor, PhiTreeNode]:
    """
    Converts the given tensor to floating point format with the currently specified precision.
    
    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.
    
    See the `phi.math` module documentation at https://tum-pbs.github.io/PhiFlow/Math.html

    See Also:
        `cast()`.

    Args:
        x: `Tensor` or `phi.math.magic.PhiTreeNode` to convert

    Returns:
        `Tensor` or `phi.math.magic.PhiTreeNode` matching `x`.
    """
    return _backend_op1(x, Backend.to_float)


def to_int32(x) -> Union[Tensor, PhiTreeNode]:
    """ Converts the `Tensor` or `phi.math.magic.PhiTreeNode` `x` to 32-bit integer. """
    return _backend_op1(x, Backend.to_int32)


def to_int64(x) -> Union[Tensor, PhiTreeNode]:
    """ Converts the `Tensor` or `phi.math.magic.PhiTreeNode` `x` to 64-bit integer. """
    return _backend_op1(x, Backend.to_int64)


def to_complex(x) -> Union[Tensor, PhiTreeNode]:
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


def is_finite(x) -> Union[Tensor, PhiTreeNode]:
    """ Returns a `Tensor` or `phi.math.magic.PhiTreeNode` matching `x` with values `True` where `x` has a finite value and `False` otherwise. """
    return _backend_op1(x, Backend.isfinite)


def is_nan(x) -> Union[Tensor, PhiTreeNode]:
    """ Returns a `Tensor` or `phi.math.magic.PhiTreeNode` matching `x` with values `True` where `x` is `NaN` and `False` otherwise. """
    return _backend_op1(x, Backend.isnan)


def is_inf(x) -> Union[Tensor, PhiTreeNode]:
    """ Returns a `Tensor` or `phi.math.magic.PhiTreeNode` matching `x` with values `True` where `x` is `+inf` or `-inf` and `False` otherwise. """
    return _backend_op1(x, Backend.isnan)


def real(x) -> Union[Tensor, PhiTreeNode]:
    """
    See Also:
        `imag()`, `conjugate()`.

    Args:
        x: `Tensor` or `phi.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Real component of `x`.
    """
    return _backend_op1(x, Backend.real)


def imag(x) -> Union[Tensor, PhiTreeNode]:
    """
    Returns the imaginary part of `x`.
    If `x` does not store complex numbers, returns a zero tensor with the same shape and dtype as this tensor.

    See Also:
        `real()`, `conjugate()`.

    Args:
        x: `Tensor` or `phi.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Imaginary component of `x` if `x` is complex, zeros otherwise.
    """
    return _backend_op1(x, Backend.imag)


def conjugate(x) -> Union[Tensor, PhiTreeNode]:
    """
    See Also:
        `imag()`, `real()`.

    Args:
        x: Real or complex `Tensor` or `phi.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Complex conjugate of `x` if `x` is complex, else `x`.
    """
    return _backend_op1(x, Backend.conj)


def degrees(deg):
    """ Convert degrees to radians. """
    return deg * (3.1415 / 180.)


def sin(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *sin(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sin)


def arcsin(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes the inverse of *sin(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`.
    For real arguments, the result lies in the range [-π/2, π/2].
    """
    return _backend_op1(x, Backend.arcsin)


def cos(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *cos(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.cos)


def arccos(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes the inverse of *cos(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`.
    For real arguments, the result lies in the range [0, π].
    """
    return _backend_op1(x, Backend.cos)


def tan(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *tan(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.tan)


def arctan(x, divide_by=None) -> Union[Tensor, PhiTreeNode]:
    """
    Computes the inverse of *tan(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`.

    Args:
        x: Input. The single-argument `arctan` function cannot output π/2 or -π/2 since tan(π/2) is infinite.
        divide_by: If specified, computes `arctan(x/divide_by)` so that it can return π/2 and -π/2.
            This is equivalent to the common `arctan2` function.
    """
    if divide_by is None:
        return _backend_op1(x, Backend.arctan)
    else:
        divide_by = to_float(divide_by)
        return custom_op2(x, divide_by, arctan, lambda a, b: choose_backend(a, b).arctan2(a, b), 'arctan')


def sinh(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *sinh(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sinh)


def arcsinh(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes the inverse of *sinh(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arcsinh)


def cosh(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *cosh(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.cosh)


def arccosh(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes the inverse of *cosh(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arccosh)


def tanh(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *tanh(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.tanh)


def arctanh(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes the inverse of *tanh(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arctanh)


def log(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes the natural logarithm of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.log)


def log2(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *log(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x` with base 2. """
    return _backend_op1(x, Backend.log2)


def log10(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes *log(x)* of the `Tensor` or `phi.math.magic.PhiTreeNode` `x` with base 10. """
    return _backend_op1(x, Backend.log10)


def sigmoid(x) -> Union[Tensor, PhiTreeNode]:
    """ Computes the sigmoid function of the `Tensor` or `phi.math.magic.PhiTreeNode` `x`. """
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


def safe_div(x: Union[float, Tensor], y: Union[float, Tensor]):
    """ Computes *x/y* with the `Tensor`s `x` and `y` but returns 0 where *y=0*. """
    return custom_op2(x, y,
                      l_operator=safe_div,
                      l_native_function=lambda x_, y_: choose_backend(x_, y_).divide_no_nan(x_, y_),
                      r_operator=lambda y_, x_: safe_div(x_, y_),
                      r_native_function=lambda y_, x_: choose_backend(x_, y_).divide_no_nan(x_, y_),
                      op_name='divide_no_nan')


def maximum(x: Union[Tensor, float], y: Union[Tensor, float]):
    """ Computes the element-wise maximum of `x` and `y`. """
    return custom_op2(x, y, maximum, lambda x_, y_: choose_backend(x_, y_).maximum(x_, y_), op_name='maximum')


def minimum(x: Union[Tensor, float], y: Union[Tensor, float]):
    """ Computes the element-wise minimum of `x` and `y`. """
    return custom_op2(x, y, minimum, lambda x_, y_: choose_backend(x_, y_).minimum(x_, y_), op_name='minimum')


def clip(x: Tensor, lower_limit: Union[float, Tensor], upper_limit: Union[float, Tensor]):
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
        value = pad(value, {dim: (kernel.shape.get_size(dim) // 2, (kernel.shape.get_size(dim) - 1) // 2) for dim in conv_shape.names}, extrapolation)
    native_kernel = reshaped_native(kernel, (batch, out_channels, in_channels, *conv_shape.names), force_expand=in_channels)
    native_value = reshaped_native(value, (batch, in_channels, *conv_shape.names), force_expand=batch)
    backend = choose_backend(native_value, native_kernel)
    native_result = backend.conv(native_value, native_kernel, zero_padding=extrapolation == e_.ZERO)
    result = reshaped_tensor(native_result, (batch, out_channels, *conv_shape))
    return result


def boolean_mask(x: Tensor, dim: DimFilter, mask: Tensor):
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
    dim, original_dim = mask.shape.only(dim), dim  # ToDo
    assert dim, f"mask dimension '{original_dim}' must be present on the mask {mask.shape}"
    assert dim.rank == 1, f"boolean mask only supports 1D selection"
    
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


def gather(values: Tensor, indices: Tensor, dims: Union[DimFilter, None] = None):
    """
    Gathers the entries of `values` at positions described by `indices`.
    All non-channel dimensions of `indices` that are part of `values` but not indexed are treated as batch dimensions.

    See Also:
        `scatter()`.

    Args:
        values: `Tensor` containing values to gather.
        indices: `int` `Tensor`. Multidimensional position references in `values`.
            Must contain a single channel dimension for the index vector matching the number of dimensons to index.
            This channel dimension should list the dimension names to index as item names unless explicitly specified as `dims`.
        dims: (Optional) Dimensions indexed by `indices`.
            Alternatively, the dimensions can be specified as the item names of the channel dimension of `indices`.
            If `None` and no index item names are specified, will default to all spatial dimensions or all instance dimensions, depending on which ones are present (but not both).

    Returns:
        `Tensor` with combined batch dimensions, channel dimensions of `values` and spatial/instance dimensions of `indices`.
    """
    assert channel(indices).rank < 2, f"indices can at most have one channel dimension but got {indices.shape}"
    if dims is None:
        if channel(indices) and channel(indices).item_names[0]:
            dims = channel(indices).item_names[0]
        else:  # Fallback to spatial / instance
            warnings.warn(f"Indexing without item names is not recommended. Got indices {indices.shape}", SyntaxWarning, stacklevel=2)
            assert values.shape.instance.is_empty or values.shape.spatial.is_empty, f"Specify gather dimensions for values with both instance and spatial dimensions. Got {values.shape}"
            dims = values.shape.instance if values.shape.spatial.is_empty else values.shape.spatial
    if indices.dtype.kind == bool:
        indices = to_int32(indices)
    dims = parse_dim_order(dims)
    assert dims in values.shape, f"Trying to index non-existant dimensions with indices {indices.shape} into values {values.shape}"
    treat_as_batch = non_channel(indices).only(values.shape).without(dims)
    batch_ = (values.shape.batch & indices.shape.batch).without(dims) & treat_as_batch
    channel_ = values.shape.without(dims).without(batch_)
    index_list_dims = indices.shape.non_channel.without(batch_)
    squeeze_index_list = False
    if not index_list_dims:
        index_list_dims = instance('_single_index')
        squeeze_index_list = True
    native_values = reshaped_native(values, [batch_, *dims, channel_])
    native_indices = reshaped_native(indices, [batch_, *index_list_dims, channel(indices)])
    backend = choose_backend(native_values, native_indices)
    native_result = backend.batched_gather_nd(native_values, native_indices)
    result = reshaped_tensor(native_result, [batch_, *index_list_dims, channel_], convert=False)
    if squeeze_index_list:
        result = result[{'_single_index': 0}]
    return result


def scatter(base_grid: Union[Tensor, Shape],
            indices: Union[Tensor, dict],
            values: Union[Tensor, float],
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
    if isinstance(indices, dict):  # update a slice
        if len(indices) == 1 and isinstance(next(iter(indices.values())), (str, int, slice)):  # update a range
            dim, sel = next(iter(indices.items()))
            full_dim = base_grid.shape[dim]
            if isinstance(sel, str):
                sel = full_dim.item_names[0].index(sel)
            if isinstance(sel, int):
                sel = slice(sel, sel+1)
            assert isinstance(sel, slice), f"Selection must be a str, int or slice but got {type(sel)}"
            values = expand(values, full_dim.after_gather({dim: sel}))
            parts = [
                base_grid[{dim: slice(sel.start)}],
                values,
                base_grid[{dim: slice(sel.stop, None)}]
            ]
            return concat(parts, dim)
        else:
            raise NotImplementedError("scattering into non-continuous values not yet supported by dimension")
    grid_shape = base_grid if isinstance(base_grid, Shape) else base_grid.shape
    assert channel(indices).rank < 2
    if channel(indices) and channel(indices).item_names[0]:
        indexed_dims = channel(indices).item_names[0]
        assert indexed_dims in grid_shape, f"Scatter indices {indices.shape} point to missing dimensions in grid {grid_shape}"
        if indexed_dims != grid_shape.only(indexed_dims).names:
            indices = indices.vector[grid_shape.only(indexed_dims).names]
        indexed_dims = grid_shape.only(indexed_dims)
    else:
        assert channel(indices).rank == 1 or (grid_shape.spatial_rank + grid_shape.instance_rank == 1 and indices.shape.channel_rank == 0)
        indexed_dims = grid_shape.spatial or grid_shape.instance
        assert channel(indices).volume == indexed_dims.rank
    values = wrap(values)
    batches = values.shape.non_channel.non_instance & indices.shape.non_channel.non_instance
    channels = grid_shape.without(indexed_dims).without(batches) & values.shape.channel
    # --- Set up grid ---
    if isinstance(base_grid, Shape):
        with choose_backend_t(indices, values):
            base_grid = zeros(base_grid & batches & values.shape.channel, dtype=values.dtype)
        if mode != 'add':
            base_grid += math.nan
    # --- Handle outside indices ---
    if outside_handling == 'clamp':
        indices = clip(indices, 0, tensor(indexed_dims, channel('vector')) - 1)
    elif outside_handling == 'discard':
        indices_linear = pack_dims(indices, instance, instance(_scatter_instance=1))
        indices_inside = min_((round_(indices_linear) >= 0) & (round_(indices_linear) < tensor(indexed_dims, channel('vector'))), 'vector')
        indices_linear = boolean_mask(indices_linear, '_scatter_instance', indices_inside)
        if instance(values).rank > 0:
            values_linear = pack_dims(values, instance, instance(_scatter_instance=1))
            values_linear = boolean_mask(values_linear, '_scatter_instance', indices_inside)
            values = unpack_dim(values_linear, '_scatter_instance', instance(values))
        indices = unpack_dim(indices_linear, '_scatter_instance', instance(indices))
        if indices.shape.is_non_uniform:
            raise NotImplementedError()
    lists = indices.shape.instance & values.shape.instance

    def scatter_forward(base_grid, indices, values):
        indices = to_int32(round_(indices))
        native_grid = reshaped_native(base_grid, [batches, *indexed_dims, channels])
        native_values = reshaped_native(values, [batches, lists, channels])
        native_indices = reshaped_native(indices, [batches, lists, 'vector'])
        backend = choose_backend(native_indices, native_values, native_grid)
        if mode in ('add', 'update'):
            native_result = backend.scatter(native_grid, native_indices, native_values, mode=mode)
        else:  # mean
            zero_grid = backend.zeros_like(native_grid)
            summed = backend.scatter(zero_grid, native_indices, native_values, mode='add')
            count = backend.scatter(zero_grid, native_indices, backend.ones_like(native_values), mode='add')
            native_result = summed / backend.maximum(count, 1)
            native_result = backend.where(count == 0, native_grid, native_result)
        return reshaped_tensor(native_result, [batches, *indexed_dims, channels], check_sizes=True)

    def scatter_backward(args: dict, _output, d_output):
        from ._nd import spatial_gradient
        values_grad = gather(d_output, args['indices'])
        spatial_gradient_indices = gather(spatial_gradient(d_output, dims=indexed_dims), args['indices'])
        indices_grad = mean(spatial_gradient_indices * args['values'], 'vector_')
        return None, indices_grad, values_grad

    from ._functional import custom_gradient
    scatter_function = custom_gradient(scatter_forward, scatter_backward) if indices_gradient else scatter_forward
    result = scatter_function(base_grid, indices, values)
    return result


def histogram(values: Tensor, bins: Shape or Tensor = spatial(bins=30), weights=1, same_bins: DimFilter = None):
    """
    Compute a histogram of a distribution of values.

    *Important Note:* In its current implementation, values outside the range of bins may or may not be added to the outermost bins.

    Args:
        values: `Tensor` listing the values to be binned along spatial or instance dimensions.
            `values´ may not contain channel or dual dimensions.
        bins: Either `Shape` specifying the number of equally-spaced bins to use or bin edge positions as `Tensor` with a spatial or instance dimension.
        weights: `Tensor` assigning a weight to every value in `values` that will be added to the bin, default 1.
        same_bins: Only used if `bins` is given as a `Shape`.
            Use the same bin sizes and positions across these batch dimensions.
            By default, bins will be chosen independently for each example.

    Returns:
        hist: `Tensor` containing all batch dimensions and the `bins` dimension with dtype matching `weights`.
        bin_edges: `Tensor`
        bin_center: `Tensor`
    """
    assert isinstance(values, Tensor), f"values must be a Tensor but got {type(values)}"
    assert channel(values).is_empty, f"Only 1D histograms supported but values have a channel dimension: {values.shape}"
    assert dual(values).is_empty, f"values cannot contain dual dimensions but got shape {values.shape}"
    weights = wrap(weights)
    if isinstance(bins, Shape):
        def equal_bins(v):
            return linspace(finite_min(v, shape), finite_max(v, shape), bins.with_size(bins.size + 1))
        bins = broadcast_op(equal_bins, [values], iter_dims=(batch(values) & batch(weights)).without(same_bins))
    assert isinstance(bins, Tensor), f"bins must be a Tensor but got {type(bins)}"
    assert non_batch(bins).rank == 1, f"bins must contain exactly one spatial or instance dimension listing the bin edges but got shape {bins.shape}"
    assert channel(bins).rank == dual(bins).rank == 0, f"bins cannot have any channel or dual dimensions but got shape {bins.shape}"
    tensors = [values, bins] if weights is None else [values, weights, bins]
    backend = choose_backend_t(*tensors)

    def histogram_uniform(values: Tensor, bin_edges: Tensor, weights):
        batch_dims = batch(values) & batch(bin_edges) & batch(weights)
        value_dims = non_batch(values) & non_batch(weights)
        values_native = reshaped_native(values, [batch_dims, value_dims])
        weights_native = reshaped_native(weights, [batch_dims, value_dims])
        bin_edges_native = reshaped_native(bin_edges, [batch_dims, non_batch(bin_edges)])
        hist_native = backend.histogram1d(values_native, weights_native, bin_edges_native)
        hist = reshaped_tensor(hist_native, [batch_dims, non_batch(bin_edges).with_size(non_batch(bin_edges).size - 1)])
        return hist
        # return stack_tensors([bin_edges, hist], channel(vector=[bin_edges.shape.name, 'hist']))

    bin_center = (bins[{non_batch(bins).name: slice(1, None)}] + bins[{non_batch(bins).name: slice(0, -1)}]) / 2
    bin_center = expand(bin_center, channel(vector=non_batch(bins).names))
    bin_edges = stack_tensors([bins], channel(values)) if channel(values) else bins
    return broadcast_op(histogram_uniform, [values, bins, weights]), bin_edges, bin_center


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
    elif isinstance(tensor1, CompressedSparseMatrix):
        if isinstance(tensor2, CompressedSparseMatrix):
            _assert_close(tensor1._values, tensor2._values, rel_tolerance, abs_tolerance, msg, verbose)
            _assert_close(tensor1._indices, tensor2._indices, 0, 0, msg, verbose)
            _assert_close(tensor1._pointers, tensor2._pointers, 0, 0, msg, verbose)
        elif tensor1._compressed_dims.only(tensor2.shape):
            _assert_close(dense(tensor1), tensor2, rel_tolerance, abs_tolerance, msg, verbose)
        else:
            _assert_close(tensor1._values, tensor2._values, rel_tolerance, abs_tolerance, msg, verbose)
    elif isinstance(tensor2, CompressedSparseMatrix):
        return _assert_close(tensor2, tensor1, rel_tolerance, abs_tolerance, msg, verbose)
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
        x: `Tensor` or `phi.math.magic.PhiTreeNode` for which gradients should be disabled.

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


def pairwise_distances(positions: Tensor,
                       max_distance: Union[float, Tensor] = None,
                       format: str = 'dense',
                       default: Optional[float] = None,
                       method: str = 'sparse') -> Tensor:
    """
    Computes the distance matrix containing the pairwise position differences between each pair of points.
    Points that are further apart than `max_distance` (if specified) are assigned a distance value of `0`.
    The diagonal of the matrix (self-distance) also consists purely of zero-vectors and may or may not be stored explicitly.

    Args:
        positions: `Tensor`.
            Channel dimensions are interpreted as position components.
            Instance and spatial dimensions list nodes.
        max_distance: Scalar or `Tensor` specifying a max_radius for each point separately.
            Can contain additional batch dimensions but spatial/instance dimensions must match `positions` if present.
            If not specified, uses an infinite cutoff radius, i.e. all points will be considered neighbors.
        format: Matrix format as `str` or concrete sparsity pattern as `Tensor`.
            Allowed strings are `'dense', `'csr'`, `'coo'`, `'csc'`.
            When a `Tensor` is passed, it needs to have all instance and spatial dims as `positions` as well as corresponding dual dimensions.
            The distances will be evaluated at all stored entries of the `format` tensor.
        default: Value the sparse tensor returns for non-stored values. Must be `0` or `None`.

    Returns:
        Distance matrix as sparse or dense `Tensor`, depending on `format`.
        For each spatial/instance dimension in `positions`, the matrix also contains a dual dimension of the same name and size.
        The matrix also contains all batch dimensions of `positions` and one channel dimension called `vector`.

    Examples:
        >>> pos = vec(x=0, y=tensor([0, 1, 2.5], instance('particles')))
        >>> dx = pairwise_distances(pos, format='dense', max_distance=2)
        >>> dx.particles[0]
        (x=0.000, y=0.000); (x=0.000, y=1.000); (x=0.000, y=0.000) (~particlesᵈ=3, vectorᶜ=x,y)
    """
    assert isinstance(positions, Tensor), f"positions must be a Tensor but got {type(positions)}"
    assert default in [0, None], f"default value must be either 0 or None but got '{default}'"
    primal_dims = positions.shape.non_batch.non_channel.non_dual
    dual_dims = dual(**primal_dims.untyped_dict)
    if isinstance(format, Tensor):  # sparse connectivity specified, no neighborhood search required
        assert max_distance is None, "max_distance not allowed when connectivity is specified (passing a Tensor for format)"
        return map_pairs(lambda p1, p2: p2 - p1, positions, format)
    # --- Dense ---
    elif format == 'dense':
        dx = unpack_dim(pack_dims(positions, non_batch(positions).non_channel.non_dual, instance('_tmp')), '_tmp', dual_dims) - positions
        if max_distance is not None:
            neighbors = sum_(dx ** 2, channel) <= max_distance ** 2
            default = float('nan') if default is None else default
            dx = where(neighbors, dx, default)
        return dx
    # --- Sparse neighbor search from here on ---
    assert max_distance is not None, "max_distance must be specified when computing distance in sparse format"
    max_distance = wrap(max_distance)
    index_dtype = DType(int, 32)
    backend = choose_backend_t(positions, max_distance)
    batch_shape = batch(positions) & batch(max_distance)
    if not dual_dims.well_defined:
        assert dual_dims.rank == 1, f"others_dims sizes must be specified when passing more then one dimension but got {dual_dims}"
        dual_dims = dual_dims.with_size(primal_dims.volume)
    # --- Determine mode ---
    tmp_pair_count = None
    pair_count = None
    table_len = None
    mode = 'vectorize' if batch_shape.volume > 1 and batch_shape.is_uniform else 'loop'
    if backend.is_available(positions):
        if mode == 'vectorize':
            # ToDo determine limits from positions? build_cells+bincount would be enough
            pair_count = 7
    else:  # tracing
        if backend.requires_fixed_shapes_when_tracing():
            # ToDo use fixed limits (set by user)
            pair_count = 7
            mode = 'vectorize'
    # --- Run neighborhood search ---
    from .backend._partition import find_neighbors, find_neighbors_matscipy, find_neighbors_sklearn
    if mode == 'loop':
        indices = []
        values = []
        for b in batch_shape.meshgrid():
            native_positions = reshaped_native(positions[b], [primal_dims, channel(positions)])
            native_max_dist = max_distance[b].native()
            if method == 'sparse':
                nat_rows, nat_cols, nat_vals = find_neighbors(native_positions, native_max_dist, None, periodic=False, default=default)
            elif method == 'matscipy':
                assert positions.available, f"Cannot jit-compile matscipy neighborhood search"
                nat_rows, nat_cols, nat_vals = find_neighbors_matscipy(native_positions, native_max_dist, None, periodic=False)
            elif method == 'sklearn':
                assert positions.available, f"Cannot jit-compile matscipy neighborhood search"
                nat_rows, nat_cols, nat_vals = find_neighbors_sklearn(native_positions, native_max_dist)
            else:
                raise ValueError(method)
            nat_indices = backend.stack([nat_rows, nat_cols], -1)
            indices.append(reshaped_tensor(nat_indices, [instance('pairs'), channel(vector=primal_dims.names + dual_dims.names)], convert=False))
            values.append(reshaped_tensor(nat_vals, [instance('pairs'), channel(positions)]))
        indices = stack(indices, batch_shape)
        values = stack(values, batch_shape)
    elif mode == 'vectorize':
        raise NotImplementedError
        # native_positions = reshaped_native(positions, [batch_shape, primal_dims, channel(positions)])
        # native_max_dist = reshaped_native(max_distance, [batch_shape, primal_dims], force_expand=False)
        # def single_search(pos, r):
        #     return find_neighbors(pos, r, None, periodic=False, pair_count=pair_count, default=default)
        # nat_rows, nat_cols, nat_vals = backend.vectorized_call(single_search, native_positions, native_max_dist, output_dtypes=(index_dtype, index_dtype, positions.dtype))
        # nat_indices = backend.stack([nat_rows, nat_cols], -1)
        # indices = reshaped_tensor(nat_indices, [batch_shape, instance('pairs'), channel(vector=primal_dims.names + dual_dims.names)], convert=False)
        # values = reshaped_tensor(nat_vals, [batch_shape, instance('pairs'), channel(positions)])
    else:
        raise RuntimeError
    # --- Assemble sparse matrix ---
    dense_shape = primal_dims & dual_dims
    coo = SparseCoordinateTensor(indices, values, dense_shape, can_contain_double_entries=False, indices_sorted=False, default=default)
    return to_format(coo, format)


def map_pairs(map_function: Callable, values: Tensor, connections: Tensor):
    """
    Evaluates `map_function` on all pairs of elements present in the sparsity pattern of `connections`.

    Args:
        map_function: Function with signature `(Tensor, Tensor) -> Tensor`.
        values: Values to evaluate `map_function` on.
            Needs to have a spatial or instance dimension but must not have a dual dimension.
        connections: Sparse tensor.

    Returns:
        `Tensor` with the sparse dimensions of `connections` and all non-instance dimensions returned by `map_function`.
    """
    assert dual(values).is_empty, f"values must not have a dual dimension but got {values.shape}"
    inst_dim = non_batch(values).non_channel.non_dual.name
    indices = stored_indices(connections, invalid='clamp')
    origin = values[{inst_dim: indices[inst_dim]}]
    target = values[{inst_dim: indices['~' + inst_dim]}]
    result = map_function(origin, target)
    return tensor_like(connections, result, value_order='as existing')

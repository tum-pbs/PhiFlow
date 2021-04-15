import functools
import math
import re
import time
import warnings
from contextlib import contextmanager
from numbers import Number
from typing import Tuple, Callable, Any

import numpy as np

from phi import math as _math
from .backend import default_backend, choose_backend, Backend, get_precision, convert as b_convert, \
    BACKENDS
from .backend._backend_helper import combined_dim
from .backend._dtype import DType, combine_types
from ._shape import BATCH_DIM, CHANNEL_DIM, SPATIAL_DIM, Shape, EMPTY_SHAPE, spatial_shape, shape as shape_, \
    _infer_dim_type_from_name, combine_safe, parse_dim_order, batch_shape, channel_shape
from ._tensors import Tensor, wrap, tensor, broadcastable_native_tensors, NativeTensor, TensorStack, CollapsedTensor, custom_op2, tensors, TensorDim
from . import extrapolation as extrapolation_
from .backend._optim import SolveResult, Solve, ConvergenceException, NotConverged, Diverged
from .backend._profile import get_current_profile


def choose_backend_t(*values, prefer_default=False) -> Backend:
    """ Choose backend for given `Tensor` or native tensor values. """
    natives = sum([v._natives() if isinstance(v, Tensor) else (v,) for v in values], ())
    return choose_backend(*natives, prefer_default=prefer_default)


def convert(value: Tensor, backend: Backend = None, use_dlpack=True):
    """
    Convert the native representation of a `Tensor` to the native format of `backend`.

    *Warning*: This operation breaks the automatic differentiation chain.

    See Also:
        `phi.math.backend.convert()`.

    Args:
        value: `Tensor` to convert.
        backend: Target backend. If `None`, uses the current default backend, see `phi.math.backend.default_backend()`.

    Returns:
        `Tensor` with native representation belonging to `backend`.
    """
    return value._op1(lambda native: b_convert(native, backend, use_dlpack=use_dlpack))


def all_available(*values: Tensor):
    """
    Tests if the values of all given tensors are known and can be read at this point.
    
    Tensors are typically available when the backend operates in eager mode.

    Args:
      values: tensors to check
      *values: Tensor: 

    Returns:
      bool

    """
    for value in values:
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
        choose_backend(value, raise_error=True)
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
                    force_expand: Any = False):
    """
    Returns a native representation of `value` where dimensions are laid out according to `groups`.

    See Also:
        `native()`, `join_dimensions()`, `reshaped_tensor()`.

    Args:
        value: `Tensor`
        groups: Sequence of dimension names as `str` or groups of dimensions to be joined as `Shape`.
        force_expand: `bool` or sequence of dimensions.
            If `True`, repeats the tensor along missing dimensions.
            If `False`, puts singleton dimensions where possible.
            If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.

    Returns:
        Native tensor with dimensions matching `groups`.
    """
    assert isinstance(value, Tensor), f"value must be a Tensor but got {type(value)}"
    order = []
    for i, group in enumerate(groups):
        if isinstance(group, Shape):
            present = value.shape.only(group)
            if force_expand is True or present.volume > 1 or (force_expand is not False and group.only(force_expand).volume > 1):
                value = _expand_dims(value, group)
            value = join_dimensions(value, group, f"group{i}")
            order.append(f"group{i}")
        else:
            assert isinstance(group, str), f"Groups must be either str or Shape but got {group}"
            order.append(group)
    return value.native(order=order)


def reshaped_tensor(value: Any,
                    groups: tuple or list,
                    check_sizes=False,
                    convert=True):
    """
    Creates a `Tensor` from a native tensor or tensor-like whereby the dimensions of `value` are split according to `groups`.

    See Also:
        `phi.math.tensor()`, `reshaped_native()`, `split_dimension()`.

    Args:
        value: Native tensor or tensor-like.
        groups: Sequence of dimension names as `str` or groups of dimensions to be joined as `Shape`.
        check_sizes: If True, group sizes must match the sizes of `value` exactly. Otherwise, allows singleton dimensions.
        convert: If True, converts the data to the native format of the current default backend.
            If False, wraps the data in a `Tensor` but keeps the given data reference if possible.

    Returns:
        `Tensor` with all dimensions from `groups`
    """
    names = [group if isinstance(group, str) else f'group{i}' for i, group in enumerate(groups)]
    value = tensor(value, names, convert=convert)
    for i, group in enumerate(groups):
        if isinstance(group, Shape):
            if value.shape.get_size(f'group{i}') == group.volume:
                value = split_dimension(value, f'group{i}', group)
            else:
                if check_sizes:
                    raise AssertionError()
                value = value.dimension(f'group{i}')[0]  # remove group dim
                value = _expand_dims(value, group)
    return value


def copy(value: Tensor):
    """
    Copies the data buffer and encapsulating `Tensor` object.

    Args:
        value: `Tensor` to be copied.

    Returns:
        Copy of `value`.
    """
    if value._is_special:
        warnings.warn("Tracing tensors cannot be copied.")
        return value
    return value._op1(lambda native: choose_backend(native).copy(native))


def native_call(f: Callable, *inputs: Tensor, channels_last=None, channel_dim='vector'):
    """
    Calls `f` with the native representations of the `inputs` tensors in standard layout and returns the result as a `Tensor`.

    All inputs are converted to native tensors depending on `channels_last`:

    * `channels_last=True`: Dimension layout `(total_batch_size, spatial_dims..., total_channel_size)`
    * `channels_last=False`: Dimension layout `(total_batch_size, total_channel_size, spatial_dims...)`

    All batch dimensions are compressed into a single dimension with `total_batch_size = input.shape.batch.volume`.
    The same is done for all channel dimensions.

    Additionally, missing batch and spatial dimensions are added so that all `inputs` have the same batch and spatial shape.

    Args:
        f: Function to be called on native tensors of `inputs`.
            The function output must have the same dimension layout as the inputs and the batch size must be identical.
        *inputs: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
            If `None`, the channels are put in the default position associated with the current backend,
            see `phi.math.backend.Backend.prefers_channels_last()`.
        channel_dim: Name of the channel dimension of the result.

    Returns:
        `Tensor` with batch and spatial dimensions of `inputs` and single channel dimension `channel_dim`.
    """
    if channels_last is None:
        backend = choose_backend_t(*inputs, prefer_default=True)
        channels_last = backend.prefers_channels_last()
    batch = combine_safe(*[i.shape.batch for i in inputs])
    spatial = combine_safe(*[i.shape.spatial for i in inputs])
    natives = []
    for i in inputs:
        groups = (batch, *i.shape.spatial.names, i.shape.channel) if channels_last else (batch, i.shape.channel, *i.shape.spatial.names)
        natives.append(reshaped_native(i, groups))
    output = f(*natives)
    if isinstance(output, (tuple, list)):
        raise NotImplementedError()
    else:
        groups = (batch, *spatial.names, channel_dim) if channels_last else (batch, channel_dim, *spatial.names)
        return reshaped_tensor(output, groups)


def print_(value: Tensor = None, name: str = ""):
    """
    Print a tensor with no more than two spatial dimensions, slicing it along all batch and channel dimensions.
    
    Unlike NumPy's array printing, the dimensions are sorted.
    Elements along the alphabetically first dimension is printed to the right, the second dimension upward.
    Typically, this means x right, y up.

    Args:
        name: name of the tensor
        value: tensor-like
        value: Tensor:  (Default value = None)
        name: str:  (Default value = None)

    Returns:

    """
    if value is None:
        print()
        return
    if name:
        print(" " * 16 + name)
    value = wrap(value)
    dim_order = tuple(sorted(value.shape.spatial.names, reverse=True))
    if value.shape.spatial_rank == 0:
        print(value.numpy())
    elif value.shape.spatial_rank == 1:
        for index_dict in value.shape.non_spatial.meshgrid():
            if value.shape.non_spatial.volume > 1:
                print(f"--- {', '.join('%s=%d' % (name, idx) for name, idx in index_dict.items())} ---")
            text = np.array2string(value[index_dict].numpy(dim_order), precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', text))
    elif value.shape.spatial_rank == 2:
        for index_dict in value.shape.non_spatial.meshgrid():
            if value.shape.non_spatial.volume > 1:
                print(f"--- {', '.join('%s=%d' % (name, idx) for name, idx in index_dict.items())} ---")
            text = np.array2string(value[index_dict].numpy(dim_order)[::-1], precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', re.sub('\],', '', text)))
    else:
        raise NotImplementedError('Can only print tensors with up to 2 spatial dimensions.')


def print_gradient(value: Tensor, name="", detailed=False) -> Tensor:
    """
    Prints the gradient vector of `value` when computed.
    The gradient at `value` is the vector-Jacobian product of all operations between the output of this function and the loss value.

    The gradient is not printed in jit mode, see `jit_compile()`.

    Example:
        ```python
        def f(x):
            x = math.print_gradient(x, 'dx')
            return math.l1_loss(x)

        math.functional_gradient(f)(math.ones(x=6))
        ```

    Args:
        value: `Tensor` for which the gradient may be computed later.
        name: (Optional) Name to print along with the gradient values
        detailed: If `False`, prints a short summary of the gradient tensor.

    Returns:
        `identity(value)` which when differentiated, prints the gradient vector.
    """
    def print_grad(_x, _y, dx):
        if all_available(_x, dx):
            if detailed:
                print_(dx, name=name)
            else:
                print(f"{name}:  \t{dx}")
        return dx,
    identity = custom_gradient(lambda x: x, print_grad)
    return identity(value)


def map_(function, *values: Tensor) -> Tensor:
    """
    Calls `function` on all elements of `value`.

    Args:
        function: Function to be called on single elements contained in `value`. Must return a value that can be stored in tensors.
        values: Tensors to iterate over. Number of tensors must match `function` signature.

    Returns:
        `Tensor` of same shape as `value`.
    """
    shape = combine_safe(*[v.shape for v in values])
    values_reshaped = [CollapsedTensor(v, shape) for v in values]
    flat = [flatten(v) for v in values_reshaped]
    result = []
    for items in zip(*flat):
        result.append(function(*items))
    if None in result:
        assert all(r is None for r in result), f"map function returned None for some elements, {result}"
        return
    return wrap(result).vector.split(shape)


def _initialize(uniform_initializer, shape=EMPTY_SHAPE, dtype=None, **dimensions):
    shape &= shape_(**dimensions)
    if shape.is_non_uniform:
        stack_dim = shape.shape.without('dims')[0:1]
        shapes = shape.unstack(stack_dim.name)
        tensors = [_initialize(uniform_initializer, s, dtype) for s in shapes]
        return _stack(tensors, stack_dim.name, stack_dim.types[0])
    else:
        return uniform_initializer(shape, dtype)


def zeros(shape=EMPTY_SHAPE, dtype=None, **dimensions):
    """
    Define a tensor with specified shape with value 0 / False everywhere.
    
    This method may not immediately allocate the memory to store the values.

    Args:
      shape: base tensor shape (Default value = EMPTY_SHAPE)
      dtype: data type (Default value = None)
      dimensions: additional dimensions, types are determined from names
      **dimensions: 

    Returns:
      tensor of specified shape

    """
    return _initialize(lambda shape, dtype: CollapsedTensor(NativeTensor(default_backend().zeros((), dtype=dtype), EMPTY_SHAPE), shape), shape, dtype, **dimensions)


def zeros_like(tensor: Tensor):
    return zeros(tensor.shape, dtype=tensor.dtype)


def ones(shape=EMPTY_SHAPE, dtype=None, **dimensions):
    """
    Define a tensor with specified shape with value 1 / True everywhere.
    
    This method may not immediately allocate the memory to store the values.

    Args:
      shape: base tensor shape (Default value = EMPTY_SHAPE)
      dtype: data type (Default value = None)
      dimensions: additional dimensions, types are determined from names
      **dimensions: 

    Returns:
      tensor of specified shape

    """
    return _initialize(lambda shape, dtype: CollapsedTensor(NativeTensor(default_backend().ones((), dtype=dtype), EMPTY_SHAPE), shape), shape, dtype, **dimensions)


def ones_like(tensor: Tensor):
    return zeros(tensor.shape, dtype=tensor.dtype) + 1


def random_normal(shape=EMPTY_SHAPE, dtype=None, **dimensions):

    def uniform_random_normal(shape, dtype):
        native = choose_backend(*shape.sizes, prefer_default=True).random_normal(shape.sizes)
        native = native if dtype is None else native.astype(dtype)
        return NativeTensor(native, shape)

    return _initialize(uniform_random_normal, shape, dtype, **dimensions)


def random_uniform(shape=EMPTY_SHAPE, dtype=None, **dimensions):

    def uniform_random_uniform(shape, dtype):
        native = choose_backend(*shape.sizes, prefer_default=True).random_uniform(shape.sizes)
        native = native if dtype is None else native.astype(dtype)
        return NativeTensor(native, shape)

    return _initialize(uniform_random_uniform, shape, dtype, **dimensions)


def transpose(value, axes):
    if isinstance(value, Tensor):
        return CollapsedTensor(value, value.shape[axes])
    else:
        return choose_backend(value).transpose(value, axes)


def fftfreq(resolution: Shape, dtype: DType = None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    Args:
      resolution: Grid resolution measured in cells
      dtype: Data type of the returned tensor (Default value = None)

    Returns:
      tensor holding the frequencies of the corresponding values computed by math.fft

    """
    resolution = spatial_shape(resolution)
    k = meshgrid(**{dim: np.fft.fftfreq(int(n)) for dim, n in resolution.named_sizes})
    return to_float(k) if dtype is None else cast(k, dtype)


def meshgrid(**dimensions):
    """
    generate a TensorStack meshgrid from keyword dimensions

    Args:
      **dimensions: 

    Returns:

    """
    assert 'vector' not in dimensions
    dim_values = {}
    for dim, spec in dimensions.items():
        if isinstance(spec, int):
            dim_values[dim] = tuple(range(spec))
        elif isinstance(spec, Tensor):
            dim_values[dim] = spec.native()
        else:
            dim_values[dim] = spec
    backend = choose_backend(*dim_values.values(), prefer_default=True)
    indices_list = backend.meshgrid(*dim_values.values())
    single_shape = Shape([len(val) for val in dim_values.values()], dim_values.keys(), [SPATIAL_DIM] * len(dim_values))
    channels = [NativeTensor(t, single_shape) for t in indices_list]
    return TensorStack(channels, 'vector', CHANNEL_DIM)


def linspace(start, stop, number: int, dim='linspace'):
    native = choose_backend(start, stop, number, prefer_default=True).linspace(start, stop, number)
    return NativeTensor(native, shape_(**{dim: number}))


def arange(start_or_stop: int, stop: int or None = None, step=1, dim='range'):
    if stop is None:
        start, stop = 0, start_or_stop
    else:
        start = start_or_stop
    native = choose_backend(start, stop, prefer_default=True).range(start, stop, step, DType(int, 32))
    return NativeTensor(native, shape_(**{dim: stop - start}))


def range_tensor(shape: Shape, **dims):
    shape &= shape_(**dims)
    data = arange(0, shape.volume)
    result = split_dimension(data, 'range', shape)
    return result


def channel_stack(values, dim: str):
    return _stack(values, dim, CHANNEL_DIM)


def batch_stack(values, dim: str = 'batch'):
    return _stack(values, dim, BATCH_DIM)


def spatial_stack(values, dim: str):
    return _stack(values, dim, SPATIAL_DIM)


def _stack(values: tuple or list,
           dim: str,
           dim_type: str):
    values = cast_same(*values)

    def inner_stack(*values):
        return TensorStack(values, dim, dim_type)

    result = broadcast_op(inner_stack, values)
    return result


def concat(values: tuple or list, dim: str) -> Tensor:
    """
    Concatenates a sequence of tensors along one dimension.
    The shapes of all values must be equal, except for the size of the concat dimension.

    Args:
      values: Tensors to concatenate
      dim: concat dimension, must be present in all values
      values: tuple or list: 
      dim: str: 

    Returns:
      concatenated tensor

    """
    assert len(values) > 0, "concat() got empty sequence"
    broadcast_shape = values[0].shape
    natives = [v.native(order=broadcast_shape.names) for v in values]
    backend = choose_backend(*natives)
    concatenated = backend.concat(natives, broadcast_shape.index(dim))
    return NativeTensor(concatenated, broadcast_shape.with_sizes(backend.staticshape(concatenated)))


def pad(value: Tensor, widths: dict, mode: 'extrapolation_.Extrapolation') -> Tensor:
    """
    Pads a tensor along the specified dimensions, determining the added values using the given extrapolation_.
    
    This is equivalent to calling `mode.pad(value, widths)`.

    Args:
      value: tensor to be padded
      widths: name: str -> (lower: int, upper: int)
      mode: Extrapolation object
      value: Tensor: 
      widths: dict: 
      mode: 'extrapolation_.Extrapolation': 

    Returns:
      padded Tensor

    """
    return mode.pad(value, widths)


def closest_grid_values(grid: Tensor,
                        coordinates: Tensor,
                        extrap: 'extrapolation_.Extrapolation',
                        stack_dim_prefix='closest_'):
    """
    Finds the neighboring grid points in all spatial directions and returns their values.
    The result will have 2^d values for each vector in coordiantes in d dimensions.

    Args:
      grid: grid data. The grid is spanned by the spatial dimensions of the tensor
      coordinates: tensor with 1 channel dimension holding vectors pointing to locations in grid index space
      extrap: grid extrapolation
      stack_dim_prefix: For each spatial dimension `dim`, stacks lower and upper closest values along dimension `stack_dim_prefix+dim`.

    Returns:
      Tensor of shape (batch, coord_spatial, grid_spatial=(2, 2,...), grid_channel)

    """
    return broadcast_op(functools.partial(_closest_grid_values, extrap=extrap, stack_dim_prefix=stack_dim_prefix), [grid, coordinates])


def _closest_grid_values(grid: Tensor,
                         coordinates: Tensor,
                         extrap: 'extrapolation_.Extrapolation',
                         stack_dim_prefix='closest_'):
    # alternative method: pad array for all 2^d combinations, then stack to simplify gather.
    # --- Pad tensor where transform is not possible ---
    non_copy_pad = {dim: (0 if extrap[dim, 0].is_copy_pad else 1, 0 if extrap[dim, 1].is_copy_pad else 1)
                    for dim in grid.shape.spatial.names}
    grid = extrap.pad(grid, non_copy_pad)
    coordinates += wrap([not extrap[dim, 0].is_copy_pad for dim in grid.shape.spatial.names], 'vector')
    # --- Transform coordiantes ---
    min_coords = to_int(floor(coordinates))
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
        return spatial_stack([values_left, values_right], f"{stack_dim_prefix}{grid.shape.spatial.names[ax_idx]}")

    result = left_right(np.array([False] * grid.shape.spatial_rank), 0)
    return result


def grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation_.Extrapolation'):
    result = broadcast_op(functools.partial(_grid_sample, extrap=extrap), [grid, coordinates])
    return result


def _grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation_.Extrapolation' or None):
    if grid.shape.batch == coordinates.shape.batch or grid.shape.batch.volume == 1 or coordinates.shape.batch.volume == 1:
        # reshape batch dimensions, delegate to backend.grid_sample()
        grid_batched = join_dimensions(join_dimensions(grid, grid.shape.batch, 'batch'), grid.shape.channel, 'vector')
        coordinates_batched = join_dimensions(coordinates, coordinates.shape.batch, 'batch')
        backend = choose_backend(*grid._natives())
        result = NotImplemented
        if extrap is None:
            result = backend.grid_sample(grid_batched.native(),
                                         grid.shape.index(grid.shape.spatial),
                                         coordinates_batched.native(),
                                         'undefined')
        elif extrap.native_grid_sample_mode:
            result = backend.grid_sample(grid_batched.native(),
                                         grid.shape.index(grid.shape.spatial),
                                         coordinates_batched.native(),
                                         extrap.native_grid_sample_mode)
        if result is NotImplemented:
            # pad one layer
            grid_batched = pad(grid_batched, {dim: (1, 1) for dim in grid.shape.spatial.names}, extrap or extrapolation_.ZERO)
            if extrap is not None:
                from .extrapolation import _CopyExtrapolation
                if isinstance(extrap, _CopyExtrapolation):
                    inner_coordinates = extrap.transform_coordinates(coordinates_batched, grid.shape) + 1
                else:
                    inner_coordinates = extrap.transform_coordinates(coordinates_batched + 1, grid_batched.shape)
            else:
                inner_coordinates = coordinates_batched + 1
            result = backend.grid_sample(grid_batched.native(),
                                         grid.shape.index(grid.shape.spatial),
                                         inner_coordinates.native(),
                                         'boundary')
        if result is not NotImplemented:
            result_shape = shape_(batch=max(grid.shape.batch.volume, coordinates.shape.batch.volume)) & coordinates_batched.shape.spatial & grid_batched.shape.channel
            result = NativeTensor(result, result_shape)
            result = result.batch.split(grid.shape.batch & coordinates.shape.batch).vector.split(grid.shape.channel)
            return result
    # fallback to slower grid sampling
    neighbors = _closest_grid_values(grid, coordinates, extrap or extrapolation_.ZERO, 'closest_')
    binary = meshgrid(**{f'closest_{dim}': (0, 1) for dim in grid.shape.spatial.names})
    right_weights = coordinates % 1
    binary, right_weights = join_spaces(binary, right_weights)
    weights = prod(binary * right_weights + (1 - binary) * (1 - right_weights), 'vector')
    result = sum_(neighbors * weights, dim=[f"closest_{dim}" for dim in grid.shape.spatial.names])
    return result


def join_spaces(*tensors):
    spatial = functools.reduce(lambda s1, s2: s1.combined(s2, combine_spatial=True), [t.shape.spatial for t in tensors])
    return [CollapsedTensor(t, t.shape.non_spatial & spatial) for t in tensors]


def broadcast_op(operation: Callable,
                 tensors: tuple or list,
                 iter_dims: set or tuple or list or Shape = None,
                 no_return=False):
    if iter_dims is None:
        iter_dims = set()
        for tensor in tensors:
            if isinstance(tensor, TensorStack) and tensor.requires_broadcast:
                iter_dims.add(tensor.stack_dim_name)
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
            if dim in tensor.shape:
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
            return TensorStack(result_unstacked, dim, dim_type)


def split_dimension(value: Tensor, dim: str, split_dims: Shape):
    """
    Decompresses a tensor dimension by unstacking the elements along it.
    The compressed dimension `dim` is assumed to contain elements laid out according to the order or `split_dims`.

    See Also:
        `join_dimensions()`

    Args:
        value: `Tensor` for which one dimension should be split.
        dim: Compressed dimension to be decompressed.
        split_dims: Ordered new dimensions to replace `dim` as `Shape`.

    Returns:
        `Tensor` with decompressed shape
    """
    if split_dims.rank == 0:
        return value.dimension(dim)[0]  # remove dim
    if split_dims.rank == 1:
        new_shape = value.shape.without(dim).expand(split_dims.sizes[0], split_dims.name, split_dims.types[0], pos=value.shape.index(dim))
        return value._with_shape_replaced(new_shape)
    else:
        native = value.native()
        new_shape = value.shape.without(dim)
        i = value.shape.index(dim)
        for size, name, dim_type in split_dims.dimensions:
            new_shape = new_shape.expand(size, name, dim_type, pos=i)
            i += 1
        native_reshaped = choose_backend(native).reshape(native, new_shape.sizes)
        return NativeTensor(native_reshaped, new_shape)


def join_dimensions(value: Tensor,
                    dims: Shape or tuple or list,
                    joined_dim_name: str,
                    pos: int or None = None):
    """
    Compresses multiple dimensions into a single dimension by concatenating the elements.
    Elements along the new dimensions are laid out according to the order of `dims`.
    If the order of `dims` differs from the current dimension order, the tensor is transposed accordingly.

    The type of the new dimension will be equal to the types of `dims`.
    If `dims` have varying types, the new dimension will be a batch dimension.

    See Also:
        `split_dimension()`

    Args:
        value: Tensor containing the dimensions `dims`.
        dims: Dimensions to be compressed in the specified order.
        joined_dim_name: Name of the new dimension.
        pos: Index of new dimension. `None` for automatic, `-1` for last, `0` for first.

    Returns:
        `Tensor` with compressed shape.
    """
    if len(dims) == 0:
        return CollapsedTensor(value, value.shape.expand(1, joined_dim_name, _infer_dim_type_from_name(joined_dim_name), pos))
    if len(dims) == 1:
        old_dim = dims.name if isinstance(dims, Shape) else dims[0]
        new_shape = value.shape.with_names([joined_dim_name if name == old_dim else name for name in value.shape.names])
        return value._with_shape_replaced(new_shape)
    order = value.shape.order_group(dims)
    native = value.native(order)
    types = value.shape.get_type(dims)
    dim_type = types[0] if len(set(types)) == 1 else BATCH_DIM
    if pos is None:
        pos = min(value.shape.indices(dims))
    new_shape = value.shape.without(dims).expand(value.shape.only(dims).volume, joined_dim_name, dim_type, pos)
    native = choose_backend(native).reshape(native, new_shape.sizes)
    return NativeTensor(native, new_shape)


def flatten(value: Tensor, flat_dim: str = 'flat'):
    return join_dimensions(value, value.shape, flat_dim)


def where(condition: Tensor or float or int, value_true: Tensor or float or int, value_false: Tensor or float or int):
    """
    Builds a tensor by choosing either values from `value_true` or `value_false` depending on `condition`.
    If `condition` is not of type boolean, non-zero values are interpreted as True.
    
    This function requires non-None values for `value_true` and `value_false`.
    To get the indices of True / non-zero values, use :func:`nonzero`.

    Args:
      condition: determines where to choose values from value_true or from value_false
      value_true: values to pick where condition != 0 / True
      value_false: values to pick where condition == 0 / False
      condition: Tensor or float or int: 
      value_true: Tensor or float or int: 
      value_false: Tensor or float or int: 

    Returns:
      tensor containing dimensions of all inputs

    """
    condition, value_true, value_false = tensors(condition, value_true, value_false)
    shape, (c, vt, vf) = broadcastable_native_tensors(condition, value_true, value_false)
    result = choose_backend(c, vt, vf).where(c, vt, vf)
    return NativeTensor(result, shape)


def nonzero(value: Tensor, list_dim='nonzero', index_dim='vector'):
    """
    Get spatial indices of non-zero / True values.
    
    Batch dimensions are preserved by this operation.
    If channel dimensions are present, this method returns the indices where any entry is nonzero.

    Args:
      value: spatial tensor to find non-zero / True values in.
      list_dim: name of dimension listing non-zero values (Default value = 'nonzero')
      index_dim: name of index dimension (Default value = 'vector')
      value: Tensor: 

    Returns:
      tensor of shape (batch dims..., list_dim=#non-zero, index_dim=value.shape.spatial_rank)

    """
    if value.shape.channel_rank > 0:
        value = sum_(abs(value), value.shape.channel.names)

    def unbatched_nonzero(value):
        native = value.native()
        backend = choose_backend(native)
        indices = backend.nonzero(native)
        indices_shape = Shape(backend.staticshape(indices), (list_dim, index_dim), (BATCH_DIM, CHANNEL_DIM))
        return NativeTensor(indices, indices_shape)

    return broadcast_op(unbatched_nonzero, [value], iter_dims=value.shape.batch.names)


def _reduce(value: Tensor or list or tuple,
            dim: str or tuple or list or Shape or None,
            native_function: Callable,
            collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
            unaffected_function: Callable = lambda value: value) -> Tensor:
    """

    Args:
        value:
        dim:
        native_function:
        collapsed_function: handles collapsed dimensions, called as `collapsed_function(inner_reduced, collapsed_dims_to_reduce)`
        unaffected_function: returns `unaffected_function(value)` if `len(dims) > 0` but none of them are part of `value`

    Returns:

    """
    if dim in ((), [], EMPTY_SHAPE):
        return value
    if isinstance(value, (tuple, list)):
        values = [wrap(v) for v in value]
        value = _stack(values, '_reduce', BATCH_DIM)
        if dim is None:
            pass  # continue below
        elif dim == 0:
            dim = '_reduce'
        else:
            raise ValueError('dim must be 0 or None when passing a sequence of tensors')
    else:
        value = wrap(value)
    dims = _resolve_dims(dim, value.shape)
    return value.__tensor_reduce__(dims, native_function, collapsed_function, unaffected_function)


def _resolve_dims(dim: str or tuple or list or Shape or None,
                  t_shape: Shape) -> Tuple[str]:
    if dim is None:
        return t_shape.names
    return parse_dim_order(dim)


def sum_(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.sum(native, dim),
                   collapsed_function=lambda inner, red_shape: inner * red_shape.volume)


def prod(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.prod(native, dim),
                   collapsed_function=lambda inner, red_shape: inner ** red_shape.volume)


def mean(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.mean(native, dim))


def std(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.std(native, dim),
                   collapsed_function=lambda inner, red_shape: inner,
                   unaffected_function=lambda value: value * 0)


def any_(boolean_tensor: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(boolean_tensor, dim,
                   native_function=lambda backend, native, dim: backend.any(native, dim))


def all_(boolean_tensor: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(boolean_tensor, dim,
                   native_function=lambda backend, native, dim: backend.all(native, dim))


def max_(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.max(native, dim))


def min_(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.min(native, dim))


def dot(x: Tensor,
        x_dims: str or tuple or list or Shape,
        y: Tensor,
        y_dims: str or tuple or list or Shape) -> Tensor:
    """
    Computes the dot product along the specified dimensions.
    Contracts `x_dims` with `y_dims` by first multiplying the elements and then summing them up.

    For one dimension, this is equal to matrix-matrix or matrix-vector multiplication.

    Args:
        x: First `Tensor`
        x_dims: Dimensions of `x` to reduce against `y`
        y: Second `Tensor`
        y_dims: Dimensions of `y` to reduce against `x`.

    Returns:
        Dot product as `Tensor`.
    """
    x_dims = _resolve_dims(x_dims, x.shape)
    y_dims = _resolve_dims(y_dims, y.shape)
    x_native = x.native()
    y_native = y.native()
    backend = choose_backend(x_native, y_native)
    remaining_shape_x = x.shape.without(x_dims)
    remaining_shape_y = y.shape.without(y_dims)
    if remaining_shape_y.only(remaining_shape_x).is_empty:  # no shared batch dimensions -> tensordot
        result_native = backend.tensordot(x_native, x.shape.index(x_dims), y_native, y.shape.index(y_dims))
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
                if dim in x_letter_map:
                    y_letters.append(x_letter_map[dim])
                else:
                    y_letters.append(KEEP_LETTERS.pop(0))
        keep_letters = list('abcdefgh')[:-len(KEEP_LETTERS)]
        subscripts = f'{"".join(x_letters)},{"".join(y_letters)}->{"".join(keep_letters)}'
        result_native = backend.einsum(subscripts, x_native, y_native)
    result_shape = combine_safe(x.shape.without(x_dims), y.shape.without(y_dims))  # don't check group match
    return NativeTensor(result_native, result_shape)


def _backend_op1(x: Tensor, unbound_method) -> Tensor:
    return x._op1(lambda native: getattr(choose_backend(native), unbound_method.__name__)(native))


def abs(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.abs)


def sign(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.sign)


def round_(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.round)


def ceil(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.ceil)


def floor(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.floor)


def sqrt(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.sqrt)


def exp(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.exp)


def to_float(x: Tensor) -> Tensor:
    """
    Converts the given tensor to floating point format with the currently specified precision.
    
    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.
    
    See the `phi.math` module documentation at https://tum-pbs.github.io/PhiFlow/Math.html

    Args:
      x: values to convert
      x: Tensor: 

    Returns:
      Tensor of same shape as `x`

    """
    return _backend_op1(x, Backend.to_float)


def to_int(x: Tensor, int64=False) -> Tensor:
    return x._op1(lambda native: choose_backend(native).to_int(native, int64=int64))


def to_complex(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.to_complex)


def isfinite(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.isfinite)


def imag(complex: Tensor) -> Tensor:
    return _backend_op1(complex, Backend.imag)


def real(complex: Tensor) -> Tensor:
    return _backend_op1(complex, Backend.real)


def sin(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.sin)


def cos(x: Tensor) -> Tensor:
    return _backend_op1(x, Backend.cos)


def cast(x: Tensor, dtype: DType) -> Tensor:
    return x._op1(lambda native: choose_backend(native).cast(native, dtype=dtype))


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
    dtypes = [v.dtype for v in values]
    if any(dt != dtypes[0] for dt in dtypes):
        common_type = combine_types(*dtypes, fp_precision=get_precision())
        return tuple([cast(v, common_type) for v in values])
    else:
        return values


def divide_no_nan(x, y):
    return custom_op2(x, y, divide_no_nan, lambda x_, y_: choose_backend(x_, y_).divide_no_nan(x_, y_), lambda y_, x_: divide_no_nan(x_, y_), lambda y_, x_: choose_backend(x_, y_).divide_no_nan(x_, y_))


def maximum(x: Tensor or float, y: Tensor or float):
    return custom_op2(x, y, maximum, lambda x_, y_: choose_backend(x_, y_).maximum(x_, y_))


def minimum(x: Tensor or float, y: Tensor or float):
    return custom_op2(x, y, minimum, lambda x_, y_: choose_backend(x_, y_).minimum(x_, y_))


def clip(x: Tensor, lower_limit: float or Tensor, upper_limit: float or Tensor):
    if isinstance(lower_limit, Number) and isinstance(upper_limit, Number):

        def clip_(x):
            return x._op1(lambda native: choose_backend(native).clip(native, lower_limit, upper_limit))

        return broadcast_op(clip_, [x])
    else:
        return maximum(lower_limit, minimum(x, upper_limit))


def convolve(value: Tensor,
             kernel: Tensor,
             extrapolation: 'extrapolation_.Extrapolation' = None) -> Tensor:
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
    if extrapolation is not None and extrapolation != extrapolation_.ZERO:
        value = pad(value, {dim: (kernel.shape.get_size(dim) // 2, (kernel.shape.get_size(dim) - 1) // 2)
                            for dim in conv_shape.name}, extrapolation)
    native_kernel = reshaped_native(kernel, (batch, out_channels, in_channels, *conv_shape.names), force_expand=in_channels)
    native_value = reshaped_native(value, (batch, in_channels, *conv_shape.names), force_expand=batch)
    backend = choose_backend(native_value, native_kernel)
    native_result = backend.conv(native_value, native_kernel, zero_padding=extrapolation == extrapolation_.ZERO)
    result = reshaped_tensor(native_result, (batch, out_channels, *conv_shape.names))
    return result


def unstack(value: Tensor, dim: str):
    """ Alias for `Tensor.unstack()` """
    return value.unstack(dim)


def boolean_mask(x: Tensor, dim: str, mask: Tensor):
    """
    Discards values `x.dim[i]` where `mask.dim[i]=False`.

    All dimensions of `mask` that are not `dim` are treated as batch dimensions.

    Args:
        x: `Tensor` of values.
        dim: Dimension of `x` to along which to discard slices.
        mask: Boolean `Tensor` marking which values to keep. Must have the dimension `dim` matching `x´.

    Returns:
        Selected values of `x` as `Tensor` with dimensions from `x` and `mask`.
    """
    def uniform_boolean_mask(x: Tensor, mask_1d: Tensor):
        if dim in x.shape:
            x_native = x.native()
            mask_native = mask_1d.native()
            backend = choose_backend(x_native, mask_native)
            result_native = backend.boolean_mask(x_native, mask_native, axis=x.shape.index(dim))
            new_shape = x.shape.with_sizes(backend.staticshape(result_native))
            return NativeTensor(result_native, new_shape)
        else:
            total = int(sum_(to_int(mask_1d)))
            new_shape = mask_1d.shape.with_sizes([total])
            return _expand_dims(x, new_shape)

    return broadcast_op(uniform_boolean_mask, [x, mask], iter_dims=mask.shape.without(dim))


def gather(values: Tensor, indices: Tensor):
    b_values = join_dimensions(values, values.shape.batch, 'batch')
    b_values = join_dimensions(b_values, b_values.shape.channel, 'channel', pos=-1)
    b_indices = _expand_dims(indices, values.shape.batch)
    b_indices = join_dimensions(b_indices, values.shape.batch, 'batch')
    native_values = b_values.native()
    native_indices = b_indices.native()
    backend = choose_backend(native_values, native_indices)
    native_result = backend.batched_gather_nd(native_values, native_indices)
    result_shape = Shape(backend.staticshape(native_result),
                         ('batch', *indices.shape.non_channel.without(values.shape.batch).names, 'vector'),
                         (BATCH_DIM, *indices.shape.non_channel.without(values.shape.batch).types, CHANNEL_DIM))
    b_result = NativeTensor(native_result, result_shape)
    result = split_dimension(b_result, 'vector', values.shape.channel)
    result = split_dimension(result, 'batch', values.shape.batch)
    return result


def scatter(base_grid: Tensor or Shape,
            indices: Tensor,
            values: Tensor,
            scatter_dims: str or tuple or list or 'Shape',
            mode: str = 'update',
            outside_handling: str = 'discard',
            indices_gradient=False):
    """
    Scatters `values` into `base_grid` at `indices`.
    Depending on `mode`, this method has one of the following effects:

    * `mode='update'`: Replaces the values of `base_grid` at `indices` by `values`. The result is undefined if `indices` contains duplicates.
    * `mode='add'`: Adds `values` to `base_grid` at `indices`. The values corresponding to duplicate indices are accumulated.
    * `mode='mean'`: Replaces the values of `base_grid` at `indices` by the mean of all `values` with the same index.

    Args:
        base_grid: `Tensor` into which `values` are scattered.
        indices: `Tensor` of n-dimensional indices at which to place `values`.
            Must have a single channel dimension with size matching the number of spatial dimensions of `base_grid`.
            This dimension is optional if the spatial rank is 1.
            Must also contain all `scatter_dims`.
        values: `Tensor` of values to scatter at `indices`.
        scatter_dims: Dimensions of `values` and/or `indices` to reduce during scattering.
            These dimensions are not treated as batch dimensions.
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
    assert indices.shape.channel.names == ('vector',) or (grid_shape.spatial_rank == 1 and indices.shape.channel_rank == 0)

    batches = (values.shape.non_channel & indices.shape.non_channel).without(scatter_dims)
    lists = indices.shape.only(scatter_dims)
    channels = (grid_shape.channel & values.shape.channel).without(scatter_dims)
    # --- Reshape base_grid to (batch, *base_grid.shape, vector) ---
    shaped_base_grid = join_dimensions(base_grid, batches, 'batch_', pos=0)
    shaped_base_grid = expand_channel(shaped_base_grid, vector_=channels.volume)
    # --- Reshape indices to (batch, list, vector) ---
    shaped_indices = join_dimensions(indices, batches, 'batch_', pos=0)
    shaped_indices = join_dimensions(shaped_indices, lists, 'list_', pos=1)
    # --- Reshape values to (batch, list, vector) ---
    values = _expand_dims(values, channels)
    shaped_values = join_dimensions(values, batches, 'batch_', pos=0)
    shaped_values = join_dimensions(shaped_values, lists, 'list_', pos=1)
    shaped_values = join_dimensions(shaped_values, channels, 'vector_', pos=-1)

    # --- Set up grid ---
    if isinstance(base_grid, Shape):
        with choose_backend_t(indices, values):
            base_grid = zeros(base_grid & batches & values.shape.channel)
        if mode != 'add':
            base_grid += math.nan
    # --- Handle outside indices ---
    if outside_handling == 'clamp':
        shaped_indices = clip(shaped_indices, 0, tensor(grid_shape.spatial, 'vector') - 1)
    elif outside_handling == 'discard':
        indices_inside = min_((_math.round(shaped_indices) >= tensor(0.)) &
                              (_math.round(shaped_indices) < tensor(grid_shape.spatial, 'vector')), 'vector')
        shaped_indices = shaped_indices.list_[indices_inside]
        shaped_values = shaped_values.list_[indices_inside]
        if shaped_indices.shape.is_non_uniform:
            raise NotImplementedError()

    def scatter_forward(shaped_base_grid_, shaped_indices_, shaped_values_):
        shaped_indices_ = _math.to_int(_math.round(shaped_indices_))
        native_grid = reshaped_native(base_grid, (batches, *base_grid.shape.spatial.names, channels), force_expand=True)
        native_values = shaped_values_.native('batch_, list_, vector_')
        native_indices = shaped_indices_.native('batch_, list_, vector')
        backend = choose_backend(native_indices, native_values, native_grid)
        if mode in ('add', 'update'):
            native_result = backend.scatter(native_grid, native_indices, native_values, mode=mode)
        else:  # mean
            zero_grid = backend.zeros_like(native_grid)
            summed = backend.scatter(zero_grid, native_indices, native_values, mode='add')
            count = backend.scatter(zero_grid, native_indices, backend.ones_like(native_values), mode='add')
            native_result = summed / backend.maximum(count, 1)
            native_result = backend.where(count == 0, native_grid, native_result)
        return tensor(native_result, shaped_base_grid_.shape)

    def scatter_backward(shaped_base_grid_, shaped_indices_, shaped_values_, output, d_output):
        values_grad = gather(d_output, shaped_indices_)
        spatial_gradient_indices = gather(_math.spatial_gradient(d_output), shaped_indices_)
        indices_grad = mean(spatial_gradient_indices * shaped_values_, 'vector_')
        return None, indices_grad, values_grad

    scatter_function = scatter_forward
    if indices_gradient:
        scatter_function = custom_gradient(scatter_forward, scatter_backward)

    result = scatter_function(shaped_base_grid, shaped_indices, shaped_values)
    return reshaped_tensor(result, (batches, *base_grid.shape.spatial.names, channels), check_sizes=True)


def fft(x: Tensor):
    """
    Performs a fast Fourier transform (FFT) on all spatial dimensions of x.
    
    The inverse operation is :func:`ifft`.

    Args:
      x: tensor of type float or complex
      x: Tensor: 

    Returns:
      FFT(x) of type complex

    """
    native, assemble = _invertible_standard_form(x)
    result = choose_backend(native).fft(native)
    return assemble(result)


def ifft(k: Tensor):
    native, assemble = _invertible_standard_form(k)
    result = choose_backend(native).ifft(native)
    return assemble(result)


def dtype(x):
    if isinstance(x, Tensor):
        return x.dtype
    else:
        return choose_backend(x).dtype(x)


def expand_batch(value: Tensor, **dims):
    return _expand_dims(value, batch_shape(dims))


def expand_spatial(value: Tensor, **dims):
    return _expand_dims(value, spatial_shape(dims))


def expand_channel(value: Tensor, **dims):
    return _expand_dims(value, channel_shape(dims))


def expand(value: Tensor, add_shape: Shape = EMPTY_SHAPE, **dims):
    return _expand_dims(value, add_shape & shape_(**dims))


def _expand_dims(value: Tensor, new_dims: Shape):
    value = wrap(value)
    shape = value.shape
    for size, dim, dim_type in new_dims.dimensions:
        if dim in value.shape:
            assert shape.get_size(dim) == size
            assert shape.get_type(dim) == dim_type
        else:
            shape = shape.expand(size, dim, dim_type)
    return CollapsedTensor(value, shape)


def _invertible_standard_form(value: Tensor):
    """
    Reshapes the tensor into the shape (batch, spatial..., channel) with a single batch and channel dimension.

    Args:
      value: tensor to reshape
      value: Tensor: 

    Returns:
      reshaped native tensor, inverse function

    """
    normal_order = value.shape.normal_order()
    native = value.native(normal_order.names)
    backend = choose_backend(native)
    standard_form = (value.shape.batch.volume,) + value.shape.spatial.sizes + (value.shape.channel.volume,)
    reshaped = backend.reshape(native, standard_form)

    def assemble(reshaped):
        un_reshaped = backend.reshape(reshaped, backend.shape(native))
        return NativeTensor(un_reshaped, normal_order)

    return reshaped, assemble


def close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks whether all tensors have equal values within the specified tolerance.
    
    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    Args:
      tensors: tensor or tensor-like (constant) each
      rel_tolerance: relative tolerance (Default value = 1e-5)
      abs_tolerance: absolute tolerance (Default value = 0)
      *tensors: 

    Returns:

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
    any_tensor = next(filter(lambda t: isinstance(t, Tensor), values))
    if any_tensor is None:
        values = [wrap(t) for t in values]
    else:  # use Tensor to infer dimensions
        values = [any_tensor._tensor(t).__simplify__() for t in values]
    for other in values[1:]:
        _assert_close(values[0], other, rel_tolerance, abs_tolerance, msg, verbose)


def _assert_close(tensor1, tensor2, rel_tolerance: float, abs_tolerance: float, error_message: str = "", verbose: bool = True):
    if tensor2 is tensor1:
        return
    if isinstance(tensor2, (int, float, bool)):
        np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)

    def inner_assert_close(tensor1, tensor2):
        new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
        np1 = choose_backend(native1).numpy(native1)
        np2 = choose_backend(native2).numpy(native2)
        if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
            np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance, err_msg=error_message, verbose=verbose)

    broadcast_op(inner_assert_close, [tensor1, tensor2], no_return=True)


def jit_compile(f: Callable) -> Callable:
    """
    Compiles a graph based on the function `f`.
    The graph compilation is performed just-in-time (jit) when the returned function is called for the first time.

    The traced function will compute the same result as `f` but may run much faster.
    Some checks may be disabled in the compiled function.

    Args:
        f: Function to be traced.
            All arguments must be of type `Tensor` returning a single `Tensor` or a `tuple` or `list` of tensors.

    Returns:
        Function with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
    """
    wrapper, _, _, _ = _native_wrapper(f, lambda nf, backend: backend.jit_compile(nf), persistent_refs=True)
    return wrapper


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
        warnings.warn(f"Backend '{backend}' not supported. Returning original function.")
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


def custom_gradient(f: Callable, gradient: Callable):
    """
    Creates a function based on `f` that uses a custom gradient for the backpropagation pass.

    *Warning* This method can lead to memory leaks if the gradient funcion is not called.
    Make sure to pass tensors without gradients if the gradient is not required, see `stop_gradient()`.

    Args:
        f: Forward function mapping `Tensor` arguments `x` to a single `Tensor` output or sequence of tensors `y`.
        gradient: Function to compute the vector-Jacobian product for backpropropagation. Will be called as `gradient(*x, *y, *dy) -> *dx`.

    Returns:
        Function with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
    """
    def native_custom_gradient(fun: Callable, backend: Backend):
        def native_gradient(x_natives, y_natives, dy_natives):
            x_natives = list(x_natives)
            y_natives = list(y_natives)
            dy_natives = list(dy_natives)
            x = [t._op1(lambda _: x_natives.pop(0)) for t in input_tensors]
            y = [t._op1(lambda _: y_natives.pop(0)) for t in output_tensors]
            dy = [t._op1(lambda _: dy_natives.pop(0)) for t in output_tensors]
            result = gradient(*x, *y, *dy)
            assert isinstance(result, (tuple, list)), "Gradient function must return tuple or list"
            input_tensors.clear()  # release tensors manually since persistent_refs=True
            output_tensors.clear()
            return [r.native() if r is not None else None for r in result]
        return backend.custom_gradient(fun, native_gradient)

    wrapper, _, input_tensors, output_tensors = _native_wrapper(f, native_custom_gradient, persistent_refs=True)
    return wrapper


def functional_gradient(f: Callable, wrt: tuple or list = (0,), get_output=False) -> Callable:
    """
    Creates a function which computes the spatial_gradient of `f`.

    Example:

    ```python
    def loss_function(x, y):
        prediction = f(x)
        loss = math.l2_loss(prediction - y)
        return loss, prediction

    dx, = functional_gradient(loss_function)(x, y)

    loss, prediction, dx, dy = functional_gradient(loss_function, wrt=(0, 1),
                                                 get_output=True)(x, y)
    ```

    Args:
        f: Function to be differentiated.
            `f` must return a floating point `Tensor` with rank zero.
            It can return additional tensors which are treated as auxiliary data and will be returned by the spatial_gradient function if `return_values=True`.
            All arguments for which the spatial_gradient is computed must be of dtype float or complex.
        get_output: Whether the spatial_gradient function should also return the return values of `f`.
        wrt: Arguments of `f` with respect to which the spatial_gradient should be computed.
            Example: `wrt_indices=[0]` computes the spatial_gradient with respect to the first argument of `f`.

    Returns:
        Function with the same arguments as `f` that returns the value of `f`, auxiliary data and spatial_gradient of `f` if `get_output=True`, else just the spatial_gradient of `f`.
    """
    INPUT_TENSORS = []
    OUTPUT_TENSORS = []
    ARG_INDICES = []

    def native_function(*natives):
        natives = list(natives)
        values = [t._op1(lambda _: natives.pop(0)) for t in INPUT_TENSORS]
        assert len(natives) == 0, "Not all arguments were converted"
        result = f(*values)
        results = [result] if not isinstance(result, (tuple, list)) else result
        assert all(isinstance(t, Tensor) for t in results), f"Function output must be Tensor or sequence of tensors but got {result}."
        OUTPUT_TENSORS.clear()
        OUTPUT_TENSORS.extend(results)
        return sum([v._natives() for v in results], ())

    backend = default_backend()

    class GradientFunction:

        def __init__(self):
            self.gradf = None

        def __call__(self, *args, **kwargs):
            assert not len(kwargs)
            shifted_wrt = [i for i in range(len(ARG_INDICES)) if ARG_INDICES[i] in wrt]
            if self.gradf is None:
                self.gradf = backend.functional_gradient(native_function, shifted_wrt, get_output=get_output)
            return self.gradf(*args)

    grad_native = GradientFunction()

    def wrapper(*values: Tensor):
        assert all(isinstance(v, Tensor) for v in values)
        INPUT_TENSORS.clear()
        INPUT_TENSORS.extend(values)
        ARG_INDICES.clear()
        natives = []
        for arg_index, v in enumerate(values):
            v._expand()
            n = v._natives()
            natives.extend(n)
            for _ in range(len(n)):
                ARG_INDICES.append(arg_index)
        results_native = list(grad_native(*natives))
        assert None not in results_native, f"None found in gradient result {results_native}"
        proto_tensors = []
        if get_output:
            proto_tensors.extend(OUTPUT_TENSORS)
        proto_tensors.extend([t for i, t in enumerate(INPUT_TENSORS) if i in wrt])
        results = [t._with_natives_replaced(results_native) for t in proto_tensors]
        INPUT_TENSORS.clear()
        OUTPUT_TENSORS.clear()
        assert len(results_native) == 0
        return results

    return wrapper


def minimize(function, x0: Tensor, solve_params: Solve, callback: Callable = None) -> Tensor:
    """
    Finds a minimum of the scalar function `function(x)` where `x` is a `Tensor` like `x0`.
    The `solver` argument of `solve_params` determines which method is used.
    All methods supported by `scipy.optimize.minimize` can be used, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html .

    For backends that support gradients (PyTorch, TensorFlow, Jax), `functional_gradient()` is used to determine the optimization direction.
    Otherwise (NumPy), the gradient is determined using finite differences and a warning is printed.

    Information about the performed solve will be added to `solve_params.result`.

    Args:
        function: Function to be minimized
        x0: Initial guess for `x`
        solve_params: Specifies solver type and parameters. Additional solve information will be stored in `solve_params.result`.

    Returns:
        x: solution, the minimum point `x`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the optimization failed prematurely.
    """
    assert solve_params.relative_tolerance == 0, f"relative_tolerance must be zero for minimize() but got {solve_params.relative_tolerance}"
    backend = choose_backend_t(x0, prefer_default=True)
    x0._expand()
    natives = x0._natives()
    natives_flat = [backend.flatten(n) for n in natives]
    x0_flat = backend.concat(natives_flat, 0)

    def unflatten_assemble(x_native):
        i = 0
        x_natives = []
        for native, native_flat in zip(natives, natives_flat):
            vol = backend.shape(native_flat)[0]
            x_natives.append(backend.reshape(x_native[i:i + vol], backend.shape(native)))
            i += vol
        x = x0._op1(lambda _: x_natives.pop(0))  # assemble x0 structure
        return x

    def native_function(native_x):
        x = unflatten_assemble(native_x)
        y = function(x)
        if isinstance(y, (tuple, list)):
            return [y_.native() for y_ in y]
        else:
            return y.native()

    try:
        x_native = backend.minimize(native_function, x0_flat, solve_params)
        if x_native is NotImplemented:
            x_native = _default_minimize(native_function, x0_flat, x0.dtype, backend, solve_params, callback)
        x = unflatten_assemble(x_native)
        return x
    except ConvergenceException as exc:
        x = unflatten_assemble(exc.x)
        raise type(exc)(exc.solve, x0, x, exc.msg)


def _default_minimize(native_function, x0_flat, x_dtype, backend: Backend, solve_params: Solve, callback: Callable = None):
    import scipy.optimize as sopt
    x0 = backend.numpy(x0_flat)
    if x_dtype.kind == complex:
        x0 = x0.view(np.float32 if x0.dtype == np.complex64 else np.float64)
    method = solve_params.solver or 'L-BFGS-B'
    loss_curve = []  # by evaluation

    if backend.supports(Backend.functional_gradient):
        val_and_grad = backend.functional_gradient(native_function, [0], get_output=True)

        def min_target(x: np.ndarray):
            if x_dtype.kind == complex:
                x = x.view(np.complex64 if x.dtype == np.float32 else np.complex128)
            x = backend.as_tensor(x, convert_external=True)
            eval = val_and_grad(x)
            loss = eval[0]
            loss_curve.append(loss)
            grad = eval[-1]
            if callback is not None:
                callback(x, *eval[:-1])
            if x_dtype.kind == complex:
                grad = backend.numpy(grad).astype(np.complex128)
                grad = grad.view(np.float64)
            else:
                grad = backend.numpy(grad).astype(np.float64)
            return backend.numpy(loss).astype(np.float64), grad

        res = sopt.minimize(fun=min_target, x0=x0, jac=True, method=method, tol=solve_params.absolute_tolerance,
                            options={'maxiter': solve_params.max_iterations})
    else:
        warnings.warn(f"{backend} does not support gradients. minimize() will use finite difference gradients which may be slow.")

        def min_target(x):
            if x_dtype.kind == complex:
                x = x.view(np.complex64 if x.dtype == np.float32 else np.complex128)
            x = backend.as_tensor(x, convert_external=True)
            loss = native_function(x)
            loss_curve.append(loss)
            if isinstance(loss, (tuple, list)):
                loss = loss[0]
            return backend.numpy(loss).astype(np.float64)

        finite_diff_epsilon = {64: 1e-9, 32: 1e-4, 16: 1e-1}[get_precision()]
        res = sopt.minimize(fun=min_target, x0=x0, method=method, tol=solve_params.absolute_tolerance,
                            options={'maxiter': solve_params.max_iterations, 'eps': finite_diff_epsilon})
    solve_params.result = SolveResult(res.nit, loss_curve=loss_curve)
    if not res.success:
        raise NotConverged(solve_params, x0, res.x, msg=res.message)
    return res.x


def linear_function(f: Callable, jit_compile=True) -> Callable:
    """
    Mark `f` as a linear function.
    This is required for performing a linear solve using `solve()`.

    Can be used as a decorator:

    ```python
    @math.linear_function
    def my_linear_function(x):
    ```

    If `jit_compile=True`, a sparse matrix representation of `f` will be created when the returned function is called.

    Args:
        f: Linear function with `Tensor` arguments and return value(s).
        jit_compile: Whether to compile a matrix representation for `f`.

    Returns:
        Function with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
    """
    if not jit_compile:
        return LinearFunction(f)
    backend = default_backend()
    if not backend.supports(Backend.sparse_tensor):
        warnings.warn(f"Cannot compile linear function with {backend} because sparse matrices are not supported.")
        return LinearFunction(f)
    return JitLinearFunction(f, backend)


class LinearFunction:

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        assert not kwargs, "kwargs not supported, pass all values as positional arguments."
        return self.f(*args)


class JitLinearFunction(LinearFunction):

    def __init__(self, f, backend: Backend):
        LinearFunction.__init__(self, f)
        self.backend = backend
        self._cached_coo = None
        self._trace_result = None

    def sparse_coordinate_matrix(self, x0):
        if self._cached_coo is None:
            self.build(x0)
        return self._cached_coo

    def build(self, x0):
        from ._trace import lin_placeholder, ShiftLinOp
        with self.backend:
            trace_time = time.perf_counter()
            x_track = lin_placeholder(x0)
            self._trace_result = self.f(x_track)
            assert isinstance(self._trace_result, ShiftLinOp), 'Baking sparse matrix failed. Make sure only supported linear operations are used.'
            trace_time = time.perf_counter() - trace_time
            try:
                build_time = time.perf_counter()
                self._cached_coo = self._trace_result.build_sparse_coordinate_matrix()
                build_time = time.perf_counter() - build_time
            except NotImplementedError as err:
                raise AssertionError(f"Failed to build sparse matrix for linear function", err)
            if get_current_profile():
                get_current_profile().add_external_message(
                    f"Sparse Linear track: {round(trace_time * 1000)} ms, \tbuild: {round(build_time * 1000)} ms")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Currently only supported in solve()")
        # assert not kwargs, "kwargs not supported, pass all values as positional arguments."
        # if len(args) > 1:
        #     raise NotImplementedError("Only single-argument functions can be compiled")
        #
        # x0 = tensor(args[0])
        #
        # if self._trace_result is None:
        #     self.build(x0)


def solve(f: Callable,
          y: Tensor,
          x0: Tensor,
          solve_params: Solve,
          constants: tuple or list = (),
          callback=None) -> Tensor:
    """
    Solves the system of linear or nonlinear equations *f(x) = y*.
    To solve a linear system, decorate `f` with `@math.linear_function`.
    Otherwise, a nonlinear solver will be used which may be slower.

    Information about the performed solve will be stored in `solve_params.result`.
    In case the gradient of this operation is computed during backpropagation, information about the gradient solve will be added to `solve_params.gradient_solve.result`.

    See Also:
        `minimize()`, `linear_function()`.

    Args:
        f: Function `operator(x)` or matrix
        y: Desired output of `operator · x`
        x0: Initial guess for `x`
        solve_params: Specifies solver type and parameters. Additional solve information will be stored in `solve_params.result`.
        callback: Function to be called after each iteration as `callback(x_n)`. *This argument may be ignored by some backends.*

    Returns:
        x: solution of the linear system of equations `operator · x = y`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the solve failed prematurely.
    """
    if not isinstance(f, LinearFunction):
        from ._nd import l2_loss

        def min_func(x):
            diff = f(x) - y
            l2 = l2_loss(diff)
            return l2

        rel_tol_to_abs = solve_params.relative_tolerance * l2_loss(y, batch_norm=True)
        solve_params.absolute_tolerance = rel_tol_to_abs
        solve_params.relative_tolerance = 0
        return minimize(min_func, x0, solve_params=solve_params)
    if solve_params.solver != 'CG':
        raise NotImplementedError("Only 'CG' solver currently supported")

    for c in constants:
        c._expand()

    x0, y = tensors(x0, y)
    backend = choose_backend(*x0._natives(), *y._natives())

    if isinstance(f, JitLinearFunction):
        operator_or_matrix = f.sparse_coordinate_matrix(x0)
    else:
        def operator_or_matrix(native_x):
            native_x_shaped = backend.reshape(native_x, x0.shape.non_batch.sizes)
            x = NativeTensor(native_x_shaped, x0.shape.non_batch)
            Ax = f(x)
            Ax_native = backend.reshape(Ax.native(), backend.shape(native_x))
            return Ax_native

    loop_time = time.perf_counter()
    if backend.supports(Backend.conjugate_gradient):
        x0_native = backend.reshape(x0.native(), (x0.shape.batch.volume, -1))
        y_native = backend.reshape(y.native(), (y.shape.batch.volume, -1))
        x = backend.conjugate_gradient(operator_or_matrix, y_native, x0_native, solve_params, callback)
        batch = (y.shape & x0.shape).batch
        x = backend.reshape(x, batch.sizes + x0.shape.non_batch.sizes)
        x = NativeTensor(x, batch.combined(x0.shape.non_batch))
    else:
        x = _default_cg(operator_or_matrix, y, x0, solve_params, callback)
    if get_current_profile():
        iterations = solve_params.result.iterations
        loop_time = time.perf_counter() - loop_time
        info = "  \tProfile with trace=False to get more accurate results." if get_current_profile()._trace else ""
        get_current_profile().add_external_message(f"CG \tloop: {round(loop_time * 1000)} ms / {iterations} iterations {info}")
    return x


def _default_cg(A, y_t: Tensor, x0: Tensor, solve_params: Solve, callback=None):
    """ This conjugate gradient algorithm should work with all backends. The gradient pass performs a second CG. """
    if callback is not None:
        warnings.warn("callback is not yet supported and will not be called.")
    batch = (y_t.shape & x0.shape).batch
    backend = choose_backend_t(y_t, x0)

    if callable(A):
        function = A
    else:
        A = backend.as_tensor(A)
        A_shape = backend.staticshape(A)
        assert len(A_shape) == 2, f"A must be a square matrix but got shape {A_shape}"
        assert A_shape[0] == A_shape[1], f"A must be a square matrix but got shape {A_shape}"

        def function(vec):
            return backend.matmul(A, vec)

    def cg(y: Tensor, x0_t: Tensor, params: Solve):
        y = backend.to_float(backend.reshape(y.native(), (y.shape.batch.volume, -1)))
        x0 = backend.copy(backend.to_float(backend.reshape(x0_t.native(), (x0_t.shape.batch.volume, -1))), only_mutable=True)
        batch_size = combined_dim(x0.shape[0], y.shape[0])
        if x0.shape[0] < batch_size:
            x0 = backend.tile(x0, [batch_size, 1])
        tolerance_sq = backend.maximum(params.relative_tolerance ** 2 * backend.sum(y ** 2, -1), params.absolute_tolerance ** 2)
        x = x0
        dx = residual = y - function(x)
        dy = function(dx)
        iterations = 0
        while backend.all(backend.sum(residual ** 2, -1) > tolerance_sq):
            if iterations == params.max_iterations:
                solve_params.result = SolveResult(iterations)
                raise NotConverged(solve_params, x0, x)
            iterations += 1
            dx_dy = backend.sum(dx * dy, axis=-1, keepdims=True)
            step_size = backend.divide_no_nan(backend.sum(dx * residual, axis=-1, keepdims=True), dx_dy)
            x += step_size * dx
            residual -= step_size * dy
            dx = residual - backend.divide_no_nan(backend.sum(residual * dy, axis=-1, keepdims=True) * dx, dx_dy)
            dy = function(dx)
        if not backend.all(backend.isfinite(x)):
            solve_params.result = SolveResult(iterations)
            raise Diverged(solve_params, x0, x)
        x = backend.reshape(x, batch.sizes + x0_t.shape.non_batch.sizes)
        params.result = SolveResult(iterations)
        return NativeTensor(x, batch.combined(x0_t.shape.non_batch))

    return custom_gradient(lambda y: cg(y, x0, solve_params),
                           lambda y, x, dx: (cg(dx, zeros_like(x0), solve_params.gradient_solve),))(y_t)


# def variable(x: Tensor) -> Tensor:
#     """
#     Returns a copy of `x` for which gradients are recorded automatically.
#     The function `gradients()` may be used to compute gradients of a Tensor derived from `x` w.r.t. `x`.
#
#     If the backend of `x` does not support variables, converts `x` to the default backend.
#     If the default backend does not support variables, raises an exception.
#
#     Alternatively, `record_gradients()` may be used to record gradients only for specific operations.
#
#     Args:
#         x: Parameter for which gradients of the form dL/dx may be computed
#
#     Returns:
#         copy of `x`
#     """
#     x._expand()  # CollapsedTensors need to be expanded early
#
#     def create_var(native):
#         native_backend = choose_backend(native)
#         native_backend_var = native_backend.variable(native)
#         if native_backend_var is not NotImplemented:
#             return native_backend_var
#         default_be = default_backend()
#         if default_be == native_backend:
#             raise AssertionError(f"The backend '{native_backend}' does not support variables.")
#         native = default_be.as_tensor(native, convert_external=True)
#         default_backend_var = default_be.variable(native)
#         if default_backend_var is not NotImplemented:
#             return default_backend_var
#         raise AssertionError(f"The default backend '{default_be}' does not support variables.")
#
#     result = x._op1(create_var)
#     return result


@contextmanager
def record_gradients(*x: Tensor, persistent=False):
    """
    Context expression to record gradients for operations within that directly or indirectly depend on `x`.

    The function `gradients()` may be called within the context to evaluate the gradients of a Tensor derived from `x` w.r.t. `x`.

    Args:
        *x: Parameters for which gradients of the form dL/dx may be computed
        persistent: if `False`, `gradients()` may only be called once within the context
    """
    for x_ in x:
        x_._expand()
    natives = sum([x_._natives() for x_ in x], ())
    backend = choose_backend(*natives)
    ctx = backend.record_gradients(natives, persistent=persistent)
    _PARAM_STACK.append(x)
    ctx.__enter__()
    try:
        yield None
    finally:
        ctx.__exit__(None, None, None)
        _PARAM_STACK.pop(0)


_PARAM_STACK = []  # list of tuples


def gradients(y: Tensor,
              *x: Tensor,
              grad_y: Tensor or None = None):
    """
    Computes the gradients dy/dx for each `x`.
    The parameters `x` must be marked prior to all operations for which gradients should be recorded using `record_gradients()`.

    Args:
        y: Scalar `Tensor` computed from `x`, typically loss.
        *x: (Optional) Parameter tensors which were previously marked in `record_gradients()`.
            If empty, computes the gradients w.r.t. all marked tensors.
        grad_y: (optional) Gradient at `y`, defaults to 1.

    Returns:
        Single `Tensor` if one `x` was passed, else sequence of tensors.
    """
    assert isinstance(y, NativeTensor), f"{type(y)}"
    if len(x) == 0:
        x = _PARAM_STACK[-1]
    backend = choose_backend_t(y, *x)
    x_natives = sum([x_._natives() for x_ in x], ())
    native_gradients = list(backend.gradients(y.native(), x_natives, grad_y.native() if grad_y is not None else None))
    for i, grad in enumerate(native_gradients):
        assert grad is not None, f"Missing spatial_gradient for source with shape {x_natives[i].shape}"
    grads = [x_._op1(lambda native: native_gradients.pop(0)) for x_ in x]
    return grads[0] if len(x) == 1 else grads


def stop_gradient(value: Tensor):
    """
    Disables gradients for the given tensor.
    This may switch off the gradients for `value` itself or create a copy of `value` with disabled gradients.

    Args:
        value: tensor for which gradients should be disabled.

    Returns:
        Copy of `value` or `value`.
    """
    return value._op1(lambda native: choose_backend(native).stop_gradient(native))

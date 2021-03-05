import functools
import re
import time
import warnings
from contextlib import contextmanager
from numbers import Number
from typing import Tuple, Callable

import numpy as np

from .backend import default_backend, choose_backend, Solve, LinearSolve, Backend, get_precision
from .backend._dtype import DType, combine_types
from ._shape import BATCH_DIM, CHANNEL_DIM, SPATIAL_DIM, Shape, EMPTY_SHAPE, spatial_shape, shape as shape_, \
    _infer_dim_type_from_name, combine_safe
from ._tensors import Tensor, wrap, tensor, broadcastable_native_tensors, NativeTensor, TensorStack, CollapsedTensor, custom_op2, tensors, TensorDim
from . import extrapolation
from .backend._profile import get_current_profile


def choose_backend_t(*values, prefer_default=False):
    """ Choose backend for given `Tensor` or native tensor values. """
    natives = sum([v._natives() if isinstance(v, Tensor) else (v,) for v in values], ())
    return choose_backend(*natives, prefer_default=prefer_default)


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


def print_(value: Tensor = None, name: str = None):
    """
    Print a tensor with no more than two spatial dimensions, splitting it along all batch and channel dimensions.
    
    Unlike regular printing, the primary dimension, typically x, is oriented to the right.

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
    if name is not None:
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
            text = np.array2string(value[index_dict].numpy(dim_order), precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', re.sub('\],', '', text)))
    else:
        raise NotImplementedError('Can only print tensors with up to 2 spatial dimensions.')


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
    broadcast_shape = values[0].shape
    natives = [v.native(order=broadcast_shape.names) for v in values]
    backend = choose_backend(*natives)
    concatenated = backend.concat(natives, broadcast_shape.index(dim))
    return NativeTensor(concatenated, broadcast_shape.with_sizes(backend.staticshape(concatenated)))


def pad(value: Tensor, widths: dict, mode: 'extrapolation.Extrapolation') -> Tensor:
    """
    Pads a tensor along the specified dimensions, determining the added values using the given extrapolation.
    
    This is equivalent to calling `mode.pad(value, widths)`.

    Args:
      value: tensor to be padded
      widths: name: str -> (lower: int, upper: int)
      mode: Extrapolation object
      value: Tensor: 
      widths: dict: 
      mode: 'extrapolation.Extrapolation': 

    Returns:
      padded Tensor

    """
    return mode.pad(value, widths)


def closest_grid_values(grid: Tensor,
                        coordinates: Tensor,
                        extrap: 'extrapolation.Extrapolation',
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
                         extrap: 'extrapolation.Extrapolation',
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


def grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    result = broadcast_op(functools.partial(_grid_sample, extrap=extrap), [grid, coordinates])
    return result


def _grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation' or None):
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
            grid_batched = pad(grid_batched, {dim: (1, 1) for dim in grid.shape.spatial.names}, extrap or extrapolation.ZERO)
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
    neighbors = _closest_grid_values(grid, coordinates, extrap or extrapolation.ZERO, 'closest_')
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
                 iter_dims: set or tuple or list = None,
                 no_return=False):
    if iter_dims is None:
        iter_dims = set()
        for tensor in tensors:
            if isinstance(tensor, TensorStack) and tensor.requires_broadcast:
                iter_dims.add(tensor.stack_dim_name)
    if len(iter_dims) == 0:
        return operation(*tensors)
    else:
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


def split_dimension(dim: TensorDim, split_dims: Shape):
    """
    Decompresses a tensor dimension by unstacking the elements along it.
    The compressed dimension `dim` is assumed to contain elements laid out according to the order or `split_dims`.

    See Also:
        `join_dimensions()`

    Args:
        dim: Compressed dimension to be decompressed.
        split_dims: Ordered new dimensions to replace `dim` as `Shape`.

    Returns:
        `Tensor` with decompressed shape
    """
    if split_dims.rank == 0:
        return dim[0]
    if split_dims.rank == 1:
        value = dim.tensor
        new_shape = value.shape.with_names([split_dims.name if name == dim.name else name for name in value.shape.names])
        return value._with_shape_replaced(new_shape)
    else:
        value = dim.tensor
        native = value.native()
        new_shape = value.shape.without(dim.name)
        i = dim.index
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
                  shape: Shape) -> Tuple[str]:
    if dim is None:
        return shape.names
    if isinstance(dim, list):
        return tuple(dim)
    if isinstance(dim, tuple):
        return dim
    if isinstance(dim, str):
        return dim,
    if isinstance(dim, Shape):
        return dim.names
    raise ValueError(dim)


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


def dot(a, b, axes) -> Tensor:
    """ Not yet implemented. """
    raise NotImplementedError()


def matmul(A, b) -> Tensor:
    """ Not yet implemented. """
    raise NotImplementedError()


def einsum(equation, *tensors) -> Tensor:
    """ Not yet implemented. """
    raise NotImplementedError()


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


def conv(value: Tensor, kernel: Tensor, padding='same'):
    """ Not yet implemented. """
    raise NotImplementedError()


def unstack(value: Tensor, dim: str):
    """ Alias for `Tensor.unstack()` """
    return value.unstack(dim)


def boolean_mask(x: Tensor, mask):
    """ Not yet implemented. """
    raise NotImplementedError()


def gather(values: Tensor, indices: Tensor):
    b_values = join_dimensions(values, values.shape.batch, 'batch')
    b_values = join_dimensions(b_values, b_values.shape.channel, 'channel', pos=-1)
    b_indices = join_dimensions(indices, indices.shape.batch, 'batch')
    native_values = b_values.native()
    native_indices = b_indices.native()
    native_result = choose_backend(native_values, native_indices).batched_gather_nd(native_values, native_indices)
    b_result = tensor(native_result, ('batch', *indices.shape.spatial.names, 'vector'))
    result = split_dimension(b_result.vector, values.shape.channel)
    result = split_dimension(result.batch, values.shape.batch & indices.shape.batch)
    return result


def scatter(indices: Tensor,
            values: Tensor,
            size: Shape,
            scatter_dims: str or tuple or list or 'Shape',
            duplicates_handling: str = 'undefined',
            outside_handling: str = 'discard'):
    """
    Create a dense tensor from sparse values.

    Args:
        indices: n-dimensional indices corresponding to values
        values: values to scatter at indices
        size: spatial size of dense tensor
        scatter_dims: dimensions of values/indices to reduce during scattering
        duplicates_handling: one of ('undefined', 'add', 'mean', 'any') (Default value = 'undefined')
        outside_handling: one of ('discard', 'clamp', 'undefined') (Default value = 'discard')

    Returns:
        Tensor of shape `size` and dtype matching `values`.
    """
    indices_ = indices.native()
    values_ = values.native(values.shape.combined(indices.shape.non_channel).names)
    backend = choose_backend(indices_, values_)
    result_ = backend.scatter(indices_, values_, tuple(size), duplicates_handling=duplicates_handling, outside_handling=outside_handling)
    result_shape = size & indices.shape.batch & values.shape.non_spatial
    result_shape = result_shape.without(scatter_dims)
    return NativeTensor(result_, result_shape)


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


def tile(value, multiples):
    """ Not yet implemented. """
    raise NotImplementedError()


def expand_batch(value: Tensor, dim_name: str, dim_size: int = 1):
    return _expand_dim(value, dim_name, dim_size, BATCH_DIM)


def expand_spatial(value: Tensor, dim_name: str, dim_size: int = 1):
    return _expand_dim(value, dim_name, dim_size, SPATIAL_DIM)


def expand_channel(value: Tensor, dim_name: str, dim_size: int = 1):
    return _expand_dim(value, dim_name, dim_size, CHANNEL_DIM)


def expand(value: Tensor, dim_name: str, dim_size: int = 1):
    dim_type = _infer_dim_type_from_name(dim_name)
    return _expand_dim(value, dim_name, dim_size, dim_type)


def _expand_dim(value: Tensor, dim_name: str, dim_size: int, dim_type: str):
    value = wrap(value)
    if dim_name in value.shape:
        assert value.shape.get_size(dim_name) == dim_size
        assert value.shape.get_type(dim_name) == dim_type
        return value
    new_shape = value.shape.expand(dim_size, dim_name, dim_type)
    return CollapsedTensor(value, new_shape)


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


def assert_close(*tensors,
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0):
    """
    Checks that all tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.
    
    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    Args:
      tensors: tensor or tensor-like (constant) each
      rel_tolerance: relative tolerance (Default value = 1e-5)
      abs_tolerance: absolute tolerance (Default value = 0)
      *tensors: 

    Returns:

    """
    any_tensor = next(filter(lambda t: isinstance(t, Tensor), tensors))
    if any_tensor is None:
        tensors = [wrap(t) for t in tensors]
    else:  # use Tensor to infer dimensions
        tensors = [any_tensor._tensor(t).__simplify__() for t in tensors]
    for other in tensors[1:]:
        _assert_close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)


def _assert_close(tensor1, tensor2, rel_tolerance: float, abs_tolerance: float):
    if tensor2 is tensor1:
        return
    if isinstance(tensor2, (int, float, bool)):
        np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)

    def inner_assert_close(tensor1, tensor2):
        new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
        np1 = choose_backend(native1).numpy(native1)
        np2 = choose_backend(native2).numpy(native2)
        if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
            np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance)

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
    INPUT_TENSORS = []
    OUTPUT_TENSORS = []

    def native_function(*natives):
        natives = list(natives)
        values = [t._op1(lambda _: natives.pop(0)) for t in INPUT_TENSORS]
        assert len(natives) == 0, "Not all arguments were converted"
        result = f(*values)
        results = [result] if not isinstance(result, (tuple, list)) else result
        OUTPUT_TENSORS.clear()
        OUTPUT_TENSORS.extend(results)
        return sum([v._natives() for v in results], ())

    backend = default_backend()
    traced = backend.jit_compile(native_function)
    if traced is NotImplemented:
        warnings.warn(f"Backend '{backend}' does not support function tracing. Returning original function.")
        return f

    def wrapper(*values: Tensor):
        INPUT_TENSORS.clear()
        INPUT_TENSORS.extend(values)
        for v in values:
            v._expand()
        natives = sum([v._natives() for v in values], ())
        results_native = list(traced(*natives))
        results = [t._with_natives_replaced(results_native) for t in OUTPUT_TENSORS]
        assert len(results_native) == 0
        return results[0] if len(results) == 1 else results

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
        proto_tensors = []
        if get_output:
            proto_tensors.extend(OUTPUT_TENSORS)
        proto_tensors.extend([t for i, t in enumerate(INPUT_TENSORS) if i in wrt])
        results = [t._with_natives_replaced(results_native) for t in proto_tensors]
        assert len(results_native) == 0
        return results

    return wrapper


def minimize(function, x0: Tensor, solve_params: Solve) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Finds a minimum of the scalar function `function(x)` where `x` is a `Tensor` like `x0`.

    Args:
        function: Function to be minimized
        x0: Initial guess for `x`
        solve_params: Specifies solver type and parameters. Additional solve information will be stored in `solve_params.result`.

    Returns:
        converged: scalar bool tensor representing whether the solve found a solution within the specified accuracy within the allowed iterations
        x: solution, the minimum point `x`.
        iterations: number of iterations performed
    """
    backend = choose_backend_t(x0)
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
        return y.native()

    x_native = backend.minimize(native_function, x0_flat, solve_params)
    x = unflatten_assemble(x_native)
    return wrap(solve_params.result.success), x, wrap(solve_params.result.iterations)


def solve(operator,
          y: Tensor,
          x0: Tensor,
          solve_params: Solve,
          constants: tuple or list = (),
          callback=None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Solves the system of linear or nonlinear equations *operator · x = y*.

    Args:
        operator: Function `operator(x)` or matrix
        y: Desired output of `operator · x`
        x0: Initial guess for `x`
        solve_params: Specifies solver type and parameters. Additional solve information will be stored in `solve_params.result`.
        callback: Function to be called after each iteration as `callback(x_n)`. *This argument may be ignored by some backends.*

    Returns:
        converged: scalar bool tensor representing whether the solve found a solution within the specified accuracy within the allowed iterations
        x: solution of the linear system of equations `operator · x = y`.
        iterations: number of iterations performed
    """
    if not isinstance(solve_params, LinearSolve):
        from ._nd import l2_loss

        def min_func(x):
            diff = operator(x) - y
            l2 = l2_loss(diff)
            return l2

        rel_tol_to_abs = solve_params.relative_tolerance * l2_loss(y, batch_norm=True)
        solve_params.absolute_tolerance = rel_tol_to_abs
        solve_params.relative_tolerance = None
        return minimize(min_func, x0, solve_params=solve_params)
    if solve_params.solver not in (None, 'CG'):
        raise NotImplementedError("Only 'CG' solver currently supported")

    for c in constants:
        c._expand()

    from ._track import lin_placeholder, ShiftLinOp
    x0, y = tensors(x0, y)
    backend = choose_backend(*x0._natives(), *y._natives())
    batch = (y.shape & x0.shape).batch
    x0_native = backend.reshape(x0.native(), (x0.shape.batch.volume, -1))
    y_native = backend.reshape(y.native(), (y.shape.batch.volume, -1))

    if callable(operator):
        operator_or_matrix = None
        if solve_params.bake == 'sparse':
            track_time = time.perf_counter()
            x_track = lin_placeholder(x0)
            Ax_track = operator(x_track)
            assert isinstance(Ax_track, ShiftLinOp), 'Baking sparse matrix failed. Make sure only supported linear operations are used.'
            track_time = time.perf_counter() - track_time
            build_time = time.perf_counter()
            try:
                operator_or_matrix = Ax_track.build_sparse_coordinate_matrix()
            except NotImplementedError as err:
                warnings.warn(f"Failed to build sparse matrix, using function directly. {err}")
            # TODO reshape x0, y so that independent dimensions are batch
            build_time = time.perf_counter() - build_time
        if operator_or_matrix is None:
            def operator_or_matrix(native_x):
                native_x_shaped = backend.reshape(native_x, x0.shape.non_batch.sizes)
                x = NativeTensor(native_x_shaped, x0.shape.non_batch)
                Ax = operator(x)
                Ax_native = backend.reshape(Ax.native(), backend.shape(native_x))
                return Ax_native
    else:
        operator_or_matrix = backend.reshape(operator.native(), (y.shape.non_batch.volume, x0.shape.non_batch.volume))

    loop_time = time.perf_counter()
    x = backend.conjugate_gradient(operator_or_matrix, y_native, x0_native, solve_params, callback)
    converged = solve_params.result.success
    iterations = solve_params.result.iterations
    loop_time = time.perf_counter() - loop_time
    if get_current_profile():
        info = "  \tProfile with trace=False to get more accurate results." if get_current_profile()._trace else ""
        get_current_profile().add_external_message(f"CG   track: {round(track_time * 1000)} ms  \tbuild: {round(build_time * 1000)} ms  \tloop: {round(loop_time * 1000)} ms / {iterations} iterations {info}")
    x = backend.reshape(x, batch.sizes + x0.shape.non_batch.sizes)
    return NativeTensor(converged, EMPTY_SHAPE), NativeTensor(x, batch.combined(x0.shape.non_batch)), NativeTensor(iterations, EMPTY_SHAPE)


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

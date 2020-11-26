import functools
import numbers
import re
import time
from functools import partial

import numpy as np

from .backend import math, Solve, LinearSolve
from ._shape import BATCH_DIM, CHANNEL_DIM, SPATIAL_DIM, Shape, EMPTY_SHAPE, spatial_shape, shape as shape_
from . import _extrapolation as extrapolation
from ._tensors import Tensor, tensor, broadcastable_native_tensors, NativeTensor, CollapsedTensor, TensorStack, \
    custom_op2, tensors
from phi.math.backend._scipy_backend import SCIPY_BACKEND


def is_tensor(x):
    return isinstance(x, Tensor)


def as_tensor(x, convert_external=True):
    if convert_external:
        return tensor(x)
    else:
        return x


def copy(tensor, only_mutable=False):
    raise NotImplementedError()


def print_(value: Tensor = None, name: str = None):
    """
    Print a tensor with no more than two spatial dimensions, splitting it along all batch and channel dimensions.

    Unlike regular printing, the primary axis, typically x, is oriented to the right.

    :param name: name of the tensor
    :param value: tensor-like
    """
    if value is None:
        print()
        return
    if name is not None:
        print(" " * 16 + name)
    value = tensor(value)
    axis_order = tuple(sorted(value.shape.spatial.names, reverse=True))
    if value.shape.spatial_rank == 0:
        print(value.numpy())
    elif value.shape.spatial_rank == 1:
        for index_dict in value.shape.non_spatial.meshgrid():
            if value.shape.non_spatial.volume > 1:
                print(f"--- {', '.join('%s=%d' % (name, idx) for name, idx in index_dict.items())} ---")
            text = np.array2string(value[index_dict].numpy(axis_order), precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', text))
    elif value.shape.spatial_rank == 2:
        for index_dict in value.shape.non_spatial.meshgrid():
            if value.shape.non_spatial.volume > 1:
                print(f"--- {', '.join('%s=%d' % (name, idx) for name, idx in index_dict.items())} ---")
            text = np.array2string(value[index_dict].numpy(axis_order), precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', re.sub('\],', '', text)))
    else:
        raise NotImplementedError('Can only print tensors with up to 2 spatial dimensions.')


def transpose(tensor, axes):
    if isinstance(tensor, Tensor):
        return CollapsedTensor(tensor, tensor.shape[axes])
    else:
        return math.transpose(tensor, axes)


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

    :param shape: base tensor shape
    :param dtype: data type
    :param dimensions: additional dimensions, types are determined from names
    :return: tensor of specified shape
    """
    return _initialize(lambda shape, dtype: CollapsedTensor(NativeTensor(math.zeros((), dtype=dtype), EMPTY_SHAPE), shape), shape, dtype, **dimensions)


def ones(shape=EMPTY_SHAPE, dtype=None, **dimensions):
    """
    Define a tensor with specified shape with value 1 / True everywhere.

    This method may not immediately allocate the memory to store the values.

    :param shape: base tensor shape
    :param dtype: data type
    :param dimensions: additional dimensions, types are determined from names
    :return: tensor of specified shape
    """
    return _initialize(lambda shape, dtype: CollapsedTensor(NativeTensor(math.ones((), dtype=dtype), EMPTY_SHAPE), shape), shape, dtype, **dimensions)


def random_normal(shape=EMPTY_SHAPE, dtype=None, **dimensions):

    def uniform_random_normal(shape, dtype):
        native = math.random_normal(shape.sizes)
        native = native if dtype is None else native.astype(dtype)
        return NativeTensor(native, shape)

    return _initialize(uniform_random_normal, shape, dtype, **dimensions)


def random_uniform(shape=EMPTY_SHAPE, dtype=None, **dimensions):

    def uniform_random_uniform(shape, dtype):
        native = math.random_uniform(shape.sizes)
        native = native if dtype is None else native.astype(dtype)
        return NativeTensor(native, shape)

    return _initialize(uniform_random_uniform, shape, dtype, **dimensions)


def fftfreq(resolution, dtype=None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    :param resolution: grid resolution measured in cells
    :param dtype: data type of the returned tensor
    :return: tensor holding the frequencies of the corresponding values computed by math.fft
    """
    resolution = spatial_shape(resolution)
    k = meshgrid(**{dim: np.fft.fftfreq(int(n)) for dim, n in resolution.named_sizes})
    return to_float(k) if dtype is None else cast(k, dtype)


def meshgrid(**dimensions):
    """generate a TensorStack meshgrid from keyword dimensions"""
    assert 'vector' not in dimensions
    dimensions = {dim: np.arange(val) if isinstance(val, int) else val for dim, val in dimensions.items()}
    indices_list = math.meshgrid(*dimensions.values())
    single_shape = Shape([len(val) for val in dimensions.values()], dimensions.keys(), [SPATIAL_DIM] * len(dimensions))
    channels = [NativeTensor(t, single_shape) for t in indices_list]
    return TensorStack(channels, 'vector', CHANNEL_DIM)


def channel_stack(values, axis: str):
    return _stack(values, axis, CHANNEL_DIM)


def batch_stack(values, axis: str = 'batch'):
    return _stack(values, axis, BATCH_DIM)


def spatial_stack(values, axis: str):
    return _stack(values, axis, SPATIAL_DIM)


def _stack(values, dim: str, dim_type: str):
    assert isinstance(dim, str)

    def inner_stack(*values):
        varying_shapes = any([v.shape != values[0].shape for v in values[1:]])
        from ._track import SparseLinearOperation
        tracking = any([isinstance(v, SparseLinearOperation) for v in values])
        inner_keep_separate = any([isinstance(v, TensorStack) and v.keep_separate for v in values])
        return TensorStack(values, dim, dim_type, keep_separate=varying_shapes or tracking or inner_keep_separate)

    result = broadcast_op(inner_stack, values)
    return result


def concat(values: tuple or list, dim: str) -> Tensor:
    """
    Concatenates a sequence of tensors along one axis.
    The shapes of all values must be equal, except for the size of the concat dimension.

    :param values: Tensors to concatenate
    :param dim: concat dimension, must be present in all values
    :return: concatenated tensor
    """
    broadcast_shape = values[0].shape
    tensors = [v.native(order=broadcast_shape.names) for v in values]
    concatenated = math.concat(tensors, broadcast_shape.index(dim))
    return NativeTensor(concatenated, broadcast_shape.with_sizes(math.staticshape(concatenated)))


def spatial_pad(value, pad_width: tuple or list, mode: 'extrapolation.Extrapolation') -> Tensor:
    value = tensor(value)
    return pad(value, {n: w for n, w in zip(value.shape.spatial.names, pad_width)}, mode=mode)


def pad(value: Tensor, widths: dict, mode: 'extrapolation.Extrapolation') -> Tensor:
    """
    Pads a tensor along the specified dimensions, determining the added values using the given extrapolation.

    This is equivalent to calling `mode.pad(value, widths)`.

    :param value: tensor to be padded
    :param widths: name: str -> (lower: int, upper: int)
    :param mode: Extrapolation object
    :return: padded Tensor
    """
    return mode.pad(value, widths)


def closest_grid_values(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    """
    Finds the neighboring grid points in all spatial directions and returns their values.
    The result will have 2^d values for each vector in coordiantes in d dimensions.

    :param extrap: grid extrapolation
    :param grid: grid data. The grid is spanned by the spatial dimensions of the tensor
    :param coordinates: tensor with 1 channel dimension holding vectors pointing to locations in grid index space
    :return: Tensor of shape (batch, coord_spatial, grid_spatial=(2, 2,...), grid_channel)
    """
    # alternative method: pad array for all 2^d combinations, then stack to simplify gather_nd.
    assert all(name not in grid.shape for name in coordinates.shape.spatial.names), 'grid and coordinates must have different spatial dimensions'
    # --- Pad tensor where transform is not possible ---
    non_copy_pad = {dim: (0 if extrap[dim, 0].is_copy_pad else 1, 0 if extrap[dim, 1].is_copy_pad else 1) for dim in grid.shape.spatial.names}
    grid = extrap.pad(grid, non_copy_pad)
    coordinates += [not extrap[dim, 0].is_copy_pad for dim in grid.shape.spatial.names]
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
        return spatial_stack([values_left, values_right], grid.shape.names[ax_idx])

    result = left_right(np.array([False] * grid.shape.spatial_rank), 0)
    return result


def grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    result = broadcast_op(functools.partial(_grid_sample, extrap=extrap), [grid, coordinates])
    return result


def _grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    coord_names = ['_coord_' + dim.name if dim.is_spatial else dim.name for dim in coordinates.shape.unstack()]
    coordinates = coordinates._with_shape_replaced(coordinates.shape.with_names(coord_names))
    neighbors = closest_grid_values(grid, coordinates, extrap)
    binary = meshgrid(**{dim: (0, 1) for dim in grid.shape.spatial.names})
    right_weights = coordinates % 1
    binary, right_weights = join_spaces(binary, right_weights)
    weights = prod(binary * right_weights + (1 - binary) * (1 - right_weights), 'vector')
    result = sum_(neighbors * weights, axis=grid.shape.spatial.names)
    result_names = [dim.name[7:] if dim.is_spatial else dim.name for dim in result.shape.unstack()]
    result = result._with_shape_replaced(result.shape.with_names(result_names))
    return result


def join_spaces(*tensors):
    spatial = functools.reduce(lambda s1, s2: s1.combined(s2, combine_spatial=True), [t.shape.spatial for t in tensors])
    return [CollapsedTensor(t, t.shape.non_spatial & spatial) for t in tensors]


def broadcast_op(operation, tensors, iter_dims: set or tuple or list = None):
    if iter_dims is None:
        iter_dims = set()
        for tensor in tensors:
            if isinstance(tensor, TensorStack) and tensor.keep_separate:
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
        return TensorStack(result_unstacked, dim, dim_type, keep_separate=True)


def reshape(value: Tensor, *operations: str):
    # '(x, y) -> list', 'batch -> (batch=5, group=2)'
    raise NotImplementedError()


def join_dimensions(value: Tensor, dims: Shape or tuple or list, joined_dim_name: str):
    order = value.shape.order_group(dims)
    native = value.native(order)
    types = value.shape.get_type(dims)
    dim_type = types[0] if len(set(types)) == 1 else BATCH_DIM
    first_dim_index = min(*value.shape.index(dims))
    new_shape = value.shape.without(dims).expand(value.shape.only(dims).volume, joined_dim_name, dim_type, pos=first_dim_index)
    native = math.reshape(native, new_shape.sizes)
    return NativeTensor(native, new_shape)


def prod(value, axis=None):
    if axis is None and isinstance(value, (tuple, list)) and all(isinstance(v, numbers.Number) for v in value):
        return SCIPY_BACKEND.prod(value)
    if isinstance(value, Tensor):
        native = math.prod(value.native(), value.shape.index(axis))
        return NativeTensor(native, value.shape.without(axis))
    raise NotImplementedError(f"{type(value)} not supported. Only Tensor allowed.")


def where(condition: Tensor or float or int, value_true: Tensor or float or int, value_false: Tensor or float or int):
    """
    Builds a tensor by choosing either values from `value_true` or `value_false` depending on `condition`.
    If `condition` is not of type boolean, non-zero values are interpreted as True.

    This function requires non-None values for `value_true` and `value_false`.
    To get the indices of True / non-zero values, use :func:`nonzero`.

    :param condition: determines where to choose values from value_true or from value_false
    :param value_true: values to pick where condition != 0 / True
    :param value_false: values to pick where condition == 0 / False
    :return: tensor containing dimensions of all inputs
    """
    condition, value_true, value_false = tensors(condition, value_true, value_false)
    shape, (c, vt, vf) = broadcastable_native_tensors(condition, value_true, value_false)
    result = math.where(c, vt, vf)
    return NativeTensor(result, shape)


def nonzero(value: Tensor, list_dim='nonzero', index_dim='vector'):
    """
    Get spatial indices of non-zero / True values.

    Batch dimensions are preserved by this operation.
    If channel dimensions are present, this method returns the indices where any entry is nonzero.

    :param value: spatial tensor to find non-zero / True values in.
    :param list_dim: name of dimension listing non-zero values
    :param index_dim: name of index dimension
    :return: tensor of shape (batch dims..., list_dim=#non-zero, index_dim=value.shape.spatial_rank)
    """
    if value.shape.channel_rank > 0:
        value = sum_(abs(value), value.shape.channel.names)

    def unbatched_nonzero(value):
        indices = math.nonzero(value.native())
        indices_shape = Shape(math.staticshape(indices), (list_dim, index_dim), (BATCH_DIM, CHANNEL_DIM))
        return NativeTensor(indices, indices_shape)

    return broadcast_op(unbatched_nonzero, [value], iter_dims=value.shape.batch.names)


def sum_(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.sum)


def _reduce(value: Tensor or list or tuple, axis, native_function):
    if axis in ((), [], EMPTY_SHAPE):
        return value
    if isinstance(value, (tuple, list)):
        values = [tensor(v) for v in value]
        value = _stack(values, '_reduce', BATCH_DIM)
        if axis is None:
            pass  # continue below
        elif axis == 0:
            axis = '_reduce'
        else:
            raise ValueError('axis must be 0 or None when passing a sequence of tensors')
    else:
        value = tensor(value)
    axes = _axis(axis, value.shape)
    if isinstance(value, NativeTensor):
        result = native_function(value.native(), axis=value.shape.index(axes))
        return NativeTensor(result, value.shape.without(axes))
    elif isinstance(value, TensorStack):
        # --- inner reduce ---
        inner_axes = [ax for ax in axes if ax != value.stack_dim_name]
        red_inners = [_reduce(t, inner_axes, native_function) for t in value.tensors]
        # --- outer reduce ---
        from ._track import ShiftLinOp, sum_operators
        if value.stack_dim_name in axes:
            if any([isinstance(t, ShiftLinOp) for t in red_inners]):
                return sum(red_inners[1:], red_inners[0])
            natives = [t.native() for t in red_inners]
            result = native_function(natives, axis=0)
            return NativeTensor(result, red_inners[0].shape)
        else:
            return TensorStack(red_inners, value.stack_dim_name, value.stack_dim_type, keep_separate=value.keep_separate)
    else:
        raise NotImplementedError(f"{type(value)} not supported. Only (NativeTensor, TensorStack) allowed.")


def _axis(axis, shape: Shape):
    if axis is None:
        return shape.names
    if isinstance(axis, (tuple, list)):
        return axis
    if isinstance(axis, (str, int)):
        return [axis]
    if isinstance(axis, Shape):
        return axis.names
    raise ValueError(axis)


def mean(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.mean)


def std(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.std)


def any_(boolean_tensor: Tensor or list or tuple, axis=None):
    return _reduce(boolean_tensor, axis, math.any)


def all_(boolean_tensor: Tensor or list or tuple, axis=None):
    return _reduce(boolean_tensor, axis, math.all)


def max_(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.max)


def min_(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.min)


def zeros_like(tensor: Tensor):
    return zeros(tensor.shape, dtype=tensor.dtype)


def ones_like(tensor: Tensor):
    return zeros(tensor.shape, dtype=tensor.dtype) + 1


def dot(a, b, axes):
    raise NotImplementedError()


def matmul(A, b):
    raise NotImplementedError()


def einsum(equation, *tensors):
    raise NotImplementedError()


def abs(x: Tensor):
    return x._op1(math.abs)


def sign(x: Tensor):
    return x._op1(math.sign)


def round(x: Tensor):
    return x._op1(math.round)


def ceil(x: Tensor):
    return x._op1(math.ceil)


def floor(x: Tensor):
    return x._op1(math.floor)


def divide_no_nan(x, y):
    return custom_op2(x, y, divide_no_nan, math.divide_no_nan, lambda y_, x_: divide_no_nan(x_, y_), lambda y_, x_: math.divide_no_nan(x_, y_))


def maximum(x, y):
    return custom_op2(x, y, maximum, math.maximum)


def minimum(x, y):
    return custom_op2(x, y, minimum, math.minimum)


def clip(x, minimum, maximum):
    def _clip(x, minimum, maximum):
        new_shape, (x_, min_, max_) = broadcastable_native_tensors(*tensors(x, minimum, maximum))
        result_tensor = math.clip(x_, min_, max_)
        return NativeTensor(result_tensor, new_shape)
    return broadcast_op(_clip, tensors(x, minimum, maximum))


def with_custom_gradient(function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
    raise NotImplementedError()


def sqrt(x):
    return tensor(x)._op1(math.sqrt)


def exp(x):
    return tensor(x)._op1(math.exp)


def conv(tensor, kernel, padding='same'):
    raise NotImplementedError()


def dim_sizes(value):
    return value.shape.sizes if isinstance(value, Tensor) else math.shape(value)


def ndims(value):
    return value.rank if isinstance(value, Tensor) else math.ndims(value)


def dim_sizes_static(value):
    if isinstance(value, Tensor):
        return value.shape.sizes
    else:
        return math.staticshape(value)


def to_float(x, float64=False):
    return tensor(x)._op1(partial(math.to_float, float64=float64))


def to_int(x, int64=False):
    return tensor(x)._op1(partial(math.to_int, int64=int64))


def to_complex(x):
    return tensor(x)._op1(math.to_complex)


def unstack(tensor, axis=0):
    assert isinstance(tensor, Tensor)
    return tensor.unstack(tensor.shape.names[axis])


def boolean_mask(x, mask):
    raise NotImplementedError()


def isfinite(x):
    return tensor(x)._op1(lambda t: math.isfinite(t))


def gather(value: Tensor, indices: Tensor):
    v_ = value.native()
    i_ = indices.native()
    if value.shape.channel_rank == 0:
        v_ = math.expand_dims(v_, -1)
    result = math.gather_nd(v_, i_, batch_dims=value.shape.batch_rank)
    if value.shape.channel_rank == 0:
        result = result[..., 0]
    new_shape = value.shape.batch & indices.shape.non_channel & value.shape.channel
    return NativeTensor(result, new_shape)


def scatter(indices: Tensor, values: Tensor, size: Shape, scatter_dims, duplicates_handling='undefined', outside_handling='discard'):
    """
    Create a dense tensor from sparse values.

    :param indices: n-dimensional indices corresponding to values
    :param values: values to scatter at indices
    :param size: spatial size of dense tensor
    :param scatter_dims: dimensions of values/indices to reduce during scattering
    :param duplicates_handling: one of ('undefined', 'add', 'mean', 'any')
    :param outside_handling: one of ('discard', 'clamp', 'undefined')
    """
    indices_ = indices.native()
    values_ = values.native(values.shape.combined(indices.shape.non_channel).names)
    result_ = math.scatter(indices_, values_, tuple(size), duplicates_handling=duplicates_handling, outside_handling=outside_handling)
    result_shape = size & indices.shape.batch & values.shape.non_spatial
    result_shape = result_shape.without(scatter_dims)
    return NativeTensor(result_, result_shape)


def fft(x: Tensor):
    """
    Performs a fast Fourier transform (FFT) on all spatial dimensions of x.

    The inverse operation is :func:`ifft`.

    :param x: tensor of type float or complex
    :return: FFT(x) of type complex
    """
    native, assemble = _invertible_standard_form(x)
    result = math.fft(native)
    return assemble(result)


def ifft(k: Tensor):
    native, assemble = _invertible_standard_form(k)
    result = math.ifft(native)
    return assemble(result)


def imag(complex):
    return complex._op1(math.imag)


def real(complex: Tensor):
    return complex._op1(math.real)


def cast(x: Tensor, dtype):
    return x._op1(partial(math.cast, dtype=dtype))


def sin(x):
    return tensor(x)._op1(math.sin)


def cos(x):
    return tensor(x)._op1(math.cos)


def dtype(x):
    if isinstance(x, Tensor):
        return x.dtype
    else:
        return math.dtype(x)


def tile(value, multiples):
    raise NotImplementedError()


def expand_channel(value, dim_size, dim_name):
    return _expand(value, dim_size, dim_name, CHANNEL_DIM)


def _expand(value: Tensor, dim_size: int, dim_name: str, dim_type: str):
    value = tensor(value)
    new_shape = value.shape.expand(dim_size, dim_name, dim_type)
    if isinstance(value, CollapsedTensor):
        return CollapsedTensor(value.tensor, new_shape)
    else:
        return CollapsedTensor(value, new_shape)


def sparse_tensor(indices, values, shape):
    raise NotImplementedError()


def _invertible_standard_form(value: Tensor):
    """
    Reshapes the tensor into the shape (batch, spatial..., channel) with a single batch and channel dimension.

    :param value: tensor to reshape
    :return: reshaped native tensor, inverse function
    """
    normal_order = value.shape.normal_order()
    native = value.native(normal_order.names)
    standard_form = (value.shape.batch.volume,) + value.shape.spatial.sizes + (value.shape.channel.volume,)
    reshaped = math.reshape(native, standard_form)

    def assemble(reshaped):
        un_reshaped = math.reshape(reshaped, math.shape(native))
        return NativeTensor(un_reshaped, normal_order)

    return reshaped, assemble


def close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks whether all tensors have equal values within the specified tolerance.

    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    :param tensors: tensor or tensor-like (constant) each
    :param rel_tolerance: relative tolerance
    :param abs_tolerance: absolute tolerance
    """
    tensors = [tensor(t) for t in tensors]
    for other in tensors[1:]:
        if not _close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance):
            return False
    return True


def _close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return True
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = math.numpy(native1)
    np2 = math.numpy(native2)
    return np.allclose(np1, np2, rel_tolerance, abs_tolerance)


def assert_close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks that all tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.

    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    :param tensors: tensor or tensor-like (constant) each
    :param rel_tolerance: relative tolerance
    :param abs_tolerance: absolute tolerance
    """
    any_tensor = next(filter(lambda t: isinstance(t, Tensor), tensors))
    if any_tensor is None:
        tensors = [tensor(t) for t in tensors]
    else:  # use Tensor to infer dimensions
        tensors = [any_tensor._tensor(t) for t in tensors]
    for other in tensors[1:]:
        _assert_close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)


def _assert_close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return
    if isinstance(tensor2, (int, float, bool)):
        np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = math.numpy(native1)
    np2 = math.numpy(native2)
    if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
        np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance)


def solve(operator, y: Tensor, x0: Tensor, solve_params: Solve, callback=None):
    if not isinstance(solve_params, LinearSolve):
        raise NotImplementedError("Only linear solve is currently supported. Pass a LinearSolve object")
    if solve_params.solver not in (None, 'CG'):
        raise NotImplementedError("Only 'CG' solver currently supported")

    from ._track import lin_placeholder, ShiftLinOp
    x0, y = tensors(x0, y)
    batch = (y.shape & x0.shape).batch
    x0_native = math.reshape(x0.native(), (x0.shape.batch.volume, x0.shape.non_batch.volume))
    y_native = math.reshape(y.native(), (y.shape.batch.volume, y.shape.non_batch.volume))
    if callable(operator):
        A_ = None
        if solve_params.solver_arguments['bake'] == 'sparse':
            build_time = time.time()
            x_track = lin_placeholder(x0)
            Ax_track = operator(x_track)
            assert isinstance(Ax_track, ShiftLinOp), 'Baking sparse matrix failed. Make sure only supported linear operations are used.'
            A_ = Ax_track.build_sparse_coordinate_matrix()
            # print_(tensor(A_.todense(), spatial_dims=2))
            # TODO reshape x0, y so that independent dimensions are batch
            print("CG: matrix build time: %s" % (time.time() - build_time))
        if A_ is None:
            def A_(native_x):
                native_x_shaped = math.reshape(native_x, x0.shape.non_batch.sizes)
                x = NativeTensor(native_x_shaped, x0.shape.non_batch)
                Ax = operator(x)
                Ax_native = math.reshape(Ax.native(), math.shape(native_x))
                return Ax_native
    else:
        A_ = math.reshape(operator.native(), (y.shape.non_batch.volume, x0.shape.non_batch.volume))

    cg_time = time.time()
    converged, x, iterations = math.conjugate_gradient(A_, y_native, x0_native, solve_params.relative_tolerance, solve_params.absolute_tolerance, solve_params.max_iterations, 'implicit', callback)
    print("CG: loop time: %s (%s iterations)" % (time.time() - cg_time, iterations))
    converged = math.reshape(converged, batch.sizes)
    x = math.reshape(x, batch.sizes + x0.shape.non_batch.sizes)
    iterations = math.reshape(iterations, batch.sizes)
    return NativeTensor(converged, batch), NativeTensor(x, batch.combined(x0.shape.non_batch)), NativeTensor(iterations, batch)

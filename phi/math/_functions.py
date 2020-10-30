import functools
import numbers
import re
import time
import warnings
from functools import partial

import numpy as np

from ._shape import BATCH_DIM, CHANNEL_DIM, SPATIAL_DIM, Shape, EMPTY_SHAPE, spatial_shape, shape_from_dict
from . import _extrapolation as extrapolation
from ._track import as_sparse_linear_operation, SparseLinearOperation, sum_operators
from .backend import math
from ._tensors import Tensor, tensor, broadcastable_native_tensors, NativeTensor, CollapsedTensor, TensorStack, combined_shape
from phi.math.backend._scipy_backend import SCIPY_BACKEND
from ._config import GLOBAL_AXIS_ORDER


def is_tensor(x):
    return isinstance(x, Tensor)


def as_tensor(x, convert_external=True):
    if convert_external:
        return tensor(x)
    else:
        return x


def copy(tensor, only_mutable=False):
    raise NotImplementedError()


def print_(value, name: str = None):
    """
    Print a tensor with no more than two spatial dimensions, splitting it along all batch and channel dimensions.

    Unlike regular printing, the primary axis, typically x, is oriented to the right.

    :param name: name of the tensor
    :param value: tensor-like
    """
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


def zeros(shape=EMPTY_SHAPE, dtype=None, **dimensions):
    """

    :param shape: base tensor shape
    :param dtype: data type
    :param dimensions: additional dimensions, types are determined from names: 'vector' -> channel, 'x','y','z' -> spatial, else batch
    :return:
    """
    shape &= shape_from_dict(dimensions)
    native_zero = math.zeros((), dtype=dtype)
    collapsed = NativeTensor(native_zero, EMPTY_SHAPE)
    return CollapsedTensor(collapsed, shape)


def ones(shape=EMPTY_SHAPE, dtype=None, **dimensions):
    """

    :param channels: int or (int,)
    :param batch: int or {name: size}
    :param dtype:
    :param spatial:
    :return:
    """
    shape &= shape_from_dict(dimensions)
    native_one = math.ones((), dtype=dtype)
    collapsed = NativeTensor(native_one, EMPTY_SHAPE)
    return CollapsedTensor(collapsed, shape)


def random_normal(shape=EMPTY_SHAPE, dtype=None, **dimensions):
    shape &= shape_from_dict(dimensions)
    native = math.random_normal(shape.sizes)
    native = native if dtype is None else native.astype(dtype)
    return NativeTensor(native, shape)


def fftfreq(resolution, dtype=None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    :param resolution: grid resolution measured in cells
    :param dtype: data type of the returned tensor
    :return: tensor holding the frequencies of the corresponding values computed by math.fft
    """
    resolution = spatial_shape(resolution)
    k = math.meshgrid(*[np.fft.fftfreq(int(n)) for n in resolution.sizes])
    k = [math.to_float(channel) if dtype is None else math.cast(channel, dtype) for channel in k]
    channel_shape = spatial_shape(k[0].shape)
    k = [NativeTensor(channel, channel_shape) for channel in k]
    return TensorStack(k, 'vector', CHANNEL_DIM)


def meshgrid(*coordinates, names=None):
    indices_list = math.meshgrid(*coordinates)
    single_shape = spatial_shape([len(coo) for coo in coordinates], names)
    channels = [NativeTensor(t, single_shape) for t in indices_list]
    return TensorStack(channels, 'vector', CHANNEL_DIM)


def channel_stack(values, axis: str):
    return _stack(values, axis, CHANNEL_DIM)


def batch_stack(values, axis: str = 'batch'):
    return _stack(values, axis, BATCH_DIM)


def spatial_stack(values, axis: str):
    return _stack(values, axis, SPATIAL_DIM)


def _stack(values, dim: str, dim_type: int):
    assert isinstance(dim, str)
    def inner_stack(*values):
        varying_shapes = any([v.shape != values[0].shape for v in values[1:]])
        tracking = any([isinstance(v, SparseLinearOperation) for v in values])
        inner_keep_separate = any([isinstance(v, TensorStack) and v.keep_separate for v in values])
        return TensorStack(values, dim, dim_type, keep_separate=varying_shapes or tracking or inner_keep_separate)

    result = broadcast_op(inner_stack, values)
    return result


def concat(values, axis):
    tensors = broadcastable_native_tensors(values)
    concatenated = math.concat(tensors, axis)
    return NativeTensor(concatenated, values[0].shape)


def spatial_pad(value, pad_width: tuple or list, mode: 'extrapolation.Extrapolation'):
    value = tensor(value)
    return pad(value, {n: w for n, w in zip(value.shape.spatial.names, pad_width)}, mode=mode)


def pad(value: Tensor, widths: dict, mode: 'extrapolation.Extrapolation'):
    """

    :param value: tensor to be padded
    :param widths: name: str -> (lower: int, upper: int)
    :param mode: extrapolation object
    :return:
    """
    return mode.pad(value, widths)

    value = tensor(value)
    if isinstance(value, NativeTensor):
        native = value.tensor
        ordered_pad_widths = value.shape.order(pad_width, default=0)
        ordered_mode = value.shape.order(mode, default=extrapolation.ZERO)
        result_tensor = math.pad(native, ordered_pad_widths, ordered_mode)
        new_shape = value.shape.with_sizes(math.staticshape(result_tensor))
        return NativeTensor(result_tensor, new_shape)
    elif isinstance(value, CollapsedTensor):
        inner = value.tensor
        inner_widths = {dim: w for dim, w in pad_width.items() if dim in inner.shape}
        if len(inner_widths) > 0:
            inner = pad(inner, pad_width, mode=mode)
        new_sizes = []
        for size, dim, dim_type in value.shape.dimensions:
            if dim not in pad_width:
                new_sizes.append(size)
            else:
                delta = sum_(pad_width[dim]) if isinstance(pad_width[dim], (tuple, list)) else 2 * pad_width[dim]
                new_sizes.append(size + int(delta))
        new_shape = value.shape.with_sizes(new_sizes)
        return CollapsedTensor(inner, new_shape)
    elif isinstance(value, SparseLinearOperation):
        return pad_operator(value, pad_width, mode)
    elif isinstance(value, TensorStack):
        if not value.requires_broadcast:
            return pad(value._cache())
        inner_widths = {dim: w for dim, w in pad_width.items() if dim != value.stack_dim_name}
        tensors = [pad(t, inner_widths, mode) for t in value.tensors]
        return TensorStack(tensors, value.stack_dim_name, value.stack_dim_type, value.keep_separate)
    else:
        raise NotImplementedError()


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
    grid_names = ['grid_' + dim.name if dim.is_spatial else dim.name for dim in grid.shape.unstack()]
    grid_names_sp = ['grid_' + dim.name for dim in grid.shape.spatial.unstack()]
    grid = grid._with_shape_replaced(grid.shape.with_names(grid_names))
    neighbors = closest_grid_values(grid, coordinates, extrap)
    binary = meshgrid(*[[0, 1]] * grid.shape.spatial_rank, names=grid_names_sp)
    right_weights = coordinates % 1
    binary, right_weights = join_spaces(binary, right_weights)
    weights = prod(binary * right_weights + (1 - binary) * (1 - right_weights), 'vector')
    return sum_(neighbors * weights, axis=grid.shape.spatial.names)


def join_spaces(*tensors):
    spatial = functools.reduce(lambda s1, s2: s1.combined(s2, combine_spatial=True), [t.shape.spatial for t in tensors])
    return [CollapsedTensor(t, t.shape.non_spatial & spatial) for t in tensors]


def broadcast_op(operation, tensors):
    non_atomic_dims = set()
    for tensor in tensors:
        if isinstance(tensor, TensorStack) and tensor.keep_separate:
            non_atomic_dims.add(tensor.stack_dim_name)
    if len(non_atomic_dims) == 0:
        return operation(*tensors)
    elif len(non_atomic_dims) == 1:
        dim = next(iter(non_atomic_dims))
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
            result_unstacked.append(operation(*gathered))
        return TensorStack(result_unstacked, dim, dim_type, keep_separate=True)
    else:
        raise NotImplementedError()


def reshape(value: Tensor, shape: Shape):
    native = value.native(shape.names)
    return NativeTensor(native, shape)


def prod(value, axis=None):
    if axis is None and isinstance(value, (tuple, list)) and all(isinstance(v, numbers.Number) for v in value):
        return SCIPY_BACKEND.prod(value)
    if isinstance(value, Tensor):
        native = math.prod(value.native(), value.shape.index(axis))
        return NativeTensor(native, value.shape.without(axis))
    raise NotImplementedError()


def divide_no_nan(x, y):
    x = tensor(x)
    return x._op2(y, lambda t1, t2: math.divide_no_nan(t1, t2))


def where(condition, x: Tensor, y: Tensor):
    condition = x._tensor(condition)
    shape, (c_, x_, y_) = broadcastable_native_tensors(condition, x, y)
    result = math.where(c_, x_, y_)
    return NativeTensor(result, shape)


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
        if value.stack_dim_name in axes:
            if any([isinstance(t, SparseLinearOperation) for t in red_inners]):
                return sum_operators(red_inners)  # TODO other functions
            natives = [t.native() for t in red_inners]
            result = native_function(natives, axis=0)
            return NativeTensor(result, red_inners[0].shape)
        else:
            return TensorStack(red_inners, value.stack_dim_name, value.stack_dim_type, keep_separate=value.keep_separate)
    else:
        raise NotImplementedError()


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


def max(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.max)


def min(value: Tensor or list or tuple, axis=None):
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


def maximum(a, b):
    a_, b_ = tensor(a, b)
    return a_._op2(b_, math.maximum)


def minimum(a, b):
    a_, b_ = tensor(a, b)
    return a_._op2(b_, math.minimum)


def clip(x, minimum, maximum):
    def _clip(x, minimum, maximum):
        new_shape, (x_, min_, max_) = broadcastable_native_tensors(*tensor(x, minimum, maximum))
        result_tensor = math.clip(x_, min_, max_)
        return NativeTensor(result_tensor, new_shape)
    return broadcast_op(_clip, tensor(x, minimum, maximum))


def with_custom_gradient(function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
    raise NotImplementedError()


def sqrt(x):
    return tensor(x)._op1(math.sqrt)


def exp(x):
    return tensor(x)._op1(math.exp)


def conv(tensor, kernel, padding='same'):
    raise NotImplementedError()


def shape(tensor):
    return tensor.shape.sizes if isinstance(tensor, Tensor) else math.shape(tensor)


def ndims(tensor):
    return tensor.rank if isinstance(tensor, Tensor) else math.ndims(tensor)


def staticshape(tensor):
    if isinstance(tensor, Tensor):
        return tensor.shape.sizes
    else:
        return math.staticshape(tensor)


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


def fft(x):
    raise NotImplementedError()


def ifft(k):
    native, assemble = _invertible_standard_form(k)
    result = math.ifft(native)
    return assemble(result)


def imag(complex):
    return complex._op1(math.imag)


def real(complex: Tensor):
    return complex._op1(math.real)


def cast(x: Tensor, dtype):
    return x._op1(lambda t: math.cast(x, dtype))


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


def expand_channel(x, dim_size, dim_name):
    x = tensor(x)
    shape = x.shape.expand(dim_size, dim_name, CHANNEL_DIM)
    return CollapsedTensor(x, shape)


def sparse_tensor(indices, values, shape):
    raise NotImplementedError()


def _invertible_standard_form(tensor: Tensor):
    normal_order = tensor.shape.normal_order()
    native = tensor.native(normal_order.names)
    standard_form = (tensor.shape.batch.volume,) + tensor.shape.spatial.sizes + (tensor.shape.channel.volume,)
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


def conjugate_gradient(operator, y, x0, relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, gradient: str = 'implicit', callback=None, bake='sparse'):
    x0, y = tensor(x0, y)
    batch = combined_shape(y, x0).batch
    x0_native = math.reshape(x0.native(), (x0.shape.batch.volume, x0.shape.non_batch.volume))
    y_native = math.reshape(y.native(), (y.shape.batch.volume, y.shape.non_batch.volume))
    if callable(operator):
        A_ = None
        if bake == 'sparse':
            build_time = time.time()
            x_track = as_sparse_linear_operation(x0)
            try:
                Ax_track = operator(x_track)
                if isinstance(Ax_track, SparseLinearOperation):
                    A_ = Ax_track.dependency_matrix
                    print("CG: matrix build time: %s" % (time.time() - build_time))
                else:
                    warnings.warn("Could not create matrix for conjugate_gradient() because non-linear operations were used.")
            except NotImplementedError as err:
                warnings.warn("Could not create matrix for conjugate_gradient():\n%s" % err)
                raise err
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
    converged, x, iterations = math.conjugate_gradient(A_, y_native, x0_native, relative_tolerance, absolute_tolerance, max_iterations, gradient, callback)
    print("CG: loop time: %s (%s iterations)" % (time.time() - cg_time, iterations))
    converged = math.reshape(converged, batch.sizes)
    x = math.reshape(x, batch.sizes + x0.shape.non_batch.sizes)
    iterations = math.reshape(iterations, batch.sizes)
    return NativeTensor(converged, batch), NativeTensor(x, batch.combined(x0.shape.non_batch)), NativeTensor(iterations, batch)

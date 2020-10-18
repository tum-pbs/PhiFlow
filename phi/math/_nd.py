# Because division is different in Python 2 and 3
from __future__ import division

import warnings

import numpy as np

from phi import struct
from . import _extrapolation as extrapolation
from ._extrapolation import Extrapolation
from . import _tensor_math as math
from ._shape import CHANNEL_DIM, spatial_shape, channel_shape
from ._tensor_math import broadcast_op, batch_stack, channel_stack
from ._tensors import tensor, Tensor, TensorStack, NativeTensor
from ._helper import _get_pad_width, _contains_axis, _multi_roll


def indices_tensor(tensor, dtype=None):
    """
    Returns an index tensor of the same spatial shape as the given tensor.
    Each index denotes the location within the tensor starting from zero.
    Indices are encoded as vectors in the index tensor.

    :param tensor: a tensor of shape (batch size, spatial dimensions..., component size)
    :param dtype: NumPy data type or `None` for default
    :return: an index tensor of shape (1, spatial dimensions..., spatial rank)
    """
    spatial_dimensions = list(tensor.shape[1:-1])
    idx_zyx = math.meshgrid(*[range(dim) for dim in spatial_dimensions])
    idx = np.stack(idx_zyx, axis=-1).reshape([1, ] + spatial_dimensions + [len(spatial_dimensions)])
    if dtype is not None:
        return idx.astype(dtype)
    else:
        return math.to_float(idx)


def normalize_to(target: Tensor, source: Tensor, epsilon=1e-5):
    """
    Multiplies the target so that its total content matches the source.

    :param target: a tensor
    :param source: a tensor or number
    :param epsilon: small number to prevent division by zero or None.
    :return: normalized tensor of the same shape as target
    """
    target_total = math.sum_(target, axis=target.shape.non_batch.names)
    denominator = math.maximum(target_total, epsilon) if epsilon is not None else target_total
    source_total = math.sum_(source, axis=source.shape.non_batch.names)
    return target * (source_total / denominator)


def l1_loss(tensor, batch_norm=True, reduce_batches=True):
    if struct.isstruct(tensor):
        all_tensors = struct.flatten(tensor)
        return sum(l1_loss(tensor, batch_norm, reduce_batches) for tensor in all_tensors)
    if reduce_batches:
        total_loss = math.sum_(math.abs(tensor))
    else:
        total_loss = math.sum_(math.abs(tensor), axis=list(range(1, len(tensor.shape))))
    if batch_norm and reduce_batches:
        batch_size = math.shape(tensor)[0]
        return math.div(total_loss, math.to_float(batch_size))
    else:
        return total_loss


def l2_loss(tensor, batch_norm=True):
    return l_n_loss(tensor, 2, batch_norm=batch_norm)


def l_n_loss(tensor, n, batch_norm=True):
    if struct.isstruct(tensor):
        all_tensors = struct.flatten(tensor)
        return sum(l_n_loss(tensor, n, batch_norm) for tensor in all_tensors)
    total_loss = math.sum_(tensor ** n) / n
    if batch_norm:
        batch_size = math.shape(tensor)[0]
        return math.div(total_loss, math.to_float(batch_size))
    else:
        return total_loss


def frequency_loss(tensor, frequency_falloff=100, reduce_batches=True):
    """
    Instead of minimizing each entry of the tensor, minimize the frequencies of the tensor, emphasizing lower frequencies over higher ones.

    :param reduce_batches: whether to reduce the batch dimension of the loss by adding the losses along the first dimension
    :param tensor: typically actual - target
    :param frequency_falloff: large values put more emphasis on lower frequencies, 1.0 weights all frequencies equally.
    :return: scalar loss value
    """
    if struct.isstruct(tensor):
        all_tensors = struct.flatten(tensor)
        return sum(frequency_loss(tensor, frequency_falloff, reduce_batches) for tensor in all_tensors)
    diff_fft = abs_square(math.fft(tensor))
    k = fftfreq(tensor.shape[1:-1], mode='absolute')
    weights = math.exp(-0.5 * k ** 2 * frequency_falloff ** 2)
    return l1_loss(diff_fft * weights, reduce_batches=reduce_batches)


def abs_square(complex):
    return math.imag(complex) ** 2 + math.real(complex) ** 2


# Divergence

# def divergence(tensor, dx=1, difference='central', padding='constant', axes=None):
#     """
#     Computes the spatial divergence of a vector channel from finite differences.
#
#     :param tensor: vector field; tensor of shape (batch size, spatial dimensions..., spatial rank)
#     :param dx: distance between adjacent grid points (default 1)
#     :param difference: type of difference, one of ('forward', 'central') (default 'forward')
#     :return: tensor of shape (batch size, spatial dimensions..., 1)
#     """
#     assert difference in ('central', 'forward', 'backward'), difference
#     rank = spatial_rank(tensor)
#     if difference == 'forward':
#         return _divergence_nd(tensor, padding, (0, 1), axes) / dx ** rank  # TODO why dx^rank?
#     elif difference == 'backward':
#         return _divergence_nd(tensor, padding, (-1, 0), axes) / dx ** rank
#     else:
#         return _divergence_nd(tensor, padding, (-1, 1), axes) / (2 * dx) ** rank
#
#
# def _divergence_nd(x_, padding, relative_shifts, axes=None):
#     x = tensor(x_)
#     assert x.shape.channel.rank == 1
#     axes = axes if axes is not None else x.shape.spatial.names
#     x = math.pad(x, {axis: (-relative_shifts[0], relative_shifts[1]) for axis in axes}, mode=padding)
#     components = []
#     for dimension in axes:
#         dim_index_in_spatial = x.shape.spatial.reset_indices().index(dimension)
#         lower, upper = _multi_roll(x, dimension, relative_shifts, diminish_others=(-relative_shifts[0], relative_shifts[1]), names=axes, base_selection={0: rank - dimension - 1})
#         components.append(upper - lower)
#     return math.sum_(components, 0)


def shift(x: Tensor, offsets: tuple, axes: tuple or None = None, padding: Extrapolation or None = extrapolation.BOUNDARY, stack_dim='shift') -> list:
    x = tensor(x)
    axes = axes if axes is not None else x.shape.spatial.names
    pad_lower = max(0, -min(offsets))
    pad_upper = max(0, max(offsets))
    if padding is not None:
        x = math.pad(x, {axis: (pad_lower, pad_upper) for axis in axes}, mode=padding)
    offset_tensors = []
    for offset in offsets:
        components = []
        for dimension in axes:
            slices = {dim: slice(pad_lower + offset, -pad_upper + offset) if dim == dimension else slice(pad_lower, -pad_upper) for dim in axes}
            slices = {dim: slice(sl.start, sl.stop if sl.stop < 0 else None) for dim, sl in slices.items()}  # replace stop=0 by stop=None
            components.append(x[slices])
        offset_tensors.append(channel_stack(components, stack_dim))
    return offset_tensors


# Gradient

def gradient(tensor, dx=1, difference='central', padding=extrapolation.BOUNDARY, axes=None):
    """
    Calculates the gradient of a scalar channel from finite differences.
    The gradient vectors are in reverse order, lowest dimension first.

    :param axes: (optional) sequence of dimension names
    :param tensor: channel with shape (batch_size, spatial_dimensions..., 1)
    :param dx: physical distance between grid points (default 1)
    :param difference: type of difference, one of ('forward', 'backward', 'central') (default 'forward')
    :param padding: tensor padding mode
    :return: tensor of shape (batch_size, spatial_dimensions..., spatial rank)
    """
    if difference.lower() == 'central':
        return _gradient_nd(tensor, padding, (-1, 1), axes) / (dx * 2)
    elif difference.lower() == 'forward':
        return _gradient_nd(tensor, padding, (0, 1), axes) / dx
    elif difference.lower() == 'backward':
        return _gradient_nd(tensor, padding, (-1, 0), axes) / dx
    else:
        raise ValueError('Invalid difference type: {}. Can be CENTRAL or FORWARD'.format(difference))


def _gradient_nd(x, padding, relative_shifts, axes):
    left, right = shift(tensor(x), relative_shifts, axes, padding, stack_dim='gradient')
    return right - left


# Laplace

def laplace(x, dx=1, padding=extrapolation.BOUNDARY, axes=None):
    """
    Spatial Laplace operator as defined for scalar fields.
    If a vector field is passed, the laplace is computed component-wise.

    :param x: n-dimensional field of shape (batch, spacial dimensions..., components)
    :param dx: scalar or 1d tensor
    :param padding: extrapolation
    :type padding: Ex
    :param axes: The second derivative along these axes is summed over
    :type axes: list
    :return: tensor of same shape
    """
    dx = tensor(dx)
    if isinstance(x, Extrapolation):
        return x.gradient()
    left, center, right = shift(tensor(x), (-1, 0, 1), axes, padding, stack_dim='_laplace')
    result = (left + right - 2 * center) / tensor(dx, names='_laplace')
    result = math.sum_(result, '_laplace')
    return result


def fourier_laplace(tensor, times=1):
    """
Applies the spatial laplce operator to the given tensor with periodic boundary conditions.

*Note:* The results of `fourier_laplace` and `laplace` are close but not identical.

This implementation computes the laplace operator in Fourier space.
The result for periodic fields is exact, i.e. no numerical instabilities can occur, even for higher-order derivatives.
    :param tensor: tensor, assumed to have periodic boundary conditions
    :param times: number of times the laplace operator is applied. The computational cost is independent of this parameter.
    :return: tensor of same shape as `tensor`
    """
    frequencies = math.fft(math.to_complex(tensor))
    k = fftfreq(math.staticshape(tensor)[1:-1], mode='square')
    fft_laplace = -(2 * np.pi)**2 * k
    return math.real(math.ifft(frequencies * fft_laplace ** times))


def fourier_poisson(tensor, times=1):
    """ Inverse operation to `fourier_laplace`. """
    frequencies = math.fft(math.to_complex(tensor))
    k = fftfreq(math.staticshape(tensor)[1:-1], mode='square')
    fft_laplace = -(2 * np.pi)**2 * k
    fft_laplace[(0,) * math.ndims(k)] = np.inf
    return math.cast(math.real(math.ifft(math.divide_no_nan(frequencies, fft_laplace**times))), math.dtype(tensor))


# Downsample / Upsample

def downsample2x(tensor, interpolation='linear', axes=None):
    if struct.isstruct(tensor):
        return struct.map(lambda s: downsample2x(s, interpolation, axes),
                          tensor, recursive=False)

    if interpolation.lower() != 'linear':
        raise ValueError('Only linear interpolation supported')
    rank = spatial_rank(tensor)
    if axes is None:
        axes = range(rank)
    tensor = math.pad(tensor,
                      [[0, 0]]
                      + [([0, 1] if (dim % 2) != 0 and _contains_axis(axes, ax, rank) else [0, 0]) for ax, dim in enumerate(tensor.shape[1:-1])]
                      + [[0, 0]], 'replicate')
    for axis in axes:
        upper_slices = tuple([(slice(1, None, 2) if i == axis else slice(None)) for i in range(rank)])
        lower_slices = tuple([(slice(0, None, 2) if i == axis else slice(None)) for i in range(rank)])
        tensor_sum = tensor[(slice(None),) + upper_slices + (slice(None),)] + tensor[(slice(None),) + lower_slices + (slice(None),)]
        tensor = tensor_sum / 2
    return tensor


def upsample2x(tensor, interpolation='linear'):
    if struct.isstruct(tensor):
        return struct.map(lambda s: upsample2x(s, interpolation), tensor, recursive=False)

    if interpolation.lower() != 'linear':
        raise ValueError('Only linear interpolation supported')
    dims = range(spatial_rank(tensor))
    vlen = tensor.shape[-1]
    spatial_dims = tensor.shape[1:-1]
    rank = spatial_rank(tensor)
    tensor = math.pad(tensor, _get_pad_width(rank), 'replicate')
    for dim in dims:
        lower, center, upper = _dim_shifted(tensor, dim, (-1, 0, 1))
        combined = math.stack([0.25 * lower + 0.75 * center, 0.75 * center + 0.25 * upper], axis=2 + dim)
        tensor = math.reshape(combined, [-1] + [spatial_dims[dim] * 2 if i == dim else tensor.shape[i + 1] for i in dims] + [vlen])
    return tensor


def spatial_sum(tensor):
    summed = math.sum_(tensor, axis=math.dimrange(tensor))
    for i in math.dimrange(tensor):
        summed = math.expand_dims(summed, i)
    return summed


def interpolate_linear(tensor: Tensor, start, size):
    for sta, siz, dim in zip(start, size, tensor.shape.spatial.names):
        tensor = tensor.dimension(dim)[int(sta):int(sta)+siz + (1 if sta != 0 else 0)]
    upper_weight = start % 1
    lower_weight = 1 - upper_weight
    for i, dimension in enumerate(tensor.shape.spatial.names):
        if upper_weight[i] not in (0, 1):
            lower, upper = _multi_roll(tensor, dimension, (0, 1), names=tensor.shape.spatial.names)
            tensor = upper * upper_weight[i] + lower * lower_weight[i]
    return tensor


def vec_abs(tensor: Tensor):
    return math.sqrt(math.sum_(tensor ** 2, axis=tensor.shape.channel.names))


def vec_squared(tensor: Tensor):
    return math.sum_(tensor ** 2, axis=tensor.shape.channel.names)

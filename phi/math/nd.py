# Because division is different in Python 2 and 3
from __future__ import division

import numpy as np

from phi import struct
from phi.struct.tensorop import collapsed_gather_nd
from .base_backend import DYNAMIC_BACKEND as math


def spatial_rank(tensor):
    """ The spatial rank of a tensor is ndims - 2. """
    return math.ndims(tensor) - 2


def spatial_dimensions(obj):
    return tuple(range(1, len(math.staticshape(obj)) - 1))


def axes(obj):
    return tuple(range(len(math.staticshape(obj)) - 2))


def all_dimensions(tensor):
    return range(len(math.staticshape(tensor)))


def is_scalar(obj):
    return len(math.staticshape(obj)) == 0


def indices_tensor(tensor, dtype=np.float32):
    """
    Returns an index tensor of the same spatial shape as the given tensor.
    Each index denotes the location within the tensor starting from zero.
    Indices are encoded as vectors in the index tensor.

    :param tensor: a tensor of shape (batch size, spatial dimensions..., component size)
    :param dtype: a numpy data type (default float32)
    :return: an index tensor of shape (1, spatial dimensions..., spatial rank)
    """
    spatial_dimensions = list(tensor.shape[1:-1])
    idx_zyx = np.meshgrid(*[range(dim) for dim in spatial_dimensions], indexing='ij')
    idx = np.stack(idx_zyx, axis=-1).reshape([1, ] + spatial_dimensions + [len(spatial_dimensions)])
    return idx.astype(dtype)


def normalize_to(target, source=1, epsilon=1e-5, batch_dims=1):
    """
    Multiplies the target so that its total content matches the source.

    :param target: a tensor
    :param source: a tensor or number
    :param epsilon: small number to prevent division by zero or None.
    :return: normalized tensor of the same shape as target
    """
    target_total = math.sum(target, axis=tuple(range(batch_dims, math.ndims(target))), keepdims=True)
    denominator = math.maximum(target_total, epsilon) if epsilon is not None else target_total
    source_total = math.sum(source, axis=tuple(range(batch_dims, math.ndims(source))), keepdims=True)
    return target * (source_total / denominator)


def batch_align(tensor, innate_dims, target, convert_to_same_backend=True):
    if isinstance(tensor, (tuple, list)):
        return [batch_align(t, innate_dims, target) for t in tensor]
    # --- Convert type ---
    if convert_to_same_backend:
        backend = math.choose_backend([tensor, target])
        tensor = backend.as_tensor(tensor)
        target = backend.as_tensor(target)
    # --- Batch align ---
    ndims = len(math.staticshape(tensor))
    if ndims <= innate_dims:
        return tensor  # There is no batch dimension
    target_ndims = len(math.staticshape(target))
    assert target_ndims >= ndims
    if target_ndims == ndims:
        return tensor
    return math.expand_dims(tensor, axis=-innate_dims-1, number=target_ndims - ndims)


def batch_align_scalar(tensor, innate_spatial_dims, target):
    if math.staticshape(tensor)[-1] != 1:
        tensor = math.expand_dims(tensor, -1)
    result = batch_align(tensor, innate_spatial_dims+1, target)
    return result


def blur(field, radius, cutoff=None, kernel="1/1+x"):
    """
Warning: This function can cause NaN in the gradients, reason unknown.

Runs a blur kernel over the given tensor.
    :param field: tensor
    :param radius: weight function curve scale
    :param cutoff: kernel size
    :param kernel: Type of blur kernel (str). Must be in ('1/1+x', 'gauss')
    :return:
    """
    if cutoff is None:
        cutoff = min(int(round(radius * 3)), *field.shape[1:-1])

    xyz = np.meshgrid(*[range(-int(cutoff), (cutoff)+1) for _ in field.shape[1:-1]])
    d = np.float32(np.sqrt(np.sum([x ** 2 for x in xyz], axis=0)))
    if kernel == "1/1+x":
        weights = np.float32(1) / ( d / radius + 1)
    elif kernel.lower() == "gauss":
        weights = math.exp(- d / radius / 2)
    else:
        raise ValueError("Unknown kernel: %s" % kernel)
    weights /= math.sum(weights)
    weights = math.reshape(weights, list(weights.shape) + [1, 1])
    return math.conv(field, weights)


def l1_loss(tensor, batch_norm=True, reduce_batches=True):
    if struct.isstruct(tensor):
        all_tensors = struct.flatten(tensor)
        return sum(l1_loss(tensor, batch_norm, reduce_batches) for tensor in all_tensors)
    if reduce_batches:
        total_loss = math.sum(math.abs(tensor))
    else:
        total_loss = math.sum(math.abs(tensor), axis=list(range(1, len(tensor.shape))))
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
    total_loss = math.sum(tensor ** n) / n
    if batch_norm:
        batch_size = math.shape(tensor)[0]
        return math.div(total_loss, math.to_float(batch_size))
    else:
        return total_loss


# Divergence

def divergence(vel, dx=1, difference='central'):
    """
    Computes the spatial divergence of a vector channel from finite differences.

    :param vel: tensor of shape (batch size, spatial dimensions..., spatial rank)
    :param dx: distance between adjacent grid points (default 1)
    :param difference: type of difference, one of ('forward', 'central') (default 'forward')
    :return: tensor of shape (batch size, spatial dimensions..., 1)
    """
    assert difference in ('central', 'forward')
    rank = spatial_rank(vel)
    if difference == 'forward':
        return _forward_divergence_nd(vel) / dx ** rank
    else:
        return _central_divergence_nd(vel) / (2 * dx) ** rank


def _forward_divergence_nd(field):
    rank = spatial_rank(field)
    dims = range(rank)
    components = []
    for dimension in dims:
        vq = field[...,rank-dimension-1]
        upper_slices = [(slice(1, None) if i == dimension else slice(None)) for i in dims]
        lower_slices = [(slice(-1)      if i == dimension else slice(None)) for i in dims]
        diff = vq[(slice(None),)+upper_slices] - vq[(slice(None),)+lower_slices]
        padded = math.pad(diff, [[0,0]] + [([0,1] if i == dimension else [0,0]) for i in dims])
        components.append(padded)
    return math.expand_dims(math.sum(components, 0), -1)


def _central_divergence_nd(tensor):
    rank = spatial_rank(tensor)
    dims = range(rank)
    components = []
    tensor = math.pad(tensor, [[0, 0]] + [[1, 1]]*rank + [[0, 0]])
    for dimension in dims:
        upper_slices = [(slice(2, None) if i == dimension else slice(1, -1)) for i in dims]
        lower_slices = [(slice(-2)      if i == dimension else slice(1, -1)) for i in dims]
        diff = tensor[(slice(None),) + upper_slices + [rank - dimension - 1]] \
             - tensor[(slice(None),) + lower_slices + [rank - dimension - 1]]
        components.append(diff)
    return math.expand_dims(math.sum(components, 0), -1)


# Gradient

def gradient(tensor, dx=1, difference='forward', padding='replicate'):
    """
    Calculates the gradient of a scalar channel from finite differences.
    The gradient vectors are in reverse order, lowest dimension first.

    :param tensor: channel with shape (batch_size, spatial_dimensions..., 1)
    :param dx: physical distance between grid points (default 1)
    :param difference: type of difference, one of ('forward', 'backward', 'central') (default 'forward')
    :return: tensor of shape (batch_size, spatial_dimensions..., spatial rank)
    """
    if tensor.shape[-1] != 1:
        raise ValueError('Gradient requires a scalar channel as input')
    dims = range(spatial_rank(tensor))
    field = tensor[..., 0]

    if 1 in field.shape[1:]:
        raise ValueError('All spatial dimensions must have size larger than 1, got {}'.format(tensor.shape))

    if difference.lower() == 'central':
        return _central_diff_nd(tensor, dims, padding) / (dx * 2)
    elif difference.lower() == 'forward':
        return _forward_diff_nd(field, dims, padding) / dx
    elif difference.lower() == 'backward':
        return _backward_diff_nd(field, dims, padding) / dx
    else:
        raise ValueError('Invalid difference type: {}. Can be CENTRAL or FORWARD'.format(difference))


def _backward_diff_nd(field, dims, padding):
    df_dq = []
    for dimension in dims:
        upper_slices = tuple([(slice(1, None) if i==dimension else slice(None)) for i in dims])
        lower_slices = tuple([(slice(-1)      if i==dimension else slice(None)) for i in dims])
        diff = field[(slice(None),)+upper_slices] - field[(slice(None),)+lower_slices]
        padded = math.pad(diff, [[0,0]]+[([1,0] if i == dimension else [0,0]) for i in dims], mode=padding)
        df_dq.append(padded)
    return math.stack(df_dq, axis=-1)


def _forward_diff_nd(field, dims, padding):
    df_dq = []
    for dimension in dims:
        upper_slices = tuple([(slice(1, None) if i==dimension else slice(None)) for i in dims])
        lower_slices = tuple([(slice(-1)      if i==dimension else slice(None)) for i in dims])
        diff = field[(slice(None),) + upper_slices] - field[(slice(None),) + lower_slices]
        padded = math.pad(diff, [[0,0]]+[([0,1] if i == dimension else [0,0]) for i in dims], mode=padding)
        df_dq.append(padded)
    return math.stack(df_dq, axis=-1)


def _central_diff_nd(field, dims, padding):
    field = math.pad(field, [[0,0]] + [[1,1]]*spatial_rank(field) + [[0, 0]], mode=padding)
    df_dq = []
    for dimension in dims:
        upper_slices = tuple([(slice(2, None) if i==dimension else slice(1,-1)) for i in dims])
        lower_slices = tuple([(slice(-2)      if i==dimension else slice(1,-1)) for i in dims])
        diff = field[(slice(None),) + upper_slices + (0,)] - field[(slice(None),) + lower_slices + (0,)]
        df_dq.append(diff)
    return math.stack(df_dq, axis=-1)


def axis_gradient(tensor, spatial_axis):
    dims = range(spatial_rank(tensor))
    upper_slices = tuple([(slice(1, None) if i == spatial_axis else slice(None)) for i in dims])
    lower_slices = tuple([(slice(-1) if i == spatial_axis else slice(None)) for i in dims])
    diff = tensor[(slice(None),) + upper_slices + (slice(None),)] \
         - tensor[(slice(None),) + lower_slices + (slice(None),)]
    return diff


# Laplace

def laplace(tensor, padding='replicate', axes=None):
    """
    Spatial Laplace operator as defined for scalar fields.
    If a vector field is passed, the laplace is computed component-wise.

    :param tensor: n-dimensional field of shape (batch, spacial dimensions..., components)
    :param padding: 'valid', 'constant', 'reflect', 'replicate', 'cyclic'
    :param axes: The second derivative along these axes is summed over
    :type axes: list
    :return: tensor of same shape
    """
    if padding.lower() == 'cyclic':
        return fourier_laplace(tensor)
    rank = spatial_rank(tensor)
    if padding.lower() in ('constant', 'reflect', 'replicate'):
        tensor = math.pad(tensor, [[0,0]] + [([1,1] if _contains_axis(axes, i, rank) else [0,0]) for i in range(rank)] + [[0,0]], padding)
    # --- convolutional laplace ---
    if axes is not None:
        return _sliced_laplace_nd(tensor, axes)
    if rank == 2:
        return _conv_laplace_2d(tensor)
    elif rank == 3:
        return _conv_laplace_3d(tensor)
    else:
        return _sliced_laplace_nd(tensor)


def _conv_laplace_2d(tensor):
    kernel = np.zeros((3, 3, 1, 1), np.float32)
    kernel[1, 1, 0, 0] = -4
    kernel[(0,1,1,2), (1,0,2,1), 0, 0] = 1
    if tensor.shape[-1] == 1:
        return math.conv(tensor, kernel, padding='VALID')
    else:
        return math.concat([math.conv(tensor[..., i:i+1], kernel, padding='VALID') for i in range(tensor.shape[-1])], -1)


def _conv_laplace_3d(tensor):
    kernel = np.zeros((3, 3, 3, 1, 1), np.float32)
    kernel[1, 1, 1, 0, 0] = -6
    kernel[(0,1,1,1,1,2), (1,0,2,1,1,1), (1,1,1,0,2,1), 0, 0] = 1
    if tensor.shape[-1] == 1:
        return math.conv(tensor, kernel, padding='VALID')
    else:
        return math.concat([math.conv(tensor[..., i:i+1], kernel, padding='VALID') for i in range(tensor.shape[-1])], -1)


def _sliced_laplace_nd(tensor, axes=None):
    # Laplace code for n dimensions
    rank = spatial_rank(tensor)
    dims = range(rank)
    components = []
    for ax in dims:
        if _contains_axis(axes, ax, rank):
            center_slices = tuple([(slice(1, -1) if i == ax else (slice(1,-1)) if _contains_axis(axes, i, rank) else slice(None)) for i in dims])
            upper_slices = tuple([(slice(2, None) if i == ax else (slice(1,-1)) if _contains_axis(axes, i, rank) else slice(None)) for i in dims])
            lower_slices = tuple([(slice(-2) if i == ax else (slice(1,-1)) if _contains_axis(axes, i, rank) else slice(None)) for i in dims])
            diff = tensor[(slice(None),) + upper_slices + (slice(None),)] \
                   + tensor[(slice(None),) + lower_slices + (slice(None),)] \
                   - 2 * tensor[(slice(None),) + center_slices + (slice(None),)]
            components.append(diff)
    return math.sum(components, 0)


def _contains_axis(axes, axis, sp_rank):
    assert -sp_rank <= axis < sp_rank
    return axes is None or axis in axes or axis+sp_rank in axes


def map_for_axes(function, obj, axes, rank):
    if axes is None:
        return function(obj)
    else:
        return [(function(collapsed_gather_nd(obj, i)) if _contains_axis(axes, i, rank) else collapsed_gather_nd(obj, i)) for i in range(rank)]


def fourier_laplace(tensor):
    frequencies = math.fft(math.to_complex(tensor))
    k = fftfreq(math.staticshape(tensor)[1:-1], mode='square')
    fft_laplace = -(2*np.pi)**2 * k
    return math.ifft(frequencies * fft_laplace)


def fftfreq(resolution, mode='vector', dtype=np.float32):
    assert mode in ('vector', 'absolute', 'square')
    k = np.meshgrid(*[np.fft.fftfreq(int(n)) for n in resolution], indexing='ij')
    k = math.expand_dims(math.stack(k, -1), 0)
    k = k.astype(dtype)
    if mode == 'vector':
        return k
    k = math.sum(k ** 2, axis=-1, keepdims=True)
    if mode == 'square':
        return k
    else:
        return math.sqrt(k)


# Downsample / Upsample

def downsample2x(tensor, interpolation='linear'):
    if struct.isstruct(tensor):
        return struct.map(lambda s: downsample2x(s, interpolation), tensor, recursive=False)

    if interpolation.lower() != 'linear':
        raise ValueError('Only linear interpolation supported')
    dims = range(spatial_rank(tensor))
    tensor = math.pad(tensor, [[0,0]]+
                          [([0, 1] if (dim % 2) != 0 else [0,0]) for dim in tensor.shape[1:-1]]
                          + [[0,0]], 'replicate')
    for dimension in dims:
        upper_slices = tuple([(slice(1, None, 2) if i==dimension else slice(None)) for i in dims])
        lower_slices = tuple([(slice(0, None, 2) if i==dimension else slice(None)) for i in dims])
        sum = tensor[(slice(None),)+upper_slices+(slice(None),)] + tensor[(slice(None),)+lower_slices+(slice(None),)]
        tensor = sum / 2
    return tensor


def upsample2x(tensor, interpolation='linear'):
    if struct.isstruct(tensor):
        return struct.map(lambda s: upsample2x(s, interpolation), tensor, recursive=False)

    if interpolation.lower() != 'linear':
        raise ValueError('Only linear interpolation supported')
    dims = range(spatial_rank(tensor))
    vlen = tensor.shape[-1]
    spatial_dims = tensor.shape[1:-1]
    tensor = math.pad(tensor, [[0, 0]] + [[1, 1]]*spatial_rank(tensor) + [[0, 0]], 'replicate')
    for dim in dims:
        left_slices_1 =  tuple([(slice(2, None) if i==dim else slice(None)) for i in dims])
        left_slices_2 =  tuple([(slice(1,-1)    if i==dim else slice(None)) for i in dims])
        right_slices_1 = tuple([(slice(1, -1)   if i==dim else slice(None)) for i in dims])
        right_slices_2 = tuple([(slice(-2)      if i==dim else slice(None)) for i in dims])
        left = 0.75 * tensor[(slice(None),)+left_slices_2+(slice(None),)] + 0.25 * tensor[(slice(None),)+left_slices_1+(slice(None),)]
        right = 0.25 * tensor[(slice(None),)+right_slices_2+(slice(None),)] + 0.75 * tensor[(slice(None),)+right_slices_1+(slice(None),)]
        combined = math.stack([right, left], axis=2+dim)
        tensor = math.reshape(combined, [-1] + [spatial_dims[dim] * 2 if i == dim else tensor.shape[i + 1] for i in dims] + [vlen])
    return tensor


def spatial_sum(tensor):
    summed = math.sum(tensor, axis=math.dimrange(tensor))
    for i in math.dimrange(tensor):
        summed = math.expand_dims(summed, i)
    return summed


def interpolate_linear(tensor, upper_weight, dimensions):
    """

    :param tensor:
    :param upper_weight: tensor of floats (leading dimensions must be 1) or nan to ignore interpolation along this axis
    :param dimensions: list or tuple of dimensions (first spatial axis=1) to be interpolated. Other axes are ignored.
    :return:
    """
    lower_weight = 1 - upper_weight
    for dimension in spatial_dimensions(tensor):
        if dimension in dimensions:
            upper_slices = tuple([(slice(1, None) if i == dimension else slice(None)) for i in all_dimensions(tensor)])
            lower_slices = tuple([(slice(-1) if i == dimension else slice(None)) for i in all_dimensions(tensor)])
            tensor = math.mul(tensor[upper_slices], upper_weight[...,dimension-1]) + math.mul(tensor[lower_slices], lower_weight[...,dimension-1])
    return tensor

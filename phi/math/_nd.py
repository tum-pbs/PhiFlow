# Because division is different in Python 2 and 3
from __future__ import division

import warnings

import numpy as np

from phi import struct
from . import _extrapolation as extrapolation
from ._extrapolation import Extrapolation
from . import _functions as math
from ._shape import CHANNEL_DIM, spatial_shape, channel_shape, Shape
from ._functions import broadcast_op, batch_stack, channel_stack
from ._tensors import tensor, Tensor, TensorStack, NativeTensor
from ._tensors import Tensor
from ._config import GLOBAL_AXIS_ORDER
from .backend.tensorop import collapsed_gather_nd


def spatial_sum(value: Tensor):
    return math.sum_(value, axis=value.shape.spatial.names)


def vec_abs(vec: Tensor):
    return math.sqrt(math.sum_(vec ** 2, axis=vec.shape.channel.names))


def vec_squared(vec: Tensor):
    return math.sum_(vec ** 2, axis=vec.shape.channel.names)


def cross_product(vec1: Tensor, vec2: Tensor):
    vec1, vec2 = math.tensor(vec1, vec2)
    spatial_rank = vec1.vector.size if 'vector' in vec1.shape else vec2.vector.size
    if spatial_rank == 2:  # Curl in 2D
        dist_0, dist_1 = vec2.vector.unstack()
        if GLOBAL_AXIS_ORDER.is_x_first:
            velocity = vec1 * math.channel_stack([-dist_1, dist_0], 'vector')
        else:
            velocity = vec1 * math.channel_stack([dist_1, -dist_0], 'vector')
        return velocity
    elif spatial_rank == 3:  # Curl in 3D
        raise NotImplementedError(f'spatial_rank={spatial_rank} not yet implemented')
    else:
        raise AssertionError(f'dims = {spatial_rank}. Vector product not available in > 3 dimensions')


def indices_tensor(tensor: Tensor, dtype=None):
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


def l1_loss(tensor: Tensor, batch_norm=True, reduce_batches=True):
    """get L1 loss"""
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


def l2_loss(tensor: Tensor, batch_norm=True):
    """get L2 loss"""
    return l_n_loss(tensor, 2, batch_norm=batch_norm)


def l_n_loss(tensor: Tensor, n: int, batch_norm=True):
    """get Ln loss"""
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
    k_squared = math.sum_(math.fftfreq(tensor.shape[1:-1]) ** 2, 'vector')
    weights = math.exp(-0.5 * k_squared * frequency_falloff ** 2)
    return l1_loss(diff_fft * weights, reduce_batches=reduce_batches)


def abs_square(complex):
    """get the square magnitude

    :param complex: complex input data
    :type complex: Tensor
    :return: real valued magnitude squared
    :rtype: Tensor
    """
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


def shift(x: Tensor,
          offsets: tuple,
          axes: tuple or None = None,
          padding: Extrapolation or None = extrapolation.BOUNDARY,
          stack_dim: str = 'shift') -> list:
    """shift Tensor by a fixed offset and abiding by extrapolation

    :param x: Input data
    :type x: Tensor
    :param offsets: Shift size
    :type offsets: tuple
    :param axes: Axes along which to shift, defaults to None
    :type axes: tuple or None, optional
    :param padding: padding to be performed at the boundary, defaults to extrapolation.BOUNDARY
    :type padding: Extrapolation or None, optional
    :param stack_dim: dimensions to be stacked, defaults to 'shift'
    :type stack_dim: str, optional
    :return: offset_tensor
    :rtype: list
    """
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

def gradient(grid: Tensor,
             dx: float or int = 1,
             difference: str = 'central',
             padding: Extrapolation = extrapolation.BOUNDARY,
             axes: tuple or None = None):
    """
    Calculates the gradient of a scalar channel from finite differences.
    The gradient vectors are in reverse order, lowest dimension first.

    :param axes: (optional) sequence of dimension names
    :type axes: integer, iterable of integers
    :param x: channel with shape (batch_size, spatial_dimensions..., 1)
    :param dx: physical distance between grid points (default 1)
    :param difference: type of difference, one of ('forward', 'backward', 'central') (default 'forward')
    :param padding: tensor padding mode
    :return: tensor of shape (batch_size, spatial_dimensions..., spatial rank)
    """
    grid = tensor(grid)
    if difference.lower() == 'central':
        left, right = shift(grid, (-1, 1), axes, padding, stack_dim='gradient')
        return (right - left) / (dx * 2)
    elif difference.lower() == 'forward':
        left, right = shift(grid, (0, 1), axes, padding, stack_dim='gradient')
        return (right - left) / dx
    elif difference.lower() == 'backward':
        left, right = shift(grid, (-1, 0), axes, padding, stack_dim='gradient')
        return (right - left) / dx
    else:
        raise ValueError('Invalid difference type: {}. Can be CENTRAL or FORWARD'.format(difference))


# Laplace

def laplace(x: Tensor,
            dx: float = 1,
            padding: Extrapolation = extrapolation.BOUNDARY,
            axes: tuple or None = None):
    """
    Spatial Laplace operator as defined for scalar fields.
    If a vector field is passed, the laplace is computed component-wise.

    :param x: n-dimensional field of shape (batch, spacial dimensions..., components)
    :param dx: scalar or 1d tensor
    :param padding: extrapolation
    :type padding: Extrapolation
    :param axes: The second derivative along these axes is summed over
    :type axes: list
    :return: tensor of same shape
    """
    if not isinstance(dx, (int, float)):
        dx = tensor(dx, names='_laplace')
    if isinstance(x, Extrapolation):
        return x.gradient()
    left, center, right = shift(tensor(x), (-1, 0, 1), axes, padding, stack_dim='_laplace')
    result = (left + right - 2 * center) / dx
    result = math.sum_(result, '_laplace')
    return result


def fourier_laplace(grid: Tensor,
                    dx: Tensor or Shape or float or list or tuple,
                    times=1):
    """
    Applies the spatial laplace operator to the given tensor with periodic boundary conditions.

    *Note:* The results of `fourier_laplace` and `laplace` are close but not identical.

    This implementation computes the laplace operator in Fourier space.
    The result for periodic fields is exact, i.e. no numerical instabilities can occur, even for higher-order derivatives.

    :param grid: tensor, assumed to have periodic boundary conditions
    :param dx: distance between grid points, tensor-like, scalar or vector
    :param times: number of times the laplace operator is applied. The computational cost is independent of this parameter.
    :return: tensor of same shape as `tensor`
    """
    frequencies = math.fft(math.to_complex(grid))
    k_squared = math.sum_(math.fftfreq(grid.shape) ** 2, 'vector')
    fft_laplace = -(2 * np.pi)**2 * k_squared
    result = math.real(math.ifft(frequencies * fft_laplace ** times))
    return math.cast(result / tensor(dx) ** 2, grid.dtype)


def fourier_poisson(grid: Tensor, dx, times=1):
    """ Inverse operation to `fourier_laplace`. """
    frequencies = math.fft(math.to_complex(grid))
    k_squared = math.sum_(math.fftfreq(grid.shape) ** 2, 'vector')
    fft_laplace = -(2 * np.pi)**2 * k_squared
    fft_laplace.tensor[(0,) * math.ndims(k_squared)] = np.inf  # assume NumPy array to edit
    result = math.real(math.ifft(math.divide_no_nan(frequencies, fft_laplace ** times)))
    return math.cast(result * tensor(dx) ** 2, grid.dtype)


# Downsample / Upsample

def downsample2x(tensor,
                 interpolation: str = 'linear',
                 axes: tuple or None = None) -> Tensor:
    """get half sized tensor using given interpolation method

    :param tensor: [description]
    :type tensor: [type]
    :param interpolation: [description], defaults to 'linear'
    :type interpolation: str, optional
    :param axes: axes along which this is applied, defaults to None
    :type axes: iterable, optional
    :raises ValueError: if interpolation != linear
    :return: Downsampled Tensor (half the size)
    :rtype: Tensor
    """
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
                      + [([0, 1] if (dim % 2) != 0 and _contains_axis(axes, ax, rank)
                          else [0, 0])
                          for ax, dim in enumerate(tensor.shape[1:-1])]
                      + [[0, 0]], 'replicate')
    for axis in axes:
        upper_slices = tuple([(slice(1, None, 2) if i == axis else slice(None)) for i in range(rank)])
        lower_slices = tuple([(slice(0, None, 2) if i == axis else slice(None)) for i in range(rank)])
        tensor_sum = tensor[(slice(None),) + upper_slices + (slice(None),)] + tensor[(slice(None),) + lower_slices + (slice(None),)]
        tensor = tensor_sum / 2
    return tensor


def upsample2x(tensor, interpolation='linear') -> Tensor:
    """get double sized tensor using given interpolation method

    :param tensor: Data to be upsampled
    :type tensor: Tensor
    :param interpolation: interpolation method, defaults to 'linear'
    :type interpolation: str, optional (only linear allowed)
    :raises ValueError: if not linear
    :return: Upsampled Tensor (double the size)
    :rtype: Tensor
    """
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


def interpolate_linear(tensor: Tensor, start, size):
    for sta, siz, dim in zip(start, size, tensor.shape.spatial.names):
        tensor = tensor.dimension(dim)[int(sta):int(sta) + siz + (1 if sta != 0 else 0)]
    upper_weight = start % 1
    lower_weight = 1 - upper_weight
    for i, dimension in enumerate(tensor.shape.spatial.names):
        if upper_weight[i] not in (0, 1):
            lower, upper = _multi_roll(tensor, dimension, (0, 1), names=tensor.shape.spatial.names)
            tensor = upper * upper_weight[i] + lower * lower_weight[i]
    return tensor


def _get_pad_width_axes(rank, axes, val_true=(1, 1), val_false=(0, 0)):
    mid_shape = []
    for i in range(rank):
        if _contains_axis(axes, i, rank):
            mid_shape.append(val_true)
        else:
            mid_shape.append(val_false)
    return [[0, 0]] + mid_shape + [[0, 0]]


def _get_pad_width(rank, axis_widths=(1, 1)):
    return [[0, 0]] + [axis_widths] * rank + [[0, 0]]


def _multi_roll(tensor, roll_name, relative_shifts, diminish_others=(0, 0), names=None, base_selection={}):
    assert isinstance(tensor, Tensor), tensor
    assert len(relative_shifts) >= 2
    total_shift = max(relative_shifts) - min(relative_shifts)
    slice_others = slice(diminish_others[0], -diminish_others[1] if diminish_others[1] != 0 else None)
    # --- Slice tensor to create shifts ---
    shifted_tensors = []
    for shift in relative_shifts:
        slices = dict(base_selection)
        for name in names:
            if name == roll_name:
                shift_start = shift - min(relative_shifts)
                shift_end = shift_start - total_shift
                if shift_end == 0:
                    shift_end = None
                slices[name] = slice(shift_start, shift_end)
            else:
                slices[name] = slice_others
        sliced_tensor = tensor[slices]
        shifted_tensors.append(sliced_tensor)
    return shifted_tensors


def _contains_axis(axes, axis, sp_rank):
    assert -sp_rank <= axis < sp_rank
    return (axes is None) or (axis in axes) or (axis + sp_rank in axes)


def map_for_axes(function, obj, axes: tuple or None, rank: int):
    """apply function to each axes contained in the object

    :param function: function to be applied
    :type function: function
    :param obj: object with axes
    :type obj: Tensor or similar
    :param axes: axes along which to apply the function
    :type axes: tuple or None
    :param rank: number of dimensions
    :type rank: int
    :return: applied function
    :rtype: object or list
    """
    if axes is None:
        return function(obj)
    else:
        return [(function(collapsed_gather_nd(obj, i)) if _contains_axis(axes, i, rank)
                 else collapsed_gather_nd(obj, i))
                for i in range(rank)]


# Poisson Brackets


def poisson_bracket(grid1, grid2):
    if all([grid1.rank == grid2.rank == 2,
            grid1.boundary == grid2.boundary == extrapolation.PERIODIC,
            len(set(list(grid1.dx) + list(grid2.dx))) == 1]):
        return _periodic_2d_arakawa_poisson_bracket(grid1.values, grid2.values, grid1.dx)
    else:
        raise NotImplementedError("\n".join[
            "Not implemented for:"
            f"ranks ({grid1.rank}, {grid2.rank}) != 2",
            f"boundary ({grid1.boundary}, {grid2.boundary}) != {extrapolation.PERIODIC}",
            f"dx uniform ({grid1.dx}, {grid2.dx})"
        ])


def _periodic_2d_arakawa_poisson_bracket(tensor1: Tensor, tensor2: Tensor, dx: float):
    """Solves the poisson bracket using the Arakawa Scheme [tensor1, tensor2]

    Only works in 2D, with equal spaced grids, and periodic boundary conditions

    :param tensor1: first field in the poisson bracket
    :type tensor1: Tensor
    :param tensor2: second field in the poisson bracket
    :type tensor2: Tensor
    :param dx: Grid size (equal in x-y)
    :type dx: float
    """
    zeta = math.pad(value=tensor1, widths={'x': (1, 1), 'y': (1, 1)}, mode=extrapolation.PERIODIC)
    psi = math.pad(value=tensor2, widths={'x': (1, 1), 'y': (1, 1)}, mode=extrapolation.PERIODIC)
    return (zeta.x[2:].y[1:-1] * (psi.x[1:-1].y[2:] - psi.x[1:-1].y[0:-2] + psi.x[2:].y[2:] - psi.x[2:].y[0:-2])
            - zeta.x[0:-2].y[1:-1] * (psi.x[1:-1].y[2:] - psi.x[1:-1].y[0:-2] + psi.x[0:-2].y[2:] - psi.x[0:-2].y[0:-2])
            - zeta.x[1:-1].y[2:] * (psi.x[2:].y[1:-1] - psi.x[0:-2].y[1:-1] + psi.x[2:].y[2:] - psi.x[0:-2].y[2:])
            + zeta.x[1:-1].y[0:-2] * (psi.x[2:].y[1:-1] - psi.x[0:-2].y[1:-1] + psi.x[2:].y[0:-2] - psi.x[0:-2].y[0:-2])
            + zeta.x[2:].y[0:-2] * (psi.x[2:].y[1:-1] - psi.x[1:-1].y[0:-2])
            + zeta.x[2:].y[2:] * (psi.x[1:-1].y[2:] - psi.x[2:].y[1:-1])
            - zeta.x[0:-2].y[2:] * (psi.x[1:-1].y[2:] - psi.x[0:-2].y[1:-1])
            - zeta.x[0:-2].y[0:-2] * (psi.x[0:-2].y[1:-1] - psi.x[1:-1].y[0:-2])) / (12 * dx**2)

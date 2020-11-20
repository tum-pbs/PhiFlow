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

# def divergence(tensor, dx=1, difference='central', padding='constant', dimensions=None):
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
#         return _divergence_nd(tensor, padding, (0, 1), dims) / dx ** rank  # TODO why dx^rank?
#     elif difference == 'backward':
#         return _divergence_nd(tensor, padding, (-1, 0), dims) / dx ** rank
#     else:
#         return _divergence_nd(tensor, padding, (-1, 1), dims) / (2 * dx) ** rank
#
#
# def _divergence_nd(x_, padding, relative_shifts, dims=None):
#     x = tensor(x_)
#     assert x.shape.channel.rank == 1
#     dims = dims if dims is not None else x.shape.spatial.names
#     x = math.pad(x, {axis: (-relative_shifts[0], relative_shifts[1]) for axis in dims}, mode=padding)
#     components = []
#     for dimension in dims:
#         dim_index_in_spatial = x.shape.spatial.reset_indices().index(dimension)
#         lower, upper = _multi_roll(x, dimension, relative_shifts, diminish_others=(-relative_shifts[0], relative_shifts[1]), names=dims, base_selection={0: rank - dimension - 1})
#         components.append(upper - lower)
#     return math.sum_(components, 0)


def shift(x: Tensor,
          offsets: tuple,
          dims: tuple or None = None,
          padding: Extrapolation or None = extrapolation.BOUNDARY,
          stack_dim: str or None = 'shift') -> list:
    """shift Tensor by a fixed offset and abiding by extrapolation

    :param x: Input data
    :param offsets: Shift size
    :param dims: Dimensions along which to shift, defaults to None
    :param padding: padding to be performed at the boundary, defaults to extrapolation.BOUNDARY
    :param stack_dim: dimensions to be stacked, defaults to 'shift'
    :return: offset_tensor
    :rtype: list
    """
    if stack_dim is None:
        assert len(dims) == 1
    x = tensor(x)
    dims = dims if dims is not None else x.shape.spatial.names
    pad_lower = max(0, -min(offsets))
    pad_upper = max(0, max(offsets))
    if padding is not None:
        x = math.pad(x, {axis: (pad_lower, pad_upper) for axis in dims}, mode=padding)
    offset_tensors = []
    for offset in offsets:
        components = []
        for dimension in dims:
            slices = {dim: slice(pad_lower + offset, -pad_upper + offset) if dim == dimension else slice(pad_lower, -pad_upper) for dim in dims}
            slices = {dim: slice(sl.start, sl.stop if sl.stop < 0 else None) for dim, sl in slices.items()}  # replace stop=0 by stop=None
            components.append(x[slices])
        offset_tensors.append(channel_stack(components, stack_dim) if stack_dim is not None else components[0])
    return offset_tensors


# Gradient

def gradient(grid: Tensor,
             dx: float or int = 1,
             difference: str = 'central',
             padding: Extrapolation = extrapolation.BOUNDARY,
             dims: tuple or None = None):
    """
    Calculates the gradient of a scalar channel from finite differences.
    The gradient vectors are in reverse order, lowest dimension first.

    :param grid: grid values
    :param dims: (optional) sequence of dimension names
    :param dx: physical distance between grid points (default 1)
    :param difference: type of difference, one of ('forward', 'backward', 'central') (default 'forward')
    :param padding: tensor padding mode
    :return: tensor of shape (batch_size, spatial_dimensions..., spatial rank)
    """
    grid = tensor(grid)
    if difference.lower() == 'central':
        left, right = shift(grid, (-1, 1), dims, padding, stack_dim='gradient')
        return (right - left) / (dx * 2)
    elif difference.lower() == 'forward':
        left, right = shift(grid, (0, 1), dims, padding, stack_dim='gradient')
        return (right - left) / dx
    elif difference.lower() == 'backward':
        left, right = shift(grid, (-1, 0), dims, padding, stack_dim='gradient')
        return (right - left) / dx
    else:
        raise ValueError('Invalid difference type: {}. Can be CENTRAL or FORWARD'.format(difference))


# Laplace

def laplace(x: Tensor,
            dx: Tensor or float = 1,
            padding: Extrapolation = extrapolation.BOUNDARY,
            dims: tuple or None = None):
    """
    Spatial Laplace operator as defined for scalar fields.
    If a vector field is passed, the laplace is computed component-wise.

    :param x: n-dimensional field of shape (batch, spacial dimensions..., components)
    :param dx: scalar or 1d tensor
    :param padding: extrapolation
    :param dims: The second derivative along these dimensions is summed over
    :return: tensor of same shape
    """
    if not isinstance(dx, (int, float)):
        dx = tensor(dx, names='_laplace')
    if isinstance(x, Extrapolation):
        return x.gradient()
    left, center, right = shift(tensor(x), (-1, 0, 1), dims, padding, stack_dim='_laplace')
    result = (left + right - 2 * center) / dx
    result = math.sum_(result, '_laplace')
    return result


def fourier_laplace(grid: Tensor,
                    dx: Tensor or Shape or float or list or tuple,
                    times: int = 1):
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


def fourier_poisson(grid: Tensor,
                    dx: Tensor or Shape or float or list or tuple,
                    times: int = 1):
    """ Inverse operation to `fourier_laplace`. """
    frequencies = math.fft(math.to_complex(grid))
    k_squared = math.sum_(math.fftfreq(grid.shape) ** 2, 'vector')
    fft_laplace = -(2 * np.pi)**2 * k_squared
    # fft_laplace.tensor[(0,) * math.ndims(k_squared)] = math.inf  # assume NumPy array to edit
    result = math.real(math.ifft(math.divide_no_nan(frequencies, fft_laplace ** times)))
    return math.cast(result * tensor(dx) ** 2, grid.dtype)


# Downsample / Upsample

def downsample2x(grid: Tensor,
                 padding: Extrapolation = extrapolation.BOUNDARY,
                 dims: tuple or None = None) -> Tensor:
    """
    Resamples a regular grid to half the number of spatial sample points per dimension.
    The grid values at the new points are determined via linear interpolation.

    :param grid: full size grid
    :param padding: grid extrapolation. Used to insert an additional value for odd spatial dims
    :param dims: dims along which down-sampling is applied. If None, down-sample along all spatial dims.
    :return: half-size grid
    """
    dims = grid.shape.spatial.only(dims).names
    odd_dimensions = [dim for dim in dims if grid.shape.get_size(dim) % 2 != 0]
    grid = math.pad(grid, {dim: (0, 1) for dim in odd_dimensions}, padding)
    for dim in dims:
        grid = (grid[{dim: slice(1, None, 2)}] + grid[{dim: slice(0, None, 2)}]) / 2
    return grid


def upsample2x(grid: Tensor,
               padding: Extrapolation = extrapolation.BOUNDARY,
               dims: tuple or None = None) -> Tensor:
    """
    Resamples a regular grid to double the number of spatial sample points per dimension.
    The grid values at the new points are determined via linear interpolation.

    :param grid: half-size grid
    :param padding: grid extrapolation
    :param dims: dims along which up-sampling is applied. If None, up-sample along all spatial dims.
    :return: double-size grid
    """
    for i, dim in enumerate(grid.shape.spatial.only(dims).names):
        left, center, right = shift(grid, (-1, 0, 1), (dim,), padding, None)
        interp_left = 0.25 * left + 0.75 * center
        interp_right = 0.75 * center + 0.25 * right
        stacked = math.spatial_stack([interp_left, interp_right], '_interleave')
        grid = math.join_dimensions(stacked, (dim, '_interleave'), dim)
    return grid


def sample_subgrid(grid: Tensor, start: Tensor, size: Shape) -> Tensor:
    """
    Samples a sub-grid from `grid` with equal distance between sampling points.
    The values at the new sample points are determined via linear interpolation.

    :param grid: full size grid to be resampled
    :param start: origin point of sub-grid within `grid`, measured in number of cells.
    Must have a single dimension called `vector`.
    Example: `start=(1, 0.5)` would slice off the first grid point in dim 1 and take the mean of neighbouring points in dim 2.
    The order of dims must be equal to `size` and `grid.shape.spatial`.
    :param size: resolution of the sub-grid. Must not be larger than the resolution of `grid`.
    The order of dims must be equal to `start` and `grid.shape.spatial`.
    :return: sampled sub-grid
    """
    assert start.shape.names == ('vector',)
    assert grid.shape.spatial.names == size.names
    discard = {}
    for dim, d_start, d_size in zip(grid.shape.spatial.names, start, size):
        discard[dim] = slice(int(d_start), int(d_start) + d_size + (1 if d_start != 0 else 0))
    grid = grid[discard]
    upper_weight = start % 1
    lower_weight = 1 - upper_weight
    for i, dim in enumerate(grid.shape.spatial.names):
        if upper_weight[i] not in (0, 1):
            lower, upper = shift(grid, (0, 1), [dim], padding=None, stack_dim=None)
            grid = upper * upper_weight[i] + lower * lower_weight[i]
    return grid


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

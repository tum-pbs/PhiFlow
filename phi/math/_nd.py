# Because division is different in Python 2 and 3
from __future__ import division

from typing import Tuple

import numpy as np

from . import _ops as math
from . import extrapolation as extrapolation
from ._config import GLOBAL_AXIS_ORDER
from ._ops import stack
from ._shape import Shape, channel, batch, spatial
from ._tensors import Tensor, TensorLike, variable_values
from ._tensors import wrap
from .extrapolation import Extrapolation


def vec_abs(vec: Tensor, vec_dim: str or tuple or list or Shape = None):
    """ Computes the vector length of `vec`. If `vec_dim` is None, the combined channel dimensions of `vec` are interpreted as a vector. """
    return math.sqrt(math.sum_(vec ** 2, dim=vec.shape.channel if vec_dim is None else vec_dim))


def vec_squared(vec: Tensor, vec_dim: str or tuple or list or Shape = None):
    """ Computes the squared length of `vec`. If `vec_dim` is None, the combined channel dimensions of `vec` are interpreted as a vector. """
    return math.sum_(vec ** 2, dim=vec.shape.channel if vec_dim is None else vec_dim)


def vec_normalize(vec: Tensor, vec_dim: str or tuple or list or Shape = None):
    """ Normalizes the vectors in `vec`. If `vec_dim` is None, the combined channel dimensions of `vec` are interpreted as a vector. """
    return vec / vec_abs(vec, vec_dim=vec_dim)


def cross_product(vec1: Tensor, vec2: Tensor) -> Tensor:
    """
    Computes the cross product of two vectors in 2D.

    Args:
        vec1: `Tensor` with a single channel dimension called `'vector'`
        vec2: `Tensor` with a single channel dimension called `'vector'`

    Returns:
        `Tensor`
    """
    vec1 = math.tensor(vec1)
    vec2 = math.tensor(vec2)
    spatial_rank = vec1.vector.size if 'vector' in vec1.shape else vec2.vector.size
    if spatial_rank == 2:  # Curl in 2D
        assert vec2.vector.exists
        if vec1.vector.exists:
            v1_x, v1_y = vec1.vector.unstack()
            v2_x, v2_y = vec2.vector.unstack()
            if GLOBAL_AXIS_ORDER.is_x_first:
                return v1_x * v2_y - v1_y * v2_x
            else:
                return - v1_x * v2_y + v1_y * v2_x
        else:
            v2_x, v2_y = vec2.vector.unstack()
            if GLOBAL_AXIS_ORDER.is_x_first:
                return vec1 * math.stack([-v2_y, v2_x], channel('vector'))
            else:
                return vec1 * math.stack([v2_y, -v2_x], channel('vector'))
    elif spatial_rank == 3:  # Curl in 3D
        raise NotImplementedError(f'spatial_rank={spatial_rank} not yet implemented')
    else:
        raise AssertionError(f'dims = {spatial_rank}. Vector product not available in > 3 dimensions')


def normalize_to(target: Tensor, source: float or Tensor, epsilon=1e-5):
    """
    Multiplies the target so that its sum matches the source.

    Args:
        target: `Tensor`
        source: `Tensor` or constant
        epsilon: Small number to prevent division by zero.

    Returns:
        Normalized tensor of the same shape as target
    """
    target_total = math.sum_(target)
    denominator = math.maximum(target_total, epsilon) if epsilon is not None else target_total
    source_total = math.sum_(source)
    return target * (source_total / denominator)


def l1_loss(x) -> Tensor:
    """
    Computes *∑<sub>i</sub> ||x<sub>i</sub>||<sub>1</sub>*, summing over all non-batch dimensions.

    Args:
        x: `Tensor` or `TensorLike`.
            For `TensorLike` objects, only value the sum over all value attributes is computed.

    Returns:
        loss: `Tensor`
    """
    if isinstance(x, Tensor):
        return math.sum_(abs(x), x.shape.non_batch)
    elif isinstance(x, TensorLike):
        return sum([l1_loss(getattr(x, a)) for a in variable_values(x)])
    else:
        raise ValueError(x)


def l2_loss(x) -> Tensor:
    """
    Computes *∑<sub>i</sub> ||x<sub>i</sub>||<sub>2</sub><sup>2</sup> / 2*, summing over all non-batch dimensions.

    Args:
        x: `Tensor` or `TensorLike`.
            For `TensorLike` objects, only value the sum over all value attributes is computed.

    Returns:
        loss: `Tensor`
    """
    if isinstance(x, Tensor):
        if x.dtype.kind == complex:
            x = abs(x)
        return math.sum_(x ** 2, x.shape.non_batch) * 0.5
    elif isinstance(x, TensorLike):
        return sum([l2_loss(getattr(x, a)) for a in variable_values(x)])
    else:
        raise ValueError(x)


def frequency_loss(x,
                   frequency_falloff: float = 100,
                   threshold=1e-5,
                   ignore_mean=False) -> Tensor:
    """
    Penalizes the squared `values` in frequency (Fourier) space.
    Lower frequencies are weighted more strongly then higher frequencies, depending on `frequency_falloff`.

    Args:
        x: `Tensor` or `TensorLike` Values to penalize, typically `actual - target`.
        frequency_falloff: Large values put more emphasis on lower frequencies, 1.0 weights all frequencies equally.
            *Note*: The total loss is not normalized. Varying the value will result in losses of different magnitudes.
        threshold: Frequency amplitudes below this value are ignored.
            Setting this to zero may cause infinities or NaN values during backpropagation.
        ignore_mean: If `True`, does not penalize the mean value (frequency=0 component).

    Returns:
      Scalar loss value
    """
    if isinstance(x, Tensor):
        if ignore_mean:
            x -= math.mean(x, x.shape.non_batch)
        k_squared = vec_squared(math.fftfreq(x.shape.spatial))
        weights = math.exp(-0.5 * k_squared * frequency_falloff ** 2)
        diff_fft = abs_square(math.fft(x) * weights)
        diff_fft = math.sqrt(math.maximum(diff_fft, threshold))
        return l2_loss(diff_fft)
    elif isinstance(x, TensorLike):
        return sum([frequency_loss(getattr(x, a), frequency_falloff, threshold, ignore_mean) for a in variable_values(x)])
    else:
        raise ValueError(x)


def abs_square(complex_values: Tensor) -> Tensor:
    """
    Squared magnitude of complex values.

    Args:
      complex_values: complex `Tensor`

    Returns:
        Tensor: real valued magnitude squared

    """
    return math.imag(complex_values) ** 2 + math.real(complex_values) ** 2


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
          stack_dim: Shape or None = channel('shift')) -> list:
    """
    shift Tensor by a fixed offset and abiding by extrapolation

    Args:
        x: Input data
        offsets: Shift size
        dims: Dimensions along which to shift, defaults to None
        padding: padding to be performed at the boundary, defaults to extrapolation.BOUNDARY
        stack_dim: dimensions to be stacked, defaults to 'shift'

    Returns:
        list: offset_tensor

    """
    if stack_dim is None:
        assert len(dims) == 1
    x = wrap(x)
    dims = dims if dims is not None else x.shape.spatial.names
    pad_lower = max(0, -min(offsets))
    pad_upper = max(0, max(offsets))
    if padding:
        x = math.pad(x, {axis: (pad_lower, pad_upper) for axis in dims}, mode=padding)
    offset_tensors = []
    for offset in offsets:
        components = []
        for dimension in dims:
            if padding:
                slices = {dim: slice(pad_lower + offset, (-pad_upper + offset) or None) if dim == dimension else slice(pad_lower, -pad_upper or None) for dim in dims}
            else:
                slices = {dim: slice(pad_lower + offset, (-pad_upper + offset) or None) if dim == dimension else slice(None, None) for dim in dims}
            components.append(x[slices])
        offset_tensors.append(stack(components, stack_dim) if stack_dim is not None else components[0])
    return offset_tensors


def extrapolate_valid_values(values: Tensor, valid: Tensor, distance_cells: int = 1) -> Tuple[Tensor, Tensor]:
    """
    Extrapolates the values of `values` which are marked by the nonzero values of `valid` for `distance_cells` steps in all spatial directions.
    Overlapping extrapolated values get averaged. Extrapolation also includes diagonals.

    Examples (1-step extrapolation), x marks the values for extrapolation:
        200   000    111        004   00x    044        102   000    144
        010 + 0x0 => 111        000 + 000 => 234        004 + 00x => 234
        040   000    111        200   x00    220        200   x00    234

    Args:
        values: Tensor which holds the values for extrapolation
        valid: Tensor with same size as `x` marking the values for extrapolation with nonzero values
        distance_cells: Number of extrapolation steps

    Returns:
        values: Extrapolation result
        valid: mask marking all valid values after extrapolation
    """

    def binarize(x):
        return math.divide_no_nan(x, x)

    distance_cells = min(distance_cells, max(values.shape.sizes))
    for _ in range(distance_cells):
        valid = binarize(valid)
        valid_values = valid * values
        overlap = valid
        for dim in values.shape.spatial.names:
            values_l, values_r = shift(valid_values, (-1, 1), dims=dim, padding=extrapolation.ZERO)
            valid_values = math.sum_(values_l + values_r + valid_values, dim='shift')
            mask_l, mask_r = shift(overlap, (-1, 1), dims=dim, padding=extrapolation.ZERO)
            overlap = math.sum_(mask_l + mask_r + overlap, dim='shift')
        extp = math.divide_no_nan(valid_values, overlap)  # take mean where extrapolated values overlap
        values = math.where(valid, values, math.where(binarize(overlap), extp, values))
        valid = overlap
    return values, binarize(valid)


# Gradient

def spatial_gradient(grid: Tensor,
                     dx: float or int = 1,
                     difference: str = 'central',
                     padding: Extrapolation or None = extrapolation.BOUNDARY,
                     dims: tuple or None = None,
                     stack_dim: Shape = channel('gradient')):
    """
    Calculates the spatial_gradient of a scalar channel from finite differences.
    The spatial_gradient vectors are in reverse order, lowest dimension first.

    Args:
      grid: grid values
      dims: optional) sequence of dimension names
      dx: physical distance between grid points (default 1)
      difference: type of difference, one of ('forward', 'backward', 'central') (default 'forward')
      padding: tensor padding mode
      stack_dim: name of the new vector dimension listing the spatial_gradient w.r.t. the various axes

    Returns:
      tensor of shape (batch_size, spatial_dimensions..., spatial rank)

    """
    grid = wrap(grid)
    if difference.lower() == 'central':
        left, right = shift(grid, (-1, 1), dims, padding, stack_dim=stack_dim)
        return (right - left) / (dx * 2)
    elif difference.lower() == 'forward':
        left, right = shift(grid, (0, 1), dims, padding, stack_dim=stack_dim)
        return (right - left) / dx
    elif difference.lower() == 'backward':
        left, right = shift(grid, (-1, 0), dims, padding, stack_dim=stack_dim)
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

    Args:
        x: n-dimensional field of shape (batch, spacial dimensions..., components)
        dx: scalar or 1d tensor
        padding: extrapolation
        dims: The second derivative along these dimensions is summed over

    Returns:
        `phi.math.Tensor` of same shape as `x`

    """
    if isinstance(dx, (tuple, list)):
        dx = wrap(dx, batch('_laplace'))
    elif isinstance(dx, Tensor) and dx.vector.exists:
        dx = math.rename_dims(dx, 'vector', batch('_laplace'))
    if isinstance(x, Extrapolation):
        return x.spatial_gradient()
    left, center, right = shift(wrap(x), (-1, 0, 1), dims, padding, stack_dim=batch('_laplace'))
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

    Args:
      grid: tensor, assumed to have periodic boundary conditions
      dx: distance between grid points, tensor-like, scalar or vector
      times: number of times the laplace operator is applied. The computational cost is independent of this parameter.
      grid: Tensor: 
      dx: Tensor or Shape or float or list or tuple: 
      times: int:  (Default value = 1)

    Returns:
      tensor of same shape as `tensor`

    """
    frequencies = math.fft(math.to_complex(grid))
    k_squared = math.sum_(math.fftfreq(grid.shape) ** 2, 'vector')
    fft_laplace = -(2 * np.pi) ** 2 * k_squared
    result = math.real(math.ifft(frequencies * fft_laplace ** times))
    return math.cast(result / wrap(dx) ** 2, grid.dtype)


def fourier_poisson(grid: Tensor,
                    dx: Tensor or Shape or float or list or tuple,
                    times: int = 1):
    """
    Inverse operation to `fourier_laplace`.

    Args:
      grid: Tensor: 
      dx: Tensor or Shape or float or list or tuple: 
      times: int:  (Default value = 1)

    Returns:

    """
    frequencies = math.fft(math.to_complex(grid))
    k_squared = math.sum_(math.fftfreq(grid.shape) ** 2, 'vector')
    fft_laplace = -(2 * np.pi) ** 2 * k_squared
    # fft_laplace.tensor[(0,) * math.ndims(k_squared)] = math.inf  # assume NumPy array to edit
    result = math.real(math.ifft(math.divide_no_nan(frequencies, math.to_complex(fft_laplace ** times))))
    return math.cast(result * wrap(dx) ** 2, grid.dtype)


# Downsample / Upsample

def downsample2x(grid: Tensor,
                 padding: Extrapolation = extrapolation.BOUNDARY,
                 dims: tuple or None = None) -> Tensor:
    """
    Resamples a regular grid to half the number of spatial sample points per dimension.
    The grid values at the new points are determined via mean (linear interpolation).

    Args:
      grid: full size grid
      padding: grid extrapolation. Used to insert an additional value for odd spatial dims
      dims: dims along which down-sampling is applied. If None, down-sample along all spatial dims.
      grid: Tensor: 
      padding: Extrapolation:  (Default value = extrapolation.BOUNDARY)
      dims: tuple or None:  (Default value = None)

    Returns:
      half-size grid

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

    Args:
      grid: half-size grid
      padding: grid extrapolation
      dims: dims along which up-sampling is applied. If None, up-sample along all spatial dims.
      grid: Tensor: 
      padding: Extrapolation:  (Default value = extrapolation.BOUNDARY)
      dims: tuple or None:  (Default value = None)

    Returns:
      double-size grid

    """
    for i, dim in enumerate(grid.shape.spatial.only(dims)):
        left, center, right = shift(grid, (-1, 0, 1), dim.names, padding, None)
        interp_left = 0.25 * left + 0.75 * center
        interp_right = 0.75 * center + 0.25 * right
        stacked = math.stack([interp_left, interp_right], spatial('_interleave'))
        grid = math.pack_dims(stacked, (dim.name, '_interleave'), dim)
    return grid


def sample_subgrid(grid: Tensor, start: Tensor, size: Shape) -> Tensor:
    """
    Samples a sub-grid from `grid` with equal distance between sampling points.
    The values at the new sample points are determined via linear interpolation.

    Args:
        grid: `Tensor` to be resampled. Values are assumed to be sampled at cell centers.
        start: Origin point of sub-grid within `grid`, measured in number of cells.
            Must have a single dimension called `vector`.
            Example: `start=(1, 0.5)` would slice off the first grid point in dim 1 and take the mean of neighbouring points in dim 2.
            The order of dims must be equal to `size` and `grid.shape.spatial`.
        size: Resolution of the sub-grid. Must not be larger than the resolution of `grid`.
            The order of dims must be equal to `start` and `grid.shape.spatial`.

    Returns:
      Sub-grid as `Tensor`
    """
    assert start.shape.names == ('vector',)
    assert grid.shape.spatial.names == size.names
    assert math.all_available(start), "Cannot perform sample_subgrid() during tracing, 'start' must be known."
    discard = {}
    for dim, d_start, d_size in zip(grid.shape.spatial.names, start, size.sizes):
        discard[dim] = slice(int(d_start), int(d_start) + d_size + (1 if d_start != 0 else 0))
    grid = grid[discard]
    upper_weight = start % 1
    lower_weight = 1 - upper_weight
    for i, dim in enumerate(grid.shape.spatial.names):
        if upper_weight[i].native() not in (0, 1):
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
        raise NotImplementedError("\n".join([
                                      "Not implemented for:"
                                      f"ranks ({grid1.rank}, {grid2.rank}) != 2",
                                      f"boundary ({grid1.boundary}, {grid2.boundary}) != {extrapolation.PERIODIC}",
                                      f"dx uniform ({grid1.dx}, {grid2.dx})"
                                  ]))


def _periodic_2d_arakawa_poisson_bracket(tensor1: Tensor, tensor2: Tensor, dx: float):
    """
    Solves the poisson bracket using the Arakawa Scheme [tensor1, tensor2]
    
    Only works in 2D, with equal spaced grids, and periodic boundary conditions

    Args:
      tensor1(Tensor): first field in the poisson bracket
      tensor2(Tensor): second field in the poisson bracket
      dx(float): Grid size (equal in x-y)
      tensor1: Tensor: 
      tensor2: Tensor: 
      dx: float: 

    Returns:

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
            - zeta.x[0:-2].y[0:-2] * (psi.x[0:-2].y[1:-1] - psi.x[1:-1].y[0:-2])) / (12 * dx ** 2)

from typing import Tuple, Optional

import numpy as np

from . import _ops as math
from . import extrapolation as extrapolation
from ._magic_ops import stack, rename_dims, concat, variable_values
from ._shape import Shape, channel, batch, spatial, DimFilter, parse_dim_order
from ._tensors import Tensor, wrap
from .magic import PhiTreeNode
from .extrapolation import Extrapolation


def vec(name='vector', **components) -> Tensor:
    """
    Lay out the given values along a channel dimension without converting them to the current backend.

    Args:
        **components: Values by component name.
        name: Dimension name.

    Returns:
        `Tensor`
    """
    return stack(components, channel(name))


def const_vec(value: float or Tensor, dim: Shape or tuple or list or str):
    """
    Creates a single-dimension tensor with all values equal to `value`.
    `value` is not converted to the default backend, even when it is a Python primitive.

    Args:
        value: Value for filling the vector.
        dim: Either single-dimension non-spatial Shape or `Shape` consisting of any number of spatial dimensions.
            In the latter case, a new channel dimension named `'vector'` will be created from the spatial shape.

    Returns:
        `Tensor`
    """
    if isinstance(dim, Shape):
        if dim.spatial:
            assert not dim.non_spatial, f"When creating a vector given spatial dimensions, the shape may only contain spatial dimensions but got {dim}"
            shape = channel(vector=dim.names)
        else:
            assert dim.rank == 1, f"Cannot create vector from {dim}"
            shape = dim
    else:
        dims = parse_dim_order(dim)
        shape = channel(vector=dims)
    return wrap([value] * shape.size, shape)


def vec_abs(vec: Tensor, vec_dim: DimFilter = channel, eps: float or Tensor = None):
    """
    Computes the vector length of `vec`.

    Args:
        eps: Minimum vector length. Use to avoid `inf` gradients for zero-length vectors.
    """
    squared = vec_squared(vec, vec_dim)
    if eps is not None:
        squared = math.maximum(squared, eps)
    return math.sqrt(squared)


def vec_squared(vec: Tensor, vec_dim: DimFilter = channel):
    """ Computes the squared length of `vec`. If `vec_dim` is None, the combined channel dimensions of `vec` are interpreted as a vector. """
    return math.sum_(vec ** 2, dim=vec_dim)


def vec_normalize(vec: Tensor, vec_dim: DimFilter = channel):
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
            v1_x, v1_y = vec1.vector
            v2_x, v2_y = vec2.vector
            return v1_x * v2_y - v1_y * v2_x
        else:
            v2_x, v2_y = vec2.vector
            return vec1 * math.stack_tensors([-v2_y, v2_x], channel('vector'))
    elif spatial_rank == 3:  # Curl in 3D
        raise NotImplementedError(f'spatial_rank={spatial_rank} not yet implemented')
    else:
        raise AssertionError(f'dims = {spatial_rank}. Vector product not available in > 3 dimensions')


def rotate_vector(vector: math.Tensor, angle: float or math.Tensor) -> Tensor:
    """
    Rotates `vector` around the origin.

    Args:
        vector: n-dimensional vector with a channel dimension called `'vector'`
        angle: Euler angle. The direction is the rotation axis and the length is the amount (in radians).

    Returns:
        Rotated vector as `Tensor`
    """
    assert 'vector' in vector.shape, "vector must have 'vector' dimension."
    if vector.vector.size == 2:
        sin = wrap(math.sin(angle))
        cos = wrap(math.cos(angle))
        x, y = vector.vector
        rot_x = cos * x - sin * y
        rot_y = sin * x + cos * y
        return math.stack_tensors([rot_x, rot_y], channel(vector=vector.vector.item_names))
    elif vector.vector.size == 1:
        raise AssertionError(f"Cannot rotate a 1D vector. shape={vector.shape}")
    else:
        raise NotImplementedError(f"Rotation in {vector.vector.size}D not yet implemented.")


def dim_mask(all_dims: Shape or tuple or list, dims: DimFilter, mask_dim=channel('vector')) -> Tensor:
    """
    Creates a masked vector with 1 elements for `dims` and 0 for all other dimensions in `all_dims`.

    Args:
        all_dims: All dimensions for which the vector should have an entry.
        dims: Dimensions marked as 1.
        mask_dim: Dimension of the masked vector. Item names are assigned automatically.

    Returns:
        `Tensor`
    """
    assert isinstance(all_dims, (Shape, tuple, list)), f"all_dims must be a tuple or Shape but got {type(all_dims)}"
    assert isinstance(mask_dim, Shape) and mask_dim.rank == 1, f"mask_dim must be a single-dimension Shape but got {mask_dim}"
    if isinstance(all_dims, (tuple, list)):
        all_dims = spatial(*all_dims)
    dims = all_dims.only(dims)
    mask = [1 if dim in dims else 0 for dim in all_dims]
    mask_dim = mask_dim._with_item_names((all_dims.names,))
    return wrap(mask, mask_dim)


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


def l1_loss(x, reduce: DimFilter = math.non_batch) -> Tensor:
    """
    Computes *∑<sub>i</sub> ||x<sub>i</sub>||<sub>1</sub>*, summing over all non-batch dimensions.

    Args:
        x: `Tensor` or `PhiTreeNode` or 0D or 1D native tensor.
            For `PhiTreeNode` objects, only value the sum over all value attributes is computed.
        reduce: Dimensions to reduce as `DimFilter`.

    Returns:
        loss: `Tensor`
    """
    if isinstance(x, Tensor):
        return math.sum_(abs(x), reduce)
    elif isinstance(x, PhiTreeNode):
        return sum([l1_loss(getattr(x, a), reduce) for a in variable_values(x)])
    else:
        try:
            backend = math.choose_backend(x)
            shape = backend.staticshape(x)
            if len(shape) == 0:
                return abs(x)
            elif len(shape) == 1:
                return backend.sum(abs(x))
            else:
                raise ValueError("l2_loss is only defined for 0D and 1D native tensors. For higher-dimensional data, use Φ-Flow tensors.")
        except math.NoBackendFound:
            raise ValueError(x)


def l2_loss(x, reduce: DimFilter = math.non_batch) -> Tensor:
    """
    Computes *∑<sub>i</sub> ||x<sub>i</sub>||<sub>2</sub><sup>2</sup> / 2*, summing over all non-batch dimensions.

    Args:
        x: `Tensor` or `PhiTreeNode` or 0D or 1D native tensor.
            For `PhiTreeNode` objects, only value the sum over all value attributes is computed.
        reduce: Dimensions to reduce as `DimFilter`.

    Returns:
        loss: `Tensor`
    """
    if isinstance(x, Tensor):
        if x.dtype.kind == complex:
            x = abs(x)
        return math.sum_(x ** 2, reduce) * 0.5
    elif isinstance(x, PhiTreeNode):
        return sum([l2_loss(getattr(x, a), reduce) for a in variable_values(x)])
    else:
        try:
            backend = math.choose_backend(x)
            shape = backend.staticshape(x)
            if len(shape) == 0:
                return x ** 2 * 0.5
            elif len(shape) == 1:
                return backend.sum(x ** 2) * 0.5
            else:
                raise ValueError("l2_loss is only defined for 0D and 1D native tensors. For higher-dimensional data, use Φ-Flow tensors.")
        except math.NoBackendFound:
            raise ValueError(x)


def frequency_loss(x,
                   frequency_falloff: float = 100,
                   threshold=1e-5,
                   ignore_mean=False,
                   n=2) -> Tensor:
    """
    Penalizes the squared `values` in frequency (Fourier) space.
    Lower frequencies are weighted more strongly then higher frequencies, depending on `frequency_falloff`.

    Args:
        x: `Tensor` or `PhiTreeNode` Values to penalize, typically `actual - target`.
        frequency_falloff: Large values put more emphasis on lower frequencies, 1.0 weights all frequencies equally.
            *Note*: The total loss is not normalized. Varying the value will result in losses of different magnitudes.
        threshold: Frequency amplitudes below this value are ignored.
            Setting this to zero may cause infinities or NaN values during backpropagation.
        ignore_mean: If `True`, does not penalize the mean value (frequency=0 component).

    Returns:
      Scalar loss value
    """
    assert n in (1, 2)
    if isinstance(x, Tensor):
        if ignore_mean:
            x -= math.mean(x, x.shape.non_batch)
        k_squared = vec_squared(math.fftfreq(x.shape.spatial))
        weights = math.exp(-0.5 * k_squared * frequency_falloff ** 2)

        diff_fft = abs_square(math.fft(x) * weights)
        diff_fft = math.sqrt(math.maximum(diff_fft, threshold))
        return l2_loss(diff_fft) if n == 2 else l1_loss(diff_fft)
    elif isinstance(x, PhiTreeNode):
        losses = [frequency_loss(getattr(x, a), frequency_falloff, threshold, ignore_mean, n) for a in variable_values(x)]
        return sum(losses)
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
          dims: DimFilter = math.spatial,
          padding: Extrapolation or None = extrapolation.BOUNDARY,
          stack_dim: Optional[Shape] = channel('shift')) -> list:
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
    if dims is None:
        raise ValueError("dims=None is not supported anymore.")
    dims = x.shape.only(dims).names
    if stack_dim is None:
        assert len(dims) == 1
    x = wrap(x)
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


def masked_fill(values: Tensor, valid: Tensor, distance: int = 1) -> Tuple[Tensor, Tensor]:
    """
    Extrapolates the values of `values` which are marked by the nonzero values of `valid` for `distance` steps in all spatial directions.
    Overlapping extrapolated values get averaged. Extrapolation also includes diagonals.

    Args:
        values: Tensor which holds the values for extrapolation
        valid: Tensor with same size as `x` marking the values for extrapolation with nonzero values
        distance: Number of extrapolation steps

    Returns:
        values: Extrapolation result
        valid: mask marking all valid values after extrapolation
    """
    def binarize(x):
        return math.divide_no_nan(x, x)
    distance = min(distance, max(values.shape.sizes))
    for _ in range(distance):
        valid = binarize(valid)
        valid_values = valid * values
        overlap = valid  # count how many values we are adding
        for dim in values.shape.spatial.names:
            values_l, values_r = shift(valid_values, (-1, 1), dims=dim, padding=extrapolation.ZERO)
            valid_values = math.sum_(values_l + values_r + valid_values, dim='shift')
            mask_l, mask_r = shift(overlap, (-1, 1), dims=dim, padding=extrapolation.ZERO)
            overlap = math.sum_(mask_l + mask_r + overlap, dim='shift')
        extp = math.divide_no_nan(valid_values, overlap)  # take mean where extrapolated values overlap
        values = math.where(valid, values, math.where(binarize(overlap), extp, values))
        valid = overlap
    return values, binarize(valid)


def finite_fill(values: Tensor, dims: DimFilter = spatial, distance: int = 1, diagonal: bool = True, padding=extrapolation.BOUNDARY) -> Tuple[Tensor, Tensor]:
    """
    Fills non-finite (NaN, inf, -inf) values from nearby finite values.
    Extrapolates the finite values of `values` for `distance` steps along `dims`.
    Where multiple finite values could fill an invalid value, the average is computed.

    Args:
        values: Floating-point `Tensor`. All non-numeric values (`NaN`, `inf`, `-inf`) are interpreted as invalid.
        dims: Dimensions along which to fill invalid values from finite ones.
        distance: Number of extrapolation steps, each extrapolating one cell out.
        diagonal: Whether to extrapolate values to their diagonal neighbors per step.
        padding: Extrapolation of `values`. Determines whether to extrapolate from the edges as well.

    Returns:
        `Tensor` of same shape as `values`.
    """
    if diagonal:
        distance = min(distance, max(values.shape.sizes))
        dims = values.shape.only(dims)
        for _ in range(distance):
            valid = math.is_finite(values)
            valid_values = math.where(valid, values, 0)
            overlap = valid
            for dim in dims:
                values_l, values_r = shift(valid_values, (-1, 1), dims=dim, padding=padding)
                valid_values = math.sum_(values_l + values_r + valid_values, dim='shift')
                mask_l, mask_r = shift(overlap, (-1, 1), dims=dim, padding=padding)
                overlap = math.sum_(mask_l + mask_r + overlap, dim='shift')
            values = math.where(valid, values, valid_values / overlap)
    else:
        distance = min(distance, sum(values.shape.sizes))
        for _ in range(distance):
            neighbors = concat(shift(values, (-1, 1), dims, padding=padding, stack_dim=channel('neighbors')), 'neighbors')
            finite = math.is_finite(neighbors)
            avg_neighbors = math.sum_(math.where(finite, neighbors, 0), 'neighbors') / math.sum_(finite, 'neighbors')
            values = math.where(math.is_finite(values), values, avg_neighbors)
    return values


# Gradient

def spatial_gradient(grid: Tensor,
                     dx: float or Tensor = 1,
                     difference: str = 'central',
                     padding: Extrapolation or None = extrapolation.BOUNDARY,
                     dims: DimFilter = spatial,
                     stack_dim: Shape or None = channel('gradient')) -> Tensor:
    """
    Calculates the spatial_gradient of a scalar channel from finite differences.
    The spatial_gradient vectors are in reverse order, lowest dimension first.

    Args:
        grid: grid values
        dims: (Optional) Dimensions along which the spatial derivative will be computed. sequence of dimension names
        dx: Physical distance between grid points, `float` or `Tensor`.
            When passing a vector-valued `Tensor`, the dx values should be listed along `stack_dim`, matching `dims`.
        difference: type of difference, one of ('forward', 'backward', 'central') (default 'forward')
        padding: tensor padding mode
        stack_dim: name of the new vector dimension listing the spatial_gradient w.r.t. the various axes

    Returns:
        `Tensor`
    """
    grid = wrap(grid)
    if stack_dim is not None and stack_dim in grid.shape:
        assert grid.shape.only(stack_dim).size == 1, f"spatial_gradient() cannot list components along {stack_dim.name} because that dimension already exists on grid {grid}"
        grid = grid[{stack_dim.name: 0}]
    dims = grid.shape.only(dims)
    dx = wrap(dx)
    if dx.vector.exists:
        dx = dx.vector[dims]
        if dx.vector.size in (None, 1):
            dx = dx.vector[0]
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
            dims: DimFilter = spatial):
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
        dx = rename_dims(dx, 'vector', batch('_laplace'))
    if isinstance(x, Extrapolation):
        return x.spatial_gradient()
    left, center, right = shift(wrap(x), (-1, 0, 1), dims, padding, stack_dim=batch('_laplace'))
    result = (left + right - 2 * center) / (dx ** 2)
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
                 dims: DimFilter = spatial) -> Tensor:
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
    dims = grid.shape.only(dims).names
    odd_dimensions = [dim for dim in dims if grid.shape.get_size(dim) % 2 != 0]
    grid = math.pad(grid, {dim: (0, 1) for dim in odd_dimensions}, padding)
    for dim in dims:
        grid = (grid[{dim: slice(1, None, 2)}] + grid[{dim: slice(0, None, 2)}]) / 2
    return grid


def upsample2x(grid: Tensor,
               padding: Extrapolation = extrapolation.BOUNDARY,
               dims: DimFilter = spatial) -> Tensor:
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
    for dim in grid.shape.only(dims):
        left, center, right = shift(grid, (-1, 0, 1), dim.names, padding, None)
        interp_left = 0.25 * left + 0.75 * center
        interp_right = 0.75 * center + 0.25 * right
        stacked = math.stack_tensors([interp_left, interp_right], channel(_interleave='left,right'))
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
    crop = {}
    for dim, d_start, d_size in zip(grid.shape.spatial.names, start, size.sizes):
        crop[dim] = slice(int(d_start), int(d_start) + d_size + (0 if d_start % 1 in (0, 1) else 1))
    grid = grid[crop]
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

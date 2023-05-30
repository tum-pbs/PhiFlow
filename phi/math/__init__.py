"""
Vectorized operations, tensors with named dimensions.

This package provides a common interface for tensor operations.
Is internally uses NumPy, TensorFlow or PyTorch.

Main classes: `Tensor`, `Shape`, `DType`, `Extrapolation`.

The provided operations are not implemented directly.
Instead, they delegate the actual computation to either NumPy, TensorFlow or PyTorch, depending on the configuration.
This allows the user to write simulation code once and have it run with various computation backends.

See the documentation at https://tum-pbs.github.io/PhiFlow/Math.html
"""

from .backend._dtype import DType
from .backend import NUMPY, precision, set_global_precision, get_precision

from ._shape import (
    shape, Shape, EMPTY_SHAPE, DimFilter,
    spatial, channel, batch, instance, dual,
    non_batch, non_spatial, non_instance, non_channel, non_dual, non_primal, primal,
    merge_shapes, concat_shapes, IncompatibleShapes,
    enable_debug_checks,
)

from ._magic_ops import (
    slice_ as slice, unstack,
    stack, concat, expand,
    rename_dims, rename_dims as replace_dims, pack_dims, unpack_dim, flatten,
    b2i, c2b, i2b, s2b, si2d,
    copy_with, replace
)

from ._tensors import wrap, tensor, layout, Tensor, Dict, to_dict, from_dict, is_scalar, BROADCAST_FORMATTER as f

from ._sparse import dense, get_sparsity, get_format, sparse_tensor, stored_indices, stored_values, tensor_like

from .extrapolation import Extrapolation

from ._ops import (
    choose_backend_t as choose_backend, all_available, convert, seed, to_device,
    native, numpy, reshaped_native, reshaped_tensor, reshaped_numpy, copy, native_call,
    print_ as print,
    map_ as map,
    zeros, ones, fftfreq, random_normal, random_uniform, meshgrid, linspace, arange as range, range_tensor,  # creation operators (use default backend)
    zeros_like, ones_like,
    pad,
    transpose,  # reshape operations
    safe_div, safe_div as divide_no_nan,
    where, nonzero,
    sum_ as sum, finite_sum, mean, finite_mean, std, prod, max_ as max, finite_max, min_ as min, finite_min, any_ as any, all_ as all, quantile, median,  # reduce
    dot,
    abs_ as abs, sign,
    round_ as round, ceil, floor,
    maximum, minimum, clip,
    sqrt, exp, log, log2, log10, sigmoid, soft_plus,
    sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh, log_gamma, factorial,
    to_float, to_int32, to_int64, to_complex, imag, real, conjugate,
    degrees,
    boolean_mask,
    is_finite, is_finite as isfinite, is_nan, is_inf,
    closest_grid_values, grid_sample, scatter, gather,
    histogram,
    fft, ifft, convolve, cumulative_sum,
    dtype, cast,
    close, assert_close,
    stop_gradient,
    pairwise_distances, map_pairs,
)

from ._nd import (
    shift,
    vec, const_vec, vec_abs, vec_abs as vec_length, vec_squared, vec_normalize, cross_product, rotate_vector, dim_mask,
    normalize_to,
    l1_loss, l2_loss, frequency_loss,
    spatial_gradient, laplace,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, sample_subgrid,
    masked_fill, finite_fill
)

from ._trace import matrix_from_function

from ._functional import (
    LinearFunction, jit_compile_linear, jit_compile,
    jacobian, jacobian as gradient, functional_gradient, custom_gradient, print_gradient,
    map_types, map_s2b, map_i2b, map_c2b,
    broadcast,
    iterate,
    identity,
    trace_check,
)

from ._optimize import solve_linear, solve_nonlinear, minimize, Solve, SolveInfo, ConvergenceException, NotConverged, Diverged, SolveTape, factor_ilu

PI = 3.14159265358979323846
"""Value of π to double precision """
pi = PI  # intentionally undocumented, use PI instead. Exists only as an anlog to numpy.pi

INF = float("inf")
""" Floating-point representation of positive infinity. """
inf = INF  # intentionally undocumented, use INF instead. Exists only as an anlog to numpy.inf


NAN = float("nan")
""" Floating-point representation of NaN (not a number). """
nan = NAN  # intentionally undocumented, use NAN instead. Exists only as an anlog to numpy.nan

NUMPY = NUMPY  # to show up in pdoc
"""Default backend for NumPy arrays and SciPy objects."""

f = f
"""
Automatic mapper for broadcast string formatting of tensors, resulting in tensors of strings.
Used with the special `-f-` syntax.

Examples:
    >>> from phi.math import f
    >>> -f-f'String containing {tensor1} and {tensor2:.1f}'
    # Result is a str tensor containing all dims of tensor1 and tensor2
"""

__all__ = [key for key in globals().keys() if not key.startswith('_')]

__pdoc__ = {
    'Extrapolation': False,
    'Shape.__init__': False,
    'SolveInfo.__init__': False,
    'TensorDim.__init__': False,
    'ConvergenceException.__init__': False,
    'Diverged.__init__': False,
    'NotConverged.__init__': False,
    'LinearFunction.__init__': False,
}

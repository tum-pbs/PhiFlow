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
    spatial, channel, batch, instance,
    non_batch, non_spatial, non_instance, non_channel,
    merge_shapes, concat_shapes, IncompatibleShapes
)
from ._magic_ops import unstack, stack, concat, expand, rename_dims, pack_dims, unpack_dim, unpack_dim as unpack_dims, flatten, copy_with
from ._tensors import wrap, tensor, layout, Tensor, Dict, to_dict, from_dict, is_scalar
from .extrapolation import Extrapolation
from ._ops import (
    choose_backend_t as choose_backend, all_available, convert, seed,
    native, numpy, reshaped_native, reshaped_tensor, reshaped_numpy, copy, native_call,
    print_ as print,
    map_ as map,
    zeros, ones, fftfreq, random_normal, random_uniform, meshgrid, linspace, arange as range, range_tensor,  # creation operators (use default backend)
    zeros_like, ones_like,
    pad,
    transpose,  # reshape operations
    divide_no_nan,
    where, nonzero,
    sum_ as sum, finite_sum, mean, finite_mean, std, prod, max_ as max, finite_max, min_ as min, finite_min, any_ as any, all_ as all, quantile, median,  # reduce
    dot,
    abs_ as abs, sign,
    round_ as round, ceil, floor,
    maximum, minimum, clip,
    sqrt, exp, sin, cos, tan, log, log2, log10, sigmoid, arcsin, arccos,
    to_float, to_int32, to_int64, to_complex, imag, real, conjugate,
    degrees,
    boolean_mask,
    is_finite, is_finite as isfinite,
    closest_grid_values, grid_sample, scatter, gather,
    fft, ifft, convolve, cumulative_sum,
    dtype, cast,
    close, assert_close,
    stop_gradient
)
from ._nd import (
    shift,
    vec, const_vec, vec_abs, vec_abs as vec_length, vec_squared, vec_normalize, cross_product, rotate_vector, dim_mask,
    normalize_to,
    l1_loss, l2_loss, frequency_loss,
    spatial_gradient, laplace,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, sample_subgrid,
    masked_fill, finite_fill,
)
from ._functional import (
    LinearFunction, jit_compile_linear, jit_compile,
    jacobian, jacobian as gradient, functional_gradient, custom_gradient, print_gradient, hessian,
    solve_linear, solve_nonlinear, minimize, Solve, SolveInfo, ConvergenceException, NotConverged, Diverged, SolveTape,
    map_types, map_s2b, map_i2b,
    iterate,
)


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

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

from ._config import GLOBAL_AXIS_ORDER
from ._shape import Shape, EMPTY_SHAPE, spatial, channel, batch, instance, merge_shapes, concat_shapes
from ._tensors import wrap, tensor, Tensor, TensorDim, TensorLike, Dict
from .extrapolation import Extrapolation
from ._ops import (
    choose_backend_t as choose_backend, all_available, convert, seed,
    native, numpy, reshaped_native, reshaped_tensor, copy, native_call,
    print_ as print,
    map_ as map,
    zeros, ones, fftfreq, random_normal, random_uniform, meshgrid, linspace, arange as range, range_tensor,  # creation operators (use default backend)
    zeros_like, ones_like,
    stack, unstack, concat,
    pad,
    pack_dims, unpack_dims, rename_dims, flatten, expand, transpose,  # reshape operations
    divide_no_nan,
    where, nonzero,
    sum_ as sum, mean, std, prod, max_ as max, min_ as min, any_ as any, all_ as all, quantile, median,  # reduce
    dot,
    abs_ as abs, sign,
    round_ as round, ceil, floor,
    maximum, minimum, clip,
    sqrt, exp, sin, cos, tan, log, log2, log10,
    to_float, to_int32, to_int64, to_complex, imag, real, conjugate,
    boolean_mask,
    isfinite,
    closest_grid_values, grid_sample, scatter, gather,
    fft, ifft, convolve, cumulative_sum,
    dtype, cast,
    close, assert_close,
    record_gradients, gradients, stop_gradient
)
from ._nd import (
    shift,
    vec_abs, vec_squared, vec_normalize, cross_product,
    normalize_to,
    l1_loss, l2_loss, frequency_loss,
    spatial_gradient, laplace,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, sample_subgrid,
    extrapolate_valid_values,
)
from ._functional import (
    LinearFunction, jit_compile_linear, jit_compile,
    functional_gradient, custom_gradient, print_gradient,
    solve_linear, solve_nonlinear, minimize, Solve, SolveInfo, ConvergenceException, NotConverged, Diverged, SolveTape,
)


PI = 3.14159265358979323846
"""Value of π to double precision """
pi = PI

NUMPY = NUMPY  # to show up in pdoc
"""Default backend for NumPy arrays and SciPy objects."""

__all__ = [key for key in globals().keys() if not key.startswith('_')]

__pdoc__ = {
    'Extrapolation.__init__': False,
    'Shape.__init__': False,
    'SolveInfo.__init__': False,
    'TensorDim.__init__': False,
    'ConvergenceException.__init__': False,
    'Diverged.__init__': False,
    'NotConverged.__init__': False,
    'LinearFunction.__init__': False,
    'TensorLike.__variable_attrs__': True,
    'TensorLike.__value_attrs__': True,
    'TensorLike.__with_attrs__': True,
}

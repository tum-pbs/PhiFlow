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

from .backend._optim import Solve, SolveResult, ConvergenceException, NotConverged, Diverged
from .backend._dtype import DType
from .backend import NUMPY_BACKEND, precision, set_global_precision, get_precision

from ._config import GLOBAL_AXIS_ORDER
from ._shape import Shape, spatial_shape, EMPTY_SHAPE, batch_shape, channel_shape, shape
from ._tensors import wrap, tensor, tensors, Tensor, TensorDim
from .extrapolation import Extrapolation
from ._ops import (
    choose_backend_t as choose_backend, all_available, convert, seed,
    native, numpy, reshaped_native, reshaped_tensor, copy, native_call,
    print_ as print, print_gradient,
    map_ as map,
    jit_compile, jit_compile_linear, functional_gradient, custom_gradient,
    zeros, ones, fftfreq, random_normal, random_uniform, meshgrid, linspace, arange as range, range_tensor,  # creation operators (use default backend)
    zeros_like, ones_like,
    batch_stack, spatial_stack, channel_stack, unstack, concat,
    pad,
    join_dimensions, split_dimension, flatten, expand, expand_batch, expand_spatial, expand_channel, transpose,  # reshape operations
    divide_no_nan,
    where, nonzero,
    sum_ as sum, mean, std, prod, max_ as max, min_ as min, any_ as any, all_ as all,  # reduce
    dot,
    abs_ as abs, sign,
    round_ as round, ceil, floor,
    maximum, minimum, clip,
    sqrt, exp, sin, cos, tan, log, log2, log10,
    to_float, to_int, to_complex, imag, real,
    boolean_mask,
    isfinite,
    closest_grid_values, grid_sample, scatter, gather,
    fft, ifft, convolve,
    dtype, cast,
    close, assert_close,
    solve_linear, solve_nonlinear, minimize,
    record_gradients, gradients, stop_gradient
)
from ._nd import (
    shift,
    spatial_sum, vec_abs, vec_squared, cross_product,
    normalize_to,
    l1_loss, l2_loss, l_n_loss, frequency_loss,
    spatial_gradient, laplace,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, sample_subgrid,
    extrapolate_valid_values,
)
from ._trace import LinearFunction


PI = 3.14159265358979323846
"""Value of π to double precision """
pi = PI

NUMPY_BACKEND = NUMPY_BACKEND  # to show up in pdoc
"""Default backend for NumPy arrays and SciPy objects."""

__all__ = [key for key in globals().keys() if not key.startswith('_')]

__pdoc__ = {
    'Extrapolation.__init__': False,
    'Shape.__init__': False,
    'SolveResult.__init__': False,
    'TensorDim.__init__': False,
    'ConvergenceException.__init__': False,
    'Diverged.__init__': False,
    'NotConverged.__init__': False,
    'LinearFunction.__init__': False,
}

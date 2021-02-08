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

from .backend import precision, set_global_precision, get_precision, Solve, LinearSolve, DType, SCIPY_BACKEND

from .extrapolation import Extrapolation

from ._config import GLOBAL_AXIS_ORDER

from ._shape import Shape, spatial_shape, EMPTY_SHAPE, batch_shape, channel_shape, shape
from ._tensors import tensor, tensors, Tensor
from ._functions import (
    choose_backend_t as choose_backend,
    all_available,
    print_ as print,
    transpose,
    zeros, ones, fftfreq, random_normal, random_uniform, meshgrid, linspace,  # creation operators (use default backend)
    batch_stack, spatial_stack, channel_stack, unstack, concat,
    pad, spatial_pad,
    join_dimensions, split_dimension, expand, expand_batch, expand_spatial, expand_channel,  # reshape operations
    divide_no_nan,
    where, nonzero,
    sum_ as sum, mean, std, prod, max_ as max, min_ as min, any_ as any, all_ as all,  # reduce
    zeros_like, ones_like,
    dot,
    matmul,
    einsum,
    abs,
    sign,
    round_ as round, ceil, floor,
    maximum, minimum, clip,
    with_custom_gradient,
    sqrt, exp, sin, cos,
    conv,
    to_float, to_int, to_complex, imag, real,
    boolean_mask,
    isfinite,
    closest_grid_values, grid_sample, scatter,
    fft, ifft,
    dtype, cast,
    tile,
    sparse_tensor,
    close, assert_close,
    solve,
    record_gradients, gradients, stop_gradient
)
from ._nd import (
    shift,
    spatial_sum, vec_abs, vec_squared, cross_product,
    normalize_to,
    l1_loss, l2_loss, l_n_loss, frequency_loss,
    gradient, laplace,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, sample_subgrid,
    extrapolate_valid_values,
    distribute_points
)

PI = 3.14159265358979323846
"""Value of π to double precision """
pi = PI

SCIPY_BACKEND = SCIPY_BACKEND  # to show up in pdoc
"""Default backend for NumPy arrays and SciPy objects."""

__all__ = [key for key in globals().keys() if not key.startswith('_')]

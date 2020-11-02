"""
The phi.math package is the main API for tensor / array operations in PhiFlow.
It provides a common inferface for mathematical operations over tensors which currently supports NumPy, TensorFlow and PyTorch.

Provides

* A tensor base class with multiple implementations
* A NumPy-like API for mathematical operations over tensors as well as tensor generation

The provided operations are not implemented directly.
Instead, they delegate the actual computation to either NumPy, TensorFlow or PyTorch, depending on the configuration.
This allows the user to write simulation code once and have it run with various computation backends.

Main classes:

* Tensor
* Shape
"""

from .backend import DYNAMIC_BACKEND, set_precision
from .backend._scipy_backend import SCIPY_BACKEND

from . import _extrapolation as extrapolation
from ._extrapolation import Extrapolation

from ._config import GLOBAL_AXIS_ORDER

from ._shape import Shape, spatial_shape, infer_shape, EMPTY_SHAPE, batch_shape, channel_shape
from ._tensors import tensor, Tensor, combined_shape, Tensor as Tensor
from ._functions import (
    is_tensor, as_tensor,
    copy,
    print_ as print,
    transpose,
    zeros, ones, fftfreq, random_normal, random_uniform, meshgrid,  # creation operators (use default backend)
    batch_stack, spatial_stack, channel_stack, unstack, concat,
    pad, spatial_pad,
    reshape,
    prod,
    divide_no_nan,
    where,
    sum_ as sum, mean, std,
    zeros_like, ones_like,
    dot,
    matmul,
    einsum,
    abs,
    sign,
    round, ceil, floor,
    max, min, maximum, minimum, clip,
    with_custom_gradient,
    sqrt, exp, sin, cos,
    conv,
    shape, staticshape, ndims,
    to_float, to_int, to_complex, imag, real,
    boolean_mask,
    isfinite,
    closest_grid_values, grid_sample, scatter,
    any_ as any, all_ as all,
    fft, ifft,
    dtype, cast,
    tile, expand_channel,
    sparse_tensor,
    close, assert_close,
    conjugate_gradient,
)
from ._nd import (
    shift,
    indices_tensor,
    normalize_to,
    l1_loss, l2_loss, l_n_loss, frequency_loss,
    gradient, laplace,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, interpolate_linear,
    spatial_sum, vec_abs, vec_squared
)

choose_backend = DYNAMIC_BACKEND.choose_backend
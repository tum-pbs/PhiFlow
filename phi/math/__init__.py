from phi.backend.backend import Backend
from phi.backend.dynamic_backend import DYNAMIC_BACKEND
from phi.backend.scipy_backend import SciPyBackend
from phi.struct.struct_backend import StructBroadcastBackend
from .math_util import types, is_static_shape, zeros, ones, randn, randfreq, interpolate
from .helper import is_scalar, axes, rank
from .nd import (spatial_rank, spatial_dimensions, all_dimensions,
                 indices_tensor,
                 normalize_to,
                 batch_align, batch_align_scalar,
                 blur,
                 l1_loss, l2_loss, l_n_loss, frequency_loss,
                 divergence, gradient, axis_gradient, laplace,
                 fourier_laplace, fourier_poisson, fftfreq, abs_square,
                 downsample2x, upsample2x, interpolate_linear,
                 spatial_sum,)
from .batched import BATCHED, ShapeMismatch
from . import optim


# Setup Backend
DYNAMIC_BACKEND.add_backend(SciPyBackend())
DYNAMIC_BACKEND.add_backend(StructBroadcastBackend(DYNAMIC_BACKEND))


def set_precision(floating_point_bits):
    """
    Sets the floating point precision of DYNAMIC_BACKEND which affects all registered backends.

    If `floating_point_bits` is an integer, all floating point tensors created henceforth will be of the corresponding data type, float16, float32 or float64.
    Operations may also convert floating point values to this precision, even if the input had a different precision.

    If `floating_point_bits` is None, new tensors will default to float32 unless specified otherwise.
    The output of math operations has the same precision as its inputs.

    :param floating_point_bits: one of (16, 32, 64, None)
    """
    DYNAMIC_BACKEND.precision = floating_point_bits


# Enable importing methods directly from math
choose_backend = DYNAMIC_BACKEND.choose_backend

abs = DYNAMIC_BACKEND.abs
add = DYNAMIC_BACKEND.add
all = DYNAMIC_BACKEND.all
any = DYNAMIC_BACKEND.any
as_tensor = DYNAMIC_BACKEND.as_tensor
batch_gather = DYNAMIC_BACKEND.batch_gather
boolean_mask = DYNAMIC_BACKEND.boolean_mask
cast = DYNAMIC_BACKEND.cast
ceil = DYNAMIC_BACKEND.ceil
clip = DYNAMIC_BACKEND.clip
copy = DYNAMIC_BACKEND.copy
cos = DYNAMIC_BACKEND.cos
concat = DYNAMIC_BACKEND.concat
conv = DYNAMIC_BACKEND.conv
dimrange = DYNAMIC_BACKEND.dimrange
div = DYNAMIC_BACKEND.div
divide_no_nan = DYNAMIC_BACKEND.divide_no_nan
dot = DYNAMIC_BACKEND.dot
dtype = DYNAMIC_BACKEND.dtype
einsum = DYNAMIC_BACKEND.einsum
equal = DYNAMIC_BACKEND.equal
exp = DYNAMIC_BACKEND.exp
expand_dims = DYNAMIC_BACKEND.expand_dims
fft = DYNAMIC_BACKEND.fft
flatten = DYNAMIC_BACKEND.flatten
floor = DYNAMIC_BACKEND.floor
gather = DYNAMIC_BACKEND.gather
gather_nd = DYNAMIC_BACKEND.gather_nd
ifft = DYNAMIC_BACKEND.ifft
imag = DYNAMIC_BACKEND.imag
isfinite = DYNAMIC_BACKEND.isfinite
is_tensor = DYNAMIC_BACKEND.is_tensor
matmul = DYNAMIC_BACKEND.matmul
max = DYNAMIC_BACKEND.max
maximum = DYNAMIC_BACKEND.maximum
mean = DYNAMIC_BACKEND.mean
min = DYNAMIC_BACKEND.min
minimum = DYNAMIC_BACKEND.minimum
mul = DYNAMIC_BACKEND.mul
name = DYNAMIC_BACKEND.name
ndims = DYNAMIC_BACKEND.ndims
ones_like = DYNAMIC_BACKEND.ones_like
pad = DYNAMIC_BACKEND.pad
pow = DYNAMIC_BACKEND.pow
py_func = DYNAMIC_BACKEND.py_func
random_uniform = DYNAMIC_BACKEND.random_uniform
range = DYNAMIC_BACKEND.range
real = DYNAMIC_BACKEND.real
resample = DYNAMIC_BACKEND.resample
reshape = DYNAMIC_BACKEND.reshape
round = DYNAMIC_BACKEND.round
sign = DYNAMIC_BACKEND.sign
size = DYNAMIC_BACKEND.size
scatter = DYNAMIC_BACKEND.scatter
shape = DYNAMIC_BACKEND.shape
sin = DYNAMIC_BACKEND.sin
sparse_tensor = DYNAMIC_BACKEND.sparse_tensor
sqrt = DYNAMIC_BACKEND.sqrt
stack = DYNAMIC_BACKEND.stack
staticshape = DYNAMIC_BACKEND.staticshape
std = DYNAMIC_BACKEND.std
sub = DYNAMIC_BACKEND.sub
sum = DYNAMIC_BACKEND.sum
prod = DYNAMIC_BACKEND.prod
tile = DYNAMIC_BACKEND.tile
to_complex = DYNAMIC_BACKEND.to_complex
to_float = DYNAMIC_BACKEND.to_float
to_int = DYNAMIC_BACKEND.to_int
unstack = DYNAMIC_BACKEND.unstack
where = DYNAMIC_BACKEND.where
while_loop = DYNAMIC_BACKEND.while_loop
with_custom_gradient = DYNAMIC_BACKEND.with_custom_gradient
zeros_like = DYNAMIC_BACKEND.zeros_like

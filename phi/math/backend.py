from .base import DynamicBackend
from .scipy_backend import SciPyBackend, as_tensor
from .struct_backend import StructBroadcastBackend


# Setup Backend
backend = DynamicBackend()
backend.backends.append(SciPyBackend())
backend.backends.append(StructBroadcastBackend(backend))

#locals().update(backend)
# Excplicitly unpack namespace from backend
abs = backend.abs
add = backend.add
all = backend.all
any = backend.any
boolean_mask = backend.boolean_mask
cast = backend.cast
ceil = backend.ceil
cos = backend.cos
dtype = backend.dtype
floor = backend.floor
concat = backend.concat
conv = backend.conv
dimrange = backend.dimrange
divide_no_nan = backend.divide_no_nan
dot = backend.dot
exp = backend.exp
expand_dims = backend.expand_dims
fft = backend.fft
flatten = backend.flatten
gather = backend.gather
ifft = backend.ifft
imag = backend.imag
isfinite = backend.isfinite
matmul = backend.matmul
max = backend.max
maximum = backend.maximum
mean = backend.mean
min = backend.min
minimum = backend.minimum
name = backend.name
ones_like = backend.ones_like
pad = backend.pad
py_func = backend.py_func
random_like = backend.random_like
real = backend.real
resample = backend.resample
reshape = backend.reshape
round = backend.round
sign = backend.sign
scatter = backend.scatter
shape = backend.shape
sin = backend.sin
sqrt = backend.sqrt
stack = backend.stack
staticshape = backend.staticshape
std = backend.std
sum = backend.sum
prod = backend.prod
tile = backend.tile
to_complex = backend.to_complex
to_float = backend.to_float
to_int = backend.to_int
unstack = backend.unstack
where = backend.where
while_loop = backend.while_loop
with_custom_gradient = backend.with_custom_gradient
zeros_like = backend.zeros_like

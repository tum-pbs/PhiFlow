
class Backend:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def matches_name(self, name):
        return self.name.lower() == name.lower()

    def is_applicable(self, values):
        return False

    def stack(self, values, axis=0):
        raise NotImplementedError(self)

    def concat(self, values, axis):
        raise NotImplementedError(self)

    def pad(self, value, pad_width, mode="constant", constant_values=0):
        raise NotImplementedError(self)

    def add(self, values):
        raise NotImplementedError(self)

    def reshape(self, value, shape):
        raise NotImplementedError(self)

    def sum(self, value, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def mean(self, value, axis=None):
        raise NotImplementedError(self)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        raise NotImplementedError(self)

    def resample(self, inputs, sample_coords, interpolation="LINEAR", boundary="ZERO"):
        raise NotImplementedError(self)

    def zeros_like(self, tensor):
        raise NotImplementedError(self)

    def ones_like(self, tensor):
        raise NotImplementedError(self)

    def dot(self, a, b, axes):
        raise NotImplementedError(self)

    def matmul(self, A, b):
        raise NotImplementedError(self)

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True,
                   swap_memory=False, name=None, maximum_iterations=None):
        raise NotImplementedError(self)

    def abs(self, x):
        raise NotImplementedError(self)

    def ceil(self, x):
        raise NotImplementedError(self)

    def floor(self, x):
        raise NotImplementedError(self)

    def max(self, x, axis=None):
        raise NotImplementedError(self)

    def maximum(self, a, b):
        raise NotImplementedError(self)

    def minimum(self, a, b):
        raise NotImplementedError(self)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base="custom_gradient_func"):
        raise NotImplementedError(self)

    def sqrt(self, x):
        raise NotImplementedError(self)

    def exp(self, x):
        raise NotImplementedError(self)

    def conv(self, tensor, kernel, padding="SAME"):
        raise NotImplementedError(self)

    def expand_dims(self, a, axis=0):
        raise NotImplementedError(self)

    def shape(self, tensor):
        raise NotImplementedError(self)

    def staticshape(self, tensor):
        raise NotImplementedError(self)

    def to_float(self, x):
        raise NotImplementedError(self)

    def to_int(self, x, int64=False):
        raise NotImplementedError(self)

    def to_complex(self, x):
        raise NotImplementedError(self)

    def dimrange(self, tensor):
        return range(1, len(tensor.shape)-1)

    def gather(self, values, indices):
        raise NotImplementedError(self)

    def flatten(self, x):
        return self.reshape(x, (-1,) )

    def unstack(self, tensor, axis=0):
        raise NotImplementedError(self)

    def std(self, x, axis=None):
        raise NotImplementedError(self)

    def boolean_mask(self, x, mask):
        raise NotImplementedError(self)

    def isfinite(self, x):
        raise NotImplementedError(self)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def fft(self, x):
        """
Computes the n-dimensional FFT along all but the first and last dimensions.
        :param x: tensor of dimension 3 or higher
        """
        raise NotImplementedError(self)

    def ifft(self, k):
        """
Computes the n-dimensional inverse FFT along all but the first and last dimensions.
        :param k: tensor of dimension 3 or higher
        """
        raise NotImplementedError(self)

    def imag(self, complex):
        raise NotImplementedError(self)

    def real(self, complex):
        raise NotImplementedError(self)

    def cast(self, x, dtype):
        raise NotImplementedError(self)



class DynamicBackend(Backend):

    def __init__(self):
        Backend.__init__(self, "Dynamic")
        self.backends = []

    def choose_backend(self, values):
        if not isinstance(values, tuple) and not isinstance(values, list):
            values = [values]
        for backend in self.backends:
            if backend.is_applicable(values):
                return backend
        raise NoBackendFound("No backend found for values %s; registered backends are %s" % (values, self.backends))

    def is_applicable(self, values):
        if not isinstance(values, tuple) and not isinstance(values, list):
            values = [values]
        for backend in self.backends:
            if backend.is_applicable(values):
                return True
        return False

    def stack(self, values, axis=0):
        return self.choose_backend(values).stack(values, axis)

    def concat(self, values, axis):
        return self.choose_backend(values).concat(values, axis)

    def pad(self, value, pad_width, mode="constant", constant_values=0):
        return self.choose_backend(value).pad(value, pad_width, mode, constant_values)

    def add(self, values):
        return self.choose_backend(values).add(values)

    def reshape(self, value, shape):
        return self.choose_backend(value).reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        return self.choose_backend(value).sum(value, axis=axis, keepdims=keepdims)

    def mean(self, value, axis=None):
        return self.choose_backend(value).mean(value, axis)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        return self.choose_backend(inputs).py_func(func, inputs, Tout, shape_out, stateful, name, grad)

    def resample(self, inputs, sample_coords, interpolation="LINEAR", boundary="ZERO"):
        return self.choose_backend((inputs, sample_coords)).resample(inputs, sample_coords, interpolation, boundary)

    def zeros_like(self, tensor):
        return self.choose_backend(tensor).zeros_like(tensor)

    def ones_like(self, tensor):
        return self.choose_backend(tensor).ones_like(tensor)

    def dot(self, a, b, axes):
        return self.choose_backend((a, b)).dot(a, b, axes)

    def matmul(self, A, b):
        return self.choose_backend((A, b)).matmul(A, b)

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True,
                   swap_memory=False, name=None, maximum_iterations=None):
        return self.choose_backend(loop_vars).while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations,
                                                         back_prop, swap_memory, name, maximum_iterations)

    def abs(self, x):
        return self.choose_backend(x).abs(x)

    def ceil(self, x):
        return self.choose_backend(x).ceil(x)

    def floor(self, x):
        return self.choose_backend(x).floor(x)

    def max(self, x, axis=None):
        return self.choose_backend(x).max(x, axis)

    def maximum(self, a, b):
        return self.choose_backend([a,b]).maximum(a, b)

    def minimum(self, a, b):
        return self.choose_backend([a,b]).minimum(a, b)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base="custom_gradient_func"):
        return self.choose_backend(inputs[0]).with_custom_gradient(function, inputs, gradient, input_index, output_index, name_base)

    def sqrt(self, x):
        return self.choose_backend(x).sqrt(x)

    def exp(self, x):
        return self.choose_backend(x).exp(x)

    def conv(self, tensor, kernel, padding="SAME"):
        return self.choose_backend([tensor, kernel]).conv(tensor, kernel, padding)

    def expand_dims(self, a, axis=0):
        return self.choose_backend(a).expand_dims(a, axis)

    def shape(self, tensor):
        return self.choose_backend(tensor).shape(tensor)

    def staticshape(self, tensor):
        return self.choose_backend(tensor).staticshape(tensor)

    def to_float(self, x):
        return self.choose_backend(x).to_float(x)

    def to_int(self, x, int64=False):
        return self.choose_backend(x).to_int(x, int64=int64)

    def to_complex(self, x):
        return self.choose_backend(x).to_complex(x)

    def gather(self, values, indices):
        return self.choose_backend([values, indices]).gather(values, indices)

    def unstack(self, tensor, axis=0):
        return self.choose_backend(tensor).unstack(tensor, axis)

    def std(self, x, axis=None):
        return self.choose_backend(x).std(x, axis)

    def boolean_mask(self, x, mask):
        return self.choose_backend((x, mask)).boolean_mask(x, mask)

    def isfinite(self, x):
        return self.choose_backend(x).isfinite(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return self.choose_backend(boolean_tensor).any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return self.choose_backend(boolean_tensor).all(boolean_tensor, axis=axis, keepdims=keepdims)

    def fft(self, x):
        return self.choose_backend(x).fft(x)

    def ifft(self, k):
        return self.choose_backend(k).ifft(k)

    def imag(self, complex):
        return self.choose_backend(complex).imag(complex)

    def real(self, complex):
        return self.choose_backend(complex).real(complex)

    def cast(self, x, dtype):
        return self.choose_backend(x).cast(x, dtype)


class NoBackendFound(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)



backend = DynamicBackend()

from phi.math.scipy_backend import SciPyBackend, as_tensor
backend.backends.append(SciPyBackend())

abs = backend.abs
add = backend.add
all = backend.all
any = backend.any
boolean_mask = backend.boolean_mask
cast = backend.cast
ceil = backend.ceil
floor = backend.floor
concat = backend.concat
conv = backend.conv
dimrange = backend.dimrange
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
minimum = backend.minimum
name = backend.name
ones_like = backend.ones_like
pad = backend.pad
py_func = backend.py_func
real = backend.real
resample = backend.resample
reshape = backend.reshape
shape = backend.shape
sqrt = backend.sqrt
stack = backend.stack
staticshape = backend.staticshape
std = backend.std
sum = backend.sum
to_complex = backend.to_complex
to_float = backend.to_float
to_int = backend.to_int
unstack = backend.unstack
while_loop = backend.while_loop
with_custom_gradient = backend.with_custom_gradient
zeros_like = backend.zeros_like
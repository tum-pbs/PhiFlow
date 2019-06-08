
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
        raise NotImplementedError()

    def concat(self, values, axis):
        raise NotImplementedError()

    def pad(self, value, pad_width, mode="constant", constant_values=0):
        raise NotImplementedError()

    def add(self, values):
        raise NotImplementedError()

    def reshape(self, value, shape):
        raise NotImplementedError()

    def sum(self, value, axis=None):
        raise NotImplementedError()

    def mean(self, value, axis=None):
        raise NotImplementedError()

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        raise NotImplementedError()

    def resample(self, inputs, sample_coords, interpolation="LINEAR", boundary="ZERO"):
        raise NotImplementedError()

    def zeros_like(self, tensor):
        raise NotImplementedError()

    def ones_like(self, tensor):
        raise NotImplementedError()

    def dot(self, a, b, axes):
        raise NotImplementedError()

    def matmul(self, A, b):
        raise NotImplementedError()

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True,
                   swap_memory=False, name=None, maximum_iterations=None):
        raise NotImplementedError()

    def abs(self, x):
        raise NotImplementedError()

    def ceil(self, x):
        raise NotImplementedError()

    def floor(self, x):
        raise NotImplementedError()

    def max(self, x, axis=None):
        raise NotImplementedError()

    def maximum(self, a, b):
        raise NotImplementedError()

    def minimum(self, a, b):
        raise NotImplementedError()

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base="custom_gradient_func"):
        raise NotImplementedError()

    def sqrt(self, x):
        raise NotImplementedError()

    def exp(self, x):
        raise NotImplementedError()

    def conv(self, tensor, kernel, padding="SAME"):
        raise NotImplementedError()

    def expand_dims(self, a, axis=0):
        raise NotImplementedError()

    def shape(self, tensor):
        raise NotImplementedError()

    def to_float(self, x):
        raise NotImplementedError()

    def to_int(self, x, int64=False):
        raise NotImplementedError()

    def dimrange(self, tensor):
        return range(1, len(tensor.shape)-1)

    def gather(self, values, indices):
        raise NotImplementedError()

    def flatten(self, x):
        return self.reshape(x, (-1,) )

    def unstack(self, tensor, axis=0):
        raise NotImplementedError()

    def std(self, x, axis=None):
        raise NotImplementedError()

    def boolean_mask(self, x, mask):
        raise NotImplementedError()

    def isfinite(self, x):
        raise NotImplementedError()

    def any(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError()

    def all(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError()



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

    def sum(self, value, axis=None):
        return self.choose_backend(value).sum(value, axis)

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

    def to_float(self, x):
        return self.choose_backend(x).to_float(x)

    def to_int(self, x, int64=False):
        return self.choose_backend(x).to_int(x, int64=int64)

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
ceil = backend.ceil
floor = backend.floor
concat = backend.concat
conv = backend.conv
dimrange = backend.dimrange
dot = backend.dot
exp = backend.exp
expand_dims = backend.expand_dims
flatten = backend.flatten
gather = backend.gather
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
resample = backend.resample
reshape = backend.reshape
shape = backend.shape
sqrt = backend.sqrt
stack = backend.stack
std = backend.std
sum = backend.sum
to_float = backend.to_float
to_int = backend.to_int
unstack = backend.unstack
while_loop = backend.while_loop
with_custom_gradient = backend.with_custom_gradient
zeros_like = backend.zeros_like
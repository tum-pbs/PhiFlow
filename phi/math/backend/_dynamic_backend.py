import warnings

from ._backend import Backend


class NoBackendFound(Exception):

    def __init__(self, msg):
        Exception.__init__(self, msg)


class DynamicBackend(Backend):

    def __init__(self):
        self.backends = []
        self.default_backend: Backend = None
        Backend.__init__(self, 'Dynamic')

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision
        for backend in self.backends:
            backend.precision = precision

    def choose_backend(self, values: list or tuple, creation=False) -> Backend:
        if not isinstance(values, (list, tuple)):
            values = [values]

        def applicable(backend):
            for value in values:
                if not backend.is_tensor(value, only_native=False):
                    return False
            return True

        # --- Default Backend has priority ---
        if creation and self.default_backend is not None and applicable(self.default_backend):
            return self.default_backend
        # --- Filter out non-applicable ---
        backends = list(backend for backend in self.backends if applicable(backend))
        if len(backends) == 0:
            list(backend for backend in self.backends if applicable(backend))
            raise NoBackendFound('No backend found for values %s; registered backends are %s' % (values, self.backends))
        # --- Native tensors? ---
        for backend in backends:
            for value in values:
                if backend.is_tensor(value, only_native=True):
                    return backend
        else:
            return backends[0]

    def add_backend(self, backend, priority=None):
        for existing in self.backends:
            if existing.name == backend.name:
                return False
        backend.precision = self.precision
        if priority is None:
            self.backends.append(backend)
        else:
            self.backends.insert(0, backend)
        return True

    def is_tensor(self, x, only_native=False):
        try:
            return self.choose_backend(x).is_tensor(x, only_native=only_native)
        except NoBackendFound:
            return False

    def as_tensor(self, x, convert_external=True):
        return self.choose_backend(x).as_tensor(x, convert_external=convert_external)

    def numpy(self, tensor):
        return self.choose_backend(tensor).numpy(tensor)

    def copy(self, tensor, only_mutable=False):
        return self.choose_backend(tensor).copy(tensor, only_mutable=only_mutable)

    def transpose(self, tensor, axes):
        return self.choose_backend(tensor).transpose(tensor, axes=axes)

    def equal(self, x, y):
        return self.choose_backend([x, y]).equal(x, y)

    def random_uniform(self, shape):
        return self.choose_backend(shape).random_uniform(shape)

    def random_normal(self, shape):
        return self.choose_backend(shape).random_normal(shape)

    def stack(self, values, axis=0):
        return self.choose_backend(values).stack(values, axis)

    def concat(self, values, axis):
        return self.choose_backend(values).concat(values, axis)

    def tile(self, value, multiples):
        return self.choose_backend(value).tile(value, multiples)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        return self.choose_backend(value).pad(value, pad_width, mode, constant_values)

    def reshape(self, value, shape):
        return self.choose_backend(value).reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        return self.choose_backend(value).sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        return self.choose_backend(value).prod(value, axis)

    def divide_no_nan(self, x, y):
        return self.choose_backend([x, y]).divide_no_nan(x, y)

    def where(self, condition, x=None, y=None):
        # For Tensorflow x,y the condition can be a Numpy array, but not the other way around. If possible, choose backend based on first input, otherwise based on condition.
        return self.choose_backend([condition, x, y]).where(condition, x, y)

    def mean(self, value, axis=None, keepdims=False):
        return self.choose_backend(value).mean(value, axis, keepdims=keepdims)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        return self.choose_backend(inputs).py_func(func, inputs, Tout, shape_out, stateful, name, grad)

    def resample(self, inputs, sample_coords, interpolation='linear', boundary='constant', constant_values=0):
        return self.choose_backend([inputs, sample_coords]).resample(inputs, sample_coords, interpolation=interpolation, boundary=boundary, constant_values=constant_values)

    def range(self, start, limit=None, delta=1, dtype=None):
        possible_tensors = [t for t in (start, limit, delta) if t is not None]
        return self.choose_backend(possible_tensors, creation=True).range(start, limit, delta, dtype)

    def zeros(self, shape, dtype=None):
        return self.choose_backend([], creation=True).zeros(shape, dtype)

    def zeros_like(self, tensor):
        return self.choose_backend(tensor, creation=True).zeros_like(tensor)

    def ones(self, shape, dtype=None):
        return self.choose_backend([], creation=True).zeros(shape, dtype)

    def ones_like(self, tensor):
        return self.choose_backend(tensor, creation=True).ones_like(tensor)

    def meshgrid(self, *coordinates):
        return self.choose_backend(coordinates, creation=True).meshgrid(*coordinates)

    def dot(self, a, b, axes):
        return self.choose_backend([a, b]).dot(a, b, axes)

    def matmul(self, A, b):
        return self.choose_backend([A, b]).matmul(A, b)

    def einsum(self, equation, *tensors):
        return self.choose_backend(tensors).einsum(equation, *tensors)

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None, maximum_iterations=None):
        return self.choose_backend(loop_vars).while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations)

    def abs(self, x):
        return self.choose_backend(x).abs(x)

    def sign(self, x):
        return self.choose_backend(x).sign(x)

    def round(self, x):
        return self.choose_backend(x).round(x)

    def ceil(self, x):
        return self.choose_backend(x).ceil(x)

    def floor(self, x):
        return self.choose_backend(x).floor(x)

    def max(self, x, axis=None, keepdims=False):
        return self.choose_backend(x).max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return self.choose_backend(x).min(x, axis=axis, keepdims=keepdims)

    def maximum(self, a, b):
        return self.choose_backend([a,b]).maximum(a, b)

    def minimum(self, a, b):
        return self.choose_backend([a,b]).minimum(a, b)

    def clip(self, x, minimum, maximum):
        return self.choose_backend([x, minimum, maximum]).clip(x, minimum, maximum)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
        return self.choose_backend(inputs[0]).with_custom_gradient(function, inputs, gradient, input_index, output_index, name_base)

    def sqrt(self, x):
        return self.choose_backend(x).sqrt(x)

    def exp(self, x):
        return self.choose_backend(x).exp(x)

    def conv(self, tensor, kernel, padding='SAME'):
        return self.choose_backend([tensor, kernel]).conv(tensor, kernel, padding)

    def expand_dims(self, a, axis=0, number=1):
        return self.choose_backend(a).expand_dims(a, axis, number)

    def shape(self, tensor):
        return self.choose_backend(tensor).shape(tensor)

    def to_float(self, x, float64=False):
        if float64: warnings.warn('float64 argument is deprecated, set Backend.precision = 64 to use 64 bit operations.', DeprecationWarning)
        return self.choose_backend(x).to_float(x, float64=float64)

    def staticshape(self, tensor):
        return self.choose_backend(tensor).staticshape(tensor)

    def to_int(self, x, int64=False):
        return self.choose_backend(x).to_int(x, int64=int64)

    def to_complex(self, x):
        return self.choose_backend(x).to_complex(x)

    def gather(self, values, indices):
        return self.choose_backend([values]).gather(values, indices)

    def gather_nd(self, values, indices, batch_dims=0):
        return self.choose_backend([values]).gather_nd(values, indices, batch_dims=batch_dims)

    def unstack(self, tensor, axis=0, keepdims=False):
        return self.choose_backend(tensor).unstack(tensor, axis, keepdims=keepdims)

    def std(self, x, axis=None, keepdims=False):
        return self.choose_backend(x).std(x, axis, keepdims=keepdims)

    def boolean_mask(self, x, mask):
        return self.choose_backend([x, mask]).boolean_mask(x, mask)

    def isfinite(self, x):
        return self.choose_backend(x).isfinite(x)

    def scatter(self, points, indices, values, shape, duplicates_handling='undefined'):
        return self.choose_backend([points, indices, values]).scatter(points, indices, values, shape, duplicates_handling=duplicates_handling)

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

    def sin(self, x):
        return self.choose_backend(x).sin(x)

    def cos(self, x):
        return self.choose_backend(x).cos(x)

    def sparse_tensor(self, indices, values, shape):
        if isinstance(indices, (tuple, list)):
            backend = self.choose_backend([values, *indices])
        else:
            backend = self.choose_backend([indices, values])
        return backend.sparse_tensor(indices, values, shape)

    def coordinates(self, tensor, unstack_coordinates=False):
        return self.choose_backend(tensor).coordinates(tensor, unstack_coordinates)

    def conjugate_gradient(self, A, y, x0, relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, gradient: str = 'implicit', callback=None):
        backend = self.choose_backend([y, x0]) if callable(A) else self.choose_backend([A, y, x0])
        return backend.conjugate_gradient(A, y, x0, relative_tolerance, absolute_tolerance, max_iterations, gradient, callback)

    def dtype(self, array):
        return self.choose_backend(array).dtype(array)

    def add(self, a, b):
        return self.choose_backend([a, b]).add(a, b)

    def sub(self, a, b):
        return self.choose_backend([a, b]).sub(a, b)

    def mul(self, a, b):
        return self.choose_backend([a, b]).mul(a, b)

    def div(self, numerator, denominator):
        return self.choose_backend([numerator, denominator]).div(numerator, denominator)

    def pow(self, base, exp):
        return self.choose_backend([base, exp]).pow(base, exp)

    def mod(self, dividend, divisor):
        return self.choose_backend([dividend, divisor]).mod(dividend, divisor)


DYNAMIC_BACKEND = DynamicBackend()


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

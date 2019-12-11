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

    # --- Abstract math functions ---

    def is_tensor(self, x):
        raise NotImplementedError()

    def as_tensor(self, x):
        raise NotImplementedError()

    def random_uniform(self, shape):
        raise NotImplementedError(self)

    def stack(self, values, axis=0):
        raise NotImplementedError(self)

    def concat(self, values, axis):
        raise NotImplementedError(self)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        """
        Pad a tensor.

        :param value:
        :param pad_width: 2D tensor specifying the number of values padded to the edges of each axis in the form [[before axis 0, after axis 0], ...].
        :param mode: 'constant', 'symmetric', 'reflect', 'wrap'
        :param constant_values: used for out-of-bounds points if mode='constant'
        """
        raise NotImplementedError(self)

    def add(self, values):
        raise NotImplementedError(self)

    def reshape(self, value, shape):
        raise NotImplementedError(self)

    def sum(self, value, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def prod(self, value, axis=None):
        raise NotImplementedError(self)

    def divide_no_nan(self, x, y):
        raise NotImplementedError(self)

    def where(self, condition, x=None, y=None):
        raise NotImplementedError(self)

    def mean(self, value, axis=None):
        raise NotImplementedError(self)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        raise NotImplementedError(self)

    def resample(self, inputs, sample_coords, interpolation='LINEAR', boundary='zero'):
        raise NotImplementedError(self)

    def range(self, start, limit=None, delta=1, dtype=None):
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

    def sign(self, x):
        raise NotImplementedError(self)

    def round(self, x):
        raise NotImplementedError(self)

    def ceil(self, x):
        raise NotImplementedError(self)

    def floor(self, x):
        raise NotImplementedError(self)

    def max(self, x, axis=None):
        raise NotImplementedError(self)

    def min(self, x, axis=None):
        raise NotImplementedError(self)

    def maximum(self, a, b):
        raise NotImplementedError(self)

    def minimum(self, a, b):
        raise NotImplementedError(self)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
        raise NotImplementedError(self)

    def sqrt(self, x):
        raise NotImplementedError(self)

    def exp(self, x):
        raise NotImplementedError(self)

    def conv(self, tensor, kernel, padding='SAME'):
        raise NotImplementedError(self)

    def expand_dims(self, a, axis=0, number=1):
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
        return range(1, len(tensor.shape) - 1)

    def gather(self, values, indices):
        raise NotImplementedError(self)

    def gather_nd(self, values, indices):
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

    def scatter(self, points, indices, values, shape, duplicates_handling='undefined'):
        """
        This method expects the first dimension of indices and values to be the batch dimension.
        The batch dimension need not be specified in the indices array.

        All indices must be non-negative and are expected to be within bounds. Otherwise the behaviour is undefined.

        :param indices:
        :param values:
        :param shape:
        :param duplicates_handling: one of ('undefined', 'add', 'mean', 'any', 'last', 'no duplicates')
        """
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

    def sin(self, x):
        raise NotImplementedError(self)

    def cos(self, x):
        raise NotImplementedError(self)

    def dtype(self, array):
        raise NotImplementedError(self)

    def tile(self, value, multiples):
        raise NotImplementedError(self)

    # --- Math function with default implementation ---

    def ndims(self, tensor):
        return len(self.staticshape(tensor))

    def size(self, array):
        return self.prod(self.shape(array))

    def batch_gather(self, tensor, batches):
        if isinstance(batches, int):
            batches = [batches]
        return tensor[batches, ...]


class DynamicBackend(Backend):

    def __init__(self):
        Backend.__init__(self, 'Dynamic')
        self.backends = []

    def choose_backend(self, values):
        if not isinstance(values, tuple) and not isinstance(values, list):
            values = [values]
        for backend in self.backends:
            if backend.is_applicable(values):
                return backend
        raise NoBackendFound('No backend found for values %s; registered backends are %s' % (values, self.backends))

    def add_backend(self, backend):
        for existing in self.backends:
            if existing.name == backend.name:
                return False
        self.backends.append(backend)
        return True

    def is_applicable(self, values):
        if not isinstance(values, tuple) and not isinstance(values, list):
            values = [values]
        for backend in self.backends:
            if backend.is_applicable(values):
                return True
        return False

    def is_tensor(self, x):
        try:
            return self.choose_backend(x).is_tensor(x)
        except NoBackendFound:
            return False

    def as_tensor(self, x):
        return self.choose_backend(x).as_tensor(x)

    def random_uniform(self, tensor):
        return self.choose_backend(tensor).random_uniform(tensor)

    def stack(self, values, axis=0):
        return self.choose_backend(values).stack(values, axis)

    def concat(self, values, axis):
        return self.choose_backend(values).concat(values, axis)

    def tile(self, value, multiples):
        return self.choose_backend(value).tile(value, multiples)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        return self.choose_backend(value).pad(value, pad_width, mode, constant_values)

    def add(self, values):
        return self.choose_backend(values).add(values)

    def reshape(self, value, shape):
        return self.choose_backend(value).reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        return self.choose_backend(value).sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        return self.choose_backend(value).prod(value, axis)

    def divide_no_nan(self, x, y):
        return self.choose_backend((x,y)).divide_no_nan(x, y)

    def where(self, condition, x=None, y=None):
        # For Tensorflow x,y the condition can be a Numpy array, but not the other way around. If possible, choose backend based on first input, otherwise based on condition.
        return self.choose_backend((condition, x, y)).where(condition, x, y)

    def mean(self, value, axis=None):
        return self.choose_backend(value).mean(value, axis)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        return self.choose_backend(inputs).py_func(func, inputs, Tout, shape_out, stateful, name, grad)

    def resample(self, inputs, sample_coords, interpolation='LINEAR', boundary='zero'):
        return self.choose_backend((inputs, sample_coords)).resample(inputs, sample_coords, interpolation, boundary)

    def range(self, start, limit=None, delta=1, dtype=None):
        return self.choose_backend((start, limit, delta)).range(start, limit, delta, dtype)

    def zeros_like(self, tensor):
        return self.choose_backend(tensor).zeros_like(tensor)

    def ones_like(self, tensor):
        return self.choose_backend(tensor).ones_like(tensor)

    def dot(self, a, b, axes):
        return self.choose_backend((a, b)).dot(a, b, axes)

    def matmul(self, A, b):
        return self.choose_backend((A, b)).matmul(A, b)

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

    def max(self, x, axis=None):
        return self.choose_backend(x).max(x, axis)

    def min(self, x, axis=None):
        return self.choose_backend(x).min(x, axis)

    def maximum(self, a, b):
        return self.choose_backend([a,b]).maximum(a, b)

    def minimum(self, a, b):
        return self.choose_backend([a,b]).minimum(a, b)

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
        return self.choose_backend(x).to_float(x, float64=float64)

    def staticshape(self, tensor):
        return self.choose_backend(tensor).staticshape(tensor)

    def to_int(self, x, int64=False):
        return self.choose_backend(x).to_int(x, int64=int64)

    def to_complex(self, x):
        return self.choose_backend(x).to_complex(x)

    def gather(self, values, indices):
        return self.choose_backend([values]).gather(values, indices)

    def gather_nd(self, values, indices):
        return self.choose_backend([values]).gather_nd(values, indices)

    def unstack(self, tensor, axis=0):
        return self.choose_backend(tensor).unstack(tensor, axis)

    def std(self, x, axis=None):
        return self.choose_backend(x).std(x, axis)

    def boolean_mask(self, x, mask):
        return self.choose_backend((x, mask)).boolean_mask(x, mask)

    def isfinite(self, x):
        return self.choose_backend(x).isfinite(x)

    def scatter(self, points, indices, values, shape, duplicates_handling='undefined'):
        return self.choose_backend(points).scatter(points, indices, values, shape, duplicates_handling=duplicates_handling)

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

    def dtype(self, array):
        return self.choose_backend(array).dtype(array)


class NoBackendFound(Exception):

    def __init__(self, msg):
        Exception.__init__(self, msg)


DYNAMIC_BACKEND = DynamicBackend()

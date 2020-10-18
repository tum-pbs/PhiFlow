import numpy as np


class Backend:

    def __init__(self, name, precision=32):
        self._name = name
        self.precision = precision

    @property
    def name(self):
        return self._name

    @property
    def precision(self):
        """
        If `precision` is an integer, any Backend method may convert floating point values to this precision, even if the input had a different precision.

        If `precision` is `None`, the output of math operations has the same precision as its inputs.
        """
        return self._precision

    @precision.setter
    def precision(self, precision):
        """
        If `precision` is an integer, any Backend method may convert floating point values to this precision, even if the input had a different precision.

        If `precision` is `None`, the output of math operations has the same precision as its inputs.

        :param precision: one of (16, 32, 64, None)
        """
        assert precision in (None, 16, 32, 64)
        self._precision = precision

    @property
    def float_type(self):
        return {16: np.float16, 32: np.float32, 64: np.float64, None: np.float32}[self.precision]

    @property
    def complex_type(self):
        return {16: np.complex64, 32: np.complex64, 64: np.complex128, None: np.complex64}[self.precision]

    def combine_types(self, *dtypes):
        # all bool?
        if all(dt.kind == 'b' for dt in dtypes):
            return dtypes[0]
        # all int / bool?
        if all(dt.kind == 'i' or dt.kind == 'b' for dt in dtypes):
            largest = max(dtypes, key=lambda dt: dt.itemsize)
            return largest
        # all real?
        if all(dt.kind == 'f' or dt.kind == 'i' or dt.kind == 'b' for dt in dtypes):
            return self.float_type
        # complex
        if all(dt.kind == 'c' or dt.kind == 'f' or dt.kind == 'i' or dt.kind == 'b' for dt in dtypes):
            return self.complex_type
        raise ValueError(dtypes)

    def auto_cast(self, *tensors):
        """
        Determins the appropriate data type resulting from operations involving the tensors as input.

        This method is called by the default implementations of basic operators.
        Backends can override this method to prevent unnecessary casting.
        """
        dtypes = [self.dtype(t) for t in tensors]
        result_type = self.combine_types(*dtypes)
        tensors = [self.cast(t, result_type) for t in tensors]
        return tensors

    @property
    def has_fixed_precision(self):
        return self.precision is not None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def matches_name(self, name):
        return self.name.lower() == name.lower()

    # --- Abstract math functions ---

    def is_tensor(self, x, only_native=False):
        """
        An object is considered a native tensor by a backend if no internal conversion is required by backend methods.
        An object is considered a tensor (nativer or otherwise) by a backend if it is not a struct (e.g. tuple, list) and all methods of the backend accept it as a tensor argument.

        :param x: object to check
        :param only_native: If True, only accepts true native tensor representations, not Python numbers or others that are also supported as tensors
        :return: whether `x` is considered a tensor by this backend
        :rtype: bool
        """
        raise NotImplementedError()

    def as_tensor(self, x, convert_external=True):
        """
        Converts a tensor-like object to the native tensor representation of this backend.
        If x is a native tensor of this backend, it is returned without modification.
        If x is a Python number (numbers.Number instance), `convert_numbers` decides whether to convert it unless the backend cannot handle Python numbers.

        *Note:* There may be objects that are considered tensors by this backend but are not native and thus, will be converted by this method.

        :param x: tensor-like, e.g. list, tuple, Python number, tensor
        :param convert_external: if False and `x` is a Python number that is understood by this backend, this method returns the number as is. This can help prevent type clashes like int32 vs int64.
        :return: tensor representation of `x`
        """
        raise NotImplementedError()

    def numpy(self, tensor):
        raise NotImplementedError()

    def copy(self, tensor, only_mutable=False):
        raise NotImplementedError()

    def transpose(self, tensor, axes):
        raise NotImplementedError()

    def equal(self, x, y):
        raise NotImplementedError()

    def random_uniform(self, shape):
        raise NotImplementedError(self)

    def random_normal(self, shape):
        raise NotImplementedError(self)

    def stack(self, values, axis=0):
        raise NotImplementedError(self)

    def concat(self, values, axis):
        raise NotImplementedError(self)

    def pad(self, value, pad_width, mode: str = 'constant', constant_values=0):
        """
        Pad a tensor with values as specified by `mode` and `constant_values`.
        :param value: tensor
        :param pad_width: 2D tensor specifying the number of values padded to the edges of each axis in the form [[axis 0 lower, axis 0 upper], ...] including batch and component axes.
        :param mode: 'constant', 'boundary', 'periodic', 'symmetric', 'reflect'
        :param constant_values: used for out-of-bounds points if mode='constant'
        """
        raise NotImplementedError(self)

    def reshape(self, value, shape):
        raise NotImplementedError(self)

    def sum(self, value, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def prod(self, value, axis=None):
        raise NotImplementedError(self)

    def divide_no_nan(self, x, y):
        """ Computes x/y but returns 0 if y=0. """
        raise NotImplementedError(self)

    def where(self, condition, x=None, y=None):
        raise NotImplementedError(self)

    def mean(self, value, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        raise NotImplementedError(self)

    def range(self, start, limit=None, delta=1, dtype=None):
        raise NotImplementedError(self)

    def zeros(self, shape, dtype=None):
        raise NotImplementedError(self)

    def zeros_like(self, tensor):
        raise NotImplementedError(self)

    def ones(self, shape, dtype=None):
        raise NotImplementedError(self)

    def ones_like(self, tensor):
        raise NotImplementedError(self)

    def meshgrid(self, *coordinates):
        raise NotImplementedError(self)

    def dot(self, a, b, axes):
        raise NotImplementedError(self)

    def matmul(self, A, b):
        raise NotImplementedError(self)

    def einsum(self, equation, *tensors):
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

    def max(self, x, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def min(self, x, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def maximum(self, a, b):
        raise NotImplementedError(self)

    def minimum(self, a, b):
        raise NotImplementedError(self)

    def clip(self, x, minimum, maximum):
        raise NotImplementedError(self)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
        raise NotImplementedError(self)

    def sqrt(self, x):
        raise NotImplementedError(self)

    def exp(self, x):
        raise NotImplementedError(self)

    def conv(self, tensor, kernel, padding='same'):
        raise NotImplementedError(self)

    def expand_dims(self, a, axis=0, number=1):
        raise NotImplementedError(self)

    def shape(self, tensor):
        raise NotImplementedError(self)

    def staticshape(self, tensor):
        raise NotImplementedError(self)

    def to_float(self, x, float64=False):
        """
Converts a tensor to floating point values.
If this Backend uses a fixed precision, the tensor will be converted to that precision.
Otherwise, non-float inputs are converted to float32 (unless `float64=True`).

If `x` is mutable and of the correct floating type, returns a copy of `x`.

To convert float tensors to the backend precision but leave non-float tensors untouched, use `Backend.as_tensor()`.
        :param x: tensor
        :param float64: deprecated. Set Backend.precision = 64 to use 64 bit operations.
        """
        raise NotImplementedError(self)

    def to_int(self, x, int64=False):
        raise NotImplementedError(self)

    def to_complex(self, x):
        raise NotImplementedError(self)

    def dimrange(self, tensor):
        return range(1, len(tensor.shape) - 1)

    def gather(self, values, indices):
        raise NotImplementedError(self)

    def gather_nd(self, values, indices, batch_dims=0):
        raise NotImplementedError(self)

    def flatten(self, x):
        return self.reshape(x, (-1,))

    def std(self, x, axis=None, keepdims=False):
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
        """
Repeats the tensor along each axis the number of times given by multiples.
If `multiples` has more dimensions than `value`, these dimensions are added to `value` as outer dimensions.
        :param value: tensor
        :param multiples: tuple or list of integers
        :return: tile tensor
        """
        raise NotImplementedError(self)

    def sparse_tensor(self, indices, values, shape):
        """

        :param indices: tuple/list matching the dimensions (pair for matrix)
        :param values:
        :param shape:
        """
        raise NotImplementedError(self)

    def coordinates(self, tensor, unstack_coordinates=False):
        """
        Returns the coordinates and data of a tensor.

        The first returned value is a tensor holding the coordinate vectors in the last dimension if unstack_coordinates=False.
        In case unstack_coordinates=True, the coordiantes are returned as a tuple of tensors, i.e. (row, col) for matrices

        :param tensor: dense or sparse tensor
        :return: indices (tensor or tuple), data
        """
        raise NotImplementedError(self)

    def conjugate_gradient(self, A, y, x0, relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, gradient: str = 'implicit', callback=None):
        """
        Solve the system of linear equations
          A * x = y

        :param A: batch of sparse / dense matrices or or linear function A(x). 3rd order tensor or list of matrices.
        :param y: target result of A * x. 2nd order tensor (batch, vector) or list of vectors.
        :param x0: initial guess for x. 2nd order tensor (batch, vector) or list of vectors.
        :param relative_tolerance: stops when norm(residual) <= max(relative_tolerance * norm(y), absolute_tolerance)
        :param absolute_tolerance: stops when norm(residual) <= max(relative_tolerance * norm(y), absolute_tolerance)
        :param max_iterations: maximum number of iterations or None for unlimited
        :param gradient: one of ('implicit', 'inverse', 'autodiff')
        :param callback: Function to call after each iteration. It is called with the current solution as callback(x).
        :rtype: tuple
        :return: converged, solution, iterations
        """
        raise NotImplementedError(self)

    # --- Math function with default implementation ---

    def resample(self, inputs, sample_coords, interpolation='linear', boundary='constant', constant_values=0):
        """
        Interpolates a regular grid at the specified coordinates.
        :param inputs: grid data
        :param sample_coords: tensor of floating grid indices. The last dimension must match the dimensions of inputs. The first grid point of dimension i lies at position 0, the last at data.shape[i]-1.
        :param interpolation: only 'linear' is currently supported
        :param boundary: values to use for coordinates outside the grid, can be specified for each face, options are 'constant', 'boundary', 'periodic', 'symmetric', 'reflect'
        :param constant_values: Value used for constant boundaries, can be specified for each face
        """
        from . import general_grid_sample_nd
        assert interpolation == 'linear'
        return general_grid_sample_nd(inputs, sample_coords, boundary, constant_values, self)

    def ndims(self, tensor):
        return len(self.staticshape(tensor))

    def size(self, array):
        return self.prod(self.shape(array))

    def batch_gather(self, tensor, batches):
        if isinstance(batches, int):
            batches = [batches]
        return tensor[batches, ...]

    def unstack(self, tensor, axis=0, keepdims=False):
        if axis < 0:
            axis += len(tensor.shape)
        if axis >= len(tensor.shape) or axis < 0:
            raise ValueError("Illegal axis value")
        result = []
        for slice_idx in range(tensor.shape[axis]):
            if keepdims:
                component = tensor[tuple([slice(slice_idx, slice_idx + 1) if d == axis else slice(None) for d in range(len(tensor.shape))])]
            else:
                component = tensor[tuple([slice_idx if d == axis else slice(None) for d in range(len(tensor.shape))])]
            result.append(component)
        return tuple(result)

    def add(self, a, b):
        a, b = self.auto_cast(a, b)
        return a + b

    def sub(self, a, b):
        a, b = self.auto_cast(a, b)
        return a - b

    def mul(self, a, b):
        a, b = self.auto_cast(a, b)
        return a * b

    def div(self, numerator, denominator):
        numerator, denominator = self.auto_cast(numerator, denominator)
        return numerator / denominator

    def pow(self, base, exp):
        base, exp = self.auto_cast(base, exp)
        return base ** exp

    def mod(self, dividend, divisor):
        dividend, divisor = self.auto_cast(dividend, divisor)
        return dividend % divisor

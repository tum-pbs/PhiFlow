from contextlib import contextmanager
from typing import Tuple, List, Any, Callable

import numpy

from ._dtype import DType, combine_types
from ._optim import Solve, LinearSolve


class ComputeDevice:
    """
    A physical device that can be selected to perform backend computations.
    """

    def __init__(self, backend: 'Backend', name: str, device_type: str, memory: int, processor_count: int, description: str, ref=None):
        self.name: str = name
        """ Name of the compute device. CPUs are typically called `'CPU'`. """
        self.device_type: str = device_type
        """ Type of device such as `'CPU'`, `'GPU'` or `'TPU'`. """
        self.memory: int = memory
        """ Maximum memory of the device that can be allocated (in bytes). -1 for n/a. """
        self.processor_count: int = processor_count
        """ Number of CPU cores or GPU multiprocessors. -1 for n/a. """
        self.description: str = description
        """ Further information about the device such as driver version. """
        self.ref = ref
        """ (Optional) Reference to the internal device representation. """
        self.backend: 'Backend' = backend
        """ Backend that this device belongs to. Different backends represent the same device with different objects. """

    def __repr__(self):
        mem = f"{(self.memory / 1024 ** 2)} MB" if self.memory > 0 else "memory: n/a"
        pro = f"{self.processor_count} processors" if self.processor_count > 0 else "processors: n/a"
        descr = self.description.replace('\n', '  ')
        if len(descr) > 30:
            descr = descr[:28] + "..."
        return f"'{self.name}' ({self.device_type}) | {mem} | {pro} | {descr}"


class Backend:

    def __init__(self, name: str, default_device: ComputeDevice):
        """
        Backends delegate low-level operations to a compute library or emulate them.

        The methods of `Backend` form a comprehensive list of available operations.

        To support a compute library, subclass `Backend` and register it by adding it to `BACKENDS`.

        Args:
            name: Human-readable string
            default_device: `ComputeDevice` being used by default
        """
        self._name = name
        self._default_device = default_device

    def __enter__(self):
        _DEFAULT.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _DEFAULT.pop(-1)

    @property
    def name(self) -> str:
        return self._name

    def supports(self, feature: str or Callable) -> bool:
        """
        Tests if this backend supports the given feature.
        Features correspond to a method of this backend that must be implemented if the feature is supported.

        Possible features:

        * `sparse_tensor`
        * `gradients

        Args:
            feature: `str` or unbound Backend method, e.g. `Backend.sparse_tensor`

        Returns:
            Whether the feature is supported.
        """
        feature = feature if isinstance(feature, str) else feature.__name__
        if not hasattr(Backend, feature):
            raise ValueError(f"Not a valid feature: '{feature}'")
        backend_fun = getattr(Backend, feature)
        impl_fun = getattr(self.__class__, feature)
        return impl_fun is not backend_fun

    @property
    def precision(self) -> int:
        """ Short for math.backend.get_precision() """
        return get_precision()

    @property
    def float_type(self) -> DType:
        return DType(float, self.precision)

    @property
    def complex_type(self) -> DType:
        return DType(complex, max(64, self.precision))

    def combine_types(self, *dtypes: DType) -> DType:
        return combine_types(*dtypes, fp_precision=self.precision)

    def auto_cast(self, *tensors) -> list:
        """
        Determins the appropriate values type resulting from operations involving the tensors as input.
        
        This method is called by the default implementations of basic operators.
        Backends can override this method to prevent unnecessary casting.

        Args:
          *tensors: tensors to cast and to consider when determining the common data type

        Returns:
            tensors cast to a common data type
        """
        dtypes = [self.dtype(t) for t in tensors]
        result_type = self.combine_types(*dtypes)
        if result_type.kind in (int, float, complex, bool):
            tensors = [self.cast(t, result_type) for t in tensors]
        return tensors

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def list_devices(self, device_type: str or None = None) -> List[ComputeDevice]:
        """
        Fetches information about all available compute devices this backend can use.

        Args:
            device_type: (optional) Return only devices of this type, e.g. `'GPU'`. See `ComputeDevice.device_type`.

        Returns:
            Tuple of all currently available devices.
        """
        raise NotImplementedError()

    def get_default_device(self) -> ComputeDevice:
        return self._default_device

    def set_default_device(self, device: ComputeDevice or str):
        if isinstance(device, str):
            devices = self.list_devices(device)
            assert len(devices) >= 1, f"{self.name}: Cannot select '{device} because no device of this type is available."
            device = devices[0]
        self._default_device = device

    def is_tensor(self, x, only_native=False):
        """
        An object is considered a native tensor by a backend if no internal conversion is required by backend methods.
        An object is considered a tensor (nativer or otherwise) by a backend if it is not a struct (e.g. tuple, list) and all methods of the backend accept it as a tensor argument.

        Args:
          x: object to check
          only_native: If True, only accepts true native tensor representations, not Python numbers or others that are also supported as tensors (Default value = False)

        Returns:
          bool: whether `x` is considered a tensor by this backend

        """
        raise NotImplementedError()

    def as_tensor(self, x, convert_external=True):
        """
        Converts a tensor-like object to the native tensor representation of this backend.
        If x is a native tensor of this backend, it is returned without modification.
        If x is a Python number (numbers.Number instance), `convert_numbers` decides whether to convert it unless the backend cannot handle Python numbers.
        
        *Note:* There may be objects that are considered tensors by this backend but are not native and thus, will be converted by this method.

        Args:
          x: tensor-like, e.g. list, tuple, Python number, tensor
          convert_external: if False and `x` is a Python number that is understood by this backend, this method returns the number as is. This can help prevent type clashes like int32 vs int64. (Default value = True)

        Returns:
          tensor representation of `x`

        """
        raise NotImplementedError()

    def is_available(self, tensor) -> bool:
        """
        Tests if the value of the tensor is known and can be read at this point.
        If true, `numpy(tensor)` must return a valid NumPy representation of the value.
        
        Tensors are typically available when the backend operates in eager mode.

        Args:
          tensor: backend-compatible tensor

        Returns:
          bool

        """
        raise NotImplementedError()

    def numpy(self, tensor) -> numpy.ndarray:
        """
        Returns a NumPy representation of the given tensor.
        If `tensor` is already a NumPy array, it is returned without modification.
        
        This method raises an error if the value of the tensor is not known at this point, e.g. because it represents a node in a graph.
        Use `is_available(tensor)` to check if the value can be represented as a NumPy array.

        Args:
          tensor: backend-compatible tensor

        Returns:
          NumPy representation of the values stored in the tensor

        """
        raise NotImplementedError()

    def copy(self, tensor, only_mutable=False):
        raise NotImplementedError()

    def jit_compile(self, f: Callable) -> Callable:
        return NotImplemented

    def call(self, f: Callable, *args, name=None):
        """
        Calls `f(*args)` and returns the result.
        This method may be used to register internal calls with the profiler.

        Usage:

            choose_backend(key).call(custom_function, *args)
        """
        return f(*args)

    def custom_gradient(self, f: Callable, gradient: Callable) -> Callable:
        """
        Creates a function based on `f` that uses a custom spatial_gradient for backprop.

        Args:
            f: Forward function. All arguments must be
            gradient: Function for backprop. Will be called as `spatial_gradient(*d_out)` to compute the spatial_gradient of `f`.

        Returns:
            Function with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
        """
        return NotImplemented

    def transpose(self, tensor, axes):
        raise NotImplementedError()

    def random_uniform(self, shape):
        """ Float tensor of selected precision containing random values in the range [0, 1) """
        raise NotImplementedError(self)

    def random_normal(self, shape):
        """ Float tensor of selected precision containing random values sampled from a normal distribution with mean 0 and std 1. """
        raise NotImplementedError(self)

    def stack(self, values, axis=0):
        raise NotImplementedError(self)

    def concat(self, values, axis):
        raise NotImplementedError(self)

    def pad(self, value, pad_width, mode: str = 'constant', constant_values=0):
        """
        Pad a tensor with values as specified by `mode` and `constant_values`.
        
        If the mode is not supported, returns NotImplemented.

        Args:
          value: tensor
          pad_width: 2D tensor specifying the number of values padded to the edges of each axis in the form [[axis 0 lower, axis 0 upper], ...] including batch and component axes.
          mode: constant', 'boundary', 'periodic', 'symmetric', 'reflect'
          constant_values: used for out-of-bounds points if mode='constant' (Default value = 0)
          mode: str:  (Default value = 'constant')

        Returns:
          padded tensor or NotImplemented

        """
        raise NotImplementedError(self)

    def reshape(self, value, shape):
        raise NotImplementedError(self)

    def flip(self, value, axes: tuple or list):
        slices = tuple(slice(None, None, -1 if i in axes else None) for i in range(self.ndims(value)))
        return value[slices]

    def sum(self, value, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def prod(self, value, axis=None):
        raise NotImplementedError(self)

    def divide_no_nan(self, x, y):
        """
        Computes x/y but returns 0 if y=0.

        Args:
          x: 
          y: 

        Returns:

        """
        raise NotImplementedError(self)

    def where(self, condition, x=None, y=None):
        raise NotImplementedError(self)

    def nonzero(self, values):
        """
        

        Args:
          values: 

        Returns:
          

        """
        raise NotImplementedError(self)

    def mean(self, value, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def range(self, start, limit=None, delta=1, dtype: DType = None):
        raise NotImplementedError(self)

    def zeros(self, shape, dtype: DType = None):
        raise NotImplementedError(self)

    def zeros_like(self, tensor):
        raise NotImplementedError(self)

    def ones(self, shape, dtype: DType = None):
        raise NotImplementedError(self)

    def ones_like(self, tensor):
        raise NotImplementedError(self)

    def meshgrid(self, *coordinates):
        raise NotImplementedError(self)

    def linspace(self, start, stop, number):
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

    def cast(self, x, dtype: DType):
        raise NotImplementedError(self)

    def to_float(self, x):
        """
        Converts a tensor to floating point values with precision equal to the currently set default precision.

        See Also:
            `Backend.precision()`.

        If `x` is mutable and of the correct floating type, returns a copy of `x`.

        To convert float tensors to the backend precision but leave non-float tensors untouched, use `Backend.as_tensor()`.

        Args:
            x: tensor of bool, int or float

        Returns:
            Values of `x` as float tensor
        """
        return self.cast(x, self.float_type)

    def to_int(self, x, int64=False):
        return self.cast(x, DType(int, 64 if int64 else 32))

    def to_complex(self, x):
        return self.cast(x, DType(complex, max(64, min(self.precision * 2, 128))))

    def gather(self, values, indices):
        raise NotImplementedError(self)

    def batched_gather_nd(self, values, indices):
        """
        Gathers values from the tensor `values` at locations `indices`.
        The first dimension of `values` and `indices` is the batch dimension which must be either equal for both or one for either.

        Args:
            values: tensor of shape (batch, spatial..., channel)
            indices: int tensor of shape (batch, any..., multi_index) where the size of multi_index is values.rank - 2.

        Returns:
            Gathered values as tensor of shape (batch, any..., channel)
        """
        raise NotImplementedError(self)

    def flatten(self, x):
        return self.reshape(x, (-1,))

    def std(self, x, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def boolean_mask(self, x, mask):
        raise NotImplementedError(self)

    def isfinite(self, x):
        raise NotImplementedError(self)

    def scatter(self, indices, values, shape, duplicates_handling='undefined', outside_handling='undefined'):
        """
        This method expects the first dimension of indices and values to be the batch dimension.
        The batch dimension need not be specified in the indices array.

        Args:
          indices: n-dimensional indices corresponding to values
          values: values to scatter at indices
          shape: spatial shape of the result tensor, 1D int array
          duplicates_handling: one of ('undefined', 'add', 'mean', 'any') (Default value = 'undefined')
          outside_handling: one of ('discard', 'clamp', 'undefined') (Default value = 'undefined')

        Returns:

        """
        raise NotImplementedError(self)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def fft(self, x):
        """
        Computes the n-dimensional FFT along all but the first and last dimensions.

        Args:
          x: tensor of dimension 3 or higher

        Returns:

        """
        raise NotImplementedError(self)

    def ifft(self, k):
        """
        Computes the n-dimensional inverse FFT along all but the first and last dimensions.

        Args:
          k: tensor of dimension 3 or higher

        Returns:

        """
        raise NotImplementedError(self)

    def imag(self, complex):
        raise NotImplementedError(self)

    def real(self, complex):
        raise NotImplementedError(self)

    def sin(self, x):
        raise NotImplementedError(self)

    def cos(self, x):
        raise NotImplementedError(self)

    def dtype(self, array) -> DType:
        raise NotImplementedError(self)

    def tile(self, value, multiples):
        """
        Repeats the tensor along each axis the number of times given by multiples.
        If `multiples` has more dimensions than `value`, these dimensions are added to `value` as outer dimensions.

        Args:
          value: tensor
          multiples: tuple or list of integers

        Returns:
          tile tensor

        """
        raise NotImplementedError(self)

    def sparse_tensor(self, indices, values, shape):
        """
        Optional features. If overridden,

        Args:
          indices: tuple/list matching the dimensions (pair for matrix)
          values: param shape:
          shape: 

        Returns:

        """
        raise NotImplementedError(self)

    def coordinates(self, tensor, unstack_coordinates=False):
        """
        Returns the coordinates and values of a tensor.
        
        The first returned value is a tensor holding the coordinate vectors in the last dimension if unstack_coordinates=False.
        In case unstack_coordinates=True, the coordiantes are returned as a tuple of tensors, i.e. (row, col) for matrices

        Args:
          tensor: dense or sparse tensor
          unstack_coordinates:  (Default value = False)

        Returns:
          indices (tensor or tuple), values

        """
        raise NotImplementedError(self)

    def minimize(self, function, x0, solve_params: Solve):
        raise NotImplementedError(self)

    def conjugate_gradient(self, A, y, x0, solve_params=LinearSolve(), callback=None):
        """
        Solve the system of linear equations
          A * x = y

        Args:
          A: batch of sparse / dense matrices or or linear function A(x). 3rd order tensor or list of matrices.
          y: target result of A * x. 2nd order tensor (batch, vector) or list of vectors.
          x0: initial guess for x. 2nd order tensor (batch, vector) or list of vectors.
          solve_params: Determines stop criteria. Stops when norm(residual) <= max(relative_tolerance * norm(y), absolute_tolerance) or maximum number of iterations reached.
          callback: Function to call after each iteration. It is called with the current solution as callback(x). (Default value = None)

        Returns:
            x: the solution as a tensor

        """
        raise NotImplementedError(self)

    def functional_gradient(self, f, wrt: tuple or list, get_output: bool):
        raise NotImplementedError(self)

    def gradients(self, y, xs: tuple or list, grad_y) -> tuple:
        raise NotImplementedError(self)

    def record_gradients(self, xs: tuple or list, persistent=False):
        raise NotImplementedError(self)

    def stop_gradient(self, value):
        raise NotImplementedError(self)

    def grid_sample(self, grid, spatial_dims: tuple, coordinates, extrapolation='constant'):
        """
        Interpolates a regular grid at the specified coordinates.

        Args:
          grid: Tensor
          spatial_dims: Dimension indices that correspond to coordinate vectors
          coordinates: Tensor of floating grid indices. The last dimension must match `spaital_dims`. The first grid point of dimension i lies at position 0, the last at values.shape[i]-1.
          extrapolation: Values to use for coordinates outside the grid. One of `('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect')`.

        Returns:
            sampled values with linear interpolation
        """
        return NotImplemented

    def variable(self, value):
        return NotImplemented

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

    def equal(self, x, y):
        """ Element-wise equality check """
        raise NotImplementedError(self)

    def not_equal(self, x, y):
        return ~self.equal(x, y)

    def greater_than(self, x, y):
        x, y = self.auto_cast(x, y)
        return x > y

    def greater_or_equal(self, x, y):
        x, y = self.auto_cast(x, y)
        return x >= y

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


BACKENDS = []  # documented in __init__.py
_DEFAULT = []  # [0] = global default, [1:] from 'with' blocks
_PRECISION = [32]  # [0] = global precision in bits, [1:] from 'with' blocks


def choose_backend(*values, prefer_default=False, raise_error=True) -> Backend:
    """
    Selects a suitable backend to handle the given values.

    This function is used by most math functions operating on `Tensor` objects to delegate the actual computations.

    Args:
        *values:
        prefer_default: if True, selects the default backend assuming it can handle handle the values, see `default_backend()`.
        raise_error: Determines the behavior of this function if no backend can handle the given values.
            If True, raises a `NoBackendFound` error, else returns `None`.

    Returns:
        the selected `Backend`
    """
    # --- Default Backend has priority ---
    if _is_applicable(_DEFAULT[-1], values) and (prefer_default or _is_specific(_DEFAULT[-1], values)):
        return _DEFAULT[-1]
    # --- Filter out non-applicable ---
    backends = [backend for backend in BACKENDS if _is_applicable(backend, values)]
    if len(backends) == 0:
        if raise_error:
            raise NoBackendFound(f"No backend found for types {[type(v).__name__ for v in values]}; registered backends are {BACKENDS}")
        else:
            return None
    # --- Native tensors? ---
    for backend in backends:
        if _is_specific(backend, values):
            return backend
    else:
        return backends[0]


class NoBackendFound(Exception):
    """
    Thrown by `choose_backend` if no backend can handle the given values.
    """

    def __init__(self, msg):
        Exception.__init__(self, msg)


def default_backend() -> Backend:
    """
    The default backend is preferred by `choose_backend()`.

    The default backend can be set globally using `set_global_default_backend()` and locally using `with backend:`.

    Returns:
        current default `Backend`
    """
    return _DEFAULT[-1]


def context_backend() -> Backend or None:
    """
    Returns the backend set by the inner-most surrounding `with backend:` block.
    If called outside a backend context, returns `None`.

    Returns:
        `Backend` or `None`
    """
    return _DEFAULT[-1] if len(_DEFAULT) > 1 else None


def set_global_default_backend(backend: Backend):
    """
    Sets the given backend as default.
    This setting can be overridden using `with backend:`.

    See `default_backend()`, `choose_backend()`.

    Args:
        backend: `Backend` to set as default
    """
    assert isinstance(backend, Backend)
    _DEFAULT[0] = backend


def set_global_precision(floating_point_bits: int):
    """
    Sets the floating point precision of DYNAMIC_BACKEND which affects all registered backends.

    If `floating_point_bits` is an integer, all floating point tensors created henceforth will be of the corresponding data type, float16, float32 or float64.
    Operations may also convert floating point values to this precision, even if the input had a different precision.

    If `floating_point_bits` is None, new tensors will default to float32 unless specified otherwise.
    The output of math operations has the same precision as its inputs.

    Args:
      floating_point_bits: one of (16, 32, 64, None)
    """
    _PRECISION[0] = floating_point_bits


def get_precision() -> int:
    """
    Gets the current target floating point precision in bits.
    The precision can be set globally using `set_global_precision()` or locally using `with precision(p):`.

    Any Backend method may convert floating point values to this precision, even if the input had a different precision.

    Returns:
        16 for half, 32 for single, 64 for double
    """
    return _PRECISION[-1]


@contextmanager
def precision(floating_point_bits: int):
    """
    Sets the floating point precision for the local context.

    Usage: `with precision(p):`

    This overrides the global setting, see `set_global_precision()`.

    Args:
        floating_point_bits: 16 for half, 32 for single, 64 for double
    """
    _PRECISION.append(floating_point_bits)
    try:
        yield None
    finally:
        _PRECISION.pop(-1)


# Backend choice utility functions

def _is_applicable(backend, values):
    for value in values:
        if not backend.is_tensor(value, only_native=False):
            return False
    return True


def _is_specific(backend, values):
    for value in values:
        if backend.is_tensor(value, only_native=True):
            return True
    return False

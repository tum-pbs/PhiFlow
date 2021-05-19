from collections import namedtuple
from contextlib import contextmanager
from threading import Barrier
from typing import List, Callable, Tuple, Any

import numpy

from ._dtype import DType, combine_types


BasicSolveResult = namedtuple('BasicSolveResult', [
    'x', 'converged', 'diverged',
])
FullSolveResult = namedtuple('FullSolveResult', [
    'x', 'residual', 'iterations', 'function_evaluations', 'converged', 'diverged', 'message',
])


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

    def prefers_channels_last(self) -> bool:
        raise NotImplementedError()

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

    def seed(self, seed: int):
        raise NotImplementedError()

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
          convert_external: if False and `x` is a Python number that is understood by this backend, this method returns the number as-is. This can help prevent type clashes like int32 vs int64. (Default value = True)

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

    def to_dlpack(self, tensor):
        raise NotImplementedError()

    def from_dlpack(self, capsule):
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
        Creates a function based on `f` that uses a custom gradient for backprop.

        Args:
            f: Forward function.
            gradient: Function for backprop. Will be called as `gradient(*d_out)` to compute the gradient of `f`.

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

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
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

    def tensordot(self, a, a_axes: tuple or list, b, b_axes: tuple or list):
        """ Multiply-sum-reduce a_axes of a with b_axes of b. """
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

    def conv(self, value, kernel, zero_padding=True):
        """
        Convolve value with kernel.
        Depending on the tensor rank, the convolution is either 1D (rank=3), 2D (rank=4) or 3D (rank=5).
        Higher dimensions may not be supported.

        Args:
            value: tensor of shape (batch_size, in_channel, spatial...)
            kernel: tensor of shape (batch_size or 1, out_channel, in_channel, spatial...)
            zero_padding: If True, pads the edges of `value` with zeros so that the result has the same shape as `value`.

        Returns:
            Convolution result as tensor of shape (batch_size, out_channel, spatial...)
        """
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

    def to_int32(self, x):
        return self.cast(x, DType(int, 32))

    def to_int64(self, x):
        return self.cast(x, DType(int, 64))

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

    def boolean_mask(self, x, mask, axis=0):
        """

        Args:
            x:
            mask:
            axis: Axis index >= 0

        Returns:

        """
        raise NotImplementedError(self)

    def isfinite(self, x):
        raise NotImplementedError(self)

    def scatter(self, base_grid, indices, values, mode: str):
        """
        Depending on `mode`, performs scatter_update or scatter_add.

        Args:
            base_grid: Tensor into which scatter values are inserted at indices. Tensor of shape (batch_size, spatial..., channels)
            indices: Tensor of shape (batch_size or 1, update_count, index_vector)
            values: Values to scatter at indices. Tensor of shape (batch_size or 1, update_count or 1, channels or 1)
            mode: One of ('update', 'add')

        Returns:
            Copy of base_grid with values at `indices` updated by `values`.
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

    def imag(self, x):
        raise NotImplementedError(self)

    def real(self, x):
        raise NotImplementedError(self)

    def sin(self, x):
        raise NotImplementedError(self)

    def cos(self, x):
        raise NotImplementedError(self)

    def tan(self, x):
        raise NotImplementedError(self)

    def log(self, x):
        """ Natural logarithm """
        raise NotImplementedError(self)

    def log2(self, x):
        raise NotImplementedError(self)

    def log10(self, x):
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

    def minimize(self, method: str, f, x0, atol, max_iter, ret: str):
        from scipy.optimize import OptimizeResult, minimize
        assert self.supports(Backend.functional_gradient)
        x0 = solve.x0
        assert len(self.staticshape(x0)) == 2  # (batch, parameters)
        fg = self.functional_gradient(f, [0], get_output=True)
        batch_size = self.staticshape(x0)[0]
        iterations = [0] * batch_size
        function_evaluations = [0] * batch_size
        losses = []
        last_result = [None]
        if trajectory is not None:
            function_evaluations += 1
            trajectory.append(SolveResult(solve, x0, ))

        call_function_barrier = Barrier(batch_size)

        def min_target(x: numpy.ndarray):
            call_function_barrier.wait()
            # TODO
            function_evaluations[b] += 1
            x = self.expand_dims(self.as_tensor(x, convert_external=True), 0)
            output, (grad,) = fg(x)
            loss = output[0] if isinstance(output, (tuple, list)) else output
            grad = self.numpy(grad).astype(numpy.float64)
            losses.append(loss)
            return self.numpy(loss).astype(numpy.float64), grad

        def callback(x, *args):  # L-BFGS-B only passes x but the documentation says (x, state)
            iterations[b] += 1
            loss = min(losses)
            losses.clear()
            last_result[0] = SolveResult(solve, x, loss, iterations[b], function_evaluations[b], converged=False, diverged=False)
            if trajectory is not None:
                trajectory.append(last_result[0])

        results = []
        for b in range(batch_size):
            from threading import Thread
            Thread(target=lambda: minimize()).start()
            res = minimize(fun=min_target, x0=x0[b], jac=True, method=solve.method, tol=solve.absolute_tolerance,
                           options={'maxiter': solve.max_iterations}, callback=callback)
            batch_result = SolveResult(solve, res.x, batch_result.final_state.residual,
                                       iteration=res.nit,
                                       function_evaluations=res.nfev,
                                       converged=res.success,
                                       diverged=res.status in (0, 1),
                                       termination_message=res.message)
            results.append(batch_result)
        return batch_combine_solve_results(results, self)

    def linear_solve(self, method: str, lin, y, x0, rtol, atol, max_iter, ret: type) -> Any:
        """
        Solve the system of linear equations A · x = y.
        This method need not provide a gradient for the operation.

        Args:
            method: Which algorithm to use. One of `('auto', 'CG', 'CG-adaptive')`.
            lin: Linear operation. One of
                * sparse/dense matrix valid for all instances
                * tuple/list of sparse/dense matrices for varying matrices along batch, must have the same nonzero locations.
                * linear function A(x), must be called on all instances in parallel
            y: target result of A * x. 2nd order tensor (batch, vector) or list of vectors.
            x0: Initial guess of size (batch, parameters)
            rtol: Relative tolerance of size (batch,)
            atol: Absolute tolerance of size (batch,)
            max_iter: Maximum number of iterations of size (batch,)
            ret: Return type, one of ('BasicSolveResult', 'FullSolveResult', 'list')

        Returns:
            result: Type depends on `ret`.
        """
        if method == 'auto':
            return self.conjugate_gradient_adaptive(lin, y, x0, rtol, atol, max_iter, ret)
        elif method == 'CG':
            return self.conjugate_gradient(lin, y, x0, rtol, atol, max_iter, ret)
        elif method == 'CG-adaptive':
            return self.conjugate_gradient_adaptive(lin, y, x0, rtol, atol, max_iter, ret)
        else:
            raise NotImplementedError(f"Method '{method}' not supported for linear solve.")

    def conjugate_gradient(self, lin, y, x0, rtol, atol, max_iter, ret: type) -> Any:
        """ Standard conjugate gradient algorithm. Signature matches to `Backend.linear_solve()`. """
        # Based on "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" by Jonathan Richard Shewchuk
        # symbols: dx=d, dy=q, step_size=alpha, residual_squared=delta, residual=r, y=b
        f = as_linear_function(lin, self)
        y = self.to_float(y)
        x0 = self.copy(self.to_float(x0), only_mutable=True)
        batch_size = combined_dim(self.shape(y)[0], self.shape(x0)[0])
        tolerance_sq = self.maximum(rtol ** 2 * self.sum(y ** 2, -1), atol ** 2)
        x = x0
        dx = residual = y - f(x)
        it_counter = 0
        iterations = self.zeros([batch_size], DType(int, 32))
        function_evaluations = self.ones([batch_size], DType(int, 32))
        residual_squared = self.sum(residual ** 2, -1, keepdims=True)
        trajectory = [] if ret == 'trajectory' else None
        while True:
            diverged = ~self.all(self.isfinite(x), axis=(1,))
            converged = self.all(residual_squared <= tolerance_sq, axis=(1,))
            if trajectory is not None:
                trajectory.append(FullSolveResult(x, residual, iterations, function_evaluations, converged, diverged, ""))
            finished = converged | diverged | (iterations >= max_iter); not_finished_1 = self.to_int32(~finished)  # ; active = self.to_float(self.expand_dims(not_finished_1, -1))
            if self.all(finished):
                return BasicSolveResult(x, converged, diverged) if ret == BasicSolveResult else (FullSolveResult(x, residual, iterations, function_evaluations, converged, diverged, "") if ret == FullSolveResult else trajectory)
            it_counter += 1; iterations += not_finished_1
            dy = f(dx); function_evaluations += not_finished_1
            dx_dy = self.sum(dx * dy, axis=-1, keepdims=True)
            step_size = self.divide_no_nan(residual_squared, dx_dy)
            x += step_size * dx
            if it_counter % 50 == 0:
                residual = y - f(x); function_evaluations += 1
            else:
                residual -= step_size * dy
            residual_squared_old = residual_squared
            residual_squared = self.sum(residual ** 2, -1, keepdims=True)
            dx = residual + self.divide_no_nan(residual_squared, residual_squared_old) * dx

    def conjugate_gradient_adaptive(self, lin, y, x0, rtol, atol, max_iter, ret: type) -> Any:
        """ Conjugate gradient algorithm with adaptive step size. Signature matches to `Backend.linear_solve()`. """
        # Based on the variant described in "Methods of Conjugate Gradients for Solving Linear Systems" by Magnus R. Hestenes and Eduard Stiefel
        # https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf
        A = as_linear_function(lin, self)
        y = self.to_float(y)
        x0 = self.copy(self.to_float(x0), only_mutable=True)
        batch_size = combined_dim(self.shape(y)[0], self.shape(x0)[0])
        tolerance_sq = self.maximum(rtol ** 2 * self.sum(y ** 2, -1), atol ** 2)
        x = x0
        dx = residual = y - A(x)
        dy = A(dx)
        it_counter = 0
        iterations = self.zeros([batch_size], DType(int, 32))
        function_evaluations = self.ones([batch_size], DType(int, 32))
        trajectory = [] if ret == 'trajectory' else None
        while True:
            residual_squared = self.sum(residual ** 2, -1, keepdims=True)
            diverged = ~self.all(self.isfinite(x), axis=(1,))
            converged = self.all(residual_squared <= tolerance_sq, axis=(1,))
            if trajectory is not None:
                trajectory.append(FullSolveResult(x, residual, iterations, function_evaluations, converged, diverged, ""))
            finished = converged | diverged | (iterations >= max_iter); not_finished_1 = self.to_int32(~finished)  # ; active = self.to_float(self.expand_dims(not_finished_1, -1))
            if self.all(finished):
                return BasicSolveResult(x, converged, diverged) if ret == BasicSolveResult else (FullSolveResult(x, residual, iterations, function_evaluations, converged, diverged,"") if ret == FullSolveResult else trajectory)
            it_counter += 1; iterations += not_finished_1
            dx_dy = self.sum(dx * dy, axis=-1, keepdims=True)
            step_size = self.divide_no_nan(self.sum(dx * residual, axis=-1, keepdims=True), dx_dy)
            x += step_size * dx
            residual -= step_size * dy
            dx = residual - self.divide_no_nan(self.sum(residual * dy, axis=-1, keepdims=True) * dx, dx_dy)
            dy = A(dx); function_evaluations += not_finished_1

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
          coordinates: Tensor of floating grid indices. The last dimension must match `spatial_dims`. The first grid point of dimension i lies at position 0, the last at values.shape[i]-1.
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

    def unstack(self, tensor, axis=0, keepdims=False) -> tuple:
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

    def and_(self, a, b):
        a, b = self.auto_cast(a, b)
        return a & b

    def or_(self, a, b):
        a, b = self.auto_cast(a, b)
        return a | b

    def xor(self, a, b):
        a, b = self.auto_cast(a, b)
        return a ^ b

    def floordiv(self, a, b):
        a, b = self.auto_cast(a, b)
        return a // b



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


def convert(tensor, backend: Backend = None, use_dlpack=True):
    """
    Convert a Tensor to the native format of `backend`.
    If the target backend can operate natively on `tensor`, returns `tensor`.

    If both backends support *DLPack* and `use_dlpack=True`, uses zero-copy conversion using the DLPack library.
    Else, intermediately converts `tensor` to a NumPy array.

    *Warning*: This operation breaks the automatic differentiation chain.

    Args:
        tensor: Native tensor belonging to any registered backend.
        backend: Target backend. If `None`, uses the current default backend, see `default_backend()`.

    Returns:
        Tensor belonging to `backend`.
    """
    backend = backend or default_backend()
    current_backend = choose_backend(tensor, prefer_default=False)
    if backend.is_tensor(tensor, True) or backend is current_backend:
        return tensor
    if use_dlpack and current_backend.supports(Backend.to_dlpack) and backend.supports(Backend.from_dlpack):
        capsule = current_backend.to_dlpack(tensor)
        return backend.from_dlpack(capsule)
    else:
        nparray = current_backend.numpy(tensor)
        return backend.as_tensor(nparray)


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


# Other low-level helper functions

def combined_dim(dim1, dim2, type_str: str = 'batch'):
    if dim1 is None and dim2 is None:
        return None
    if dim1 is None or dim1 == 1:
        return dim2
    if dim2 is None or dim2 == 1:
        return dim1
    assert dim1 == dim2, f"Incompatible {type_str} dimensions: x0 {dim1}, y {dim2}"
    return dim1


def batch_combine_solve_results(results: List[BasicSolveResult], backend: Backend):
    def batch_combine_solver_states(states: List[SolverState] or Tuple[SolverState]):
        x = backend.stack([s.x for s in states])
        residual = None
        if states[0].residual is not None:
            residual = backend.stack([s.residual for s in states])
        return SolverState(x, residual, max([s.iteration for s in states]),
                           max([s.function_evaluations for s in states]),
                           all(s.converged for s in states),
                           any(s.diverged for s in states))

    result = SolveResult(results[0].parameters)
    result.final_state = batch_combine_solver_states([r.final_state for r in results])
    if results[0].trajectory:
        tlen = max([len(r.trajectory) for r in results])
        trajectories = [r.trajectory + [r.final_state] * (tlen - len(r.trajectory)) for r in results]
        result.trajectory = [batch_combine_solver_states(states) for states in zip(*trajectories)]
    for r in results:
        if r._termination_message:
            result._termination_message = r._termination_message
            break
    return result


def as_linear_function(lin, backend):
    if callable(lin):
        f = lin
    elif isinstance(lin, (tuple, list)):
        for lin_i in lin:
            lin_shape = backend.staticshape(lin_i)
            assert len(lin_shape) == 2

        def f(vector):
            results = [backend.matmul(m, v) for m, v in zip(lin, backend.unstack(vector))]
            return backend.stack(results)
    else:
        lin_shape = backend.staticshape(lin)
        assert len(lin_shape) == 2, f"A must be a matrix but got shape {lin_shape}"

        def f(vector):
            return backend.matmul(lin, vector)
    return f

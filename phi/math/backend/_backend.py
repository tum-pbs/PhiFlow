import sys
import warnings
from collections import namedtuple
from contextlib import contextmanager
from typing import List, Callable, TypeVar, Tuple, Any, Union

import logging
import numpy

from ._dtype import DType, combine_types, to_numpy_dtype


SolveResult = namedtuple('SolveResult', [
    'method', 'x', 'residual', 'iterations', 'function_evaluations', 'converged', 'diverged', 'message',
])

TensorType = TypeVar('TensorType')


class ComputeDevice:
    """
    A physical device that can be selected to perform backend computations.
    """

    def __init__(self, backend: 'Backend', name: str, device_type: str, memory: int, processor_count: int, description: str, ref):
        assert device_type in ('CPU', 'GPU', 'TPU')
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
        """ Reference to the internal device representation. Two devices are equal if their refs are equal. """
        self.backend: 'Backend' = backend
        """ Backend that this device belongs to. Different backends represent the same device with different objects. """

    def __repr__(self):
        mem = f"{(self.memory / 1024 ** 2):.0f} MB" if self.memory > 0 else "memory: n/a"
        pro = f"{self.processor_count} processors" if self.processor_count > 0 else "processors: n/a"
        ref = f" '{self.ref}'" if isinstance(self.ref, str) else ""
        descr = self.description.replace('\n', '  ')
        if len(descr) > 30:
            descr = descr[:28] + "..."
        return f"{self.backend} device '{self.name}' ({self.device_type}{ref}) | {mem} | {pro} | {descr}"

    def __eq__(self, other):
        return isinstance(other, ComputeDevice) and other.ref == self.ref

    def __hash__(self):
        return hash(self.ref)


class Backend:

    def __init__(self, name: str, devices: List[ComputeDevice], default_device: ComputeDevice):
        """
        Backends delegate low-level operations to a compute library or emulate them.

        The methods of `Backend` form a comprehensive list of available operations.

        To support a compute library, subclass `Backend` and register it by adding it to `BACKENDS`.

        Args:
            name: Human-readable string
            default_device: `ComputeDevice` being used by default
        """
        self._name = name
        self._devices = tuple(devices)
        self._default_device = default_device

    def __enter__(self):
        _DEFAULT.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _DEFAULT.pop(-1)

    @property
    def name(self) -> str:
        return self._name

    def supports(self, feature: Union[str, Callable]) -> bool:
        """
        Tests if this backend supports the given feature.
        Features correspond to a method of this backend that must be implemented if the feature is supported.

        Possible features:

        * `sparse_coo_tensor`
        * `gradients

        Args:
            feature: `str` or unbound Backend method, e.g. `Backend.sparse_coo_tensor`

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

    def requires_fixed_shapes_when_tracing(self) -> bool:
        return False

    @property
    def precision(self) -> int:
        """ Short for math.backend.get_precision() """
        return get_precision()

    @property
    def float_type(self) -> DType:
        return DType(float, self.precision)

    @property
    def as_registered(self) -> 'Backend':
        from phi.math.backend import BACKENDS
        for backend in BACKENDS:
            if self.name in backend.name:
                return backend
        raise RuntimeError(f"Backend '{self}' is not visible.")

    @property
    def complex_type(self) -> DType:
        return DType(complex, max(64, self.precision))

    def combine_types(self, *dtypes: DType) -> DType:
        return combine_types(*dtypes, fp_precision=self.precision)

    def auto_cast(self, *tensors, bool_to_int=False, int_to_float=False) -> list:
        """
        Determins the appropriate values type resulting from operations involving the tensors as input.
        
        This method is called by the default implementations of basic operators.
        Backends can override this method to prevent unnecessary casting.

        Args:
            *tensors: tensors to cast and to consider when determining the common data type
            bool_to_int: Whether to convert boolean values to integers if all values are boolean.

        Returns:
            tensors cast to a common data type
        """
        dtypes = [self.dtype(t) for t in tensors]
        result_type = self.combine_types(*dtypes)
        if result_type.kind == bool and bool_to_int:
            result_type = DType(int, 32)
        if result_type.kind == int and int_to_float:
            result_type = DType(float, self.precision)
        if result_type.kind in (int, float, complex, bool):  # do not cast everything to string!
            tensors = [self.cast(t, result_type) for t in tensors]
        return tensors

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def list_devices(self, device_type: Union[str, None] = None) -> List[ComputeDevice]:
        """
        Fetches information about all available compute devices this backend can use.

        Implementations:

        * NumPy: [`os.cpu_count`](https://docs.python.org/3/library/os.html#os.cpu_count)
        * PyTorch: [`torch.cuda.get_device_properties`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.get_device_properties)
        * TensorFlow: `tensorflow.python.client.device_lib.list_local_devices`
        * Jax: [`jax.devices`](https://jax.readthedocs.io/en/latest/jax.html#jax.devices)

        See Also:
            `Backend.set_default_device()`.

        Args:
            device_type: (optional) Return only devices of this type, e.g. `'GPU'` or `'CPU'`. See `ComputeDevice.device_type`.

        Returns:
            `list` of all currently available devices.
        """
        if device_type is None:
            return list(self._devices)
        else:
            assert device_type in ('CPU', 'GPU', 'TPU'), "Device"
            return [d for d in self._devices if d.device_type == device_type]

    def get_default_device(self) -> ComputeDevice:
        return self._default_device

    def set_default_device(self, device: Union[ComputeDevice, str]) -> bool:
        """
        Sets the device new tensors will be allocated on.
        This function will do nothing if the target device type is not available.

        See Also:
            `Backend.list_devices()`, `Backend.get_default_device()`.

        Args:
            device: `ComputeDevice` or device type as `str`, such as `'CPU'` or `'GPU'`.

        Returns:
            `bool` whether the device was successfully set.
        """
        if isinstance(device, str):
            devices = self.list_devices(device)
            if not devices:
                warnings.warn(f"{self.name}: Cannot select '{device}' because no device of this type is available.", RuntimeWarning)
                return False
            device = devices[0]
        assert device.backend is self, f"Cannot set default device to {device.name} for backend {self.name} because the devices belongs to backend {device.backend.name}"
        self._default_device = device
        return True

    def get_device(self, tensor: TensorType) -> ComputeDevice:
        """ Returns the device `tensor` is located on. """
        raise NotImplementedError()

    def get_device_by_ref(self, ref):
        for device in self._devices:
            if device.ref == ref:
                return device
        raise KeyError(f"{self.name} has no device with ref '{ref}'. Available: {[d.ref for d in self._devices]}")

    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        """
        Moves `tensor` to `device`. May copy the tensor if it is already on the device.

        Args:
            tensor: Existing tensor native to this backend.
            device: Target device, associated with this backend.
        """
        raise NotImplementedError()

    def seed(self, seed: int):
        raise NotImplementedError()

    def is_module(self, obj) -> bool:
        """
        Tests if `obj` is of a type that is specific to this backend, e.g. a neural network.
        If `True`, this backend will be chosen for operations involving `obj`.

        See Also:
            `Backend.is_tensor()`.

        Args:
            obj: Object to test.
        """
        raise NotImplementedError()

    def is_tensor(self, x, only_native=False):
        """
        An object is considered a native tensor by a backend if no internal conversion is required by backend methods.
        An object is considered a tensor (nativer or otherwise) by a backend if it is not a struct (e.g. tuple, list) and all methods of the backend accept it as a tensor argument.

        If `True`, this backend will be chosen for operations involving `x`.

        See Also:
            `Backend.is_module()`.

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

    def copy_leaves(self, tree, only_mutable=False):
        if isinstance(tree, tuple):
            return tuple([self.copy_leaves(e, only_mutable) for e in tree])
        elif isinstance(tree, list):
            return [self.copy_leaves(e, only_mutable) for e in tree]
        elif isinstance(tree, dict):
            return {k: self.copy_leaves(e, only_mutable) for k, e in tree.items()}
        else:
            return self.copy(tree, only_mutable=only_mutable)

    def call(self, f: Callable, *args, name=None):
        """
        Calls `f(*args)` and returns the result.
        This method may be used to register internal calls with the profiler.

        Usage:

            choose_backend(key).call(custom_function, *args)
        """
        return f(*args)

    def block_until_ready(self, values):
        pass

    def vectorized_call(self, f, *args):
        batch_dim = self.determine_size(args, 0)
        result = []
        for b in range(batch_dim):
            result.append(f(*[t[min(b, self.staticshape(t)[0] - 1)] for t in args]))
        return self.stack(result)

    def determine_size(self, tensors, axis):
        sizes = [self.staticshape(t)[axis] for t in tensors]
        non_singleton_sizes = [b for b in sizes if b != 1]
        size = non_singleton_sizes[0] if non_singleton_sizes else 1
        assert all([b in (1, size) for b in sizes])
        return size

    def tile_to(self, x, axis, size):
        current_size = self.staticshape(x)[axis]
        if current_size == size:
            return x
        assert size > current_size
        assert size % current_size == 0
        multiples = [size // current_size if i == axis else 1 for i in range(self.ndims(x))]
        return self.tile(x, multiples)

    def jit_compile(self, f: Callable) -> Callable:
        return NotImplemented

    def jacobian(self, f: Callable, wrt: Union[tuple, list], get_output: bool, is_f_scalar: bool):
        """
        Args:
            f: Function to differentiate. Returns a tuple containing `(reduced_loss, output)`
            wrt: Argument indices for which to compute the gradient.
            get_output: Whether the derivative function should return the output of `f` in addition to the gradient.
            is_f_scalar: Whether `f` is guaranteed to return a scalar output.

        Returns:
            A function `g` with the same arguments as `f`.
            If `get_output=True`, `g` returns a `tuple`containing the outputs of `f` followed by the gradients.
            The gradients retain the dimensions of `reduced_loss` in order as outer (first) dimensions.
        """
        raise NotImplementedError(self)

    def hessian(self, f: Callable, wrt: Union[tuple, list], get_output: bool, get_gradient: bool) -> tuple:
        """
        First dimension of all inputs/outputs of `f` is assumed to be a batch dimension.
        Element-wise Hessians will be computed along the batch dimension.
        All other dimensions are parameter dimensions and will appear twice in the Hessian matrices.

        Args:
            f: Function whose first output is a scalar float or complex value.
            wrt:
            get_output:
            get_gradient:

        Returns:
            Function returning `(f(x), g(x), H(x))` or less depending on `get_output` and `get_gradient`.
            The result is always a `tuple` holding at most these three items.
        """
        raise NotImplementedError(self)

    def custom_gradient(self, f: Callable, gradient: Callable, get_external_cache: Callable = None, on_call_skipped: Callable = None) -> Callable:
        """
        Creates a function based on `f` that uses a custom gradient for backprop.

        Args:
            f: Forward function.
            gradient: Function for backprop. Will be called as `gradient(*d_out)` to compute the gradient of `f`.

        Returns:
            Function with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
        """
        return NotImplemented

    def jit_compile_grad(self, f: Callable, wrt: Union[tuple, list], get_output: bool, is_f_scalar: bool):
        raise NotImplementedError()

    def jit_compile_hessian(self, f: Callable, wrt: Union[tuple, list], get_output: bool, get_gradient: bool):
        raise NotImplementedError()

    def transpose(self, tensor, axes):
        """ Transposes the dimensions of `tensor` given the new axes order. The tensor will be cast to the default precision in the process. """
        raise NotImplementedError()

    def random_uniform(self, shape, low, high, dtype: Union[DType, None]):
        """ Float tensor of selected precision containing random values in the range [0, 1) """
        raise NotImplementedError(self)

    def random_normal(self, shape, dtype: DType):
        """ Float tensor of selected precision containing random values sampled from a normal distribution with mean 0 and std 1. """
        raise NotImplementedError(self)

    def stack(self, values, axis=0):
        raise NotImplementedError(self)

    def stack_leaves(self, trees: Union[tuple, list], axis=0):
        tree0 = trees[0]
        if isinstance(tree0, tuple):
            return tuple([self.stack_leaves([tree[i] for tree in trees], axis=axis) for i in range(len(tree0))])
        elif isinstance(tree0, list):
            return [self.stack_leaves([tree[i] for tree in trees], axis=axis) for i in range(len(tree0))]
        elif isinstance(tree0, dict):
            return {k: self.stack_leaves([tree[k] for tree in trees], axis=axis) for k in tree0}
        else:
            return self.stack(trees, axis=axis)

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

    def flip(self, value, axes: Union[tuple, list]):
        slices = tuple(slice(None, None, -1 if i in axes else None) for i in range(self.ndims(value)))
        return value[slices]

    def sum(self, value, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def prod(self, value, axis=None):
        raise NotImplementedError(self)

    def divide_no_nan(self, x, y):
        """ Computes x/y but returns 0 if y=0. """
        raise NotImplementedError(self)

    def where(self, condition, x=None, y=None):
        raise NotImplementedError(self)

    def nonzero(self, values):
        """
        Args:
            values: Tensor with only spatial dimensions

        Returns:
            non-zero multi-indices as tensor of shape (nnz, vector)
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

    def tensordot(self, a, a_axes: Union[tuple, list], b, b_axes: Union[tuple, list]):
        """ Multiply-sum-reduce a_axes of a with b_axes of b. """
        raise NotImplementedError(self)

    def mul_matrix_batched_vector(self, A, b):
        raise NotImplementedError(self)

    def einsum(self, equation, *tensors):
        raise NotImplementedError(self)

    def cumsum(self, x, axis: int):
        raise NotImplementedError(self)

    def while_loop(self, loop: Callable, values: tuple, max_iter: Union[int, Tuple[int, ...], List[int]]):
        """
        If `max_iter is None`, runs

        ```python
        while any(values[0]):
            values = loop(*values)
        return values
        ```

        This operation does not support backpropagation.

        Args:
            loop: Loop function, must return a `tuple` with entries equal to `values` in shape and data type.
            values: Initial values of loop variables.
            max_iter: Maximum number of iterations to run, single `int` or sequence of integers.
        Returns:
            Loop variables upon loop completion if `max_iter` is a single integer.
            If `max_iter` is a sequence, stacks the variables after each entry in `max_iter`, adding an outer dimension of size `<= len(max_iter)`.
            If the condition is fulfilled before the maximum max_iter is reached, the loop may be broken or not, depending on the implementation.
            If the loop is broken, the values returned by the last loop are expected to be constant and filled.
        """
        values = self.stop_gradient_tree(values)
        if isinstance(max_iter, (tuple, list)):
            trj = [self.copy_leaves(values, only_mutable=True)] if 0 in max_iter else []
            for i in range(1, max(max_iter) + 1):
                values = loop(*values)
                if i in max_iter:
                    trj.append(self.copy_leaves(values, only_mutable=True))
                if not self.any(values[0]):
                    break
            trj.extend([trj[-1]] * (len(max_iter) - len(trj)))  # fill trj with final values
            return self.stop_gradient_tree(self.stack_leaves(trj))
        else:
            for i in range(1, max_iter + 1):
                if not self.any(values[0]):
                    break
                values = loop(*values)
            return self.stop_gradient_tree(values)

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

    def softplus(self, x):
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
        """
        Returns the shape of a tensor.
        The shape is iterable and implements `len()`.
        For non-eager tensors, undefined dimensions should return a placeholder value representing the size.

        See Also:
            `Backend.staticshape()`.

        Args:
            tensor: Native tensor compatible with this backend.

        Returns:
            Shape of `tensor`
        """
        raise NotImplementedError(self)

    def staticshape(self, tensor) -> tuple:
        """
        Evaluates the static shape of a native tensor.
        If the tensor is eager, the shape is a `tuple[int]`.
        For placeholder tensors, unknown dimensions are represented as `None`.

        See Also:
            `Backend.shape()`.

        Args:
            tensor: Native tensor compatible with this backend.

        Returns:
            `tuple` of sizes. Each size is an `int` if the size is defined, else `None`.
        """
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
        return self.cast(x, DType(complex, max(64, self.precision * 2)))

    def unravel_index(self, flat_index, shape):
        strides = [1]
        for size in reversed(shape[1:]):
            strides.append(strides[-1] * size)
        strides = strides[::-1]
        result = []
        for i in range(len(shape)):
            result.append(flat_index // strides[i] % shape[i])
        return self.stack(result, -1)

    def ravel_multi_index(self, multi_index, shape, mode: Union[str, int] = 'undefined'):
        """
        Args:
            multi_index: (batch..., index_dim)
            shape: 1D tensor or tuple/list
            mode: `'undefined'`, `'periodic'`, `'clamp'` or an `int` to use for all invalid indices.

        Returns:
            Integer tensor of shape (batch...)
        """
        strides = [self.ones((), DType(int, 32))]
        for size in reversed(shape[1:]):
            strides.append(strides[-1] * size)
        strides = self.stack(strides[::-1])
        if mode == 'periodic':
            multi_index %= self.as_tensor(shape)
        elif mode == 'clamp':
            multi_index = self.clip(multi_index, 0, self.as_tensor(shape) - 1)
        result = self.sum(multi_index * strides, -1)
        if isinstance(mode, int):
            inside = self.all((0 <= multi_index) & (multi_index < self.as_tensor(shape)), -1)
            result = self.where(inside, result, mode)
        return result

    def gather(self, values, indices, axis: int):
        """
        Gathers values from the tensor `values` at locations `indices`.

        Args:
            values: tensor
            indices: 1D tensor
            axis: Axis along which to gather slices

        Returns:
            tensor, with size along `axis` being the length of `indices`
        """
        raise NotImplementedError(self)

    def gather_by_component_indices(self, values, *component_indices):
        return values[component_indices]

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

    def batched_gather_1d(self, values, indices):
        values = self.expand_dims(values, -1)
        indices = self.expand_dims(indices, -1)
        return self.batched_gather_nd(values, indices)[..., 0]

    def gather_1d(self, values, indices):
        return self.gather(values, indices, 0)

    def flatten(self, x):
        return self.reshape(x, (-1,))

    def std(self, x, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def boolean_mask(self, x, mask, axis=0, new_length=None, fill_value=0):
        """
        Args:
            x: tensor with any number of dimensions
            mask: 1D mask tensor
            axis: Axis index >= 0
            new_length: Maximum size of the output along `axis`. This must be set when jit-compiling with Jax.
            fill_value: If `new_length` is larger than the filtered result, the remaining values will be set to `fill_value`.
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

    def histogram1d(self, values, weights, bin_edges):
        """
        Args:
            values: (batch, values)
            bin_edges: (batch, edges)
            weights: (batch, values)

        Returns:
            (batch, edges) with dtype matching weights
        """
        raise NotImplementedError(self)

    def bincount(self, x, weights, bins: int):
        raise NotImplementedError(self)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError(self)

    def quantile(self, x, quantiles):
        """
        Reduces the last / inner axis of x.

        Args:
            x: Tensor
            quantiles: List or 1D tensor of quantiles to compute.

        Returns:
            Tensor with shape (quantiles, *x.shape[:-1])
        """
        raise NotImplementedError(self)

    def argsort(self, x, axis=-1):
        raise NotImplementedError(self)

    def searchsorted(self, sorted_sequence, search_values, side: str, dtype=DType(int, 32)):
        raise NotImplementedError(self)

    def fft(self, x, axes: Union[tuple, list]):
        """
        Computes the n-dimensional FFT along all but the first and last dimensions.

        Args:
          x: tensor of dimension 3 or higher
          axes: Along which axes to perform the FFT

        Returns:
            Complex tensor `k`
        """
        raise NotImplementedError(self)

    def ifft(self, k, axes: Union[tuple, list]):
        """
        Computes the n-dimensional inverse FFT along all but the first and last dimensions.

        Args:
          k: tensor of dimension 3 or higher
          axes: Along which axes to perform the inverse FFT

        Returns:
            Complex tensor `x`
        """
        raise NotImplementedError(self)

    def imag(self, x):
        raise NotImplementedError(self)

    def real(self, x):
        raise NotImplementedError(self)

    def conj(self, x):
        raise NotImplementedError(self)

    def sin(self, x):
        raise NotImplementedError(self)

    def arcsin(self, x):
        raise NotImplementedError(self)

    def cos(self, x):
        raise NotImplementedError(self)

    def arccos(self, x):
        raise NotImplementedError(self)

    def tan(self, x):
        raise NotImplementedError(self)

    def arctan(self, x):
        raise NotImplementedError(self)

    def arctan2(self, y, x):
        raise NotImplementedError(self)

    def sinh(self, x):
        raise NotImplementedError(self)

    def arcsinh(self, x):
        raise NotImplementedError(self)

    def cosh(self, x):
        raise NotImplementedError(self)

    def arccosh(self, x):
        raise NotImplementedError(self)

    def tanh(self, x):
        raise NotImplementedError(self)

    def arctanh(self, x):
        raise NotImplementedError(self)

    def log(self, x):
        """ Natural logarithm """
        raise NotImplementedError(self)

    def log2(self, x):
        raise NotImplementedError(self)

    def log10(self, x):
        raise NotImplementedError(self)

    def sigmoid(self, x):
        return 1 / (1 + self.exp(-x))

    def dtype(self, array) -> DType:
        raise NotImplementedError(self)

    def tile(self, value, multiples):
        """
        Repeats the full tensor along each axis the number of times given by multiples.
        If `multiples` has more dimensions than `value`, these dimensions are added to `value` as outer dimensions.

        Args:
            value: tensor
            multiples: tuple or list of integers

        Returns:
            tiled tensor
        """
        raise NotImplementedError(self)

    def repeat(self, x, repeats, axis: int, new_length=None):
        """
        Repeats the elements along `axis` `repeats` times.

        Args:
            x: Tensor
            repeats: How often to repeat each element. 1D tensor of length x.shape[axis]
            axis: Which axis to repeat elements along
            new_length: Set the length of `axis` after repeating. This is required for jit compilation with Jax.

        Returns:
            repeated Tensor
        """
        raise NotImplementedError(self)

    def get_diagonal(self, matrices, offset=0):
        """

        Args:
            matrices: (batch, rows, cols, channels)
            offset: 0=diagonal, positive=above diagonal, negative=below diagonal

        Returns:
            diagonal: (batch, max(rows,cols), channels)
        """
        raise NotImplementedError(self)

    def indexed_segment_sum(self, x, indices, axis: int):
        """
        Args:
            x: Values to sum. Segments are laid out contiguously along `axis`. (batch, ...)
            indices: should start with 0 along `axis`. (batch, indices)
            axis: Axis along which to sum

        Returns:
            Tensor with `len(indices)` elements along `axis`. (batch, ..., indices, ...)
        """
        raise NotImplementedError(self)

    def sparse_coo_tensor(self, indices: Union[tuple, list], values, shape: tuple):
        """
        Create a sparse matrix in coordinate list (COO) format.

        Optional feature.

        See Also:
            `Backend.csr_matrix()`, `Backend.csc_matrix()`.

        Args:
            indices: 2D tensor of shape `(nnz, dims)`.
            values: 1D values tensor matching `indices`
            shape: Shape of the sparse matrix

        Returns:
            Native representation of the sparse matrix
        """
        raise NotImplementedError(self)

    def sparse_coo_tensor_batched(self, indices: Union[tuple, list], values, shape: tuple):
        """
        Args:
            indices: shape (batch_size, dims, nnz)
            values: Values tensor matching `indices`, shape (batch_size, nnz)
            shape: tuple of two ints representing the dense shape, (dims...)
        """
        raise NotImplementedError(self)

    def mul_coo_dense(self, indices, values, shape, dense):
        """
        Multiply a batch of sparse coordinate matrices by a batch of dense matrices.
        Every backend should implement this feature.
        This is the fallback if CSR multiplication is not supported.

        Args:
            indices: (batch, nnz, ndims)
            values: (batch, nnz, channels)
            shape: Shape of the full matrix, tuple of length ndims
            dense: (batch, dense_rows=sparse_cols, channels, dense_cols)

        Returns:
            (batch, channels, dense_rows=sparse_cols, dense_cols)
        """
        values, dense = self.auto_cast(values, dense)
        batch_size, nnz, channel_count = self.staticshape(values)
        _, dense_rows, _, dense_cols = self.staticshape(dense)
        dense_formatted = self.reshape(dense, (batch_size, dense_rows, dense_cols * channel_count))
        dense_gathered = self.batched_gather_nd(dense_formatted, indices[:, :, 1:2])
        base_grid = self.zeros((batch_size, shape[0], dense.shape[3] * dense_cols), self.dtype(dense))
        assert dense_cols == 1
        result = self.scatter(base_grid, indices[:, :, 0:1], values * dense_gathered, mode='add')
        return self.reshape(result, (batch_size, channel_count, dense_rows, dense_cols))

    def coo_to_dense(self, indices, values, shape, contains_duplicates: bool):
        batch_size, nnz, channel_count = self.staticshape(values)
        base = self.zeros((batch_size, *shape, channel_count))
        result = self.scatter(base, indices, values, mode='add' if contains_duplicates else 'update')
        return result

    def ilu_coo(self, indices, values, shape, iterations: int, safe: bool):
        """ See incomplete_lu_coo() in _linalg """
        from ._linalg import incomplete_lu_coo
        assert self.dtype(values).kind in (bool, int, float)
        return incomplete_lu_coo(self, indices, self.to_float(values), shape, iterations, safe)

    def ilu_dense(self, matrix, iterations: int, safe: bool):
        """ See incomplete_lu_dense() in _linalg """
        from ._linalg import incomplete_lu_dense
        assert self.dtype(matrix).kind in (bool, int, float)
        return incomplete_lu_dense(self, self.to_float(matrix), iterations, safe)

    def csr_matrix(self, column_indices, row_pointers, values, shape: Tuple[int, int]):
        """
        Create a sparse matrix in compressed sparse row (CSR) format.

        Optional feature.

        See Also:
            `Backend.sparse_coo_tensor()`, `Backend.csc_matrix()`.

        Args:
            column_indices: Column indices corresponding to `values`, 1D tensor
            row_pointers: Indices in `values` where any row starts, 1D tensor of length `rows + 1`
            values: Non-zero values, 1D tensor
            shape: Shape of the full matrix

        Returns:
            Native representation of the sparse matrix
        """
        raise NotImplementedError(self)

    def csr_matrix_batched(self, column_indices, row_pointers, values, shape: Tuple[int, int]):
        """
        Args:
            column_indices: Column indices corresponding to `values`, shape (batch_size, nnz)
            row_pointers: Indices in `values` where any row starts, shape (batch_size, rows+1)
            values: Non-zero values, shape (batch_size, nnz, channels)
            shape: tuple of two ints representing the dense shape, (cols, rows)
        """
        raise NotImplementedError(self)

    def mul_csr_dense(self, column_indices, row_pointers, values, shape: Tuple[int, int], dense):
        """
        Multiply a batch of compressed sparse row matrices by a batch of dense matrices.

        Optional feature.

        See Also:
            `Backend.sparse_coo_tensor()`, `Backend.csc_matrix()`.

        Args:
            column_indices: (batch, nnz)
            row_pointers: (batch, rows + 1)
            values: (batch, nnz, channels)
            shape: Shape of the full matrix (cols, rows)
            dense: (batch, dense_rows=sparse_cols, channels, dense_cols)

        Returns:
            (batch, channels, dense_rows=sparse_cols, dense_cols)
        """
        # if not self.supports(Backend.indexed_segment_sum):
        native_coo_indices = self.csr_to_coo(column_indices, row_pointers)
        return self.mul_coo_dense(native_coo_indices, values, shape, dense)
        # values, dense = self.auto_cast(values, dense)
        # batch_size, nnz, channel_count = self.staticshape(values)
        # _, dense_rows, _, dense_cols = self.staticshape(dense)
        # assert dense_cols == 1
        # dense_formatted = self.reshape(dense, (batch_size, dense_rows, channel_count * dense_cols))
        # dense_gathered = self.batched_gather_nd(dense_formatted, self.expand_dims(column_indices, -1))  # (batch, nnz, channels*rhs_cols)
        # dense_gathered = self.reshape(dense_gathered, (batch_size, nnz, channel_count, dense_cols))
        # values = self.reshape(values, (batch_size, nnz, channel_count, 1))
        # result = self.indexed_segment_sum(values * dense_gathered, row_pointers[:, :-1], 1)
        # return self.reshape(result, (batch_size, channel_count, rhs_rows, rhs_cols))

    def csr_to_coo(self, column_indices, row_pointers):
        """
        Convert a batch of compressed sparse matrices to sparse coordinate matrices.

        Args:
            column_indices: (batch, nnz)
            row_pointers: (batch, rows + 1)

        Returns:
            indices: (batch, nnz, 2)
        """
        batch_size = self.staticshape(column_indices)[0]
        repeats = row_pointers[:, 1:] - row_pointers[:, :-1]
        row_count = self.shape(repeats)[-1]
        row_indices = [self.repeat(self.range(row_count, dtype=self.dtype(column_indices)), repeats[b], -1) for b in range(batch_size)]
        return self.stack([self.stack(row_indices), column_indices], axis=-1)

    def csr_to_dense(self, column_indices, row_pointers, values, shape: Tuple[int, int]):
        indices = self.csr_to_coo(column_indices, row_pointers)
        return self.coo_to_dense(indices, values, shape, contains_duplicates=False)

    def csc_matrix(self, column_pointers, row_indices, values, shape: Tuple[int, int]):
        """
        Create a sparse matrix in compressed sparse column (CSC) format.

        Optional feature.

        See Also:
            `Backend.sparse_coo_tensor()`, `Backend.csr_matrix()`.

        Args:
            column_pointers: Indices in `values` where any column starts, 1D tensor of length `cols + 1`
            row_indices: Row indices corresponding to `values`.
            values: Non-zero values, 1D tensor
            shape: Shape of the full matrix

        Returns:
            Native representation of the sparse matrix
        """
        raise NotImplementedError(self)

    def csc_matrix_batched(self, column_pointers, row_indices, values, shape: Tuple[int, int]):
        """
        Args:
            column_pointers: Indices in `values` where any row starts, shape (batch_size, cols+1)
            row_indices: Row indices corresponding to `values`, shape (batch_size, nnz)
            values: Non-zero values, shape (batch_size, nnz, channels)
            shape: tuple of two ints representing the dense shape, (cols, rows)
        """
        raise NotImplementedError(self)

    def pairwise_distances(self, positions, max_radius, format: str, index_dtype=DType(int, 32)) -> list:
        """

        Args:
            positions: Point locations of shape (batch, instances, vector)
            max_radius: Scalar or (batch,) or (batch, instances)
            format: 'csr',  Not yet implemented: 'sparse', 'coo', 'csc'
            index_dtype: Either int32 or int64

        Returns:
            Sequence of batch_size sparse distance matrices
        """
        from sklearn import neighbors
        batch_size, point_count, _vec_count = self.staticshape(positions)
        positions_np_batched = self.numpy(positions)
        result = []
        for i in range(batch_size):
            tree = neighbors.KDTree(positions_np_batched[i])
            radius = float(max_radius) if len(self.staticshape(max_radius)) == 0 else max_radius[i]
            nested_neighbors = tree.query_radius(positions_np_batched[i], r=radius)  # ndarray[ndarray]
            if format == 'csr':
                column_indices = numpy.concatenate(nested_neighbors).astype(to_numpy_dtype(index_dtype))  # flattened_neighbors
                neighbor_counts = [len(nlist) for nlist in nested_neighbors]
                row_pointers = numpy.concatenate([[0], numpy.cumsum(neighbor_counts)]).astype(to_numpy_dtype(index_dtype))
                pos_neighbors = self.gather(positions[i], column_indices, 0)
                pos_self = self.repeat(positions[i], neighbor_counts, axis=0)
                values = pos_neighbors - pos_self
                result.append((column_indices, row_pointers, values))
                # sparse_matrix = self.csr_matrix(column_indices, row_pointers, values, (point_count, point_count))
                # sparse_matrix.eliminate_zeros()  # setdiag(0) keeps zero entries
            else:
                raise NotImplementedError(format)
        return result

    def minimize(self, method: str, f, x0, atol, max_iter, trj: bool):
        if method == 'auto':
            method = 'L-BFGS-B'
        if method == 'GD':
            from ._minimize import gradient_descent
            return gradient_descent(self, f, x0, atol, max_iter, trj)
        else:
            from ._minimize import scipy_minimize
            return scipy_minimize(self, method, f, x0, atol, max_iter, trj)

    def linear_solve(self, method: str, lin, y, x0, tol_sq, max_iter) -> SolveResult:
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
            tol_sq: Squared absolute tolerance of size (batch,)
            max_iter: Maximum number of iterations of size (batch,) or a sequence of maximum iterations to obtain a trajectory.

        Returns:
            `SolveResult`
        """
        if method == 'auto':
            return self.conjugate_gradient_adaptive(lin, y, x0, tol_sq, max_iter)
        elif method == 'CG':
            return self.conjugate_gradient(lin, y, x0, tol_sq, max_iter)
        elif method == 'CG-adaptive':
            return self.conjugate_gradient_adaptive(lin, y, x0, tol_sq, max_iter)
        elif method in ['biCG', 'biCG-stab(0)']:
            raise NotImplementedError("Unstabilized Bi-CG not yet supported")
            # return self.bi_conjugate_gradient_original(lin, y, x0, tol_sq, max_iter)
        elif method == 'biCG-stab':
            return self.bi_conjugate_gradient(lin, y, x0, tol_sq, max_iter, poly_order=1)
        elif method.startswith('biCG-stab('):
            order = int(method[len('biCG-stab('):-1])
            return self.bi_conjugate_gradient(lin, y, x0, tol_sq, max_iter, poly_order=order)
        else:
            raise NotImplementedError(f"Method '{method}' not supported for linear solve.")

    def conjugate_gradient(self, lin, y, x0, tol_sq, max_iter) -> SolveResult:
        """ Standard conjugate gradient algorithm. Signature matches to `Backend.linear_solve()`. """
        from ._linalg import cg, stop_on_l2
        return cg(self, lin, y, x0, stop_on_l2(self, tol_sq, max_iter), max_iter)

    def conjugate_gradient_adaptive(self, lin, y, x0, tol_sq, max_iter) -> SolveResult:
        """ Conjugate gradient algorithm with adaptive step size. Signature matches to `Backend.linear_solve()`. """
        from ._linalg import cg_adaptive, stop_on_l2
        return cg_adaptive(self, lin, y, x0, stop_on_l2(self, tol_sq, max_iter), max_iter)

    def bi_conjugate_gradient(self, lin, y, x0, tol_sq, max_iter, poly_order=2) -> SolveResult:
        """ Generalized stabilized biconjugate gradient algorithm. Signature matches to `Backend.linear_solve()`. """
        from ._linalg import bicg, stop_on_l2
        return bicg(self, lin, y, x0, stop_on_l2(self, tol_sq, max_iter), max_iter, poly_order)

    def linear(self, lin, vector):
        if callable(lin):
            return lin(vector)
        elif isinstance(lin, (tuple, list)):
            for lin_i in lin:
                lin_shape = self.staticshape(lin_i)
                assert len(lin_shape) == 2
            return self.stack([self.mul_matrix_batched_vector(m, v) for m, v in zip(lin, self.unstack(vector))])
        else:
            lin_shape = self.staticshape(lin)
            assert len(lin_shape) == 2, f"A must be a matrix but got shape {lin_shape}"
            return self.mul_matrix_batched_vector(lin, vector)

    def matrix_solve_least_squares(self, matrix: TensorType, rhs: TensorType) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        """
        Args:
            matrix: Shape (batch, vec, constraints)
            rhs: Shape (batch, vec, batch_per_matrix)

        Returns:
            solution: Solution vector of Shape (batch, constraints, batch_per_matrix)
            residuals: Optional, can be `None`
            rank: Optional, can be `None`
            singular_values: Optional, can be `None`
        """
        raise NotImplementedError(self)

    def solve_triangular_dense(self, matrix, rhs, lower: bool, unit_diagonal: bool):
        """

        Args:
            matrix: (batch_size, rows, cols)
            rhs: (batch_size, cols)
            lower:
            unit_diagonal:

        Returns:
            (batch_size, cols)
        """
        raise NotImplementedError(self)

    def stop_gradient(self, value):
        raise NotImplementedError(self)

    def stop_gradient_tree(self, tree):
        if isinstance(tree, tuple):
            return tuple([self.stop_gradient_tree(v) for v in tree])
        if isinstance(tree, list):
            return [self.stop_gradient_tree(v) for v in tree]
        if isinstance(tree, dict):
            return {k: self.stop_gradient_tree(v) for k, v in tree.items()}
        return self.stop_gradient(tree)

    def grid_sample(self, grid, coordinates, extrapolation: str):
        """
        Interpolates a regular grid at the specified coordinates.

        Args:
            grid: Tensor of shape (batch, spatial..., channel)
            coordinates: Tensor of floating grid indices of shape (batch, instance..., vector).
                The last dimension must match `spatial_dims`.
                The first grid point of dimension i lies at position 0, the last at values.shape[i]-1.
            extrapolation: Values to use for coordinates outside the grid.
                One of `('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect')`.

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

    def multi_slice(self, tensor, slices: tuple):
        """
        Args:
            tensor: value to slice
            slices: `tuple` of `slice`, `int`, or scalar integer tensors
        """
        return tensor[slices]

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
        a, b = self.auto_cast(a, b, bool_to_int=True)
        return a + b

    def sub(self, a, b):
        a, b = self.auto_cast(a, b, bool_to_int=True)
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

    def shift_bits_left(self, a, b):
        a, b = self.auto_cast(a, b)
        return a << b

    def shift_bits_right(self, a, b):
        a, b = self.auto_cast(a, b)
        return a >> b


BACKENDS = []
""" Global list of all registered backends. Register a `Backend` by adding it to the list. """
_DEFAULT = []  # [0] = global default, [1:] from 'with' blocks
_PRECISION = [32]  # [0] = global precision in bits, [1:] from 'with' blocks


def choose_backend(*values, prefer_default=False) -> Backend:
    """
    Selects a suitable backend to handle the given values.

    This function is used by most math functions operating on `Tensor` objects to delegate the actual computations.

    Backends need to be registered to be available, e.g. via the global import `phi.<backend>` or `phi.detect_backends()`.

    Args:
        *values:
        prefer_default: Whether to always select the default backend if it can work with `values`, see `default_backend()`.

    Returns:
        The selected `Backend`
    """
    # --- Default Backend has priority ---
    if _is_applicable(_DEFAULT[-1], values) and (prefer_default or _is_specific(_DEFAULT[-1], values)):
        return _DEFAULT[-1]
    # --- Filter out non-applicable ---
    backends = [backend for backend in BACKENDS if _is_applicable(backend, values)]
    if len(backends) == 0:
        raise NoBackendFound(f"No backend found for types {[type(v).__name__ for v in values]}; registered backends are {BACKENDS}")
    # --- Native tensors? ---
    for backend in backends:
        if _is_specific(backend, values):
            return backend
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


def context_backend() -> Union[Backend, None]:
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
        if not (backend.is_tensor(value, only_native=False) or backend.is_module(value)):
            return False
    return True


def _is_specific(backend: Backend, values):
    for value in values:
        if backend.is_tensor(value, only_native=True) or backend.is_module(value):
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


_SPATIAL_DERIVATIVE_CONTEXT = [0]
_FUNCTIONAL_DERIVATIVE_CONTEXT = [0]


@contextmanager
def spatial_derivative_evaluation(order=1):
    _SPATIAL_DERIVATIVE_CONTEXT.append(order)
    try:
        yield None
    finally:
        assert _SPATIAL_DERIVATIVE_CONTEXT.pop(-1) == order


def get_spatial_derivative_order():
    """
    Extrapolations may behave differently when extrapolating the derivative of a grid.
    Returns 1 inside a CG loop, and 0 by default.
    """
    return _SPATIAL_DERIVATIVE_CONTEXT[-1]


@contextmanager
def functional_derivative_evaluation(order=1):
    _FUNCTIONAL_DERIVATIVE_CONTEXT.append(order)
    try:
        yield None
    finally:
        assert _FUNCTIONAL_DERIVATIVE_CONTEXT.pop(-1) == order


def get_functional_derivative_order():
    """
    Operations that do not define a first or higher-order derivative may use slower alternative code paths when the derivative is `>0`.
    This is set when calling a function created by `math.jacobian()` or `math.hessian()`.
    """
    return _FUNCTIONAL_DERIVATIVE_CONTEXT[-1]


PHI_LOGGER = logging.getLogger('Φ')  # used for warnings and debug messages by all internal PhiFlow functions
_LOG_CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
_LOG_CONSOLE_HANDLER.setFormatter(logging.Formatter("%(message)s (%(levelname)s), %(asctime)sn\n"))
_LOG_CONSOLE_HANDLER.setLevel(logging.NOTSET)
PHI_LOGGER.addHandler(_LOG_CONSOLE_HANDLER)

import numbers
from contextlib import contextmanager
from functools import wraps, partial
from typing import List, Callable

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from ..math.backend._backend import combined_dim
from ..math.backend._dtype import DType, to_numpy_dtype, from_numpy_dtype
from phi.math.backend import Backend, ComputeDevice, NUMPY_BACKEND
from ._tf_cuda_resample import resample_cuda, use_cuda


class TFBackend(Backend):

    def __init__(self):
        Backend.__init__(self, "TensorFlow", default_device=None)

    def prefers_channels_last(self) -> bool:
        return True

    def list_devices(self, device_type: str or None = None) -> List[ComputeDevice]:
        tf_devices = device_lib.list_local_devices()
        devices = []
        for device in tf_devices:
            if device_type in (None, device.device_type):
                devices.append(ComputeDevice(self, device.name, device.device_type, device.memory_limit,
                                             processor_count=-1,
                                             description=str(device),
                                             ref=device))
        return devices

    def seed(self, seed: int):
        tf.random.set_seed(seed)

    def is_tensor(self, x, only_native=False):
        is_tf_tensor = tf.is_tensor(x) is True  # tf.is_tensor() can return non-bool values which indicates not a Tensor
        if only_native:
            return is_tf_tensor
        else:
            return is_tf_tensor or NUMPY_BACKEND.is_tensor(x, only_native=False)

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            return x
        tensor = tf.convert_to_tensor(x)
        # --- Enforce Precision ---
        if not isinstance(tensor, numbers.Number):
            if isinstance(tensor, np.ndarray):
                tensor = NUMPY_BACKEND.as_tensor(tensor)
            elif tensor.dtype.is_floating:
                tensor = self.to_float(tensor)
        return tensor

    def is_available(self, tensor) -> bool:
        if self.is_tensor(tensor, only_native=True):
            return tf.executing_eagerly()
        else:
            return True

    def numpy(self, tensor):
        if tf.is_tensor(tensor):
            return tensor.numpy()
        return NUMPY_BACKEND.numpy(tensor)

    def to_dlpack(self, tensor):
        from tensorflow import experimental
        return experimental.dlpack.to_dlpack(tensor)

    def from_dlpack(self, capsule):
        from tensorflow import experimental
        return experimental.dlpack.from_dlpack(capsule)

    def copy(self, tensor, only_mutable=False):
        if not only_mutable or tf.executing_eagerly():
            return tf.identity(tensor)
        else:
            return tensor

    def jit_compile(self, f: Callable) -> Callable:
        return tf.function(f)

    def custom_gradient(self, f: Callable, gradient: Callable = None) -> Callable:
        @tf.custom_gradient
        def tf_function(*args, **kwargs):
            def grad(*grad_args):
                return gradient(args, y, grad_args)
            y = f(*args, **kwargs)
            return y, grad
        return tf_function

    def transpose(self, tensor, axes):
        return tf.transpose(tensor, perm=axes)

    def equal(self, x, y):
        return tf.equal(x, y)

    def divide_no_nan(self, x, y):
        x, y = self.auto_cast(x, y)
        return tf.math.divide_no_nan(x, y)

    def random_uniform(self, shape):
        return tf.random.uniform(shape, dtype=to_numpy_dtype(self.float_type))

    def random_normal(self, shape):
        return tf.random.normal(shape, dtype=to_numpy_dtype(self.float_type))

    def rank(self, value):
        return len(value.shape)

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
        return tf.range(start, limit, delta, to_numpy_dtype(dtype))

    def tile(self, value, multiples):
        if isinstance(multiples, (tuple, list)) and self.ndims(value) < len(multiples):
            value = self.expand_dims(value, axis=0, number=len(multiples) - self.ndims(value))
        return tf.tile(value, multiples)

    def stack(self, values, axis=0):
        return tf.stack(values, axis=axis)

    def concat(self, values, axis):
        return tf.concat(values, axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        if mode == 'boundary' and np.all(np.array(pad_width) <= 1):
            mode = 'symmetric'
        if mode in ('constant', 'symmetric', 'reflect'):
            return tf.pad(value, pad_width, mode.upper(), constant_values=constant_values)
        else:
            return NotImplemented

    def reshape(self, value, shape):
        return tf.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        if isinstance(value, tf.SparseTensor):
            return tf.sparse.reduce_sum(value, axis=axis, keepdims=keepdims, output_is_sparse=False)
        if isinstance(value, (tuple, list)) and any([isinstance(x, tf.SparseTensor) for x in value]):
            result = value[0]
            for v in value[1:]:
                result = tf.sparse.add(result, v, threshold=0)
            return result
        return tf.reduce_sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        if value.dtype == bool:
            return tf.reduce_all(value, axis=axis)
        return tf.reduce_prod(value, axis=axis)

    def where(self, condition, x=None, y=None):
        c = self.cast(condition, self.dtype(x))
        return c * x + (1 - c) * y
        # return tf.where(condition, x, y)  # TF1 has an inconsistent broadcasting rule for where

    def nonzero(self, values):
        return tf.where(tf.not_equal(values, 0))

    def mean(self, value, axis=None, keepdims=False):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        return tf.reduce_mean(value, axis, keepdims=keepdims)

    def grid_sample(self, grid, spatial_dims: tuple, coordinates, extrapolation='constant'):
        if use_cuda(grid):
            # TODO reshape for spatial_dims
            return resample_cuda(grid, coordinates, extrapolation)
        else:
            return NotImplemented

    def zeros(self, shape, dtype: DType = None):
        return tf.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def zeros_like(self, tensor):
        return tf.zeros_like(tensor)

    def ones(self, shape, dtype: DType = None):
        return tf.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def ones_like(self, tensor):
        return tf.ones_like(tensor)

    def meshgrid(self, *coordinates):
        result = tf.meshgrid(*coordinates, indexing='ij')
        return result

    def linspace(self, start, stop, number):
        return self.to_float(tf.linspace(start, stop, number))

    def tensordot(self, a, a_axes: tuple or list, b, b_axes: tuple or list):
        return tf.tensordot(a, b, (a_axes, b_axes))

    def matmul(self, A, b):
        if isinstance(A, tf.SparseTensor):
            result_T = tf.sparse.sparse_dense_matmul(A, tf.transpose(b))  # result shape contains unknown size
            result = tf.transpose(result_T)
            result.set_shape(tf.TensorShape([b.shape[0], A.shape[0]]))
            return result
        else:
            return tf.matmul(A, b)

    def einsum(self, equation, *tensors):
        return tf.einsum(equation, *tensors)

    def while_loop(self, loop: Callable, values: tuple):
        cond = lambda c, *vals: tf.reduce_any(c)
        return tf.nest.map_structure(tf.stop_gradient, tf.while_loop(cond, loop, values))

    def abs(self, x):
        return tf.abs(x)

    def sign(self, x):
        return tf.sign(x)

    def round(self, x):
        return tf.round(x)

    def ceil(self, x):
        return tf.math.ceil(x)

    def floor(self, x):
        return tf.floor(x)

    def max(self, x, axis=None, keepdims=False):
        if isinstance(x, (tuple, list)):
            x = tf.stack(x)
        if x.dtype == tf.bool:
            return tf.cast(tf.reduce_max(tf.cast(x, tf.uint8), axis=axis, keepdims=keepdims), tf.bool)  # reduce_max allows no bool
        return tf.reduce_max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        if isinstance(x, (tuple, list)):
            x = tf.stack(x)
        if x.dtype == tf.bool:
            return tf.cast(tf.reduce_min(tf.cast(x, tf.uint8), axis=axis, keepdims=keepdims), tf.bool)  # reduce_min allows no bool
        return tf.reduce_min(x, axis=axis, keepdims=keepdims)

    def maximum(self, a, b):
        a, b = self.auto_cast(a, b)
        return tf.maximum(a, b)

    def minimum(self, a, b):
        a, b = self.auto_cast(a, b)
        return tf.minimum(a, b)

    def clip(self, x, minimum, maximum):
        x, minimum, maximum = self.auto_cast(x, minimum, maximum)
        return tf.clip_by_value(x, minimum, maximum)

    def sqrt(self, x):
        return tf.sqrt(x)

    def exp(self, x):
        return tf.exp(x)

    def conv(self, value, kernel, zero_padding=True):
        value = self.to_float(value)
        kernel = self.to_float(kernel)  # should use auto_cast but TensorFlow only supports DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32
        if zero_padding:
            value_padding = [[0, 0]] * 2 + [[s // 2, (s - 1) // 2] for s in kernel.shape[3:]]
            value = tf.pad(value, value_padding)
        convf = {3: partial(tf.nn.conv1d, stride=1),
                 4: partial(tf.nn.conv2d, strides=[1, 1, 1, 1]),
                 5: partial(tf.nn.conv3d, strides=[1, 1, 1, 1, 1])}[len(value.shape)]
        value = tf.transpose(value, [0, *range(2, value.ndim), 1])  # could use data_format='NC...' but it's supported neither on CPU and for int tensors
        kernel = tf.transpose(kernel, [0, *range(3, kernel.ndim), 2, 1])
        if kernel.shape[0] == 1:
            result = convf(value, kernel[0, ...], padding='VALID')
        else:
            result = []
            for b in range(kernel.shape[0]):
                result.append(convf(value[b:b+1, ...], kernel[b], padding='VALID'))
            result = tf.concat(result, 0)
        result = tf.transpose(result, [0, result.ndim - 1, *range(1, result.ndim - 1)])
        return result

    def expand_dims(self, a, axis=0, number=1):
        if number == 0:
            return a
        for _i in range(number):
            a = tf.expand_dims(a, axis)
        return a

    def shape(self, tensor):
        return tf.shape(tensor)

    def staticshape(self, tensor):
        if self.is_tensor(tensor, only_native=True):
            return tuple(tensor.shape.as_list())
        else:
            return np.shape(tensor)

    def batched_gather_nd(self, values, indices):
        if self.staticshape(values)[0] == 1 and self.staticshape(indices)[0] != 1:
            result = tf.gather_nd(values[0, ...], indices, batch_dims=0)
            return result
        return tf.gather_nd(values, indices, batch_dims=1)

    def unstack(self, tensor, axis=0, keepdims=False):
        unstacked = tf.unstack(tensor, axis=axis)
        if keepdims:
            unstacked = [self.expand_dims(c, axis=axis) for c in unstacked]
        return unstacked

    def std(self, x, axis=None, keepdims=False):
        _mean, var = tf.nn.moments(x, axis, keepdims=keepdims)
        return tf.sqrt(var)

    def boolean_mask(self, x, mask, axis=0):
        return tf.boolean_mask(x, mask, axis=axis)

    def isfinite(self, x):
        return tf.math.is_finite(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return tf.reduce_any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return tf.reduce_all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, base_grid, indices, values, mode: str):
        base_grid, values = self.auto_cast(base_grid, values)
        indices = self.as_tensor(indices)
        batch_size = combined_dim(combined_dim(indices.shape[0], values.shape[0]), base_grid.shape[0])
        scatter = tf.tensor_scatter_nd_add if mode == 'add' else tf.tensor_scatter_nd_update
        result = []
        for b in range(batch_size):
            b_grid = base_grid[b, ...]
            b_indices = indices[min(b, indices.shape[0] - 1), ...]
            b_values = values[min(b, values.shape[0] - 1), ...]
            result.append(scatter(b_grid, b_indices, b_values))
        return self.stack(result, axis=0)

    def fft(self, x):
        rank = len(x.shape) - 2
        assert rank >= 1
        x = self.to_complex(x)
        if rank == 1:
            return tf.stack([tf.signal.fft(c) for c in tf.unstack(x, axis=-1)], axis=-1)
        elif rank == 2:
            return tf.stack([tf.signal.fft2d(c) for c in tf.unstack(x, axis=-1)], axis=-1)
        elif rank == 3:
            return tf.stack([tf.signal.fft3d(c) for c in tf.unstack(x, axis=-1)], axis=-1)
        else:
            raise NotImplementedError('n-dimensional FFT not implemented.')

    def ifft(self, k):
        rank = len(k.shape) - 2
        assert rank >= 1
        if rank == 1:
            return tf.stack([tf.signal.ifft(c) for c in tf.unstack(k, axis=-1)], axis=-1)
        elif rank == 2:
            return tf.stack([tf.signal.ifft2d(c) for c in tf.unstack(k, axis=-1)], axis=-1)
        elif rank == 3:
            return tf.stack([tf.signal.ifft3d(c) for c in tf.unstack(k, axis=-1)], axis=-1)
        else:
            raise NotImplementedError('n-dimensional inverse FFT not implemented.')

    def imag(self, complex):
        return tf.math.imag(complex)

    def real(self, complex):
        return tf.math.real(complex)

    def cast(self, x, dtype: DType):
        if not self.is_tensor(x, only_native=True):
            x = self.as_tensor(x, convert_external=True)
        if self.dtype(x) == dtype:
            return x
        else:
            return tf.cast(x, to_numpy_dtype(dtype))

    def sin(self, x):
        return tf.math.sin(x)

    def cos(self, x):
        return tf.math.cos(x)

    def tan(self, x):
        return tf.math.tan(x)

    def log(self, x):
        return tf.math.log(x)

    def log2(self, x):
        return tf.math.log(x) / 0.6931471805599453094  # log(x) / log(2)

    def log10(self, x):
        return tf.math.log(x) / 2.3025850929940456840  # log(x) / log(10)

    def dtype(self, array) -> DType:
        if tf.is_tensor(array):
            dt = array.dtype.as_numpy_dtype
            return from_numpy_dtype(dt)
        else:
            return NUMPY_BACKEND.dtype(array)

    def sparse_tensor(self, indices, values, shape):
        indices = [tf.convert_to_tensor(i, tf.int64) for i in indices]
        indices = tf.cast(tf.stack(indices, axis=-1), tf.int64)
        return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

    def coordinates(self, tensor, unstack_coordinates=False):
        if isinstance(tensor, tf.SparseTensor):
            idx = tensor.indices
            if unstack_coordinates:
                idx = tf.unstack(idx, axis=-1)
            return idx, tensor.values
        else:
            raise NotImplementedError()

    def add(self, a, b):
        if isinstance(a, tf.SparseTensor) or isinstance(b, tf.SparseTensor):
            return tf.sparse.add(a, b, threshold=1e-5)
        else:
            return Backend.add(self, a, b)

    def functional_gradient(self, f, wrt: tuple or list, get_output: bool):
        @wraps(f)
        def eval_grad(*args):
            args = [self.as_tensor(arg, True) if i in wrt else arg for i, arg in enumerate(args)]
            wrt_args = [arg for i, arg in enumerate(args) if i in wrt]
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                for arg in wrt_args:
                    assert arg.dtype in (tf.float16, tf.float32, tf.float64, tf.complex64, tf.complex128), f"Gradients can only be computed for float or complex tensors but got {arg.dtype} for argument with shape {arg.shape}"
                    tape.watch(arg)
                output = f(*args)
            loss, aux = (output[0], output[1:]) if isinstance(output, (tuple, list)) else (output, None)
            grads = list(tape.gradient(loss, wrt_args))
            assert None not in grads, f"Gradient could not be computed for wrt argument {grads.index(None)} (argument {wrt[grads.index(None)]}) with shape {wrt_args[grads.index(None)].shape}. TensorFlow returned gradient=None."
            if get_output:
                if aux is not None:
                    return (loss, *aux, *grads)
                else:
                    return (loss, *grads)
            else:
                return grads
        return eval_grad

    # def variable(self, value):  # not supported, variables must record gradients outside a context
    #     return tf.Variable(value, trainable=True)

    def gradients(self, y, xs: tuple or list, grad_y):
        if _TAPES:
            tape = _TAPES[-1]
            return tape.gradient(y, xs, grad_y)
        return tf.gradients(y, xs, grad_y)

    @contextmanager
    def record_gradients(self, xs: tuple or list, persistent=False):
        tape = tf.GradientTape(persistent=persistent)
        tape.__enter__()
        for x in xs:
            tape.watch(x)
        _TAPES.append(tape)

        try:
            yield None
        finally:
            tape.__exit__(None, None, None)
            _TAPES.pop(-1)

    def stop_gradient(self, value):
        return tf.stop_gradient(value)


TF_BACKEND = TFBackend()
_TAPES = []

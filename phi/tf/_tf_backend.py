import numbers
import uuid
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from phi.math.backend import Backend, DType, to_numpy_dtype, from_numpy_dtype, ComputeDevice
from phi.math.backend._scipy_backend import SCIPY_BACKEND, SciPyBackend
from ._tf_cuda_resample import resample_cuda, use_cuda
from ..math.backend._backend_helper import combined_dim


class TFBackend(Backend):

    def __init__(self):
        Backend.__init__(self, "TensorFlow")

    def list_devices(self, device_type: str or None = None) -> List[ComputeDevice]:
        tf_devices = device_lib.list_local_devices()
        devices = []
        for device in tf_devices:
            if device_type in (None, device.device_type):
                devices.append(ComputeDevice(device.name,
                                             device.device_type,
                                             device.memory_limit,
                                             -1,
                                             str(device)))
        return devices

    def is_tensor(self, x, only_native=False):
        if only_native:
            return tf.is_tensor(x)
        else:
            return tf.is_tensor(x) or SCIPY_BACKEND.is_tensor(x, only_native=False)

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            tensor = x
        elif isinstance(x, np.ndarray):
            tensor = tf.convert_to_tensor(SciPyBackend(precision=self.precision).as_tensor(x))
        else:
            tensor = tf.convert_to_tensor(x)
        # --- Enforce Precision ---
        if not isinstance(tensor, numbers.Number):
            if isinstance(tensor, np.ndarray):
                tensor = SciPyBackend(precision=self.precision).as_tensor(tensor)
            elif tensor.dtype.is_floating and self.has_fixed_precision:
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
        return SCIPY_BACKEND.numpy(tensor)

    def copy(self, tensor, only_mutable=False):
        if not only_mutable or tf.executing_eagerly():
            return tf.identity(tensor)
        else:
            return tensor

    def transpose(self, tensor, axes):
        return tf.transpose(tensor, perm=axes)

    def equal(self, x, y):
        return tf.equal(x, y)

    def divide_no_nan(self, x, y):
        if x.dtype != y.dtype:
            # TODO: cast to complex is somehow broken
            x, y = self.auto_cast((x, y))
        return tf.math.divide_no_nan(x, y)

    def random_uniform(self, shape):
        return tf.random.uniform(shape, dtype=to_numpy_dtype(self.float_type))

    def random_normal(self, shape):
        return tf.random.normal(shape, dtype=to_numpy_dtype(self.float_type))

    def rank(self, value):
        return len(value.shape)

    def range(self, start, limit=None, delta=1, dtype=None):
        return tf.range(start, limit, delta, dtype)

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

    def mean(self, value, axis=None, keepdims=False):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        return tf.reduce_mean(value, axis, keepdims=keepdims)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        if grad is None:
            result = tf.py_func(func, inputs, Tout, stateful=stateful, name=name)
        else:
            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'PyFuncGrad' + str(uuid.uuid4())

            tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                result = tf.py_func(func, inputs, Tout, stateful=stateful, name=name)
        if shape_out is not None:
            result.set_shape(shape_out)
        return result

    def resample(self, inputs, sample_coords, interpolation='linear', boundary='constant', constant_values=0):
        assert interpolation == 'linear'
        if use_cuda(inputs):
            return resample_cuda(inputs, sample_coords, boundary)
        else:
            return general_grid_sample_nd(inputs, sample_coords, boundary, constant_values, self)  # while this is a bit slower than niftynet, it give consisten results at the boundaries

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

    def dot(self, a, b, axes):
        return tf.tensordot(a, b, axes)

    def matmul(self, A, b):
        if isinstance(A, tf.SparseTensor):
            result = tf.sparse.sparse_dense_matmul(A, tf.transpose(b))
            result = tf.transpose(result)
            # result.set_shape(tf.TensorShape([b.shape[0], A.shape[0]]))
            return result
        else:
            return tf.matmul(A, b)

    def einsum(self, equation, *tensors):
        return tf.einsum(equation, *tensors)

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True,
                   swap_memory=False, name=None, maximum_iterations=None):
        return tf.while_loop(cond, body, loop_vars,
                             shape_invariants=shape_invariants,
                             parallel_iterations=parallel_iterations,
                             back_prop=back_prop,
                             swap_memory=swap_memory,
                             name=name,
                             maximum_iterations=maximum_iterations)

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
        return tf.reduce_max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return tf.reduce_min(x, axis=axis, keepdims=keepdims)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base="custom_gradient_func"):
        # Setup custom gradient
        gradient_name = name_base + "_" + str(uuid.uuid4())
        tf.RegisterGradient(gradient_name)(gradient)

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": gradient_name}):
            fake_function = tf.identity(inputs[input_index])

        outputs = function(*inputs)
        output = outputs if output_index is None else outputs[output_index]
        output_with_gradient = fake_function + tf.stop_gradient(output - fake_function)
        if output_index is None:
            return output_with_gradient
        else:
            outputs = list(outputs)
            outputs[output_index] = output_with_gradient
            return outputs

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

    def conv(self, tensor, kernel, padding="SAME"):
        rank = len(tensor.shape) - 2
        padding = padding.upper()
        if rank == 1:
            result = tf.nn.conv1d(tensor, kernel, 1, padding)
        elif rank == 2:
            result = tf.nn.conv2d(tensor, kernel, [1, 1, 1, 1], padding)
        elif rank == 3:
            result = tf.nn.conv3d(tensor, kernel, [1, 1, 1, 1, 1], padding)
        else:
            raise ValueError("Tensor must be of rank 1, 2 or 3 but is %d" % rank)
        return result

    def expand_dims(self, a, axis=0, number=1):
        if number == 0:
            return a
        for _i in range(number):
            a = tf.expand_dims(a, axis)
        return a

    def shape(self, tensor):
        return tf.shape(tensor)

    def to_float(self, x):
        return tf.cast(x, to_numpy_dtype(self.float_type))

    def staticshape(self, tensor):
        if self.is_tensor(tensor, only_native=True):
            return tuple(tensor.shape.as_list())
        else:
            return np.shape(tensor)

    def to_int(self, x, int64=False):
        return tf.cast(x, tf.int64) if int64 else tf.cast(x, tf.int32)

    def to_complex(self, x):
        if self.dtype(x) in (np.complex64, np.complex128):
            return x
        if self.dtype(x) == np.float64:
            return tf.cast(x, tf.complex128)
        else:
            return tf.cast(x, tf.complex64)

    def gather(self, values, indices):
        if isinstance(values, tf.SparseTensor):
            if isinstance(indices, (tuple, list)) and indices[1] == slice(None):
                result = sparse_select_indices(values, indices[0], axis=0, are_indices_sorted=True, are_indices_uniqua=True)
                return result
        if isinstance(indices, slice):
            return values[indices]
        return tf.gather(values, indices)

    def gather_nd(self, values, indices, batch_dims=0):
        return tf.gather_nd(values, indices, batch_dims=batch_dims)

    def unstack(self, tensor, axis=0, keepdims=False):
        unstacked = tf.unstack(tensor, axis=axis)
        if keepdims:
            unstacked = [self.expand_dims(c, axis=axis) for c in unstacked]
        return unstacked

    def std(self, x, axis=None, keepdims=False):
        _mean, var = tf.nn.moments(x, axis, keepdims=keepdims)
        return tf.sqrt(var)

    def boolean_mask(self, x, mask):
        return tf.boolean_mask(x, mask)

    def isfinite(self, x):
        return tf.is_finite(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return tf.reduce_any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return tf.reduce_all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, indices, values, shape, duplicates_handling='undefined', outside_handling='undefined'):
        assert duplicates_handling in ('undefined', 'add', 'mean', 'any')
        assert outside_handling in ('discard', 'clamp', 'undefined')
        if duplicates_handling == 'undefined':
            pass

        # Change indexing so batch number is included as first element of the index, for example: [0,31,24] indexes the first batch (batch 0) and 2D coordinates (31,24).
        buffer = tf.zeros(shape, dtype=values.dtype)

        repetitions = []
        for dim in range(len(indices.shape) - 1):
            if values.shape[dim] == 1:
                repetitions.append(indices.shape[dim])
            else:
                assert indices.shape[dim] == values.shape[dim]
                repetitions.append(1)
        repetitions.append(1)
        values = self.tile(values, repetitions)

        if duplicates_handling == 'add':
            # Only for Tensorflow with custom gradient
            @tf.custom_gradient
            def scatter_density(points, indices, values):
                result = tf.tensor_scatter_add(buffer, indices, values)

                def grad(dr):
                    return self.resample(gradient(dr, difference='central'), points), None, None

                return result, grad

            return scatter_density(points, indices, values)
        elif duplicates_handling == 'mean':
            # Won't entirely work with out of bounds particles (still counted in mean)
            count = tf.tensor_scatter_add(buffer, indices, tf.ones_like(values))
            total = tf.tensor_scatter_add(buffer, indices, values)
            return total / tf.maximum(1.0, count)
        else:  # last, any, undefined
            # indices = self.to_int(indices, int64=True)
            # st = tf.SparseTensor(indices, values, shape)  # ToDo this only supports 2D shapes
            # st = tf.sparse.reorder(st)   # only needed if not ordered
            # return tf.sparse.to_dense(st)
            count = tf.tensor_scatter_add(buffer, indices, tf.ones_like(values))
            total = tf.tensor_scatter_add(buffer, indices, values)
            return total / tf.maximum(1.0, count)

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
        return tf.cast(x, to_numpy_dtype(dtype))

    def sin(self, x):
        return tf.math.sin(x)

    def cos(self, x):
        return tf.math.cos(x)

    def dtype(self, array) -> DType:
        if tf.is_tensor(array):
            dt = array.dtype.as_numpy_dtype
            return from_numpy_dtype(dt)
        else:
            return SCIPY_BACKEND.dtype(array)

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

    def conjugate_gradient(self, A, y, x0,
                           relative_tolerance: float = 1e-5,
                           absolute_tolerance: float = 0.0,
                           max_iterations: int = 1000,
                           gradient: str = 'implicit',
                           callback=None):
        backend = self

        batch_size = combined_dim(x0.shape[0], y.shape[0])
        if x0.shape[0] < batch_size:
            x0 = tf.tile(x0, [batch_size, 1])

        class LinOp(tf.linalg.LinearOperator):
            def __init__(self):
                tf.linalg.LinearOperator.__init__(self, y.dtype, graph_parents=None, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True, is_square=True)

            def _matmul(self, x, adjoint=False, adjoint_arg=False):
                if callable(A):
                    return A(x)
                else:
                    x = tf.reshape(x, x0.shape)
                    result = backend.matmul(A, x)
                    return tf.expand_dims(result, -1)

            def _shape(self):
                return y.shape

            def _shape_tensor(self):
                return y.shape

        result = tf.linalg.experimental.conjugate_gradient(LinOp(), y, preconditioner=None, x=x0, tol=absolute_tolerance + relative_tolerance, max_iter=max_iterations)
        converged = result.i < max_iterations
        iterations = result.i
        return converged, result.x, iterations

    def add(self, a, b):
        if isinstance(a, tf.SparseTensor) or isinstance(b, tf.SparseTensor):
            return tf.sparse.add(a, b, threshold=1e-5)
        else:
            return Backend.add(self, a, b)


TF_BACKEND = TFBackend()


def sparse_select_indices(sp_input, indices, axis=0, are_indices_uniqua=False, are_indices_sorted=False):
    if not are_indices_uniqua:
        indices, _ = tf.unique(indices)
    n_indices = tf.size(indices)
    # Only necessary if indices may not be sorted
    if not are_indices_sorted:
        indices, _ = tf.math.top_k(indices, n_indices)
        indices = tf.reverse(indices, [0])
    # Get indices for the axis
    idx = sp_input.indices[:, axis]
    # Find where indices match the selection
    eq = tf.equal(tf.expand_dims(idx, 1), tf.cast(indices, tf.int64))
    # Mask for selected values
    sel = tf.reduce_any(eq, axis=1)
    # Selected values
    values_new = tf.boolean_mask(sp_input.values, sel, axis=0)
    # New index value for selected elements
    n_indices = tf.cast(n_indices, tf.int64)
    idx_new = tf.reduce_sum(tf.cast(eq, tf.int64) * tf.range(n_indices), axis=1)
    idx_new = tf.boolean_mask(idx_new, sel, axis=0)
    # New full indices tensor
    indices_new = tf.boolean_mask(sp_input.indices, sel, axis=0)
    indices_new = tf.concat([indices_new[:, :axis], tf.expand_dims(idx_new, 1), indices_new[:, axis + 1:]], axis=1)
    # New shape
    shape_new = tf.concat([sp_input.dense_shape[:axis], [n_indices], sp_input.dense_shape[axis + 1:]], axis=0)
    return tf.SparseTensor(indices_new, values_new, shape_new)

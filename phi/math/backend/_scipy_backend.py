import numbers
import warnings

import numpy as np
import scipy.signal
import scipy.sparse
from scipy.sparse.linalg import cg, LinearOperator

from . import Backend
from ._backend_helper import combined_dim
from ._dtype import from_numpy_dtype, to_numpy_dtype, DType


class SciPyBackend(Backend):

    """
    Core Python Backend using NumPy & SciPy
    """

    def __init__(self, precision=32):
        Backend.__init__(self, "SciPy", precision=precision)

    @property
    def precision_dtype(self):
        return to_numpy_dtype(self.float_type)

    # --- Abstract math functions ---

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            array = x
        else:
            array = np.array(x)
        # --- Enforce Precision ---
        if not isinstance(array, numbers.Number):
            if array.dtype in (np.float16, np.float32, np.float64, np.longdouble) and self.has_fixed_precision:
                array = self.to_float(array)
        return array

    def is_tensor(self, x, only_native=False):
        if isinstance(x, np.ndarray) and x.dtype != np.object:
            return True
        if scipy.sparse.issparse(x):
            return True
        if isinstance(x, np.bool_):
            return True
        # --- Above considered native ---
        if only_native:
            return False
        # --- Non-native types
        if isinstance(x, (numbers.Number, bool)):
            return True
        if isinstance(x, (tuple, list)):
            return all([self.is_tensor(item, False) for item in x])
        return False

    def numpy(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)

    def copy(self, tensor, only_mutable=False):
        return np.copy(tensor)

    def transpose(self, tensor, axes):
        return np.transpose(tensor, axes)

    def equal(self, x, y):
        """ array equality comparison """
        return np.equal(x, y)

    def divide_no_nan(self, x, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = x / y
        return np.where(y == 0, 0, result)

    def random_uniform(self, shape):
        """ random array [0.0, 1.0) """
        return np.random.random(shape).astype(self.precision_dtype)

    def random_normal(self, shape):
        return np.random.standard_normal(shape).astype(self.precision_dtype)

    def range(self, start, limit=None, delta=1, dtype=None):
        """ range syntax to arange syntax """
        if limit is None:
            start, limit = 0, start
        return np.arange(start, limit, delta, dtype)

    def tile(self, value, multiples):
        return np.tile(value, multiples)

    def stack(self, values, axis=0):
        return np.stack(values, axis)

    def concat(self, values, axis):
        return np.concatenate(values, axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        assert mode in ('constant', 'symmetric', 'periodic', 'reflect', 'boundary'), mode
        if mode == 'constant':
            return np.pad(value, pad_width, 'constant', constant_values=constant_values)
        else:
            if mode in ('periodic', 'boundary'):
                mode = {'periodic': 'wrap', 'boundary': 'edge'}[mode]
            return np.pad(value, pad_width, mode)

    def reshape(self, value, shape):
        return np.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if value.dtype == bool:
            return np.all(value, axis=axis)
        return np.prod(value, axis=axis)

    def where(self, condition, x=None, y=None):
        if x is None or y is None:
            return np.argwhere(condition)
        return np.where(condition, x, y)

    def nonzero(self, values):
        return np.argwhere(values)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        result = func(*inputs)
        assert result.dtype == Tout, "returned value has wrong type: {}, expected {}".format(result.dtype, Tout)
        assert result.shape == shape_out, "returned value has wrong shape: {}, expected {}".format(result.shape, shape_out)
        return result

    def zeros(self, shape, dtype: DType = None):
        return np.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def zeros_like(self, tensor):
        return np.zeros_like(tensor)

    def ones(self, shape, dtype: DType = None):
        return np.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def ones_like(self, tensor):
        return np.ones_like(tensor)

    def meshgrid(self, *coordinates):
        return np.meshgrid(*coordinates, indexing='ij')

    def mean(self, value, axis=None, keepdims=False):
        return np.mean(value, axis, keepdims=keepdims)

    def dot(self, a, b, axes):
        return np.tensordot(a, b, axes)

    def mul(self, a, b):
        if scipy.sparse.issparse(a):
            return a.multiply(b)
        elif scipy.sparse.issparse(b):
            return b.multiply(a)
        else:
            return a * b

    def matmul(self, A, b):
        return np.stack([A.dot(b[i]) for i in range(b.shape[0])])

    def einsum(self, equation, *tensors):
        return np.einsum(equation, *tensors)

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True,
                   swap_memory=False, name=None, maximum_iterations=None):
        i = 0
        while cond(*loop_vars):
            if maximum_iterations is not None and i == maximum_iterations:
                break
            loop_vars = body(*loop_vars)
            i += 1
        return loop_vars

    def abs(self, x):
        return np.abs(x)

    def sign(self, x):
        return np.sign(x)

    def round(self, x):
        return np.round(x)

    def ceil(self, x):
        return np.ceil(x)

    def floor(self, x):
        return np.floor(x)

    def max(self, x, axis=None, keepdims=False):
        return np.max(x, axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return np.min(x, axis, keepdims=keepdims)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base="custom_gradient_func"):
        return function(*inputs)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def clip(self, x, minimum, maximum):
        return np.clip(x, minimum, maximum)

    def sqrt(self, x):
        return np.sqrt(x)

    def exp(self, x):
        return np.exp(x)

    def conv(self, tensor, kernel, padding="SAME"):
        """ apply convolution of kernel on tensor """
        assert tensor.shape[-1] == kernel.shape[-2]
        # kernel = kernel[[slice(None)] + [slice(None, None, -1)] + [slice(None)]*(len(kernel.shape)-3) + [slice(None)]]
        if padding.lower() == "same":
            result = np.zeros(tensor.shape[:-1] + (kernel.shape[-1],), dtype=self.precision_dtype)
        elif padding.lower() == "valid":
            valid = [tensor.shape[i + 1] - (kernel.shape[i] + 1) // 2 for i in range(tensor_spatial_rank(tensor))]
            result = np.zeros([tensor.shape[0]] + valid + [kernel.shape[-1]], dtype=self.precision_dtype)
        else:
            raise ValueError("Illegal padding: %s" % padding)
        for batch in range(tensor.shape[0]):
            for o in range(kernel.shape[-1]):
                for i in range(tensor.shape[-1]):
                    result[batch, ..., o] += scipy.signal.correlate(tensor[batch, ..., i], kernel[..., i, o], padding.lower())
        return result

    def expand_dims(self, a, axis=0, number=1):
        for _i in range(number):
            a = np.expand_dims(a, axis)
        return a

    def shape(self, tensor):
        return np.shape(tensor)

    def staticshape(self, tensor):
        return np.shape(tensor)

    def to_float(self, x, float64=False):
        if float64:
            warnings.warn('float64 argument is deprecated, set Backend.precision = 64 to use 64 bit operations.', DeprecationWarning)
            return np.array(x).astype(np.float64)
        else:
            return np.array(x).astype(self.precision_dtype)

    def to_int(self, x, int64=False):
        return np.array(x).astype(np.int64 if int64 else np.int32)

    def to_complex(self, x):
        x = self.as_tensor(x)
        if x.dtype in (np.complex64, np.complex128):
            return x
        elif x.dtype == np.float64:
            return x.astype(np.complex128)
        else:
            return x.astype(np.complex64)

    def cast(self, x, dtype: DType):
        if self.is_tensor(x, only_native=True) and from_numpy_dtype(x.dtype) == dtype:
            return x
        else:
            return np.array(x, to_numpy_dtype(dtype))

    def auto_cast(self, *tensors):
        return tensors

    def gather(self, values, indices):
        if scipy.sparse.issparse(values):
            if scipy.sparse.isspmatrix_coo(values):
                values = values.tocsc()
        return values[indices]

    def gather_nd(self, values, indices, batch_dims=0):
        assert indices.shape[-1] == self.ndims(values) - batch_dims - 1
        if batch_dims == 0:
            indices_list = self.unstack(indices, axis=-1)
            result = values[indices_list]
            return result
        for dim in range(batch_dims):
            assert indices.shape[dim] == values.shape[dim] or values.shape[dim] == 1 or indices.shape[dim] == 1, 'Batch dimension %d does not match: %s (values) and %s (indices)' % (dim, values.shape, indices.shape)
        values_batch_max = np.array(values.shape[:batch_dims]) - 1
        indices_batch_max = np.array(indices.shape[:batch_dims]) - 1

        def inner_gather_nd(*pos):
            batch_idx_values = tuple([np.minimum(pos[i], values_batch_max[i]) for i in range(len(pos))])
            values_batch = values[batch_idx_values]
            batch_idx_indices = tuple([np.minimum(pos[i], indices_batch_max[i]) for i in range(len(pos))])
            indices_batch = indices[batch_idx_indices]
            result = values_batch[self.unstack(indices_batch, axis=-1)]
            return result
        # --- Iterate over batch dimensions ---
        batch_pos = np.meshgrid(*[range(dim) for dim in indices.shape[:batch_dims]], indexing='ij')
        batch_pos = np.stack(batch_pos, axis=-1).reshape([-1, batch_dims])
        result = np.empty(indices.shape[:-1] + (values.shape[-1],), values.dtype)
        for i in range(batch_pos.shape[0]):
            gathered = inner_gather_nd(*batch_pos[i])
            result[tuple(batch_pos[i])] = gathered
        return result

    def std(self, x, axis=None, keepdims=False):
        return np.std(x, axis, keepdims=keepdims)

    def boolean_mask(self, x, mask):
        return x[mask]

    def isfinite(self, x):
        return np.isfinite(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return np.any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return np.all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, indices, values, shape, duplicates_handling='undefined', outside_handling='undefined'):
        assert duplicates_handling in ('undefined', 'add', 'mean', 'any')
        assert outside_handling in ('discard', 'clamp', 'undefined')
        shape = np.array(shape, np.int32)
        if outside_handling == 'clamp':
            indices = np.maximum(0, np.minimum(indices, shape - 1))
        elif outside_handling == 'discard':
            indices_inside = (indices >= 0) & (indices < shape)
            indices_inside = np.min(indices_inside, axis=-1)
            filter_indices = np.argwhere(indices_inside)
            indices = indices[filter_indices][..., 0, :]
            if values.shape[0] > 1:
                values = values[filter_indices.reshape(-1)]
        array = np.zeros(tuple(shape) + values.shape[indices.ndim-1:], self.precision_dtype if self.has_fixed_precision else values.dtype)
        indices = self.unstack(indices, axis=-1)
        if duplicates_handling == 'add':
            np.add.at(array, tuple(indices), values)
        elif duplicates_handling == 'mean':
            count = np.zeros(shape, np.int32)
            np.add.at(array, tuple(indices), values)
            np.add.at(count, tuple(indices), 1)
            count = np.maximum(1, count)
            return array / count
        else:  # last, any, undefined
            array[indices] = values
        return array

    def fft(self, x):
        rank = len(x.shape) - 2
        assert rank >= 1
        if rank == 1:
            return np.fft.fft(x, axis=1)
        elif rank == 2:
            return np.fft.fft2(x, axes=[1, 2])
        else:
            return np.fft.fftn(x, axes=list(range(1, rank + 1)))

    def ifft(self, k):
        rank = len(k.shape) - 2
        assert rank >= 1
        if rank == 1:
            return np.fft.ifft(k, axis=1)
        elif rank == 2:
            return np.fft.ifft2(k, axes=[1, 2])
        else:
            return np.fft.ifftn(k, axes=list(range(1, rank + 1)))

    def imag(self, complex_arr):
        return np.imag(complex_arr)

    def real(self, complex_arr):
        return np.real(complex_arr)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def dtype(self, array) -> DType:
        if isinstance(array, int):
            return DType(int, 32)
        if isinstance(array, float):
            return self.float_type
        if isinstance(array, complex):
            return self.complex_type
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        return from_numpy_dtype(array.dtype)

    def sparse_tensor(self, indices, values, shape):
        if not isinstance(indices, (tuple, list)):
            indices = self.unstack(indices, -1)
        if len(indices) == 2:
            return scipy.sparse.csc_matrix((values, indices), shape=shape)
        else:
            raise NotImplementedError(f"len(indices) = {len(indices)} not supported. Only (2) allowed.")

    def coordinates(self, tensor, unstack_coordinates=False):
        if scipy.sparse.issparse(tensor):
            coo = tensor.tocoo()
            return (coo.row, coo.col), coo.data
        else:
            raise NotImplementedError("Only sparse tensors supported.")

    def conjugate_gradient(self, A, y, x0, relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, gradient: str = 'implicit', callback=None):
        bs_y = self.staticshape(y)[0]
        bs_x0 = self.staticshape(x0)[0]
        batch_size = combined_dim(bs_y, bs_x0)

        if callable(A):
            A = LinearOperator(dtype=y.dtype, shape=(self.staticshape(y)[-1], self.staticshape(x0)[-1]), matvec=A)
        elif isinstance(A, (tuple, list)) or self.ndims(A) == 3:
            batch_size = combined_dim(batch_size, self.staticshape(A)[0])

        iterations = [0] * batch_size
        converged = []
        results = []

        def count_callback(*args):
            iterations[batch] += 1
            if callback is not None:
                callback(*args)

        for batch in range(batch_size):
            y_ = y[min(batch, bs_y - 1)]
            x0_ = x0[min(batch, bs_x0 - 1)]
            x, ret_val = cg(A, y_, x0_, tol=relative_tolerance, atol=absolute_tolerance, maxiter=max_iterations, callback=count_callback)
            converged.append(ret_val == 0)
            results.append(x)
        return np.array(converged), self.stack(results), np.array(iterations)


def clamp(coordinates, shape):
    assert coordinates.shape[-1] == len(shape)
    for i in range(len(shape)):
        coordinates[...,i] = np.maximum(0, np.minimum(shape[i] - 1, coordinates[..., i]))
    return coordinates


def tensor_spatial_rank(field):
    dims = len(field.shape) - 2
    assert dims > 0, "channel has no spatial dimensions"
    return dims


SCIPY_BACKEND = SciPyBackend()

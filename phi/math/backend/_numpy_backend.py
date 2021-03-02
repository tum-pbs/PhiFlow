import numbers
import os
import sys
import warnings
from typing import List

import numpy as np
import scipy.signal
import scipy.sparse
from scipy.sparse.linalg import cg, LinearOperator

from . import Backend, ComputeDevice
from ._backend_helper import combined_dim
from ._dtype import from_numpy_dtype, to_numpy_dtype, DType
from ._optim import Solve, LinearSolve, SolveResult


class NumPyBackend(Backend):
    """Core Python Backend using NumPy & SciPy"""

    def __init__(self):
        if sys.platform != "win32" and sys.platform != "darwin":
            mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        else:
            mem_bytes = -1
        processors = os.cpu_count()
        self.cpu = ComputeDevice(self, "CPU", 'CPU', mem_bytes, processors, "")
        Backend.__init__(self, "NumPy", self.cpu)

    def list_devices(self, device_type: str or None = None) -> List[ComputeDevice]:
        return [self.cpu]

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            array = x
        else:
            array = np.array(x)
        # --- Enforce Precision ---
        if not isinstance(array, numbers.Number):
            if array.dtype in (np.float16, np.float32, np.float64, np.longdouble):
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
        if isinstance(x, (numbers.Number, bool, str)):
            return True
        if isinstance(x, (tuple, list)):
            return all([self.is_tensor(item, False) for item in x])
        return False

    def is_available(self, tensor):
        return True

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
        if isinstance(x, np.ndarray) and x.dtype.char == 'U':  # string comparison
            x = x.astype(np.object)
        if isinstance(x, str):
            x = np.array(x, np.object)
        return np.equal(x, y)

    def divide_no_nan(self, x, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = x / y
        return np.where(y == 0, 0, result)

    def random_uniform(self, shape):
        return np.random.random(shape).astype(to_numpy_dtype(self.float_type))

    def random_normal(self, shape):
        return np.random.standard_normal(shape).astype(to_numpy_dtype(self.float_type))

    def range(self, start, limit=None, delta=1, dtype=None):
        """
        range syntax to arange syntax

        Args:
          start: 
          limit:  (Default value = None)
          delta:  (Default value = 1)
          dtype:  (Default value = None)

        Returns:

        """
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

    def linspace(self, start, stop, number):
        return np.linspace(start, stop, number, dtype=to_numpy_dtype(self.float_type))

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
            return Backend.mul(self, a, b)

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
        """
        apply convolution of kernel on tensor

        Args:
          tensor: 
          kernel: 
          padding:  (Default value = "SAME")

        Returns:

        """
        assert tensor.shape[-1] == kernel.shape[-2]
        # kernel = kernel[[slice(None)] + [slice(None, None, -1)] + [slice(None)]*(len(kernel.shape)-3) + [slice(None)]]
        if padding.lower() == "same":
            result = np.zeros(tensor.shape[:-1] + (kernel.shape[-1],), dtype=to_numpy_dtype(self.float_type))
        elif padding.lower() == "valid":
            valid = [tensor.shape[i + 1] - (kernel.shape[i] + 1) // 2 for i in range(tensor_spatial_rank(tensor))]
            result = np.zeros([tensor.shape[0]] + valid + [kernel.shape[-1]], dtype=to_numpy_dtype(self.float_type))
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

    def cast(self, x, dtype: DType):
        if self.is_tensor(x, only_native=True) and from_numpy_dtype(x.dtype) == dtype:
            return x
        else:
            return np.array(x, to_numpy_dtype(dtype))

    def gather(self, values, indices):
        if scipy.sparse.issparse(values):
            if scipy.sparse.isspmatrix_coo(values):
                values = values.tocsc()
        return values[indices]

    def batched_gather_nd(self, values, indices):
        assert indices.shape[-1] == self.ndims(values) - 2
        batch_size = combined_dim(values.shape[0], indices.shape[0])
        result = np.empty((batch_size, *indices.shape[1:-1], values.shape[-1],), values.dtype)
        for b in range(batch_size):
            b_values = values[min(b, values.shape[0] - 1)]
            b_indices = self.unstack(indices[min(b, indices.shape[0] - 1)], -1)
            result[b] = b_values[b_indices]
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
        array = np.zeros(tuple(shape) + values.shape[indices.ndim-1:], to_numpy_dtype(self.float_type))
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
        assert self.dtype(k).kind == complex
        rank = len(k.shape) - 2
        assert rank >= 1
        if rank == 1:
            return np.fft.ifft(k, axis=1).astype(k.dtype)
        elif rank == 2:
            return np.fft.ifft2(k, axes=[1, 2]).astype(k.dtype)
        else:
            return np.fft.ifftn(k, axes=list(range(1, rank + 1))).astype(k.dtype)

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
            return DType(float, 64)
        if isinstance(array, complex):
            return DType(complex, 128)
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

    def conjugate_gradient(self, A, y, x0, solve_params=LinearSolve(), callback=None):
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
            x, ret_val = cg(A, y_, x0_, tol=solve_params.relative_tolerance, atol=solve_params.absolute_tolerance, maxiter=solve_params.max_iterations, callback=count_callback)
            converged.append(ret_val == 0)
            results.append(x)
        solve_params.result = SolveResult(all(converged), max(iterations))
        return self.stack(results)


def clamp(coordinates, shape):
    assert coordinates.shape[-1] == len(shape)
    for i in range(len(shape)):
        coordinates[...,i] = np.maximum(0, np.minimum(shape[i] - 1, coordinates[..., i]))
    return coordinates


def tensor_spatial_rank(field):
    dims = len(field.shape) - 2
    assert dims > 0, "channel has no spatial dimensions"
    return dims


NUMPY_BACKEND = NumPyBackend()

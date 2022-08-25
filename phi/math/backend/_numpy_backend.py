import numbers
import os
import sys
from typing import List, Any, Callable

import numpy as np
import numpy.random
import scipy.signal
import scipy.sparse
from scipy.sparse import issparse
from scipy.sparse.linalg import cg, spsolve

from . import Backend, ComputeDevice
from ._backend import combined_dim, SolveResult, TensorType
from ._dtype import from_numpy_dtype, to_numpy_dtype, DType


class NumPyBackend(Backend):
    """Core Python Backend using NumPy & SciPy"""

    def __init__(self):
        if sys.platform != "win32" and sys.platform != "darwin":
            mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        else:
            mem_bytes = -1
        processors = os.cpu_count()
        cpu = ComputeDevice(self, "CPU", 'CPU', mem_bytes, processors, "", 'CPU')
        Backend.__init__(self, "NumPy", [cpu], cpu)

    def prefers_channels_last(self) -> bool:
        return True

    seed = np.random.seed
    clip = staticmethod(np.clip)
    minimum = np.minimum
    maximum = np.maximum
    ones_like = staticmethod(np.ones_like)
    zeros_like = staticmethod(np.zeros_like)
    nonzero = staticmethod(np.argwhere)
    reshape = staticmethod(np.reshape)
    concat = staticmethod(np.concatenate)
    stack = staticmethod(np.stack)
    tile = staticmethod(np.tile)
    transpose = staticmethod(np.transpose)
    sqrt = np.sqrt
    exp = np.exp
    sin = np.sin
    arcsin = np.arcsin
    cos = np.cos
    arccos = np.arccos
    tan = np.tan
    log = np.log
    log2 = np.log2
    log10 = np.log10
    isfinite = np.isfinite
    abs = np.abs
    sign = np.sign
    round = staticmethod(np.round)
    ceil = np.ceil
    floor = np.floor
    shape = staticmethod(np.shape)
    staticshape = staticmethod(np.shape)
    imag = staticmethod(np.imag)
    real = staticmethod(np.real)
    conj = staticmethod(np.conjugate)
    einsum = staticmethod(np.einsum)
    cumsum = staticmethod(np.cumsum)

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            array = x
        else:
            array = np.array(x)
        # --- Enforce Precision ---
        if not isinstance(array, numbers.Number):
            if self.dtype(array).kind == float:
                array = self.to_float(array)
            elif self.dtype(array).kind == complex:
                array = self.to_complex(array)
        return array

    def is_module(self, obj):
        return False

    def is_tensor(self, x, only_native=False):
        if isinstance(x, np.ndarray) and x.dtype != object and x.dtype != str:
            return True
        if issparse(x):
            return True
        if isinstance(x, (np.bool_, np.float32, np.float64, np.float16, np.int8, np.int16, np.int32, np.int64, np.complex128, np.complex64)):
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

    def is_available(self, tensor):
        return True

    def numpy(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)

    def copy(self, tensor, only_mutable=False):
        return np.copy(tensor)

    def get_device(self, tensor) -> ComputeDevice:
        return self._default_device

    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        assert device == self._default_device, f"NumPy Can only allocate on the CPU but got device {device}"
        return tensor

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

    def random_uniform(self, shape, low, high, dtype: DType or None):
        dtype = dtype or self.float_type
        if dtype.kind == float:
            return np.random.uniform(low, high, shape).astype(to_numpy_dtype(dtype))
        elif dtype.kind == complex:
            return (np.random.uniform(low.real, high.real, shape) + 1j * np.random.uniform(low.imag, high.imag, shape)).astype(to_numpy_dtype(dtype))
        elif dtype.kind == int:
            return numpy.random.randint(low, high, shape, dtype=to_numpy_dtype(dtype))
        else:
            raise ValueError(dtype)

    def random_normal(self, shape, dtype: DType):
        dtype = dtype or self.float_type
        return np.random.standard_normal(shape).astype(to_numpy_dtype(dtype))

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
        if limit is None:
            start, limit = 0, start
        return np.arange(start, limit, delta, to_numpy_dtype(dtype))

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        assert mode in ('constant', 'symmetric', 'periodic', 'reflect', 'boundary'), mode
        if mode == 'constant':
            return np.pad(value, pad_width, 'constant', constant_values=constant_values)
        else:
            if mode in ('periodic', 'boundary'):
                mode = {'periodic': 'wrap', 'boundary': 'edge'}[mode]
            return np.pad(value, pad_width, mode)

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

    def zeros(self, shape, dtype: DType = None):
        return np.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def ones(self, shape, dtype: DType = None):
        return np.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def meshgrid(self, *coordinates):
        return np.meshgrid(*coordinates, indexing='ij')

    def linspace(self, start, stop, number):
        return np.linspace(start, stop, number, dtype=to_numpy_dtype(self.float_type))

    def mean(self, value, axis=None, keepdims=False):
        return np.mean(value, axis, keepdims=keepdims)

    def tensordot(self, a, a_axes: tuple or list, b, b_axes: tuple or list):
        return np.tensordot(a, b, (a_axes, b_axes))

    def mul(self, a, b):
        if scipy.sparse.issparse(a):
            return a.multiply(b)
        elif scipy.sparse.issparse(b):
            return b.multiply(a)
        else:
            return Backend.mul(self, a, b)

    def matmul(self, A, b):
        return np.stack([A.dot(b[i]) for i in range(b.shape[0])])

    def while_loop(self, loop: Callable, values: tuple):
        while np.any(values[0]):
            values = loop(*values)
        return values

    def max(self, x, axis=None, keepdims=False):
        return np.max(x, axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return np.min(x, axis, keepdims=keepdims)

    def conv(self, value, kernel, zero_padding=True):
        assert kernel.shape[0] in (1, value.shape[0])
        assert value.shape[1] == kernel.shape[2], f"value has {value.shape[1]} channels but kernel has {kernel.shape[2]}"
        assert value.ndim + 1 == kernel.ndim
        if zero_padding:
            result = np.zeros((value.shape[0], kernel.shape[1], *value.shape[2:]), dtype=to_numpy_dtype(self.float_type))
        else:
            valid = [value.shape[i + 2] - kernel.shape[i + 3] + 1 for i in range(value.ndim - 2)]
            result = np.zeros([value.shape[0], kernel.shape[1], *valid], dtype=to_numpy_dtype(self.float_type))
        mode = 'same' if zero_padding else 'valid'
        for b in range(value.shape[0]):
            b_kernel = kernel[min(b, kernel.shape[0] - 1)]
            for o in range(kernel.shape[1]):
                for i in range(value.shape[1]):
                    result[b, o, ...] += scipy.signal.correlate(value[b, i, ...], b_kernel[o, i, ...], mode=mode)
        return result

    def expand_dims(self, a, axis=0, number=1):
        for _i in range(number):
            a = np.expand_dims(a, axis)
        return a

    def cast(self, x, dtype: DType):
        if self.is_tensor(x, only_native=True) and from_numpy_dtype(x.dtype) == dtype:
            return x
        else:
            return np.array(x, to_numpy_dtype(dtype))

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

    def boolean_mask(self, x, mask, axis=0):
        slices = [mask if i == axis else slice(None) for i in range(len(x.shape))]
        return x[tuple(slices)]

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return np.any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return np.all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, base_grid, indices, values, mode: str):
        assert mode in ('add', 'update')
        assert isinstance(base_grid, np.ndarray)
        assert isinstance(indices, (np.ndarray, tuple))
        assert isinstance(values, np.ndarray)
        assert indices.ndim == 3
        assert values.ndim == 3
        assert base_grid.ndim >= 3
        batch_size = combined_dim(combined_dim(base_grid.shape[0], indices.shape[0]), values.shape[0])
        if base_grid.shape[0] == batch_size:
            result = np.copy(base_grid)
        else:
            result = np.tile(base_grid, (batch_size, *[1] * (base_grid.ndim - 1)))
        if not isinstance(indices, (tuple, list)):
            indices = self.unstack(indices, axis=-1)
        if mode == 'add':
            for b in range(batch_size):
                np.add.at(result, (b, *[i[min(b, i.shape[0]-1)] for i in indices]), values[min(b, values.shape[0]-1)])
        else:  # update
            for b in range(batch_size):
                result[(b, *[i[min(b, i.shape[0]-1)] for i in indices])] = values[min(b, values.shape[0]-1)]
        # elif duplicates_handling == 'mean':
        #     count = np.zeros(shape, np.int32)
        #     np.add.at(array, tuple(indices), values)
        #     np.add.at(count, tuple(indices), 1)
        #     count = np.maximum(1, count)
        #     return array / count
        return result

    def quantile(self, x, quantiles):
        return np.quantile(x, quantiles, axis=-1)

    def fft(self, x, axes: tuple or list):
        x = self.to_complex(x)
        if not axes:
            return x
        if len(axes) == 1:
            return np.fft.fft(x, axis=axes[0]).astype(x.dtype)
        elif len(axes) == 2:
            return np.fft.fft2(x, axes=axes).astype(x.dtype)
        else:
            return np.fft.fftn(x, axes=axes).astype(x.dtype)

    def ifft(self, k, axes: tuple or list):
        if not axes:
            return k
        if len(axes) == 1:
            return np.fft.ifft(k, axis=axes[0]).astype(k.dtype)
        elif len(axes) == 2:
            return np.fft.ifft2(k, axes=axes).astype(k.dtype)
        else:
            return np.fft.ifftn(k, axes=axes).astype(k.dtype)

    def dtype(self, array) -> DType:
        if isinstance(array, int):
            return DType(int, 32)
        if isinstance(array, float):
            return DType(float, 64)
        if isinstance(array, complex):
            return DType(complex, 128)
        if not hasattr(array, 'dtype'):
            array = np.array(array)
        return from_numpy_dtype(array.dtype)

    def sparse_coo_tensor(self, indices, values, shape):
        if not isinstance(indices, (tuple, list)):
            indices = self.unstack(indices, -1)
        if len(shape) == 2:
            return scipy.sparse.coo_matrix((values, indices), shape=shape)
        else:
            raise NotImplementedError(f"len(indices) = {len(indices)} not supported. Only (2) allowed.")

    def csr_matrix(self, column_indices, row_pointers, values, shape: tuple):
        return scipy.sparse.csr_matrix((values, column_indices, row_pointers), shape=shape)

    def csc_matrix(self, column_pointers, row_indices, values, shape: tuple):
        return scipy.sparse.csc_matrix((values, row_indices, column_pointers), shape=shape)

    def coordinates(self, tensor):
        assert scipy.sparse.issparse(tensor)
        coo = tensor.tocoo()
        return (coo.row, coo.col), coo.data

    def stop_gradient(self, value):
        return value

    # def jacobian(self, f, wrt: tuple or list, get_output: bool):
    #     warnings.warn("NumPy does not support analytic gradients and will use differences instead. This may be slow!", RuntimeWarning)
    #     eps = {64: 1e-9, 32: 1e-4, 16: 1e-1}[self.precision]
    #
    #     def gradient(*args, **kwargs):
    #         output = f(*args, **kwargs)
    #         loss = output[0] if isinstance(output, (tuple, list)) else output
    #         grads = []
    #         for wrt_ in wrt:
    #             x = args[wrt_]
    #             assert isinstance(x, np.ndarray)
    #             if x.size > 64:
    #                 raise RuntimeError("NumPy does not support analytic gradients. Use PyTorch, TensorFlow or Jax.")
    #             grad = np.zeros_like(x).flatten()
    #             for i in range(x.size):
    #                 x_flat = x.flatten()  # makes a copy
    #                 x_flat[i] += eps
    #                 args_perturbed = list(args)
    #                 args_perturbed[wrt_] = np.reshape(x_flat, x.shape)
    #                 output_perturbed = f(*args_perturbed, **kwargs)
    #                 loss_perturbed = output_perturbed[0] if isinstance(output, (tuple, list)) else output_perturbed
    #                 grad[i] = (loss_perturbed - loss) / eps
    #             grads.append(np.reshape(grad, x.shape))
    #         if get_output:
    #             return output, grads
    #         else:
    #             return grads
    #     return gradient

    def linear_solve(self, method: str, lin, y, x0, rtol, atol, max_iter, trj: bool) -> Any:
        if method == 'direct':
            return self.direct_linear_solve(lin, y)
        elif method == 'CG-native':
            return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.cg)
        elif method == 'GMres':
            return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.gmres)
        elif method == 'biCG':
            return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.bicg)
        elif method == 'CGS':
            return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.cgs)
        elif method == 'lGMres':
            return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.lgmres)
        # elif method == 'minres':
        #     return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.minres)
        elif method == 'QMR':
            return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.qmr)
        elif method == 'GCrotMK':
            return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.gcrotmk)
        elif method == 'auto':
            return self.conjugate_gradient_adaptive(lin, y, x0, rtol, atol, max_iter, trj)
            # return self.conjugate_gradient(lin, y, x0, rtol, atol, max_iter, trj)
        else:
            return Backend.linear_solve(self, method, lin, y, x0, rtol, atol, max_iter, trj)

    def direct_linear_solve(self, lin, y) -> Any:
        batch_size = self.staticshape(y)[0]
        xs = []
        converged = []
        if isinstance(lin, (tuple, list)):
            assert all(issparse(l) for l in lin)
        else:
            assert issparse(lin)
            lin = [lin] * batch_size
        # Solve each example independently
        for batch in range(batch_size):
            # use_umfpack=self.precision == 64
            x = spsolve(lin[batch], y[batch])  # returns nan when diverges
            xs.append(x)
            converged.append(np.all(np.isfinite(x)))
        x = np.stack(xs)
        converged = np.stack(converged)
        diverged = ~converged
        iterations = [-1] * batch_size  # spsolve does not perform iterations
        return SolveResult('scipy.sparse.linalg.spsolve', x, None, iterations, iterations, converged, diverged, "")

    def conjugate_gradient(self, lin, y, x0, rtol, atol, max_iter, trj: bool) -> Any:
        if trj or callable(lin):
            return Backend.conjugate_gradient(self, lin, y, x0, rtol, atol, max_iter, trj)  # generic implementation
        else:
            return self.scipy_iterative_sparse_solve(lin, y, x0, rtol, atol, max_iter, scipy_function=scipy.sparse.linalg.bicg)  # more stable than cg

    def scipy_iterative_sparse_solve(self, lin, y, x0, rtol, atol, max_iter, scipy_function=cg) -> Any:
        bs_y = self.staticshape(y)[0]
        bs_x0 = self.staticshape(x0)[0]
        batch_size = combined_dim(bs_y, bs_x0)
        # if callable(A):
        #     A = LinearOperator(dtype=y.dtype, shape=(self.staticshape(y)[-1], self.staticshape(x0)[-1]), matvec=A)
        if isinstance(lin, (tuple, list)):
            assert len(lin) == batch_size
        else:
            lin = [lin] * batch_size

        def count_callback(x_n):  # called after each step, not with x0
            iterations[b] += 1

        xs = []
        iterations = [0] * batch_size
        converged = []
        diverged = []
        for b in range(batch_size):
            x, ret_val = scipy_function(lin[b], y[b], x0=x0[b], tol=rtol[b], atol=atol[b], maxiter=max_iter[b], callback=count_callback)
            # ret_val: 0=success, >0=not converged, <0=error
            xs.append(x)
            converged.append(ret_val == 0)
            diverged.append(ret_val < 0 or np.any(~np.isfinite(x)))
        x = np.stack(xs)
        f_eval = [i + 1 for i in iterations]
        return SolveResult(f'scipy.sparse.linalg.{scipy_function.__name__}', x, None, iterations, f_eval, converged, diverged, "")

    def matrix_solve_least_squares(self, matrix: TensorType, rhs: TensorType) -> TensorType:
        solution, residuals, rank, singular_values = [], [], [], []
        for b in range(self.shape(rhs)[0]):
            solution_b, residual_b, rnk_b, s_b = np.linalg.lstsq(matrix[b], rhs[b], rcond=None)
            solution.append(solution_b)
            residuals.append(residual_b)
            rank.append(rnk_b)
            singular_values.append(s_b)
        return np.stack(solution), np.stack(residuals), np.stack(rank), np.stack(singular_values)

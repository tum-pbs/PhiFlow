import numbers
from functools import wraps
from typing import List, Callable

import numpy as np

from phi.math.backend._optim import SolveResult

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as scipy
    from jax.scipy.sparse.linalg import cg
    from jax import random
except ImportError as err:
    print(err)

from phi.math.backend import Backend, ComputeDevice, to_numpy_dtype, from_numpy_dtype
from phi.math import Solve, LinearSolve, DType
from phi.math.backend._backend_helper import combined_dim


class JaxBackend(Backend):

    def __init__(self):
        Backend.__init__(self, "SciPy", default_device=None)
        try:
            self.rnd_key = jax.random.PRNGKey(seed=0)
        except NameError:  # Jax not imported
            self.rnd_key = None
        except RuntimeError:
            self.rnd_key = None

    def list_devices(self, device_type: str or None = None) -> List[ComputeDevice]:
        jax_devices = jax.devices()
        devices = []
        for jax_dev in jax_devices:
            jax_dev_type = jax_dev.platform.upper()
            if device_type is None or device_type == jax_dev_type:
                description = f"id={jax_dev.id}"
                devices.append(ComputeDevice(jax_dev.device_kind, jax_dev_type, -1, -1, description))
        return devices

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            array = x
        else:
            array = jnp.array(x)
        # --- Enforce Precision ---
        if not isinstance(array, numbers.Number):
            if array.dtype in (jnp.float16, jnp.float32, jnp.float64):
                array = self.to_float(array)
        return array

    def is_tensor(self, x, only_native=False):
        if isinstance(x, jnp.ndarray) and not isinstance(x, np.ndarray):  # NumPy arrays inherit from Jax arrays
            return True
        # if scipy.sparse.issparse(x):  # TODO
        #     return True
        if isinstance(x, jnp.bool_):
            return True
        # --- Above considered native ---
        if only_native:
            return False
        # --- Non-native types ---
        if isinstance(x, np.ndarray):
            return True
        if isinstance(x, (numbers.Number, bool, str)):
            return True
        if isinstance(x, (tuple, list)):
            return all([self.is_tensor(item, False) for item in x])
        return False

    def is_available(self, tensor):
        return True

    def numpy(self, tensor):
        if isinstance(tensor, jnp.ndarray):
            return tensor
        else:
            return jnp.array(tensor)

    def copy(self, tensor, only_mutable=False):
        return jnp.array(tensor, copy=True)

    def trace_function(self, f: Callable) -> Callable:
        return jax.jit(f)

    def gradient_function(self, f, wrt: tuple or list, get_output: bool):
        if get_output:
            @wraps(f)
            def aux_f(*args):
                result = f(*args)
                return (result[0], result[1:]) if isinstance(result, (tuple, list)) else (result[0], None)
            jax_grad_f = jax.value_and_grad(aux_f, argnums=wrt, has_aux=True)
            @wraps(f)
            def unwrap_outputs(*args):
                (loss, aux), grads = jax_grad_f(*args)
                return (loss, *aux, *grads)
            return unwrap_outputs
        else:
            @wraps(f)
            def nonaux_f(*args):
                result = f(*args)
                return result[0] if isinstance(result, (tuple, list)) else result
            return jax.grad(nonaux_f, argnums=wrt, has_aux=False)

    def custom_gradient(self, f: Callable, gradient: Callable) -> Callable:
        jax_fun = jax.custom_jvp(f)
        @jax_fun.defjvp
        def jax_grad(primals, tangents):
            grad = gradient(*tangents)
            return jax_fun(primals), grad
        return jax_fun

    def transpose(self, tensor, axes):
        return jnp.transpose(tensor, axes)

    def equal(self, x, y):
        return jnp.equal(x, y)

    def divide_no_nan(self, x, y):
        return jnp.nan_to_num(x / y, copy=True, nan=0)

    def random_uniform(self, shape):
        self.rnd_key, subkey = jax.random.split(self.rnd_key)
        return random.uniform(subkey, shape, dtype=to_numpy_dtype(self.float_type))

    def random_normal(self, shape):
        self.rnd_key, subkey = jax.random.split(self.rnd_key)
        return random.normal(subkey, shape, dtype=to_numpy_dtype(self.float_type))

    def range(self, start, limit=None, delta=1, dtype=None):
        if limit is None:
            start, limit = 0, start
        return jnp.arange(start, limit, delta, dtype)

    def tile(self, value, multiples):
        return jnp.tile(value, multiples)

    def stack(self, values, axis=0):
        return jnp.stack(values, axis)

    def concat(self, values, axis):
        return jnp.concatenate(values, axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        assert mode in ('constant', 'symmetric', 'periodic', 'reflect', 'boundary'), mode
        if mode == 'constant':
            return jnp.pad(value, pad_width, 'constant', constant_values=constant_values)
        else:
            if mode in ('periodic', 'boundary'):
                mode = {'periodic': 'wrap', 'boundary': 'edge'}[mode]
            return jnp.pad(value, pad_width, mode)

    def reshape(self, value, shape):
        return jnp.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        if isinstance(value, (tuple, list)):
            assert axis == 0
            return sum(value[1:], value[0])
        return jnp.sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        if not isinstance(value, jnp.ndarray):
            value = jnp.array(value)
        if value.dtype == bool:
            return jnp.all(value, axis=axis)
        return jnp.prod(value, axis=axis)

    def where(self, condition, x=None, y=None):
        if x is None or y is None:
            return jnp.argwhere(condition)
        return jnp.where(condition, x, y)

    def nonzero(self, values):
        return jnp.argwhere(values)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        result = func(*inputs)
        assert result.dtype == Tout, "returned value has wrong type: {}, expected {}".format(result.dtype, Tout)
        assert result.shape == shape_out, "returned value has wrong shape: {}, expected {}".format(result.shape, shape_out)
        return result

    def zeros(self, shape, dtype: DType = None):
        return jnp.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def zeros_like(self, tensor):
        return jnp.zeros_like(tensor)

    def ones(self, shape, dtype: DType = None):
        return jnp.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def ones_like(self, tensor):
        return jnp.ones_like(tensor)

    def meshgrid(self, *coordinates):
        return jnp.meshgrid(*coordinates, indexing='ij')

    def linspace(self, start, stop, number):
        return jnp.linspace(start, stop, number, dtype=to_numpy_dtype(self.float_type))

    def mean(self, value, axis=None, keepdims=False):
        return jnp.mean(value, axis, keepdims=keepdims)

    def dot(self, a, b, axes):
        return jnp.tensordot(a, b, axes)

    def mul(self, a, b):
        # if scipy.sparse.issparse(a):  # TODO sparse?
        #     return a.multiply(b)
        # elif scipy.sparse.issparse(b):
        #     return b.multiply(a)
        # else:
            return Backend.mul(self, a, b)

    def matmul(self, A, b):
        return jnp.stack([A.dot(b[i]) for i in range(b.shape[0])])

    def einsum(self, equation, *tensors):
        return jnp.einsum(equation, *tensors)

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
        return jnp.abs(x)

    def sign(self, x):
        return jnp.sign(x)

    def round(self, x):
        return jnp.round(x)

    def ceil(self, x):
        return jnp.ceil(x)

    def floor(self, x):
        return jnp.floor(x)

    def max(self, x, axis=None, keepdims=False):
        return jnp.max(x, axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return jnp.min(x, axis, keepdims=keepdims)

    def maximum(self, a, b):
        return jnp.maximum(a, b)

    def minimum(self, a, b):
        return jnp.minimum(a, b)

    def clip(self, x, minimum, maximum):
        return jnp.clip(x, minimum, maximum)

    def sqrt(self, x):
        return jnp.sqrt(x)

    def exp(self, x):
        return jnp.exp(x)

    def conv(self, tensor, kernel, padding="SAME"):
        assert tensor.shape[-1] == kernel.shape[-2]
        # kernel = kernel[[slice(None)] + [slice(None, None, -1)] + [slice(None)]*(len(kernel.shape)-3) + [slice(None)]]
        if padding.lower() == "same":
            result = jnp.zeros(tensor.shape[:-1] + (kernel.shape[-1],), dtype=to_numpy_dtype(self.float_type))
        elif padding.lower() == "valid":
            valid = [tensor.shape[i + 1] - (kernel.shape[i] + 1) // 2 for i in range(tensor_spatial_rank(tensor))]
            result = jnp.zeros([tensor.shape[0]] + valid + [kernel.shape[-1]], dtype=to_numpy_dtype(self.float_type))
        else:
            raise ValueError("Illegal padding: %s" % padding)
        for batch in range(tensor.shape[0]):
            for o in range(kernel.shape[-1]):
                for i in range(tensor.shape[-1]):
                    result[batch, ..., o] += scipy.signal.correlate(tensor[batch, ..., i], kernel[..., i, o], padding.lower())
        return result

    def expand_dims(self, a, axis=0, number=1):
        for _i in range(number):
            a = jnp.expand_dims(a, axis)
        return a

    def shape(self, tensor):
        return jnp.shape(tensor)

    def staticshape(self, tensor):
        return jnp.shape(tensor)

    def cast(self, x, dtype: DType):
        if self.is_tensor(x, only_native=True) and from_numpy_dtype(x.dtype) == dtype:
            return x
        else:
            return jnp.array(x, to_numpy_dtype(dtype))

    def gather(self, values, indices):
        # if scipy.sparse.issparse(values):  # TODO no sparse matrices?
        #     if scipy.sparse.isspmatrix_coo(values):
        #         values = values.tocsc()
        return values[indices]

    def gather_nd(self, values, indices, batch_dims=0):
        assert indices.shape[-1] == self.ndims(values) - batch_dims - 1
        if batch_dims == 0:
            indices_list = self.unstack(indices, axis=-1)
            result = values[indices_list]
            return result
        for dim in range(batch_dims):
            assert indices.shape[dim] == values.shape[dim] or values.shape[dim] == 1 or indices.shape[dim] == 1, 'Batch dimension %d does not match: %s (values) and %s (indices)' % (dim, values.shape, indices.shape)
        values_batch_max = jnp.array(values.shape[:batch_dims]) - 1
        indices_batch_max = jnp.array(indices.shape[:batch_dims]) - 1

        def inner_gather_nd(*pos):
            batch_idx_values = tuple([jnp.minimum(pos[i], values_batch_max[i]) for i in range(len(pos))])
            values_batch = values[batch_idx_values]
            batch_idx_indices = tuple([jnp.minimum(pos[i], indices_batch_max[i]) for i in range(len(pos))])
            indices_batch = indices[batch_idx_indices]
            result = values_batch[self.unstack(indices_batch, axis=-1)]
            return result
        # --- Iterate over batch dimensions ---
        batch_pos = jnp.meshgrid(*[range(dim) for dim in indices.shape[:batch_dims]], indexing='ij')
        batch_pos = jnp.stack(batch_pos, axis=-1).reshape([-1, batch_dims])
        result = jnp.empty(indices.shape[:-1] + (values.shape[-1],), values.dtype)
        for i in range(batch_pos.shape[0]):
            gathered = inner_gather_nd(*batch_pos[i])
            result[tuple(batch_pos[i])] = gathered
        return result

    def std(self, x, axis=None, keepdims=False):
        return jnp.std(x, axis, keepdims=keepdims)

    def boolean_mask(self, x, mask):
        return x[mask]

    def isfinite(self, x):
        return jnp.isfinite(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return jnp.any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return jnp.all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, indices, values, shape, duplicates_handling='undefined', outside_handling='undefined'):
        assert duplicates_handling in ('undefined', 'add', 'mean', 'any')
        assert outside_handling in ('discard', 'clamp', 'undefined')
        shape = jnp.array(shape, jnp.int32)
        if outside_handling == 'clamp':
            indices = jnp.maximum(0, jnp.minimum(indices, shape - 1))
        elif outside_handling == 'discard':
            indices_inside = (indices >= 0) & (indices < shape)
            indices_inside = jnp.min(indices_inside, axis=-1)
            filter_indices = jnp.argwhere(indices_inside)
            indices = indices[filter_indices][..., 0, :]
            if values.shape[0] > 1:
                values = values[filter_indices.reshape(-1)]
        array = jnp.zeros(tuple(shape) + values.shape[indices.ndim-1:], to_numpy_dtype(self.float_type))
        indices = self.unstack(indices, axis=-1)
        if duplicates_handling == 'add':
            jnp.add.at(array, tuple(indices), values)
        elif duplicates_handling == 'mean':
            count = jnp.zeros(shape, jnp.int32)
            jnp.add.at(array, tuple(indices), values)
            jnp.add.at(count, tuple(indices), 1)
            count = jnp.maximum(1, count)
            return array / count
        else:  # last, any, undefined
            array[indices] = values
        return array

    def fft(self, x):
        rank = len(x.shape) - 2
        assert rank >= 1
        if rank == 1:
            return jnp.fft.fft(x, axis=1)
        elif rank == 2:
            return jnp.fft.fft2(x, axes=[1, 2])
        else:
            return jnp.fft.fftn(x, axes=list(range(1, rank + 1)))

    def ifft(self, k):
        assert self.dtype(k).kind == complex
        rank = len(k.shape) - 2
        assert rank >= 1
        if rank == 1:
            return jnp.fft.ifft(k, axis=1).astype(k.dtype)
        elif rank == 2:
            return jnp.fft.ifft2(k, axes=[1, 2]).astype(k.dtype)
        else:
            return jnp.fft.ifftn(k, axes=list(range(1, rank + 1))).astype(k.dtype)

    def imag(self, complex_arr):
        return jnp.imag(complex_arr)

    def real(self, complex_arr):
        return jnp.real(complex_arr)

    def sin(self, x):
        return jnp.sin(x)

    def cos(self, x):
        return jnp.cos(x)

    def dtype(self, array) -> DType:
        if isinstance(array, int):
            return DType(int, 32)
        if isinstance(array, float):
            return DType(float, 64)
        if isinstance(array, complex):
            return DType(complex, 128)
        if not isinstance(array, jnp.ndarray):
            array = jnp.array(array)
        return from_numpy_dtype(array.dtype)

    def sparse_tensor(self, indices, values, shape):
        raise NotImplementedError()  # TODO
        # if not isinstance(indices, (tuple, list)):
        #     indices = self.unstack(indices, -1)
        # if len(indices) == 2:
        #     return scipy.sparse.csc_matrix((values, indices), shape=shape)
        # else:
        #     raise NotImplementedError(f"len(indices) = {len(indices)} not supported. Only (2) allowed.")

    def coordinates(self, tensor, unstack_coordinates=False):
        raise NotImplementedError()  # TODO
        # if scipy.sparse.issparse(tensor):
        #     coo = tensor.tocoo()
        #     return (coo.row, coo.col), coo.data
        # else:
        #     raise NotImplementedError("Only sparse tensors supported.")

    def conjugate_gradient(self, A, y, x0, solve_params=LinearSolve(), gradient: str = 'implicit', callback=None):
        bs_y = self.staticshape(y)[0]
        bs_x0 = self.staticshape(x0)[0]
        batch_size = combined_dim(bs_y, bs_x0)

        if isinstance(A, (tuple, list)) or self.ndims(A) == 3:
            batch_size = combined_dim(batch_size, self.staticshape(A)[0])

        results = []

        for batch in range(batch_size):
            y_ = y[min(batch, bs_y - 1)]
            x0_ = x0[min(batch, bs_x0 - 1)]
            x, ret_val = cg(A, y_, x0_, tol=solve_params.relative_tolerance, atol=solve_params.absolute_tolerance, maxiter=solve_params.max_iterations)

            results.append(x)
        solve_params.result = SolveResult(success=True, iterations=-1)
        return self.stack(results)


def clamp(coordinates, shape):
    assert coordinates.shape[-1] == len(shape)
    for i in range(len(shape)):
        coordinates[...,i] = jnp.maximum(0, jnp.minimum(shape[i] - 1, coordinates[..., i]))
    return coordinates


def tensor_spatial_rank(field):
    dims = len(field.shape) - 2
    assert dims > 0, "channel has no spatial dimensions"
    return dims


JAX_BACKEND = JaxBackend()

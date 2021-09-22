import numbers
import warnings
from functools import wraps, partial
from typing import List, Callable

import logging
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as scipy
from jax.core import Tracer
from jax.interpreters.xla import DeviceArray
from jax.scipy.sparse.linalg import cg
from jax import random

from phi.math import SolveInfo, Solve, DType
from ..math.backend._dtype import to_numpy_dtype, from_numpy_dtype
from phi.math.backend import Backend, ComputeDevice
from phi.math.backend._backend import combined_dim, SolveResult


class JaxBackend(Backend):

    def __init__(self):
        Backend.__init__(self, "Jax", default_device=None)
        try:
            self.rnd_key = jax.random.PRNGKey(seed=0)
        except RuntimeError as err:
            warnings.warn(f"{err}")
            self.rnd_key = None

    def prefers_channels_last(self) -> bool:
        return True

    def list_devices(self, device_type: str or None = None) -> List[ComputeDevice]:
        devices = []
        for jax_dev in jax.devices():
            jax_dev_type = jax_dev.platform.upper()
            if device_type is None or device_type == jax_dev_type:
                description = f"id={jax_dev.id}"
                devices.append(ComputeDevice(self, jax_dev.device_kind, jax_dev_type, -1, -1, description, jax_dev))
        return devices

    # def set_default_device(self, device: ComputeDevice or str):
    #     if device == 'CPU':
    #         jax.config.update('jax_platform_name', 'cpu')
    #     elif device == 'GPU':
    #         jax.config.update('jax_platform_name', 'gpu')
    #     else:
    #         raise NotImplementedError()

    def _check_float64(self):
        if self.precision == 64:
            if not jax.config.read('jax_enable_x64'):
                jax.config.update('jax_enable_x64', True)
            assert jax.config.read('jax_enable_x64'), "FP64 is disabled for Jax."

    def seed(self, seed: int):
        self.rnd_key = jax.random.PRNGKey(seed)

    def as_tensor(self, x, convert_external=True):
        self._check_float64()
        if self.is_tensor(x, only_native=convert_external):
            array = x
        else:
            array = jnp.array(x)
        # --- Enforce Precision ---
        if not isinstance(array, numbers.Number):
            if self.dtype(array).kind == float:
                array = self.to_float(array)
            elif self.dtype(array).kind == complex:
                array = self.to_complex(array)
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
        return not isinstance(tensor, Tracer)

    def numpy(self, x):
        return np.array(x)

    def to_dlpack(self, tensor):
        from jax import dlpack
        return dlpack.to_dlpack(tensor)

    def from_dlpack(self, capsule):
        from jax import dlpack
        return dlpack.from_dlpack(capsule)

    def copy(self, tensor, only_mutable=False):
        return jnp.array(tensor, copy=True)

    sqrt = staticmethod(jnp.sqrt)
    exp = staticmethod(jnp.exp)
    sin = staticmethod(jnp.sin)
    cos = staticmethod(jnp.cos)
    tan = staticmethod(jnp.tan)
    log = staticmethod(jnp.log)
    log2 = staticmethod(jnp.log2)
    log10 = staticmethod(jnp.log10)
    isfinite = staticmethod(jnp.isfinite)
    abs = staticmethod(jnp.abs)
    sign = staticmethod(jnp.sign)
    round = staticmethod(jnp.round)
    ceil = staticmethod(jnp.ceil)
    floor = staticmethod(jnp.floor)
    nonzero = staticmethod(jnp.nonzero)
    flip = staticmethod(jnp.flip)
    stop_gradient = staticmethod(jax.lax.stop_gradient)
    transpose = staticmethod(jnp.transpose)
    equal = staticmethod(jnp.equal)
    tile = staticmethod(jnp.tile)
    stack = staticmethod(jnp.stack)
    concat = staticmethod(jnp.concatenate)
    zeros_like = staticmethod(jnp.zeros_like)
    ones_like = staticmethod(jnp.ones_like)
    maximum = staticmethod(jnp.maximum)
    minimum = staticmethod(jnp.minimum)
    clip = staticmethod(jnp.clip)
    shape = staticmethod(jnp.shape)
    staticshape = staticmethod(jnp.shape)
    imag = staticmethod(jnp.imag)
    real = staticmethod(jnp.real)
    conj = staticmethod(jnp.conjugate)
    einsum = staticmethod(jnp.einsum)
    cumsum = staticmethod(jnp.cumsum)

    def jit_compile(self, f: Callable) -> Callable:
        def run_jit_f(*args):
            logging.debug(f"JaxBackend: running jit-compiled '{f.__name__}' with shapes {[arg.shape for arg in args]} and dtypes {[arg.dtype.name for arg in args]}")
            return self.as_registered.call(jit_f, *args, name=f"run jit-compiled '{f.__name__}'")

        run_jit_f.__name__ = f"Jax-Jit({f.__name__})"
        jit_f = jax.jit(f)
        return run_jit_f

    def block_until_ready(self, values):
        if isinstance(values, DeviceArray):
            values.block_until_ready()
        if isinstance(values, (tuple, list)):
            for v in values:
                self.block_until_ready(v)

    def functional_gradient(self, f, wrt: tuple or list, get_output: bool):
        if get_output:
            @wraps(f)
            def aux_f(*args):
                output = f(*args)
                if isinstance(output, (tuple, list)) and len(output) == 1:
                    output = output[0]
                result = (output[0], output[1:]) if isinstance(output, (tuple, list)) else (output, None)
                if result[0].ndim > 0:
                    result = jnp.sum(result[0]), result[1]
                return result
            jax_grad_f = jax.value_and_grad(aux_f, argnums=wrt, has_aux=True)
            @wraps(f)
            def unwrap_outputs(*args):
                (loss, aux), grads = jax_grad_f(*args)
                return (loss, *aux, *grads) if aux is not None else (loss, *grads)
            return unwrap_outputs
        else:
            @wraps(f)
            def nonaux_f(*args):
                output = f(*args)
                result = output[0] if isinstance(output, (tuple, list)) else output
                if result.ndim > 0:
                    result = jnp.sum(result)
                return result
            return jax.grad(nonaux_f, argnums=wrt, has_aux=False)

    def custom_gradient(self, f: Callable, gradient: Callable) -> Callable:
        jax_fun = jax.custom_vjp(f)  # custom vector-Jacobian product (reverse-mode differentiation)

        def forward(*x):
            y = f(*x)
            return y, (x, y)

        def backward(x_y, dy):
            x, y = x_y
            dx = gradient(x, y, dy)
            return tuple(dx)

        jax_fun.defvjp(forward, backward)
        return jax_fun

    def divide_no_nan(self, x, y):
        return jnp.nan_to_num(x / y, copy=True, nan=0)

    def random_uniform(self, shape):
        self._check_float64()
        self.rnd_key, subkey = jax.random.split(self.rnd_key)
        return random.uniform(subkey, shape, dtype=to_numpy_dtype(self.float_type))

    def random_normal(self, shape):
        self._check_float64()
        self.rnd_key, subkey = jax.random.split(self.rnd_key)
        return random.normal(subkey, shape, dtype=to_numpy_dtype(self.float_type))

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
        if limit is None:
            start, limit = 0, start
        return jnp.arange(start, limit, delta, to_numpy_dtype(dtype))

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        assert mode in ('constant', 'symmetric', 'periodic', 'reflect', 'boundary'), mode
        if mode == 'constant':
            constant_values = jnp.array(constant_values, dtype=value.dtype)
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

    def zeros(self, shape, dtype: DType = None):
        self._check_float64()
        return jnp.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def ones(self, shape, dtype: DType = None):
        self._check_float64()
        return jnp.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def meshgrid(self, *coordinates):
        self._check_float64()
        return jnp.meshgrid(*coordinates, indexing='ij')

    def linspace(self, start, stop, number):
        self._check_float64()
        return jnp.linspace(start, stop, number, dtype=to_numpy_dtype(self.float_type))

    def mean(self, value, axis=None, keepdims=False):
        return jnp.mean(value, axis, keepdims=keepdims)

    def tensordot(self, a, a_axes: tuple or list, b, b_axes: tuple or list):
        return jnp.tensordot(a, b, (a_axes, b_axes))

    def mul(self, a, b):
        # if scipy.sparse.issparse(a):  # TODO sparse?
        #     return a.multiply(b)
        # elif scipy.sparse.issparse(b):
        #     return b.multiply(a)
        # else:
            return Backend.mul(self, a, b)

    def matmul(self, A, b):
        return jnp.stack([A.dot(b[i]) for i in range(b.shape[0])])

    def while_loop(self, loop: Callable, values: tuple):
        if all(self.is_available(t) for t in values):
            while jnp.any(values[0]):
                values = loop(*values)
            return values
        else:
            cond = lambda vals: jnp.any(vals[0])
            body = lambda vals: loop(*vals)
            return jax.lax.while_loop(cond, body, values)

    def max(self, x, axis=None, keepdims=False):
        return jnp.max(x, axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return jnp.min(x, axis, keepdims=keepdims)

    def conv(self, value, kernel, zero_padding=True):
        assert kernel.shape[0] in (1, value.shape[0])
        assert value.shape[1] == kernel.shape[2], f"value has {value.shape[1]} channels but kernel has {kernel.shape[2]}"
        assert value.ndim + 1 == kernel.ndim
        # AutoDiff may require jax.lax.conv_general_dilated
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
            a = jnp.expand_dims(a, axis)
        return a

    def cast(self, x, dtype: DType):
        if self.is_tensor(x, only_native=True) and from_numpy_dtype(x.dtype) == dtype:
            return x
        else:
            return jnp.array(x, to_numpy_dtype(dtype))

    def batched_gather_nd(self, values, indices):
        assert indices.shape[-1] == self.ndims(values) - 2
        batch_size = combined_dim(values.shape[0], indices.shape[0])
        results = []
        for b in range(batch_size):
            b_values = values[min(b, values.shape[0] - 1)]
            b_indices = self.unstack(indices[min(b, indices.shape[0] - 1)], -1)
            results.append(b_values[b_indices])
        return jnp.stack(results)

    def std(self, x, axis=None, keepdims=False):
        return jnp.std(x, axis, keepdims=keepdims)

    def boolean_mask(self, x, mask, axis=0):
        slices = [mask if i == axis else slice(None) for i in range(len(x.shape))]
        return x[tuple(slices)]

    def any(self, boolean_tensor, axis=None, keepdims=False):
        if isinstance(boolean_tensor, (tuple, list)):
            boolean_tensor = jnp.stack(boolean_tensor)
        return jnp.any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        if isinstance(boolean_tensor, (tuple, list)):
            boolean_tensor = jnp.stack(boolean_tensor)
        return jnp.all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, base_grid, indices, values, mode: str):
        base_grid, values = self.auto_cast(base_grid, values)
        batch_size = combined_dim(combined_dim(indices.shape[0], values.shape[0]), base_grid.shape[0])
        spatial_dims = tuple(range(base_grid.ndim - 2))
        dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(1,),  # channel dim of updates (batch dim removed)
                                                inserted_window_dims=spatial_dims,  # no idea what this does but spatial_dims seems to work
                                                scatter_dims_to_operand_dims=spatial_dims)  # spatial dims of base_grid (batch dim removed)
        scatter = jax.lax.scatter_add if mode == 'add' else jax.lax.scatter
        result = []
        for b in range(batch_size):
            b_grid = base_grid[b, ...]
            b_indices = indices[min(b, indices.shape[0] - 1), ...]
            b_values = values[min(b, values.shape[0] - 1), ...]
            result.append(scatter(b_grid, b_indices, b_values, dnums))
        return jnp.stack(result)

    def quantile(self, x, quantiles):
        return jnp.quantile(x, quantiles, axis=-1)

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
        if not isinstance(array, jnp.ndarray):
            array = jnp.array(array)
        return from_numpy_dtype(array.dtype)

    def linear_solve(self, method: str, lin, y, x0, rtol, atol, max_iter, trj: bool) -> SolveResult or List[SolveResult]:
        if method == 'auto' and not trj and not self.is_available(y):
            return self.conjugate_gradient(lin, y, x0, rtol, atol, max_iter, trj)
        else:
            return Backend.linear_solve(self, method, lin, y, x0, rtol, atol, max_iter, trj)

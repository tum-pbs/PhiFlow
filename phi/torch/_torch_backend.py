import numbers
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import List, Callable, Optional, Set, Tuple, Any

import numpy as np
import torch
import torch.fft
import torch.nn.functional as torchf
from packaging import version

from phi.math import DType
from phi.math.backend import Backend, NUMPY, ComputeDevice, PHI_LOGGER
from phi.math.backend._backend import combined_dim, SolveResult, get_functional_derivative_order, TensorType


class TorchBackend(Backend):

    def __init__(self):
        cpu = NUMPY.get_default_device()
        devices = [ComputeDevice(self, "CPU", 'CPU', cpu.memory, cpu.processor_count, cpu.description, ref='cpu')]
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(ComputeDevice(self, properties.name, 'GPU', properties.total_memory, properties.multi_processor_count, f"compute capability {properties.major}.{properties.minor}", f'cuda:{index}'))
        Backend.__init__(self, 'PyTorch', devices, devices[1 if len(devices) > 1 else 0])

    def prefers_channels_last(self) -> bool:
        return False

    def is_module(self, obj):
        return isinstance(obj, (JITFunction, torch.nn.Module))

    def is_tensor(self, x, only_native=False):
        if isinstance(x, torch.Tensor):
            return True
        if only_native:
            return False
        if isinstance(x, numbers.Number):
            return True
        if isinstance(x, (tuple, list)) and all(isinstance(c, numbers.Number) for c in x):
            return True
        if isinstance(x, np.ndarray) and x.dtype != np.object:
            return True  # this is pretty much required, else we couldn't perform NP+PyTorch operations
        return False

    def as_tensor(self, x, convert_external=True):
        if isinstance(x, torch.nn.Module):
            return x
        if self.is_tensor(x, only_native=convert_external):
            tensor = x
        elif isinstance(x, np.ndarray):
            try:
                tensor = torch.from_numpy(x)
            except ValueError:  # or TypeError?
                tensor = torch.from_numpy(x.copy())
            tensor = tensor.to(self.get_default_device().ref)
        elif isinstance(x, (tuple, list)):
            try:
                x = np.stack(x)
                tensor = torch.tensor(x, device=self.get_default_device().ref)
            except ValueError:  # there may be Tensors inside the list
                components = [self.as_tensor(c) for c in x]
                tensor = torch.stack(components, dim=0)
        else:
            tensor = torch.tensor(x, device=self.get_default_device().ref)
        # --- Enforce Precision ---
        if self.is_tensor(tensor, only_native=True):
            dtype = self.dtype(tensor)
            if dtype.kind == float:
                tensor = self.to_float(tensor)
            elif dtype.kind == complex:
                tensor = self.to_complex(tensor)
        # --- Move to default device ---
        if isinstance(tensor, torch.Tensor) and tensor.device != self.get_default_device().ref:
            tensor = tensor.to(self.get_default_device().ref)
        return tensor

    def recursive_as_tensor(self, obj):
        if isinstance(obj, (tuple, list)):
            return type(obj)([self.recursive_as_tensor(item) for item in obj])
        elif isinstance(obj, dict):
            raise NotImplementedError()
        else:
            return self.as_tensor(obj)

    def auto_cast(self, *tensors, **kwargs) -> list:
        tensors = [t if isinstance(t, (numbers.Number, bool)) else self.as_tensor(t, True) for t in tensors]
        return Backend.auto_cast(self, *tensors, **kwargs)

    def is_available(self, tensor) -> bool:
        # return True
        return torch._C._get_tracing_state() is None  # TODO can we find out whether this tensor specifically is being traced?

    def numpy(self, tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()
        if hasattr(tensor, 'resolve_conj'):
            tensor = tensor.resolve_conj()
        return tensor.cpu().numpy()

    def to_dlpack(self, tensor):
        from torch.utils import dlpack
        return dlpack.to_dlpack(tensor)

    def from_dlpack(self, capsule):
        from torch.utils import dlpack
        tensor = dlpack.from_dlpack(capsule)
        tensor = tensor.to(self.get_default_device().ref)
        return tensor

    def copy(self, tensor, only_mutable=False):
        return torch.clone(tensor)

    def get_device(self, tensor: TensorType) -> ComputeDevice:
        return self.get_device_by_ref(str(tensor.device))

    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        return self.as_tensor(tensor).to(device.ref)

    def multi_slice(self, tensor, slices: tuple):
        neg_slices = [i for i, s in enumerate(slices) if isinstance(s, slice) and s.step is not None and s.step < 0]
        if neg_slices:
            tensor = torch.flip(tensor, neg_slices)
        pos_slices = [slice(s.start, s.stop, -s.step) if i in neg_slices else s for i, s in enumerate(slices)]
        return tensor[tuple(pos_slices)]

    sqrt = torch.sqrt
    exp = torch.exp
    sin = torch.sin
    arcsin = torch.arcsin
    cos = torch.cos
    arccos = torch.arccos
    tan = torch.tan
    log = torch.log
    log2 = torch.log2
    log10 = torch.log10
    sigmoid = torch.sigmoid
    isfinite = torch.isfinite
    abs = torch.abs
    sign = torch.sign
    round = torch.round
    ceil = torch.ceil
    floor = torch.floor
    nonzero = torch.nonzero
    flip = torch.flip
    seed = staticmethod(torch.manual_seed)
    einsum = staticmethod(torch.einsum)

    def jit_compile(self, f: Callable) -> Callable:
        return JITFunction(self, f)

    def custom_gradient(self, f: Callable, gradient: Callable = None, get_external_cache: Callable = None, on_call_skipped: Callable = None) -> Callable:
        """ See PyTorch_Jit.md """
        def select_jit(*args):
            args = [self.as_tensor(arg) for arg in args]
            if not CURRENT_JIT_CALLS:
                return torch_function.apply(*args)
            jit = CURRENT_JIT_CALLS[-1]
            if torch._C._get_tracing_state() is not None:  # second call: we are tracing
                compiled_function, ext_cache = jit.get_compiled_function(torch_function, args)  # increases counter
                if on_call_skipped:
                    on_call_skipped(ext_cache)
                return compiled_function.apply(*args)  # this adds the compiled function to TorchScript. The function must not call any torch functions while being traced lest they be double-executed later.
            else:  # first call: record this function
                output = torch_function.apply(*args)
                ext_cache = get_external_cache() if get_external_cache else None
                jit.record_autograd_function_call(torch_function, args, output, ext_cache)
                return output

        torch_function = construct_torch_custom_function(f, None, None, gradient, is_f_traced=False, backend=self)
        return select_jit

    def transpose(self, tensor, axes):
        return tensor.permute(axes)

    def equal(self, x, y):
        x, y = self.auto_cast(x, y)
        return x == y

    def random_uniform(self, shape, low, high, dtype: DType or None):
        dtype = dtype or self.float_type
        if dtype.kind == float:
            return low + (high - low) * torch.rand(size=shape, dtype=to_torch_dtype(dtype), device=self.get_default_device().ref)
        elif dtype.kind == complex:
            real = low.real + (high.real - low.real) * torch.rand(size=shape, dtype=to_torch_dtype(DType(float, dtype.precision)), device=self.get_default_device().ref)
            imag = low.imag + (high.imag - low.imag) * torch.rand(size=shape, dtype=to_torch_dtype(DType(float, dtype.precision)), device=self.get_default_device().ref)
            return real + 1j * imag
        elif dtype.kind == int:
            return torch.randint(low, high, shape, dtype=to_torch_dtype(dtype))
        else:
            raise ValueError(dtype)

    def random_normal(self, shape, dtype: DType):
        return torch.randn(size=shape, dtype=to_torch_dtype(dtype or self.float_type), device=self.get_default_device().ref)

    def stack(self, values, axis=0):
        values = [self.as_tensor(v) for v in values]
        return torch.stack(values, dim=axis)

    def concat(self, values, axis):
        values = [self.as_tensor(v) for v in values]
        return torch.cat(values, dim=axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        """
        pad tensor using mode

        Args:
          value(torch.Tensor): values
          pad_width(iterable): left, right, upper, lower
          mode(str, optional, optional): type of padding to be applied, defaults to 'constant'
          constant_values(int, optional, optional): value to pad, defaults to 0

        Returns:
          torch.Tensor: padded tensor
        """
        mode = {'constant': 'constant', 'reflect': 'reflect', 'boundary': 'replicate', 'periodic': 'circular'}.get(mode, None)
        if not mode:
            return NotImplemented
        # transpose for leading zero-pad: [(0, 0), (0, 0), ...]
        ndims = self.ndims(value)
        if ndims > 2 and pad_width[0] == pad_width[1] == (0, 0):
            reordered = value
            pad_width_reordered = pad_width[2:]
            undo_transform = lambda x: x
        elif ndims > 2 and pad_width[0] == (0, 0) and self.ndims(value) < 5:
            reordered = torch.unsqueeze(value, 0)
            pad_width_reordered = pad_width[1:]
            undo_transform = lambda x: torch.squeeze(x, 0)
        elif ndims < 4:
            reordered = torch.unsqueeze(torch.unsqueeze(value, 0), 0)
            pad_width_reordered = pad_width
            undo_transform = lambda x: torch.squeeze(torch.squeeze(x, 0), 0)
        else:
            raise NotImplementedError()  # TODO transpose to get (0, 0) to the front
        pad_width_spatial = [item for sublist in reversed(pad_width_reordered) for item in sublist]  # flatten
        try:
            constant_values = self.dtype(value).kind(constant_values)
            result = torchf.pad(reordered, pad_width_spatial, mode, value=constant_values)  # supports 3D to 5D (2 + 1D to 3D)
        except RuntimeError as err:
            warnings.warn(f"PyTorch error {err}", RuntimeWarning)
            return NotImplemented
        result = undo_transform(result)
        return result

    def grid_sample(self, grid, coordinates, extrapolation: str):
        assert extrapolation in ('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect'), extrapolation
        if get_functional_derivative_order() > 1:
            return NotImplemented  # PyTorch's grid_sample operator does not define higher-order derivatives
        extrapolation = {'undefined': 'zeros', 'zeros': 'zeros', 'boundary': 'border', 'reflect': 'reflection'}.get(extrapolation, None)
        if extrapolation is None:
            return NotImplemented
        grid = channels_first(self.as_tensor(grid))
        coordinates = self.as_tensor(coordinates)
        if coordinates.shape[0] != grid.shape[0]:  # repeating yields wrong result
            return NotImplemented
        if coordinates.ndim != grid.ndim or coordinates.ndim not in (4, 5):
            return NotImplemented  # torchf.grid_sample cannot handle this case
        if coordinates.dtype.is_floating_point and not grid.dtype.is_complex and not grid.dtype.is_floating_point:
            grid = self.to_float(grid)
        resolution = torch.tensor(self.staticshape(grid)[2:], dtype=coordinates.dtype, device=coordinates.device)
        coordinates = 2 * coordinates / (resolution - 1) - 1
        coordinates = torch.flip(coordinates, dims=[-1])
        batch_size = combined_dim(coordinates.shape[0], grid.shape[0])
        coordinates = coordinates.repeat(batch_size, *[1] * (len(coordinates.shape-1))) if coordinates.shape[0] < batch_size else coordinates
        grid = grid.repeat(batch_size, *[1] * (len(grid.shape)-1)) if grid.shape[0] < batch_size else grid
        result = torchf.grid_sample(grid, coordinates, mode='bilinear', padding_mode=extrapolation, align_corners=True)  # can cause segmentation violation if NaN or inf are present
        result = channels_last(result)
        return result

    def reshape(self, value, shape):
        return torch.reshape(self.as_tensor(value), shape)

    def sum(self, value, axis=None, keepdims=False):
        if axis is None:
            axis = tuple(range(len(value.shape)))
        return torch.sum(value, dim=axis, keepdim=keepdims)

    def prod(self, value, axis=None):
        if isinstance(axis, (tuple, list)):
            for dim in reversed(sorted(axis)):
                value = torch.prod(value, dim=dim)
            return value
        return torch.prod(value, dim=axis)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        boolean_tensor = self.as_tensor(boolean_tensor, convert_external=True)
        if self.dtype(boolean_tensor).kind != bool:
            boolean_tensor = boolean_tensor != 0
        if axis is None:
            return torch.any(boolean_tensor)
        else:
            axes = axis if isinstance(axis, (tuple, list)) else [axis]
            for axis in reversed(sorted(axes)):
                boolean_tensor = torch.any(boolean_tensor, dim=axis, keepdim=keepdims)
            return boolean_tensor

    def all(self, boolean_tensor, axis=None, keepdims=False):
        boolean_tensor = self.as_tensor(boolean_tensor, convert_external=True)
        if self.dtype(boolean_tensor).kind != bool:
            boolean_tensor = boolean_tensor != 0
        if axis is None:
            return torch.all(boolean_tensor)
        else:
            axes = axis if isinstance(axis, (tuple, list)) else [axis]
            for axis in reversed(sorted(axes)):
                boolean_tensor = torch.all(boolean_tensor, dim=axis, keepdim=keepdims)
            return boolean_tensor

    def quantile(self, x, quantiles):
        x = self.to_float(x)
        result = torch.quantile(x, quantiles, dim=-1)
        return result

    def divide_no_nan(self, x, y):
        x, y = self.auto_cast(x, y)
        return divide_no_nan(x, y)

    def where(self, condition, x=None, y=None):
        condition = self.as_tensor(condition).bool()
        x, y = self.auto_cast(x, y)
        x = self.as_tensor(x)
        y = self.as_tensor(y)
        return torch.where(condition, x, y)

    def mean(self, value, axis=None, keepdims=False):
        if self.dtype(value).kind not in (float, complex):
            value = self.to_float(value)
        return torch.mean(value, dim=axis, keepdim=keepdims)

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
        if limit is None:
            start, limit = 0, start
        return torch.arange(start, limit, delta, dtype=to_torch_dtype(dtype))

    def zeros(self, shape, dtype=None):
        return torch.zeros(shape, dtype=to_torch_dtype(dtype or self.float_type), device=self.get_default_device().ref)

    def zeros_like(self, tensor):
        return torch.zeros_like(self.as_tensor(tensor), device=self.get_default_device().ref)

    def ones(self, shape, dtype: DType = None):
        return torch.ones(shape, dtype=to_torch_dtype(dtype or self.float_type), device=self.get_default_device().ref)

    def ones_like(self, tensor):
        return torch.ones_like(self.as_tensor(tensor), device=self.get_default_device().ref)

    def meshgrid(self, *coordinates):
        coordinates = [self.as_tensor(c) for c in coordinates]
        from packaging import version
        if version.parse(torch.__version__) >= version.parse('1.10'):
            return torch.meshgrid(*coordinates, indexing='ij')
        else:
            return torch.meshgrid(*coordinates)

    def linspace(self, start, stop, number):
        return torch.linspace(start, stop, number, dtype=to_torch_dtype(self.float_type), device=self.get_default_device().ref)

    def tensordot(self, a, a_axes: tuple or list, b, b_axes: tuple or list):
        a, b = self.auto_cast(a, b)
        return torch.tensordot(a, b, (a_axes, b_axes))

    def matmul(self, A, b):
        A, b = self.auto_cast(A, b)
        if isinstance(A, torch.Tensor) and A.is_sparse:
            result = torch.sparse.mm(A, torch.transpose(b, 0, 1))
            return torch.transpose(result, 0, 1)
        raise NotImplementedError(type(A), type(b))

    def cumsum(self, x, axis: int):
        return torch.cumsum(x, dim=axis)

    def while_loop(self, loop: Callable, values: tuple):
        if torch._C._get_tracing_state() is not None:
            if isinstance(loop, torch.ScriptFunction):
                jit_loop = loop
                while torch.any(values[0]):
                    values = jit_loop(*values)
                return values
            else:
                warnings.warn("Tracing a PyTorch while loop requires an additional tracing pass. You can avoid this by passing a torch.ScriptFunction.", RuntimeWarning)
                raise NotImplementedError()
                # def trace_later():
                #     jit_loop = torch.jit.trace(loop, check_trace=False)
                #     @torch.jit.script
                #     def loop_script(values: Tuple[torch.Tensor], loop_script: Callable):
                #         while torch.any(values[0]):
                #             values = loop_script(*values)
                #         return values
                # CURRENT_JIT_CALLS[-1].post_trace.append(trace_later)
        else:
            while torch.any(values[0]):
                values = loop(*values)
            return values

    def max(self, x, axis=None, keepdims=False):
        if axis is None:
            result = torch.max(x)
            if keepdims:
                result = self.expand_dims(result, axis=0, number=self.ndims(x))
            return result
        elif isinstance(axis, (tuple, list)):
            for dim in reversed(sorted(axis)):
                x, _ = torch.max(x, dim=dim, keepdim=keepdims)
            return x
        else:
            return torch.max(x, dim=axis, keepdim=keepdims)[0]

    def min(self, x, axis=None, keepdims=False):
        if axis is None:
            result = torch.min(x)
            if keepdims:
                result = self.expand_dims(result, axis=0, number=self.ndims(x))
            return result
        elif isinstance(axis, (tuple, list)):
            for dim in reversed(sorted(axis)):
                x, _ = torch.min(x, dim=dim, keepdim=keepdims)
            return x
        else:
            return torch.min(x, dim=axis, keepdim=keepdims)[0]

    def maximum(self, a, b):
        a_ = self.as_tensor(a)
        b_ = self.as_tensor(b)
        return torch.max(a_, other=b_)

    def minimum(self, a, b):
        a_ = self.as_tensor(a)
        b_ = self.as_tensor(b)
        return torch.min(a_, other=b_)

    def clip(self, x, minimum, maximum):
        if isinstance(minimum, numbers.Number) and isinstance(maximum, numbers.Number):
            return torch.clamp(self.as_tensor(x), minimum, maximum)
        else:
            return self.maximum(minimum, self.minimum(x, maximum))

    def conv(self, value, kernel, zero_padding=True):
        value = self.as_tensor(value)
        kernel = self.as_tensor(kernel)
        value, kernel = self.auto_cast(value, kernel)
        if self.dtype(value).kind in (bool, int):
            value = self.to_float(value)
            kernel = self.to_float(kernel)
        if zero_padding:
            if all(s % 2 == 1 for s in kernel.shape[3:]):
                padding = [s // 2 for s in kernel.shape[3:]]
            else:
                padding = 0
                value_padding = sum([[s // 2, (s - 1) // 2] for s in kernel.shape[3:]], [])
                value = torchf.pad(value, value_padding)
        else:
            padding = 0
        convf = {3: torchf.conv1d, 4: torchf.conv2d, 5: torchf.conv3d}[len(value.shape)]
        if kernel.shape[0] == 1:
            result = convf(value, kernel[0, ...], padding=padding)
        else:
            result = []
            for b in range(kernel.shape[0]):
                result.append(convf(value[b:b+1, ...], kernel[b, ...], padding=padding))
            result = torch.cat(result, 0)
        return result

    def expand_dims(self, a, axis=0, number=1):
        for _ in range(number):
            a = torch.unsqueeze(a, dim=axis)
        return a

    def shape(self, tensor):
        if self.is_tensor(tensor, only_native=True):
            return tensor.shape
        else:
            return NUMPY.shape(tensor)

    def staticshape(self, tensor):
        if isinstance(tensor, torch.nn.Module):
            return ()
        if self.is_tensor(tensor, only_native=True):
            return tuple([int(s) for s in tensor.shape])
        else:
            return NUMPY.staticshape(tensor)

    def batched_gather_nd(self, values, indices):
        values = self.as_tensor(values)
        indices = self.as_tensor(indices).long()
        batch_size = combined_dim(values.shape[0], indices.shape[0])
        result = []
        for b in range(batch_size):
            b_indices = self.unstack(indices[min(b, indices.shape[0] - 1)], -1)
            result.append(values[(min(b, values.shape[0] - 1),) + b_indices])
        return self.stack(result, axis=0)

    def unstack(self, tensor, axis=0, keepdims=False):
        unstacked = torch.unbind(tensor, dim=axis)
        if keepdims:
            unstacked = [self.expand_dims(c, axis=axis) for c in unstacked]
        return unstacked

    def std(self, x, axis=None, keepdims=False):
        if self.dtype(x).kind not in (float, complex):
            x = self.to_float(x)
        return torch.std(x, dim=axis, keepdim=keepdims, unbiased=False)

    def boolean_mask(self, x, mask, axis=0):
        x = self.as_tensor(x)
        mask = self.as_tensor(mask)
        result = []
        for selected, data in zip(mask, self.unstack(x, axis)):
            if selected:
                result.append(data)
        return self.stack(result, axis)
        # return torch.masked_select(x_, mask_)

    def scatter(self, base_grid, indices, values, mode: str):
        base_grid, values = self.auto_cast(base_grid, values)
        indices = self.as_tensor(indices)
        batch_size = combined_dim(combined_dim(indices.shape[0], values.shape[0]), base_grid.shape[0])
        scatter = torch.scatter_add if mode == 'add' else torch.scatter
        if indices.shape[0] < batch_size:
            indices = indices.repeat([batch_size] + [1] * (len(indices.shape)-1))
        if values.shape[0] < batch_size or values.shape[1] == 1:
            values = values.repeat([batch_size // values.shape[0], indices.shape[1] // indices.shape[1]] + [1] * (len(values.shape)-2))
        if len(base_grid.shape) > 3:
            resolution = base_grid.shape[1:-1]
            ravel = [1]
            for i in range(1, len(resolution)):
                ravel.insert(0, ravel[0] * resolution[-i])
            ravel = self.to_int64(self.as_tensor(ravel, True))
            indices = torch.sum(indices * ravel, dim=-1, keepdim=True)
        base_grid_flat = torch.reshape(base_grid, [base_grid.shape[0], -1, base_grid.shape[-1]])
        indices = indices.long().repeat([1, 1, values.shape[-1]])
        result = scatter(base_grid_flat, dim=1, index=indices, src=values)
        return torch.reshape(result, base_grid.shape)

    def fft(self, x, axes: tuple or list):
        if not x.is_complex():
            x = self.to_complex(x)
        for i in axes:
            x = torch.fft.fft(x, dim=i)
        return x

    def ifft(self, k, axes: tuple or list):
        if not k.is_complex():
            k = self.to_complex(k)
        for i in axes:
            k = torch.fft.ifft(k, dim=i)
        return k

    def imag(self, x):
        dtype = self.dtype(x)
        if dtype.kind == complex:
            return torch.imag(x)
        else:
            return self.zeros(x.shape, DType(float, dtype.precision))

    def real(self, x):
        if self.dtype(x).kind == complex:
            return torch.real(x)
        else:
            return x

    def conj(self, x):
        if self.dtype(x).kind == complex:
            return torch.conj(x)
        else:
            return x

    def cast(self, x, dtype: DType):
        if isinstance(x, (numbers.Number, bool)):
            return dtype.kind(x)  # Creating a Tensor here would raise warnings during tracing.
        if not self.is_tensor(x, only_native=True):
            x = self.as_tensor(x)
        if self.dtype(x) == dtype:
            return x
        else:
            return x.to(to_torch_dtype(dtype))

    def dtype(self, array) -> DType:
        if self.is_tensor(array, only_native=True):
            return from_torch_dtype(array.dtype)
        else:
            return NUMPY.dtype(array)

    def tile(self, value, multiples):
        if isinstance(multiples, np.ndarray):
            multiples = multiples.tolist()
        return self.as_tensor(value).repeat(multiples)

    def sparse_coo_tensor(self, indices, values, shape):
        indices_ = self.to_int64(indices)
        values_ = self.to_float(values)
        if not self.is_available(values_):
            # the output of torch.sparse_coo_tensor is considered constant
            @torch.jit.script
            def sparse_coo_tensor(values, indices, cols: int, rows: int, dtype: torch.dtype) -> torch.sparse.Tensor:
                size = torch.Size([cols, rows])
                return torch.sparse_coo_tensor(indices, values, size=size, dtype=dtype)
            result = sparse_coo_tensor(values_, indices_, shape[0], shape[1], to_torch_dtype(self.float_type))
        else:
            result = torch.sparse_coo_tensor(indices_, values_, shape, dtype=to_torch_dtype(self.float_type))
        return result

    def coordinates(self, tensor):
        assert isinstance(tensor, torch.Tensor) and tensor.is_sparse
        idx = tensor._indices()
        idx = self.unstack(idx, axis=0)
        return idx, tensor._values()

    def conjugate_gradient(self, lin, y, x0, rtol, atol, max_iter, trj: bool) -> SolveResult or List[SolveResult]:
        if callable(lin) or trj:
            assert self.is_available(y), "Tracing conjugate_gradient with linear operator is not yet supported."
            return Backend.conjugate_gradient(self, lin, y, x0, rtol, atol, max_iter, trj)
        assert isinstance(lin, torch.Tensor) and lin.is_sparse, "Batched matrices are not yet supported"
        y = self.to_float(y)
        x0 = self.copy(self.to_float(x0))
        rtol = self.as_tensor(rtol)
        atol = self.as_tensor(atol)
        max_iter = self.as_tensor(max_iter)
        x, residual, iterations, function_evaluations, converged, diverged = torch_sparse_cg(lin, y, x0, rtol, atol, max_iter)
        return SolveResult(f"Φ-Flow CG ({'PyTorch*' if self.is_available(y) else 'TorchScript'})", x, residual, iterations, function_evaluations, converged, diverged, "")

    def conjugate_gradient_adaptive(self, lin, y, x0, rtol, atol, max_iter, trj: bool) -> SolveResult or List[SolveResult]:
        if callable(lin) or trj:
            assert self.is_available(y), "Tracing conjugate_gradient with linear operator is not yet supported."
            return Backend.conjugate_gradient_adaptive(self, lin, y, x0, rtol, atol, max_iter, trj)
        assert isinstance(lin, torch.Tensor) and lin.is_sparse, "Batched matrices are not yet supported"
        y = self.to_float(y)
        x0 = self.copy(self.to_float(x0))
        rtol = self.as_tensor(rtol)
        atol = self.as_tensor(atol)
        max_iter = self.as_tensor(max_iter)
        x, residual, iterations, function_evaluations, converged, diverged = torch_sparse_cg_adaptive(lin, y, x0, rtol, atol, max_iter)
        return SolveResult(f"Φ-Flow CG ({'PyTorch*' if self.is_available(y) else 'TorchScript'})", x, residual, iterations, function_evaluations, converged, diverged, "")

    def matrix_solve_least_squares(self, matrix: TensorType, rhs: TensorType) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        assert version.parse(torch.__version__) >= version.parse('1.9.0'), "least squares requires PyTorch >= 1.9.0"
        matrix, rhs = self.auto_cast(matrix, rhs)
        solution, residuals, rank, singular_values = torch.linalg.lstsq(matrix, rhs)
        return solution, residuals, rank, singular_values

    def _prepare_graph_inputs(self, args: tuple, wrt: tuple or list):
        args = [self.as_tensor(arg, True) if i in wrt else arg for i, arg in enumerate(args)]
        args = [self.to_float(arg) if self.dtype(arg).kind == int else arg for arg in args]
        for i, arg in enumerate(args):
            if self.is_tensor(arg, True) and arg.requires_grad and not arg.is_leaf:
                arg = torch.clone(arg).detach()
                arg.requires_grad = i in wrt
                args[i] = arg
            elif i in wrt:
                arg = self.as_tensor(arg, True)
                arg = arg.detach()  # returns a new tensor in any case
                arg.requires_grad = True
                args[i] = arg
        wrt_args = [arg for i, arg in enumerate(args) if i in wrt]
        for t in wrt_args:
            assert t.requires_grad
        return args, wrt_args

    def jacobian(self, f, wrt: tuple or list, get_output: bool, is_f_scalar: bool):
        @wraps(f)
        def eval_grad(*args):
            args, wrt_args = self._prepare_graph_inputs(args, wrt)
            loss, output = f(*args)
            if np.prod(self.staticshape(loss)) == 1:
                grads = torch.autograd.grad(loss, wrt_args)  # grad() cannot be called during jit trace
            else:
                raise NotImplementedError()
                grads = torch.autograd.grad(loss, wrt_args, retain_graph=True)
            return (*output, *grads) if get_output else grads
        return eval_grad

    def hessian(self, f: Callable, wrt: tuple or list, get_output: bool, get_gradient: bool):
        # if not get_output and not get_gradient:
        # @wraps(f)
        # def eval_hessian(*args):
        #     batch_size = args[0].shape[0]
        #     for arg in args:
        #         assert arg.shape[0] == batch_size, f"All arguments must have a matching batch dimension as their first dimension. Got shapes {[arg.shape for arg in args]}"
        #
        #     def f_only_wrt_inputs(*wrt_args_only, reduce_batch=False):
        #         all_args = list(args)
        #         for i, arg in zip(wrt, wrt_args_only):
        #             all_args[i] = arg
        #         output = f(*all_args)
        #         loss, aux = (output[0], output[1:]) if isinstance(output, (tuple, list)) else (output, None)
        #         if reduce_batch:
        #             if loss.ndim > 0:
        #                 loss = loss.sum()
        #         else:
        #             assert np.prod(loss.shape) == 1, f"Loss (first output of f) must be scalar but has shape {loss.shape}"
        #             loss = loss.sum()
        #         return loss
        #
        #     wrt_args = tuple([self.as_tensor(arg, True) for i, arg in enumerate(args) if i in wrt])
        #     result = ()
        #     if get_output:
        #         result += f(*args),
        #     if get_gradient:
        #         result += torch.autograd.functional.jacobian(lambda *a: f_only_wrt_inputs(*a, reduce_batch=True), wrt_args),
        #     if hasattr(torch, 'vmap'):
        #         # single_hessian_f = lambda *args: torch.autograd.functional.hessian(f_only_wrt_inputs, args)
        #         # multi_hessian_f = torch.vmap
        #         raise NotImplementedError()
        #     else:
        #         hessian = tuple([tuple([[] for _1 in range(len(wrt))]) for _2 in range(len(wrt))])  # n x n matrix of lists
        #         for b in range(batch_size):
        #             h = torch.autograd.functional.hessian(f_only_wrt_inputs, tuple([arg[b:b + 1] for arg in wrt_args]))
        #             for i in range(len(wrt)):
        #                 for j in range(len(wrt)):
        #                     fake_batch_dim = args[i].ndim
        #                     hessian[i][j].append(torch.squeeze(torch.squeeze(h[i][j], fake_batch_dim), 0))
        #         hessian = [[torch.stack(hessian[i][j]) for j in range(len(wrt))] for i in range(len(wrt))]
        #         # hessian = torch.stack([torch.autograd.functional.hessian(f_only_wrt_inputs, tuple([arg[b:b+1] for arg in wrt_args])) for b in range(batch_size)])  # manual batch loop
        #     result += hessian,
        #     return result
        # else:
        @wraps(f)
        def eval_hessian(*args):
            args, wrt_args = self._prepare_graph_inputs(args, wrt)
            output = f(*args)
            loss, aux = (output[0], output[1:]) if isinstance(output, (tuple, list)) else (output, None)
            scalar_loss = loss.sum() if loss.ndim > 0 else loss
            grads = torch.autograd.grad(scalar_loss, wrt_args, create_graph=True, retain_graph=True)  # grad() cannot be called during jit trace
            hessian = []
            for grad in grads:
                if not grad.requires_grad:
                    raise NotImplementedError("Linear dependency detected. Hessian = 0.")
                hessian.append([[] for _ in grads])
                for lin_index in range(int(np.prod(grad.shape[1:]))):
                    multi_index = np.unravel_index(lin_index, grad.shape[1:])
                    h = torch.autograd.grad(grad[(slice(None),) + multi_index].sum(), wrt_args, allow_unused=True, retain_graph=True)  # grad of every entry in grad
                    # Warning: This returns incorrect values for certain inputs. Hessian of x^2 returns 0 at x=0 but is correct everywhere else.
                    # ToDo torch.autograd.functional.hessian does not seem to have this issue. Wait for torch.vmap(), then conditionally switch.
                    for i, h_ in enumerate(h):
                        hessian[-1][i].append(h_)
            for col in hessian:
                for i, row in enumerate(col):
                    if len(row) > 1:
                        col[i] = torch.stack(row, dim=1)
                    else:
                        col[i] = row[0]
                    h_shape = tuple(grads[i].shape) + tuple(grads[i].shape[1:])
                    col[i] = torch.reshape(col[i], h_shape)

            result = ()
            if get_output:
                loss = loss.detach()
                if aux is not None:
                    aux = [aux_.detach() if isinstance(aux_, torch.Tensor) else aux_ for aux_ in aux]
                    result += (loss, *aux),
                else:
                    result += loss,
            if get_gradient:
                result += tuple([g.detach() for g in grads]),
            result += hessian,
            return result

        return eval_hessian

    def jit_compile_grad(self, f, wrt: tuple or list, get_output: bool, is_f_scalar: bool):
        jit = self.jit_compile(f)
        return self.jacobian(jit, wrt, get_output, is_f_scalar)

    def jit_compile_hessian(self, f, wrt: tuple or list, get_output: bool, get_gradient: bool):
        jit = self.jit_compile(f)
        return self.hessian(jit, wrt, get_output, get_gradient)

    def stop_gradient(self, value):
        return value.detach()


def channels_first(x):
    return x.permute(*((0, -1) + tuple(range(1, len(x.shape) - 1))))


def channels_last(x):
    return x.permute((0,) + tuple(range(2, len(x.shape))) + (1,))


class JITFunction:
    """
    PyTorch Tracing Procedure:
    1. Call function non-tracing, record all called nn.Modules and autograd.Function forward calls with their args
    2. Compile autograd.Functions forward and backward passes
    3. Add nn.Modules to JIT Module
    4. Trace JIT Module

    Nested jit calls are ignored.

    See PyTorch_Jit.md
    """

    def __init__(self, backend: TorchBackend, f):
        self.backend = backend
        self.f = f
        self.traced = None
        self.autograd_function_calls = []  # (TorchCustomFunction, args, output, ext_cache)
        self.compiled_functions = []  # (TorchCustomFunction, TorchCustomFunction)
        self.autograd_function_call_counts = 0
        self.called_modules: Set[torch.nn.Module] = set()

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise NotImplementedError("kwargs not supported for traced function")
        if CURRENT_JIT_CALLS:
            warnings.warn(f"PyTorch does not support nested tracing. The inner JIT of {self.f.__name__} will be ignored.", RuntimeWarning)
            return self.f(*args)
        args = self.backend.recursive_as_tensor(args)
        if self.traced is None:
            self_jit = self
            CURRENT_JIT_CALLS.append(self)
            self.f(*args)  # records all autograd.Function / nn.Module calls with their args -> self.autograd_function_calls, self.called_modules
            for i, (rec_function, rec_args, rec_output, _ext_cache) in enumerate(self.autograd_function_calls):
                self.compiled_functions.append((rec_function, rec_function.compile(rec_args, rec_output)))
            assert self.autograd_function_call_counts == 0

            class JitModule(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    for submodule in self_jit.called_modules:
                        self.add_module(str(f"{type(submodule).__name__}_{id(submodule)}"), submodule)

                def forward(self, *args):
                    PHI_LOGGER.debug(f"Tracing Pytorch jit module for {self_jit.f.__name__}")
                    return self_jit.f(*args)

            module = JitModule()
            self.traced = torch.jit.trace(module, tuple(args), check_trace=False, strict=False)
            assert self.autograd_function_call_counts == len(self.autograd_function_calls), "Not all custom-gradient functions were called during tracing! Nested custom gradients are not supported."
            assert CURRENT_JIT_CALLS.pop(-1) == self
        from phi.math.backend import choose_backend
        return choose_backend(self).call(self.traced, *args, name=f"run jit-compiled '{self.f.__name__}'")

    def record_autograd_function_call(self, function: torch.autograd.Function, args, output, ext_cache):
        self.autograd_function_calls.append((function, args, output, ext_cache))

    def get_compiled_function(self, function: torch.autograd.Function, args) -> Tuple[torch.autograd.Function, Any]:
        assert torch._C._get_tracing_state() is not None
        assert self.autograd_function_call_counts < len(self.autograd_function_calls), f"More custom-gradient functions were called during tracing!\nLast encountered: {function}"
        assert len(self.autograd_function_calls) == len(self.compiled_functions)
        original_function, compiled_function = self.compiled_functions[self.autograd_function_call_counts]
        assert isinstance(compiled_function, torch.autograd.Function)
        function, args, output, ext_cache = self.autograd_function_calls[self.autograd_function_call_counts]
        self.autograd_function_call_counts += 1
        return compiled_function, ext_cache

    def __repr__(self):
        return f"TorchScript[{self.f.__name__}]"


CURRENT_JIT_CALLS: List[JITFunction] = []  # should contain no more than 1 element; PyTorch doesn't support nested tracing


def register_module_call(module: torch.nn.Module):
    if CURRENT_JIT_CALLS:
        CURRENT_JIT_CALLS[-1].called_modules.add(module)


def construct_torch_custom_function(f: Callable, jit_f: Optional[Callable], f_example_output, g: Callable, is_f_traced: bool, backend: TorchBackend):
    jit_g = []

    class TorchCustomFunction(torch.autograd.Function):
        """ See PyTorch_Jit.md """

        @staticmethod
        def forward(ctx, *args, **kwargs):  # The result of this is used in the graph.
            if torch._C._get_tracing_state():
                PHI_LOGGER.debug(f"torch.jit.trace encountered forward pass of {f.__name__}. Returning cached output to avoid double execution.")
                # jit_context = CURRENT_JIT_CALLS[-1]; jit_context.cached_output[torch_custom_function]
                return f_example_output
            y = (jit_f or f)(*args, **kwargs)
            ctx.save_for_backward(*args, *y)
            ctx.input_count = len(args)
            return y

        # @torch.jit.unused, @torch.jit.ignore(drop=True)  do not work here

        @staticmethod
        def backward(ctx, *grad_args):  # Breakpoints not supported here
            x = ctx.saved_tensors[:ctx.input_count]
            y = ctx.saved_tensors[ctx.input_count:]
            if is_f_traced:
                if not jit_g:
                    # backward pass can return None but that's not allowed in JIT functions
                    needs_input_grad = ctx.needs_input_grad
                    none_indices = []  # jit function cannot return None but gradient returns None to indicate there is no gradient

                    def filter_required_grads(*args):  # traced once
                        grads = g(*args)
                        filtered = [gv for gv, need in zip(grads, needs_input_grad) if need]
                        none_indices.clear()
                        none_indices.extend([i for i, g in enumerate(filtered) if g is None])
                        filtered = [gv for gv in filtered if gv is not None]
                        assert len(filtered) > 0, "Custom backward function must return at least one valid gradient."
                        assert all([isinstance(gv, torch.Tensor) for gv in filtered]), [type(gv) for gv in grads]
                        return filtered

                    PHI_LOGGER.debug(f"Tracing backward pass of '{f.__name__}' which uses a custom gradient")
                    needed_g = backend.jit_compile(filter_required_grads)
                    # needed_g = torch.jit.trace(filter_required_grads, tuple([x, y, grad_args]), check_trace=False, strict=False)

                    def g_(*args):  # called each time, not jitted
                        needed = backend.as_registered.call(needed_g, *args, name=f"run jit-compiled custom backward '{g.__name__}'")
                        assert isinstance(needed, (tuple, list))
                        needed = list(needed)
                        for i in none_indices:
                            needed.insert(i, None)
                        result = [(needed.pop(0) if need else None) for need in needs_input_grad]
                        return result

                    jit_g.append(g_)
                output = jit_g[0](x, y, grad_args)
            else:
                output = g(x, y, grad_args)
            result = output[0] if len(output) == 1 else (*output,)
            return result

        @staticmethod
        def compile(args: tuple, output):
            assert jit_f is None
            PHI_LOGGER.debug(f"Tracing forward pass of '{f.__name__}' which uses a custom gradient")
            jit_f_ = torch.jit.trace(f, args, strict=False, check_trace=False)
            return construct_torch_custom_function(f, jit_f_, output, g, is_f_traced=True, backend=backend)

        @property
        def __name__(self):
            return f"TorchCustomFunction-{'jit' if is_f_traced else 'non-jit'}[{f.__name__}]"

        def __repr__(self):
            return self.__name__

    torch_custom_function = TorchCustomFunction()
    return torch_custom_function


def to_torch_dtype(dtype: DType):
    return _TO_TORCH[dtype]


def from_torch_dtype(torch_dtype):
    if torch_dtype in _FROM_TORCH:
        return _FROM_TORCH[torch_dtype]
    else:
        kind = {'i': int, 'b': bool, 'f': float, 'c': complex}[torch_dtype.kind]
        return DType(kind, torch_dtype.itemsize * 8)


_TO_TORCH = {
    DType(float, 16): torch.float16,
    DType(float, 32): torch.float32,
    DType(float, 64): torch.float64,
    DType(complex, 64): torch.complex64,
    DType(complex, 128): torch.complex128,
    DType(int, 8): torch.int8,
    DType(int, 16): torch.int16,
    DType(int, 32): torch.int32,
    DType(int, 64): torch.int64,
    DType(bool): torch.bool,
}
_FROM_TORCH = {np: dtype for dtype, np in _TO_TORCH.items()}


@torch.jit._script_if_tracing
def torch_sparse_cg(lin, y, x0, rtol, atol, max_iter):
    batch_size = y.shape[0]
    tolerance_sq = torch.maximum(rtol ** 2 * torch.sum(y ** 2, -1), atol ** 2)
    x = x0
    dx = residual = y - sparse_matmul(lin, x)
    it_counter = torch.tensor(0, dtype=torch.int32, device=x.device)
    iterations = torch.zeros([batch_size], dtype=torch.int32, device=x.device)
    function_evaluations = torch.ones([batch_size], dtype=torch.int32, device=x.device)
    residual_squared = rsq0 = torch.sum(residual ** 2, -1, keepdim=True)
    diverged = torch.any(~torch.isfinite(x), dim=1)
    converged = torch.all(residual_squared <= tolerance_sq, dim=1)
    finished = converged | diverged | (iterations >= max_iter); not_finished_1 = (~finished).to(torch.int32)
    while ~torch.all(finished):
        it_counter += 1; iterations += not_finished_1
        dy = sparse_matmul(lin, dx); function_evaluations += not_finished_1
        dx_dy = torch.sum(dx * dy, dim=-1, keepdim=True)
        step_size = divide_no_nan(residual_squared, dx_dy)
        step_size *= torch.unsqueeze(not_finished_1.to(y.dtype), -1)  # this is not really necessary but ensures batch-independence
        x += step_size * dx
        if it_counter % 20 == 0:
            residual = y - sparse_matmul(lin, x); function_evaluations += 1
        else:
            residual = residual - step_size * dy  # in-place subtraction affects convergence
        residual_squared_old = residual_squared
        residual_squared = torch.sum(residual ** 2, -1, keepdim=True)
        dx = residual + divide_no_nan(residual_squared, residual_squared_old) * dx
        diverged = torch.any(residual_squared / rsq0 > 100, dim=1) & (iterations >= 8)
        converged = torch.all(residual_squared <= tolerance_sq, dim=1)
        finished = converged | diverged | (iterations >= max_iter); not_finished_1 = (~finished).to(torch.int32)
    return x, residual, iterations, function_evaluations, converged, diverged


@torch.jit._script_if_tracing
def torch_sparse_cg_adaptive(lin, y, x0, rtol, atol, max_iter):
    batch_size = y.shape[0]
    tolerance_sq = torch.maximum(rtol ** 2 * torch.sum(y ** 2, -1), atol ** 2)
    x = x0
    dx = residual = y - sparse_matmul(lin, x)
    it_counter = torch.tensor(0, dtype=torch.int32, device=x.device)
    iterations = torch.zeros([batch_size], dtype=torch.int32, device=x.device)
    function_evaluations = torch.ones([batch_size], dtype=torch.int32, device=x.device)
    residual_squared = rsq0 = torch.sum(residual ** 2, -1, keepdim=True)
    diverged = torch.any(~torch.isfinite(x), dim=1)
    converged = torch.all(residual_squared <= tolerance_sq, dim=1)
    finished = converged | diverged | (iterations >= max_iter); not_finished_1 = (~finished).to(torch.int32)
    while ~torch.all(finished):
        it_counter += 1; iterations += not_finished_1
        dy = sparse_matmul(lin, dx); function_evaluations += not_finished_1
        dx_dy = torch.sum(dx * dy, dim=-1, keepdim=True)
        step_size = divide_no_nan(torch.sum(dx * residual, dim=1, keepdim=True), dx_dy)
        step_size *= torch.unsqueeze(not_finished_1.to(y.dtype), -1)  # this is not really necessary but ensures batch-independence
        x += step_size * dx
        if it_counter % 20 == 0:
            residual = y - sparse_matmul(lin, x); function_evaluations += 1
        else:
            residual = residual - step_size * dy  # in-place subtraction affects convergence
        residual_squared = torch.sum(residual ** 2, -1, keepdim=True)
        dx = residual - divide_no_nan(torch.sum(residual * dy, dim=1, keepdim=True) * dx, dx_dy)
        diverged = torch.any(residual_squared / rsq0 > 100, dim=1) & (iterations >= 8)
        converged = torch.all(residual_squared <= tolerance_sq, dim=1)
        finished = converged | diverged | (iterations >= max_iter); not_finished_1 = (~finished).to(torch.int32)
    return x, residual, iterations, function_evaluations, converged, diverged


def sparse_matmul(matrix: torch.sparse.Tensor, b: torch.Tensor):
    return torch.transpose(torch.sparse.mm(matrix, torch.transpose(b, 0, 1)), 0, 1)


def divide_no_nan(x: torch.Tensor, y: torch.Tensor):
    # --- PyTorch backward pass of where produces nan gradients when inf values are present.
    # Workaround is to avoid zero division by replacing zeros with ones (which then get filtered
    # in the return where). ---
    result = x / torch.where(y == 0, torch.ones_like(y), y)
    result = torch.where(y == 0, torch.zeros_like(result), result)
    return result

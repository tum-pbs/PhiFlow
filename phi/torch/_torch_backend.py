import numbers
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import List, Callable

import numpy as np
import torch
import torch.fft
import torch.nn.functional as torchf

from phi.math import DType
from phi.math.backend import Backend, NUMPY, ComputeDevice
from phi.math.backend._backend import combined_dim, SolveResult


class TorchBackend(Backend):

    def __init__(self):
        cpu = NUMPY.cpu
        self.cpu = ComputeDevice(self, "CPU", 'CPU', cpu.memory, cpu.processor_count, cpu.description, ref='cpu')
        Backend.__init__(self, 'PyTorch', default_device=self.cpu)

    def prefers_channels_last(self) -> bool:
        return False

    def list_devices(self, device_type: str or None = None) -> List[ComputeDevice]:
        devices = []
        if device_type in (None, 'CPU'):
            devices.append(self.cpu)
        if device_type in (None, 'GPU'):
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(ComputeDevice(self,
                                             properties.name,
                                             'GPU',
                                             properties.total_memory,
                                             properties.multi_processor_count,
                                             f"compute capability {properties.major}.{properties.minor}",
                                             ref=f'cuda:{index}'))
        return devices

    def is_tensor(self, x, only_native=False):
        if isinstance(x, (torch.Tensor, JITFunction)):
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
        return tensor

    def auto_cast(self, *tensors) -> list:
        tensors = [t if isinstance(t, (torch.Tensor, numbers.Number, bool)) else self.as_tensor(t, True) for t in tensors]
        return Backend.auto_cast(self, *tensors)

    def is_available(self, tensor) -> bool:
        return torch._C._get_tracing_state() is None  # TODO can we find out whether this tensor specifically is being traced?

    def numpy(self, tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
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

    sqrt = torch.sqrt
    exp = torch.exp
    sin = torch.sin
    cos = torch.cos
    tan = torch.tan
    log = torch.log
    log2 = torch.log2
    log10 = torch.log10
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
        return JITFunction(f)

    def custom_gradient(self, f: Callable, gradient: Callable = None) -> Callable:
        TRACED_F = {}
        TRACED_B = {}

        class TorchFunction(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args, **kwargs):
                current_jit = CURRENT_JIT_CALLS[-1] if CURRENT_JIT_CALLS else None
                if torch._C._get_tracing_state() is not None:
                    assert current_jit, "Detected PyTorch tracing outside Φ-Flow"
                    f_ = f

                    def trace_later():
                        TRACED_F[current_jit] = torch.jit.trace(f, args)  # nested traces not allowed in PyTorch
                    current_jit.post_trace.append(trace_later)
                elif CURRENT_JIT_CALLS:
                    jit_f = TRACED_F[CURRENT_JIT_CALLS[-1]]
                    f_ = jit_f
                else:
                    f_ = f
                y = f_(*args, **kwargs)
                ctx.save_for_backward(*args, *y)
                ctx.input_count = len(args)
                ctx.jit_f = current_jit
                return y

            @staticmethod
            def backward(ctx, *grad_args):
                x = ctx.saved_tensors[:ctx.input_count]
                y = ctx.saved_tensors[ctx.input_count:]
                if torch._C._get_tracing_state() is not None:
                    assert CURRENT_JIT_CALLS
                    raise NotImplementedError()
                    # g_ = gradient
                    #
                    # def trace_later():
                    #     TRACED_F[current_jit] = torch.jit.trace(f, args)  # nested traces not allowed in PyTorch
                elif ctx.jit_f:
                    jit_f = ctx.jit_f
                    if ctx.jit_f in TRACED_B:
                        g_ = TRACED_B[ctx.jit_f]
                    else:
                        # jit-compile the gradient function
                        # jit-functions cannot return None, so we have to filter out non-required gradients
                        needs_input_grad = ctx.needs_input_grad
                        with ctx.jit_f:
                            def filter_required_grads(*args):
                                grads = gradient(*args)
                                filtered = [g for g, need in zip(grads, needs_input_grad) if need]
                                return filtered

                            needed_g = torch.jit.trace(filter_required_grads, tuple([x, y, grad_args]), check_trace=False)

                            def g_(*args):
                                with jit_f:
                                    needed = self.as_registered.call(needed_g, *args, name=f"run jit-compiled custom backward '{gradient.__name__}'")
                                assert not jit_f.post_trace
                                assert isinstance(needed, (tuple, list))
                                needed = list(needed)
                                result = [(needed.pop(0) if need else None) for need in needs_input_grad]
                                return result

                            TRACED_B[ctx.jit_f] = g_
                else:
                    g_ = gradient
                output = g_(x, y, grad_args)
                result = output[0] if len(output) == 1 else (*output, )
                return result

        return TorchFunction.apply

    def transpose(self, tensor, axes):
        return tensor.permute(axes)

    def equal(self, x, y):
        x, y = self.auto_cast(x, y)
        return x == y

    def random_uniform(self, shape):
        return torch.rand(size=shape, dtype=to_torch_dtype(self.float_type), device=self.get_default_device().ref)

    def random_normal(self, shape):
        return torch.randn(size=shape, dtype=to_torch_dtype(self.float_type), device=self.get_default_device().ref)

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
            result = torchf.pad(reordered, pad_width_spatial, mode, value=constant_values)  # supports 3D to 5D (2 + 1D to 3D)
        except RuntimeError as err:
            warnings.warn(f"PyTorch error {err}")
            return NotImplemented
        result = undo_transform(result)
        return result

    def grid_sample(self, grid, coordinates, extrapolation='constant'):
        assert extrapolation in ('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect'), extrapolation
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
        if axis is None:
            return torch.any(boolean_tensor)
        else:
            axes = axis if isinstance(axis, (tuple, list)) else [axis]
            for axis in reversed(sorted(axes)):
                boolean_tensor = torch.any(boolean_tensor, dim=axis, keepdim=keepdims)
            return boolean_tensor

    def all(self, boolean_tensor, axis=None, keepdims=False):
        boolean_tensor = self.as_tensor(boolean_tensor, convert_external=True)
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
        return torch.where(condition, x, y)

    def mean(self, value, axis=None, keepdims=False):
        return torch.mean(value, dim=axis, keepdim=keepdims)

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
        if limit is None:
            start, limit = 0, start
        return torch.arange(start, limit, delta, dtype=to_torch_dtype(dtype))

    def zeros(self, shape, dtype=None):
        return torch.zeros(shape, dtype=to_torch_dtype(dtype or self.float_type), device=self.get_default_device().ref)

    def zeros_like(self, tensor):
        return torch.zeros_like(tensor, device=self.get_default_device().ref)

    def ones(self, shape, dtype: DType = None):
        return torch.ones(shape, dtype=to_torch_dtype(dtype or self.float_type), device=self.get_default_device().ref)

    def ones_like(self, tensor):
        return torch.ones_like(tensor, device=self.get_default_device().ref)

    def meshgrid(self, *coordinates):
        coordinates = [self.as_tensor(c) for c in coordinates]
        return torch.meshgrid(coordinates)

    def linspace(self, start, stop, number):
        return torch.linspace(start, stop, number, dtype=to_torch_dtype(self.float_type), device=self.get_default_device().ref)

    def tensordot(self, a, a_axes: tuple or list, b, b_axes: tuple or list):
        return torch.tensordot(a, b, (a_axes, b_axes))

    def matmul(self, A, b):
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
                warnings.warn("Tracing a PyTorch while loop requires an additional tracing pass. You can avoid this by passing a torch.ScriptFunction.")
                raise NotImplementedError()
                # def trace_later():
                #     jit_loop = torch.jit.trace(loop)
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
        return torch.std(x, dim=axis, keepdim=keepdims)

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
            return x  # Creating a Tensor here would raise warnings during tracing.
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

    def sparse_tensor(self, indices, values, shape):
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

    def functional_gradient(self, f, wrt: tuple or list, get_output: bool):
        @wraps(f)
        def eval_grad(*args):
            args = [self.as_tensor(arg, True) if i in wrt else arg for i, arg in enumerate(args)]
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
            output = f(*args)
            loss, aux = (output[0], output[1:]) if isinstance(output, (tuple, list)) else (output, None)
            if loss.ndim > 0:
                loss = loss.sum()
            grads = torch.autograd.grad(loss, wrt_args)  # grad() cannot be called during jit trace
            if get_output:
                loss = loss.detach()
                if aux is not None:
                    aux = [aux_.detach() for aux_ in aux]
                    return (loss, *aux, *grads)
                else:
                    return (loss, *grads)
            else:
                return grads
        return eval_grad

    def jit_compile_grad(self, f, wrt: tuple or list, get_output: bool):
        jit = self.jit_compile(f)
        return self.functional_gradient(jit, wrt, get_output)

    def gradients(self, y, xs: tuple or list, grad_y) -> tuple:
        if self.ndims(y) > 0:
            y = self.sum(y)
        grad = torch.autograd.grad(y, xs, grad_y)
        return grad

    @contextmanager
    def record_gradients(self, xs: tuple or list, persistent=False):
        for x in xs:
            assert self.is_tensor(x, only_native=True), f"Must be a PyTorch tensor but got {x}"
        xs = [x if x.is_leaf else x.detach_() for x in xs]
        assert not any(x.requires_grad for x in xs)
        for x in xs:
            x.requires_grad = True
        try:
            yield None
        finally:
            for x in xs:
                x.requires_grad = False

    def stop_gradient(self, value):
        return value.detach()


def channels_first(x):
    return x.permute(*((0, -1) + tuple(range(1, len(x.shape) - 1))))


def channels_last(x):
    return x.permute((0,) + tuple(range(2, len(x.shape))) + (1,))


class JITFunction:

    def __init__(self, f):
        self.f = f
        self.traced = None
        self.post_trace = []

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise NotImplementedError("kwargs not supported for traced function")
        if self.traced is None:
            with self:
                self.traced = torch.jit.trace(self.f, example_inputs=args, check_trace=False)
        with self:
            from phi.math.backend import choose_backend
            return choose_backend(self).call(self.traced, *args, name=f"run jit-compiled '{self.f.__name__}'")

    def __repr__(self):
        return f"jit-TorchScript[{self.f.__name__}]"

    def __enter__(self):
        CURRENT_JIT_CALLS.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert CURRENT_JIT_CALLS.pop(-1) == self
        while self.post_trace:
            self.post_trace.pop(0)()



CURRENT_JIT_CALLS: List[JITFunction] = []


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

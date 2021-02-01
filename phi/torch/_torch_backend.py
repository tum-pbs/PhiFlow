import numbers
import time
import warnings
from typing import Tuple, List

import numpy as np

import torch
import torch.fft
import torch.nn.functional as torchf

from phi.math.backend import Backend, DType, SCIPY_BACKEND, ComputeDevice
from phi.math.backend._backend_helper import combined_dim


class TorchBackend(Backend):

    def __init__(self):
        Backend.__init__(self, 'PyTorch')

    def list_devices(self, device_type: str or None = None) -> List[ComputeDevice]:
        devices = []
        if device_type in (None, 'CPU'):
            devices.extend(SCIPY_BACKEND.list_devices(device_type='CPU'))
        if device_type in (None, 'GPU'):
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(ComputeDevice(properties.name,
                                          'GPU',
                                          properties.total_memory,
                                          properties.multi_processor_count,
                                          f"major={properties.major}, minor={properties.minor}"))
        return devices

    def auto_cast(self, *tensors):
        """
        Determins the appropriate values type resulting from operations involving the tensors as input.
        
        This method is called by the default implementations of basic operators.
        Backends can override this method to prevent unnecessary casting.

        Args:
          *tensors: 

        Returns:

        """
        dtypes = [self.dtype(t) for t in tensors]
        result_type = self.combine_types(*dtypes)
        tensors = [self.cast(t, result_type) for t in tensors]
        return tensors

    def is_tensor(self, x, only_native=False):
        if isinstance(x, torch.Tensor):
            return True
        if only_native:
            return False
        if isinstance(x, numbers.Number):
            return True
        if isinstance(x, (tuple, list)) and all(isinstance(c, numbers.Number) for c in x):
            return True
        return False

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            tensor = x
        elif isinstance(x, np.ndarray):
            tensor = torch.from_numpy(SCIPY_BACKEND.as_tensor(x))
        elif isinstance(x, (tuple, list)):
            try:
                tensor = torch.tensor(x)
            except ValueError:  # there may be Tensors inside the list
                components = [self.as_tensor(c) for c in x]
                tensor = torch.stack(components, dim=0)
        else:
            tensor = torch.tensor(x)
        # --- Enforce Precision ---
        if self.is_tensor(tensor, only_native=True):
            if tensor.dtype.is_floating_point:
                tensor = self.to_float(tensor)
        return tensor

    def is_available(self, tensor) -> bool:
        return True  # ToDo may require different handling for TorchScript

    def numpy(self, tensor):
        return tensor.numpy()

    def copy(self, tensor, only_mutable=False):
        return torch.clone(tensor)

    def transpose(self, tensor, axes):
        return tensor.permute(axes)

    def equal(self, x, y):
        return x == y

    def random_uniform(self, shape):
        return torch.rand(size=shape, dtype=to_torch_dtype(self.float_type))

    def random_normal(self, shape):
        return torch.randn(size=shape, dtype=to_torch_dtype(self.float_type))

    def stack(self, values, axis=0):
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

    def _single_mode_single_constant_pad(self, value, pad_width, single_mode, constant_value=0):
        assert single_mode in ('constant', 'symmetric', 'circular', 'reflect', 'replicate'), single_mode
        if single_mode == 'constant':
            pad = sum(pad_width[::-1], [] if isinstance(pad_width, list) else ())
            return torchf.pad(value, pad, mode='constant', value=constant_value)
        if single_mode == 'symmetric':
            if np.any(np.array(pad_width) > 1):
                return symmetric_pad(value, pad_width, self)
            else:
                single_mode = 'replicate'
        value = channels_first(value)
        reversed_axis_pad = pad_width[1:-1][::-1]
        pad = sum(reversed_axis_pad, [] if isinstance(pad_width, list) else ())
        result = torchf.pad(value, pad, mode=single_mode, value=constant_value)  # reflect, replicate, circular (constant handled above)
        result = channels_last(result)
        return result

    def grid_sample(self, grid, spatial_dims: tuple, coordinates, extrapolation='constant'):
        assert extrapolation in ('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect'), extrapolation
        extrapolation = {'undefined': 'zeros', 'zeros': 'zeros', 'boundary': 'border', 'reflect': 'reflection'}.get(extrapolation, None)
        if not extrapolation:
            return NotImplemented
        grid = channels_first(self.as_tensor(grid))
        coordinates = self.as_tensor(coordinates)
        resolution = torch.Tensor(self.staticshape(grid)[2:])
        coordinates = 2 * coordinates / (resolution - 1) - 1
        coordinates = torch.flip(coordinates, dims=[-1])
        result = torchf.grid_sample(grid, coordinates, mode='bilinear', padding_mode=extrapolation, align_corners=True)  # can cause segmentation violation if NaN or inf are present
        result = channels_last(result)
        return result

    def reshape(self, value, shape):
        # if not value.is_complex():
        #     value = value.view_as_complex()
        return torch.reshape(value, shape)

    def flip(self, value, axes: tuple or list):
        return torch.flip(value, axes)

    def sum(self, value, axis=None, keepdims=False):
        value = self.as_tensor(value)
        if axis is None:
            axis = tuple(range(len(value.shape)))
        return torch.sum(value, dim=axis, keepdim=keepdims)

    def prod(self, value, axis=None):
        if isinstance(axis, (tuple, list)):
            for dim in axis:
                value = torch.prod(value, dim=dim)
            return value
        return torch.prod(value, dim=axis)

    def divide_no_nan(self, x, y):
        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(y)
        result = self.as_tensor(x) / self.as_tensor(y)
        return torch.where(y == 0, torch.zeros_like(result), result)

    def where(self, condition, x=None, y=None):
        condition = self.as_tensor(condition).bool()
        x = self.as_tensor(x)
        y = self.as_tensor(y)
        return torch.where(condition, x, y)

    def mean(self, value, axis=None, keepdims=False):
        return torch.mean(value, dim=axis, keepdim=keepdims)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        raise NotImplementedError()

    def range(self, start, limit=None, delta=1, dtype=None):
        if limit is None:
            start, limit = 0, start
        if dtype is None:
            dtype = torch.int32
        elif not isinstance(dtype, torch.dtype):
            dtype = self.translate_dtype(dtype)
        return torch.arange(start, limit, delta, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return torch.zeros(shape, dtype=to_torch_dtype(dtype or self.float_type))

    def zeros_like(self, tensor):
        return torch.zeros_like(tensor)

    def ones(self, shape, dtype: DType = None):
        return torch.ones(shape, dtype=to_torch_dtype(dtype or self.float_type))

    def ones_like(self, tensor):
        return torch.ones_like(tensor)

    def meshgrid(self, *coordinates):
        coordinates = [self.as_tensor(c) for c in coordinates]
        return torch.meshgrid(coordinates)

    def linspace(self, start, stop, number):
        return torch.linspace(start, stop, number, dtype=to_torch_dtype(self.float_type))

    def dot(self, a, b, axes):
        return torch.tensordot(a, b, axes)

    def matmul(self, A, b):
        if isinstance(A, torch.sparse.FloatTensor):
            result = torch.sparse.mm(A, torch.transpose(b, 0, 1))
            return torch.transpose(result, 0, 1)
        raise NotImplementedError()

    def einsum(self, equation, *tensors):
        return torch.einsum(equation, *tensors)

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None, maximum_iterations=None):
        i = 0
        while cond(*loop_vars):
            if maximum_iterations is not None and i == maximum_iterations:
                break
            loop_vars = body(*loop_vars)
            i += 1
        return loop_vars

    def abs(self, x):
        return torch.abs(x)

    def sign(self, x):
        return torch.sign(x)

    def round(self, x):
        return torch.round(x)

    def ceil(self, x):
        return torch.ceil(x)

    def floor(self, x):
        return torch.floor(x)

    def max(self, x, axis=None, keepdims=False):
        if isinstance(x, (tuple, list)):
            x = torch.stack(x)
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
        if isinstance(x, (tuple, list)):
            x = torch.stack(x)
        if axis is None:
            result = torch.min(x, keepdim=keepdims)
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

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
        return function(*inputs)  # ToDo

    def sqrt(self, x):
        return torch.sqrt(x)

    def exp(self, x):
        return torch.exp(x)

    def conv(self, tensor, kernel, padding='same'):
        tensor = self.as_tensor(tensor)
        kernel = self.as_tensor(kernel)
        if padding.lower() == 'valid':
            padding = 0
        elif padding.lower() == 'same':
            shape = kernel.shape
            padding = sum([[d // 2, (d + 1) // 2] for d in shape], [])
        else:
            raise ValueError(padding)
        tensor = channels_first(tensor)
        kernel = kernel.permute((-2, -1) + tuple(range(len(kernel.shape) - 2)))
        convf = {3: torchf.conv1d, 4: torchf.conv2d, 5: torchf.conv3d}[len(tensor.shape)]
        result = convf(tensor, kernel, padding=padding)
        result = channels_last(result)
        return result

    def expand_dims(self, a, axis=0, number=1):
        for _ in range(number):
            a = torch.unsqueeze(a, dim=axis)
        return a

    def shape(self, tensor):
        return tensor.shape

    def staticshape(self, tensor):
        return tuple(tensor.shape)

    def to_float(self, x):
        if not self.is_tensor(x):
            x = self.as_tensor(x)
        elif self.precision == 16:
            return x.half()
        elif self.precision == 32:
            return x.float()
        elif self.precision == 64:
            return x.double()
        else:
            raise AssertionError(self.precision)

    def to_int(self, x, int64=False):
        x = self.as_tensor(x)
        return x.int()

    def to_complex(self, real, imag=None):
        if real.is_complex():
            return real
        if imag is None:
            imag = torch.zeros_like(real)
        return torch.complex(real, imag)

    def gather(self, values, indices):
        # return torch.gather(values, dim=0, index=indices)
        raise NotImplementedError()

    def gather_nd(self, values, indices, batch_dims=0):
        values = self.as_tensor(values)
        indices = self.as_tensor(indices).long()
        if batch_dims == 0:
            dim_indices = self.unstack(indices, axis=-1)
            result = values[list(dim_indices)]
        elif batch_dims == 1:
            batch_size = combined_dim(values.shape[0], indices.shape[0])
            result = []
            for i in range(batch_size):
                dim_indices = self.unstack(indices[i], axis=-1)
                result.append(values[[i] + list(dim_indices)])
            result = self.stack(result, axis=0)
        else:
            raise NotImplementedError("Only batch_dims <= 1 are supported.")
        return result

    def unstack(self, tensor, axis=0, keepdims=False):
        unstacked = torch.unbind(tensor, dim=axis)
        if keepdims:
            unstacked = [self.expand_dims(c, axis=axis) for c in unstacked]
        return unstacked

    def std(self, x, axis=None, keepdims=False):
        return torch.std(x, dim=axis, keepdim=keepdims)

    def boolean_mask(self, x, mask):
        return torch.masked_select(x, mask)

    def isfinite(self, x):
        return torch.isfinite(x)

    def scatter(self, points, indices, values, shape, duplicates_handling='undefined'):
        raise NotImplementedError()

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return torch.any(boolean_tensor, dim=axis, keepdim=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        if axis is None:
            return torch.all(boolean_tensor)
        return torch.all(boolean_tensor, dim=axis, keepdim=keepdims)

    def fft(self, x):
        if not x.is_complex():
            x = self.to_complex(x)
        rank = len(x.shape) - 2
        x = channels_first(x)
        x = torch.view_as_real(x)
        k = torch.fft.fft(x, rank)
        if k.is_complex():
            k = self.real(k).contiguous()
        k = torch.view_as_complex(k)
        k = channels_last(k)
        return k

    def ifft(self, k):
        if not k.is_complex():
            k = self.to_complex(k)
        rank = len(k.shape) - 2
        k = channels_first(k)
        x = torch.fft.ifft(torch.view_as_real(k), rank)
        if x.is_complex():
            x = self.real(x).contiguous()
        x = torch.view_as_complex(x)
        x = channels_last(x)
        return x

    def imag(self, complex):
        if isinstance(complex, torch.Tensor):
            return complex.imag
        else:
            if isinstance(complex, np.ndarray):
                complex = np.imag(complex)
            return torch.zeros_like(self.as_tensor(complex))

    def real(self, complex):
        if isinstance(complex, torch.Tensor):
            return complex.real
        else:
            if isinstance(complex, np.ndarray):
                complex = np.real(complex)
            return self.as_tensor(complex)

    def cast(self, x, dtype: DType):
        x = self.as_tensor(x)
        return x.to(to_torch_dtype(dtype))

    def sin(self, x):
        return torch.sin(x)

    def cos(self, x):
        return torch.cos(x)

    def dtype(self, array) -> DType:
        if self.is_tensor(array, only_native=True):
            return from_torch_dtype(array.dtype)
        else:
            return SCIPY_BACKEND.dtype(array)

    def tile(self, value, multiples):
        if isinstance(multiples, np.ndarray):
            multiples = multiples.tolist()
        return self.as_tensor(value).repeat(multiples)

    def sparse_tensor(self, indices, values, shape):
        indices_ = torch.LongTensor(indices)
        values_ = torch.FloatTensor(values)
        result = torch.sparse.FloatTensor(indices_, values_, shape)
        return result

    def conjugate_gradient(self, A, y, x0,
                           relative_tolerance: float = 1e-5,
                           absolute_tolerance: float = 0.0,
                           max_iterations: int = 1000,
                           gradient: str = 'implicit',
                           callback=None):
        if callable(A):
            function = A
        else:
            A = self.as_tensor(A)
            A_shape = self.staticshape(A)
            assert len(A_shape) == 2, f"A must be a square matrix but got shape {A_shape}"
            assert A_shape[0] == A_shape[1], f"A must be a square matrix but got shape {A_shape}"

            def function(vec):
                return self.matmul(A, vec)

        # from SciPy: max(float(atol), tol * float(bnrm2))
        tolerance_sq = self.maximum(relative_tolerance ** 2 * torch.sum(y ** 2, -1), absolute_tolerance ** 2)

        y = self.to_float(y)
        x = self.to_float(x0)
        batch_size = combined_dim(x.shape[0], y.shape[0])
        if x.shape[0] < batch_size:
            x = x.repeat([batch_size, 1])

        dx = residual = y - function(x)
        dy = function(dx)
        iterations = 0
        converged = True
        while self.all(self.sum(residual ** 2, -1) > tolerance_sq):
            if iterations == max_iterations:
                converged = False
                break
            iterations += 1
            dx_dy = self.sum(dx * dy, axis=-1, keepdims=True)
            step_size = self.divide_no_nan(self.sum(dx * residual, axis=-1, keepdims=True), dx_dy)
            x += step_size * dx
            residual -= step_size * dy
            dx = residual - self.divide_no_nan(self.sum(residual * dy, axis=-1, keepdims=True) * dx, dx_dy)
            dy = function(dx)
        return converged, x, iterations


TORCH_BACKEND = TorchBackend()


def channels_first(x):
    return x.permute(*((0, -1) + tuple(range(1, len(x.shape) - 1))))


def channels_last(x):
    return x.permute((0,) + tuple(range(2, len(x.shape))) + (1,))


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


def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves a batch of matrix linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.
    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


class CG(torch.autograd.Function):

    def __init__(self, A_bmm, M_bmm=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        self.A_bmm = A_bmm
        self.M_bmm = M_bmm
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose

    def forward(self, B, X0=None):
        X, _ = cg_batch(self.A_bmm, B, M_bmm=self.M_bmm, X0=X0, rtol=self.rtol,
                     atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return X

    def backward(self, dX):
        dB, _ = cg_batch(self.A_bmm, dX, M_bmm=self.M_bmm, rtol=self.rtol,
                      atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return dB
import numbers
import warnings

import numpy as np

import torch
import torch.nn.functional as torchf

from phi.backend.backend import Backend
from phi.backend.backend_helper import split_multi_mode_pad, PadSettings, general_grid_sample_nd, combined_dim, symmetric_pad
from phi.backend.scipy_backend import SciPyBackend


class TorchBackend(Backend):

    def __init__(self):
        Backend.__init__(self, 'PyTorch')

    @property
    def precision_dtype(self):
        return {16: torch.float16, 32: torch.float32, 64: torch.float64, None: torch.float32}[self.precision]

    def is_tensor(self, x, only_native=False):
        if not only_native and isinstance(x, numbers.Number):
            return True
        return isinstance(x, (torch.Tensor, ComplexTensor))

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            tensor = x
        elif isinstance(x, np.ndarray):
            tensor = torch.from_numpy(SciPyBackend(precision=self.precision).as_tensor(x))
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
            if tensor.dtype.is_floating_point and self.has_fixed_precision:
                tensor = self.to_float(tensor)
        return tensor

    def copy(self, tensor, only_mutable=False):
        return torch.clone(tensor)

    def equal(self, x, y):
        return x == y

    def random_uniform(self, shape):
        return torch.rand(size=shape, dtype=self.precision_dtype)

    def random_normal(self, shape):
        return torch.randn(size=shape, dtype=self.precision_dtype)

    def stack(self, values, axis=0):
        return torch.stack(values, dim=axis)

    def concat(self, values, axis):
        return torch.cat(values, dim=axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        passes = split_multi_mode_pad(self.ndims(value), PadSettings(pad_width, mode, constant_values), split_by_constant_value=True)
        for pad_pass in passes:
            value = self._single_mode_single_constant_pad(value, *pad_pass)
        return value

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

    def resample(self, inputs, sample_coords, interpolation='linear', boundary='constant', constant_values=0):
        assert interpolation == 'linear'
        assert constant_values == 0
        return general_grid_sample_nd(inputs, sample_coords, boundary, constant_values, self)
        # return self._native_resample(inputs, sample_coords, interpolation, boundary)

    def _native_resample(self, inputs, sample_coords, interpolation='linear', boundary='constant'):
        """ Around 5% faster than general_grid_sample_nd on the CPU. Does not support multi-boundary resampling or constant values. """
        inputs = channels_first(self.as_tensor(inputs))
        sample_coords = self.as_tensor(sample_coords)
        # --- Interpolation ---
        if interpolation.lower() == 'linear':
            interpolation = 'bilinear'
        elif interpolation.lower() == 'nearest':
            interpolation = 'nearest'
        else:
            raise NotImplementedError(interpolation)
        # --- Boundary ---
        if boundary == 'zero' or boundary == 'constant':
            boundary = 'zeros'
        elif boundary == 'replicate':
            boundary = 'border'
        elif boundary == 'circular':
            shape = self.to_float(inputs.shape[2:])
            sample_coords = torch.fmod(sample_coords, shape)
            inputs = torchf.pad(inputs, [0, 1] * (len(inputs.shape)-2), mode='circular')
            boundary = 'zeros'
        else:
            raise NotImplementedError(boundary)
        resolution = torch.Tensor(self.staticshape(inputs)[2:])
        sample_coords = 2 * sample_coords / (resolution-1) - 1
        sample_coords = torch.flip(sample_coords, dims=[-1])
        result = torchf.grid_sample(inputs, sample_coords, mode=interpolation, padding_mode=boundary, align_corners=True)  # can cause segmentation violation if NaN or inf are present
        result = channels_last(result)
        return result

    def reshape(self, value, shape):
        return torch.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        value = self.as_tensor(value)
        if axis is None:
            axis = tuple(range(len(value.shape)))
        return torch.sum(value, dim=axis, keepdim=keepdims)

    def prod(self, value, axis=None):
        return torch.prod(value, dim=axis)

    def divide_no_nan(self, x, y):
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
        return torch.arange(start, limit, delta, dtype=dtype)

    def zeros_like(self, tensor):
        return torch.zeros_like(tensor)

    def ones_like(self, tensor):
        return torch.ones_like(tensor)

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
            if maximum_iterations is not None and i == maximum_iterations: break
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
        if axis is None:
            result = torch.max(x)
            if keepdims:
                result = self.expand_dims(result, axis=0, number=self.ndims(x))
            return result
        return torch.max(x, dim=axis, keepdim=keepdims)

    def min(self, x, axis=None, keepdims=False):
        if axis is None:
            result = torch.min(x, keepdim=keepdims)
            if keepdims:
                result = self.expand_dims(result, axis=0, number=self.ndims(x))
            return result
        return torch.min(x, dim=axis, keepdim=keepdims)

    def maximum(self, a, b):
        a_ = self.as_tensor(a)
        b_ = self.as_tensor(b).to(a_.dtype)
        return torch.max(a_, other=b_)

    def minimum(self, a, b):
        a_ = self.as_tensor(a)
        b_ = self.as_tensor(b).to(a_.dtype)
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
            padding = sum([[d//2, (d+1)//2] for d in shape], [])
        else:
            raise ValueError(padding)
        tensor = channels_first(tensor)
        kernel = kernel.permute((-2, -1) + tuple(range(len(kernel.shape)-2)))
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

    def to_float(self, x, float64=False):
        if not self.is_tensor(x):
            x = self.as_tensor(x)
        if float64:
            warnings.warn('float64 argument is deprecated, set Backend.precision = 64 to use 64 bit operations.', DeprecationWarning)
            return x.double()
        else:
            if not self.has_fixed_precision:
                return x if x.dtype.is_floating else x.float()
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

    def to_complex(self, x):
        if isinstance(x, ComplexTensor):
            return x
        x = self.as_tensor(x)
        return ComplexTensor(self.stack([x, torch.zeros_like(x)], -1))

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
            batch_size = combined_dim(self.staticshape(values)[0], self.staticshape(indices)[0])
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
        torch.std(x, dim=axis, keepdim=keepdims)

    def boolean_mask(self, x, mask):
        return torch.masked_select(x, mask)

    def isfinite(self, x):
        return torch.isfinite(x)

    def scatter(self, points, indices, values, shape, duplicates_handling='undefined'):
        raise NotImplementedError()

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return torch.any(boolean_tensor, dim=axis, keepdim=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return torch.all(boolean_tensor, dim=axis, keepdim=keepdims)

    def fft(self, x):
        if not isinstance(x, ComplexTensor):
            x = self.to_complex(x)
        rank = len(x.shape) - 2
        x = channels_first(x).tensor
        k = torch.fft(x, rank)
        k = ComplexTensor(k)
        k = channels_last(k)
        return k

    def ifft(self, k):
        if not isinstance(k, ComplexTensor):
            k = self.to_complex(k)
        rank = len(k.shape) - 2
        k = channels_first(k)
        x = torch.ifft(k.tensor, rank)
        x = ComplexTensor(x)
        x = channels_last(x)
        return x

    def imag(self, complex):
        if isinstance(complex, ComplexTensor):
            return complex.imag
        else:
            if isinstance(complex, np.ndarray):
                complex = np.imag(complex)
            return torch.zeros_like(self.as_tensor(complex))

    def real(self, complex):
        if isinstance(complex, ComplexTensor):
            return complex.real
        else:
            if isinstance(complex, np.ndarray):
                complex = np.real(complex)
            return self.as_tensor(complex)

    def cast(self, x, dtype):
        if not isinstance(dtype, torch.dtype):
            dtype = {np.float16: torch.float16, np.float32: torch.float32, np.float64: torch.float64, np.bool: torch.bool, np.int8: torch.int8, np.int16: torch.int16, np.int32: torch.int32, np.int64: torch.int64}[dtype]
        x = self.as_tensor(x)
        return x.to(dtype)

    def sin(self, x):
        return torch.sin(x)

    def cos(self, x):
        return torch.cos(x)

    def dtype(self, array):
        return array.dtype

    def tile(self, value, multiples):
        if isinstance(multiples, np.ndarray):
            multiples = multiples.tolist()
        return self.as_tensor(value).repeat(multiples)

    def sparse_tensor(self, indices, values, shape):
        indices_ = torch.transpose(torch.LongTensor(indices), 0, 1)
        values_ = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(indices_, values_, shape)


def channels_first(x):
    if isinstance(x, ComplexTensor):
        x = x.tensor
        y = x.permute(*((0, -2) + tuple(range(1, len(x.shape) - 2)) + (-1,)))
        return ComplexTensor(y)
    else:
        return x.permute(*((0, -1) + tuple(range(1, len(x.shape) - 1))))


def channels_last(x):
    if isinstance(x, ComplexTensor):
        x = x.tensor
        x = x.permute((0,) + tuple(range(2, len(x.shape)-1)) + (1, -1))
        return ComplexTensor(x)
    else:
        return x.permute((0,) + tuple(range(2, len(x.shape))) + (1,))


class ComplexTensor(object):

    def __init__(self, tensor):
        self.tensor = tensor

    @property
    def shape(self):
        return self.tensor.shape[:-1]

    @property
    def real(self):
        return self.tensor[...,0]

    @property
    def imag(self):
        return self.tensor[...,1]

    def __mul__(self, other):
        math = TorchBackend()
        real = self.real * math.real(other) - self.imag * math.imag(other)
        imag = self.real * math.imag(other) + self.imag * math.real(other)
        result = math.stack([real, imag], -1)
        return ComplexTensor(result)

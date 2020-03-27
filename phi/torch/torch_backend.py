import warnings

import numpy as np

import torch
import torch.nn.functional as torchf

from phi.backend.backend import Backend


class TorchBackend(Backend):

    def __init__(self):
        Backend.__init__(self, 'PyTorch')

    def is_tensor(self, x):
        return isinstance(x, (torch.Tensor, ComplexTensor))

    def as_tensor(self, x):
        if self.is_tensor(x):
            return x
        if isinstance(x, np.ndarray):
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            return torch.from_numpy(x)
        if isinstance(x, (tuple, list)):
            try:
                return torch.tensor(x)
            except ValueError:  # there may be Tensors inside the list
                components = [self.as_tensor(c) for c in x]
                return torch.stack(components, dim=0)
        return torch.tensor(x)

    def copy(self, tensor, only_mutable=False):
        return torch.clone(tensor)

    def equal(self, x, y):
        return x == y

    def random_uniform(self, shape):
        return torch.rand(shape)

    def stack(self, values, axis=0):
        return torch.stack(values, dim=axis)

    def concat(self, values, axis):
        return torch.cat(values, dim=axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        mode = mode.lower()
        if mode == 'wrap':
            warnings.warn("'wrap' is deprecated, use 'circular' instead", DeprecationWarning, stacklevel=2)
            mode = 'circular'
        if mode == 'constant':
            pad = sum(pad_width[::-1], [] if isinstance(pad_width, list) else ())
            return torchf.pad(value, pad, mode=mode, value=constant_values)  # constant, reflect, replicate, circular
        if mode == 'symmetric':
            warnings.warn("mode 'symmetric' is not supported by PyTorch. Defaults to 'replicate'.")
            mode = 'replicate'
        value = channels_first(value)
        reversed_axis_pad = pad_width[1:-1][::-1]
        pad = sum(reversed_axis_pad, [] if isinstance(pad_width, list) else ())
        result = torchf.pad(value, pad, mode=mode, value=constant_values)  # constant, reflect, replicate, circular
        result = channels_last(result)
        return result

    def reshape(self, value, shape):
        return torch.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        value = self.as_tensor(value)
        if axis is None:
            axis = range(len(value.shape))
        return torch.sum(value, dim=axis, keepdim=keepdims)

    def prod(self, value, axis=None):
        return torch.prod(value, dim=axis)

    def divide_no_nan(self, x, y):
        result = self.as_tensor(x) / self.as_tensor(y)
        return torch.where(y == 0, torch.zeros_like(result), result)

    def where(self, condition, x=None, y=None):
        return torch.where(condition, x, y)

    def mean(self, value, axis=None, keepdims=False):
        return torch.mean(value, dim=axis, keepdim=keepdims)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        raise NotImplementedError()

    def resample(self, inputs, sample_coords, interpolation='linear', boundary='constant'):
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
        result = torchf.grid_sample(inputs, sample_coords, mode=interpolation, padding_mode=boundary)  # can cause segmentation violation if NaN or inf are present
        result = channels_last(result)
        return result

    def range(self, start, limit=None, delta=1, dtype=None):
        raise NotImplementedError()

    def zeros_like(self, tensor):
        return torch.zeros_like(tensor)

    def ones_like(self, tensor):
        return torch.ones_like(tensor)

    def dot(self, a, b, axes):
        raise NotImplementedError()

    def matmul(self, A, b):
        if isinstance(A, torch.sparse.FloatTensor):
            result = torch.sparse.mm(A, torch.transpose(b, 0, 1))
            return torch.transpose(result, 0, 1)
        raise NotImplementedError()

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

    def max(self, x, axis=None):
        if axis is None:
            return torch.max(x)
        return torch.max(x, dim=axis)

    def min(self, x, axis=None):
        if axis is None:
            return torch.min(x)
        return torch.min(x, dim=axis)

    def maximum(self, a, b):
        b = self.as_tensor(b)
        return torch.max(a, other=b)

    def minimum(self, a, b):
        return torch.min(a, other=b)

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

    def to_float(self, x):
        x = self.as_tensor(x)
        return x.float()

    def to_int(self, x, int64=False):
        x = self.as_tensor(x)
        return x.int()

    def to_complex(self, x):
        x = self.as_tensor(x)
        return ComplexTensor(self.stack([x, torch.zeros_like(x)], -1))

    def gather(self, values, indices):
        raise NotImplementedError()

    def gather_nd(self, values, indices):
        raise NotImplementedError()

    def unstack(self, tensor, axis=0, keepdims=False):
        unstacked = torch.unbind(tensor, dim=axis)
        if keepdims:
            unstacked = [self.expand_dims(c, axis=axis) for c in unstacked]
        return unstacked

    def std(self, x, axis=None, keepdims=False):
        raise NotImplementedError()

    def boolean_mask(self, x, mask):
        raise NotImplementedError()

    def isfinite(self, x):
        raise NotImplementedError()

    def scatter(self, points, indices, values, shape, duplicates_handling='undefined'):
        raise NotImplementedError()

    def any(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError()

    def all(self, boolean_tensor, axis=None, keepdims=False):
        raise NotImplementedError()

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
        if dtype == np.float32:
            return self.to_float(x)
        if dtype == np.int32:
            return self.to_int(x)
        if dtype == np.int64:
            return self.to_int(x, int64=True)
        if dtype == np.complex64:
            return self.to_complex(x)
        raise NotImplementedError()

    def sin(self, x):
        return torch.sin(x)

    def cos(self, x):
        return torch.cos(x)

    def dtype(self, array):
        return array.dtype

    def tile(self, value, multiples):
        raise NotImplementedError()

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

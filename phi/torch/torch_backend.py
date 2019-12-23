import numpy as np

import torch
import torch.nn.functional as torchf

from phi.math.base_backend import Backend


class TorchBackend(Backend):

    def __init__(self):
        Backend.__init__(self, 'PyTorch')

    def is_tensor(self, x):
        return isinstance(x, torch.Tensor)

    def as_tensor(self, x):
        if self.is_tensor(x):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            return torch.Tensor(x)

    def equal(self, x, y):
        return x == y

    def random_uniform(self, shape):
        return torch.rand(shape)

    def stack(self, values, axis=0):
        return torch.stack(values, dim=axis)

    def concat(self, values, axis):
        return torch.cat(values, dim=axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        pad = sum(pad_width, [] if isinstance(pad_width, list) else ())
        return torchf.pad(value, pad, mode=mode, value=constant_values)

    def reshape(self, value, shape):
        return torch.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        return torch.sum(value, dim=axis, keepdim=keepdims)

    def prod(self, value, axis=None):
        return torch.prod(value, dim=axis)

    def divide_no_nan(self, x, y):
        raise NotImplementedError()

    def where(self, condition, x=None, y=None):
        return torch.where(condition, x, y)

    def mean(self, value, axis=None):
        raise NotImplementedError()

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        raise NotImplementedError()

    def resample(self, inputs, sample_coords, interpolation='linear', boundary='zero'):
        if interpolation.lower() == 'linear':
            interpolation = 'bilinear'
        elif interpolation.lower() == 'nearest':
            interpolation = 'nearest'
        else:
            raise NotImplementedError(interpolation)
        if boundary == 'zero':
            boundary = 'zeros'
        else:
            raise NotImplementedError(boundary)
        resolution = torch.Tensor(self.staticshape(inputs)[1:-1])
        sample_coords = 2 * sample_coords / (resolution-1) - 1
        inputs = channels_first(inputs)
        sample_coords = torch.flip(sample_coords, dims=[-1])
        result = torchf.grid_sample(inputs, sample_coords, mode=interpolation, padding_mode=boundary)
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
        raise NotImplementedError()

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None, maximum_iterations=None):
        raise NotImplementedError()

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
        return torch.max(x, dim=axis)

    def min(self, x, axis=None):
        return torch.min(x, dim=axis)

    def maximum(self, a, b):
        return torch.max(a, other=b)

    def minimum(self, a, b):
        return torch.min(a, other=b)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
        raise NotImplementedError()

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
        kernel = kernel.permute(tuple(range(2, len(kernel.shape))) + (0, 1))
        result = torchf.conv2d(tensor, kernel, padding=padding)
        result = channels_last(result)
        return result

    def expand_dims(self, a, axis=0, number=1):
        raise NotImplementedError()

    def shape(self, tensor):
        return tensor.shape

    def staticshape(self, tensor):
        return tuple(tensor.shape)

    def to_float(self, x):
        raise NotImplementedError()

    def to_int(self, x, int64=False):
        raise NotImplementedError()

    def to_complex(self, x):
        raise NotImplementedError()

    def gather(self, values, indices):
        raise NotImplementedError()

    def gather_nd(self, values, indices):
        raise NotImplementedError()

    def unstack(self, tensor, axis=0):
        raise NotImplementedError()

    def std(self, x, axis=None):
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
        raise NotImplementedError()

    def ifft(self, k):
        raise NotImplementedError()

    def imag(self, complex):
        raise NotImplementedError()

    def real(self, complex):
        raise NotImplementedError()

    def cast(self, x, dtype):
        raise NotImplementedError()

    def sin(self, x):
        return torch.sin(x)

    def cos(self, x):
        return torch.cos(x)

    def dtype(self, array):
        return array.dtype

    def tile(self, value, multiples):
        raise NotImplementedError()


def channels_first(x):
    return x.permute(*((0, -1) + tuple(range(1, len(x.shape) - 1))))


def channels_last(x):
    return x.permute((0,) + tuple(range(2, len(x.shape))) + (1,))

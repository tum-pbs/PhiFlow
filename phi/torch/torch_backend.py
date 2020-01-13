import warnings

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
        else:
            raise NotImplementedError(boundary)
        inputs = self.as_tensor(inputs)
        sample_coords = self.as_tensor(sample_coords)
        resolution = torch.Tensor(self.staticshape(inputs)[1:-1])
        sample_coords = 2 * sample_coords / (resolution-1) - 1
        inputs = channels_first(inputs)
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

    def sparse_tensor(self, indices, values, shape):
        indices_ = torch.transpose(torch.LongTensor(indices), 0, 1)
        values_ = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(indices_, values_, shape)


def channels_first(x):
    return x.permute(*((0, -1) + tuple(range(1, len(x.shape) - 1))))


def channels_last(x):
    return x.permute((0,) + tuple(range(2, len(x.shape))) + (1,))

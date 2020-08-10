import numbers
import uuid
import warnings
from packaging import version
import six

import numpy as np
import six
import tensorflow as tf
from packaging import version

from phi.backend.backend_helper import split_multi_mode_pad, PadSettings, general_grid_sample_nd, equalize_shapes, circular_pad, replicate_pad
from phi.backend.scipy_backend import SciPyBackend
from phi.tf.tf_cuda_resample import *
from . import tf

from phi.backend.backend import Backend
from phi.backend.tensorop import expand, collapsed_gather_nd


class TFBackend(Backend):

    def __init__(self):
        Backend.__init__(self, "TensorFlow")

    @property
    def precision_dtype(self):
        return {16: np.float16, 32: np.float32, 64: np.float64, None: np.float32}[self.precision]

    def is_tensor(self, x, only_native=False):
        if not only_native and SciPyBackend().is_tensor(x, only_native=False):
            return True
        return isinstance(x, (tf.Tensor, tf.Variable, tf.SparseTensor, tf.Operation))

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            tensor = x
        elif isinstance(x, np.ndarray):
            tensor = tf.convert_to_tensor(SciPyBackend(precision=self.precision).as_tensor(x))
        else:
            tensor = tf.convert_to_tensor(x)
        # --- Enforce Precision ---
        if not isinstance(tensor, numbers.Number):
            if isinstance(tensor, np.ndarray):
                tensor = SciPyBackend(precision=self.precision).as_tensor(tensor)
            elif tensor.dtype.is_floating and self.has_fixed_precision:
                tensor = self.to_float(tensor)
        return tensor

    def copy(self, tensor, only_mutable=False):
        if not only_mutable or tf.executing_eagerly():
            return tf.identity(tensor)
        else:
            return tensor

    def equal(self, x, y):
        return tf.equal(x, y)

    def divide_no_nan(self, x, y):
        if version.parse(tf.__version__) >= version.parse('1.11.0'):
            return tf.div_no_nan(x, y)
        else:
            result = x / y
            return tf.where(tf.is_finite(result), result, tf.zeros_like(result))

    def random_uniform(self, shape):
        return tf.random.uniform(shape, dtype=self.precision_dtype)

    def random_normal(self, shape):
        return tf.random.normal(shape, dtype=self.precision_dtype)

    def rank(self, value):
        return len(value.shape)

    def range(self, start, limit=None, delta=1, dtype=None):
        return tf.range(start, limit, delta, dtype)

    def tile(self, value, multiples):
        if isinstance(multiples, (tuple, list)) and self.ndims(value) < len(multiples):
            value = self.expand_dims(value, axis=0, number=len(multiples) - self.ndims(value))
        return tf.tile(value, multiples)

    def stack(self, values, axis=0):
        return tf.stack(values, axis=axis)

    def concat(self, values, axis):
        return tf.concat(values, axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        passes = split_multi_mode_pad(self.ndims(value), PadSettings(pad_width, mode, constant_values), split_by_constant_value=True)
        for pad_pass in passes:
            value = self._single_mode_single_constant_pad(value, *pad_pass)
        return value

    def _single_mode_single_constant_pad(self, value, pad_width, single_mode, constant_value=0):
        assert single_mode in ('constant', 'symmetric', 'circular', 'reflect', 'replicate'), single_mode
        if single_mode == 'circular':
            return circular_pad(value, pad_width, self)
        if single_mode == 'replicate':
            if np.any(np.array(pad_width) > 1):
                return replicate_pad(value, pad_width, self)
            else:
                single_mode = 'symmetric'
        return tf.pad(value, pad_width, single_mode.upper(), constant_values=constant_value)  # constant, symmetric, reflect

    def reshape(self, value, shape):
        return tf.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        return tf.reduce_sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        if value.dtype == bool:
            return tf.reduce_all(value, axis=axis)
        return tf.reduce_prod(value, axis=axis)

    def where(self, condition, x=None, y=None):
        c = self.cast(condition, self.dtype(x))
        return c * x + (1 - c) * y
        # return tf.where(condition, x, y)  # TF1 has an inconsistent broadcasting rule for where

    def mean(self, value, axis=None, keepdims=False):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        return tf.reduce_mean(value, axis, keepdims=keepdims)

    def py_func(self, func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
        if grad is None:
            result = tf.py_func(func, inputs, Tout, stateful=stateful, name=name)
        else:
            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'PyFuncGrad' + str(uuid.uuid4())

            tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                result = tf.py_func(func, inputs, Tout, stateful=stateful, name=name)
        if shape_out is not None:
            result.set_shape(shape_out)
        return result

    def resample(self, inputs, sample_coords, interpolation='linear', boundary='constant', constant_values=0):
        assert interpolation == 'linear'
        if use_cuda(inputs):
            return resample_cuda(inputs, sample_coords, boundary)
        else:
            return general_grid_sample_nd(inputs, sample_coords, boundary, constant_values, self)  # while this is a bit slower than niftynet, it give consisten results at the boundaries

    def zeros_like(self, tensor):
        return tf.zeros_like(tensor)

    def ones_like(self, tensor):
        return tf.ones_like(tensor)

    def dot(self, a, b, axes):
        return tf.tensordot(a, b, axes)

    def matmul(self, A, b):
        if isinstance(A, tf.SparseTensor):
            result = tf.sparse_tensor_dense_matmul(A, tf.transpose(b))
            result = tf.transpose(result)
            result.set_shape(tf.TensorShape([b.shape[0], A.shape[0]]))
            return result
        else:
            return tf.matmul(A, b)

    def einsum(self, equation, *tensors):
        return tf.einsum(equation, *tensors)

    def while_loop(self, cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True,
                   swap_memory=False, name=None, maximum_iterations=None):
        return tf.while_loop(cond, body, loop_vars,
                             shape_invariants=shape_invariants,
                             parallel_iterations=parallel_iterations,
                             back_prop=back_prop,
                             swap_memory=swap_memory,
                             name=name,
                             maximum_iterations=maximum_iterations)

    def abs(self, x):
        return tf.abs(x)

    def sign(self, x):
        return tf.sign(x)

    def round(self, x):
        return tf.round(x)

    def ceil(self, x):
        return tf.ceil(x)

    def floor(self, x):
        return tf.floor(x)

    def max(self, x, axis=None, keepdims=False):
        return tf.reduce_max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return tf.reduce_min(x, axis=axis, keepdims=keepdims)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base="custom_gradient_func"):
        # Setup custom gradient
        gradient_name = name_base + "_" + str(uuid.uuid4())
        tf.RegisterGradient(gradient_name)(gradient)

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": gradient_name}):
            fake_function = tf.identity(inputs[input_index])

        outputs = function(*inputs)
        output = outputs if output_index is None else outputs[output_index]
        output_with_gradient = fake_function + tf.stop_gradient(output - fake_function)
        if output_index is None:
            return output_with_gradient
        else:
            outputs = list(outputs)
            outputs[output_index] = output_with_gradient
            return outputs

    def maximum(self, a, b):
        return tf.maximum(a, b)

    def minimum(self, a, b):
        return tf.minimum(a, b)

    def clip(self, x, minimum, maximum):
        return tf.clip_by_value(x, minimum, maximum)

    def sqrt(self, x):
        return tf.sqrt(x)

    def exp(self, x):
        return tf.exp(x)

    def conv(self, tensor, kernel, padding="SAME"):
        rank = tensor_spatial_rank(tensor)
        padding = padding.upper()
        if rank == 1:
            result = tf.nn.conv1d(tensor, kernel, 1, padding)
        elif rank == 2:
            result = tf.nn.conv2d(tensor, kernel, [1, 1, 1, 1], padding)
        elif rank == 3:
            result = tf.nn.conv3d(tensor, kernel, [1, 1, 1, 1, 1], padding)
        else:
            raise ValueError("Tensor must be of rank 1, 2 or 3 but is %d" % rank)
        return result

    def expand_dims(self, a, axis=0, number=1):
        if number == 0:
            return a
        for _i in range(number):
            a = tf.expand_dims(a, axis)
        return a

    def shape(self, tensor):
        return tf.shape(tensor)

    def to_float(self, x, float64=False):
        if float64:
            warnings.warn('float64 argument is deprecated, set Backend.precision = 64 to use 64 bit operations.', DeprecationWarning)
            return tf.cast(x, tf.float64)
        else:
            return tf.cast(x, self.precision_dtype)

    def staticshape(self, tensor):
        if self.is_tensor(tensor, only_native=True):
            return tuple(tensor.shape.as_list())
        else:
            return np.shape(tensor)

    def to_int(self, x, int64=False):
        return tf.cast(x, tf.int64) if int64 else tf.cast(x, tf.int32)

    def to_complex(self, x):
        if self.dtype(x) in (np.complex64, np.complex128):
            return x
        if self.dtype(x) == np.float64:
            return tf.to_complex128(x)
        else:
            return tf.to_complex64(x)

    def gather(self, values, indices):
        if isinstance(indices, slice):
            return values[indices]
        return tf.gather(values, indices)

    def gather_nd(self, values, indices, batch_dims=0):
        if batch_dims == 0:
            return tf.gather_nd(values, indices)
        elif version.parse(tf.__version__) >= version.parse('1.14.0'):
            return tf.gather_nd(values, indices, batch_dims=batch_dims)
        else:
            if batch_dims > 1:
                raise NotImplementedError('batch_dims > 1 only supported on TensorFlow >= 1.14')
            batch_size = self.shape(values)[0]
            batch_ids = tf.reshape(tf.range(batch_size), [batch_size] + [1] * (self.ndims(indices) - 1))
            batch_ids = tf.tile(batch_ids, [1] + self.shape(indices)[1:-1] + [1])
            indices = tf.concat([batch_ids, indices], -1)
            return tf.gather_nd(values, indices)

    def unstack(self, tensor, axis=0, keepdims=False):
        unstacked = tf.unstack(tensor, axis=axis)
        if keepdims:
            unstacked = [self.expand_dims(c, axis=axis) for c in unstacked]
        return unstacked

    def std(self, x, axis=None, keepdims=False):
        _mean, var = tf.nn.moments(x, axis, keepdims=keepdims)
        return tf.sqrt(var)

    def boolean_mask(self, x, mask):
        return tf.boolean_mask(x, mask)

    def isfinite(self, x):
        return tf.is_finite(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return tf.reduce_any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return tf.reduce_all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, points, indices, values, shape, duplicates_handling='undefined'):
        # Change indexing so batch number is included as first element of the index, for example: [0,31,24] indexes the first batch (batch 0) and 2D coordinates (31,24).
        buffer = tf.zeros(shape, dtype=values.dtype)

        repetitions = []
        for dim in range(len(indices.shape) - 1):
            if values.shape[dim] == 1:
                repetitions.append(indices.shape[dim])
            else:
                assert indices.shape[dim] == values.shape[dim]
                repetitions.append(1)
        repetitions.append(1)
        values = self.tile(values, repetitions)

        if duplicates_handling == 'add':
            # Only for Tensorflow with custom gradient
            @tf.custom_gradient
            def scatter_density(points, indices, values):
                result = tf.tensor_scatter_add(buffer, indices, values)

                def grad(dr):
                    return self.resample(gradient(dr, difference='central'), points), None, None

                return result, grad

            return scatter_density(points, indices, values)
        elif duplicates_handling == 'mean':
            # Won't entirely work with out of bounds particles (still counted in mean)
            count = tf.tensor_scatter_add(buffer, indices, tf.ones_like(values))
            total = tf.tensor_scatter_add(buffer, indices, values)
            return total / tf.maximum(1.0, count)
        else:  # last, any, undefined
            # indices = self.to_int(indices, int64=True)
            # st = tf.SparseTensor(indices, values, shape)  # ToDo this only supports 2D shapes
            # st = tf.sparse.reorder(st)   # only needed if not ordered
            # return tf.sparse.to_dense(st)
            count = tf.tensor_scatter_add(buffer, indices, tf.ones_like(values))
            total = tf.tensor_scatter_add(buffer, indices, values)
            return total / tf.maximum(1.0, count)

    def fft(self, x):
        rank = len(x.shape) - 2
        assert rank >= 1
        x = self.to_complex(x)
        if rank == 1:
            return tf.stack([tf.fft(c) for c in tf.unstack(x, axis=-1)], axis=-1)
        elif rank == 2:
            return tf.stack([tf.fft2d(c) for c in tf.unstack(x, axis=-1)], axis=-1)
        elif rank == 3:
            return tf.stack([tf.fft3d(c) for c in tf.unstack(x, axis=-1)], axis=-1)
        else:
            raise NotImplementedError('n-dimensional FFT not implemented.')

    def ifft(self, k):
        rank = len(k.shape) - 2
        assert rank >= 1
        if rank == 1:
            return tf.stack([tf.ifft(c) for c in tf.unstack(k, axis=-1)], axis=-1)
        elif rank == 2:
            return tf.stack([tf.ifft2d(c) for c in tf.unstack(k, axis=-1)], axis=-1)
        elif rank == 3:
            return tf.stack([tf.ifft3d(c) for c in tf.unstack(k, axis=-1)], axis=-1)
        else:
            raise NotImplementedError('n-dimensional inverse FFT not implemented.')

    def imag(self, complex):
        return tf.imag(complex)

    def real(self, complex):
        return tf.real(complex)

    def cast(self, x, dtype):
        return tf.cast(x, dtype)

    def sin(self, x):
        return tf.sin(x)

    def cos(self, x):
        return tf.cos(x)

    def dtype(self, array):
        if self.is_tensor(array, only_native=True):
            return array.dtype.as_numpy_dtype
        else:
            return SciPyBackend().dtype(array)

    def sparse_tensor(self, indices, values, shape):
        return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


# from niftynet.layer.resampler.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/69c98e5a95cc6788ad9fb8c5e27dc24d1acec634/niftynet/layer/resampler.py


COORDINATES_TYPE = tf.int32
EPS = 1e-6


def tensor_spatial_rank(tensor):
    return len(tensor.shape) - 2


def unit_direction(dim, spatial_rank):  # ordered like z,y,x
    direction = [1 if i == dim else 0 for i in range(spatial_rank)]
    for _i in range(spatial_rank):
        direction = tf.expand_dims(direction, axis=0)
    return direction


def _resample_no_pack(grid, coords, boundary_func):
    resolution = np.array([int(d) for d in grid.shape[1:-1]])
    sp_rank = tensor_spatial_rank(grid)

    floor = boundary_func(tf.floor(coords), resolution)
    up_weights = coords - floor
    lo_weights = TFBackend().unstack(1 - up_weights, axis=-1, keepdims=True)
    up_weights = TFBackend().unstack(up_weights, axis=-1, keepdims=True)
    base_coords = tf.cast(floor, tf.int32)

    def interpolate_nd(coords, axis):
        direction = np.array([1 if ax == axis else 0 for ax in range(sp_rank)])
        print(direction.shape)
        with tf.variable_scope('coord_plus_one'):
            up_coords = coords + direction  # This is extremely slow for some reason - ToDo tile direction array to have same dimensions before calling interpolate_nd?
        if axis == sp_rank - 1:
            # up_coords = boundary_func(up_coords, resolution)
            lo_values = tf.gather_nd(grid, coords, batch_dims=1)
            up_values = tf.gather_nd(grid, up_coords, batch_dims=1)
        else:
            lo_values = interpolate_nd(coords, axis + 1)
            up_values = interpolate_nd(up_coords, axis + 1)
        with tf.variable_scope('weighted_sum_axis_%d' % axis):
            return lo_values * lo_weights[axis] + up_values * up_weights[axis]

    with tf.variable_scope('interpolate_nd'):
        result = interpolate_nd(base_coords, 0)
    return result


def _resample_linear_niftynet(inputs, sample_coords, boundary, boundary_func, float_type):
    inputs = tf.convert_to_tensor(inputs)
    sample_coords = tf.convert_to_tensor(sample_coords)

    in_spatial_size = [int(d) for d in inputs.shape[1:-1]]
    in_spatial_rank = tensor_spatial_rank(inputs)
    batch_size = tf.shape(inputs)[0]

    out_spatial_rank = tensor_spatial_rank(sample_coords)
    out_spatial_size = sample_coords.get_shape().as_list()[1:-1]

    if sample_coords.shape[0] != inputs.shape[0]:
        sample_coords = tf.tile(sample_coords, [batch_size] + [1] * (len(sample_coords.shape) - 1))

    xy = tf.unstack(sample_coords, axis=-1)
    base_coords = [tf.floor(coords) for coords in xy]
    floor_coords = [tf.cast(boundary_func(x, in_spatial_size[idx]), COORDINATES_TYPE) for (idx, x) in enumerate(base_coords)]
    ceil_coords = [tf.cast(boundary_func(x + 1.0, in_spatial_size[idx]), COORDINATES_TYPE) for (idx, x) in enumerate(base_coords)]

    if boundary.upper() == 'ZERO':
        weight_0 = [tf.expand_dims(x - tf.cast(i, float_type), -1) for (x, i) in zip(xy, floor_coords)]
        weight_1 = [tf.expand_dims(tf.cast(i, float_type) - x, -1) for (x, i) in zip(xy, ceil_coords)]
    else:
        weight_0 = [tf.expand_dims(x - i, -1) for (x, i) in zip(xy, base_coords)]
        weight_1 = [1.0 - w for w in weight_0]

    batch_ids = tf.reshape(tf.range(batch_size), [batch_size] + [1] * out_spatial_rank)
    batch_ids = tf.tile(batch_ids, [1] + out_spatial_size)
    sc = (floor_coords, ceil_coords)
    binary_neighbour_ids = [[int(c) for c in format(i, '0%ib' % in_spatial_rank)] for i in range(2 ** in_spatial_rank)]

    def get_knot(bc):
        coord = [sc[c][i] for i, c in enumerate(bc)]
        if version.parse(tf.__version__) >= version.parse('1.14.0'):
            coord = tf.stack(coord, -1)
            return tf.gather_nd(inputs, coord, batch_dims=1)  # NaN can cause negative integers here
        else:
            coord = tf.stack([batch_ids] + coord, -1)
            return tf.gather_nd(inputs, coord)  # NaN can cause negative integers here

    samples = [get_knot(bc) for bc in binary_neighbour_ids]

    def _pyramid_combination(samples, w_0, w_1):
        if len(w_0) == 1:
            return samples[0] * w_1[0] + samples[1] * w_0[0]
        f_0 = _pyramid_combination(samples[::2], w_0[:-1], w_1[:-1])
        f_1 = _pyramid_combination(samples[1::2], w_0[:-1], w_1[:-1])
        return f_0 * w_1[-1] + f_1 * w_0[-1]

    return _pyramid_combination(samples, weight_0, weight_1)


def _boundary_snap(sample_coords, spatial_shape):
    max_indices = [l - 1 for l in spatial_shape]
    for _i in range(len(spatial_shape)):
        max_indices = tf.expand_dims(max_indices, 0)
    sample_coords = tf.minimum(sample_coords, max_indices)
    sample_coords = tf.maximum(sample_coords, 0)
    return sample_coords


def _boundary_replicate(sample_coords, input_size):
    return tf.maximum(tf.minimum(sample_coords, input_size - 1), 0)


def _boundary_circular(sample_coords, input_size):
    return tf.mod(tf.mod(sample_coords, input_size) + input_size, input_size)


def _boundary_symmetric(sample_coords, input_size):
    sample_coords = _boundary_circular(sample_coords, 2 * input_size)
    return ((2 * input_size - 1) - tf.abs((2 * input_size - 1) - 2 * sample_coords)) // 2


def _boundary_reflect(sample_coords, input_size):
    sample_coords = _boundary_circular(sample_coords, 2 * input_size - 2)
    return (input_size - 1) - tf.abs((input_size - 1) - sample_coords)


SUPPORTED_BOUNDARY = {
    'zero': _boundary_replicate,
    'replicate': _boundary_replicate,
    'circular': _boundary_circular,
    'symmetric': _boundary_symmetric,
    'reflect': _boundary_reflect,
}

import logging
import uuid
import warnings

import numpy as np
import six
import tensorflow as tf
from packaging import version

from phi.math.base_backend import Backend
from phi.struct.tensorop import expand, collapsed_gather_nd

if tf.__version__[0] == '2':
    logging.info('Adjusting for tensorflow 2.0')
    tf = tf.compat.v1
    tf.disable_eager_execution()


class TFBackend(Backend):

    def __init__(self):
        Backend.__init__(self, "TensorFlow")

    def is_tensor(self, x):
        return isinstance(x, (tf.Tensor, tf.Variable, tf.SparseTensor, tf.Operation))

    def as_tensor(self, x):
        if isinstance(x, np.ndarray) and x.dtype == np.float64:
            return tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.convert_to_tensor(x)

    def equal(self, x, y):
        return tf.equal(x, y)

    def divide_no_nan(self, x, y):
        if version.parse(tf.__version__) >= version.parse('1.11.0'):
            return tf.div_no_nan(x, y)
        else:
            result = x / y
            return tf.where(tf.is_finite(result), result, tf.zeros_like(result))

    def random_uniform(self, shape):
        return tf.random.uniform(shape)

    def rank(self, value):
        return len(value.shape)

    def range(self, start, limit=None, delta=1, dtype=None):
        return tf.range(start, limit, delta, dtype)

    def tile(self, value, multiples):
        return tf.tile(value, multiples)

    def stack(self, values, axis=0):
        return tf.stack(values, axis=axis)

    def concat(self, values, axis):
        return tf.concat(values, axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        dims = range(len(self.staticshape(value)))
        if isinstance(mode, six.string_types) and len(self.staticshape(constant_values)) == 0:
            return self._single_mode_single_constant_pad(value, pad_width, mode, constant_values)
        else:
            mode = expand(mode, shape=(len(dims), 2))
            passes = [('circular', 0), ('wrap', 0), ('replicate', 0), ('symmetric', 0), ('reflect', 0)]
            constant_values = expand(constant_values, shape=(len(dims), 2))
            constant_value_set = set()
            for d in dims:
                for upper in (False, True):
                    constant_value_set.add(constant_values[d][upper])
            for const in constant_value_set:
                passes.append(('constant', const))
            for single_mode, constant_value in passes:  # order matters! wrap first
                widths = [[collapsed_gather_nd(pad_width, [d, upper]) if mode[d][upper] == single_mode and constant_values[d][upper] == constant_value else 0 for upper in (False, True)] for d in dims]
                value = self._single_mode_single_constant_pad(value, widths, single_mode, constant_value)
            return value

    def _single_mode_single_constant_pad(self, value, pad_width, single_mode, constant_value=0):
        single_mode = single_mode.lower()
        if single_mode == 'wrap':
            warnings.warn("'wrap' is deprecated, use 'circular' instead", DeprecationWarning, stacklevel=2)
            single_mode = 'circular'
        assert single_mode in ('constant', 'symmetric', 'circular', 'reflect', 'replicate'), single_mode
        if np.sum(np.array(pad_width)) == 0:
            return value
        if single_mode == 'circular':
            dims = range(len(value.shape))
            for dim in dims:
                s = value.shape[dim]
                pad_lower, pad_upper = pad_width[dim]
                if pad_lower is 0 and pad_upper is 0:
                    continue  # Nothing to pad
                lower_slices = [slice(s-pad_lower, None) if d == dim else slice(None) for d in dims]
                upper_slices = [slice(None, pad_upper) if d == dim else slice(None) for d in dims]
                lower = value[lower_slices]
                upper = value[upper_slices]
                value = tf.concat([lower, value, upper], axis=dim)
            return value
        if single_mode == 'replicate':
            if np.any(np.array(pad_width) > 1):
                raise NotImplementedError()  # ToDo: manual padding with slices
            else:
                single_mode = 'symmetric'
        return tf.pad(value, pad_width, single_mode.upper(), constant_values=constant_value)

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
        return tf.where(condition, x, y)

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

    def resample(self, inputs, sample_coords, interpolation='linear', boundary='constant'):
        if boundary.lower() == 'constant':
            boundary = 'zero'
        boundary_func = SUPPORTED_BOUNDARY[boundary.lower()]
        assert interpolation.lower() == 'linear'
        return _resample_linear_niftynet(inputs, sample_coords, boundary, boundary_func)

    def zeros_like(self, tensor):
        return tf.zeros_like(tensor)

    def ones_like(self, tensor):
        return tf.ones_like(tensor)

    def dot(self, a, b, axes):
        return tf.tensordot(a, b, axes)

    def matmul(self, A, b):
        if isinstance(A, tf.SparseTensor):
            result = tf.sparse_tensor_dense_matmul(A, tf.transpose(b))
            return tf.transpose(result)
        else:
            return tf.matmul(A, b)

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

    def max(self, x, axis=None):
        return tf.reduce_max(x, axis=axis)

    def min(self, x, axis=None):
        return tf.reduce_min(x, axis=axis)

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
        for _i in range(number):
            a = tf.expand_dims(a, axis)
        return a

    def shape(self, tensor):
        return tf.shape(tensor)

    def to_float(self, x, float64=False):
        return tf.cast(x, tf.float64) if float64 else tf.cast(x, tf.float32)

    def staticshape(self, tensor):
        if self.is_tensor(tensor):
            return tuple(tensor.shape.as_list())
        else:
            return np.shape(tensor)

    def to_int(self, x, int64=False):
        return tf.cast(x, tf.int64) if int64 else tf.cast(x, tf.int32)

    def to_complex(self, x):
        return tf.to_complex64(x)

    def gather(self, values, indices):
        return tf.gather(values, indices)

    def gather_nd(self, values, indices):
        return tf.gather_nd(values, indices)

    def unstack(self, tensor, axis=0):
        return tf.unstack(tensor, axis=axis)

    def std(self, x, axis=None):
        _mean, var = tf.nn.moments(x, axis)
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
        z = tf.zeros(shape, dtype=values.dtype)

        if duplicates_handling == 'add':
            #Only for Tensorflow with custom gradient
            @tf.custom_gradient
            def scatter_density(points, indices, values):
                result = tf.tensor_scatter_add(z, indices, values)

                def grad(dr):
                    return self.resample(gradient(dr, difference='central'), points), None, None

                return result, grad

            return scatter_density(points, indices, values)
        elif duplicates_handling == 'mean':
            # Won't entirely work with out of bounds particles (still counted in mean)
            count = tf.tensor_scatter_add(z, indices, tf.ones_like(values))
            total = tf.tensor_scatter_add(z, indices, values)
            return (total / tf.maximum(1.0, count))
        else: # last, any, undefined
            st = tf.SparseTensor(indices, values, shape)
            st = tf.sparse.reorder(st)   # only needed if not ordered
            return tf.sparse.to_dense(st)

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
        return array.dtype.as_numpy_dtype

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


def _resample_linear_niftynet(inputs, sample_coords, boundary, boundary_func):
    inputs = tf.convert_to_tensor(inputs)
    sample_coords = tf.convert_to_tensor(sample_coords)

    in_spatial_size = [int(d) for d in inputs.shape[1:-1]]
    in_spatial_rank = tensor_spatial_rank(inputs)
    batch_size = tf.shape(inputs)[0]

    out_spatial_rank = tensor_spatial_rank(sample_coords)
    out_spatial_size = sample_coords.get_shape().as_list()[1:-1]

    if sample_coords.shape[0] != inputs.shape[0]:
        sample_coords = tf.tile(sample_coords, [batch_size]+[1]*(len(sample_coords.shape)-1))

    if in_spatial_rank == 2 and boundary.upper() == 'ZERO':
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        return tf.contrib.resampler.resampler(inputs, sample_coords)

    xy = tf.unstack(sample_coords, axis=-1)
    base_coords = [tf.floor(coords) for coords in xy]
    floor_coords = [tf.cast(boundary_func(x, in_spatial_size[idx]), COORDINATES_TYPE) for (idx, x) in enumerate(base_coords)]
    ceil_coords = [tf.cast(boundary_func(x + 1.0, in_spatial_size[idx]), COORDINATES_TYPE) for (idx, x) in enumerate(base_coords)]

    if boundary.upper() == 'ZERO':
        weight_0 = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for (x, i) in zip(xy, floor_coords)]
        weight_1 = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for (x, i) in zip(xy, ceil_coords)]
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
    max_indices = [l-1 for l in spatial_shape]
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
    circular_size = input_size + input_size - 2
    return (input_size - 1) - tf.abs(
        (input_size - 1) - _boundary_circular(sample_coords, circular_size))


SUPPORTED_BOUNDARY = {
    'zero': _boundary_replicate,
    'replicate': _boundary_replicate,
    'circular': _boundary_circular,
    'symmetric': _boundary_symmetric
}

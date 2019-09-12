import tensorflow as tf
import numpy as np
from numpy import ndarray
import collections
import uuid
from phi.math.base import Backend


class TFBackend(Backend):

    def __init__(self):
        Backend.__init__(self, "TensorFlow")

    def is_applicable(self, values):
        for value in values:
            if isinstance(value, tf.Tensor): return True
            if isinstance(value, tf.Variable): return True
            if isinstance(value, tf.SparseTensor): return True
        return False

    def rank(self, value):
        return len(value.shape)

    def stack(self, values, axis=0):
        return tf.stack(values, axis=axis)

    def concat(self, values, axis):
        return tf.concat(values, axis)

    def pad(self, value, pad_width, mode="constant", constant_values=0):
        if np.sum(np.array(pad_width)) == 0:
            return value
        return tf.pad(value, pad_width, mode, constant_values=constant_values)

    def add(self, values):
        return tf.add_n(values)

    def reshape(self, value, shape):
        return tf.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        return tf.reduce_sum(value, axis=axis)

    def mean(self, value, axis=None):
        if axis is not None:
            if not isinstance(axis, int):
                axis = list(axis)
        return tf.reduce_mean(value, axis)

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

    def resample(self, inputs, sample_coords, interpolation="LINEAR", boundary="ZERO"):
        return resample_tf(inputs, sample_coords, interpolation, boundary)

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

    def ceil(self, x):
        return tf.ceil(x)

    def floor(self, x):
        return tf.floor(x)

    def max(self, x, axis=None):
        return tf.reduce_max(x, axis=axis)

    def with_custom_gradient(self, function, inputs, gradient, input_index=0, output_index=None, name_base="custom_gradient_func"):
        import uuid
        # Setup custom gradient
        gradient_name = name_base + "_" + str(uuid.uuid4())
        tf.RegisterGradient(gradient_name)(gradient)

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": gradient_name}):
            fake_function = tf.identity(inputs[input_index])

        outputs = function(*inputs)
        output = outputs if output_index is None else outputs[output_index]
        output_with_gradient = fake_function + tf.stop_gradient(output - fake_function)
        if output_index is None: return output_with_gradient
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
            raise ValueError("Tensor must be of rank 1, 2 or 3 but is %d"%rank)
        return result

    def expand_dims(self, a, axis=0):
        return tf.expand_dims(a, axis)

    def shape(self, tensor):
        return tf.shape(tensor)

    def staticshape(self, tensor):
        return tuple(tensor.shape.as_list())

    def to_float(self, x):
        return tf.to_float(x)

    def to_int(self, x, int64=False):
        return tf.to_int64(x) if int64 else tf.to_int32(x)

    def gather(self, values, indices):
        return tf.gather(values, indices)

    def unstack(self, tensor, axis=0):
        return tf.unstack(tensor, axis=axis)

    def std(self, x, axis=None):
        mean, var = tf.nn.moments(x, axis)
        return tf.sqrt(var)

    def boolean_mask(self, x, mask):
        return tf.boolean_mask(x, mask)

    def isfinite(self, x):
        return tf.is_finite(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return tf.reduce_any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return tf.reduce_all(boolean_tensor, axis=axis, keepdims=keepdims)

    def fft(self, x):
        rank = len(x.shape) - 2
        assert rank >= 1
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


# from niftynet.layer.resampler.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/69c98e5a95cc6788ad9fb8c5e27dc24d1acec634/niftynet/layer/resampler.py

COORDINATES_TYPE = tf.int32
EPS = 1e-6


def tensor_spatial_rank(tensor):
    return len(tensor.shape) - 2



# def _resample_linear(inputs, sample_coords, boundary, boundary_func):
#     # sample coords are ordered like x,y,z
#     # sample_coords = sample_coords[..., ::-1] # now ordered like z,y,x
#
#     in_size = inputs.get_shape().as_list()
#     in_spatial_size = in_size[1:-1]
#     in_spatial_rank = tensor_spatial_rank(inputs)
#     batch_size = tf.shape(inputs)[0]
#
#     out_spatial_rank = tensor_spatial_rank(sample_coords)
#     out_spatial_size = sample_coords.get_shape().as_list()[1:-1]
#
#     if in_spatial_rank == 2 and boundary == 'ZERO':
#         inputs = tf.transpose(inputs, [0, 2, 1, 3])
#         return tf.contrib.resampler.resampler(inputs, sample_coords)
#
#     xy = tf.unstack(sample_coords, axis=-1)
#     base_coords = [tf.floor(coords) for coords in xy]
#     floor_coords = [tf.cast(boundary_func(x, in_spatial_size[idx]), COORDINATES_TYPE) for (idx, x) in
#                     enumerate(base_coords)]
#     ceil_coords = [tf.cast(boundary_func(x + 1.0, in_spatial_size[idx]), COORDINATES_TYPE) for (idx, x) in
#                    enumerate(base_coords)]
#
#     if boundary == 'ZERO':
#         floor_weights = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for (x, i) in zip(xy, floor_coords)]
#         ceil_weights = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for (x, i) in zip(xy, ceil_coords)]
#     else:
#         floor_weights = [tf.expand_dims(x - i, -1) for (x, i) in zip(xy, base_coords)]
#         ceil_weights = [1.0 - w for w in floor_weights]
#
#     floor_coords = tf.stack(floor_coords, -1)
#     ceil_coords = tf.stack(ceil_coords, -1)
#
#     batch_ids = tf.reshape(tf.range(batch_size), [batch_size] + [1] * out_spatial_rank)
#     batch_ids = tf.tile(batch_ids, [1] + out_spatial_size)
#
#
#
#     int_coords = []
#     def collect_coordinates(floor_coords, dimensions):
#         if not dimensions:
#             int_coords.append(boundary_func(floor_coords, in_spatial_size))
#         else:
#             collect_coordinates(floor_coords, dimensions[1:])
#             collect_coordinates(floor_coords + unit_directions[dimensions[0]], dimensions[1:])
#     collect_coordinates(floor_coords, range(out_spatial_rank))
#
#
#     def linear_interpolation(dimensions):
#         if not dimensions:
#             batch_floor_coords = int_coords.pop(0)
#             return tf.gather_nd(inputs, batch_floor_coords)
#         else:
#             dimension = dimensions[0]
#             lower = linear_interpolation(dimensions[1:])
#             upper = linear_interpolation(dimensions[1:])
#             batch_floor_weights = tf.expand_dims(floor_weights[..., dimension], axis=-1)
#             batch_ceil_weights = tf.expand_dims(ceil_weights[..., dimension], axis=-1)
#             return lower * batch_floor_weights + upper * batch_ceil_weights
#
#     nlinear = linear_interpolation(range(out_spatial_rank))
#     return nlinear


def unit_direction(dim, spatial_rank):  # ordered like z,y,x
    direction = [1 if i==dim else 0 for i in range(spatial_rank)]
    for i in range(spatial_rank):
        direction = tf.expand_dims(direction, axis=0)
    return direction



def _resample_linear_niftynet(inputs, sample_coords, boundary, boundary_func):
    in_spatial_size = [int(d) for d in inputs.shape[1:-1]]
    in_spatial_rank = tensor_spatial_rank(inputs)
    batch_size = tf.shape(inputs)[0]

    out_spatial_rank = tensor_spatial_rank(sample_coords)
    out_spatial_size = sample_coords.get_shape().as_list()[1:-1]

    if in_spatial_rank == 2 and boundary == 'ZERO':
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        return tf.contrib.resampler.resampler(inputs, sample_coords)

    xy = tf.unstack(sample_coords, axis=-1)
    base_coords = [tf.floor(coords) for coords in xy]
    floor_coords = [ tf.cast(boundary_func(x, in_spatial_size[idx]), COORDINATES_TYPE) for (idx, x) in enumerate(base_coords)]
    ceil_coords = [ tf.cast(boundary_func(x + 1.0, in_spatial_size[idx]), COORDINATES_TYPE) for (idx, x) in enumerate(base_coords)]

    if boundary == 'ZERO':
        weight_0 = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for (x, i) in zip(xy, floor_coords)]
        weight_1 = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for (x, i) in zip(xy, ceil_coords)]
    else:
        weight_0 = [tf.expand_dims(x - i, -1) for (x, i) in zip(xy, base_coords)]
        weight_1 = [1.0 - w for w in weight_0]

    batch_ids = tf.reshape( tf.range(batch_size), [batch_size] + [1] * out_spatial_rank )
    batch_ids = tf.tile(batch_ids, [1] + out_spatial_size)
    sc = (floor_coords, ceil_coords)
    binary_neighbour_ids = [ [int(c) for c in format(i, '0%ib' % in_spatial_rank)] for i in range(2 ** in_spatial_rank)]

    def get_knot(bc):
        coord = [sc[c][i] for i, c in enumerate(bc)]
        coord = tf.stack([batch_ids] + coord, -1)
        return tf.gather_nd(inputs, coord)

    samples = [get_knot(bc) for bc in binary_neighbour_ids]


    def _pyramid_combination(samples, w_0, w_1):
        if len(w_0) == 1:
            return samples[0] * w_1[0] + samples[1] * w_0[0]
        f_0 = _pyramid_combination(samples[::2], w_0[:-1], w_1[:-1])
        f_1 = _pyramid_combination(samples[1::2], w_0[:-1], w_1[:-1])
        return f_0 * w_1[-1] + f_1 * w_0[-1]

    return _pyramid_combination(samples, weight_0, weight_1)


def resample_tf(inputs, sample_coords, interpolation="LINEAR", boundary="ZERO"):
    """
Resamples an N-dimensional tensor at the locations provided by sample_coords
    :param inputs: grid with dimensions (batch_size, spatial dimensions..., element_size)
    :param sample_coords: sample coords (batch_size, output_shape, input_dimension)
    :param interpolation: LINEAR, BSPLINE, IDW (default is LINEAR)
    :param boundary: ZERO, REPLICATE, CIRCULAR, SYMMETRIC (default is ZERO)
    :return:
    """
    # for dim in inputs.shape:
    #     if dim.value is None:
    #         raise ValueError("Shape of input must be known, got {}".format(inputs.shape))

    boundary_func = SUPPORTED_BOUNDARY[boundary]
    assert interpolation.upper() == "LINEAR"
    # return _resample_linear(inputs, sample_coords, boundary, boundary_func)
    return _resample_linear_niftynet(inputs, sample_coords, boundary, boundary_func)



def _boundary_snap(sample_coords, spatial_shape):
    max_indices = [l-1 for l in spatial_shape]
    for i in range(len(spatial_shape)):
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
    'ZERO': _boundary_replicate,
    'REPLICATE': _boundary_replicate,
    'CIRCULAR': _boundary_circular,
    'SYMMETRIC': _boundary_symmetric
}





# def _resample_linear(inputs, sample_coords, boundary, boundary_func):
#     # sample coords are ordered like x,y,z
#     # sample_coords = sample_coords[..., ::-1] # now ordered like z,y,x
#
#     in_size = inputs.get_shape().as_list()
#     in_spatial_size = in_size[1:-1]
#     in_spatial_rank = tensor_spatial_rank(inputs)
#     batch_size = tf.shape(inputs)[0]
#
#     out_spatial_rank = tensor_spatial_rank(sample_coords)
#     out_spatial_size = sample_coords.get_shape().as_list()[1:-1]
#
#     if in_spatial_rank == 2 and boundary == 'ZERO':
#         inputs = tf.transpose(inputs, [0, 2, 1, 3])
#         return tf.contrib.resampler.resampler(inputs, sample_coords)
#
#     floor_coords = tf.cast(tf.floor(sample_coords), COORDINATES_TYPE)
#     ceil_weights = sample_coords - tf.floor(sample_coords)
#     floor_weights = 1 - ceil_weights
#
#     unit_directions = [unit_direction(i, out_spatial_rank) for i in range(out_spatial_rank)]
#
#     int_coords = []
#     def collect_coordinates(floor_coords, dimensions):
#         if not dimensions:
#             int_coords.append(boundary_func(floor_coords, in_spatial_size))
#         else:
#             collect_coordinates(floor_coords, dimensions[1:])
#             collect_coordinates(floor_coords + unit_directions[dimensions[0]], dimensions[1:])
#     collect_coordinates(floor_coords, range(out_spatial_rank))
#
#
#     def one_batch(batch_index):
#
#         def linear_interpolation(dimensions):
#             if not dimensions:
#                 batch_floor_coords = int_coords.pop(0)[batch_index,...]
#                 return tf.gather_nd(inputs[batch_index,...], batch_floor_coords)
#             else:
#                 dimension = dimensions[0]
#                 lower = linear_interpolation(dimensions[1:])
#                 upper = linear_interpolation(dimensions[1:])
#                 batch_floor_weights = tf.expand_dims(floor_weights[batch_index, ..., dimension], axis=-1)
#                 batch_ceil_weights = tf.expand_dims(ceil_weights[batch_index, ..., dimension], axis=-1)
#                 return lower * batch_floor_weights + upper * batch_ceil_weights
#
#         nlinear = linear_interpolation(range(out_spatial_rank))
#         return nlinear
#
#     result = tf.map_fn(one_batch, tf.range(0, batch_size), dtype=tf.float32)
#     return result
from collections import defaultdict
from unittest import TestCase
import random as rand

from packaging import version

import numpy as np
import tensorflow.compat.v1 as tf

from phi.tf._tf_cuda_resample import resample_cuda


tf.disable_eager_execution()

COORDINATES_TYPE = tf.int32
EPS = 1e-6


def tensor_spatial_rank(tensor):
    return len(tensor.shape) - 2


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


class TestTfCudaResample(TestCase):
    N = 10
    MAX_DIFFERENCE = 0.15
    MIN_VALUE = -10
    MAX_VALUE = 10
    BOUNDARIES = ['replicate', 'circular', 'symmetric', 'reflect']
    PERCENTAGE_OUT_OF_BOUNDS = 2.0

    def get_random_dims(self):
        dims = rand.randrange(1, 5)
        max_dim_size = 100
        if dims == 2:
            max_dim_size = 100
        elif dims == 3:
            max_dim_size = 50
        elif dims == 4:
            max_dim_size = 20
        dim_sizes = ()
        for i in range(dims):
            dim_sizes = dim_sizes + (rand.randrange(1, max_dim_size),)
        return dim_sizes

    def generate_data(self):
        dim_sizes = self.get_random_dims()
        shape = (1,)
        shape = shape + dim_sizes
        shape = shape + (rand.randrange(1, 6),)
        data = np.zeros(shape, np.float32)
        for i in range(data.size):
            data.flat[i] = rand.uniform(self.MIN_VALUE, self.MAX_VALUE)
        return data

    def generate_points(self, data_shape):
        dim_sizes = self.get_random_dims()
        shape = (1,)
        shape = shape + dim_sizes
        data_dims = len(data_shape) - 2
        shape = shape + (data_dims,)
        points = np.zeros(shape, np.float32)
        dim = 0
        for i in range(points.size):
            points.flat[i] = rand.uniform(-self.PERCENTAGE_OUT_OF_BOUNDS / 200 * (data_shape[dim + 1] - 1),
                                          (data_shape[dim + 1] - 1) * (1 + self.PERCENTAGE_OUT_OF_BOUNDS / 200))
            dim = (dim + 1) % data_dims
        return points

    def global_boundaries(self, device):
        # rand.seed(42)
        with tf.device(device):
            for i in range(self.N):
                data = self.generate_data()
                points = self.generate_points(data.shape)
                boundary = rand.choice(self.BOUNDARIES)
                data_placeholder = tf.placeholder(tf.float32, name="data_placeholder", shape=data.shape)
                points_placeholder = tf.placeholder(tf.float32, name="points_placeholder", shape=points.shape)
                cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
                gradient = np.zeros(cuda_resampled.shape, np.float32)
                for j in range(gradient.size):
                    gradient.flat[j] = rand.uniform(self.MIN_VALUE, self.MAX_VALUE)
                gradient_placeholder = tf.placeholder(tf.float32, name="gradient_placeholder", shape=gradient.shape)
                cuda_data_gradient = (tf.gradients(cuda_resampled, data_placeholder, gradient_placeholder))[0]
                cuda_points_gradient = (tf.gradients(cuda_resampled, points_placeholder, gradient_placeholder))[0]
                boundary_func = SUPPORTED_BOUNDARY[boundary.lower()]
                nifty_resampled = _resample_linear_niftynet(data_placeholder, points_placeholder, boundary,
                                                            boundary_func)
                nifty_data_gradient = (tf.gradients(nifty_resampled, data_placeholder, gradient_placeholder))[0]
                nifty_points_gradient = (tf.gradients(nifty_resampled, points_placeholder, gradient_placeholder))[0]
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    result = sess.run([cuda_resampled, nifty_resampled, cuda_data_gradient, nifty_data_gradient,
                                       cuda_points_gradient, nifty_points_gradient], feed_dict={data_placeholder: data,
                                                                                                points_placeholder:
                                                                                                    points,
                                                                                                gradient_placeholder:
                                                                                                    gradient})
                for k in range(3):
                    difference = result[2 * k + 1] - result[2 * k]
                    for j in range(difference.size):
                        assert -self.MAX_DIFFERENCE < difference.flat[j] < self.MAX_DIFFERENCE

    def test_global_boundaries(self):
        self.global_boundaries('/device:GPU:0')

    def mixed_boundaries(self, device):
        with tf.device(device):
            data = np.array([[[[1.0], [2.0], [3.0]],
                              [[4.0], [5.0], [6.0]],
                              [[7.0], [8.0], [9.0]]]])
            points = np.array([[[-0.5, -0.5], [2.5, 2.5]]])
            precomputed = np.array([[[0.25, 2.25], [0.5, 4.5], [1.0, 4.0], [0.75, 4.25]],
                                    [[0.5, 4.5], [1.0, 9.0], [2.0, 8.0], [1.5, 8.5]],
                                    [[2.0, 3.0], [4.0, 6.0], [5.0, 5.0], [4.5, 5.5]],
                                    [[1.25, 3.75], [2.5, 7.5], [3.5, 6.5], [3.0, 7.0]]])
            boundaries = ['zero', 'replicate', 'circular', 'reflect']
            for i in range(4):
                for j in range(4):
                    boundary = (boundaries[i], boundaries[j])
                    data_placeholder = tf.placeholder(tf.float32, name="data_placeholder", shape=data.shape)
                    points_placeholder = tf.placeholder(tf.float32, name="points_placeholder", shape=points.shape)
                    cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
                    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                        result = sess.run(cuda_resampled, feed_dict={data_placeholder: data, points_placeholder: points})
                    assert result.flat[0] == precomputed[i, j, 0]
                    assert result.flat[1] == precomputed[i, j, 1]
            boundary = [('zero', 'replicate'), ('circular', 'reflect')]
            points = np.array([[[-0.5, -0.5], [-0.5, 2.5], [2.5, -0.5], [2.5, 2.5]]])
            points_placeholder = tf.placeholder(tf.float32, name="points_placeholder", shape=points.shape)
            cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                result = sess.run(cuda_resampled, feed_dict={data_placeholder: data, points_placeholder: points})
            assert result.flat[0] == 1.0
            assert result.flat[1] == 1.25
            assert result.flat[2] == 8
            assert result.flat[3] == 8.5

    def test_mixed_boundaries(self):
        self.mixed_boundaries('/device:GPU:0')

    def batch_sizes(self, device):
        # rand.seed(14)
        with tf.device(device):
            for n in range(self.N):
                # Generate arrays
                boundary = rand.choice(self.BOUNDARIES)
                data = self.generate_data()
                points = self.generate_points(data.shape)
                data2 = np.zeros(data.shape)
                for x in range(data2.size):
                    data2.flat[x] = rand.uniform(self.MIN_VALUE, self.MAX_VALUE)
                points2 = np.zeros(points.shape)
                dim = 0
                dims = len(data.shape) - 2
                for x in range(points.size):
                    points2.flat[x] = rand.uniform(-self.PERCENTAGE_OUT_OF_BOUNDS / 200 * data.shape[dim + 1],
                                                   data.shape[dim + 1] * (1 + self.PERCENTAGE_OUT_OF_BOUNDS / 200))
                    dim = (dim + 1) % dims
                data_combined = np.concatenate((data, data2))
                points_combined = np.concatenate((points, points2))
                data_placeholder = tf.placeholder(tf.float32, name="data_placeholder", shape=data.shape)
                data_combined_placeholder = tf.placeholder(tf.float32, name="data_combined_placeholder",
                                                           shape=data_combined.shape)
                points_placeholder = tf.placeholder(tf.float32, name="points_placeholder", shape=points.shape)
                points_combined_placeholder = tf.placeholder(tf.float32, name="points_combined_placeholder",
                                                             shape=points_combined.shape)

                # batch_size = 2
                single = resample_cuda(data_placeholder, points_placeholder, boundary)
                single_data_gradient = (tf.gradients(single, data_placeholder))[0]
                single_points_gradient = (tf.gradients(single, points_placeholder))[0]
                combined = resample_cuda(data_combined_placeholder, points_combined_placeholder, boundary)
                combined_data_gradient = (tf.gradients(combined, data_combined_placeholder))[0]
                combined_points_gradient = (tf.gradients(combined, points_combined_placeholder))[0]
                with tf.Session() as sess, tf.device(device):
                    reference1 = sess.run([single, single_data_gradient, single_points_gradient],
                                          feed_dict={data_placeholder: data, points_placeholder: points})
                    reference2 = sess.run([single, single_data_gradient, single_points_gradient],
                                          feed_dict={data_placeholder: data2, points_placeholder: points2})
                    result = sess.run([combined, combined_data_gradient, combined_points_gradient],
                                      feed_dict={data_combined_placeholder: data_combined,
                                                 points_combined_placeholder: points_combined})
                for i in range(3):
                    reference = np.concatenate((reference1[i], reference2[i]))
                    for j in range(result[i].size):
                        assert abs(reference.flat[j] - result[i].flat[j]) < self.MAX_DIFFERENCE

                # data batch_size = 2
                combined = resample_cuda(data_combined_placeholder, points_placeholder, boundary)
                combined_data_gradient = (tf.gradients(combined, data_combined_placeholder))[0]
                combined_points_gradient = (tf.gradients(combined, points_placeholder))[0]
                with tf.Session() as sess, tf.device(device):
                    reference1 = sess.run([single, single_data_gradient, single_points_gradient],
                                          feed_dict={data_placeholder: data, points_placeholder: points})
                    reference2 = sess.run([single, single_data_gradient, single_points_gradient],
                                          feed_dict={data_placeholder: data2, points_placeholder: points})
                    result = sess.run([combined, combined_data_gradient, combined_points_gradient],
                                      feed_dict={data_combined_placeholder: data_combined,
                                                 points_placeholder: points})
                for i in range(3):
                    if i == 2:
                        reference = reference1[i] + reference2[i]
                    else:
                        reference = np.concatenate((reference1[i], reference2[i]))
                    for j in range(result[i].size):
                        assert abs(reference.flat[j] - result[i].flat[j]) < self.MAX_DIFFERENCE

                # points batch_size = 2
                combined = resample_cuda(data_placeholder, points_combined_placeholder, boundary)
                combined_data_gradient = (tf.gradients(combined, data_placeholder))[0]
                combined_points_gradient = (tf.gradients(combined, points_combined_placeholder))[0]
                with tf.Session() as sess, tf.device(device):
                    reference1 = sess.run([single, single_data_gradient, single_points_gradient],
                                          feed_dict={data_placeholder: data, points_placeholder: points})
                    reference2 = sess.run([single, single_data_gradient, single_points_gradient],
                                          feed_dict={data_placeholder: data, points_placeholder: points2})
                    result = sess.run([combined, combined_data_gradient, combined_points_gradient],
                                      feed_dict={data_placeholder: data, points_combined_placeholder: points_combined})
                for i in range(3):
                    if i == 1:
                        reference = reference1[i] + reference2[i]
                    else:
                        reference = np.concatenate((reference1[i], reference2[i]))
                    for j in range(result[i].size):
                        assert abs(reference.flat[j] - result[i].flat[j]) < self.MAX_DIFFERENCE

    def test_batch_sizes(self):
        self.batch_sizes('/device:GPU:0')

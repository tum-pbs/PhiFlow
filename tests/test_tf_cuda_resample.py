from unittest import TestCase
import random as rand

from phi.tf.flow import *

from phi.tf.tf_backend import *
from phi.tf.tf_backend import _resample_linear_niftynet

from phi.tf.tf_cuda_resample import resample_cuda


class TestTfCudaResample(TestCase):
    N = 50
    MAX_DIFFERENCE = 0.01
    MIN_VALUE = -10
    MAX_VALUE = 10
    BOUNDARIES = ['replicate', 'circular', 'symmetric']
    PERCENTAGE_OUT_OF_BOUNDS = 5.0

    def get_random_dims(self):
        dims = rand.randrange(1, 5)
        max_dim_size = 1000
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

    def generate_data(self, batch_size=1):
        dim_sizes = self.get_random_dims()
        shape = (batch_size,)
        shape = shape + dim_sizes
        shape = shape + (rand.randrange(1, 6),)
        # shape = shape + (4,)
        data = np.zeros(shape, np.float32)
        for i in range(data.size):
            data.flat[i] = rand.uniform(self.MIN_VALUE, self.MAX_VALUE)
        return data

    def generate_points(self, data_shape, batch_size=1):
        dim_sizes = self.get_random_dims()
        shape = (batch_size,)
        shape = shape + dim_sizes
        data_dims = len(data_shape) - 2
        shape = shape + (data_dims,)
        points = np.zeros(shape, np.float32)
        dim = 0
        for i in range(points.size):
            points.flat[i] = rand.uniform(-self.PERCENTAGE_OUT_OF_BOUNDS / 200 * data_shape[dim + 1],
                                          data_shape[dim + 1] * (1 + self.PERCENTAGE_OUT_OF_BOUNDS / 200))
            dim = (dim + 1) % data_dims
        return points

    def test_global_boundaries(self):
        rand.seed(42)
        for i in range(self.N):
            '''data_batch_size = points_batch_size = 1
            batch_size_selector = rand.randrange(4)
            if batch_size_selector == 1:
                points_batch_size = rand.randrange(4)
            elif batch_size_selector == 2:
                data_batch_size = rand.randrange(4)
            elif batch_size_selector == 3:
                data_batch_size = points_batch_size = rand.randrange(4)'''
            data = self.generate_data()
            points = self.generate_points(data.shape)
            boundary = rand.choice(self.BOUNDARIES)
            data_placeholder = tf.placeholder(tf.float32, name="data_placeholder", shape=data.shape)
            points_placeholder = tf.placeholder(tf.float32, name="points_placeholder", shape=points.shape)
            cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
            gradient = np.zeros(cuda_resampled.shape, np.float32)
            for j in range(gradient.size):
                gradient.flat[j] = rand.uniform(self.MIN_VALUE, self.MAX_VALUE)
            cuda_data_gradient = (tf.gradients(cuda_resampled, data_placeholder, gradient))[0]
            cuda_points_gradient = (tf.gradients(cuda_resampled, points_placeholder, gradient))[0]
            boundary_func = SUPPORTED_BOUNDARY[boundary.lower()]
            nifty_resampled = _resample_linear_niftynet(data_placeholder, points_placeholder, boundary, boundary_func)
            nifty_data_gradient = (tf.gradients(nifty_resampled, data_placeholder, gradient))[0]
            nifty_points_gradient = (tf.gradients(nifty_resampled, points_placeholder, gradient))[0]
            with tf.Session() as sess:
                result = sess.run([cuda_resampled, nifty_resampled, cuda_data_gradient, nifty_data_gradient,
                                   cuda_points_gradient, nifty_points_gradient], feed_dict={data_placeholder: data,
                                                                                            points_placeholder: points})
            for k in range(3):
                difference = result[2 * k + 1] - result[2 * k]
                for j in range(difference.size):
                    '''if not (-self.MAX_DIFFERENCE < difference.flat[j] < self.MAX_DIFFERENCE):
                        print(boundary)
                        print('cuda:', result[0].flat[j])
                        print('nifty:', result[1].flat[j])
                        index = np.unravel_index(j, result[0].shape)
                        point = points[index[:-1]]
                        print(point)'''
                    assert -self.MAX_DIFFERENCE < difference.flat[j] < self.MAX_DIFFERENCE

    def test_mixed_boundaries(self):
        data = np.array([[[[1.0], [2.0], [3.0]],
                          [[4.0], [5.0], [6.0]],
                          [[7.0], [8.0], [9.0]]]])
        points = np.array([[[-0.5, -0.5], [2.5, 2.5]]])
        precomputed = np.array([[[0.25, 2.25], [0.5, 4.5], [1.0, 4.0], [0.75, 4.25]],
                                [[0.5, 4.5], [1.0, 9.0], [2.0, 8.0], [1.5, 8.5]],
                                [[2.0, 3.0], [4.0, 6.0], [5.0, 5.0], [4.5, 5.5]],
                                [[1.25, 3.75], [2.5, 7.5], [3.5, 6.5], [3.0, 7.0]]])
        boundaries = ['zero', 'replicate', 'circular', 'symmetric']
        for i in range(4):
            for j in range(4):
                boundary = (boundaries[i], boundaries[j])
                data_placeholder = tf.placeholder(tf.float32, name="data_placeholder", shape=data.shape)
                points_placeholder = tf.placeholder(tf.float32, name="points_placeholder", shape=points.shape)
                cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
                with tf.Session() as sess:
                    result = sess.run(cuda_resampled, feed_dict={data_placeholder: data, points_placeholder: points})
                assert result.flat[0] == precomputed[i, j, 0]
                assert result.flat[1] == precomputed[i, j, 1]
        boundary = [('zero', 'replicate'), ('circular', 'symmetric')]
        points = np.array([[[-0.5, -0.5], [-0.5, 2.5], [2.5, -0.5], [2.5, 2.5]]])
        points_placeholder = tf.placeholder(tf.float32, name="points_placeholder", shape=points.shape)
        cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
        with tf.Session() as sess:
            result = sess.run(cuda_resampled, feed_dict={data_placeholder: data, points_placeholder: points})
        assert result.flat[0] == 1.0
        assert result.flat[1] == 1.25
        assert result.flat[2] == 8
        assert result.flat[3] == 8.5


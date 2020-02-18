from unittest import TestCase

import numpy as np

from phi.geom import AABox
from phi.math.nd import _dim_shifted
from phi.tf import tf

# pylint: disable-msg = redefined-builtin, redefined-outer-name, unused-wildcard-import, wildcard-import
from phi.math import *

if tf.__version__[0] == '2':
    print('Adjusting for tensorflow 2.0')
    tf = tf.compat.v1
    tf.disable_eager_execution()


# placeholder, variable tested in test_tensorflow.py


class TestMath(TestCase):

    def test_fft(self):
        tf.InteractiveSession()
        for dims in range(1, 4):
            shape = [2] + [4]*dims + [3]
            x_np = np.random.randn(*shape) + 1j * np.random.randn(*shape)
            x_np = x_np.astype(np.complex64)
            x_tf = tf.constant(x_np, tf.complex64)

            k_np = fft(x_np)
            k_tf = fft(x_tf)

            self.assertLess(max(abs(k_np - k_tf.eval())), 1e-3)

            x_np = ifft(k_np)
            x_tf = ifft(k_tf)

            self.assertLess(max(abs(x_np - x_tf.eval())), 1e-3)

    def test_laplace_padding(self):
        tf.InteractiveSession()
        for dims in range(1, 4):
            shape = [2] + [4]*dims + [3]
            a = zeros(shape)
            l = laplace(a, padding='replicate')
            np.testing.assert_equal(l, 0)
            np.testing.assert_equal(l.shape, a.shape)
            l = laplace(a, padding='reflect')
            np.testing.assert_equal(l, 0)
            np.testing.assert_equal(l.shape, a.shape)
            l = laplace(a, padding='circular')
            np.testing.assert_equal(l, 0)
            np.testing.assert_equal(l.shape, a.shape)
            l = laplace(a, padding='valid')
            np.testing.assert_equal(l, 0)
            np.testing.assert_equal(l.shape, [2] + [2]*dims + [3])

    def test_struct_broadcast(self):
        s = {'a': 0, 'b': 1}
        result = cos(s)
        self.assertEqual(result['a'], 1)
        self.assertEqual(maximum(0.5, {'a': 0, 'b': 1}), {'a': 0.5, 'b': 1})
        self.assertEqual(maximum({'a': 0, 'b': 1.5}, {'a': 0.5, 'b': 1}), {'a': 0.5, 'b': 1.5})

    def test_pad_wrap(self):
        tf.InteractiveSession()
        # --- 1D ---
        a = np.array([1,2,3,4,5])
        a_ = pad(a, [[2,3]], mode='wrap')
        np.testing.assert_equal(a_, [4,5,1,2,3,4,5,1,2,3])
        a = tf.constant(a)
        a_ = pad(a, [[2,3]], mode='wrap').eval()
        np.testing.assert_equal(a_, [4,5,1,2,3,4,5,1,2,3])
        # --- 2D + batch ---
        t = [[3,1,2,3,1], [6,4,5,6,4], [3,1,2,3,1]]
        a = np.array([[1,2,3],[4,5,6]]).reshape([1,2,3,1])
        a_ = pad(a, [[0,0], [0,1], [1,1], [0,0]], mode='wrap')
        np.testing.assert_equal(a_.shape, [1,3,5,1])
        np.testing.assert_equal(a_.reshape([3,5]), t)
        a = tf.constant(a)
        a_ = pad(a, [[0,0], [0,1], [1,1], [0,0]], mode='wrap').eval()
        np.testing.assert_equal(a_.shape, [1,3,5,1])
        np.testing.assert_equal(a_.reshape([3,5]), t)

    def test_multimode_pad(self):
        a = np.array([[1,2], [3,4]])
        print(a)
        p = pad(a, [[1,1], [1,1]], mode=['replicate', ['wrap', 'constant']], constant_values=[0, [0, 10]])
        np.testing.assert_equal(p[0,1:-1], [1,2])
        np.testing.assert_equal(p[3,1:-1], [3,4])
        np.testing.assert_equal(p[1:-1,0], [2,4])
        np.testing.assert_equal(p[1:-1,3], [10, 10])
        print(p)
        tf.InteractiveSession()
        a_tf = tf.constant(a, tf.float32, shape=(2,2))
        p_tf = pad(a_tf, [[1,1], [1,1]], mode=['replicate', ['wrap', 'constant']], constant_values=[0, [0, 10]])
        np.testing.assert_equal(p, p_tf.eval())

    def test_div_no_nan(self):
        x = np.array([1, -1, 0, 1, -1], np.float32)
        y = np.array([1,  2, 0, 0, 0], np.float32)
        result = divide_no_nan(x, y)
        np.testing.assert_equal(result, [1, -0.5, 0, 0, 0])
        sess = tf.InteractiveSession()
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        result = divide_no_nan(x, y).eval()
        np.testing.assert_equal(result, [1, -0.5, 0, 0, 0])

    def test_dim_shifted(self):
        # --- 1D ---
        tensor = np.expand_dims(np.expand_dims(np.arange(10), axis=-1), axis=0)
        lower, center, upper = _dim_shifted(tensor, 0, (-1, 0, 1), components=0)
        np.testing.assert_equal(lower[0,:,0], np.arange(8))
        np.testing.assert_equal(center[0,:,0], np.arange(1,9))
        np.testing.assert_equal(upper[0,:,0], np.arange(2,10))
        # --- 2D ---
        tensor = np.ones([1, 4, 4, 2])
        lower, upper = _dim_shifted(tensor, 0, (0, 1), diminish_others=(0, 1), components=0)
        np.testing.assert_equal(lower.shape, (1, 3, 3, 1))
        np.testing.assert_equal(upper.shape, (1, 3, 3, 1))

    def test_gradient(self):
        # --- 1D ---
        tensor = np.expand_dims(np.expand_dims(np.arange(5), axis=-1), axis=0)
        grad = gradient(tensor, padding='replicate')
        np.testing.assert_equal(grad[0,:,0], [1, 1, 1, 1, 0])
        grad = gradient(tensor, padding='circular')
        np.testing.assert_equal(grad[0,:,0], [1, 1, 1, 1, -4])
        grad = gradient(tensor, dx=0.1, padding='replicate')
        np.testing.assert_equal(grad[0,:,0], [10, 10, 10, 10, 0])

    def test_upsample_downsample(self):
        # --- 1D ---
        tensor = np.expand_dims(np.expand_dims(np.arange(5), axis=-1), axis=0)
        up = upsample2x(tensor)
        inverted = downsample2x(up)
        np.testing.assert_equal(inverted[:, 1:-1, :], tensor[:, 1:-1, :])

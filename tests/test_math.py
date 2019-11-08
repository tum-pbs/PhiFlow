import tensorflow as tf
from unittest import TestCase

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

    def test_laplace(self):
        tf.InteractiveSession()
        for dims in range(1, 4):
            shape = [2] + [4]*dims + [3]
            a = zeros(shape)
            l = laplace(a, padding='symmetric')
            np.testing.assert_equal(l, 0)
            np.testing.assert_equal(l.shape, a.shape)
            l = laplace(a, padding='reflect')
            np.testing.assert_equal(l, 0)
            np.testing.assert_equal(l.shape, a.shape)
            l = laplace(a, padding='cyclic')
            np.testing.assert_equal(l, 0)
            np.testing.assert_equal(l.shape, a.shape)
            l = laplace(a, padding='valid')
            np.testing.assert_equal(l, 0)
            np.testing.assert_equal(l.shape, [2] + [2]*dims + [3])

    def test_struct_broadcast(self):
        s = {'a': 0, 'b': 1}
        result = cos(s)
        self.assertEqual(result['a'], 1)
        self.assertEqual(math.maximum(0.5, {'a': 0, 'b': 1}), {'a': 0.5, 'b': 1})
        self.assertEqual(math.maximum({'a': 0, 'b': 1.5}, {'a': 0.5, 'b': 1}), {'a': 0.5, 'b': 1.5})

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
from unittest import TestCase
from phi.math import *
import tensorflow as tf


# placeholder, variable tested in test_tensorflow.py


class TestMath(TestCase):

    def test_fft(self):
        tf.InteractiveSession()
        for dims in range(1, 4):
            shape = [2]+[4]*dims+[3]
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
            shape = [2]+[4]*dims+[3]
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
            np.testing.assert_equal(l.shape, [2]+[2]*dims+[3])

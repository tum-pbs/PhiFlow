from unittest import TestCase

from phi import math
from phi.physics import sph
from phiml.math import spatial


class TestSPH(TestCase):

    def test_evaluate_kernel(self):
        eps = 1e-8
        with math.precision(64):
            r = math.linspace(0, 1, spatial(x=100))
            for kernel in ['quintic-spline', 'wendland-c2', 'poly6']:
                val, grad = sph.evaluate_kernel(r, r, 1, 1, kernel, derivative=(0, 1))
                h_val, = sph.evaluate_kernel(r + eps, r + eps, 1, 1, kernel, derivative=(0,))
                fd_grad = (h_val - val) / eps
                math.assert_close(fd_grad, grad, rel_tolerance=1e-5, abs_tolerance=1e-5)  # gradient correct
                math.assert_close(val.x[-1], 0)  # W(1) = 0
                math.assert_close(grad.x[-1], 0)  # dW(1) = 0
                math.assert_close(1, math.sum(val / 100), abs_tolerance=0.01)  # integral = 1

    def test_evaluate_kernel_scaled(self):
        eps = 1e-8
        with math.precision(64):
            r = math.linspace(0, 10, spatial(x=100))
            for kernel in ['quintic-spline', 'wendland-c2', 'poly6']:
                val, grad = sph.evaluate_kernel(r, r, 10, 1, kernel, derivative=(0, 1))
                h_val, = sph.evaluate_kernel(r + eps, r + eps, 10, 1, kernel, derivative=(0,))
                fd_grad = (h_val - val) / eps
                math.assert_close(fd_grad, grad, rel_tolerance=1e-5, abs_tolerance=1e-5)
                math.assert_close(val.x[-1], 0)
                math.assert_close(grad.x[-1], 0)
                math.assert_close(1, math.sum(val / 10), abs_tolerance=0.01)

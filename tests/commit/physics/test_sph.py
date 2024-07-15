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
                result = sph.evaluate_kernel(r, r, 1, 1, kernel, types=['kernel', 'grad', 'laplace'])
                val, grad, laplace = result['kernel'], result['grad'], result['laplace']
                h_val = sph.evaluate_kernel(r + eps, r + eps, 1, 1, kernel, types=['kernel'])['kernel']
                fd_grad = (h_val - val) / eps
                h_grad = sph.evaluate_kernel(r + eps, r + eps, 1, 1, kernel, types=['grad'])['grad']
                fd_laplace = (h_grad - grad) / eps
                math.assert_close(fd_grad, grad, rel_tolerance=1e-5, abs_tolerance=1e-5)
                math.assert_close(fd_laplace, laplace, rel_tolerance=1e-5, abs_tolerance=1e-5)
                math.assert_close(val.x[-1], 0)
                math.assert_close(grad.x[-1], 0)
                math.assert_close(.5, math.sum(val) / 100, abs_tolerance=0.1)

    def test_evaluate_kernel_scaled(self):
        eps = 1e-8
        with math.precision(64):
            r = math.linspace(0, 10, spatial(x=100))
            for kernel in ['quintic-spline', 'wendland-c2', 'poly6']:
                result = sph.evaluate_kernel(r, r, 10, 1, kernel, types=['kernel', 'grad', 'laplace'])
                val, grad, laplace = result['kernel'], result['grad'], result['laplace']
                h_val = sph.evaluate_kernel(r + eps, r + eps, 10, 1, kernel, types=['kernel'])['kernel']
                fd_grad = (h_val - val) / eps
                h_grad = sph.evaluate_kernel(r + eps, r + eps, 10, 1, kernel, types=['grad'])['grad']
                fd_laplace = (h_grad - grad) / eps
                math.assert_close(fd_grad, grad, rel_tolerance=1e-5, abs_tolerance=1e-5)
                math.assert_close(fd_laplace, laplace, rel_tolerance=1e-5, abs_tolerance=1e-5)
                math.assert_close(val.x[-1], 0)
                math.assert_close(grad.x[-1], 0)
                math.assert_close(.5, math.sum(val) / 10, abs_tolerance=0.1)

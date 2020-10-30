from itertools import product
from unittest import TestCase
from phi import math
from phi.math import tensor, extrapolation, Tensor

import numpy as np


class AbstractTestMathND(TestCase):

    def _test_scalar_gradient(self, ones: Tensor):
        cases = dict(difference=('central', 'forward', 'backward'),
                     padding=(extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            try:
                scalar_grad = math.gradient(ones, dx=0.1, **case_dict)
                math.assert_close(scalar_grad, 0)
                self.assertEqual(scalar_grad.shape.names, ('batch', 'x', 'y', 'gradient'))
                self.assertEqual(scalar_grad.shape.sizes, (2, 4, 3, 2))
            except BaseException as e:
                raise AssertionError(e, case_dict)

    def _test_vector_gradient(self, meshgrid: Tensor, save=False):
        cases = dict(difference=('central', 'forward', 'backward'),
                     padding=(extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            grad = math.gradient(meshgrid, dx=0.1, **case_dict)
            if save:
                math.print(meshgrid, 'Base')
                np.save('reference_data/vector_grad_%s.npy' % '_'.join(f'{key}={value}' for key, value in case_dict.items()), grad.numpy())
                math.print(grad, str(case_dict))
            else:
                ref = np.load('reference_data/vector_grad_%s.npy' % '_'.join(f'{key}={value}' for key, value in case_dict.items()))
                math.assert_close(grad, ref)

    def _test_vector_laplace(self, meshgrid: Tensor, save=False):
        cases = dict(padding=(extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            laplace = math.laplace(meshgrid, dx=0.1, **case_dict)
            if save:
                math.print(meshgrid, 'Base')
                np.save('reference_data/laplace_%s.npy' % '_'.join(f'{key}={value}' for key, value in case_dict.items()), laplace.numpy())
                math.print(laplace, str(case_dict))
            else:
                ref = np.load('reference_data/laplace_%s.npy' % '_'.join(f'{key}={value}' for key, value in case_dict.items()))
                math.assert_close(laplace, ref)


class TestMathNDNumpy(AbstractTestMathND):

    def test_gradient_scalar(self):
        scalar = tensor(np.ones([2, 4, 3, 1]))
        self._test_scalar_gradient(scalar)

    def test_gradient_vector(self):
        meshgrid = math.meshgrid((0, 1, 2, 3), (0, -1))
        self._test_vector_gradient(meshgrid, save=False)

    def test_vector_laplace(self):
        meshgrid = math.meshgrid((0, 1, 2, 3), (0, -1))
        self._test_vector_laplace(meshgrid, save=False)

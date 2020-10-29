from unittest import TestCase
from phi.flow import *
from phi.math._shape import *
from phi.math._tensors import tensor

import numpy as np


class TestMathND(TestCase):

    def test_scalar_gradient(self):
        scalar = tensor(np.ones([2, 4, 3, 1]))
        scalar_grad = math.gradient(scalar, dx=0.1, difference='central', padding='replicate')
        scalar_grad.assert_close(0)
        self.assertEqual('(batch=2, y=4, x=3, 2)', repr(scalar_grad.shape))
        axis_grad = math.gradient(scalar, dx=0.1, difference='forward', padding='replicate', axes=('x',))
        self.assertEqual('(batch=2, y=4, x=3, 1)', repr(axis_grad.shape))

    def test_vector_gradient(self):
        vector = tensor(np.ones([2, 4, 3, 3]))
        vector_grad = math.gradient(vector, dx=0.1, difference='central', padding='replicate')
        self.assertEqual('(batch=2, y=4, x=3, 2, 3)', repr(vector_grad.shape))

    def test_scalar_laplace(self):
        scalar = tensor(np.ones([2, 4, 3, 1]))
        scalar_laplace = math.laplace(scalar, dx=0.1, padding='replicate')
        self.assertEqual(scalar.shape, scalar_laplace.shape)
        scalar_laplace.assert_close(0)

    def test_vector_laplace(self):
        vector = tensor(np.ones([2, 4, 3, 3]))
        vector_laplace = math.laplace(vector, dx=0.1, padding='replicate')
        self.assertEqual(vector.shape, vector_laplace.shape)
        vector_laplace.assert_close(0)

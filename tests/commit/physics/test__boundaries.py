from unittest import TestCase

import numpy
import torch
import tensorflow as tf

from phi import math
from phi.field import Noise
from phi.physics import Domain
from phi import torch as _  # register PyTorch backend
from phi import tf as _  # register TensorFlow backend


class TestFieldMath(TestCase):

    def test_domain_grid_from_constant(self):
        domain = Domain(x=4, y=3)
        # int / float
        grid = domain.scalar_grid(1)
        math.assert_close(grid.values, 1)
        # complex
        grid = domain.grid(1+1j)
        math.assert_close(grid.values, 1+1j)
        # NumPy
        grid = domain.grid(numpy.array(1))
        math.assert_close(grid.values, 1)
        # PyTorch
        grid = domain.grid(torch.tensor(1, dtype=torch.float32))
        math.assert_close(grid.values, 1)
        # TensorFlow
        grid = domain.grid(tf.constant(1, tf.float32))
        math.assert_close(grid.values, 1)

    def test_domain_grid_from_field(self):
        large_grid = Domain(x=4, y=3).grid(Noise())
        small_grid = Domain(x=3, y=2).grid(large_grid)
        math.assert_close(large_grid.values.x[:-1].y[:-1], small_grid.values)

    def test_domain_grid_from_tensor(self):
        domain = Domain(x=4, y=3)
        grid = domain.grid(Noise(vector=2))
        grid2 = domain.grid(grid.values)
        math.assert_close(grid.values, grid2.values)

    def test_domain_grid_from_function(self):
        grid = Domain(x=4, y=3).grid(lambda x: math.sum(x ** 2, 'vector'))
        math.assert_close(grid.values.x[0].y[0], 0.5)
        self.assertEqual(grid.shape.volume, 12)
        grid = Domain(x=4, y=3).grid(lambda x: 1)
        math.assert_close(grid.values, 1)

    def test_domain_grid_memory_allocation(self):
        domain = Domain(x=10000, y=10000, z=10000, w=10000)
        grid = domain.grid()
        self.assertEqual((10000,) * 4, grid.shape.sizes)
        sgrid = domain.staggered_grid()
        self.assertEqual((10000, 10000, 10000, 10000, 4), sgrid.shape.sizes)

    def test_custom_spatial_dims(self):
        domain = Domain(a=4, b=3)
        grid = domain.scalar_grid(1)
        self.assertEqual(math.shape(a=4, b=3), grid.shape)
        grid = domain.staggered_grid(1)
        self.assertEqual(math.shape(a=4, b=3, vector=2), grid.shape)

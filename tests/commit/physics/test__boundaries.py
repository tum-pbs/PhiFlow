from unittest import TestCase

import numpy

from phi import math
from phi.field import Noise, CenteredGrid, StaggeredGrid
from phi.math import spatial, channel
from phi.physics._boundaries import Domain


class TestFieldMath(TestCase):

    def test_domain_grid_from_constant(self):
        domain = Domain(x=4, y=3)
        # int / float
        grid = domain.scalar_grid(1)
        math.assert_close(grid.values, 1)
        # complex
        grid = domain.scalar_grid(1+1j)
        math.assert_close(grid.values, 1+1j)
        # NumPy
        grid = domain.scalar_grid(numpy.array(1))
        math.assert_close(grid.values, 1)

    def test_domain_grid_from_field(self):
        large_grid = Domain(x=4, y=3).grid(Noise())
        small_grid = Domain(x=3, y=2).grid(large_grid)
        math.assert_close(large_grid.values.x[:-1].y[:-1], small_grid.values)

    def test_domain_grid_from_tensor(self):
        domain = Domain(x=4, y=3)
        grid = domain.vector_grid(Noise(vector=2))
        grid2 = domain.vector_grid(grid.values)
        math.assert_close(grid.values, grid2.values)

    def test_domain_grid_from_function(self):
        grid = Domain(x=4, y=3).scalar_grid(lambda x: math.sum(x ** 2, 'vector'))
        math.assert_close(grid.values.x[0].y[0], 0.5)
        self.assertEqual(grid.shape.volume, 12)
        grid = Domain(x=4, y=3).scalar_grid(lambda x: math.ones(x.shape.non_channel))
        math.assert_close(grid.values, 1)

    def test_domain_grid_memory_allocation(self):
        grid = CenteredGrid(0, x=10000, y=10000, z=10000, w=10000)
        self.assertEqual((10000,) * 4, grid.shape.sizes)
        sgrid = StaggeredGrid(0, x=10000, y=10000, z=10000, w=10000)
        self.assertEqual((10000, 10000, 10000, 10000, 4), sgrid.shape.sizes)

    def test_custom_spatial_dims(self):
        domain = Domain(a=4, b=3)
        grid = domain.scalar_grid(1)
        self.assertEqual(spatial(a=4, b=3), grid.shape)
        grid = domain.staggered_grid(1)
        self.assertEqual(spatial(a=4, b=3) & channel(vector=2), grid.shape)

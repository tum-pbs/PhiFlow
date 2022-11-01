from unittest import TestCase

import numpy

from phi import math
from phi.field import Noise, CenteredGrid, StaggeredGrid
from phi.math import spatial, channel


class TestFieldMath(TestCase):

    def test_grid_from_constant(self):
        # int / float
        grid = CenteredGrid(1, x=4, y=3)
        math.assert_close(grid.values, 1)
        # complex
        grid = CenteredGrid(1+1j, x=4, y=3)
        math.assert_close(grid.values, 1+1j)
        # NumPy
        grid = CenteredGrid(numpy.array(1), x=4, y=3)
        math.assert_close(grid.values, 1)

    def test_grid_from_field(self):
        large_grid = CenteredGrid(Noise(), x=4, y=3)
        small_grid = CenteredGrid(large_grid, x=3, y=2)
        math.assert_close(large_grid.values.x[:-1].y[:-1], small_grid.values)

    def test_grid_from_tensor(self):
        grid = CenteredGrid(Noise(vector=2), x=4, y=3)
        grid2 = CenteredGrid(grid.values)
        math.assert_close(grid.values, grid2.values)

    def test_grid_from_function(self):
        grid = CenteredGrid(lambda x: math.sum(x ** 2, 'vector'), x=4, y=3)
        math.assert_close(grid.values.x[0].y[0], 0.5)
        self.assertEqual(grid.shape.volume, 12)
        grid = CenteredGrid(lambda x: math.ones(x.shape.non_channel), x=4, y=3)
        math.assert_close(grid.values, 1)

    def test_grid_memory_allocation(self):
        grid = CenteredGrid(0, x=10000, y=10000, z=10000, w=10000)
        self.assertEqual((10000,) * 4, grid.shape.sizes)
        sgrid = StaggeredGrid(0, x=10000, y=10000, z=10000, w=10000)
        self.assertEqual((10000, 10000, 10000, 10000, 4), sgrid.shape.sizes)

    def test_custom_spatial_dims(self):
        grid = CenteredGrid(1, a=4, b=3)
        self.assertEqual(spatial(a=4, b=3), grid.shape)
        grid = StaggeredGrid(1, a=4, b=3)
        self.assertEqual(spatial(a=4, b=3) & channel(vector='a,b'), grid.shape)

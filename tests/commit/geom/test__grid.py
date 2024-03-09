from unittest import TestCase


from phi import math
from phi.geom import UniformGrid, Box
from phi.math import batch, channel
from phi.math.magic import Shaped, Sliceable, Shapable
from phiml.math import vec, spatial


class TestBox(TestCase):

    def test_slice_int(self):
        grid = UniformGrid(x=4, y=3, z=2)
        self.assertEqual(grid[{'z': 0}].resolution, spatial(x=4, y=3))
        self.assertEqual(grid[{'z': 0}].bounds, grid.bounds['x,y'])

    def test_slice(self):
        grid = UniformGrid(x=4, y=3, z=2)
        self.assertEqual(grid[{'z': slice(1, 2)}].resolution, spatial(x=4, y=3, z=1))
        self.assertEqual(grid[{'z': slice(1, 2)}].bounds, Box(vec(x=0, y=0, z=1), vec(x=4, y=3, z=2)))

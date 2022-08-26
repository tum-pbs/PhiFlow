from unittest import TestCase

from phi.math import tensor, batch, is_finite
from phi.field import *


class TestNoise(TestCase):

    def test_multi_k(self):
        grid = CenteredGrid(Noise(vector='x,y', scale=tensor([1, 2], batch('batch'))), x=8, y=8)
        self.assertTrue(is_finite(grid.values).all)

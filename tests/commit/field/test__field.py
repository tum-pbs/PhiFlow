from unittest import TestCase

from phi import math
from phi.field import CenteredGrid, Noise
from phi.geom import Box
from phi.math import batch, spatial


class TestField(TestCase):

    def test_concat(self):
        grids = [CenteredGrid(Noise(batch(b=1)), x=32, y=32) for _ in range(2)]
        concat = math.concat(grids, 'b')
        self.assertEqual(2, concat.b.size)

    def test_stack(self):
        grids = [CenteredGrid(Noise(), x=32, y=32) for _ in range(2)]
        b_stack = math.stack(grids, batch('b'))
        self.assertEqual(2, b_stack.b.size)
        z_stack = math.stack(grids, spatial('z'), bounds=Box(z=.2))
        math.assert_close(.2, z_stack.bounds.size.vector['z'])
        z_stack = math.stack(grids, spatial('z'))
        math.assert_close(2, z_stack.bounds.size.vector['z'])

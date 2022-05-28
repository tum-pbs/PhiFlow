from unittest import TestCase

from phi import math
from phi.field import CenteredGrid, Noise, assert_close
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

    def test_embedding(self):
        grid = CenteredGrid(lambda x: math.vec_length(x), 1, x=4, y=4)
        math.print(grid.values, "Base Grid")
        inner = CenteredGrid(0, grid, x=2, y=2, bounds=Box(x=(1, 3), y=(1, 3)))
        resampled = inner @ grid
        math.assert_close(inner.values, resampled.values.x[1:-1].y[1:-1])
        math.assert_close(grid.values.x[:1].y[1:-1], resampled.values.x[:1].y[1:-1])
        math.assert_close(grid.values.x[-1:].y[1:-1], resampled.values.x[-1:].y[1:-1])
        math.assert_close(grid.values.x[1:-1].y[:1], resampled.values.x[1:-1].y[:1])
        math.assert_close(grid.values.x[1:-1].y[-1:], resampled.values.x[1:-1].y[-1:])

    def test_embedding_resample(self):
        p = CenteredGrid(Noise(), x=10, y=10, bounds=Box(x=100, y=100))
        p_emb_x0 = CenteredGrid(p, p, x=5, y=5, bounds=Box(x=(10, 60), y=(10, 60)))
        p_back = CenteredGrid(p_emb_x0, p.extrapolation, p.bounds, p.resolution)
        assert_close(0, p_back - p)

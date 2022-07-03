from unittest import TestCase

from phi import math
from phi.field import CenteredGrid, Noise, assert_close, AngularVelocity
from phi.geom import Box, Sphere
from phi.math import batch, spatial, vec


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

    def test_slice_str(self):
        vgrid = CenteredGrid(Noise(col='r,g,b'), x=4, y=3)
        math.assert_close(vgrid.col[0], vgrid['r'])
        matrix_grid = CenteredGrid(Noise(col='r,g,b', vec='x,y'), x=4, y=3)
        math.assert_close(matrix_grid.col[(0, 1)].vec['y'], matrix_grid['r,g', 'y'])

    def test_legacy_resampling(self):
        for obj in [AngularVelocity(location=vec(x=0, y=0)), Sphere(x=0, y=0, radius=1)]:
            resampled = obj >> CenteredGrid(0, x=4, y=3)
            self.assertIsInstance(resampled, CenteredGrid)

from unittest import TestCase

from phi import math
from phi.math import extrapolation


def assert_not_close(*tensors, rel_tolerance, abs_tolerance):
    try:
        math.assert_close(*tensors, rel_tolerance, abs_tolerance)
        raise BaseException(AssertionError('Values are not close'))
    except AssertionError:
        pass


class TestMathFunctions(TestCase):

    def test_assert_close(self):
        math.assert_close(math.zeros(a=10), math.zeros(a=10), math.zeros(a=10), rel_tolerance=0, abs_tolerance=0)
        assert_not_close(math.zeros(a=10), math.ones(a=10), rel_tolerance=0, abs_tolerance=0)
        for scale in (1, 0.1, 10):
            math.assert_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0, abs_tolerance=scale)
            math.assert_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=1, abs_tolerance=0)
            assert_not_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0.9, abs_tolerance=0)
            assert_not_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0, abs_tolerance=0.9 * scale)
        math.set_precision(64)
        assert_not_close(math.zeros(a=10), math.ones(a=10) * 1e-100, rel_tolerance=0, abs_tolerance=0)
        math.assert_close(math.zeros(a=10), math.ones(a=10) * 1e-100, rel_tolerance=0, abs_tolerance=1e-15)

    def test_concat(self):
        c = math.concat([math.zeros(b=3, a=2), math.ones(a=2, b=4)], 'b')
        self.assertEqual(2, c.shape.a)
        self.assertEqual(7, c.shape.b)
        math.assert_close(c.b[:3], 0)
        math.assert_close(c.b[3:], 1)

    def test_nonzero(self):
        c = math.concat([math.zeros(b=3, a=2), math.ones(a=2, b=4)], 'b')
        nz = math.nonzero(c)
        self.assertEqual(nz.shape.nonzero, 8)
        self.assertEqual(nz.shape.vector, 2)

    def test_maximum(self):
        v = math.ones(x=4, y=3, vector=2)
        maximum = math.maximum(0, v)
        math.assert_close(math.maximum(0, v), 1)
        math.assert_close(math.maximum(0, -v), 0)

    # TODO: Fix
    def test_resample(self):
        grid = math.sum(math.meshgrid(x=[1, 2, 3], y=[0, 3]), 'vector')  # 1 2 3 | 4 5 6
        coords = math.tensor([(0, 0), (0.5, 0), (0, 0.5), (-2, -1)], names=('list', 'vector'))
        closest = math.closest_grid_values(grid, coords, extrapolation.ZERO)
        interp = math.grid_sample(grid, coords, extrapolation.ZERO)
        math.assert_close(interp, [1, 1.5, 2.5, 0])

    def test_nonzero_batched(self):
        grid = math.tensor([[(0, 1)], [(0, 0)]], 'batch,x,y')
        nz = math.nonzero(grid, list_dim='nonzero', index_dim='vector')
        self.assertEqual(('batch', 'nonzero', 'vector'), nz.shape.names)
        self.assertEqual(1, nz.batch[0].shape.nonzero)
        self.assertEqual(0, nz.batch[1].shape.nonzero)

# Legacy test to be fixed
# def _resample_test(mode, constant_values, expected):
#    grid = np.tile(np.reshape(np.array([[1,2], [4,5]]), [1,2,2,1]), [1, 1, 1, 2])
#    resampled = helper_resample(grid, coords, mode, constant_values, SciPyBackend())
#    np.testing.assert_equal(resampled[..., 0], resampled[..., 1])
#    np.testing.assert_almost_equal(expected, resampled[0, :, 0], decimal=5)

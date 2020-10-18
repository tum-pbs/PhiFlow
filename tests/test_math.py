from unittest import TestCase

from phi import math
from phi.math import extrapolation


# placeholder, variable tested in test_tensorflow.py


class TestMath(TestCase):

    def test_resample(self):
        grid = math.sum(math.meshgrid([1, 2, 3], [0, 3]), 'vector')  # 1 2 3 | 4 5 6
        coords = math.tensor([(0, 0), (0.5, 0), (0, 0.5), (-2, -1)], names=('list', 'vector'))
        closest = math.closest_grid_values(grid, coords, extrapolation.ZERO)
        interp = math.grid_sample(grid, coords, extrapolation.ZERO)
        math.assert_close(interp, [1, 1.5, 2.5, 0])


def _resample_test(mode, constant_values, expected):
    grid = np.tile(np.reshape(np.array([[1,2], [4,5]]), [1,2,2,1]), [1, 1, 1, 2])
    resampled = helper_resample(grid, coords, mode, constant_values, SciPyBackend())
    np.testing.assert_equal(resampled[..., 0], resampled[..., 1])
    np.testing.assert_almost_equal(expected, resampled[0, :, 0], decimal=5)

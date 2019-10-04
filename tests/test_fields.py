from unittest import TestCase
from phi.field import *
from phi.math.geom import *


class TestMath(TestCase):

    def test_compatibility(self):
        f = CenteredGrid('f', box[0:3, 0:4], math.zeros([1,3,4,1]))
        g = CenteredGrid('g', box[0:3, 0:4], math.zeros([1,3,3,1]))
        np.testing.assert_equal(f.dx, [1, 1])
        self.assertTrue(f.points.compatible(f))
        self.assertFalse(f.compatible(g))

    def test_inner_interpolation(self):
        data = math.zeros([1, 2, 3, 1])
        data[0, :, :, 0] = [[1,2,3], [4,5,6]]
        f = CenteredGrid('f', box[0:2, 0:3], data)
        g = CenteredGrid('g', box[0:2, 0.5:2.5], math.zeros([1,2,2,1]))
        # Resample optimized
        resampled = f.resample(g, force_optimization=True)
        self.assertTrue(resampled.compatible(g))
        np.testing.assert_equal(resampled.data[0,...,0], [[1.5, 2.5], [4.5, 5.5]])
        # Resample unoptimized
        resampled2 = Field.resample(f, g)
        self.assertTrue(resampled2.compatible(g))
        np.testing.assert_equal(resampled2.data[0,...,0], [[1.5, 2.5], [4.5, 5.5]])

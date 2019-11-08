from unittest import TestCase

from phi.geom import *
from phi.physics.field import *


class TestMath(TestCase):

    def test_compatibility(self):
        f = CenteredGrid('f', box[0:3, 0:4], math.zeros([1, 3, 4, 1]))
        g = CenteredGrid('g', box[0:3, 0:4], math.zeros([1, 3, 3, 1]))
        np.testing.assert_equal(f.dx, [1, 1])
        self.assert_(f.points.compatible(f))
        self.assertFalse(f.compatible(g))

    def test_inner_interpolation(self):
        data = math.zeros([1, 2, 3, 1])
        data[0, :, :, 0] = [[1, 2, 3], [4, 5, 6]]
        f = CenteredGrid('f', box[0:2, 0:3], data)
        g = CenteredGrid('g', box[0:2, 0.5:2.5], math.zeros([1, 2, 2, 1]))
        # Resample optimized
        resampled = f.at(g, force_optimization=True)
        self.assert_(resampled.compatible(g))
        np.testing.assert_equal(resampled.data[0, ..., 0], [[1.5, 2.5], [4.5, 5.5]])
        # Resample unoptimized
        resampled2 = Field.at(f, g)
        self.assert_(resampled2.compatible(g))
        np.testing.assert_equal(resampled2.data[0, ..., 0], [[1.5, 2.5], [4.5, 5.5]])

    def test_staggered_interpolation(self):
        # 2x2 cells
        data_x = math.zeros([1, 2, 3, 1])
        data_x[0, :, :, 0] = [[1, 2, 3], [4, 5, 6]]
        data_y = math.zeros([1, 3, 2, 1])
        data_y[0, :, :, 0] = [[-1, -2], [-3, -4], [-5, -6]]
        bounds =  box[0:2, 0:3]
        with struct.anytype():
            x = CenteredGrid('f', None, data_x)
            y = CenteredGrid('f', None, data_y)
            v = StaggeredGrid('v', bounds, None, [2, 2])
        x, y = complete_staggered_properties([x, y], v)
        v = v.with_data([x, y])

    def test_staggered_format_conversion(self):
        tensor = math.zeros([1, 5, 5, 2])
        tensor[:, 0, 0, :] = 1
        components = unstack_staggered_tensor(tensor)
        self.assertEqual(len(components), 2)
        np.testing.assert_equal(components[0].shape, [1, 5, 4, 1])
        np.testing.assert_equal(components[1].shape, [1, 4, 5, 1])
        tensor2 = stack_staggered_components(components)
        np.testing.assert_equal(tensor, tensor2)

    def test_points_flag(self):
        data = math.zeros([1, 2, 3, 1])
        f = CenteredGrid('f', box[0:2, 0:3], data)
        p = f.points
        assert SAMPLE_POINTS in p.flags
        assert p.points is p

    def test_bounds(self):
        tensor = math.zeros([1, 5, 5, 2])
        f = StaggeredGrid.from_tensors('f', None, unstack_staggered_tensor(tensor))
        bounds = data_bounds(f)
        self.assertIsInstance(bounds, AABox)
        numpy.testing.assert_equal(bounds.lower, 0)
        numpy.testing.assert_equal(bounds.upper, [4, 4])

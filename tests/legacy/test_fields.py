from unittest import TestCase

import numpy as np

from phi.flow import *
from phi import math
from phi.geom import box
from phi.field import *
from phi.math._tensors import Tensor


class TestFields(TestCase):

    def test_centered_grid_properties(self):
        physics_config.x_last()
        f = CenteredGrid(math.zeros([1, 3, 4, 1]), box[0:30, 0:40])
        print(repr(f))
        self.assertIsInstance(f.box.lower, Tensor)
        f.dx.assert_close(10)
        self.assertEqual('(y=3, x=4, 2)', repr(f.elements.center.shape))

    def test_staggered_grid_properties(self):
        physics_config.x_first()
        f = StaggeredGrid.from_staggered_tensor(math.zeros([1, 4, 5, 2]), box[0:30, 0:40])
        x, y = f.x, f.y
        self.assertIsInstance(x, CenteredGrid)
        x.dx.assert_close(y.dx, 10)


    def test_compatibility(self):
        f = CenteredGrid(math.zeros([1, 3, 4, 1]), box[0:3, 0:4])
        g = CenteredGrid(math.zeros([1, 3, 3, 1]), box[0:3, 0:4])
        np.testing.assert_equal(f.dx, [1, 1])
        self.assertTrue(f.points.compatible(f))
        self.assertFalse(f.compatible(g))

    def test_inner_interpolation(self):
        data = math.zeros([1, 2, 3, 1])
        data[0, :, :, 0] = [[1, 2, 3], [4, 5, 6]]
        f = CenteredGrid(data, box[0:2, 0:3])
        g = CenteredGrid(math.zeros([1, 2, 2, 1]), box[0:2, 0.5:2.5])
        # Resample optimized
        resampled = f.at(g)
        self.assertTrue(resampled.compatible(g))
        np.testing.assert_equal(resampled.values[0, ..., 0], [[1.5, 2.5], [4.5, 5.5]])
        # Resample unoptimized
        resampled2 = Field.at(f, g)
        self.assertTrue(resampled2.compatible(g))
        np.testing.assert_equal(resampled2.values[0, ..., 0], [[1.5, 2.5], [4.5, 5.5]])

    def test_staggered_interpolation(self):
        # 2x2 cells
        data_x = math.zeros([1, 2, 3, 1])
        data_x[0, :, :, 0] = [[1, 2, 3], [4, 5, 6]]
        data_y = math.zeros([1, 3, 2, 1])
        data_y[0, :, :, 0] = [[-1, -2], [-3, -4], [-5, -6]]
        v = StaggeredGrid([data_y, data_x])
        centered = v.at_centers()
        np.testing.assert_equal(centered.values.shape, [1, 2, 2, 2])

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
        f = CenteredGrid(data, box[0:2, 0:3])
        p = f.points
        assert SAMPLE_POINTS in p.flags
        assert p.points is p

    def test_bounds(self):
        tensor = math.zeros([1, 5, 5, 2])
        f = StaggeredGrid(tensor)
        bounds = data_bounds(f)
        self.assertIsInstance(bounds, Box)
        np.testing.assert_equal(bounds.lower, 0)
        np.testing.assert_equal(bounds.upper, [4, 4])

        a = CenteredGrid(np.zeros([1, 4, 4, 1]))
        np.testing.assert_equal(a.box.size, [4, 4])

        a = CenteredGrid(np.zeros([1, 4, 4, 1]), 1)
        np.testing.assert_equal(a.box.size, 1)

    def test_staggered_construction(self):
        # pylint: disable-msg = unsubscriptable-object
        tensor = math.zeros([1, 5, 5, 2])
        staggered = StaggeredGrid(tensor, name='')
        assert len(staggered.values) == 2
        assert isinstance(staggered.values[0], CenteredGrid)
        assert staggered.values[0].component_count == 1
        np.testing.assert_equal(staggered.values[0].box.lower, [-0.5, 0])
        staggered2 = StaggeredGrid(unstack_staggered_tensor(tensor), name='')
        struct.print_differences(staggered, staggered2)
        self.assertEqual(staggered, staggered2)
        staggered3 = StaggeredGrid([staggered.values[0], staggered2.values[1]], name='')
        self.assertEqual(staggered3, staggered)

    def test_mixed_boundaries_resample(self):
        data = np.reshape([[1,2], [3,4]], (1,2,2,1))
        field = CenteredGrid(data, extrapolation=[('boundary', 'constant'), 'periodic'])
        print(data[0,...,0])
        np.testing.assert_equal(field.sample_at([(0.5,0.5)]), [[1]])
        np.testing.assert_equal(field.sample_at([[10,0.5]]), [[0]])
        np.testing.assert_equal(field.sample_at([[0.5,2.5]]), [[1]])
        np.testing.assert_equal(field.sample_at([[0.5,1.5]]), [[2]])
        np.testing.assert_equal(field.sample_at([[-10,0.5]]), [[1]])
        np.testing.assert_equal(field.sample_at([[-10,1.5]]), [[2]])

    def test_constant_resample(self):
        field = ConstantField([0, 1])
        self.assertEqual(field.component_count, 2)
        # --- Resample to CenteredGrid ---
        at_cgrid = field.at(CenteredGrid(np.zeros([1, 4, 4, 1])))
        np.testing.assert_equal(at_cgrid.values.shape, [1, 4, 4, 2])
        # --- Resample to StaggeredGrid ---
        at_sgrid = field.at(Fluid([4, 4]).velocity)
        np.testing.assert_equal(at_sgrid.unstack()[0].values.shape, [1, 5, 4, 1])
        np.testing.assert_equal(at_sgrid.unstack()[1].values.shape, [1, 4, 5, 1])

    def test_staggered_curl2d(self):
        domain = Domain([32, 32])
        pot = CenteredGrid.sample(Noise(), domain)
        vel = staggered_curl_2d(pot)
        div = vel.divergence()
        np.testing.assert_almost_equal(div.values, 0, decimal=3)

from unittest import TestCase

import numpy as np

from phi.geom import Box, Sphere, box, GLOBAL_AXIS_ORDER
from phi.field import CenteredGrid


def points():
    return CenteredGrid(np.zeros([1, 10, 10, 1]), box[0:10, 0:10]).points


class TestGeometry(TestCase):

    def test_simple_box(self):
        GLOBAL_AXIS_ORDER.x_first()
        b = Box(0, [1, 2])
        self.assertEqual('(x=1, y=1, 2)', repr(b.shape))

    def test_batched_box(self):
        mybox = Box(0, np.stack([np.ones(10), np.linspace(0, 10, 10)], axis=-1))
        # 0D indexing
        values = mybox.lies_inside(np.zeros([10, 2]) + [0, 4])
        np.testing.assert_equal(values.shape, [10, 1])
        np.testing.assert_equal(values[:, 0], [0,0,0,0,1,1,1,1,1,1])
        # 1D indexing
        values = mybox.lies_inside(np.zeros([10, 3, 2]) + [0, 4])
        np.testing.assert_equal(values.shape, [10, 3, 1])
        np.testing.assert_equal(values[:, 0, 0], [0,0,0,0,1,1,1,1,1,1])

    def test_batched_sphere(self):
        moving_sphere = Sphere(center=np.stack([np.ones(10), np.linspace(0, 10, 10)], axis=-1), radius=1)
        growing_sphere = Sphere(center=0, radius=np.linspace(0, 10, 10))
        # 0D indexing
        values = moving_sphere.lies_inside(np.zeros([10, 2]) + [1, 4])
        np.testing.assert_equal(values[:, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        values = growing_sphere.lies_inside(np.zeros([10, 2]) + [0, 4])
        np.testing.assert_equal(values[:, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        # 1D indexing
        values = moving_sphere.lies_inside(np.zeros([10, 3, 2]) + [1, 4])
        np.testing.assert_equal(values.shape, [10, 3, 1])
        np.testing.assert_equal(values[:, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        values = growing_sphere.lies_inside(np.zeros([10, 3, 2]) + [0, 4])
        np.testing.assert_equal(values.shape, [10, 3, 1])
        np.testing.assert_equal(values[:, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

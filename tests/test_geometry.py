from unittest import TestCase

import numpy as np

from phi.geom import AABox, Sphere, box
from phi.physics.field import CenteredGrid


def points():
    return CenteredGrid(np.zeros([1, 10, 10, 1]), box[0:10, 0:10]).points


class TestGeometry(TestCase):

    def test_batched_box(self):
        mybox = AABox(0, np.stack([np.ones(10), np.linspace(0, 10, 10)], axis=-1))
        # 0D indexing
        values = mybox.value_at(np.zeros([10, 2]) + [0, 4])
        np.testing.assert_equal(values.shape, [10, 1])
        np.testing.assert_equal(values[:, 0], [0,0,0,0,1,1,1,1,1,1])
        # 1D indexing
        values = mybox.value_at(np.zeros([10, 3, 2]) + [0, 4])
        np.testing.assert_equal(values.shape, [10, 3, 1])
        np.testing.assert_equal(values[:, 0, 0], [0,0,0,0,1,1,1,1,1,1])

    def test_batched_sphere(self):
        moving_sphere = Sphere(center=np.stack([np.ones(10), np.linspace(0, 10, 10)], axis=-1), radius=1)
        growing_sphere = Sphere(center=0, radius=np.linspace(0, 10, 10))
        # 0D indexing
        values = moving_sphere.value_at(np.zeros([10, 2]) + [1, 4])
        np.testing.assert_equal(values[:, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        values = growing_sphere.value_at(np.zeros([10, 2]) + [0, 4])
        np.testing.assert_equal(values[:, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        # 1D indexing
        values = moving_sphere.value_at(np.zeros([10, 3, 2]) + [1, 4])
        np.testing.assert_equal(values.shape, [10, 3, 1])
        np.testing.assert_equal(values[:, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        values = growing_sphere.value_at(np.zeros([10, 3, 2]) + [0, 4])
        np.testing.assert_equal(values.shape, [10, 3, 1])
        np.testing.assert_equal(values[:, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

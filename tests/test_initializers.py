from unittest import TestCase

import numpy as np

from phi.geom import box
from phi import math, struct
from phi.physics.field import CenteredGrid


class TestInitializers(TestCase):

    def test_direct_initializers(self):
        np.testing.assert_equal(math.zeros([1, 16]), np.zeros([1, 16]))
        self.assertEqual(math.zeros([1, 16]).dtype, np.float32)
        np.testing.assert_equal(math.ones([1, 16, 1]), np.ones([1, 16, 1]))
        np.testing.assert_equal(math.zeros_like(math.ones([1, 16, 1])), np.zeros([1, 16, 1]))
        np.testing.assert_equal(math.randn([1, 4]).shape, [1, 4])
        self.assertEqual(math.randn([1, 4]).dtype, np.float32)

    def test_struct_initializers(self):
        obj = ([4], CenteredGrid([1, 4, 1], box[0:1], content_type=struct.shape), ([9], [8, 2]))
        z = math.zeros(obj)
        self.assertIsInstance(z, tuple)
        np.testing.assert_equal(z[0], np.zeros([4]))
        z2 = math.zeros_like(z)
        np.testing.assert_equal(math.shape(z)[0], math.shape(z2)[0])

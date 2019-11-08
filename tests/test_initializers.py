import numpy as np
from unittest import TestCase

from phi import struct
from phi.math import *
from phi.flow import *


# placeholder, variable tested in test_tensorflow.py


class TestInitializers(TestCase):

    def test_direct_initializers(self):
        np.testing.assert_equal(zeros([1, 16]), np.zeros([1, 16]))
        self.assertEqual(zeros([1, 16]).dtype, np.float32)
        np.testing.assert_equal(ones([1, 16, 1]), np.ones([1, 16, 1]))
        np.testing.assert_equal(zeros_like(ones([1, 16, 1])), np.zeros([1, 16, 1]))
        np.testing.assert_equal(randn([1, 4]).shape, [1, 4])
        self.assertEqual(randn([1, 4]).dtype, np.float32)

    def test_struct_initializers(self):
        bounds = box[0:1]
        with struct.anytype():
            obj = ([4], CenteredGrid('', bounds, [1, 4, 1]), ([9], [8, 2]))
        z = zeros(obj)
        self.assertIsInstance(z, tuple)
        np.testing.assert_equal(z[0], np.zeros([4]))
        z2 = zeros_like(z)
        np.testing.assert_equal(shape(z)[0], shape(z2)[0])

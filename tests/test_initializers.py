from unittest import TestCase
from phi.flow import *
from phi.math import *


# placeholder, variable tested in test_tensorflow.py


class TestInitializers(TestCase):
    def test_direct_initializers(self):
        numpy.testing.assert_equal(zeros([1,16]), np.zeros([1,16]))
        numpy.testing.assert_equal(ones([1,16,1]), np.ones([1,16,1]))
        numpy.testing.assert_equal(zeros_like(ones([1,16,1])), np.zeros([1,16,1]))
        numpy.testing.assert_equal(randn()([1,4]).shape, [1,4])

    def test_struct_initializers(self):
        struct = ([4], StaggeredGrid([4, 1]), ([9], [8,2]))
        z = zeros(struct)
        self.assertIsInstance(z, tuple)
        numpy.testing.assert_equal(z[0], np.zeros([4]))
        z2 = zeros_like(z)
        numpy.testing.assert_equal(shape(z)[0], shape(z2)[0])
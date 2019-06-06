from unittest import TestCase
from phi.math import container
from phi.math.nd import *


class TestTensorContainer(TestCase):

    def test_zeros(self):
        original_shape = [1,4,4,2]
        s = StaggeredGrid(np.zeros(original_shape))
        shape = container.shape(s)
        np.testing.assert_almost_equal(shape.staggered, original_shape)

        s2 = container.zeros(shape)
        np.testing.assert_almost_equal(s.staggered, s2.staggered)

        s3 = container.zeros_like(s)
        np.testing.assert_almost_equal(s.staggered, s3.staggered)
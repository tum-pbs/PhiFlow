from unittest import TestCase
from phi.tf.flow import *


class TestPlaceholder(TestCase):

    def test_placeholder(self):
        shape = (StaggeredGrid((1, 4, 4, 1)), [16, 1])
        struct = placeholder(shape)
        np.testing.assert_equal(struct[0].staggered.shape.as_list(), [1, 4, 4, 1])
        np.testing.assert_equal(struct[1].shape.as_list(), [16, 1])
        self.assertEqual(struct[0].staggered.name, 'index0/staggered:0')
        self.assertEqual(struct[1].name, 'index1:0')

    def test_placeholder_like(self):
        state = Smoke(Domain([4, 4]))
        struct = placeholder_like(state)
        np.testing.assert_equal(struct.density.shape, [1, 4, 4, 1])
        np.testing.assert_equal(struct.velocity.shape, [1, 5, 5, 2])
        self.assertEqual(struct.density.name, 'density:0')
        self.assertEqual(struct.velocity.name, 'velocity/staggered:0')

        struct = placeholder_like(state, basename='state', separator='.')
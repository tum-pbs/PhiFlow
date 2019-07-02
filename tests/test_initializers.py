from unittest import TestCase
from phi.tf.flow import *


class TestInitializers(TestCase):
    def test_direct_initializers(self):
        numpy.testing.assert_equal(zeros([1,16]), np.zeros([1,16]))
        numpy.testing.assert_equal(ones([1,16,1]), np.ones([1,16,1]))
        numpy.testing.assert_equal(zeros_like(ones([1,16,1])), np.zeros([1,16,1]))
        numpy.testing.assert_equal(randn()([1,4]).shape, [1,4])
        # --- TensorFlow ---
        tf.reset_default_graph()
        p = placeholder([4])
        self.assertIsInstance(p, tf.Tensor)
        numpy.testing.assert_equal(p.shape.as_list(), [4])
        numpy.testing.assert_equal(placeholder_like(p).shape.as_list(), p.shape.as_list())
        self.assertEqual(p.name, 'Placeholder:0')
        v = variable(zeros)([2,2])
        numpy.testing.assert_equal(v.shape.as_list(), [2,2])
        self.assertIsInstance(v, tf.Variable)
        self.assertEqual(v.name, 'Variable:0')

    def test_struct_initializers(self):
        struct = ([4], StaggeredGrid([4, 1]), ([9], [8,2]))
        z = zeros(struct)
        self.assertIsInstance(z, tuple)
        numpy.testing.assert_equal(z[0], np.zeros([4]))
        z2 = zeros_like(z)
        numpy.testing.assert_equal(shape(z)[0], shape(z2)[0])
        # --- TensorFlow ---
        tf.reset_default_graph()
        p = placeholder(struct)
        self.assertEqual(p[0].name, '0:0')
        self.assertEqual(p[1].staggered.name, '1/staggered:0')
        self.assertIsInstance(p, tuple)
        p2 = placeholder_like(p)
        self.assertIsInstance(p2, tuple)
        numpy.testing.assert_equal(p2[1].staggered.shape.as_list(), [4, 1])
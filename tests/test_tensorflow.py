from unittest import TestCase
from phi.tf.flow import *


class TestPlaceholder(TestCase):

    def test_direct_placeholders(self):
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

    def test_struct_placeholders(self):
        struct = ([4], StaggeredGrid([4, 1]), ([9], [8,2]))
        tf.reset_default_graph()
        p = placeholder(struct)
        self.assertEqual(p[0].name, '0:0')
        self.assertEqual(p[1].staggered.name, '1/staggered:0')
        self.assertIsInstance(p, tuple)
        p2 = placeholder_like(p)
        self.assertIsInstance(p2, tuple)
        numpy.testing.assert_equal(p2[1].staggered.shape.as_list(), [4, 1])
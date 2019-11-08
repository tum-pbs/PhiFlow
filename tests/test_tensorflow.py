from unittest import TestCase
import numpy as np

from phi.tf.flow import *


class TestPlaceholder(TestCase):

    def test_direct_placeholders(self):
        tf.reset_default_graph()
        p = placeholder([4])
        self.assertIsInstance(p, tf.Tensor)
        np.testing.assert_equal(p.shape.as_list(), [4])
        np.testing.assert_equal(placeholder_like(p).shape.as_list(), p.shape.as_list())
        self.assertEqual(p.name, 'Placeholder:0')
        v = variable(math.zeros([2, 2]))
        np.testing.assert_equal(v.shape.as_list(), [2, 2])
        self.assertIsInstance(v, tf.Variable)
        self.assertEqual(v.name, 'Variable:0')

    def test_struct_placeholders(self):
        bounds = box[0:1]
        with struct.anytype():
            obj = ([4], CenteredGrid('', bounds, [1, 4, 1]), ([9], [8, 2]))
        tf.reset_default_graph()
        p = placeholder(obj)
        self.assertEqual(p[0].name, '0:0')
        self.assertEqual(p[1].data.name, '1/data:0')
        self.assertIsInstance(p, tuple)
        p2 = placeholder_like(p)
        self.assertIsInstance(p2, tuple)
        np.testing.assert_equal(p2[1].data.shape.as_list(), [1, 4, 1])

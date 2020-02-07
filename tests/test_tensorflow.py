from unittest import TestCase

import numpy
import tensorflow

from phi.tf.util import placeholder

from phi import math, struct
from phi.geom import box
from phi.physics.field import CenteredGrid
from phi.tf.util import variable


class TestPlaceholder(TestCase):

    def test_direct_placeholders(self):
        tensorflow.reset_default_graph()
        p = placeholder([4])
        self.assertIsInstance(p, tensorflow.Tensor)
        numpy.testing.assert_equal(p.shape.as_list(), [4])
        self.assertEqual(p.name, 'Placeholder:0')
        v = variable(math.zeros([2, 2]))
        numpy.testing.assert_equal(v.shape.as_list(), [2, 2])
        self.assertIsInstance(v, tensorflow.Variable)
        self.assertEqual(v.name, 'Variable:0')

    def test_struct_placeholders(self):
        obj = ([4], CenteredGrid([1, 4, 1], box[0:1], content_type=struct.shape), ([9], [8, 2]))
        tensorflow.reset_default_graph()
        p = placeholder(obj)
        self.assertEqual(p[0].name, '0:0')
        self.assertEqual(p[1].data.name, '1/data:0')
        self.assertIsInstance(p, tuple)

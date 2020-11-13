from unittest import TestCase

from phi import math
from phi.math import infer_shape, shape


class TestShape(TestCase):

    def test_infer_shape(self):
        shape = infer_shape([1, 2, 3, 4], batch_dims=1, channel_dims=1)
        self.assertEqual(2, shape.spatial_rank)
        self.assertEqual(shape, infer_shape([1, 2, 3, 4]))
        shape = infer_shape([1, 2, 3, 4], batch_dims=0, channel_dims=0)
        self.assertEqual(4, shape.spatial_rank)
        shape = infer_shape([1, 2, 3, 4], batch_dims=0, spatial_dims=0)
        self.assertEqual(4, shape.channel_rank)

    def test_dimension_types(self):
        v = math.ones(batch=10, x=4, y=3, vector=2)
        self.assertEqual(v.x.index, 1)
        self.assertEqual(v.x.name, 'x')
        self.assertTrue(v.x.is_spatial)
        self.assertTrue(v.batch.is_batch)
        b = v.x.as_batch()
        self.assertTrue(b.x.is_batch)

    def test_combine(self):
        self.assertEqual(shape(batch=2, x=3, y=4), shape(batch=2) & shape(x=3, y=4))
        self.assertEqual(shape(x=3, vector=2), shape(vector=2) & shape(x=3))
        self.assertEqual(shape(batch=10, x=3, vector=2), shape(vector=2) & shape(x=3) & shape(batch=10))

    def test_subshapes(self):
        s = shape(batch=10, x=4, y=3, vector=2)
        self.assertEqual(shape(batch=10), s.batch)
        self.assertEqual(shape(x=4, y=3), s.spatial)
        self.assertEqual(shape(vector=2), s.channel)

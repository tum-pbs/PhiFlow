from unittest import TestCase

from phi import math
from phi.math import shape
from phi.math._shape import shape_stack, BATCH_DIM, vector_add


class TestShape(TestCase):

    def test_dimension_types(self):
        v = math.ones(batch=10, x=4, y=3, vector=2)
        self.assertEqual(v.x.index, 1)
        self.assertEqual(v.x.name, 'x')
        self.assertEqual(('batch', 'spatial', 'spatial', 'channel'), v.shape.types)
        b = v.x.as_batch()
        self.assertEqual(('batch', 'batch', 'spatial', 'channel'), b.shape.types)

    def test_combine(self):
        self.assertEqual(shape(batch=2, x=3, y=4), shape(batch=2) & shape(x=3, y=4))
        self.assertEqual(shape(x=3, vector=2), shape(vector=2) & shape(x=3))
        self.assertEqual(shape(batch=10, x=3, vector=2), shape(vector=2) & shape(x=3) & shape(batch=10))

    def test_stack(self):
        stacked = shape_stack('stack', BATCH_DIM, shape(time=1, x=3, y=3), shape(x=3, y=4), shape())
        print(stacked)
        self.assertEqual(('stack', 'time', 'x', 'y'), stacked.names)
        self.assertEqual(3, stacked.get_size('stack'))
        self.assertEqual(1, stacked.get_size('time'))
        math.assert_close((3, 3, 1), stacked.get_size('x'))
        math.assert_close((3, 4, 1), stacked.get_size('y'))
        print(stacked.shape)
        self.assertEqual(('stack', 'dims'), stacked.shape.names)
        self.assertEqual(12, stacked.shape.volume)

    def test_subshapes(self):
        s = shape(batch=10, x=4, y=3, vector=2)
        self.assertEqual(shape(batch=10), s.batch)
        self.assertEqual(shape(x=4, y=3), s.spatial)
        self.assertEqual(shape(vector=2), s.channel)

    def test_indexing(self):
        s = shape(batch=10, x=4, y=3, vector=2)
        self.assertEqual(shape(batch=10), s[0:1])
        self.assertEqual(shape(batch=10), s[[0]])
        self.assertEqual(shape(x=4, y=3), s[1:3])

    def test_after_gather(self):
        self.assertEqual(shape(x=2), shape(x=3).after_gather({'x': slice(None, None, 2)}))

    def test_vector_add(self):
        self.assertEqual(vector_add(shape(batch=10, x=4, y=3), shape(x=1, y=-1, z=2)), shape(batch=10, x=5, y=2, z=2))


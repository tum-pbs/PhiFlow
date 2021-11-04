from unittest import TestCase

from phi import math
from phi.math import spatial, channel, batch, instance
from phi.math._shape import shape_stack, vector_add, IncompatibleShapes, EMPTY_SHAPE


class TestShape(TestCase):

    def test_dimension_types(self):
        v = math.ones(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        self.assertEqual(v.x.index, 1)
        self.assertEqual(v.x.name, 'x')
        self.assertEqual(('batch', 'spatial', 'spatial', 'channel'), v.shape.types)
        b = v.x.as_batch()
        self.assertEqual(('batch', 'batch', 'spatial', 'channel'), b.shape.types)

    def test_combine(self):
        self.assertEqual(batch(batch=10) & spatial(y=4, x=3) & channel(vector=2), batch(batch=10) & channel(vector=2) & spatial(y=4, x=3))
        try:
            spatial(y=4) & spatial(x=3)
            self.fail()
        except IncompatibleShapes:
            pass

    def test_stack(self):
        stacked = shape_stack(batch('stack'), batch(time=1) & spatial(x=3, y=3), spatial(x=3, y=4), EMPTY_SHAPE)
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
        s = batch(batch=10) & spatial(x=4, y=3) & channel(vector=2) & instance(points=1)
        self.assertEqual(batch(batch=10), s.batch)
        self.assertEqual(spatial(x=4, y=3), s.spatial)
        self.assertEqual(channel(vector=2), s.channel)
        self.assertEqual(instance(points=1), s.instance)
        self.assertEqual(batch(batch=10), batch(s))
        self.assertEqual(spatial(x=4, y=3), spatial(s))
        self.assertEqual(channel(vector=2), channel(s))
        self.assertEqual(instance(points=1), instance(s))

    def test_indexing(self):
        s = batch(batch=10) & spatial(x=4, y=3) & channel(vector=2)
        self.assertEqual(batch(batch=10), s[0:1])
        self.assertEqual(batch(batch=10), s[[0]])
        self.assertEqual(spatial(x=4, y=3), s[1:3])
        self.assertEqual(spatial(x=4), s['x'])

    def test_after_gather(self):
        self.assertEqual(spatial(x=2), spatial(x=3).after_gather({'x': slice(None, None, 2)}))
        self.assertEqual(EMPTY_SHAPE, spatial(x=3).after_gather({'x': 0}))

    def test_vector_add(self):
        self.assertEqual(vector_add(batch(batch=10) & spatial(x=4, y=3), spatial(x=1, y=-1, z=2)), batch(batch=10) & spatial(x=5, y=2, z=2))


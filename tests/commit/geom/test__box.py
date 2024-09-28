from unittest import TestCase


from phi import math
from phi.geom import Box, union, Cuboid, embed
from phi.math import batch, channel
from phi.math.magic import Shaped, Sliceable, Shapable
from phiml.math import vec


class TestBox(TestCase):

    def test_interfaces(self):
        box = Box()
        self.assertIsInstance(box, Shaped)
        self.assertIsInstance(box, Sliceable)
        self.assertIsInstance(box, Shapable)

    def test_box_constructor(self):
        try:
            Box(0, (1, 1))
            raise RuntimeError
        except AssertionError:
            pass
        math.assert_close(Box(x=1, y=1).size, 1)

    def test_box_constructor_kwargs(self):
        b = Box(x=3.5, y=4)
        math.assert_close(b.lower, 0)
        math.assert_close(b.upper, (3.5, 4))
        b = Box(x=(1, 2), y=None)
        math.assert_close(b.lower, (1, -math.INF))
        math.assert_close(b.upper, (2, math.INF))
        b = Box(x=(None, None))
        math.assert_close(b.lower, -math.INF)
        math.assert_close(b.upper, math.INF)

    def test_box_batched(self):
        lower = math.tensor([(0, 0), (1, 1)], batch('boxes'), channel(vector='x,y'))
        upper = math.wrap((1, 1), channel(vector='x,y'))
        box = Box(lower, upper)
        self.assertEqual(batch(boxes=2) & channel(vector='x,y'), box.shape)

    def test_slice(self):
        b1, b2 = Box(x=4, y=3), Box(x=2, y=1)
        u = union(b1, b2)
        self.assertEqual(b1, u.union[0])
        self.assertEqual(b2, u.union[1])

    def test_project(self):
        box = Box(x=4, y=3)
        self.assertEqual(Box(x=4), box.project('x'))
        self.assertEqual(box, box.project('x', 'y'))
        # Slicing vector
        box = Box(x=4, y=3)
        self.assertEqual(Box(x=4), box.vector['x'])
        self.assertEqual(box, box.vector['x,y'])
        self.assertEqual(Box(x=4), box['x'])

    def test_without(self):
        box = Box(x=4, y=3)
        self.assertEqual(Box(x=4), box.without(('y',)))
        self.assertEqual(Box(), box.without(('x', 'y')))
        self.assertEqual(box, box.without(()))

    def test_embed(self):
        self.assertEqual(Box(x=4, y=3, z=None), embed(Box(x=4, y=3), 'z'))
        self.assertEqual(Box(x=4, y=3, z=None), embed(Box(x=4, y=3), 'x,z')['x,y,z'])

    def test_box_product(self):
        a = Box(x=4)
        b = Box(y=3).shifted(math.wrap(1))
        ab = a * b
        self.assertEqual(2, ab.spatial_rank)
        math.assert_close(ab.size, (4, 3))
        math.assert_close(ab.lower, (0, 1))

    def test_union_same(self):
        u = union(Box(x=1, y=1), Box(x=(2, 3), y=1))
        self.assertIsInstance(u, Box)
        math.assert_close(u.approximate_signed_distance((0, 0)), u.approximate_signed_distance((3, 1)), 0)
        math.assert_close(u.approximate_signed_distance((1.5, 0)), 0.5)

    def test_stack_volume(self):
        u = math.stack([Box(x=1, y=1), Box(x=2, y=2)], batch('batch'))
        math.assert_close(u.volume, [1, 4])

    def test_stack_type(self):
        bounds1 = Box(x=1, y=1)
        bounds2 = Box(x=10, y=10)
        bounds = math.stack([bounds1, bounds2], batch('batch'))
        self.assertIsInstance(bounds, Box)

    def test_box_eq(self):
        self.assertNotEqual(Box(x=1, y=1), Box(x=1))
        self.assertEqual(Box(x=1, y=1), Box(x=1, y=1))

    def test_cuboid_constructor_kwargs(self):
        c = Cuboid(x=2., y=1.)
        math.assert_close(c.lower, -c.upper, (-1, -.5))

    def test_slicing_constructor(self):
        box = Box(x=(1, 2), y=(2, None))
        self.assertEqual(box, Box['x,y', 1:2, 2:])

    def test_rotated_half_extent(self):
        box = Box(x=50, y=10)
        math.assert_close([25, 5], box.bounding_half_extent())
        math.assert_close([21.213203, 21.213203], box.rotated(math.PI / 4).bounding_half_extent())

    def test_face_normals_2d(self):
        n = Box(x=50, y=10).face_normals
        math.assert_close(math.vec(x=0, y=-1), n[{'~side': 'lower', '~vector': 'y'}])
        math.assert_close(math.vec(x=0, y=1), n[{'~side': 'upper', '~vector': 'y'}])
        math.assert_close(math.vec(x=1, y=0), n[{'~side': 'upper', '~vector': 'x'}])
        # --- rotated ---
        n = Box(x=50, y=10).rotated(math.PI / 2).face_normals
        math.assert_close(math.vec(x=1, y=0), n[{'~side': 'lower', '~vector': 'y'}], abs_tolerance=1e-5)
        math.assert_close(math.vec(x=-1, y=0), n[{'~side': 'upper', '~vector': 'y'}], abs_tolerance=1e-5)
        math.assert_close(math.vec(x=0, y=1), n[{'~side': 'upper', '~vector': 'x'}], abs_tolerance=1e-5)

    def test_corners(self):
        c = Box(x=50, y=10).corners
        math.assert_close(vec(x=0, y=0), c[{'~x': 0, '~y': 0}])
        math.assert_close(vec(x=0, y=10), c[{'~x': 0, '~y': 1}])
        math.assert_close(vec(x=50, y=10), c[{'~x': 1, '~y': 1}])
        # --- rotated ---
        c = Cuboid(x=50, y=10).rotated(math.PI / 2).corners
        math.assert_close(vec(x=5, y=-25), c[{'~x': 0, '~y': 0}])
        math.assert_close(vec(x=-5, y=-25), c[{'~x': 0, '~y': 1}])
        math.assert_close(vec(x=-5, y=25), c[{'~x': 1, '~y': 1}])

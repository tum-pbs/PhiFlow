from unittest import TestCase

import numpy

from phi.geom import box
from phi.physics.collective import CollectiveState
from phi.physics.domain import Domain
from phi.physics.field import CenteredGrid, manta
from phi import struct
from phi.physics.smoke import Smoke
from phi.struct.tensorop import collapse, collapsed_gather_nd, expand


def generate_test_structs():
    return [manta.centered_grid(numpy.zeros([1,4,1])),
            [('Item',)],
            {'A': 'Entry A', 'Vel': manta.staggered_grid(numpy.zeros([1,5,5,2]))},
            CollectiveState((Smoke(Domain([4])),))]


class TestStruct(TestCase):

    def test_identity(self):
        for obj in generate_test_structs():
            with struct.anytype():
                obj2 = struct.map(lambda s: s, obj, recursive=False)
                self.assertEqual(obj, obj2)
                obj3 = struct.map(lambda t: t, obj, recursive=True)
                self.assertEqual(obj, obj3)
                obj4 = struct.map(lambda t: t, obj, item_condition=struct.ALL_ITEMS)
                self.assertEqual(obj, obj4)

    def test_flatten(self):
        for obj in generate_test_structs():
            flat = struct.flatten(obj)
            self.assertIsInstance(flat, list)
            self.assertGreater(len(flat), 0)
            for item in flat:
                self.assertTrue(not struct.isstruct(item), 'The result of flatten(%s) is not flat.' % obj)

    def test_names(self):
        for obj in generate_test_structs():
            with struct.anytype():
                names = struct.flatten(struct.map(lambda attr: attr.name, obj, trace=True))
                self.assertGreater(len(names), 0)
                for name in names:
                    self.assertIsInstance(name, str)

    def test_paths(self):
        obj = {'Vels': [CenteredGrid('v', numpy.zeros([1, 4, 1]), box[0:1])]}
        with struct.anytype():
            names = struct.flatten(struct.map(lambda attr: attr.path(), obj, trace=True))
        self.assertEqual(names[0], 'Vels.0.data')

    def test_copy(self):
        with struct.anytype():
            smoke = Smoke(Domain([4]), density='Density', velocity='Velocity')
            v = smoke.copied_with(velocity='V2')
            self.assertEqual(v.velocity, 'V2')
            self.assertEqual(v.density, 'Density')

            try:
                smoke.copied_with(velocity='D2')
                self.fail()
            except AssertionError:
                pass

    def test_zip(self):
        with struct.anytype():
            a = CenteredGrid('a', 'a')
            b = CenteredGrid('b', 'b')
            stacked = struct.map(lambda *x: x, struct.zip([a, b]))
            numpy.testing.assert_equal(stacked.data, ('a', 'b'))

    def test_collapse(self):
        self.assertEqual(0, collapse(numpy.zeros([2, 2])))
        self.assertEqual(0, collapse(numpy.zeros([2, 2]).tolist()))
        self.assertEqual(('a', 'a', 'b'), tuple(collapse(['a', ('a', 'a'), 'b'])))

    def collapsed_gather_nd(self):
        self.assertEqual('a', collapsed_gather_nd('a', [1, 2, 3, 4, 5]))
        self.assertEqual('b', collapsed_gather_nd(['a', 'b'], [1, 0]))
        self.assertEqual(('b', 'b'), collapsed_gather_nd(['a', ('b', 'b')], [1, 0], leaf_condition=lambda x: isinstance(x, tuple)))

    def test_expand(self):
        numpy.testing.assert_equal(expand(1, shape=(2,2)), [[1,1], [1,1]])
        numpy.testing.assert_equal(expand(['a', ('b', 'c')], shape=(2,2)), [['a', 'a'], ['b', 'c']])